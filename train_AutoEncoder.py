import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from isegm.utils.utils import Config, Progbar, to_cuda
from isegm.utils.logger import setup_logger
from isegm.data.dataloader_transformer import load_dataset
from isegm.model.modeling.AE_model import AE_Model, extract_boundary, mask_recon_inference
from isegm.utils.wrappers import cat


def get_avg_loss(loss):
    # Just for mutil gpu in ddp mode
    world_size = dist.get_world_size()
    with torch.no_grad():
        if world_size >= 2:
            dist.all_reduce(loss)
            loss /= world_size
    return loss

# [Codebook Augmentation] These helpers are only used when building the shape-prior
# codebook after AE training. They do not change the AE reconstruction training path.
def spatial_augment_mask_pair(fm_mask, vm_mask, args):
    """Apply one conservative spatial transform to paired amodal/visible masks."""
    fm_aug, vm_aug = [], []

    for fm_single, vm_single in zip(fm_mask, vm_mask):
        src_fm = fm_single
        src_vm = vm_single

        if torch.rand(1).item() < args.codebook_aug_hflip_prob:
            src_fm = torch.flip(src_fm, dims=(-1,))
            src_vm = torch.flip(src_vm, dims=(-1,))

        angle = (torch.rand(1).item() * 2.0 - 1.0) * args.codebook_aug_degrees
        scale = args.codebook_aug_scale_min + torch.rand(1).item() * (
            args.codebook_aug_scale_max - args.codebook_aug_scale_min
        )
        translate_x = (torch.rand(1).item() * 2.0 - 1.0) * args.codebook_aug_translate
        translate_y = (torch.rand(1).item() * 2.0 - 1.0) * args.codebook_aug_translate

        angle = torch.tensor(angle * np.pi / 180.0, dtype=src_fm.dtype, device=src_fm.device)
        cos_a = torch.cos(angle) / scale
        sin_a = torch.sin(angle) / scale
        theta = torch.stack([
            torch.stack([cos_a, -sin_a, torch.tensor(translate_x, dtype=src_fm.dtype, device=src_fm.device)]),
            torch.stack([sin_a, cos_a, torch.tensor(translate_y, dtype=src_fm.dtype, device=src_fm.device)]),
        ]).unsqueeze(0)

        grid = F.affine_grid(theta, size=src_fm.unsqueeze(0).shape, align_corners=False)
        aug_fm = F.grid_sample(src_fm.unsqueeze(0).float(), grid, mode='nearest',
                               padding_mode='zeros', align_corners=False).squeeze(0)
        aug_vm = F.grid_sample(src_vm.unsqueeze(0).float(), grid, mode='nearest',
                               padding_mode='zeros', align_corners=False).squeeze(0)

        aug_fm = (aug_fm > 0.5).to(fm_single.dtype)
        aug_vm = (aug_vm > 0.5).to(vm_single.dtype)

        # [Codebook Augmentation] Avoid recording degenerate empty masks.
        if torch.sum(aug_fm) == 0:
            aug_fm = fm_single
            aug_vm = vm_single

        fm_aug.append(aug_fm)
        vm_aug.append(aug_vm)

    return torch.stack(fm_aug, dim=0), torch.stack(vm_aug, dim=0)


@torch.no_grad()
def record_codebook_vectors(config, items, recon_net, args=None, augment=False, allowed_classes=None):
    """Encode masks into the existing [vm, edge, am] 1088-d shape-prior format."""
    net = recon_net.module if hasattr(recon_net, 'module') else recon_net

    fm_crop_gt = (items['fm_crop'] > 0.5).float()
    vm_crop_gt = (items['vm_crop_gt'] > 0.5).float()
    category_ids = items['category_id'].view(-1)

    if augment:
        fm_crop_gt, vm_crop_gt = spatial_augment_mask_pair(fm_crop_gt, vm_crop_gt, args)

    if allowed_classes is not None:
        allowed_classes = {int(x) for x in allowed_classes}
        keep_indices = [
            idx for idx, cls_id in enumerate(category_ids.detach().cpu().tolist())
            if int(cls_id) in allowed_classes
        ]
        if not keep_indices:
            return
        keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=fm_crop_gt.device)
        fm_crop_gt = fm_crop_gt.index_select(0, keep_indices)
        vm_crop_gt = vm_crop_gt.index_select(0, keep_indices)
        category_ids = category_ids.index_select(0, keep_indices)

    edge_mask_gt, _ = extract_boundary(fm_crop_gt, vm_crop_gt)

    latent_am = net.AM_AE_Net.encode(fm_crop_gt).flatten(1)
    latent_vm = net.VM_AE_Net.encode(vm_crop_gt).flatten(1)
    latent_edge = net.Edge_AE_Net.encode((edge_mask_gt > 0.5).float()).flatten(1)
    combined_latent = torch.cat((latent_vm, latent_edge, latent_am), dim=1).detach()

    vector_dict = {}
    for class_id in category_ids.unique():
        class_key = int(class_id.item())
        index = (category_ids == class_id).nonzero(as_tuple=False).view(-1)
        vector_dict[class_key] = combined_latent.index_select(0, index)

    net.recording_vectors(vector_dict)


def get_shortage_classes(vector_dict, target_size):
    return {
        int(class_id)
        for class_id, vectors in vector_dict.items()
        if vectors.size(0) < target_size
    }


def cluster_codebook_vectors(vector_dict, target_size):
    """Keep the old behavior for overfull classes: compress them with K-Means."""
    for class_id in list(vector_dict.keys()):
        vectors = vector_dict[class_id]
        if vectors.size(0) <= target_size:
            continue

        kmeans = KMeans(n_clusters=target_size)
        kmeans.fit(vectors.detach().cpu().numpy())
        vector_dict[class_id] = torch.as_tensor(
            kmeans.cluster_centers_, dtype=vectors.dtype, device=vectors.device
        )


def save_codebook(config, vector_dict, args, logger):
    # [Codebook Augmentation] Augmented codebooks are saved to a new file by default
    # so existing codebooks are not overwritten accidentally.
    output_path = args.codebook_output
    if output_path is None:
        suffix = '_aug' if args.augment_codebook else ''
        output_path = os.path.join(config.codebook_path, f'{config.dataset}_codebook_NewAE{suffix}.npy')

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    vector_dict_numpy = {
        key: tensor.detach().cpu().numpy()
        for key, tensor in vector_dict.items()
    }
    np.save(output_path, vector_dict_numpy)
    logger.info("codebook saved: {}".format(output_path))


def build_shape_prior_codebook(config, train_dataset, autoencoder, args, logger, rank):
    # [Codebook Build] Re-enabled shape-prior codebook export after AE training.
    # Spatial augmentation remains opt-in via --augment-codebook.
    if rank != 0:
        return

    target_size = args.codebook_size if args.codebook_size is not None else config.KMEANS
    autoencoder.module.vector_dict = {}
    autoencoder.eval()

    codebook_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        num_workers=config.test_num_workers,
        drop_last=False,
    )

    logger.info("Start collecting codebook vectors")
    with torch.no_grad():
        for items in tqdm(codebook_loader, desc='Collect codebook'):
            items = to_cuda(items, config.device)
            record_codebook_vectors(config, items, autoencoder)

    if args.augment_codebook:
        logger.info("Start spatial augmentation for classes with fewer than {} slots".format(target_size))
        for round_idx in range(args.codebook_aug_rounds):
            shortage_classes = get_shortage_classes(autoencoder.module.vector_dict, target_size)
            if not shortage_classes:
                break

            before_count = sum(v.size(0) for v in autoencoder.module.vector_dict.values())
            for items in tqdm(codebook_loader, desc=f'Augment codebook {round_idx + 1}'):
                shortage_classes = get_shortage_classes(autoencoder.module.vector_dict, target_size)
                if not shortage_classes:
                    break
                items = to_cuda(items, config.device)
                record_codebook_vectors(
                    config, items, autoencoder, args=args, augment=True,
                    allowed_classes=shortage_classes,
                )

            after_count = sum(v.size(0) for v in autoencoder.module.vector_dict.values())
            if after_count == before_count:
                break

        shortage_classes = get_shortage_classes(autoencoder.module.vector_dict, target_size)
        if shortage_classes:
            logger.warning("Classes still below target slots after augmentation: {}".format(
                sorted(shortage_classes)
            ))

    for class_id, vectors in sorted(autoencoder.module.vector_dict.items()):
        logger.info("Before clustering - class: {}, vectors: {}".format(class_id, tuple(vectors.shape)))

    logger.info("Start KMEANS clustering with target slots: {}".format(target_size))
    cluster_codebook_vectors(autoencoder.module.vector_dict, target_size)

    for class_id, vectors in sorted(autoencoder.module.vector_dict.items()):
        logger.info("Final codebook - class: {}, vectors: {}".format(class_id, tuple(vectors.shape)))

    save_codebook(config, autoencoder.module.vector_dict, args, logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    # path
    parser.add_argument('--path', type=str, required=True, help='model checkpoints path')
    parser.add_argument('--check_point_path', type=str, default="/mnt/data/linjunwei/AIS_check_points")

    # training
    parser.add_argument('--Image_W', type=int, default=256)
    parser.add_argument('--Image_H', type=int, default=256)

    # dataset
    parser.add_argument('--dataset', type=str, default="KINS", help="select dataset")
    parser.add_argument('--data_type', type=str, default="image", help="select image or video model")
    parser.add_argument('--batch', type=int, default=16)

    parser.add_argument("--local-rank", default=0, type=int, help="node rank for distributed training")

    # [Codebook Build] Build/export the shape-prior codebook after AE training by default.
    parser.add_argument('--skip-codebook-build', action='store_true',
                        help='Skip shape-prior codebook construction after AE training.')
    parser.add_argument('--codebook-size', type=int, default=None,
                        help='Target slots per category. Defaults to config.KMEANS.')
    parser.add_argument('--codebook-output', type=str, default=None,
                        help='Optional output .npy path. Defaults to config.codebook_path/{dataset}_codebook_NewAE[_aug].npy.')

    # [Codebook Augmentation] Optional CVPR-style spatial augmentation for categories
    # with fewer than the target number of codebook slots.
    parser.add_argument('--augment-codebook', action='store_true',
                        help='Use spatial augmentation to fill underrepresented codebook categories.')
    parser.add_argument('--codebook-aug-rounds', type=int, default=20,
                        help='Maximum passes over the training set when filling underrepresented categories.')
    parser.add_argument('--codebook-aug-degrees', type=float, default=10.0,
                        help='Maximum absolute rotation angle for codebook spatial augmentation.')
    parser.add_argument('--codebook-aug-translate', type=float, default=0.08,
                        help='Maximum normalized translation for codebook spatial augmentation.')
    parser.add_argument('--codebook-aug-scale-min', type=float, default=0.90,
                        help='Minimum scale for codebook spatial augmentation.')
    parser.add_argument('--codebook-aug-scale-max', type=float, default=1.10,
                        help='Maximum scale for codebook spatial augmentation.')
    parser.add_argument('--codebook-aug-hflip-prob', type=float, default=0.5,
                        help='Horizontal flip probability for codebook spatial augmentation.')
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    rank = dist.get_rank()

    args.path = os.path.join(args.check_point_path, args.path)
    os.makedirs(args.path, exist_ok=True)
    config_path = os.path.join(args.path, '{}.yml'.format(args.dataset))
    if not os.path.exists(config_path):
        copyfile('/home/univ/hongren/Projects/Amodal_Image_Segmentation/configs/{}.yml'.format(args.dataset),
                 config_path)

    # load config file
    config = Config(config_path)
    config.path = args.path
    config.batch_size = args.batch
    config.dataset = args.dataset
    # [Codebook Build] Some AE utilities expect rank to exist on config.
    config.rank = rank

    log_file = 'log-{}.txt'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger = setup_logger(os.path.join(args.path, 'logs'), logfile_name=log_file)

    if rank == 0:
        for k in config._dict:
            logger.info("{}:{}".format(k, config._dict[k]))
        writer = SummaryWriter(os.path.join(args.path, 'tensorboard'))

    # init device
    if torch.cuda.is_available():
        config.device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        config.device = torch.device("cpu")

    # Instantiating autoencoder
    autoencoder = AE_Model(config)
    autoencoder.load_NewAE(config, model_path=config.autoencoder_path, logger=logger)
    autoencoder.to(config.device)
    autoencoder = torch.nn.parallel.DistributedDataParallel(autoencoder, device_ids=[args.local_rank],
                                                            find_unused_parameters=False)

    # load dataset
    train_dataset, test_dataset = load_dataset(config, args, "train")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=config.batch_size,
        num_workers=config.train_num_workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        num_workers=config.test_num_workers,
        drop_last=False,
    )

    BCE_criterion = nn.BCELoss()
    MSE_criterion = nn.MSELoss()

    num_epochs = 100
    autoencoder.train()
    num_batches = len(train_loader)
    index = 0
    log_iters = 0
    for epoch in range(num_epochs):
        index = epoch + 1
        if rank == 0:
            train_loader_tqdm = tqdm(train_loader, total=num_batches, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch_idx, items in enumerate(train_loader):
            log_iters += 1
            items = to_cuda(items, config.device)

            vm_crop_gt = items['vm_crop_gt']  # shape[B, 1, 256, 256]
            # pred_vm, latent_vm = autoencoder.module.VM_AE_Net(vm_crop_gt)  # outputs: shape[B, 1, 256, 256]  latent_vector: shape[B, 8, 6, 6]
            # loss_vm = MSE_criterion(pred_vm, vm_crop_gt.detach())
            loss_vm = None

            fm_crop_gt = items['fm_crop']
            # pred_fm, latent_am = autoencoder.module.AM_AE_Net(fm_crop_gt)
            # loss_fm = MSE_criterion(pred_fm, fm_crop_gt.detach())
            loss_fm = None

            # get edge_GT
            edge_mask_GT, _ = extract_boundary(fm_crop_gt, vm_crop_gt)

            # image = items['img_crop'].permute((0,3,1,2)).to(torch.float32)
            # autoencoder.module.visualization_4(image, fm_crop_gt, vm_crop_gt, edge_mask_GT, items, prefix='AE_COCOA')

            if torch.all(edge_mask_GT == 0):
                continue

            pred_edge, latent_edge = autoencoder.module.Edge_AE_Net(edge_mask_GT)
            pred_edge = F.interpolate(pred_edge, size=(256, 256), mode="nearest")
            pred_edge = torch.sigmoid(pred_edge)
            loss_edge = MSE_criterion(pred_edge, edge_mask_GT.detach())
            # loss_edge = None

            # autoencoder.module.visualization_4(image, fm_crop_gt, vm_crop_gt, edge_mask_GT, items, prefix='AE_COCOA')

            autoencoder.module.backward(loss_vm=loss_vm, loss_fm=loss_fm, loss_edge=loss_edge)

            # if rank == 0:
            #     if log_iters % config.log_idx == 0:
            #         # logger.debug(f"Epoch {index}, Batch {log_iters}, loss_vm: {loss_vm.item()}, loss_fm: {loss_fm.item()}, loss_edge: {loss_edge.item()}")
            #         logger.debug(f"Epoch {index}, Batch {log_iters}, loss_vm: {loss_fm.item()}")

            # loss_vm = get_avg_loss(loss_vm)
            # loss_fm = get_avg_loss(loss_fm)
            # loss_edge = get_avg_loss(loss_edge)
            torch.cuda.empty_cache()

            # if batch_idx == 4:
            #     break

            if rank == 0:
                train_loader_tqdm.update(1)

        if rank == 0:
            # writer.add_scalar('{}_loss/loss_vm'.format(config.dataset), loss_vm, index)
            # writer.add_scalar('{}_loss/loss_fm'.format(config.dataset), loss_fm, index)
            # writer.add_scalar('{}_loss/loss_edge'.format(config.dataset), loss_edge, index)
            train_loader_tqdm.close()

        if rank == 0:
            # [Codebook Build] Some old configs leave save_AE_epoch empty. In that case,
            # skip periodic AE checkpoint saving and still allow codebook construction.
            save_ae_epoch = getattr(config, 'save_AE_epoch', None)
            if save_ae_epoch is not None and save_ae_epoch > 0 and (index) % save_ae_epoch == 0:
                autoencoder.module.save(prefix='{}'.format(index))
                logger.info("AutoEncoder saved epoch:{}".format(index))

    if dist.is_initialized():
        dist.barrier()

    # [Codebook Build] The original codebook-training block below was commented out.
    # It is now re-enabled through this guarded call, while --skip-codebook-build keeps
    # the old "train AE only" behavior available.
    if not args.skip_codebook_build:
        build_shape_prior_codebook(config, train_dataset, autoencoder, args, logger, rank)

    if dist.is_initialized():
        dist.barrier()

