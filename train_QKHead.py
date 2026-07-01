import os
import cv2
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
from utils.utils import Config, Progbar, to_cuda, get_next_points
from utils.logger import setup_logger
from utils.loss import TripletLoss
from data.dataloader_transformer import load_dataset
from src.AE_model import AE_Model, extract_boundary, mask_recon_inference
from src.QKHead_model import QKHead_Model
from utils.wrappers import cat


def get_avg_loss(loss):
    # Just for mutil gpu in ddp mode
    world_size = dist.get_world_size()
    with torch.no_grad():
        if world_size >= 2:
            dist.all_reduce(loss)
            loss /= world_size
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    # path
    parser.add_argument('--path', type=str, required=True, help='model checkpoints path')
    parser.add_argument('--check_point_path', type=str, default="/home/univ/hongren/AIS_check_points")

    # training
    parser.add_argument('--Image_W', type=int, default=256)
    parser.add_argument('--Image_H', type=int, default=256)

    # dataset
    parser.add_argument('--dataset', type=str, default="KINS", help="select dataset")
    parser.add_argument('--data_type', type=str, default="image", help="select image or video model")
    parser.add_argument('--batch', type=int, default=16)

    parser.add_argument("--local-rank", default=0, type=int, help="node rank for distributed training")
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
    QKHead_Model = QKHead_Model(config)
    QKHead_Model.AE_net.load_NewAE(config, model_path=config.autoencoder_path, logger=logger)
    QKHead_Model.to(config.device)
    QKHead_Model = torch.nn.parallel.DistributedDataParallel(QKHead_Model, device_ids=[args.local_rank],
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

    TripletLoss = TripletLoss(margin=1)

    num_epochs = 100
    QKHead_Model.train()
    QKHead_Model.module.AE_net.eval()
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

            category_ids = items['category_id']
            vm_crop = items['vm_crop']
            fm_crop = items['fm_crop']

            points = torch.full((config.batch_size, 2, 3), -1, dtype=torch.float32)
            click_indices = torch.ones(config.batch_size, dtype=torch.float32)
            point = get_next_points(vm_crop, fm_crop, points, click_indices)

            resized_gt_mask = F.interpolate(fm_crop, size=(256, 256), mode='nearest')
            resized_vm_mask = F.interpolate(vm_crop, size=(256, 256), mode='nearest')

            with torch.no_grad():
                vectors_ae_mask = QKHead_Model.module.AE_net.AM_AE_Net.encode(resized_gt_mask)
                latent_vectors_ae_mask = vectors_ae_mask.view(vectors_ae_mask.shape[0], -1)

                # latent_vectors_gt_mask = latent_vectors_ae_mask + latent_vectors_gt_mask

            vectors_prev_mask = QKHead_Model.module.QKHead(resized_gt_mask)
            # vectors_prev_mask = QKHead_Model.module.QKHead(resized_vm_mask, point)  # Qhead  add point
            latent_vectors_prev_mask = vectors_prev_mask.view(vectors_prev_mask.shape[0], -1)

            latent_vectors_prev_mask = latent_vectors_ae_mask + latent_vectors_prev_mask

            # vm, prior_mask = self.net.AE_net.nearest_decode_Cm(latent_vectors_prev_mask, category_ids, k=self.k)
            # prior_mask, neg_masks_k, K_pos, K_neg  = QKHead_Model.module.AE_net.nearest_decode_head(QKHead_Model.module.QKHead, resized_gt_mask, latent_vectors_prev_mask, category_ids, k=config.QKHead_k)
            prior_mask, neg_masks_k, K_pos, K_neg = QKHead_Model.module.AE_net.nearest_decode_head(
                QKHead_Model.module.QKHead, resized_gt_mask, latent_vectors_prev_mask, category_ids, k=config.QKHead_k,
                point=point)  # add point
            prior_mask = F.interpolate(prior_mask, size=(256, 256), mode='nearest')

            # image_vis = items['img_crop'].permute((0, 3, 1, 2)).to(torch.float32).squeeze(0)
            # fm_mask_gt = items['fm_crop'].squeeze(0)
            # # pred_mask_change = (prev_mask > 0.5)
            # QKHead_Model.module.AE_net.save_visualization_QKHead(image_vis, fm_mask_gt, fm_mask_gt, prior_mask, items, prefix='{}_SP'.format(config.dataset), iteration=1)
            # QKHead_Model.module.AE_net.save_visualization_QKHead(image_vis, fm_mask_gt, fm_mask_gt, neg_masks_k, items, prefix='{}_SP'.format(config.dataset), iteration=2)

            loss = TripletLoss(latent_vectors_prev_mask, K_pos, K_neg)

            QKHead_Model.module.backward(loss)

            if rank == 0:
                if log_iters % config.log_idx == 0:
                    # logger.debug(f"Epoch {index}, Batch {log_iters}, loss_vm: {loss_vm.item()}, loss_fm: {loss_fm.item()}, loss_edge: {loss_edge.item()}")
                    logger.debug(f"Epoch {index}, Batch {log_iters}, loss: {loss.item()}")

            loss = get_avg_loss(loss)
            torch.cuda.empty_cache()

            # if batch_idx == 4:
            #     break

            if rank == 0:
                train_loader_tqdm.update(1)

        if rank == 0:
            writer.add_scalar('{}_loss/loss'.format(config.dataset), loss, index)
            train_loader_tqdm.close()

        if rank == 0:
            if (index) % config.QKHead_save_epoch == 0:  # config.save_QKHead_epoch
                QKHead_Model.module.save(prefix='{}'.format(index))
                logger.info("QKHead saved epoch:{}".format(index))

    if dist.is_initialized():
        dist.barrier()

    # autoencoder.eval()
    # if rank == 0:
    #     train_loader = tqdm(train_loader)

    # # logger.debug("test_loader长度：", len(test_loader))

    # with torch.no_grad():
    #     for id, items in enumerate(train_loader):
    #         mask_recon_inference(config, items, autoencoder)

    #         # if id == 4:
    #         #     break

    # values_array_sum = 0
    # for key, values_array in autoencoder.module.vector_dict.items():
    #     values_array_sum += values_array.shape[0]
    #     logger.debug(f"Key: {key}, Value shape: {values_array.shape}")
    #     print(f"Key: {key}, Value shape: {values_array.shape}")

    # logger.debug(f"类别数：{len(autoencoder.module.vector_dict)}, values_array_sum: {values_array_sum}")
    # print(f"类别数：{len(autoencoder.module.vector_dict)}, values_array_sum: {values_array_sum}")

    # # # logger.debug("类别个数：", len(autoencoder.module.vector_dict))
    # logger.info("Start KMEANS clustering")
    # autoencoder.module.cluster()
    # logger.info("KMEANS clustering has finished")
    # vector_dict_numpy = {
    # key: tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
    # for key, tensor in autoencoder.module.vector_dict.items()
    # }
    # np.save('{}/{}_codebook_NewAE_64.npy'.format(config.codebook_path, config.dataset), vector_dict_numpy)
    # logger.info("codebook saved")

    # # print("------------------------------------------------------------------")

    # values_array_sum = 0
    # for key, values_array in autoencoder.module.vector_dict.items():
    #     values_array_sum += values_array.shape[0]
    #     logger.debug(f"Key: {key}, Value shape: {values_array.shape}")
    #     print(f"Key: {key}, Value shape: {values_array.shape}")

    # logger.debug(f"类别数：{len(autoencoder.module.vector_dict)}, values_array_sum: {values_array_sum}")
    # print(f"类别数：{len(autoencoder.module.vector_dict)}, values_array_sum: {values_array_sum}")

    # 之前的代码
    #   category_ids = items['category_id']
    #         vm_crop = items['vm_crop']
    #         fm_crop = items['fm_crop']

    #         resized_gt_mask = F.interpolate(fm_crop, size=(256, 256), mode='nearest')

    #         with torch.no_grad():
    #             vectors_ae_mask = QKHead_Model.module.AE_net.AM_AE_Net.encode(resized_gt_mask)
    #             latent_vectors_ae_mask = vectors_ae_mask.view(vectors_ae_mask.shape[0], -1)

    #         vectors_prev_mask = QKHead_Model.module.QKHead(resized_gt_mask)  # Qhead
    #         latent_vectors_prev_mask = vectors_prev_mask.view(vectors_prev_mask.shape[0], -1)

    #         latent_vectors_prev_mask = latent_vectors_ae_mask + latent_vectors_prev_mask

    #         # vm, prior_mask = self.net.AE_net.nearest_decode_Cm(latent_vectors_prev_mask, category_ids, k=self.k)
    #         prior_mask, neg_masks_k, K_pos, K_neg  = QKHead_Model.module.AE_net.nearest_decode_head(QKHead_Model.module.QKHead, resized_gt_mask, latent_vectors_prev_mask, category_ids, k=config.QKHead_k)
    #         prior_mask = F.interpolate(prior_mask, size=(256, 256), mode='nearest')

    #         # image_vis = items['img_crop'].permute((0, 3, 1, 2)).to(torch.float32).squeeze(0)
    #         # fm_mask_gt = items['fm_crop'].squeeze(0)
    #         # # pred_mask_change = (prev_mask > 0.5)
    #         # QKHead_Model.module.AE_net.save_visualization_QKHead(image_vis, fm_mask_gt, fm_mask_gt, prior_mask, items, prefix='{}_SP'.format(config.dataset), iteration=1)
    #         # QKHead_Model.module.AE_net.save_visualization_QKHead(image_vis, fm_mask_gt, fm_mask_gt, neg_masks_k, items, prefix='{}_SP'.format(config.dataset), iteration=2)

    #         loss = TripletLoss(latent_vectors_prev_mask, K_pos, K_neg)