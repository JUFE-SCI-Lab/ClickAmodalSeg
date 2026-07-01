import os
import cv2
import time
import torch
import random
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from utils.utils import Config, Progbar, to_cuda
from utils.logger import setup_logger
from data.dataloader_transformer import load_dataset
from src.AE_net import mask_recon_inference
from src.AE_model import AE_Model
from utils.wrappers import cat


def extract_boundary(fm_mask, vm_mask):
    fm_mask = (fm_mask > 0.5).to(torch.float32)
    vm_mask = (vm_mask > 0.5).to(torch.float32)
    fm_mask = fm_mask.to(torch.bool)
    vm_mask = vm_mask.to(torch.bool)

    # 得到被遮挡区域的GT
    different_region = (fm_mask & (vm_mask == 0)).to(torch.bool)
    result_mask = torch.zeros_like(fm_mask, dtype=torch.float32)
    result_mask[different_region] = 1.0

    # 使用 F.max_pool2d 进行膨胀操作
    kernel_size = 9  # 膨胀核大小，例如 3x3
    padding = (kernel_size - 1) // 2  # 保证输出尺寸不变
    dilated_mask = F.max_pool2d(result_mask.float(), kernel_size, padding=padding, stride=1) > 0.5
    dilated_mask = dilated_mask.to(torch.float32)
    edge_mask = (dilated_mask * vm_mask).float()

    edge_mask = edge_mask.to(torch.float32)
    fm_mask = fm_mask.to(torch.float32)
    vm_mask = vm_mask.to(torch.float32)
    result_mask = result_mask.to(torch.float32)

    return edge_mask, result_mask


def get_IoU(pt_mask, gt_mask):
    # pred_mask  [N, Image_W, Image_H]
    # gt_mask   [N, Image_W, Image_H]
    pt_mask = pt_mask.squeeze()
    gt_mask = gt_mask.squeeze()

    pt_mask = (pt_mask > 0.5).to(torch.int64)
    gt_mask = (gt_mask > 0.5).to(torch.int64)

    pt_mask = pt_mask.unsqueeze(0)
    gt_mask = gt_mask.unsqueeze(0)

    SMOOTH = 1e-10
    intersection = (pt_mask & gt_mask).sum((-1, -2)).to(torch.float32)  # [N, 1]
    union = (pt_mask | gt_mask).sum((-1, -2)).to(torch.float32)  # [N, 1]

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # [N, 1]

    return iou


def batch_compute_iou(masks1, masks2):
    """
    批量计算IoU
    输入：
        masks1: [M, 1, H, W]
        masks2: [N, 1, H, W]
    返回：
        iou_matrix: [M, N]
    """
    device = masks1.device
    masks2 = masks2.to(device)

    M = masks1.size(0)
    N = masks2.size(0)

    # 展平mask
    masks1_flat = masks1.view(M, -1)  # [M, H*W]
    masks2_flat = masks2.view(N, -1)  # [N, H*W]

    # 计算交集 [M, N]
    intersection = torch.matmul(masks1_flat.float(), masks2_flat.t().float())  # [M, N]

    # 计算并集 [M, N]
    union = masks1_flat.sum(dim=1, keepdim=True) + masks2_flat.sum(dim=1, keepdim=True).t() - intersection  # [M, N]

    # 计算IoU [M, N]
    iou_matrix = intersection / (union + 1e-6)

    return iou_matrix


def convert_to_bgr(image):
    if image.shape[-1] == 3 and image[0, 0, 0] == image[0, 0, 2]:
        return image
    else:
        image = image.astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def visualize(meta, config):
    gt_vm = meta["vm_crop_gt"]
    gt_fm = meta["fm_crop"]

    edge_mask_GT, occ_mask_GT = extract_boundary(gt_fm, gt_vm)

    edge_mask_GT = (edge_mask_GT >= 0.5).to(torch.float32)
    edge_mask_GT = edge_mask_GT.squeeze()
    edge_mask_GT = edge_mask_GT.cpu().numpy()

    occ_mask_GT = (occ_mask_GT >= 0.5).to(torch.float32)
    occ_mask_GT = occ_mask_GT.squeeze()
    occ_mask_GT = occ_mask_GT.cpu().numpy()

    gt_vm = gt_vm.squeeze()
    gt_fm = gt_fm.squeeze()

    gt_vm = gt_vm.cpu().numpy()
    gt_fm = gt_fm.cpu().numpy()
    save_dir = os.path.join(config.VIS_PATH, 'AE_{}'.format(config.dataset))
    image_id, anno_id = meta["img_id"], meta["anno_id"]
    # plt.imsave("{}/{}_{}.png".format(save_dir, int(image_id.item()), int(anno_id.item())), pred_fm)
    plt.imsave("{}/{}_{}_vm_GT.png".format(save_dir, int(image_id.item()), int(anno_id.item())), gt_vm)
    plt.imsave("{}/{}_{}_am_GT.png".format(save_dir, int(image_id.item()), int(anno_id.item())), gt_fm)
    plt.imsave("{}/{}_{}_edge_GT.png".format(save_dir, int(image_id.item()), int(anno_id.item())), edge_mask_GT)
    plt.imsave("{}/{}_{}_occ_GT.png".format(save_dir, int(image_id.item()), int(anno_id.item())), occ_mask_GT)


def overlay_mask_on_image(image, items, config):
    output_images_path = os.path.join(config.VIS_PATH, 'AE_{}'.format(config.dataset))
    output_images_path = Path(output_images_path)

    if not output_images_path.exists():
        output_images_path.mkdir(parents=True)

    img_id = int(items['img_id'].item())
    anno_id = int(items['anno_id'].item())

    print(f"img_id: {img_id}, anno_id: {anno_id}")

    def _save_image(suffix, image):
        cv2.imwrite(str(output_images_path / f'{img_id}_{anno_id}_{suffix}.jpg'),
                    image, [cv2.IMWRITE_JPEG_QUALITY, 85])

    image = image.squeeze(0)
    image = image.cpu().numpy() * 255
    image = image.transpose((1, 2, 0))
    # image = convert_to_bgr(image)  # KINS需要变换

    vm_mask_dir = os.path.join(output_images_path, "{}_{}_vm_GT.png".format(img_id, anno_id))
    vm_gt = np.array(Image.open(vm_mask_dir).convert("L"))
    vm_gt = (vm_gt == 215)

    am_mask_dir = os.path.join(output_images_path, "{}_{}_am_GT.png".format(img_id, anno_id))
    am_gt = np.array(Image.open(am_mask_dir).convert("L"))
    am_gt = (am_gt == 215)

    edge_mask_dir = os.path.join(output_images_path, "{}_{}_edge_GT.png".format(img_id, anno_id))
    edge_gt = np.array(Image.open(edge_mask_dir).convert("L"))
    edge_gt = (edge_gt == 215)

    occ_mask_dir = os.path.join(output_images_path, "{}_{}_occ_GT.png".format(img_id, anno_id))
    occ_gt = np.array(Image.open(occ_mask_dir).convert("L"))
    occ_gt = (occ_gt == 215)

    color1 = [0, 0, 200]
    color2 = np.clip(np.array(color1) + 35, 0, 255)
    vm_gt = add_mask(vm_gt, image, color1, color2, 2)
    am_gt = add_mask(am_gt, image, color1, color2, 2)

    occ_color1 = np.array([0, 0, 255])
    am_gt = add_occ(occ_gt, am_gt, occ_color1, color2, 2)

    edge_color1 = [255, 0, 0]
    edge_color2 = np.array(edge_color1)
    edge_gt_vis = add_mask(edge_gt, image, edge_color1, edge_color2, 2)

    vm_edge_gt = add_mask(edge_gt, vm_gt, edge_color1, edge_color2, 2)

    viz_image = np.hstack((image, vm_gt, edge_gt_vis, vm_edge_gt, am_gt)).astype(np.uint8)

    _save_image('overlay', viz_image[:, :, ::-1])


def add_mask(mask, img, color1, color_mask=np.array([0, 0, 255]), line_width=1):
    mask = mask.astype(bool)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    res = cv2.drawContours(img.copy(), contours, -1, color1, line_width)
    res[mask] = res[mask] * 0.7 + color_mask * 0.3
    return res


def add_occ(mask, img, color1, color_mask=np.array([0, 0, 255]), line_width=1):
    mask = mask.astype(bool)
    # contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # res = cv2.drawContours(img.copy(), contours, -1, color1, line_width)
    res = img
    res[mask] = res[mask] * 0.3 + color1 * 0.7
    return res


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
    # autoencoder.load_AE(config, model_path=config.codebook_path, logger=logger)
    autoencoder.to(config.device)
    autoencoder = torch.nn.parallel.DistributedDataParallel(autoencoder, device_ids=[args.local_rank],
                                                            find_unused_parameters=True)

    # load dataset
    train_dataset, test_dataset = load_dataset(config, args, "train")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=config.batch_size,
        num_workers=config.train_num_workers,
        drop_last=True,
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(
        dataset=test_dataset,
        sampler=test_sampler,
        batch_size=1,
        num_workers=config.test_num_workers,
        drop_last=False,
    )

    iter = 0
    iou = 0
    iou_fm = 0
    iou_vm = 0
    iou_edge = 0
    iou_count = 0
    mean_priors_ious = 0

    autoencoder.eval()
    if rank == 0:
        test_loader = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, items in enumerate(test_loader):
            iter += 1
            items = to_cuda(items, config.device)
            # inputs = items['fm_crop']

            # fm_crop_gt = items['fm_crop']
            # # pred_fm, latent_fm = autoencoder.module.AM_AE_Net(fm_crop_gt)

            # vm_crop_gt = items['vm_crop_gt']  # shape[B, 1, 256, 256]
            # # pred_vm, latent_vm = autoencoder.module.VM_AE_Net(vm_crop_gt)  # outputs: shape[B, 1, 256, 256]  latent_vector: shape[B, 8, 6, 6]

            # # get edge_GT
            # edge_mask_GT, _ = extract_boundary(fm_crop_gt, vm_crop_gt)

            # pred_edge, latent_edge = autoencoder.module.Edge_AE_Net(edge_mask_GT)
            # pred_edge = F.interpolate(pred_edge, size=(256, 256), mode="nearest")
            # pred_edge = torch.sigmoid(pred_edge)

            # image = items['img_crop'].permute((0,3,1,2)).to(torch.float32)

            # visualize(items, config)
            # overlay_mask_on_image(image, items, config)

            # if batch_idx == 4000:
            #     break

            # # old shape priors
            # with torch.no_grad():
            #     category_ids = items['category_id']
            #     vectors = autoencoder.module.VM_AE_Net.encode(pred_vm)
            #     vectors = vectors.view(vectors.shape[0], -1)
            #     vm, prior_mask = autoencoder.module.nearest_decode_old(vectors, category_ids, k=config.k)

            # # new shape priors
            # with torch.no_grad():
            #     category_ids = items['category_id']
            #     vectors_vm = autoencoder.module.VM_AE_Net.encode(vm_crop_gt)
            #     vectors_edge = autoencoder.module.Edge_AE_Net.encode(edge_mask_GT)

            #     latent_vm_flat = vectors_vm.view(vectors_vm.size(0), -1)
            #     latent_edge_flat = vectors_edge.view(vectors_edge.size(0), -1)
            #     combined_latent = torch.cat((latent_vm_flat, latent_edge_flat), dim=1)
            #     vm, prior_mask = autoencoder.module.nearest_decode(combined_latent, category_ids, k=config.k)

            # image_id, anno_id= items["img_id"], items["anno_id"]
            # if int(image_id.item()) == 1101 and int(anno_id.item()) == 95:
            #     autoencoder.module.save_visualization_shape_priors_vm(image, fm_crop_gt, vm_crop_gt, edge_mask_GT, vm, prior_mask, items, prefix='AE_{}'.format(config.dataset))
            #     break

            # autoencoder.module.save_visualization(image, fm_crop_gt, pred_fm, vm_crop_gt, pred_vm, edge_mask_GT, pred_edge, items, prefix='AE_D2SA')
            # autoencoder.module.visualization_5(image, fm_crop_gt, pred_fm, vm_crop_gt, pred_vm, items, prefix='AE_COCOA')
            # autoencoder.module.visualization_5(image, fm_crop_gt, vm_crop_gt, edge_mask_GT, pred_edge, items, prefix='AE_D2SA')
            # autoencoder.module.save_visualization_shape_priors(image, pred_fm, pred_vm, pred_edge, prior_mask, items, prefix='AE_{}'.format(config.dataset))
            # autoencoder.module.save_visualization_shape_priors_vm(image, fm_crop_gt, vm_crop_gt, edge_mask_GT, vm, prior_mask, items, prefix='AE_{}'.format(config.dataset))
            # autoencoder.module.save_visualization_shape_priors_vm_old(image, pred_fm, pred_vm, vm, prior_mask, items, prefix='AE_{}'.format(config.dataset))
            # print("-----------------------------------------")

            with torch.no_grad():
                category_ids = items['category_id']  # torch.Size([1])

                fm_crop = items['fm_crop']  # torch.Size([1, 1, 256, 256])
                resized_prev_mask = F.interpolate(fm_crop, size=(256, 256), mode='nearest')
                vectors_coarse_am = autoencoder.module.AM_AE_Net.encode(resized_prev_mask)

                latent_vectors_coarse_am = vectors_coarse_am.view(vectors_coarse_am.shape[0], -1)
                vm, prior_masks = autoencoder.module.nearest_decode_Cm(latent_vectors_coarse_am, category_ids,
                                                                       k=config.QKHead_k)
                # prior_mask = self.net.AE_net.nearest_decode_sim(self.net.QKHead, latent_vectors_coarse_am, resized_prev_mask, category_ids, k=self.k)
                # self.prior_masks = self.net.AE_net.nearest_decode_L2(self.net.QKHead, latent_vectors_coarse_am, resized_prev_mask, category_ids, k=self.k)
                prior_masks = F.interpolate(prior_masks, size=(256, 256), mode='nearest')

                prior_masks_squeezed = prior_masks.squeeze(0).unsqueeze(1)  # [10, 1, 256, 256]

                priors_ious = batch_compute_iou(fm_crop, prior_masks_squeezed)  # [1, 10]
                mean_priors_iou = priors_ious.mean().item()  # 最终平均IoU值

                mean_priors_ious += mean_priors_iou

            logger.info('Rank: {}, iter: {}, mean_priors_iou: {}'.format(
                rank,
                iter,
                mean_priors_iou,
            ))
            dist.barrier()
            torch.cuda.empty_cache()

            # if iter==4:
            #     break

    # dist.all_reduce(mean_priors_ious)
    # dist.all_reduce(iter)
    # dist.barrier()
    if rank == 0:
        logger.info('mean_priors_ious: {}'.format(mean_priors_ious / iter))
        logger.info('iter: {}'.format(iter))

    #         loss_eval_1 = autoencoder.module.get_IoU_1(pred_edge, edge_mask_GT)
    #         loss_eval = autoencoder.module.get_IoU_2(fm_crop_gt, pred_fm, vm_crop_gt, pred_vm)
    #         iter += 1
    #         iou_fm += loss_eval['iou_fm']
    #         iou_vm += loss_eval['iou_vm']
    #         iou_edge += loss_eval_1['iou_edge']
    #         iou_count += loss_eval_1['iou_count']

    #         logger.info('Rank: {}, iter: {}, iou_fm: {}, iou_vm: {}, iou_edge: {}'.format(
    #             rank,
    #             iter-1,
    #             loss_eval['iou_fm'].item(),
    #             loss_eval['iou_vm'].item(),
    #             loss_eval_1['iou_edge'].item(),
    #         ))
    #         dist.barrier()
    #         torch.cuda.empty_cache()

    #         # if iter==4:
    #         #     break

    # dist.all_reduce(iou_fm)
    # dist.all_reduce(iou_vm)
    # dist.all_reduce(iou_edge)
    # dist.all_reduce(iou_count)
    # dist.barrier()
    # if rank==0:
    #     logger.info('meanIoU_fm: {}'.format(iou_fm.item() / iou_count.item()))
    #     logger.info('meanIoU_vm: {}'.format(iou_vm.item() / iou_count.item()))
    #     logger.info('meanIoU_edge: {}'.format(iou_edge.item() / iou_count.item()))
    #     logger.info('iou_count: {}'.format(iou_count))

    #         # loss_eval = autoencoder.module.get_IoU_1(pred_edge, edge_mask_GT)
    #         loss_eval = autoencoder.module.get_IoU_2(fm_crop_gt, pred_fm, vm_crop_gt, pred_vm)
    #         iter += 1
    #         iou_fm += loss_eval['iou_fm']
    #         iou_vm += loss_eval['iou_vm']
    #         # iou_edge += loss_eval['iou_edge']
    #         iou_count += loss_eval['iou_count']

    #         logger.info('Rank {}, iter {}: iou_fm: {}, iou_vm: {}'.format(
    #             rank,
    #             iter-1,
    #             loss_eval['iou_fm'].item(),
    #             loss_eval['iou_vm'].item(),
    #             # loss_eval['iou_edge'].item(),
    #         ))
    #         dist.barrier()
    #         torch.cuda.empty_cache()

    # dist.all_reduce(iou_fm)
    # dist.all_reduce(iou_vm)
    # # dist.all_reduce(iou_edge)
    # dist.all_reduce(iou_count)
    # dist.barrier()
    # if rank==0:
    #     logger.info('meanIoU_fm: {}'.format(iou_fm.item() / iou_count.item()))
    #     logger.info('meanIoU_vm: {}'.format(iou_vm.item() / iou_count.item()))
    #     # logger.info('meanIoU_edge: {}'.format(iou_edge.item() / iou_count.item()))
    #     logger.info('iou_count: {}'.format(iou_count))
