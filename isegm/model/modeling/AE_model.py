import os
import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

from pathlib import Path
from sklearn.cluster import KMeans
from torchvision import transforms
from .AutoEncoder import VM_AE_Net, AM_AE_Net, Edge_AE_Net   
from isegm.utils.utils import torch_init_model
from isegm.utils.vis import draw_probmap
from isegm.utils.wrappers import cat
from isegm.utils.utils import Config, Progbar, to_cuda
from isegm.inference import utils

def get_IoU(pt_mask, gt_mask):
    # pred_mask  [N, Image_W, Image_H]
    # gt_mask   [N, Image_W, Image_H]
    SMOOTH = 1e-10
    intersection = (pt_mask & gt_mask).sum((-1, -2)).to(torch.float32) # [N, 1]
    union = (pt_mask | gt_mask).sum((-1, -2)).to(torch.float32) # [N, 1]

    iou = (intersection + SMOOTH) / (union + SMOOTH) # [N, 1]

    return iou

def evaluation_image(frame_pred, frame_label):
    frame_pred = (frame_pred > 0.5).to(torch.int64)
    frame_label = frame_label.to(torch.int64)
    frame_pred = frame_pred.unsqueeze(0)
    frame_label = frame_label.unsqueeze(0)

    iou_ = get_IoU(frame_pred, frame_label)
   
    return iou_.sum()

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

    return edge_mask, result_mask

@torch.no_grad()
def mask_recon_inference(config, items, recon_net):
    vector_dict = {}
    classes = []
    items = to_cuda(items, config.device)
    inputs = items['fm_crop']  # shape[B, 1, 256, 256]
    classes.append(items['category_id'])
    classes = cat(classes, dim=0)

    fm_crop_gt = items['fm_crop']  
    pred_fm, latent_am = recon_net.module.AM_AE_Net((fm_crop_gt > 0.5).float()) 
    latent_am_flat = torch.flatten(latent_am)


    vm_crop_gt = items['vm_crop_gt']  # shape[B, 1, 256, 256]
    pred_vm, latent_vm = recon_net.module.VM_AE_Net((vm_crop_gt > 0.5).float())
    latent_vm_flat = torch.flatten(latent_vm)

    edge_mask_GT, _ = extract_boundary(fm_crop_gt, vm_crop_gt)

    pred_edge, latent_edge = recon_net.module.Edge_AE_Net((edge_mask_GT > 0.5).float())
    latent_edge_flat = torch.flatten(latent_edge)

    combined_latent = torch.cat((latent_vm_flat, latent_edge_flat, latent_am_flat), dim=0)

    # recon_outputs = recon_net.module.decode(latent_vectors)

    # recon_net.module.save_visualization(outputs, recon_outputs, items, prefix='AE_COCOA')

    for i in range(len(classes.unique())):
        index = (classes == classes.unique()[i].item()).nonzero()
        vector_dict[classes.unique()[i].item()] = combined_latent.view(len(index), -1)

    recon_net.module.recording_vectors(vector_dict)

class AE_Model(nn.Module):
    def __init__(self, config):
        super(AE_Model, self).__init__()
        self.config = config
        self.conv_dims = config.CONV_DIM
        self.num_classes = config.NUM_CLASSES
        self.num_cluster = config.KMEANS
        self.vector_dict = {}

        self.VM_AE_Net = VM_AE_Net(config).to(config.device)
        self.AM_AE_Net = AM_AE_Net(config).to(config.device)
        self.Edge_AE_Net = Edge_AE_Net(config).to(config.device)

        self.VM_optimizer = optim.Adam(self.VM_AE_Net.parameters(), lr=1e-4)
        self.AM_optimizer = optim.Adam(self.AM_AE_Net.parameters(), lr=1e-4)
        self.Edge_optimizer = optim.Adam(self.Edge_AE_Net.parameters(), lr=1e-4)
        # self.Edge_optimizer = optim.SGD(self.Edge_AE_Net.parameters(), lr=0.001, momentum=0.9)



    def backward_step(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def backward(self, loss_vm, loss_fm, loss_edge):
        # self.backward_step(self.VM_optimizer, loss_vm)
        # self.backward_step(self.AM_optimizer, loss_fm)
        self.backward_step(self.Edge_optimizer, loss_edge)


    def recording_vectors(self, vector_inference):
        for key, item in vector_inference.items():
            if self.vector_dict.get(key, None) is not None:
                self.vector_dict[key] = cat([self.vector_dict[key], item], dim=0)
            else:
                self.vector_dict[key] = item

    # vectors : torch.Size([B, 288]),  category_ids : torch.Size([1]) , 288 = 8 x 6 x6 就是通过编码器后的 C x H x W
    def nearest_decode(self, vectors, pred_classes, k=16):
        # side_len = math.sqrt(fm_vectors.size(1) / self.conv_dims)  # side_len : 6
        side_len = 6
        assert side_len % 1 == 0
        memo_latent_vectors = torch.zeros((vectors.size(0), k, 288)).to(vectors.device) # memo_latent_vectors ：torch.Size([B, 16, 288])
        vm_latent_vectors = torch.zeros((vectors.size(0), k, 288)).to(vectors.device)

        classes_lst = pred_classes.unique()  # torch.Size([B])
        for i in range(len(classes_lst)):
            with torch.no_grad():
                index = (pred_classes == pred_classes.unique()[i].item()).nonzero() 
                vectors_per_classes = vectors[index[:, 0]]  # torch.Size([1, 800])
                if pred_classes.unique()[i].item() in self.vector_dict:
                    codebook_all = self.vector_dict[pred_classes.unique()[i].item()].to(vectors.device)  # torch.Size([1024, 1088])
                    codebook = codebook_all[:, :800]  # vm + edge   torch.Size([1024, 800])
                    codebook_fm = codebook_all[:, 800:]  # am  torch.Size([1024, 288])
  
                    codebook_sqr = torch.sum(codebook ** 2, dim=1)  # torch.Size([1024])
                    inputs_sqr = torch.sum(vectors_per_classes ** 2, dim=1, keepdim=True)  # torch.Size([1, 1])

                    # Compute the distances to the codebook
                    distances = torch.addmm(codebook_sqr + inputs_sqr,   # torch.Size([1, 1024])
                                            vectors_per_classes, codebook.t(), alpha=-2.0, beta=1.0)

                    indices = torch.topk(- distances, k)[1]  # torch.Size([1, 16])
                    nn_vectors = codebook_fm[indices]  # torch.Size([1, 16, 288])
                    memo_latent_vectors[index[:, 0]] = nn_vectors  # torch.Size([1, 16, 288])

                    vm_vectors = codebook[:, :288][indices]
                    vm_latent_vectors[index[:, 0]] = vm_vectors

                else:
                    vectors_per_classes = vectors_per_classes[:, :288]
                    memo_latent_vectors[index[:, 0]] = vectors_per_classes.unsqueeze(1)

                    vm_vectors = vectors_per_classes
                    vm_latent_vectors[index[:, 0]] = vm_vectors.unsqueeze(1)

        vectors = memo_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))  # torch.Size([16, 8, 6, 6])
        vm_vectors = vm_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))

        # for layer in self.decoder:
        #     vectors = layer(vectors)
        # x = self.outconv(vectors)
        x = self.AM_AE_Net.decode(vectors)
        x = x.view(pred_classes.size(0), k, x.size(2), x.size(3))

        vm = self.VM_AE_Net.decode(vm_vectors)
        vm = vm.view(pred_classes.size(0), k, vm.size(2), vm.size(3))
        return vm, x
    
    def nearest_decode_head(self, QKHead, gt_mask, vectors, pred_classes, k=10):
        side_len = 6
        assert side_len % 1 == 0

        # Initialize output tensors
        masks_vectors = torch.zeros((vectors.size(0), k * 2, 1, gt_mask.size(2), gt_mask.size(3)), 
                                device=vectors.device)  # [B, 2k, 1, 256, 256]
        
        # Get unique classes
        classes_lst = pred_classes.unique()  # torch.Size([C]), where C is number of unique classes
        
        for i in range(len(classes_lst)):
            with torch.no_grad():
                # Get indices of all samples with this class
                class_idx = classes_lst[i].item()
                index = (pred_classes == class_idx).nonzero()[:, 0]  # [M], where M is number of samples with this class
                
                # Get the first gt_mask for this class (since all samples of same class share the same processing)
                gt_mask_per = gt_mask[index].to(vectors.device)  # [M, 1, 256, 256] M is the number of elements in the same category
                
                if class_idx in self.vector_dict:
                    codebook_all = self.vector_dict[class_idx].to(vectors.device)  # [N, 1088]
                    codebook = codebook_all[:, :800]
                    codebook_fm = codebook_all[:, 800:]  # [N, 288]

                    mask_all = codebook_fm.view(codebook_all.size(0), self.conv_dims, int(side_len), int(side_len))  # [N, 8, 6, 6] N is the number of masks of a certain category in the bank
                    with torch.no_grad():
                        gt_mask_all = self.AM_AE_Net.decode(mask_all)  # [N, 1, 256, 256]
                    gt_mask_all = (gt_mask_all > 0.5).to(torch.float32)
                    gt_mask_all = gt_mask_all.view(1, codebook_all.size(0), gt_mask_all.size(2), gt_mask_all.size(3))  # [1, N, 256, 256] 
                    gt_mask_all = gt_mask_all.to(vectors.device)

                    pos_idx, neg_idx = select_samples(gt_mask_per, gt_mask_all, k=k, th=0.8) # [M, k], [M, k]
                    pos_masks, neg_masks = extract_samples(gt_mask_all, pos_idx, neg_idx)  # [M, k, 1, 256, 256]

                    # Assign to all samples of this class
                    masks_vectors[index, :k] = pos_masks # [B, k, 1, 256, 256]
                    masks_vectors[index, k:] = neg_masks # [B, k, 1, 256, 256]
                else:
                    # If no matching class, use gt_mask_per for pos and zeros for neg
                    masks_vectors[index, :k] = gt_mask_per.unsqueeze(0).expand(1, k, 1, gt_mask.size(2), gt_mask.size(3))
                    masks_vectors[index, k:] = torch.zeros(len(index), k, 1, gt_mask.size(2), gt_mask.size(3), device=vectors.device)

        # Process all masks at once
        masks_Khead = QKHead(
            masks_vectors.view(vectors.size(0) * 2 * k, 1, gt_mask.size(2), gt_mask.size(3))  # [B * 2k, 1, 256, 256]
        )  # [B * 2k, 8, 6, 6]

        # Separate pos and neg results
        vectors_masks_Khead = masks_Khead.view(vectors.size(0), 2 * k, -1)  # [B, 2k, 288]
        vectors_pos_masks_Khead = vectors_masks_Khead[:, :k, :]  # [B, k, 288]
        vectors_neg_masks_Khead = vectors_masks_Khead[:, k:, :]  # [B, k, 288]

        # Return pos_masks_k (first k)
        pos_masks_k = masks_vectors[:, :k].view(vectors.size(0), k, gt_mask.size(2), gt_mask.size(3))  # [B, k, 256, 256]

        neg_masks_k = masks_vectors[:, k:].view(vectors.size(0), k, gt_mask.size(2), gt_mask.size(3))

        return pos_masks_k, neg_masks_k,vectors_pos_masks_Khead, vectors_neg_masks_Khead
    
    def nearest_decode_sim(self, QKHead, vectors, prev_mask, pred_classes, k=16):
        side_len = 6  
        device = vectors.device
        
        masks_pos = torch.zeros((vectors.size(0), k, prev_mask.size(2), prev_mask.size(3)), 
                            device=device)

        classes_lst = pred_classes.unique()  # torch.Size([C]), C是类别数
        
        for class_idx in classes_lst:
            class_mask = (pred_classes == class_idx)
            index = class_mask.nonzero().squeeze(-1)  # [M]
            
            if not index.numel():  
                continue
                
            vectors_per_class = vectors[index]  # [M, D]
            
            if class_idx.item() not in self.vector_dict:
                masks_pos[index] = prev_mask[index].squeeze(1).unsqueeze(1).expand(-1, k, -1, -1)
                continue
                
            codebook_all = self.vector_dict[class_idx.item()].to(device)  # [N, D_total]
            codebook_fm = codebook_all[:, 800:] 
            
            mask_all = codebook_fm.view(-1, self.conv_dims, side_len, side_len)  # [N, 8, 6, 6]
            gt_mask_all = self.AM_AE_Net.decode(mask_all)  # [N, 1, 256, 256]
            
            gt_mask_all_vectors = QKHead(gt_mask_all).flatten(1)  # [N, 288]
            
            cosine_sim = F.cosine_similarity(
                vectors_per_class.unsqueeze(1),  # [M, 1, D]
                gt_mask_all_vectors.unsqueeze(0),  # [1, N, D]
                dim=2
            )
            
            _, indices = torch.topk(cosine_sim, k, dim=1)
            
            nn_masks = gt_mask_all[indices].squeeze(2)  # [M, k, 1, H, W] -> [M, k, H, W]
            masks_pos[index] = nn_masks

        return masks_pos
    

    def nearest_decode_L2(self, QKHead, vectors, prev_mask, pred_classes, k=16, point=None):
        side_len = 6  
        device = vectors.device
        prev_mask = prev_mask.float()
        
        masks_pos = torch.zeros((vectors.size(0), k, prev_mask.size(2), prev_mask.size(3)), 
                            device=device)

        # prior_features = torch.zeros((vectors.size(0), k * 8, 6, 6), 
        #                     device=device)

        classes_lst = pred_classes.unique()  # torch.Size([C]), C是类别数
        
        for class_idx in classes_lst:
            class_mask = (pred_classes == class_idx)
            index = class_mask.nonzero().squeeze(-1)  # [M]
            
            if not index.numel():  
                continue
                
            vectors_per_class = vectors[index]  # [M, D]
            
            if class_idx.item() not in self.vector_dict:
                masks_pos[index] = prev_mask[index].squeeze(1).unsqueeze(1).expand(-1, k, -1, -1)
                continue
                
            codebook_all = self.vector_dict[class_idx.item()].to(device)  # [N, D_total]
            codebook_fm = codebook_all[:, 800:] 
            
            mask_all = codebook_fm.view(-1, self.conv_dims, side_len, side_len)  # [N, 8, 6, 6]
            gt_mask_all = self.AM_AE_Net.decode(mask_all)  # [N, 1, 256, 256]
            
            mask_all_vectors = mask_all.flatten(1)  # F_k  [N, 288]  
            gt_mask_all_vectors = QKHead(gt_mask_all).flatten(1)
            # gt_mask_all_vectors = QKHead(gt_mask_all, point=None).flatten(1)  # D_k [N, 288] add point

            gt_mask_all_vectors = mask_all_vectors + gt_mask_all_vectors    # F_k + D_k  [N, 288] 
            
            # 计算平方欧氏距离 [M, N]
            distances = torch.cdist(
                vectors_per_class,          # [M, D]
                gt_mask_all_vectors,        # [N, D]
                p=2                         # L2距离
            ) ** 2                          # 平方距离
            
            # 取距离最小的k个索引（欧氏距离越小越相似）
            _, indices = torch.topk(-distances, k, dim=1)  
            
            nn_masks = gt_mask_all[indices].squeeze(2)  # [M, k, 1, H, W] -> [M, k, H, W]
            masks_pos[index] = nn_masks

            # prior_feature = mask_all[indices].reshape(-1, k * 8, 6, 6)  # [M, K*8, 6, 6]
            # prior_features[index] = prior_feature

        return masks_pos
    
    def nearest_decode_Cm(self, vectors, pred_classes, k=16):
        # side_len = math.sqrt(fm_vectors.size(1) / self.conv_dims)  # side_len : 6
        side_len = 6
        assert side_len % 1 == 0
        memo_latent_vectors = torch.zeros((vectors.size(0), k, 288)).to(vectors.device) # torch.Size([B, 16, 288])
        vm_latent_vectors = torch.zeros((vectors.size(0), k, 288)).to(vectors.device)

        classes_lst = pred_classes.unique()  # torch.Size([B])
        for i in range(len(classes_lst)):
            with torch.no_grad():
                index = (pred_classes == pred_classes.unique()[i].item()).nonzero() 
                vectors_per_classes = vectors[index[:, 0]]  # torch.Size([1, 288])
                if pred_classes.unique()[i].item() in self.vector_dict:
                    codebook_all = self.vector_dict[pred_classes.unique()[i].item()].to(vectors.device)  # torch.Size([1024, 288])
                    codebook = codebook_all[:, :800]
                    codebook_fm = codebook_all[:, 800:]

                    codebook_sqr = torch.sum(codebook_fm ** 2, dim=1)  # torch.Size([1024])
                    inputs_sqr = torch.sum(vectors_per_classes ** 2, dim=1, keepdim=True)  # torch.Size([1, 1])

                    # Compute the distances to the codebook
                    distances = torch.addmm(codebook_sqr + inputs_sqr,   # torch.Size([1, 1024])
                                            vectors_per_classes, codebook_fm.t(), alpha=-2.0, beta=1.0)

                    indices = torch.topk(- distances, k)[1]  # torch.Size([1, 16])
                    nn_vectors = codebook_fm[indices]  # torch.Size([1, 16, 288])
                    memo_latent_vectors[index[:, 0]] = nn_vectors  # torch.Size([1, 16, 288])

                    vm_vectors = codebook[:, :288][indices]
                    vm_latent_vectors[index[:, 0]] = vm_vectors

                else:
                    vectors_per_classes = vectors_per_classes[:, :288]
                    memo_latent_vectors[index[:, 0]] = vectors_per_classes.unsqueeze(1)

                    vm_vectors = vectors_per_classes
                    vm_latent_vectors[index[:, 0]] = vm_vectors.unsqueeze(1)

        vectors = memo_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))  # torch.Size([16, 8, 6, 6])
        vm_vectors = vm_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))

        # for layer in self.decoder:
        #     vectors = layer(vectors)
        # x = self.outconv(vectors)
        x = self.AM_AE_Net.decode(vectors)
        x = x.view(pred_classes.size(0), k, x.size(2), x.size(3))

        vm = self.VM_AE_Net.decode(vm_vectors)
        vm = vm.view(pred_classes.size(0), k, vm.size(2), vm.size(3))
        return vm, x
    
    def nearest_decode_Edge(self, vectors, pred_classes, k=16):
        # side_len = math.sqrt(fm_vectors.size(1) / self.conv_dims)  # side_len : 6
        side_len = 6
        assert side_len % 1 == 0
        memo_latent_vectors = torch.zeros((vectors.size(0), k, 288)).to(vectors.device) # memo_latent_vectors ：torch.Size([B, 16, 288])
        vm_latent_vectors = torch.zeros((vectors.size(0), k, 288)).to(vectors.device)

        classes_lst = pred_classes.unique()  # torch.Size([B])
        for i in range(len(classes_lst)):
            with torch.no_grad():
                index = (pred_classes == pred_classes.unique()[i].item()).nonzero() 
                vectors_per_classes = vectors[index[:, 0]]  # torch.Size([1, 288])
                if pred_classes.unique()[i].item() in self.vector_dict:
                    codebook_all = self.vector_dict[pred_classes.unique()[i].item()].to(vectors.device)  # torch.Size([1024, 1088])
                    codebook = codebook_all[:, 288:800]
                    codebook_fm = codebook_all[:, 800:]

                    codebook_sqr = torch.sum(codebook ** 2, dim=1)  # torch.Size([1024])
                    inputs_sqr = torch.sum(vectors_per_classes ** 2, dim=1, keepdim=True)  # torch.Size([1, 1])

                    # Compute the distances to the codebook
                    distances = torch.addmm(codebook_sqr + inputs_sqr,   # torch.Size([1, 1024])
                                            vectors_per_classes, codebook.t(), alpha=-2.0, beta=1.0)

                    indices = torch.topk(- distances, k)[1]  # torch.Size([1, 16])
                    nn_vectors = codebook_fm[indices]  # torch.Size([1, 16, 288])
                    memo_latent_vectors[index[:, 0]] = nn_vectors  # torch.Size([1, 16, 288])

                    vm_vectors = codebook[:, :288][indices]
                    vm_latent_vectors[index[:, 0]] = vm_vectors

                else:
                    vectors_per_classes = vectors_per_classes[:, :288]
                    memo_latent_vectors[index[:, 0]] = vectors_per_classes.unsqueeze(1)

                    vm_vectors = vectors_per_classes
                    vm_latent_vectors[index[:, 0]] = vm_vectors.unsqueeze(1)

        vectors = memo_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))  # torch.Size([16, 8, 6, 6])
        vm_vectors = vm_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))

        # for layer in self.decoder:
        #     vectors = layer(vectors)
        # x = self.outconv(vectors)
        x = self.AM_AE_Net.decode(vectors)
        x = x.view(pred_classes.size(0), k, x.size(2), x.size(3))

        vm = self.VM_AE_Net.decode(vm_vectors)
        vm = vm.view(pred_classes.size(0), k, vm.size(2), vm.size(3))
        return vm, x

    # vectors : torch.Size([B, 288]),  category_ids : torch.Size([1]) , 288 = 8 x 6 x6 就是通过编码器后的 C x H x W
    def nearest_decode_VmEdge(self, vectors, pred_classes, k=16):
        # side_len = math.sqrt(fm_vectors.size(1) / self.conv_dims)  # side_len : 6
        side_len = 6
        assert side_len % 1 == 0
        memo_latent_vectors = torch.zeros((vectors.size(0), k, 288)).to(vectors.device) # memo_latent_vectors ：torch.Size([B, 16, 288])
        vm_latent_vectors = torch.zeros((vectors.size(0), k, 288)).to(vectors.device)

        classes_lst = pred_classes.unique()  # torch.Size([B])
        for i in range(len(classes_lst)):
            with torch.no_grad():
                index = (pred_classes == pred_classes.unique()[i].item()).nonzero() 
                vectors_per_classes = vectors[index[:, 0]]  # torch.Size([1, 288])
                if pred_classes.unique()[i].item() in self.vector_dict:
                    codebook_all = self.vector_dict[pred_classes.unique()[i].item()].to(vectors.device)  # torch.Size([1024, 1088])
                    codebook = codebook_all[:, :800]
                    codebook_fm = codebook_all[:, 800:]

                    codebook_sqr = torch.sum(codebook ** 2, dim=1)  # torch.Size([1024])
                    inputs_sqr = torch.sum(vectors_per_classes ** 2, dim=1, keepdim=True)  # torch.Size([1, 1])

                    # Compute the distances to the codebook
                    distances = torch.addmm(codebook_sqr + inputs_sqr,   # torch.Size([1, 1024])
                                            vectors_per_classes, codebook.t(), alpha=-2.0, beta=1.0)

                    indices = torch.topk(- distances, k)[1]  # torch.Size([1, 16])
                    nn_vectors = codebook_fm[indices]  # torch.Size([1, 16, 288])
                    memo_latent_vectors[index[:, 0]] = nn_vectors  # torch.Size([1, 16, 288])

                    vm_vectors = codebook[:, :288][indices]
                    vm_latent_vectors[index[:, 0]] = vm_vectors

                else:
                    vectors_per_classes = vectors_per_classes[:, :288]
                    memo_latent_vectors[index[:, 0]] = vectors_per_classes.unsqueeze(1)

                    vm_vectors = vectors_per_classes
                    vm_latent_vectors[index[:, 0]] = vm_vectors.unsqueeze(1)

        vectors = memo_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))  # torch.Size([16, 8, 6, 6])
        vm_vectors = vm_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))

        # for layer in self.decoder:
        #     vectors = layer(vectors)
        # x = self.outconv(vectors)
        x = self.AM_AE_Net.decode(vectors)
        x = x.view(pred_classes.size(0), k, x.size(2), x.size(3))

        vm = self.VM_AE_Net.decode(vm_vectors)
        vm = vm.view(pred_classes.size(0), k, vm.size(2), vm.size(3))
        return vm, x
    
    def nearest_decode_Vm(self, vectors, pred_classes, k=16):
        # side_len = math.sqrt(fm_vectors.size(1) / self.conv_dims)  # side_len : 6
        side_len = 6
        assert side_len % 1 == 0
        memo_latent_vectors = torch.zeros((vectors.size(0), k, 288)).to(vectors.device) # memo_latent_vectors ：torch.Size([B, 16, 288])
        vm_latent_vectors = torch.zeros((vectors.size(0), k, 288)).to(vectors.device)

        classes_lst = pred_classes.unique()  # torch.Size([B])
        for i in range(len(classes_lst)):
            with torch.no_grad():
                index = (pred_classes == pred_classes.unique()[i].item()).nonzero() 
                vectors_per_classes = vectors[index[:, 0]]  # torch.Size([1, 288])
                if pred_classes.unique()[i].item() in self.vector_dict:
                    codebook_all = self.vector_dict[pred_classes.unique()[i].item()].to(vectors.device)  # torch.Size([1024, 1088])
                    codebook_vm = codebook_all[:, :288]
                    codebook_edge = codebook_all[:, 288:800]
                    codebook_fm = codebook_all[:, 800:]
                    # rearranged_codebook = torch.cat((codebook_fm, codebook_edge), dim=1)  

                    codebook_sqr = torch.sum(codebook_vm ** 2, dim=1)  # torch.Size([1024])
                    inputs_sqr = torch.sum(vectors_per_classes ** 2, dim=1, keepdim=True)  # torch.Size([1, 1])

                    # Compute the distances to the codebook
                    distances = torch.addmm(codebook_sqr + inputs_sqr,   # torch.Size([1, 1024])
                                            vectors_per_classes, codebook_vm.t(), alpha=-2.0, beta=1.0)

                    indices = torch.topk(- distances, k)[1]  # torch.Size([1, 16])
                    nn_vectors = codebook_fm[indices]  # torch.Size([1, 16, 288])
                    memo_latent_vectors[index[:, 0]] = nn_vectors  # torch.Size([1, 16, 288])

                    vm_vectors = codebook_vm[indices]
                    vm_latent_vectors[index[:, 0]] = vm_vectors

                else:
                    vectors_per_classes = vectors_per_classes[:, :288]
                    memo_latent_vectors[index[:, 0]] = vectors_per_classes.unsqueeze(1)

                    vm_vectors = vectors_per_classes
                    vm_latent_vectors[index[:, 0]] = vm_vectors.unsqueeze(1)

        vectors = memo_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))  # torch.Size([16, 8, 6, 6])
        vm_vectors = vm_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))

        # for layer in self.decoder:
        #     vectors = layer(vectors)
        # x = self.outconv(vectors)
        x = self.AM_AE_Net.decode(vectors)
        x = x.view(pred_classes.size(0), k, x.size(2), x.size(3))

        vm = self.VM_AE_Net.decode(vm_vectors)
        vm = vm.view(pred_classes.size(0), k, vm.size(2), vm.size(3))
        return vm, x

    # vectors : torch.Size([B, 288]),  category_ids : torch.Size([1]) , 288 = 8 x 6 x6 就是通过编码器后的 C x H x W
    def nearest_decode_CmEdge(self, vectors, pred_classes, k=16):
        # side_len = math.sqrt(fm_vectors.size(1) / self.conv_dims)  # side_len : 6
        side_len = 6
        assert side_len % 1 == 0
        memo_latent_vectors = torch.zeros((vectors.size(0), k, 288)).to(vectors.device) # memo_latent_vectors ：torch.Size([B, 16, 288])
        vm_latent_vectors = torch.zeros((vectors.size(0), k, 288)).to(vectors.device)

        classes_lst = pred_classes.unique()  # torch.Size([B])
        for i in range(len(classes_lst)):
            with torch.no_grad():
                index = (pred_classes == pred_classes.unique()[i].item()).nonzero() 
                vectors_per_classes = vectors[index[:, 0]]  # torch.Size([1, 288])
                if pred_classes.unique()[i].item() in self.vector_dict:
                    codebook_all = self.vector_dict[pred_classes.unique()[i].item()].to(vectors.device)  # torch.Size([1024, 1088])
                    codebook_vm = codebook_all[:, :288]
                    codebook_edge = codebook_all[:, 288:800]
                    codebook_fm = codebook_all[:, 800:]
                    rearranged_codebook = torch.cat((codebook_fm, codebook_edge), dim=1)  

                    codebook_sqr = torch.sum(rearranged_codebook ** 2, dim=1)  # torch.Size([1024])
                    inputs_sqr = torch.sum(vectors_per_classes ** 2, dim=1, keepdim=True)  # torch.Size([1, 1])

                    # Compute the distances to the codebook
                    distances = torch.addmm(codebook_sqr + inputs_sqr,   # torch.Size([1, 1024])
                                            vectors_per_classes, rearranged_codebook.t(), alpha=-2.0, beta=1.0)

                    indices = torch.topk(- distances, k)[1]  # torch.Size([1, 16])
                    nn_vectors = codebook_fm[indices]  # torch.Size([1, 16, 288])
                    memo_latent_vectors[index[:, 0]] = nn_vectors  # torch.Size([1, 16, 288])

                    vm_vectors = codebook_vm[indices]
                    vm_latent_vectors[index[:, 0]] = vm_vectors

                else:
                    vectors_per_classes = vectors_per_classes[:, :288]
                    memo_latent_vectors[index[:, 0]] = vectors_per_classes.unsqueeze(1)

                    vm_vectors = vectors_per_classes
                    vm_latent_vectors[index[:, 0]] = vm_vectors.unsqueeze(1)

        vectors = memo_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))  # torch.Size([16, 8, 6, 6])
        vm_vectors = vm_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))

        # for layer in self.decoder:
        #     vectors = layer(vectors)
        # x = self.outconv(vectors)
        x = self.AM_AE_Net.decode(vectors)
        x = x.view(pred_classes.size(0), k, x.size(2), x.size(3))

        vm = self.VM_AE_Net.decode(vm_vectors)
        vm = vm.view(pred_classes.size(0), k, vm.size(2), vm.size(3))
        return vm, x
    
    def nearest_decode_CmVm(self, vectors, pred_classes, k=16):
        # side_len = math.sqrt(fm_vectors.size(1) / self.conv_dims)  # side_len : 6
        side_len = 6
        assert side_len % 1 == 0
        memo_latent_vectors = torch.zeros((vectors.size(0), k, 288)).to(vectors.device) # memo_latent_vectors ：torch.Size([B, 16, 288])
        vm_latent_vectors = torch.zeros((vectors.size(0), k, 288)).to(vectors.device)

        classes_lst = pred_classes.unique()  # torch.Size([B])
        for i in range(len(classes_lst)):
            with torch.no_grad():
                index = (pred_classes == pred_classes.unique()[i].item()).nonzero() 
                vectors_per_classes = vectors[index[:, 0]]  # torch.Size([1, 288])
                if pred_classes.unique()[i].item() in self.vector_dict:
                    codebook_all = self.vector_dict[pred_classes.unique()[i].item()].to(vectors.device)  # torch.Size([1024, 1088])
                    codebook_vm = codebook_all[:, :288]
                    codebook_edge = codebook_all[:, 288:800]
                    codebook_fm = codebook_all[:, 800:]
                    rearranged_codebook = torch.cat((codebook_fm, codebook_vm), dim=1)  

                    codebook_sqr = torch.sum(rearranged_codebook ** 2, dim=1)  # torch.Size([1024])
                    inputs_sqr = torch.sum(vectors_per_classes ** 2, dim=1, keepdim=True)  # torch.Size([1, 1])

                    # Compute the distances to the codebook
                    distances = torch.addmm(codebook_sqr + inputs_sqr,   # torch.Size([1, 1024])
                                            vectors_per_classes, rearranged_codebook.t(), alpha=-2.0, beta=1.0)

                    indices = torch.topk(- distances, k)[1]  # torch.Size([1, 16])
                    nn_vectors = codebook_fm[indices]  # torch.Size([1, 16, 288])
                    memo_latent_vectors[index[:, 0]] = nn_vectors  # torch.Size([1, 16, 288])

                    vm_vectors = codebook_vm[indices]
                    vm_latent_vectors[index[:, 0]] = vm_vectors

                else:
                    vectors_per_classes = vectors_per_classes[:, :288]
                    memo_latent_vectors[index[:, 0]] = vectors_per_classes.unsqueeze(1)

                    vm_vectors = vectors_per_classes
                    vm_latent_vectors[index[:, 0]] = vm_vectors.unsqueeze(1)

        vectors = memo_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))  # torch.Size([16, 8, 6, 6])
        vm_vectors = vm_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))

        # for layer in self.decoder:
        #     vectors = layer(vectors)
        # x = self.outconv(vectors)
        x = self.AM_AE_Net.decode(vectors)
        x = x.view(pred_classes.size(0), k, x.size(2), x.size(3))

        vm = self.VM_AE_Net.decode(vm_vectors)
        vm = vm.view(pred_classes.size(0), k, vm.size(2), vm.size(3))
        return vm, x


       # vectors : torch.Size([B, 288]),  category_ids : torch.Size([1]) , 288 = 8 x 6 x6 就是通过编码器后的 C x H x W
    def nearest_decode_CmVmEdge(self, vectors, pred_classes, k=16):
        # side_len = math.sqrt(fm_vectors.size(1) / self.conv_dims)  # side_len : 6
        side_len = 6
        assert side_len % 1 == 0
        memo_latent_vectors = torch.zeros((vectors.size(0), k, 288)).to(vectors.device) # memo_latent_vectors ：torch.Size([B, 16, 288])
        vm_latent_vectors = torch.zeros((vectors.size(0), k, 288)).to(vectors.device)

        classes_lst = pred_classes.unique()  # torch.Size([B])
        for i in range(len(classes_lst)):
            with torch.no_grad():
                index = (pred_classes == pred_classes.unique()[i].item()).nonzero() 
                vectors_per_classes = vectors[index[:, 0]]  # torch.Size([1, 288])
                if pred_classes.unique()[i].item() in self.vector_dict:
                    codebook_all = self.vector_dict[pred_classes.unique()[i].item()].to(vectors.device)  # torch.Size([1024, 1088])
                    codebook = codebook_all[:, :800]
                    codebook_fm = codebook_all[:, 800:]
                    rearranged_codebook = torch.cat((codebook_fm, codebook), dim=1)  

                    codebook_sqr = torch.sum(rearranged_codebook ** 2, dim=1)  # torch.Size([1024])
                    inputs_sqr = torch.sum(vectors_per_classes ** 2, dim=1, keepdim=True)  # torch.Size([1, 1])

                    # Compute the distances to the codebook
                    distances = torch.addmm(codebook_sqr + inputs_sqr,   # torch.Size([1, 1024])
                                            vectors_per_classes, rearranged_codebook.t(), alpha=-2.0, beta=1.0)

                    indices = torch.topk(- distances, k)[1]  # torch.Size([1, 16])
                    nn_vectors = codebook_fm[indices]  # torch.Size([1, 16, 288])
                    memo_latent_vectors[index[:, 0]] = nn_vectors  # torch.Size([1, 16, 288])

                    vm_vectors = codebook[:, :288][indices]
                    vm_latent_vectors[index[:, 0]] = vm_vectors

                else:
                    vectors_per_classes = vectors_per_classes[:, :288]
                    memo_latent_vectors[index[:, 0]] = vectors_per_classes.unsqueeze(1)

                    vm_vectors = vectors_per_classes
                    vm_latent_vectors[index[:, 0]] = vm_vectors.unsqueeze(1)

        vectors = memo_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))  # torch.Size([16, 8, 6, 6])
        vm_vectors = vm_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))

        # for layer in self.decoder:
        #     vectors = layer(vectors)
        # x = self.outconv(vectors)
        x = self.AM_AE_Net.decode(vectors)
        x = x.view(pred_classes.size(0), k, x.size(2), x.size(3))

        vm = self.VM_AE_Net.decode(vm_vectors)
        vm = vm.view(pred_classes.size(0), k, vm.size(2), vm.size(3))
        return vm, x

    def cluster(self):
        for i in range(self.num_classes):
            print("Start cluster No.{} class".format(i + 1))
            if i not in self.vector_dict:
                continue
            if self.vector_dict[i].size(0) > self.num_cluster:
                codes = self.vector_dict[i]
                kmeans = KMeans(n_clusters=self.num_cluster)
                kmeans.fit(codes.cpu())
                self.vector_dict[i] = torch.FloatTensor(kmeans.cluster_centers_).cuda()


    def save(self, prefix=None):
        if prefix is not None:
            save_path = self.config.autoencoder_path + "_{}.pth".format(prefix)
        else:
            save_path = self.autoencoder_path + ".pth"

        torch.save({
            # 'VM_AE_Net': self.VM_AE_Net.state_dict(),
            # 'AM_AE_Net': self.AM_AE_Net.state_dict(),
            'Edge_AE_Net': self.Edge_AE_Net.state_dict(),
        }, save_path)

    def load_AE(self, config, model_path=None, logger=None):
        if os.path.exists(model_path):

            model_params_path = os.path.join(model_path, '{}_shape_prior_MyAE3_100.pth'.format(config.dataset))
            params = torch.load(model_params_path, map_location=torch.device('cpu'))

            if config.rank == 0:
                logger.info("load_AE:{}".format(model_params_path))

            # Handle 'module.' in key names prefix
            if isinstance(params, dict):
                from collections import OrderedDict
                new_params = OrderedDict()
                for k, v in params.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_params[name] = v
                params = new_params

            # Load autoencoder weights
            self.VM_AE_Net.load_state_dict(params)

            
            # load codebook
            codebook_path = os.path.join(model_path, '{}_codebook_MyAE3_epoch100.npy'.format(config.dataset))
            # self.vector_dict = np.load(codebook_path, allow_pickle=True)[()]

            loaded_vector_dict = np.load(codebook_path, allow_pickle=True).item()

            if config.rank == 0:
                logger.info("load_codebook:{}".format(codebook_path))

            self.vector_dict = {
                key: torch.from_numpy(arr)
                for key, arr in loaded_vector_dict.items()
            }

            # values_array_sum = 0
            # for key, values_array in self.AE_net.vector_dict.items():
            #     values_array_sum += values_array.shape[0]
            #     print(f"Key: {key}, Value count: {values_array.shape[0]}, Value shape: {values_array.shape}")

            # print("类别数：", len(self.AE_net.vector_dict))
            # print("values_array_sum：", values_array_sum)
            # print("==============================================")

        else:
            print(model_path, 'not Found')
            raise FileNotFoundError


    
    def load_NewAE(self, config, model_path=None, logger=None):
            
            # load am and vm autoencoder
            model_params_path = model_path + "_last.pth"
            if config.rank == 0:
                logger.info("load_am_vm_AE:{}".format(model_params_path))


            if os.path.exists(model_params_path):
                torch_init_model(self.VM_AE_Net, model_params_path, 'VM_AE_Net')
                torch_init_model(self.AM_AE_Net, model_params_path, 'AM_AE_Net')
            else:
                print(model_params_path, 'not Found')
                raise FileNotFoundError
            

            # load edge autoencoder
            edge_params_path = config.edge_AE_path
            
            if config.rank == 0:
                logger.info("load_edge_AE:{}".format(edge_params_path))


            if os.path.exists(edge_params_path):
                torch_init_model(self.Edge_AE_Net, edge_params_path, 'Edge_AE_Net')
            else:
                print(edge_params_path, 'not Found')
                raise FileNotFoundError
            
            # load codebook
            codebook_path = os.path.join(config.codebook_path, '{}_codebook_NewAE.npy'.format(config.dataset))
            # self.vector_dict = np.load(codebook_path, allow_pickle=True)[()]

            loaded_vector_dict = np.load(codebook_path, allow_pickle=True).item()

            self.vector_dict = {
                key: torch.from_numpy(arr)
                for key, arr in loaded_vector_dict.items()
            }

        
    def save_visualization(self, image, fm_mask_gt, pred_am_crop, vm_mask_gt, pred_vm_crop, edge_mask, result_mask, items, prefix):
        output_images_path = os.path.join(self.config.VIS_PATH, prefix)
        output_images_path = Path(output_images_path)

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        
        img_id = int(items['img_id'].item())
        # img_id = f'{img_id:06d}'

        anno_id = int(items['anno_id'].item())

        print(f"img_id: {img_id}, anno_id: {anno_id}")

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{img_id}_{anno_id}_{suffix}_edge.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        fm_mask_gt = (fm_mask_gt > 0.5).to(torch.float32)
        pred_am_crop = (pred_am_crop > 0.5).to(torch.float32)
        vm_mask_gt = (vm_mask_gt > 0.5).to(torch.float32)
        pred_vm_crop = (pred_vm_crop > 0.5).to(torch.float32)
        edge_mask = (edge_mask > 0.5).to(torch.float32)
        result_mask = (result_mask > 0.5).to(torch.float32)

        fm_mask_gt = fm_mask_gt.cpu().numpy()
        pred_am_crop = pred_am_crop.cpu().numpy()
        vm_mask_gt = vm_mask_gt.cpu().numpy()
        pred_vm_crop = pred_vm_crop.cpu().numpy()
        edge_mask = edge_mask.cpu().numpy()
        result_mask = result_mask.cpu().numpy()


        image = image.squeeze(0)

        fm_mask_gt = np.squeeze(fm_mask_gt[0], axis=0)
        pred_am_crop = np.squeeze(pred_am_crop[0], axis=0)
        vm_mask_gt = np.squeeze(vm_mask_gt[0], axis=0)
        pred_vm_crop = np.squeeze(pred_vm_crop[0], axis=0)
        edge_mask = np.squeeze(edge_mask[0], axis=0)
        result_mask = np.squeeze(result_mask[0], axis=0)

        image = image.cpu().numpy() * 255
        image = image.transpose((1, 2, 0))
        image = self.convert_to_bgr(image)

        fm_mask_gt = draw_probmap(fm_mask_gt)
        pred_am_crop = draw_probmap(pred_am_crop)
        vm_mask_gt = draw_probmap(vm_mask_gt)
        pred_vm_crop = draw_probmap(pred_vm_crop)
        edge_mask = draw_probmap(edge_mask)
        result_mask = draw_probmap(result_mask)
        

        viz_image = np.hstack((image, fm_mask_gt, pred_am_crop, vm_mask_gt, pred_vm_crop, edge_mask, result_mask)).astype(np.uint8)

        _save_image('Boundary', viz_image[:, :, ::-1])


    def visualization_5(self, image, fm_mask_gt, pred_am_crop, vm_mask_gt, pred_vm_crop, items, prefix):
        output_images_path = os.path.join(self.config.VIS_PATH, prefix)
        output_images_path = Path(output_images_path)

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        
        img_id = int(items['img_id'].item())
        # img_id = f'{img_id:06d}'

        anno_id = int(items['anno_id'].item())

        print(f"img_id: {img_id}, anno_id: {anno_id}")

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{img_id}_{anno_id}_{suffix}_edge.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            

        fm_mask_gt = (fm_mask_gt > 0.5).to(torch.float32)
        pred_am_crop = (pred_am_crop > 0.5).to(torch.float32)
        vm_mask_gt = (vm_mask_gt > 0.5).to(torch.float32)
        pred_vm_crop = (pred_vm_crop > 0.5).to(torch.float32)

        fm_mask_gt = fm_mask_gt.cpu().numpy()
        pred_am_crop = pred_am_crop.cpu().numpy()
        vm_mask_gt = vm_mask_gt.cpu().numpy()
        pred_vm_crop = pred_vm_crop.cpu().numpy()


        image = image.squeeze(0)

        fm_mask_gt = np.squeeze(fm_mask_gt[0], axis=0)
        pred_am_crop = np.squeeze(pred_am_crop[0], axis=0)
        vm_mask_gt = np.squeeze(vm_mask_gt[0], axis=0)
        pred_vm_crop = np.squeeze(pred_vm_crop[0], axis=0)

        image = image.cpu().numpy() * 255
        image = image.transpose((1, 2, 0))
        image = self.convert_to_bgr(image)

        fm_mask_gt = draw_probmap(fm_mask_gt)
        pred_am_crop = draw_probmap(pred_am_crop)
        vm_mask_gt = draw_probmap(vm_mask_gt)
        pred_vm_crop = draw_probmap(pred_vm_crop)

        viz_image = np.hstack((image, fm_mask_gt, pred_am_crop, vm_mask_gt, pred_vm_crop)).astype(np.uint8)

        _save_image('Boundary', viz_image[:, :, ::-1])


    def visualization_4(self, image, fm_mask_gt, vm_mask_gt, pred_vm_crop, items, prefix):
        output_images_path = os.path.join(self.config.VIS_PATH, prefix)
        output_images_path = Path(output_images_path)

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        
        img_id = int(items['img_id'].item())
        img_id = f'{img_id:06d}'

        anno_id = int(items['anno_id'].item())

        print(f"img_id: {img_id}, anno_id: {anno_id}")

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{img_id}_{anno_id}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        fm_mask_gt = (fm_mask_gt > 0.5).to(torch.float32)
        vm_mask_gt = (vm_mask_gt > 0.5).to(torch.float32)
        pred_vm_crop = (pred_vm_crop > 0.5).to(torch.float32)

        fm_mask_gt = fm_mask_gt.cpu().numpy()
        vm_mask_gt = vm_mask_gt.cpu().numpy()
        pred_vm_crop = pred_vm_crop.cpu().numpy()


        image = image.squeeze(0)

        fm_mask_gt = np.squeeze(fm_mask_gt[0], axis=0)
        vm_mask_gt = np.squeeze(vm_mask_gt[0], axis=0)
        pred_vm_crop = np.squeeze(pred_vm_crop[0], axis=0)

        image = image.cpu().numpy() * 255
        image = image.transpose((1, 2, 0))
        image = self.convert_to_bgr(image)

        fm_mask_gt = draw_probmap(fm_mask_gt)
        vm_mask_gt = draw_probmap(vm_mask_gt)
        pred_vm_crop = draw_probmap(pred_vm_crop)

        viz_image = np.hstack((image, fm_mask_gt, vm_mask_gt, pred_vm_crop)).astype(np.uint8)

        _save_image('first_sp', viz_image[:, :, ::-1])


    def save_visualization_shape_priors(self, image, fm_gt, pred_fm, pred_vm, pred_fm_old, pred_edge, prior_mask, items, prefix = None):
        output_images_path = os.path.join(self.config.VIS_PATH, prefix)
        output_images_path = Path(output_images_path)

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        
        img_id = int(items['img_id'].item())
        # img_id = f'{img_id:06d}'

        anno_id = int(items['anno_id'].item())

        print(f"img_id: {img_id}, anno_id: {anno_id}")

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{img_id}_{anno_id}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 100])   # 85


        fm_gt = (fm_gt > 0.5).to(torch.float32)
        pred_fm = (pred_fm > 0.5).to(torch.float32)
        pred_fm_old = (pred_fm_old > 0.5).to(torch.float32)
        pred_vm = (pred_vm > 0.5).to(torch.float32)
        pred_edge = (pred_edge > 0.5).to(torch.float32)
        prior_mask = (prior_mask > 0.5).to(torch.float32)

        fm_gt = fm_gt.cpu().numpy()
        pred_fm = pred_fm.cpu().numpy()
        pred_fm_old = pred_fm_old.cpu().numpy()
        pred_vm = pred_vm.cpu().numpy()
        pred_edge = pred_edge.cpu().numpy()
        prior_mask = prior_mask.cpu().numpy()

        image = image.squeeze(0)

        fm_gt = np.squeeze(fm_gt[0], axis=0)
        pred_fm = np.squeeze(pred_fm[0], axis=0)
        pred_fm_old = np.squeeze(pred_fm_old[0], axis=0)
        pred_vm = np.squeeze(pred_vm[0], axis=0)
        pred_edge = np.squeeze(pred_edge[0], axis=0)

        image = image.cpu().numpy() * 255
        image = image.transpose((1, 2, 0))
        image = self.convert_to_bgr(image) #D2SA需要变换

        if prior_mask.shape[1] > 1:
            recon_channel_images = []
            for i in range(prior_mask.shape[1]):
                channel_image = prior_mask[:, i:i+1, :, :]
                channel_image = np.squeeze(channel_image[0], axis=0)
                channel_image = add_border(draw_probmap(channel_image))
                recon_channel_images.append(channel_image)
            prior_mask = np.hstack(recon_channel_images)
        else:
            prior_mask = np.squeeze(prior_mask[0], axis=0)
            prior_mask = draw_probmap(prior_mask)
            prior_mask = add_border(prior_mask)
        
        fm_gt = draw_probmap(fm_gt)
        pred_fm = draw_probmap(pred_fm)
        pred_fm_old = draw_probmap(pred_fm_old)
        pred_vm = draw_probmap(pred_vm)
        pred_edge = draw_probmap(pred_edge)

        # add white border
        image = add_border(image)
        fm_gt = add_border(fm_gt)
        pred_fm = add_border(pred_fm)
        pred_fm_old = add_border(pred_fm_old)
        pred_vm = add_border(pred_vm)
        pred_edge = add_border(pred_edge)
        

        viz_image = np.hstack((image, fm_gt, pred_fm, pred_fm_old, pred_vm, pred_edge, prior_mask)).astype(np.uint8)

        _save_image('shape_prior', viz_image[:, :, ::-1])


    def save_visualization_shape_priors_vm(self, image, fm_gt, pred_fm, pred_vm, pred_fm_old, pred_edge, recon_outputs, prior_mask, items, prefix = None):
        output_images_path = os.path.join(self.config.VIS_PATH, prefix)
        output_images_path = Path(output_images_path)

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        
        img_id = int(items['img_id'].item())
        # img_id = f'{img_id:06d}'

        anno_id = int(items['anno_id'].item())

        print(f"img_id: {img_id}, anno_id: {anno_id}")

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{img_id}_{anno_id}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 100])   # 85


        fm_gt = (fm_gt > 0.5).to(torch.float32)
        pred_fm = (pred_fm > 0.5).to(torch.float32)
        pred_fm_old = (pred_fm_old > 0.5).to(torch.float32)
        pred_vm = (pred_vm > 0.5).to(torch.float32)
        pred_edge = (pred_edge > 0.5).to(torch.float32)
        recon_outputs = (recon_outputs > 0.5).to(torch.float32)
        prior_mask = (prior_mask > 0.5).to(torch.float32)

        prior_mask[prior_mask == 1] = 0.5
        con_mask = prior_mask + recon_outputs
        con_mask = torch.clamp(con_mask, max=1)

        fm_gt = fm_gt.cpu().numpy()
        pred_fm = pred_fm.cpu().numpy()
        pred_fm_old = pred_fm_old.cpu().numpy()
        pred_vm = pred_vm.cpu().numpy()
        pred_edge = pred_edge.cpu().numpy()
        recon_outputs = recon_outputs.cpu().numpy()
        prior_mask = prior_mask.cpu().numpy()
        con_mask = con_mask.cpu().numpy()


        image = image.squeeze(0)

        fm_gt = np.squeeze(fm_gt[0], axis=0)
        pred_fm = np.squeeze(pred_fm[0], axis=0)
        pred_fm_old = np.squeeze(pred_fm_old[0], axis=0)
        pred_vm = np.squeeze(pred_vm[0], axis=0)
        pred_edge = np.squeeze(pred_edge[0], axis=0)

        image = image.cpu().numpy() * 255
        image = image.transpose((1, 2, 0))
        image = self.convert_to_bgr(image) #D2SA,COCOA需要变换


        if con_mask.shape[1] > 1:
            recon_channel_images = []
            for i in range(con_mask.shape[1]):
                channel_image = con_mask[:, i:i+1, :, :]
                channel_image = np.squeeze(channel_image[0], axis=0)
                channel_image = add_border(draw_probmap(channel_image))
                recon_channel_images.append(channel_image)
            con_mask = np.hstack(recon_channel_images)
        else:
            con_mask = np.squeeze(con_mask[0], axis=0)
            con_mask = draw_probmap(con_mask)
            con_mask = add_border(con_mask)

     
        fm_gt = draw_probmap(fm_gt)
        pred_fm = draw_probmap(pred_fm)
        pred_fm_old = draw_probmap(pred_fm_old)
        pred_vm = draw_probmap(pred_vm)
        pred_edge = draw_probmap(pred_edge)

        # add white border
        image = add_border(image)
        fm_gt = add_border(fm_gt)
        pred_fm = add_border(pred_fm)
        pred_fm_old = add_border(pred_fm_old)
        pred_vm = add_border(pred_vm)
        pred_edge = add_border(pred_edge)
        

        viz_image = np.hstack((image, fm_gt, pred_fm, pred_fm_old, pred_vm, pred_edge, con_mask)).astype(np.uint8)

        _save_image('shape_prior_vm', viz_image[:, :, ::-1])

    def save_visualization_shape_priors_vm_old(self, image, fm_gt, pred_fm, pred_fm_old, pred_vm, recon_outputs, prior_mask, items, prefix):
        output_images_path = os.path.join(self.config.VIS_PATH, prefix)
        output_images_path = Path(output_images_path)

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        
        img_id = int(items['img_id'].item())
        img_id = f'{img_id:06d}'

        anno_id = int(items['anno_id'].item())

        print(f"img_id: {img_id}, anno_id: {anno_id}")

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{img_id}_{anno_id}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])


        fm_gt = (fm_gt > 0.5).to(torch.float32)
        pred_fm = (pred_fm > 0.5).to(torch.float32)
        pred_fm_old = (pred_fm_old > 0.5).to(torch.float32)
        pred_vm = (pred_vm > 0.5).to(torch.float32)
        recon_outputs = (recon_outputs > 0.5).to(torch.float32)
        prior_mask = (prior_mask > 0.5).to(torch.float32)

        prior_mask[prior_mask == 1] = 0.5
        con_mask = prior_mask + recon_outputs
        con_mask = torch.clamp(con_mask, max=1)

        fm_gt = fm_gt.cpu().numpy()
        pred_fm = pred_fm.cpu().numpy()
        pred_fm_old = pred_fm_old.cpu().numpy()
        pred_vm = pred_vm.cpu().numpy()
        # recon_outputs = recon_outputs.cpu().numpy()
        # prior_mask = prior_mask.cpu().numpy()
        con_mask = con_mask.cpu().numpy()


        image = image.squeeze(0)

        fm_gt = np.squeeze(fm_gt[0], axis=0)
        pred_fm = np.squeeze(pred_fm[0], axis=0)
        pred_fm_old = np.squeeze(pred_fm_old[0], axis=0)
        pred_vm = np.squeeze(pred_vm[0], axis=0)

        image = image.cpu().numpy() * 255
        image = image.transpose((1, 2, 0))
        #image = self.convert_to_bgr(image)  


        if con_mask.shape[1] > 1:
            recon_channel_images = []
            for i in range(con_mask.shape[1]):
                channel_image = con_mask[:, i:i+1, :, :]
                channel_image = np.squeeze(channel_image[0], axis=0)
                recon_channel_images.append(draw_probmap(channel_image))
            con_mask = np.hstack(recon_channel_images)
        else:
            con_mask = np.squeeze(con_mask[0], axis=0)
            con_mask = draw_probmap(con_mask)

        fm_gt = draw_probmap(fm_gt)
        pred_fm = draw_probmap(pred_fm)
        pred_fm_old = draw_probmap(pred_fm_old)
        pred_vm = draw_probmap(pred_vm)
        

        viz_image = np.hstack((image, fm_gt, pred_fm, pred_fm_old, pred_vm, con_mask)).astype(np.uint8)

        _save_image('shape_prior_vm_old', viz_image[:, :, ::-1])



    def convert_to_bgr(self, image):
        if image.shape[-1] == 3 and image[0, 0, 0] == image[0, 0, 2]:
            return image
        else:
            image = image.astype(np.uint8)
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        

    def get_IoU_1(self, pred_edge, edge_mask_GT):
        loss_eval = {}
        pred_edge = pred_edge.squeeze()
        pred_edge = (pred_edge > 0.5).to(torch.int64)

        iou = evaluation_image((pred_edge > 0.5).to(torch.int64), edge_mask_GT)
        loss_eval["iou_edge"] = iou
        loss_eval["iou_count"] = torch.Tensor([1]).cuda()

        return loss_eval
    
    def get_IoU_2(self, fm_crop_gt, pred_fm, vm_crop_gt, pred_vm):
        loss_eval = {}
        pred_fm = pred_fm.squeeze()
        pred_fm = (pred_fm > 0.5).to(torch.int64)

        pred_vm = pred_vm.squeeze()
        pred_vm = (pred_vm > 0.5).to(torch.int64)

        iou_fm = evaluation_image((pred_fm > 0.5).to(torch.int64), fm_crop_gt)
        loss_eval["iou_fm"] = iou_fm

        iou_vm = evaluation_image((pred_vm > 0.5).to(torch.int64), vm_crop_gt)
        loss_eval["iou_vm"] = iou_vm

        loss_eval["iou_count"] = torch.Tensor([1]).cuda()

        return loss_eval
    

def add_border(image, border_size = 10, color=(255, 255, 255)):
    bordered_image = cv2.copyMakeBorder(
        image,
        border_size,
        border_size,
        border_size,
        border_size,
        cv2.BORDER_CONSTANT,
        value=color
    )
    return bordered_image


def select_samples(gt_mask_per_class, gt_mask_all, k=5, th=0.8):
    """
    输入：
        gt_mask_per_class: [M, 1, 256, 256] M个同类别的GT mask
        gt_mask_all: [1, N, 256, 256] 所有候选GT masks
        k: 选取的top-k数量
        th: IoU阈值
    返回：
        pos_indices: 正样本的索引 [M, k]
        neg_indices: 负样本的索引 [M, k]
    """
    device = gt_mask_per_class.device
    gt_mask_all = gt_mask_all.to(device)
    M = gt_mask_per_class.size(0)
    N = gt_mask_all.size(1)
    
    # 计算批量IoU [M, N]
    ious = batch_compute_iou(gt_mask_per_class, gt_mask_all.squeeze(0))  # [M, N]
    
    # 1. 正样本选择 [M, k]
    pos_values, pos_indices = torch.topk(ious, min(k, N), dim=1)  # 最多选N个
    
    # 如果正样本不足k个，用最大的正样本补全
    if pos_indices.size(1) < k:
        padding_size = k - pos_indices.size(1)
        # 重复最大iou样本
        padding = pos_indices[:, :1].expand(-1, padding_size)
        pos_indices = torch.cat([pos_indices, padding], dim=1)
    
    # 2. 负样本选择 [M, k]
    neg_indices = torch.zeros(M, k, dtype=torch.long, device=device)
    
    for m in range(M):
        # 负样本条件：IoU < th
        neg_mask = ious[m] < th
        
        # 预处理：准备负样本IOU（排除正样本和无效样本）
        neg_ious = ious[m].clone()
        neg_ious[pos_indices[m]] = -1  # 排除已选正样本
        neg_ious[~neg_mask] = -1       # 非负样本设为-1
        
        if (neg_ious != -1).any():  
            # 有负样本时，选择iou最大的负样本
            valid_neg_count = min(k, (neg_ious != -1).sum().item())  # 实际可用负样本数
            neg_values, neg_idx = torch.topk(neg_ious, valid_neg_count)
            neg_indices[m, :len(neg_idx)] = neg_idx
            
            # 如果负样本不足k个，用当前可用的最小iou负样本补全
            if len(neg_idx) < k:
                padding_size = k - len(neg_idx)
                padding = neg_idx[-1:].expand(padding_size)
                neg_indices[m, len(neg_idx):] = padding

        else:
            # 没有负样本时，用正样本中iou最低的补全
            neg_indices[m] = pos_indices[m, -1:].expand(k) 
        
    return pos_indices, neg_indices

def extract_samples(gt_mask_all, pos_indices, neg_indices):
    """
    从gt_mask_all中批量提取正负样本的实际mask
    输入：
        gt_mask_all: [1, N, 256, 256]
        pos_indices: [M, k] 每个样本的top-k正样本索引
        neg_indices: [M, k] 每个样本的top-k负样本索引
    返回：
        pos_masks: [M, k, 1, 256, 256]
        neg_masks: [M, k, 1, 256, 256]
    """
    device = gt_mask_all.device
    pos_indices = pos_indices.to(device)
    neg_indices = neg_indices.to(device)

    M, k = pos_indices.shape
    
    # 1. 提取正样本mask
    # 使用高级索引 [1, [M*k], 256, 256] -> reshape to [M, k, 1, 256, 256]
    pos_masks = gt_mask_all[:, pos_indices.view(-1), :, :]  # [1, M*k, 256, 256]
    pos_masks = pos_masks.view(1, M, k, 256, 256)         # [1, M, k, 256, 256]
    pos_masks = pos_masks.permute(1, 2, 0, 3, 4)         # [M, k, 1, 256, 256]
    
    # 2. 提取负样本mask
    neg_masks = gt_mask_all[:, neg_indices.view(-1), :, :]  # [1, M*k, 256, 256]
    neg_masks = neg_masks.view(1, M, k, 256, 256)         # [1, M, k, 256, 256]
    neg_masks = neg_masks.permute(1, 2, 0, 3, 4)         # [M, k, 1, 256, 256]
    
    return pos_masks, neg_masks

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