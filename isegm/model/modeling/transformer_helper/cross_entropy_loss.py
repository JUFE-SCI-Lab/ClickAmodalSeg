# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100):
    """The wrapper function for :func:`F.cross_entropy`"""
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=255):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored. Default: 255

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        assert (pred.dim() == 2 and label.dim() == 1) or (
                pred.dim() == 4 and label.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
        label, weight = _expand_onehot_labels(label, weight, pred.shape,
                                              ignore_index)

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls


# class TripletLoss(nn.Module):
#     def __init__(self, margin=0.5):
#         super().__init__()
#         self.margin = margin

#     def forward(self, Q, K_pos, K_neg):
#         """
#         Triplet Loss: L = ReLU(max(D+) - min(D-) + margin)
#         输入:
#             Q: [B, 288]          
#             K_pos: [B, k_pos, 288] (每个样本对应的正样本keys)
#             K_neg: [B, k_neg, 288] (每个样本对应的负样本keys)
#         返回:
#             loss:平均损失
#         """
#         B = Q.size(0)
        
#         # 1. 使用余弦相似度计算相似度
#         # Q: [B, 1, 288] 用于与K_pos/K_neg计算
#         Q = Q.unsqueeze(1)  # [B, 1, 288]
        
#         # 正样本相似度 [B, k_pos]
#         D_pos = F.cosine_similarity(Q.expand(-1, K_pos.size(1), -1), 
#                                    K_pos, dim=2)
        
#         # 负样本相似度 [B, k_neg]
#         D_neg = F.cosine_similarity(Q.expand(-1, K_neg.size(1), -1), 
#                                    K_neg, dim=2)

#         # 2. 对每个样本计算max(D+)和min(D-)
#         max_D_pos, _ = torch.max(D_pos, dim=1)  # [B]
#         min_D_neg, _ = torch.min(D_neg, dim=1)  # [B]
        
#         # 3. 计算每个样本的loss后取平均
#         losses = F.relu(max_D_pos - min_D_neg + self.margin)  # [B]
#         loss = torch.mean(losses)  
        
#         return loss
    

# class TripletLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super().__init__()
#         self.margin = margin

#     def forward(self, Q, K_pos, K_neg):
#         """
#         Triplet Loss with Euclidean Distance: 
#         L = ReLU(max(D+) - min(D-) + margin)
#         输入:
#             Q: [B, 288]          
#             K_pos: [B, k_pos, 288] (每个样本对应的正样本keys)
#             K_neg: [B, k_neg, 288] (每个样本对应的负样本keys)
#         返回:
#             loss: 平均损失
#         """
#         B = Q.size(0)
        
#         # 1. 计算欧式距离
#         Q = Q.unsqueeze(1)  # [B, 1, 288]
        
#         # 正样本距离 [B, k_pos]
#         D_pos = torch.sum((Q - K_pos) ** 2, dim=2)  # [B, k_pos]
        
#         # 负样本距离 [B, k_neg]
#         D_neg = torch.sum((Q - K_neg) ** 2, dim=2)  # [B, k_neg]

#         # 2. 对每个样本计算min(D+)和max(D-)
#         # 注意：距离越小表示越相似，所以取正样本的最小距离
#         min_D_pos, _ = torch.min(D_pos, dim=1)  # [B]
#         max_D_neg, _ = torch.max(D_neg, dim=1)  # [B]
        
#         # 3. 计算每个样本的loss后取平均
#         # L = ReLU(min(D+) - max(D-) + margin)
#         losses = F.relu(min_D_pos - max_D_neg + self.margin)  # [B]
#         loss = torch.mean(losses)  
        
#         return loss



class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, Q, K_pos, K_neg):
        """
        Triplet Loss with Euclidean Distance: 
        L = ReLU(max(D+) - min(D-) + margin)
        输入:
            Q: [B, 288]          
            K_pos: [B, k_pos, 288] (每个样本对应的正样本keys)
            K_neg: [B, k_neg, 288] (每个样本对应的负样本keys)
        返回:
            loss: 平均损失
        """
        B = Q.size(0)
        
        # 1. 计算欧式距离
        Q = Q.unsqueeze(1)  # [B, 1, 288]
        
        # 正样本距离 [B, k_pos]
        D_pos = torch.sum((Q - K_pos) ** 2, dim=2)  # [B, k_pos]
        
        # 负样本距离 [B, k_neg]
        D_neg = torch.sum((Q - K_neg) ** 2, dim=2)  # [B, k_neg]

        # 2. 对每个样本计算max(D+)和min(D-)
        max_D_pos, _ = torch.max(D_pos, dim=1)  # [B]
        min_D_neg, _ = torch.min(D_neg, dim=1)  # [B]
        
        # 3. 计算每个样本的loss后取平均
        # L = ReLU(max(D+) - min(D-) + margin)
        losses = F.relu(max_D_pos - min_D_neg + self.margin)  # [B]
        loss = torch.mean(losses)  
        
        return loss