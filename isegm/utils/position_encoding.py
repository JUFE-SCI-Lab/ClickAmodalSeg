import torch
import torch.nn as nn
import math

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats 
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        输入:
            x: 特征图 [B, C, H, W]
            mask: 掩码 [B, H, W]（True表示填充区域）
        输出:
            pos: 位置编码 [B, C, H, W]
        """
        not_mask = ~mask  # 反转掩码（True表示有效区域）
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # 垂直方向累积和 [B, H, W]
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # 水平方向累积和 [B, H, W]

        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # 频率因子

        # 计算正弦/余弦编码
        pos_x = x_embed[:, :, :, None] / dim_t  # [B, H, W, num_pos_feats]
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # 合并并调整形状 [B, C, H, W]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos