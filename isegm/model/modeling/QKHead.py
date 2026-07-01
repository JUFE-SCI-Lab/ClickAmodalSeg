import os
import torch
import torch.nn as nn

from .utils import DistMaps, modulate_prevMask

class QKHead(nn.Module):
    def __init__(self):
        super(QKHead, self).__init__()

        self.N = 7
        self.R_max = 100

        self.dist_maps = DistMaps(norm_radius=5, spatial_scale=1.0,
                        cpu_mode=False, use_disks=True)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),  # [B, 16, 128, 128] 4
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 64, 64]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 32, 32]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # [B, 128, 16, 16]
            nn.ReLU(),
            nn.Conv2d(128, 8, kernel_size=4, stride=2, padding=1),  # [B, 8, 8, 8]
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0)     # [B, 8, 6, 6]
        )

    # def forward(self, x, point):
    def forward(self, x):  # No Point
        """
        输入: 
            x - [B, 1, 256, 256] 二值mask
        输出:
            latent_vector - [B, 8, 6, 6] 编码后的特征
        """

        x = x.float()

        # if point is not None:
        #     point = point.to(device=x.device)
        #     prev_mask_modulated = modulate_prevMask(x, point, self.N, self.R_max)
        #     coord_features = self.get_coord_features(prev_mask_modulated, point)

        #     # coord_features = self.get_point_coord_features(x, point)
        # else:
        #     coord_features = x.repeat(1, 3, 1, 1)

        # coord_features = coord_features.float().to(device=x.device)

        # x = torch.cat((x, coord_features), dim=1)

        return self.encoder(x)

    
    def get_coord_features(self, prev_masks, point):
        coord_features = self.dist_maps(prev_masks, point)
        if prev_masks is not None:
            coord_features = torch.cat((prev_masks, coord_features), dim=1)

        return coord_features


    def get_point_coord_features(self, x, points, sigma=0.1):
        """
        
        参数:
            x: 输入特征 [B, C, H, W]（从中提取H,W）
            points: 点击坐标 [B, N, 3]
                - 前N//2为正点击，后N//2为负点击
                - [..., 2]: -1表示无效点击
            sigma: 高斯核标准差
        
        返回:
            coord_features: [B, 2, H, W]
                - 通道0: 正点击特征
                - 通道1: 负点击特征
        """
        points = points.to(x.device)
        
        B, _, height, width = x.shape
        N = points.shape[1] if points.numel() > 0 else 0
        half_n = N // 2
        
        # 生成网格坐标（自动继承x的设备）
        y_grid = torch.linspace(-1, 1, height, device=x.device).view(1, height, 1, 1)
        x_grid = torch.linspace(-1, 1, width, device=x.device).view(1, 1, width, 1)
        grid = torch.cat([x_grid.expand(B, height, width, 1), 
                        y_grid.expand(B, height, width, 1)], dim=-1)

        def process_clicks(click_points):
            # 二次设备检查（防御性编程）
            click_points = click_points.to(x.device)
            
            if click_points.numel() == 0:
                return torch.zeros(B, height, width, device=x.device)
                
            coords = (click_points[..., :2] * 2 - 1).to(x.device)
            valid = (click_points[..., 2] != -1).float().to(x.device)
            
            # 向量化计算
            delta = grid.unsqueeze(1) - coords.view(B, -1, 1, 1, 2)
            dist = torch.norm(delta, dim=-1)
            dist = dist + (1 - valid).view(B, -1, 1, 1) * 1e6
            
            gauss = torch.exp(-dist.pow(2)/(2*sigma**2))
            min_dist = torch.amin(dist, dim=1)
            max_gauss = torch.amax(gauss * valid.view(B, -1, 1, 1), dim=1)
            
            no_valid = (valid.sum(dim=1) == 0)
            min_dist[no_valid] = 0
            max_gauss[no_valid] = 0
            
            return max_gauss - min_dist
        
        empty_points = torch.empty(B, 0, 3, device=x.device)
        pos_points = points[:, :half_n] if half_n > 0 else empty_points
        neg_points = points[:, half_n:] if half_n > 0 else empty_points
        
        return torch.stack([
            process_clicks(pos_points),
            process_clicks(neg_points)
        ], dim=1)
