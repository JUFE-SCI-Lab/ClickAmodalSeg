import math
import torch
import numpy as np

from torch import nn as nn


class DistMaps(nn.Module):
    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks
        if self.cpu_mode:
            from isegm.utils.cython import get_dist_maps
            self._get_dist_maps = get_dist_maps

    def get_coord_features(self, points, batchsize, rows, cols):
        if self.cpu_mode:
            coords = []
            for i in range(batchsize):
                norm_delimeter = 1.0 if self.use_disks else self.spatial_scale * self.norm_radius
                coords.append(self._get_dist_maps(points[i].cpu().float().numpy(), rows, cols,
                                                  norm_delimeter))
            coords = torch.from_numpy(np.stack(coords, axis=0)).to(points.device).float()
        else:
            num_points = points.shape[1] // 2
            points = points.view(-1, points.size(2))
            points, points_order = torch.split(points, [2, 1], dim=1)

            invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
            row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
            col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)

            coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
            coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)

            add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
            coords.add_(-add_xy)
            if not self.use_disks:
                coords.div_(self.norm_radius * self.spatial_scale)
            coords.mul_(coords)

            coords[:, 0] += coords[:, 1]
            coords = coords[:, :1]

            coords[invalid_points, :, :, :] = 1e6

            coords = coords.view(-1, num_points, 1, rows, cols)
            coords = coords.min(dim=1)[0]  # -> (bs * num_masks * 2) x 1 x h x w
            coords = coords.view(-1, 2, rows, cols)

        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
        else:
            coords.sqrt_().mul_(2).tanh_()

        return coords

    def forward(self, x, coords):
        return self.get_coord_features(coords, x.shape[0], x.shape[2], x.shape[3])


def get_last_point(points):
    last_point = torch.zeros((points.shape[0], 1, 4), device=points.device, dtype=points.dtype)
    last_point[:, 0, :3] = points[points[:, :, -1] == points[:, :, -1].max(dim=1)[0].unsqueeze(1)]
    last_point[:, 0, -1][
        torch.argwhere(points[:, :, -1] == points[:, :, -1].max(dim=1)[0].unsqueeze(1))[:, -1] < points.shape[
            1] // 2] = 1
    last_point[:, 0, -1][
        torch.argwhere(points[:, :, -1] == points[:, :, -1].max(dim=1)[0].unsqueeze(1))[:, -1] >= points.shape[
            1] // 2] = 0

    return last_point

def modulate_prevMask(prev_mask, points, N, R_max):
    # 添加输入检查
    assert not torch.isnan(prev_mask).any(), "输入包含NaN值"
    assert (prev_mask >= 0).all() & (prev_mask <= 1).all(), "输入值超出[0,1]范围"

    with torch.no_grad():
        last_point = get_last_point(points)

        if torch.any(last_point < 0):
            return prev_mask

        num_points = points.shape[1] // 2
        row_array = torch.arange(start=0, end=prev_mask.shape[2], step=1, dtype=torch.float32, device=points.device)
        col_array = torch.arange(start=0, end=prev_mask.shape[3], step=1, dtype=torch.float32, device=points.device)
        coord_rows, coord_cols = torch.meshgrid(row_array, col_array)

        prevMod = prev_mask.detach().clone().to(torch.float32)
        prev_mask = prev_mask.detach().clone()

        for bindx in range(points.shape[0]):
            pos_points = points[bindx, :num_points][points[bindx, :num_points, -1] != -1]
            neg_points = points[bindx, num_points:][points[bindx, num_points:, -1] != -1]

            y, x = last_point[bindx, 0, :2]
            p = prev_mask[bindx, 0, y.long(), x.long()]

            dist = torch.sqrt((coord_rows - y).pow(2) + (coord_cols - x).pow(2) + 1e-8)
            L2_diff = (prev_mask[bindx, 0] - p).pow(2) + 1e-8

            # if last point is positive
            if last_point[bindx, :, -1] == 1:

                # selecting radius
                if neg_points.shape[0] != 0:
                    min_dist = torch.cdist(neg_points[:, :2].float(), last_point[bindx, 0, :2].unsqueeze(0).float()).min(dim=0)[0]
                    r = min_dist / 2
                    modWindow = (dist <= r)
                    if r < 10:
                        r = 10
                        modWindow = (dist <= r)
                        if min_dist < 10:
                            in_modWindow = neg_points[
                                (torch.cdist(neg_points[:, :2].float(), last_point[bindx, 0, :2].unsqueeze(0).float()) < 10)[:, 0]]
                            for n_click in in_modWindow:
                                dist_n = torch.sqrt((coord_rows - n_click[0]) ** 2 + (coord_cols - n_click[1]) ** 2)
                                modWindow_n = (dist_n <= torch.sqrt((last_point[bindx, 0, 0] - n_click[0]) ** 2 + (
                                            last_point[bindx, 0, 1] - n_click[1]) ** 2))
                                modWindow[modWindow_n] = 0

                else:
                    r = R_max
                    modWindow = (dist <= r)

                # selecting max gamma
                if p == 0:
                    prevMod[bindx, 0][modWindow] = 1 - (dist[modWindow] / (dist[modWindow].max() + 1e-8))
                    continue
                elif p < 0.99:
                    # max_gamma = 1 / (math.log(0.99, p)+1e-8)
                    safe_p = torch.clamp(p, 1e-7, 1-1e-7)
                    max_gamma = 1 / (torch.log(torch.tensor(0.99)) / torch.log(safe_p) + 1e-8)
                else:
                    max_gamma = 1

                # selecting difference function
                # if last click number is less than N
                if last_point[bindx, 0, 2] < N:
                    # L2_diff[modWindow] = (L2_diff[modWindow] / L2_diff[modWindow].max()) * 1000
                    max_val = L2_diff[modWindow].max()
                    L2_diff[modWindow] = (L2_diff[modWindow] / (max_val + 1e-8)) * 1000

                    diff_th = L2_diff[modWindow].median()
                    exp = -(max_gamma - 1) / (diff_th.pow(3) + 1e-8) * (L2_diff[modWindow] - diff_th).pow(3) + 1
                    exp = torch.clamp(exp, min=1.0)  # 确保exp >= 1
                else:
                    exp = max_gamma * (1 - (dist[modWindow] / (r + 1e-8))) + (dist[modWindow] / (r + 1e-8))
                    exp = torch.clamp(exp, min=1e-7, max=1e7)

                # modulating prev mask
                # prevMod[bindx, 0][modWindow] = prevMod[bindx, 0][modWindow] ** (1 / exp)
                base = torch.clamp(prevMod[bindx, 0][modWindow], 1e-7, 1-1e-7)
                exponent = 1 / torch.clamp(exp, min=1e-7)
                prevMod[bindx, 0][modWindow] = torch.pow(base, exponent)
                prevMod[bindx, 0][int(y.round()), int(x.round())] = 1

            # if last point is negative
            else:
                # selecting radius
                if pos_points.shape[0] != 0:
                    min_dist = torch.cdist(pos_points[:, :2].float(), last_point[bindx, 0, :2].unsqueeze(0).float()).min(dim=0)[0]
                    r = min_dist / 2
                    modWindow = (dist <= r)
                    if r < 10:
                        r = 10
                        modWindow = (dist <= r)
                        if min_dist < 10:
                            in_modWindow = pos_points[
                                (torch.cdist(pos_points[:, :2].float(), last_point[bindx, 0, :2].unsqueeze(0).float()) < 10)[:, 0]]
                            for p_click in in_modWindow:
                                dist_p = torch.sqrt((coord_rows - p_click[0]) ** 2 + (coord_cols - p_click[1]) ** 2)
                                modWindow_p = (dist_p <= torch.sqrt((last_point[bindx, 0, 0] - p_click[0]) ** 2 + (
                                            last_point[bindx, 0, 1] - p_click[1]) ** 2))
                                modWindow[modWindow_p] = 0
                else:
                    r = R_max
                    modWindow = (dist <= r)
                # selecting max gamma
                if p == 1:
                    prevMod[bindx, 0][modWindow] = dist[modWindow] / (dist[modWindow].max() + 1e-8)
                    continue
                elif p > 0.01:
                    # max_gamma = math.log(0.01, p)
                    safe_p = torch.clamp(p, 1e-7, 1-1e-7)
                    max_gamma = torch.log(torch.tensor(0.01)) / torch.log(safe_p)
                else:
                    max_gamma = 1

                # selecting difference function
                # if last click number is less than N
                if last_point[bindx, 0, 2] < N:
                    # L2_diff[modWindow] = (L2_diff[modWindow] / L2_diff[modWindow].max()) * 1000
                    max_val = L2_diff[modWindow].max()
                    L2_diff[modWindow] = (L2_diff[modWindow] / (max_val + 1e-8)) * 1000
                    diff_th = L2_diff[modWindow].median()
                    exp = -(max_gamma - 1) / (diff_th.pow(3) + 1e-8) * (L2_diff[modWindow] - diff_th).pow(3) + 1
                    exp = torch.clamp(exp, min=1.0)
                else:
                    exp = max_gamma * (1 - (dist[modWindow] / (r + 1e-8))) + (dist[modWindow] / (r + 1e-8))
                    exp = torch.clamp(exp, min=1e-7, max=1e7)

                # modulating prev mask
                # prevMod[bindx, 0][modWindow] = prevMod[bindx, 0][modWindow] ** (exp)
                base = torch.clamp(prevMod[bindx, 0][modWindow], 1e-7, 1-1e-7)
                exponent = torch.clamp(exp, min=1e-7, max=1e7)
                prevMod[bindx, 0][modWindow] = torch.pow(base, exponent)
                prevMod[bindx, 0][int(y.round()), int(x.round())] = 0

    assert not torch.isnan(prevMod).any(), "输出包含NaN值"
    return prevMod.to(torch.float32)