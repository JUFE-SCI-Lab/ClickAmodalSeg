import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide
from isegm.utils.vis import add_tag, draw_points

class BasePredictor(object):
    def __init__(self, cfg, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 **kwargs):
        self.cfg = cfg
        self.k = cfg.k
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None
        self.prior_masks = None

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())

    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])
        
    def get_prediction(self, clicker, meta, click_indx, points, prev_mask=None):
        clicks_list = clicker.get_clicks()

        if self.click_models is not None:
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction

            # fm_crop = meta['fm_crop'].unsqueeze(0)
            # prev_mask = fm_crop.to(device=self.prev_prediction.device, dtype=self.prev_prediction.dtype)

        # if click_indx == 0:
        #     self.prior_masks = torch.zeros_like(prev_mask).repeat(1, self.k, 1, 1)  # [B, k, H, W]
        # elif click_indx == 1:
        with torch.no_grad():
            category_ids = meta['category_id'].unsqueeze(0)
            fm_crop = meta['fm_crop'].unsqueeze(0).to(device=self.prev_prediction.device, dtype=self.prev_prediction.dtype)
            # vm_crop = meta['vm_crop_gt'].unsqueeze(0).to(device=self.prev_prediction.device, dtype=self.prev_prediction.dtype)

            # old_prev_mask = prev_mask
            # prev_mask = update_mask(points, prev_mask)
            
            resized_prev_mask = F.interpolate(prev_mask, size=(256, 256), mode='nearest')  # torch.Size([1, 1, 256, 256])
            resized_fm_mask = F.interpolate(fm_crop, size=(256, 256), mode='nearest')

            # points_sp = torch.full((1, 2, 3), -1, dtype=torch.float32)
            # click_indices_sp = torch.ones(1, dtype=torch.float32)
            # point_sp = get_next_points(resized_prev_mask, resized_fm_mask, points_sp, click_indices_sp)
                
            
            vectors_ae_mask = self.net.AE_net.AM_AE_Net.encode(resized_prev_mask)   
            # vectors_prev_mask = self.net.QKHead(resized_prev_mask)   
            # vectors_prev_mask = self.net.QKHead(resized_prev_mask, point_sp)   # add point

            latent_vectors_ae_mask = vectors_ae_mask.view(vectors_ae_mask.shape[0], -1)  # F_q
            # latent_vectors_prev_mask = vectors_prev_mask.view(vectors_prev_mask.shape[0], -1)  # D_q
            
            # latent_vectors_prev_mask = latent_vectors_ae_mask + latent_vectors_prev_mask  # F_q + D_q

            vm, self.prior_masks = self.net.AE_net.nearest_decode_Cm(latent_vectors_ae_mask, category_ids, k=self.k)
            # self.prior_masks = self.net.AE_net.nearest_decode_L2(self.net.QKHead, latent_vectors_prev_mask, resized_prev_mask, category_ids, k=self.k)
            # self.prior_masks = self.net.AE_net.nearest_decode_L2(self.net.QKHead, latent_vectors_prev_mask, resized_prev_mask, category_ids, k=self.k, point=point_sp) # add point
            self.prior_masks = F.interpolate(self.prior_masks, size=(448, 448), mode='nearest')

        # prior_mask = self.prior_masks > 0.5


        image = meta['img_crop'].permute((2,0,1)).to(torch.float32)
        fm_mask_gt = meta['fm_crop']
        img_id = int(meta['img_id'].item())
        anno_id = int(meta['anno_id'].item())
        pred_mask_change = prev_mask > 0.5  
        # old_prev_mask = old_prev_mask > 0.5
        # save_visualization_shape_priors(image, fm_mask_gt, pred_mask_change, self.prior_masks, meta, prefix='test'.format(self.cfg.dataset), iteration=click_indx)
        # if click_indx != 0:  
        # if img_id == 3205 and anno_id == 377:
        #     save_visualization_shape_priors(image, fm_mask_gt, pred_mask_change, self.prior_masks, meta, prefix='test'.format(self.cfg.dataset), iteration=click_indx)


        # input_image = torch.cat((input_image, prev_mask), dim=1)
        input_image = torch.cat((input_image, prev_mask, self.prior_masks), dim=1) # add shape prior

        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list]
        )

        pred_logits = self._get_prediction(image_nd, clicks_lists, is_image_changed)
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])
        
        # clicks_list_nd = self.get_points_nd([clicks_list])
        # outputs = self.net(input_image, clicks_list_nd)
        # pred_category = outputs['category']
        pred_category = None

        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)

        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            return self.get_prediction(clicker)

        self.prev_prediction = prediction
        return prediction.cpu().numpy()[0, 0], pred_category

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        output = self.net(image_nd, points_nd)

        return output['instances']

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']


def save_visualization_shape_priors(image, fm_gt, pred_fm, prior_mask, items, prefix, iteration):
        output_images_path = os.path.join('/media/chenjunjie/cdb0f1df-df9d-4704-ade1-ed1fde64603c/hongren/save_img_Interactivate', prefix)
        output_images_path = Path(output_images_path)

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        
        img_id = int(items['img_id'].item())
        # img_id = f'{img_id:06d}'

        anno_id = int(items['anno_id'].item())

        print(f"img_id: {img_id}, anno_id: {anno_id}")

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{img_id}_{anno_id}_{suffix}_{iteration}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 100])   # 85


        fm_gt = (fm_gt > 0.5).to(torch.float32)
        prior_mask = (prior_mask > 0.5).to(torch.float32)

        fm_gt = fm_gt.cpu().numpy()
        pred_fm = pred_fm.cpu().numpy()
        prior_mask = prior_mask.cpu().numpy()

        fm_gt = np.squeeze(fm_gt, axis=0)
        pred_fm = np.squeeze(pred_fm[0], axis=0)

        image = image.cpu().numpy() * 255
        image = image.transpose((1, 2, 0))
        image = convert_to_bgr(image) #D2SA,COCOA需要变换

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

        # add white border
        image = add_border(image)
        fm_gt = add_border(fm_gt)
        pred_fm = add_border(pred_fm)
        

        viz_image = np.hstack((image, fm_gt, pred_fm, prior_mask)).astype(np.uint8)

        _save_image('shape_prior_new', viz_image[:, :, ::-1])



def convert_to_bgr(image):
    if image.shape[-1] == 3 and image[0, 0, 0] == image[0, 0, 2]:
        return image
    else:
        image = image.astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
def draw_probmap(x):
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT)

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


def update_mask(points, prev_mask):
    device = prev_mask.device
    batch_size, _, H, W = prev_mask.shape

    prev_mask = prev_mask.float()
    
    all_points = points[:, :, :2]  # [batch_size, num_points, 2]
    valid = (all_points != -1).all(dim=-1)  # [batch_size, num_points]

    num_points = all_points.shape[1]
    half_point = num_points // 2
    
    batch_idx = torch.arange(batch_size, device=device)[:, None].expand(-1, num_points)
    
    flat_valid = valid.flatten()
    coords = all_points.reshape(-1, 2)[flat_valid].long()  # [N, 2]
    batch_indices = batch_idx.flatten()[flat_valid]  # [N]
    
    pos_neg = (torch.arange(num_points, device=device) < half_point).repeat(batch_size)[flat_valid]  # [N]
    
    x, y = coords[:, 0], coords[:, 1]
    in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    
    prev_mask[batch_indices[in_bounds], 0, y[in_bounds], x[in_bounds]] = pos_neg[in_bounds].float()
    
    return prev_mask


def get_next_points(pred, gt, points, click_indices, pred_thresh=0.49):
    assert torch.all(click_indices) > 0
    pred = pred.cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh)  # 模拟正点击，fn_mask（false negatives假反例）用于标识那些在实际情况下为正类但模型预测为负类的区域
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)   # 模拟负点击，fp_mask（false positives假正例）用于标识实际为负类但预测为正类的区域

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)  # 对fn_mask 进行填充操作，目的可能是为了处理边界情况
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.size(1) // 2  # 获取点击数量， points 存储的是正负点击点
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]  # 计算 False Negative 区域 距离最近背景的距离，用于确定下一个点击点。
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]  # 计算 False Positive 区域 距离最近前景的距离。

        fn_max_dist = np.max(fn_mask_dt)  # FN区域中距离最近背景的最大值。
        fp_max_dist = np.max(fp_mask_dt)  # FP区域中距离最近前景的最大值。

        is_positive = fn_max_dist > fp_max_dist  # 如果FN误分类区域比FP更大，下一个点击应为正点击

        # 选择下一个点击区域
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0  
        indices = np.argwhere(inner_mask)  # 获取候选点坐标
        click_indx = int(click_indices[bindx].item())
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            if is_positive:  # 正点击
                points[bindx, click_indx, 0] = float(coords[0])
                points[bindx, click_indx, 1] = float(coords[1])
                points[bindx, click_indx, 2] = float(click_indx)
            else:  # 负点击
                points[bindx, num_points + click_indx - 1, 0] = float(coords[0])
                points[bindx, num_points + click_indx - 1, 1] = float(coords[1])
                points[bindx, num_points + click_indx - 1, 2] = float(click_indx)

    return points