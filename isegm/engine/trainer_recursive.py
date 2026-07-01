import os
import random
import logging
from copy import deepcopy
from collections import defaultdict

import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.vis import draw_probmap, draw_points, add_tag
from isegm.utils.misc import save_checkpoint
from isegm.utils.serialization import get_config_repr
from isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from isegm.inference.predictors.base import save_visualization_shape_priors, update_mask
from .optimizer import get_optimizer, get_optimizer_with_layerwise_decay

from isegm.data.points_sampler import generate_probs
class ISTrainer_recursive(object):
    def __init__(self, model, cfg, model_cfg, loss_cfg,
                 trainset, valset,
                 optimizer='adam',
                 optimizer_params=None,
                 layerwise_decay=False,
                 image_dump_interval=200,
                 checkpoint_interval=1,
                 tb_dump_period=25,
                 max_interactive_points=0,
                 lr_scheduler=None,
                 metrics=None,
                 additional_val_metrics=None,
                 net_inputs=('images', 'points'),
                 max_num_next_clicks=0,
                 click_models=None,
                 prev_mask_drop_prob=0.0,
                 ):
        self.cfg = cfg
        self.k = cfg.k
        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs
        self.max_num_next_clicks = max_num_next_clicks

        self.click_models = click_models
        self.prev_mask_drop_prob = prev_mask_drop_prob

        if cfg.distributed:
            cfg.batch_size //= cfg.ngpus
            cfg.val_batch_size //= cfg.ngpus

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''
        self.sw = None

        self.trainset = trainset
        self.valset = valset

        logger.info(f'Dataset of {trainset.get_samples_number()} samples was loaded for training.')
        logger.info(f'Dataset of {valset.get_samples_number()} samples was loaded for validation.')

        self.train_data = DataLoader(
            trainset, cfg.batch_size,
            sampler=get_sampler(trainset, shuffle=True, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        self.val_data = DataLoader(
            valset, cfg.val_batch_size,
            sampler=get_sampler(valset, shuffle=False, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        if layerwise_decay:
            self.optim = get_optimizer_with_layerwise_decay(model, optimizer, optimizer_params)
        else:
            self.optim = get_optimizer(model, optimizer, optimizer_params)
        model = self._load_weights(model)

        if cfg.multi_gpu:
            model = get_dp_wrapper(cfg.distributed)(model, device_ids=cfg.gpu_ids,
                                                    output_device=cfg.gpu_ids[0])  # cfg.gpu_ids[0]

        if self.is_master:
            logger.info(model)
            logger.info(get_config_repr(model._config))

        self.device = cfg.device
        self.net = model.to(self.device)
        self.lr = optimizer_params['lr']

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        if self.click_models is not None:
            for click_model in self.click_models:
                for param in click_model.parameters():
                    param.requires_grad = False
                click_model.to(self.device)
                click_model.eval()

        sample_prob_sigma = 0.8
        self.sample_probs = generate_probs(self.max_num_next_clicks, sample_prob_sigma)

    def run(self, num_epochs, start_epoch=None, validation=True):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        self.num_epochs = num_epochs
        logger.info(f'Starting Epoch: {start_epoch}')
        logger.info(f'Total Epochs: {num_epochs}')
        for epoch in range(start_epoch, num_epochs):
            self.training(epoch)
            if validation:
                self.validation(epoch)

    def training(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        if self.cfg.distributed:
            self.train_data.sampler.set_epoch(epoch)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100)\
            if self.is_master else self.train_data

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        self.net.train()
        self.net.AE_net.eval()
        self.net.QKHead.eval()
        train_loss = 0.0
        triplet_loss = 0.0

        # 添加分类统计变量
        total_samples = 0
        correct_predictions = 0

        for i, (batch_data, meta) in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i

            vis_image_list = []
            num_iters = np.random.choice(np.arange(1, self.max_num_next_clicks + 1), p=self.sample_probs)

            for j in range(num_iters):
                if j != num_iters - 1:
                    _, _, splitted_batch_data, outputs = self.batch_forward(batch_data, meta, validation=True)
                else:
                    loss, losses_logging, splitted_batch_data, outputs = \
                        self.batch_forward(batch_data, meta, validation=False)

                with torch.no_grad():
                    if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                        vis_image = self.save_visualization(splitted_batch_data, outputs, global_step, prefix='train')
                        vis_image_list.append(vis_image)
                    batch_data['points'] = splitted_batch_data['next_points']
                    batch_data['prev_mask'] = splitted_batch_data['next_prev_mask']


            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses_logging['overall'] = loss
            reduce_loss_dict(losses_logging)

            train_loss += losses_logging['overall'].item()
            # triplet_loss += losses_logging['triplet_loss'].item()

            # # 只在最后一次迭代统计分类结果
            # # if epoch == self.num_epochs - 1:  # 最后一个epoch
            # with torch.no_grad():
            #     # 假设outputs中有分类输出，且meta中有真实标签
            #     if 'category' in outputs and 'category_id' in meta:
            #         preds = torch.argmax(outputs['category'], dim=1)
            #         labels = meta['category_id'].long().to(preds.device)
            #         total_samples += labels.size(0)
            #         correct_predictions += (preds == labels).sum().item()

            if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                output_images_path = self.cfg.VIS_PATH / 'train'
                vis_image_stacked = np.vstack(vis_image_list)
                cv2.imwrite(str(output_images_path / f'instances_segmentation.jpg'),
                            vis_image_stacked, [cv2.IMWRITE_JPEG_QUALITY, 85])

            if self.is_master:
                for loss_name, loss_value in losses_logging.items():
                    self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}',
                                       value=loss_value.item(),
                                       global_step=global_step)

                for k, v in self.loss_cfg.items():
                    if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                        v.log_states(self.sw, f'{log_prefix}Losses/{k}', global_step)

                self.sw.add_scalar(tag=f'{log_prefix}States/learning_rate',
                                   value=self.lr if not hasattr(self, 'lr_scheduler') else self.lr_scheduler.get_lr()[-1],
                                   global_step=global_step)

                # tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.4f}, triplet loss {triplet_loss/(i+1):.10f}')
                tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.4f}')
                for metric in self.train_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

                
            # if i == 2:
            #     break

        # # 在最后一个epoch打印分类统计结果
        # if epoch == self.num_epochs - 1 and total_samples > 0:
        #     accuracy = 100.0 * correct_predictions / total_samples
        #     if self.is_master:
        #         logger.info(f'\nFinal Epoch Classification Stats:')
        #         logger.info(f'Total Samples: {total_samples}')
        #         logger.info(f'Correct Predictions: {correct_predictions}')
        #         logger.info(f'Accuracy: {accuracy:.2f}%')

        if self.is_master:
            for metric in self.train_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}',
                                   value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

            save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                            epoch=None, multi_gpu=self.cfg.multi_gpu)

            if isinstance(self.checkpoint_interval, (list, tuple)):
                checkpoint_interval = [x for x in self.checkpoint_interval if x[0] <= epoch][-1][1]
            else:
                checkpoint_interval = self.checkpoint_interval

            if epoch % checkpoint_interval == 0:
                save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                                epoch=epoch, multi_gpu=self.cfg.multi_gpu)

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

    def validation(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100) if self.is_master else self.val_data

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.val_data) + i
            loss, batch_losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data, validation=True)

            batch_losses_logging['overall'] = loss
            reduce_loss_dict(batch_losses_logging)
            for loss_name, loss_value in batch_losses_logging.items():
                losses_logging[loss_name].append(loss_value.item())

            val_loss += batch_losses_logging['overall'].item()

            if self.is_master:
                tbar.set_description(f'Epoch {epoch}, validation loss: {val_loss/(i + 1):.4f}')
                for metric in self.val_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                                   global_step=epoch, disable_avg=True)

            for metric in self.val_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}', value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

    def batch_forward(self, batch_data, meta, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = dict()

        if not validation:
            self.net.train()
            self.net.AE_net.eval()
            self.net.QKHead.eval()
        else:
            self.net.eval()
            self.net.AE_net.eval()
            self.net.QKHead.eval()

        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            image, gt_mask, points, prev_mask = batch_data['images'], batch_data['instances'], batch_data['points'], batch_data['prev_mask']
            orig_image, orig_gt_mask, orig_points = image.clone(), gt_mask.clone(), points.clone()

            click_indices = points[:, :, -1].max(dim=1)[0] + 1

            # shape priors
            with torch.no_grad():
                category_ids = meta['category_id']
                fm_crop = meta['fm_crop'].to(prev_mask.device)

                # resized_gt_mask = F.interpolate(fm_crop, size=(256, 256), mode='nearest')

                # prev_mask = update_mask(points, prev_mask) # add points

                resized_prev_mask = F.interpolate(prev_mask, size=(256, 256), mode='nearest')  # torch.Size([1, 1, 256, 256])
                resized_fm_mask = F.interpolate(fm_crop, size=(256, 256), mode='nearest')

                # points_sp = torch.full((self.cfg.batch_size, 2, 3), -1, dtype=torch.float32)
                # click_indices_sp = torch.ones(self.cfg.batch_size, dtype=torch.float32)
                # point_sp = get_next_points(resized_prev_mask, resized_fm_mask, points_sp, click_indices_sp)
                
                vectors_ae_mask = self.net.AE_net.AM_AE_Net.encode(resized_prev_mask)  # torch.Size([1, 8, 6, 6])
                vectors_prev_mask = self.net.QKHead(resized_prev_mask)  # Qhead  torch.Size([1, 8, 6, 6])
                # vectors_prev_mask = self.net.QKHead(resized_prev_mask, point_sp)  # Qhead add point

                latent_vectors_ae_mask = vectors_ae_mask.view(vectors_ae_mask.shape[0], -1)
                latent_vectors_prev_mask = vectors_prev_mask.view(vectors_prev_mask.shape[0], -1)

                latent_vectors_prev_mask = latent_vectors_ae_mask + latent_vectors_prev_mask

                # vm, prior_mask = self.net.AE_net.nearest_decode_Cm(latent_vectors_ae_mask, category_ids, k=self.k)
                # prior_mask, neg_masks_k, K_pos, K_neg  = self.net.AE_net.nearest_decode_head(self.net.QKHead, resized_prev_mask, latent_vectors_prev_mask, category_ids, k=self.k)
                prior_mask = self.net.AE_net.nearest_decode_L2(self.net.QKHead, latent_vectors_prev_mask, resized_prev_mask, category_ids, k=self.k)
                # prior_mask = self.net.AE_net.nearest_decode_L2(self.net.QKHead, latent_vectors_prev_mask, resized_prev_mask, category_ids, k=self.k, point=point_sp) # add point
                prior_mask = F.interpolate(prior_mask, size=(448, 448), mode='nearest').to(prev_mask.device)
                # neg_masks_k = F.interpolate(neg_masks_k, size=(448, 448), mode='nearest').to(prev_mask.device)
                # prior_mask = None

            # image_vis = meta['img_crop'].permute((0, 3, 1, 2)).to(torch.float32).squeeze(0)
            # fm_mask_gt = meta['fm_crop'].squeeze(0)
            # pred_mask_change = (prev_mask > 0.5)
            # save_visualization_shape_priors(image_vis, fm_mask_gt, pred_mask_change, prior_mask, meta, prefix='{}_SP'.format(self.cfg.dataset), iteration=1)
            # save_visualization_shape_priors(image_vis, fm_mask_gt, pred_mask_change, neg_masks_k, meta, prefix='{}_SP'.format(self.cfg.dataset), iteration=2)

            # net_input = torch.cat((image, prev_mask), dim=1) if self.net.with_prev_mask else image  # base
            net_input = torch.cat((image, prev_mask, prior_mask), dim=1) if self.net.with_prev_mask else image  # add shape prior
            output = self.net(net_input, points)

            with torch.no_grad():
                batch_data['next_prev_mask'] = torch.sigmoid(output['instances'])
                batch_data['next_points'] = get_next_points(torch.sigmoid(output['instances']).clone(), orig_gt_mask, points,
                                                            click_indices)

            loss = 0.0
            loss = self.add_loss('instance_loss', loss, losses_logging, validation,
                                 lambda: (output['instances'], batch_data['instances']))
            # loss = self.add_loss('instance_aux_loss', loss, losses_logging, validation,
            #                      lambda: (output['instances_aux'], batch_data['instances']))
            # loss  = self.add_loss('category_loss', loss, losses_logging, validation,
            #                      lambda: (output['category'], meta['category_id'].to(torch.int64).to(output['category'].device)))
            # loss = self.add_loss('triplet_loss', loss, losses_logging, validation,
            #                      lambda: (latent_vectors_prev_mask, K_pos, K_neg))
            
            assert not torch.any(torch.isnan(loss))

            if self.is_master:
                with torch.no_grad():
                    for m in metrics:
                        m.update(*(output.get(x) for x in m.pred_outputs),
                                 *(batch_data[x] for x in m.gt_outputs))
        return loss, losses_logging, batch_data, output

    def add_loss(self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)
            losses_logging[loss_name] = loss
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
        output_images_path = self.cfg.VIS_PATH / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        images = splitted_batch_data['images']
        points = splitted_batch_data['points']
        instance_masks = splitted_batch_data['instances']

        gt_instance_masks = instance_masks.cpu().numpy()
        predicted_instance_masks = torch.sigmoid(outputs['instances']).detach().cpu().numpy()
        points = points.detach().cpu().numpy()

        image_blob, points = images[0], points[0]
        gt_mask = np.squeeze(gt_instance_masks[0], axis=0)
        predicted_mask = np.squeeze(predicted_instance_masks[0], axis=0)

        image = image_blob.cpu().numpy() * 255
        image = image.transpose((1, 2, 0))

        image_with_points = draw_points(image, points[:self.max_interactive_points], (0, 255, 0))
        image_with_points = draw_points(image_with_points, points[self.max_interactive_points:], (0, 0, 255))

        gt_mask[gt_mask < 0] = 0.25
        gt_mask = draw_probmap(gt_mask)
        predicted_prob = draw_probmap(predicted_mask)
        predicted_mask = draw_probmap(predicted_mask>0.5)

        image_with_points = add_tag(image_with_points, 'Image with points')
        gt_mask = add_tag(gt_mask, 'GT mask')
        predicted_mask = add_tag(predicted_mask, 'Pred mask')
        predicted_prob = add_tag(predicted_prob, 'Pred prob')

        viz_image = np.hstack(
            (image_with_points, predicted_prob, predicted_mask, gt_mask)).astype(
            np.uint8)

        return viz_image[:, :, ::-1]

    def _load_weights(self, net):
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                load_weights(net, self.cfg.weights)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            checkpoints = list(self.cfg.CHECKPOINTS_PATH.glob(f'{self.cfg.resume_prefix}*.pth'))
            assert len(checkpoints) == 1

            checkpoint_path = checkpoints[0]
            logger.info(f'Load checkpoint from path: {checkpoint_path}')
            load_weights(net, str(checkpoint_path))
        return net

    @property
    def is_master(self):
        return self.cfg.local_rank == 0

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


def load_weights(model, path_to_weights):
    current_state_dict = model.state_dict()
    new_state_dict = torch.load(path_to_weights, map_location='cpu')['state_dict']
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict)


