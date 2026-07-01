import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from time import time
from pathlib import Path
from isegm.inference import utils
from isegm.inference.clicker import Clicker
from isegm.model.modeling.AE_model import batch_compute_iou

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(cfg, dataset, predictor,logger, **kwargs):
    all_ious = []
    all_ious_occ = []
    all_ious_occ_c2f = []
    counts_occ = 0
    total_samples = 0
    correct_predictions = 0
    mean_priors_ious = 0
    valid_samples = 0

    start_time = time()
    # # 生成一个不重复的随机索引序列
    # random_indices = torch.randperm(len(dataset))
    for i, index in tqdm(enumerate(range(len(dataset))), total=len(dataset), leave=False):
    # for i, index in tqdm(enumerate(random_indices), total=len(dataset), leave=False):
        
        total_samples += 1
        meta, sample = dataset.get_sample(index)
        for object_id in sample.objects_ids:
            _, sample_ious, sample_ious_occ, sample_ious_occ_c2f, counts_occ, _, pred_category, mean_priors_iou, valid_samples = evaluate_sample(sample.image, sample.gt_mask(object_id), predictor,
                                                sample_id=index, meta=meta, counts_occ=counts_occ, valid_samples=valid_samples,cfg=cfg,logger=logger,**kwargs)
            all_ious.append(sample_ious)
            all_ious_occ.append(sample_ious_occ)
            all_ious_occ_c2f.append(sample_ious_occ_c2f)

            mean_priors_ious +=  mean_priors_iou

            # print(f'mean_priors_iou: {mean_priors_iou}')

        # if i == int(len(dataset) * (0.05 if cfg.dataset == "KINS" else 0.3)):
        #     break

        # if i == 3000:
        #     break

        # with torch.no_grad():
        #     preds = torch.argmax(pred_category, dim=1)
        #     labels = meta['category_id'].long().to(preds.device).unsqueeze(0)
        #     total_samples += labels.size(0)
        #     correct_predictions += (preds == labels).sum().item()

    end_time = time()
    elapsed_time = end_time - start_time

    # accuracy = 100.0 * correct_predictions / total_samples
    # print(f'\nFinal Epoch Classification Stats:')
    # print(f'Total Samples: {total_samples}')
    # print(f'Correct Predictions: {correct_predictions}')
    # print(f'Accuracy: {accuracy:.2f}%')

    # fps = len(dataset) / elapsed_time
    # print(f"FPS: {fps:.2f}")

    # print(f'mean_priors_ious: {mean_priors_ious / valid_samples:.4f},total_samples:{total_samples},occ_samples:{valid_samples}')
    logger.info("mean_priors_ious: {:.4f}, total_samples: {}, occ_samples: {}".format(mean_priors_ious / valid_samples, total_samples, valid_samples))

    return all_ious, all_ious_occ, all_ious_occ_c2f, counts_occ, elapsed_time


def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None, meta=None, counts_occ=0,valid_samples=0,cfg=None, logger=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []
    ious_occ_list = []
    ious_occ_c2f_list = []
    points = torch.full((1, 2 * max_clicks, 3), -1.0) # 初始化点击
    new_point = torch.full((1, 1, 3), -1.0)

    img_id = int(meta['img_id'].item())
    anno_id = int(meta['anno_id'].item())

    

    with torch.no_grad():
        predictor.set_input_image(image)
        is_occ = False
        mean_priors_iou = 0
        click_nums = 0


        for click_indx in range(max_clicks):
            click_nums = click_indx + 1
            points, new_point= clicker.make_next_click(pred_mask, points, new_point, click_indx + 1, max_clicks)
            prev_mask = pred_mask
            pred_probs, pred_category= predictor.get_prediction(clicker, meta, click_indx, points)
            pred_mask = pred_probs > pred_thr

            vm_crop_gt = meta["vm_crop_gt"].squeeze().numpy().astype(np.int32)
            fm_crop_gt = meta["fm_crop"].squeeze().numpy().astype(np.int32)

            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)


            image_vis = meta['img_crop'].permute((2,0,1)).to(torch.float32)
            fm_mask_gt = meta['fm_crop'] 

            # prev_mask_vis = update_mask(new_point, prev_mask)
            # prev_mask_vis = prev_mask_vis > pred_thr
            # prev_mask_vis = prev_mask_vis.squeeze().cpu().numpy()
            # save_visualization_point(prev_mask, prev_mask_vis, pred_mask, points, new_point, meta, max_clicks, prefix='test', iteration=click_indx+1)
            # visualize(cfg, pred_mask, meta)
            # # overlay_mask_on_image(cfg, meta)
            # save_visualization(prev_mask, pred_mask, points, new_point, meta, max_clicks, prefix='KINS', iteration=click_indx+1)
            # save_visualization_shape_priors(image_vis, fm_mask_gt, pred_mask, predictor.prior_masks, meta, prefix='test'.format(self.cfg.dataset), iteration=1)

    
            # if img_id == 48008 and anno_id == 8776:
            #     # visualize(cfg, meta)
            #     overlay_mask_on_image(cfg, meta)
            #     save_visualization(prev_mask, pred_mask, points, new_point, meta, max_clicks, prefix='test', iteration=click_indx+1)
            #     print("###########")

            if click_indx == 0:
                if (fm_crop_gt - vm_crop_gt).sum()!=0:
                    is_occ = True

            iou = utils.get_iou(gt_mask, pred_mask)

            
            pred_mask = pred_mask.astype(np.int32)

            if is_occ:
                iou_occ = iou
                iou_occ_c2f = utils.iou(pred_mask - vm_crop_gt, gt_mask - vm_crop_gt) # occ仅遮挡样本的遮挡面积
                # if img_id == 6326 and anno_id == 10:
                #     # visualize(cfg, meta)
                #     # overlay_mask_on_image(cfg, meta)
                #     # save_visualization_shape_priors(image_vis, fm_mask_gt, pred_mask, predictor.prior_masks, meta, prefix='KINS'.format(cfg.dataset), iteration=1)
                #     save_visualization(prev_mask, pred_mask, points, new_point, meta, max_clicks, prefix='KINS', iteration=click_indx+1)
                #     print("###########")

                if img_id == 31620 and anno_id == 182:
                    # if click_nums in {10, 20}:
                    if click_nums == 7:
                        visualize(cfg, pred_mask, meta, iteration=click_indx+1)
                        save_visualization(prev_mask, pred_mask, points, new_point, meta, max_clicks, prefix='{}_100Clicks'.format(cfg.dataset), iteration=click_indx+1, logger=logger, cfg=cfg)
                        print(f"####### iou: {iou_occ} #########")
                        logger.info("####### iou: {} #########".format(iou_occ))

                # if img_id == 3205 and anno_id == 377:
                #     # save_visualization(prev_mask, pred_mask, points, new_point, meta, max_clicks, prefix='D2SA_MFP_2', iteration=click_indx+1)
                #     # save_visualization_shape_priors(image_vis, fm_mask_gt, pred_mask, predictor.prior_masks, meta, prefix='test'.format(cfg.dataset), iteration=1)
                    
                #     prior_masks = predictor.prior_masks
                #     pred_mask_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0)  # [1, 1, 448, 448]
                #     prior_masks_squeezed = prior_masks.squeeze(0).unsqueeze(1)  # [10, 1, 448, 448]

                #     priors_ious = batch_compute_iou(pred_mask_tensor, prior_masks_squeezed) # [1, 10]
                #     mean_priors_iou = priors_ious.mean().item()  # 最终平均IoU值

                #     print(f"iou_occ: {iou_occ}, mean_priors_iou: {mean_priors_iou}")
                #     print("###########")
            else:
                iou_occ = 0
                iou_occ_c2f = 0

            ious_list.append(iou)
            ious_occ_list.append(iou_occ)
            ious_occ_c2f_list.append(iou_occ_c2f)

            # # occ仅遮挡样本的遮挡面积
            # if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
            #     if not is_occ:  
            #         break
            #     elif iou_occ > max_iou_thr:  
            #         break

            # occ为遮挡样本的全部面积
            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:  
                break
            '''
            if img_id != 31620 and anno_id != 182:
                break'''

        if click_nums != 1:
            valid_samples += 1
            prior_masks = predictor.prior_masks
            pred_mask_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0)  # [1, 1, 448, 448]
            prior_masks_squeezed = prior_masks.squeeze(0).unsqueeze(1)  # [10, 1, 448, 448]

            priors_ious = batch_compute_iou(pred_mask_tensor, prior_masks_squeezed) # [1, 10]
            mean_priors_iou = priors_ious.mean().item()  # 最终平均IoU值
            # mean_priors_iou = 0

        if is_occ:
            counts_occ+=1
            
        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), np.array(ious_occ_list, dtype=np.float32), np.array(ious_occ_c2f_list, dtype=np.float32), counts_occ, pred_probs, pred_category, mean_priors_iou, valid_samples
    
  
def save_visualization(prev_mask, pred_mask, points, new_point, meta, max_clicks, prefix, iteration, logger, cfg):
    output_images_path = os.path.join('/mnt/data/linjunwei/save_img_Interactive/', prefix)
    output_images_path = Path(output_images_path)
    images_path = os.path.join('/mnt/data/linjunwei/save_img_Interactive/', '{}_test'.format(cfg.dataset))
    # images_path = os.path.join('/media/chenjunjie/cdb0f1df-df9d-4704-ade1-ed1fde64603c/hongren/save_img_Amodal_Seg', 'Vis_COCOA_my_modal')
    images_path = Path(images_path)

    if not output_images_path.exists():
        output_images_path.mkdir(parents=True)
    
    img_id = int(meta['img_id'].item())
    anno_id = int(meta['anno_id'].item())

    # print(f"img_id: {img_id}, anno_id: {anno_id}")
    logger.info("img_id: {}, anno_id: {}".format(img_id, anno_id))

    def _save_image(suffix, image):
        cv2.imwrite(str(output_images_path / f'{img_id}_{anno_id}_{suffix}_{iteration}.jpg'),
                    image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
    vm_mask_gt = meta['vm_crop_gt']
    fm_mask_gt = meta['fm_crop']
    image = meta['img_crop'].permute((2,0,1)).to(torch.float32)

    fm_mask_gt = (fm_mask_gt > 0.5).to(torch.float32)
    vm_mask_gt = (vm_mask_gt > 0.5).to(torch.float32)

    fm_mask_gt = fm_mask_gt.cpu().numpy()
    vm_mask_gt = vm_mask_gt.cpu().numpy()
    points = points.numpy()
    points = points[0]
    new_point = new_point.numpy()
    new_point = new_point[0]

    fm_mask_gt = np.squeeze(fm_mask_gt, axis=0)
    vm_mask_gt = np.squeeze(vm_mask_gt, axis=0)

    image = image.cpu().numpy() * 255
    image = image.transpose((1, 2, 0))
    image = convert_to_bgr(image) # KINS不需要变换

    fm_mask_gt[fm_mask_gt < 0] = 0.25
    fm_mask_gt = draw_probmap(fm_mask_gt)
    vm_mask_gt[vm_mask_gt < 0] = 0.25
    vm_mask_gt = draw_probmap(vm_mask_gt)
    prev_mask[prev_mask < 0] = 0.25
    prev_mask_with_point = draw_probmap(prev_mask)
    pred_mask_change = draw_probmap(pred_mask)

    mask_dir = os.path.join(images_path,"{}_{}_{}.png".format(img_id, anno_id, iteration))
    ours_fm = np.array(Image.open(mask_dir).convert("L"))
    ours_fm = (ours_fm==215)

    am_mask_dir = os.path.join(images_path,"{}_{}_am_GT.png".format(img_id, anno_id))
    am_gt = np.array(Image.open(am_mask_dir).convert("L"))
    am_gt = (am_gt==215)

    color1 =  [172, 51, 69] 
    color2 = np.array(color1)+35

    am_gt = add_mask(am_gt, image, color1, color2, 2)
    overlayed_image = add_mask(ours_fm, image, color1, color2, 2)

    # # 选择某些点可视化
    keep_mask = np.isin(points[:, 2], [1., 4., 6.])  
    points = np.where(keep_mask[:, None], points, -1.0)


    image_with_points = draw_points(overlayed_image, points[:max_clicks], (0, 255, 0))
    image_with_points = draw_points(image_with_points, points[max_clicks:], (255, 0, 0))

    # if iteration != 1:
    #     # pred_mask_change = draw_probmap_with_diff(prev_mask, pred_mask, new_point)

    #     if new_point[0][2] == 1:
    #         prev_mask_with_point = draw_points(prev_mask_with_point, new_point, (0, 255, 0))
    #     else:
    #         prev_mask_with_point = draw_points(prev_mask_with_point, new_point, (255, 0, 0))

    # add white border
    image = add_border(image)
    # overlayed_image = add_border(overlayed_image)
    image_with_points = add_border(image_with_points)
    fm_mask_gt = add_border(fm_mask_gt)
    vm_mask_gt = add_border(vm_mask_gt)
    pred_mask_change = add_border(pred_mask_change)
    am_gt = add_border(am_gt)

    viz_image = np.hstack((image, am_gt, image_with_points, fm_mask_gt, vm_mask_gt, pred_mask_change)).astype(np.uint8)
    # viz_image = np.hstack((image, am_gt, overlayed_image, fm_mask_gt, vm_mask_gt, pred_mask_change)).astype(np.uint8)
    # viz_image = np.hstack((image, image_with_points, fm_mask_gt, vm_mask_gt, prev_mask_with_point, pred_mask_change)).astype(np.uint8)

    _save_image('Click', viz_image[:, :, ::-1])


def save_visualization_shape_priors(image, fm_gt, pred_fm, prior_mask, items, prefix, iteration):
        output_images_path = os.path.join('/mnt/data/linjunwei/save_img_Interactive', prefix)
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
        # pred_fm = pred_fm.cpu().numpy()
        prior_mask = prior_mask.cpu().numpy()

        fm_gt = np.squeeze(fm_gt, axis=0)
        # pred_fm = np.squeeze(pred_fm, axis=0)

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

        _save_image('shape_prior', viz_image[:, :, ::-1])


def save_visualization_point(prev_mask, prev_mask_point, pred_mask, points, new_point, meta, max_clicks, prefix, iteration):
    output_images_path = os.path.join('/mnt/data/linjunwei/save_img_Interactive', prefix)
    output_images_path = Path(output_images_path)

    if not output_images_path.exists():
        output_images_path.mkdir(parents=True)
    
    img_id = int(meta['img_id'].item())
    anno_id = int(meta['anno_id'].item())

    print(f"img_id: {img_id}, anno_id: {anno_id}")

    def _save_image(suffix, image):
        cv2.imwrite(str(output_images_path / f'{img_id}_{anno_id}_{suffix}_{iteration}.jpg'),
                    image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
    vm_mask_gt = meta['vm_crop_gt']
    fm_mask_gt = meta['fm_crop']
    image = meta['img_crop'].permute((2,0,1)).to(torch.float32)

    fm_mask_gt = (fm_mask_gt > 0.5).to(torch.float32)
    vm_mask_gt = (vm_mask_gt > 0.5).to(torch.float32)

    fm_mask_gt = fm_mask_gt.cpu().numpy()
    vm_mask_gt = vm_mask_gt.cpu().numpy()
    points = points.numpy()
    points = points[0]
    new_point = new_point.numpy()
    new_point = new_point[0]




    fm_mask_gt = np.squeeze(fm_mask_gt, axis=0)
    vm_mask_gt = np.squeeze(vm_mask_gt, axis=0)

    image = image.cpu().numpy() * 255
    image = image.transpose((1, 2, 0))
    image = convert_to_bgr(image)

    image_with_points = draw_points(image, points[:max_clicks], (0, 255, 0))
    image_with_points = draw_points(image_with_points, points[max_clicks:], (255, 0, 0))

    fm_mask_gt[fm_mask_gt < 0] = 0.25
    fm_mask_gt = draw_probmap(fm_mask_gt)
    vm_mask_gt[vm_mask_gt < 0] = 0.25
    vm_mask_gt = draw_probmap(vm_mask_gt)
    prev_mask[prev_mask < 0] = 0.25
    prev_mask_with_point = draw_probmap(prev_mask)
    pred_mask_change = draw_probmap(pred_mask)
    prev_mask_point = draw_probmap(prev_mask_point)

    if iteration != 1:
        # pred_mask_change = draw_probmap_with_diff(prev_mask, pred_mask, new_point)

        if new_point[0][2] == 1:
            prev_mask_with_point = draw_points(prev_mask_with_point, new_point, (0, 255, 0))
        else:
            prev_mask_with_point = draw_points(prev_mask_with_point, new_point, (255, 0, 0))


    viz_image = np.hstack((image, image_with_points, fm_mask_gt, vm_mask_gt, prev_mask_with_point, prev_mask_point, pred_mask_change)).astype(np.uint8)

    _save_image('Point', viz_image[:, :, ::-1])

def visualize(config, pred_fm, meta, iteration):
    pred_fm = (pred_fm>=0.5)
    pred_fm = pred_fm.astype(np.float32)
    # pred_fm = pred_fm.squeeze()
    gt_vm = meta["vm_crop_gt"].squeeze()
    gt_fm = meta["fm_crop"].squeeze()
    save_dir = os.path.join(config.VIS_PATH, '{}_test'.format(config.dataset))
    image_id, anno_id= meta["img_id"], meta["anno_id"]
    plt.imsave("{}/{}_{}_{}.png".format(save_dir, int(image_id.item()), int(anno_id.item()), iteration), pred_fm)
    plt.imsave("{}/{}_{}_vm_GT.png".format(save_dir, int(image_id.item()), int(anno_id.item())), gt_vm)
    plt.imsave("{}/{}_{}_am_GT.png".format(save_dir, int(image_id.item()), int(anno_id.item())), gt_fm)

def overlay_mask_on_image(config, meta):
    output_images_path = os.path.join(config.VIS_PATH, 'Click_{}'.format(config.dataset))
    output_images_path = Path(output_images_path)

    if not output_images_path.exists():
        output_images_path.mkdir(parents=True)
    
    img_id = int(meta['img_id'].item())
    anno_id = int(meta['anno_id'].item())

    print(f"img_id: {img_id}, anno_id: {anno_id}")

    def _save_image(suffix, image):
        cv2.imwrite(str(output_images_path / f'{img_id}_{anno_id}_{suffix}.jpg'),
                    image, [cv2.IMWRITE_JPEG_QUALITY, 100])

    image = meta['img_crop'].permute((2,0,1)).to(torch.float32)
    # fm_mask_gt = meta['fm_crop']
    # vm_mask_gt = meta['vm_crop_gt']

    # fm_mask_gt = (fm_mask_gt > 0.5).to(torch.float32)
    # vm_mask_gt = (vm_mask_gt > 0.5).to(torch.float32)

    # fm_mask_gt = fm_mask_gt.cpu().numpy()
    # vm_mask_gt = vm_mask_gt.cpu().numpy()

    image = image.squeeze(0)

    # fm_mask_gt = np.squeeze(fm_mask_gt[0], axis=0)
    # vm_mask_gt = np.squeeze(vm_mask_gt[0], axis=0)

    image = image.cpu().numpy() * 255
    image = image.transpose((1, 2, 0))
    # image = convert_to_bgr(image)

    # fm_mask_gt = draw_probmap(fm_mask_gt)
    # vm_mask_gt = draw_probmap(vm_mask_gt)

    vm_mask_dir = os.path.join(output_images_path,"{}_{}_vm_GT.png".format(img_id, anno_id))
    vm_gt = np.array(Image.open(vm_mask_dir).convert("L"))
    vm_gt = (vm_gt==215)

    am_mask_dir = os.path.join(output_images_path,"{}_{}_am_GT.png".format(img_id, anno_id))
    am_gt = np.array(Image.open(am_mask_dir).convert("L"))
    am_gt = (am_gt==215)

    # color1 = [random.randint(0,220),random.randint(0,220),random.randint(0,220)]
    # color2 = np.array(color1)+35

    color1 =  [210, 210, 20] 
    color2 = np.array(color1)+35
    # overlayed_image = self.add_mask(ours_fm, image, color1, color3, 2)
    
    vm_gt = add_mask(vm_gt, image, color1, color2, 2)
    am_gt = add_mask(am_gt, image, color1, color2, 2)

    # viz_image = np.hstack((image, overlayed_image)).astype(np.uint8)
    viz_image = np.hstack((image, vm_gt, am_gt)).astype(np.uint8)

    _save_image('overlay', viz_image[:, :, ::-1])

    return color1, color2

def add_mask(mask, img, color1, color_mask=np.array([0, 0, 255]),line_width=1):
    mask = mask.astype(bool)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    res = cv2.drawContours(img.copy(), contours, -1, color1, line_width)
    res[mask] = res[mask] * 0.7 + color_mask * 0.3
    return res

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

def convert_to_bgr(image):
    if image.shape[-1] == 3 and image[0, 0, 0] == image[0, 0, 2]:
        return image
    else:
        image = image.astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
def draw_probmap(x):
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT)

def draw_points(image, points, color, radius=5):
    image = image.copy()
    for p in points:
        if p[0] < 0:
            continue
        # if len(p) == 3:
        #     pradius = {0: 8, 1: 6, 2: 4}[p[2]] if p[2] < 3 else 2
        else:
            pradius = radius
        image = cv2.circle(image, (int(p[1]), int(p[0])), pradius, color, -1)

    return image


def draw_probmap_with_diff(prev_mask, pred_mask, new_point):
    # 创建基础概率图
    prev_probmap = draw_probmap(prev_mask)
    pred_probmap = draw_probmap(pred_mask)
    
    # 计算差异区域
    if new_point[0][2] == 1:
        # 当new_point[0][2]==1时，显示pred_mask比prev_mask多的部分
        diff_mask = pred_mask & ~prev_mask
        # 用绿色(0,255,0)标记新增区域
        pred_probmap[diff_mask.astype(bool)] = [0, 255, 0]
    else:
        # 否则显示pred_mask比prev_mask少的部分
        diff_mask = prev_mask & ~pred_mask
        # 用红色(255,0,0)标记减少区域
        pred_probmap[diff_mask.astype(bool)] = [255, 0, 0]
    
    return pred_probmap



def update_mask(points, prev_mask):
    if isinstance(prev_mask, np.ndarray):
        prev_mask = torch.from_numpy(prev_mask)

    # 处理不同维度的输入
    if prev_mask.dim() == 2:  # [H, W] 格式
        prev_mask = prev_mask.unsqueeze(0).unsqueeze(0)  # -> [1, 1, H, W]
    elif prev_mask.dim() == 3:  # [C, H, W] 格式
        prev_mask = prev_mask.unsqueeze(0)  # -> [1, C, H, W]
    
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





