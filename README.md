# **ClickAmodalSeg** (CVPR'2026)



Official code repository for the paper:

[**Learning and Aligning Click-Aware Shape Prior for Interactive Amodal Instance Segmentation**](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Learning_and_Aligning_Click-Aware_Shape_Prior_for_Interactive_Amodal_Instance_CVPR_2026_paper.pdf)

[Junjie Chen, Junwei Lin, Hong Ren, Shengjie Liu, Yuming Fang, Feng Qian, Yifan Zuo]

<p align="center">
  <img src="https://raw.githubusercontent.com/JUFE-SCI-Lab/ClickAmodalSeg/main/overview.png" alt="overview" width="80%">
</p>

### Abstract

Amodal instance segmentation aims to segment both visible and occluded regions of object instance, which are challenging due to lacking inference support under occlusion. Most existing methods employ the prior knowledge about object mask (shape prior) to support the amodal estimation, but the shape prior is not always compatible for object instances in the test stage. In this paper, we explore the task of interactive amodal segmentation, where a few user clicks are available for better segmenting the complete masks of object instances. For this task, we propose a novel framework based on learning and aligning clickaware shape prior (termed ClickPriorNet). Specifically, we propose to learn click-aware shape prior with triplet loss, which forces the retrieved shape priors to have higher IoU with the ground-truth of target instance and thus could exactly facilitate the prediction. Besides, considering the inevitable mismatch between shape prior and target instance, we propose to adaptively align the shape prior with deformable attention. Overall, our model could make full use of the interactive clicks to retrieve and align shape priors, and thus could estimate more complete masks. Extensive experiments on three benchmark datasets demonstrate the effectiveness of our method.

## Usage

### Install

The installation is similar to [MFP](https://github.com/cwlee00/MFP), detailed packages could be found in `Click-Seg.yml`.

### Data preparation

Datasets are available in [VRSP-Net](https://github.com/YutingXiao/Amodal-Segmentation-Based-on-Visible-Region-Segmentation-and-Shape-Prior#download-resource) (**COCOA、D2SA、KINS**)

### Training and Test

#### **Pre-train**

For the shape prior codebook pre-train stage we follow the [VRSP-Net](https://github.com/YutingXiao/Amodal-Segmentation-Based-on-Visible-Region-Segmentation-and-Shape-Prior),.

We train the QK-Head in this stage.

**Take D2SA dataset for example:**

```
train Shape prior：
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 --master_port='29508' \
train_AutoEncoder.py --dataset D2SA --batch 1 --path D2SA_seg --augment-codebook

test Shape prior ：
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port='29509' \
test_AutoEncoder.py --dataset D2SA --batch 1 --data_type image --path D2SA_seg 

train QKHead：
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port='29502' \
train_QKHead.py --dataset D2SA --batch 64 --path D2SA_seg

test QKHead：
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_port='29509' \
test_QKHead.py --dataset D2SA --batch 1 --data_type image --path D2SA_seg
```

Pre-train weights could be found in [weights](https://pan.baidu.com/s/1_qPw7ntTYDEXvJenQ98h5w?pwd=4ffj) (Password: `4ffj`).

#### **Train**

For training, please follow the [MFP](https://github.com/cwlee00/MFP) to download the [MAE](https://github.com/facebookresearch/mae) pretrained weights (click to download: [ViT-Base](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)) and configure the dowloaded path in `config.yml`.

```
CUDA_VISIBLE_DEVICES=0 \
python train.py models/iter_mask/plainvit_base448_d2sa_itermask_prevMod.py \
--exp-name='plainvit_base448_d2sa_itermask_prevMod_Click' \
--dataset='D2SA' \
--batch-size=4
```

 ClickAmodalSeg model weights could be found in [weights](https://pan.baidu.com/s/1_qPw7ntTYDEXvJenQ98h5w?pwd=4ffj) (Password: `4ffj`).

#### **Test**

For test, please download the datasets and models, and then configure the path in `config.yml`.

```
CUDA_VISIBLE_DEVICES=0 \
python scripts/evaluate_model.py \
--checkpoint=/NewSP_QKHead_OldMethod_MSDAlig_last_checkpoint.pth\
--config-path=/config/D2SA.yml \
--eval-mode=cvpr \
--dataset=D2SA
```

## Citation



```
@inproceedings{chen2026click_amodal,
  title={Learning and Aligning Click-Aware Shape Prior for Interactive Amodal Instance Segmentation},
  author={Chen, Junjie and Lin, JunWei and Hong, Ren and Liu, Shengjie and Fang, Yuming and Qian, Feng and Yifan Zuo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20478--20487},
  year={2026}
}
```



## Acknowledgement

Thanks to:

-  [MFP](https://github.com/cwlee00/MFP)
- [VRSP-Net](https://github.com/YutingXiao/Amodal-Segmentation-Based-on-Visible-Region-Segmentation-and-Shape-Prior)
- [C2F-Seg](https://github.com/amazon-science/c2f-seg)

## License

This project is released under the [Apache 2.0 license](https://github.com/chenbys/MetaPoint/blob/main/LICENSE).
