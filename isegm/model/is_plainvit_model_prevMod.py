import os
import torch
import numpy as np
import torch.nn as nn
from isegm.utils.serialization import serialize
from isegm.utils.utils import torch_init_model
from .is_model_prevMod import ISModel_prevMod
from .modeling.models_vit import VisionTransformer, PatchEmbed
from .modeling.swin_transformer import SwinTransfomerSegHead_prevMod, SwinTransfomerSegHead_MSD, AlignmentModule
from .modeling.AE_model import AE_Model
from .modeling.QKHead import QKHead

class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
        super().__init__()
        self.down_4_chan = max(out_dims[0]*2, in_dim // 2)
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan),
            nn.GELU(),
            nn.ConvTranspose2d(self.down_4_chan, self.down_4_chan // 2, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan // 2),
            nn.Conv2d(self.down_4_chan // 2, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8_chan = max(out_dims[1], in_dim // 2)
        self.down_8 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_8_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_8_chan),
            nn.Conv2d(self.down_8_chan, out_dims[1], 1),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[2], 1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32_chan = max(out_dims[3], in_dim * 2)
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.Conv2d(self.down_32_chan, out_dims[3], 1),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        x_down_4 = self.down_4(x)
        x_down_8 = self.down_8(x)
        x_down_16 = self.down_16(x)
        x_down_32 = self.down_32(x)

        return [x_down_4, x_down_8, x_down_16, x_down_32]
    

class ClassificationHead(nn.Module):
    def __init__(self, in_channels_list=[128, 256, 512, 1024], num_classes=None):
        super().__init__()
        
        # 1. 更激进的通道压缩（使用1x1卷积降维）
        self.conv1 = nn.Conv2d(in_channels_list[0], 32, 1)  
        self.conv2 = nn.Conv2d(in_channels_list[1], 32, 1)  
        self.conv3 = nn.Conv2d(in_channels_list[2], 32, 1)  
        self.conv4 = nn.Conv2d(in_channels_list[3], 32, 1) 
        
        # 2. 使用更小的统一尺寸 (7x7 替代 14x14)
        self.pool = nn.AdaptiveAvgPool2d((7,7))  # 统一到7x7
        
        # 3. 简化分类器 (减少全连接层维度)
        total_channels = 32 * 4  # 4个32通道的特征图
        self.fc = nn.Sequential(
            nn.Linear(total_channels * 7 * 7, 512),  # 原2048
            nn.ReLU(),
            nn.Dropout(0.3),  # 降低dropout率
            nn.Linear(512, num_classes)
        )

    def forward(self, features):
        # 处理每个特征图（降维+统一尺寸）
        f1 = self.pool(self.conv1(features[0]))  # [B,32,7,7]
        f2 = self.pool(self.conv2(features[1]))  # [B,32,7,7]
        f3 = self.pool(self.conv3(features[2]))  # [B,32,7,7]
        f4 = self.pool(self.conv4(features[3]))  # [B,32,7,7]
        
        # 拼接所有特征
        fused = torch.cat([f1, f2, f3, f4], dim=1)  # [B,128,7,7]
        fused = fused.view(fused.size(0), -1)        # [B,128 * 7 * 7=6272]
        
        # 分类
        return self.fc(fused)

# class ClassificationHead(nn.Module):
#     def __init__(self, in_channels_list=[128, 256, 512, 1024], num_classes=None):
#         super().__init__()
#         # 各尺度特征图的处理（保持原始通道数）
#         self.conv1 = nn.Conv2d(in_channels_list[0], in_channels_list[0]//2, 1)  # 128->64
#         self.conv2 = nn.Conv2d(in_channels_list[1], in_channels_list[1]//2, 1)  # 256->128
#         self.conv3 = nn.Conv2d(in_channels_list[2], in_channels_list[2]//2, 1)  # 512->256
#         self.conv4 = nn.Conv2d(in_channels_list[3], in_channels_list[3]//2, 1)  # 1024->512
        
#         # 自适应池化（统一到最小特征图的尺寸14x14）
#         self.pool1 = nn.AdaptiveAvgPool2d((14,14))  # 112x112 -> 14x14
#         self.pool2 = nn.AdaptiveAvgPool2d((14,14))  # 56x56 -> 14x14
#         self.pool3 = nn.AdaptiveAvgPool2d((14,14))  # 28x28 -> 14x14
        
#         # 特征融合后的分类器
#         total_channels = (in_channels_list[0]//2 + in_channels_list[1]//2 + 
#                          in_channels_list[2]//2 + in_channels_list[3]//2)
#         self.fc = nn.Sequential(
#             nn.Linear(total_channels * 14 * 14, 2048),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(2048, num_classes)
#         )

#     def forward(self, features):
#         # 处理每个特征图（降维+统一尺寸）
#         f1 = self.pool1(self.conv1(features[0]))  # [B,64,14,14]
#         f2 = self.pool2(self.conv2(features[1]))  # [B,128,14,14]
#         f3 = self.pool3(self.conv3(features[2]))  # [B,256,14,14]
#         f4 = self.conv4(features[3])             # [B,512,14,14] (已经是14x14)
        
#         # 拼接所有特征
#         fused = torch.cat([f1, f2, f3, f4], dim=1)  # [B,64+128+256+512,14,14]
#         fused = fused.view(fused.size(0), -1)        # [B,(64+128+256+512)*14 * 14]
        
#         # 分类
#         logits = self.fc(fused)
#         return logits


class PlainVitModel_prevMod(ISModel_prevMod):
    @serialize
    def __init__(
        self,
        config=None,
        backbone_params={},
        neck_params={},
        head_params={},
        random_split=False,
        **kwargs
        ):

        super().__init__(**kwargs)
        self.random_split = random_split
        self.config = config

        self.patch_embed_coords = PatchEmbed(
            img_size= backbone_params['img_size'],
            patch_size=backbone_params['patch_size'],
            # in_chans=4,
            in_chans=14,
            embed_dim=backbone_params['embed_dim'],
        )

        self.backbone = VisionTransformer(**backbone_params)
        self.neck = SimpleFPN(**neck_params) 
        self.head = SwinTransfomerSegHead_prevMod(**head_params)
        # self.seg_head = SwinTransfomerSegHead_New(**head_params)
        # self.MSD_head = SwinTransfomerSegHead_MSD(**head_params)
        # self.cls_head = ClassificationHead(num_classes = config.NUM_CLASSES)

        # self.MSD_Alig = AlignmentModule()
        self.AE_net = AE_Model(config)
        # self.QKHead = QKHead()
        
        # for param in self.QKHead.parameters():
        #     param.requires_grad = False  # 冻结所有参数

    def backbone_forward(self, image, coord_features=None, prev_mask=None, prev_mask_modulated=None, prior_mask=None):
        coord_features = self.patch_embed_coords(coord_features)
        backbone_features = self.backbone.forward_backbone(image, coord_features, self.random_split)

        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape
        grid_size = self.backbone.patch_embed.grid_size

        backbone_features = backbone_features.transpose(-1,-2).view(B, C, grid_size[0], grid_size[1])
        multi_scale_features = self.neck(backbone_features)

        # category = self.cls_head(multi_scale_features)  # 分类结果

        # MSD_features = self.MSD_Alig(multi_scale_features, image, prev_mask, prev_mask_modulated, prior_mask)

        output = self.head(multi_scale_features, image, prev_mask, prev_mask_modulated)
        # output = self.MSD_head(multi_scale_features, image, prev_mask, prev_mask_modulated)
        # output = seg_head(multi_scale_features, image, prev_mask, prev_mask_modulated, MSD_features)
        
        # return {'instances': self.head(multi_scale_features, image, prev_mask, prev_mask_modulated), 'instances_aux': None}
        return {'instances': output}
        # return {'instances': output, 'instances_aux': None, 'category': category}
    

    def load_NewAE(self, config, model_path=None, logger=None):
        
        # load am and vm autoencoder
        model_params_path = model_path + "_last.pth"

        logger.info("load_am_vm_AE:{}".format(model_params_path))


        if os.path.exists(model_params_path):
            torch_init_model(self.AE_net.VM_AE_Net, model_params_path, 'VM_AE_Net')
            torch_init_model(self.AE_net.AM_AE_Net, model_params_path, 'AM_AE_Net')
        else:
            print(model_params_path, 'not Found')
            raise FileNotFoundError
        

        # load am and vm autoencoder
        edge_params_path = config.edge_AE_path
        

        logger.info("load_edge_AE:{}".format(edge_params_path))


        if os.path.exists(edge_params_path):
            torch_init_model(self.AE_net.Edge_AE_Net, edge_params_path, 'Edge_AE_Net')
        else:
            print(edge_params_path, 'not Found')
            raise FileNotFoundError
        
        # load codebook
        codebook_path = os.path.join(config.codebook_path, '{}_codebook_NewAE.npy'.format(config.dataset))
        # self.vector_dict = np.load(codebook_path, allow_pickle=True)[()]


        logger.info("load_codebook:{}".format(codebook_path))

        loaded_vector_dict = np.load(codebook_path, allow_pickle=True).item()

        self.AE_net.vector_dict = {
            key: torch.from_numpy(arr)
            for key, arr in loaded_vector_dict.items()
        }


    def load_QKHead(self, config, model_path=None, logger=None):
        # 确定权重文件路径
        if model_path is None:
            model_path = config.QKHead_path

        # 检查文件是否存在
        if not os.path.exists(model_path):
            error_msg = f"QKHead weight file {model_path} not found"
            if logger is not None:
                logger.error(error_msg)
            else:
                print(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            # 加载权重
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 检查是否包含 QKHead 的权重
            if 'QKHead' not in checkpoint:
                error_msg = f"Weight file {model_path} doesn't contain QKHead weights"
                if logger is not None:
                    logger.error(error_msg)
                else:
                    print(error_msg)
                raise KeyError(error_msg)
            
            # 加载权重到 QKHead
            self.QKHead.load_state_dict(checkpoint['QKHead'])
            
            # 日志记录
            success_msg = f"Successfully loaded QKHead weights: {model_path} "
            if logger is not None:
                logger.info(success_msg)
            else:
                print(success_msg)
                
        except Exception as e:
            error_msg = f"Failed to load QKHead weights: {str(e)}"
            if logger is not None:
                logger.error(error_msg)
            else:
                print(error_msg)
            raise RuntimeError(error_msg)

