import os
import cv2
import math
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.nn import functional as F
import torchvision.transforms as transforms

from isegm.utils.vis import draw_probmap
from isegm.utils.wrappers import cat
from isegm.utils.boxes import pairwise_iou
from isegm.utils.utils import Config, Progbar, to_cuda

class AE_Net(nn.Module):
    def __init__(self, cfg):
        self.name = "AE"
        self.cfg = cfg
        super(AE_Net, self).__init__()

        # Encoder  input: 1 x 256 x 256   latent_vector: 8 x 6 x 6  output: 1 x 256 x 256 
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),  # [B, 16, 128, 128]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # [B, 32, 64, 64]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # [B, 64, 32, 32]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# [B, 128, 16, 16]
            nn.ReLU(),
            nn.Conv2d(128, 8, kernel_size=4, stride=2, padding=1), # [B, 8, 8, 8]
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0)    # [B, 8, 6, 6]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 128, kernel_size=3, stride=1, padding=0), # [B, 128, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),# [B, 64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # [B, 32, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # [B, 16, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # [B, 8, 128, 128]
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),   # [B, 1, 256, 256]
            nn.Sigmoid()  # Use sigmoid if the mask values are in [0, 1]
        )

        # # Encoder  input: 1 x 256 x 256   latent_vector: 8 x 8 x 8  output: 1 x 256 x 256 
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),  # [B, 16, 128, 128]
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # [B, 32, 64, 64]
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # [B, 64, 32, 32]
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# [B, 128, 16, 16]
        #     nn.ReLU(),
        #     nn.Conv2d(128, 8, kernel_size=4, stride=2, padding=1), # [B, 8, 8, 8]
        # )
        
        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(8, 128, kernel_size=4, stride=2, padding=1), # [B, 128, 16, 16]
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),# [B, 32, 32, 32]
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # [B, 16, 64, 64]
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # [B, 8, 128, 128]
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # [B, 8, 256, 256]
        #     nn.Sigmoid()  # Use sigmoid if the mask values are in [0, 1]
        # )

          
        # # Encoder  input: 1 x 256 x 256   latent_vector: 8 x 8 x 8  output: 1 x 256 x 256 
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),  # [B, 16, 128, 128]
        #     nn.ReLU(),
        #     nn.BatchNorm2d(16), 
        #     nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # [B, 32, 64, 64]
        #     nn.ReLU(),
        #     nn.BatchNorm2d(32), 
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # [B, 64, 32, 32]
        #     nn.ReLU(),
        #     nn.BatchNorm2d(64), 
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# [B, 128, 16, 16]
        #     nn.ReLU(),
        #     nn.BatchNorm2d(128), 
        #     nn.Conv2d(128, 8, kernel_size=4, stride=2, padding=1), # [B, 8, 8, 8]
        # )
        
        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(8, 128, kernel_size=4, stride=2, padding=1), # [B, 128, 16, 16]
        #     nn.ReLU(),
        #     nn.BatchNorm2d(128), 
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),# [B, 32, 32, 32]
        #     nn.ReLU(),
        #     nn.BatchNorm2d(64), 
        #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # [B, 16, 64, 64]
        #     nn.ReLU(),
        #     nn.BatchNorm2d(32), 
        #     nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # [B, 8, 128, 128]
        #     nn.ReLU(),
        #     nn.BatchNorm2d(16), 
        #     nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # [B, 8, 256, 256]
        #     nn.Sigmoid()  # Use sigmoid if the mask values are in [0, 1]
        # )



    def forward(self, x):
        x = self.encoder(x)
        latent_vector = x

        x = self.decoder(x)
        return x, latent_vector

    def encode(self, x):
        x = x.float()
        x = self.encoder(x)

        return x

    def decode(self, vectors):
        x = self.decoder(vectors)

        return x


    def load_AE(self, config, model_path=None, logger=None):
        if os.path.exists(model_path):

            # model_params_path = os.path.join(model_path, 'kins_recon_net.pth')
            # params = torch.load(model_params_path, map_location=torch.device('cpu'))
            # self.load_state_dict(params)

            model_params_path = os.path.join(model_path, '{}_shape_prior_NewAE_50.pth'.format(config.dataset))
            params = torch.load(model_params_path, map_location=torch.device('cpu'))

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
            self.load_state_dict(params)

            #  # load codebook
            # codebook_path = os.path.join(model_path, 'KINS_codebook_epoch_100.npy')
            # # self.vector_dict = np.load(codebook_path, allow_pickle=True)[()]

            # loaded_vector_dict = np.load(codebook_path, allow_pickle=True).item()

            # self.vector_dict = {
            #     key: torch.from_numpy(arr)
            #     for key, arr in loaded_vector_dict.items()
            # }

            # values_array_sum = 0
            # for key, values_array in self.vector_dict.items():
            #     values_array_sum += values_array.shape[0]
            #     print(f"Key: {key}, Value count: {values_array.shape[0]}, Value shape: {values_array.shape}")

            # print("类别数：", len(self.vector_dict))
            # print("values_array_sum：", values_array_sum)
            # print("==============================================")


        else:
            print(model_path, 'not Found')
            raise FileNotFoundError
        


    def save_visualization(self , outputs, recon_outputs, items, prefix):
        output_images_path = os.path.join(self.cfg.VIS_PATH, prefix)
        output_images_path = Path(output_images_path)

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        
        img_id = int(items['img_id'].item())
        img_id = f'{img_id:06d}'

        anno_id = int(items['anno_id'].item())

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{img_id}_{anno_id}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        images = items['img_crop'].permute((0,3,1,2)).to(torch.float32)
        instance_masks = items['fm_crop']

        # images = nn.functional.interpolate(images, size=self.cfg.AE_shape,
        #                                                 mode='bilinear', align_corners=True)
        # instance_masks = nn.functional.interpolate(instance_masks, size=self.cfg.AE_shape,
        #                                                 mode='bilinear', align_corners=True)

        gt_instance_masks = instance_masks.cpu().numpy()
        predicted_instance_masks = torch.sigmoid(outputs).detach().cpu().numpy()
        recon_outputs = torch.sigmoid(recon_outputs).detach().cpu().numpy()

        image_blob = images[0]
        gt_mask = np.squeeze(gt_instance_masks[0], axis=0)
        predicted_mask = np.squeeze(predicted_instance_masks[0], axis=0)

        if recon_outputs.shape[1] > 1:
            recon_channel_images = []
            for i in range(recon_outputs.shape[1]):
                channel_image = recon_outputs[:, i:i+1, :, :]
                channel_image = np.squeeze(channel_image[0], axis=0)
                recon_channel_images.append(draw_probmap(channel_image))
            recon_outputs = np.hstack(recon_channel_images)
        else:
            recon_outputs = np.squeeze(recon_outputs[0], axis=0)
            recon_outputs = draw_probmap(recon_outputs)

        image = image_blob.cpu().numpy() * 255
        image = image.transpose((1, 2, 0))
        image = self.convert_to_bgr(image)

        gt_mask[gt_mask < 0] = 0.25
        gt_mask = draw_probmap(gt_mask)
        predicted_mask = draw_probmap(predicted_mask)
        

        viz_image = np.hstack((image, gt_mask, predicted_mask, recon_outputs)).astype(np.uint8)

        _save_image('shape_prior', viz_image[:, :, ::-1])


    def convert_to_bgr(self, image):
        if image.shape[-1] == 3 and image[0, 0, 0] == image[0, 0, 2]:
            return image
        else:
            image = image.astype(np.uint8)
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        

class Edge_AE_Net(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        super(Edge_AE_Net, self).__init__()

        # # Encoder  input: 256 x 256
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),  # [B, 16, 128, 128]
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # [B, 32, 64, 64]
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # [B, 64, 32, 32]
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# [B, 128, 16, 16]
        #     nn.ReLU(),
        #     nn.Conv2d(128, 8, kernel_size=4, stride=2, padding=1), # [B, 8, 8, 8]
        #     nn.ReLU(),
        #     nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0)    # [B, 8, 6, 6]
        # )
        
        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(8, 128, kernel_size=3, stride=1, padding=0), # [B, 128, 8, 8]
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),# [B, 64, 16, 16]
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # [B, 32, 32, 32]
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # [B, 16, 64, 64]
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # [B, 8, 128, 128]
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),   # [B, 1, 256, 256]
        #     nn.Sigmoid()  # Use sigmoid if the mask values are in [0, 1]
        # )


        # 编码器  
        self.encoder = nn.Sequential(  
            nn.Conv2d(1, 16, 3, stride=2, padding=1),    # [B, 16, 128, 128]
            nn.ReLU(True),     
            nn.BatchNorm2d(16),    
            nn.Conv2d(16, 32, 3, stride=2, padding=1),    # [B, 32, 64, 64]
            nn.ReLU(True), 
            nn.BatchNorm2d(32),     
            nn.Conv2d(32, 64, 3, stride=2, padding=1),     # [B, 64, 32, 32] 
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),     # [B, 128, 16, 16] 
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 8, 3, stride=2, padding=1),     # [B, 8, 8, 8] 
        )  

        # 解码器  
        self.decoder = nn.Sequential(  
            nn.ConvTranspose2d(8, 128, 3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(True),  
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(True),  
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(True),
            nn.BatchNorm2d(32),  
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   
            nn.ReLU(True),
            nn.BatchNorm2d(16),  
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )


        # # 编码器  
        # self.encoder = nn.Sequential(  
        #     nn.Conv2d(1, 16, 3, stride=2, padding=1),    # [B, 16, 128, 128]
        #     nn.ReLU(True),     
        #     nn.BatchNorm2d(16),    
        #     nn.Conv2d(16, 32, 3, stride=2, padding=1),    # [B, 32, 64, 64]
        #     nn.ReLU(True), 
        #     nn.BatchNorm2d(32),     
        #     nn.Conv2d(32, 64, 3, stride=2, padding=1),     # [B, 64, 32, 32] 
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(64),
        #     nn.Conv2d(64, 128, 3, stride=2, padding=1),     # [B, 128, 16, 16] 
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(128),
        #     nn.Conv2d(128, 8, 3, stride=2, padding=1),     # [B, 8, 8, 8] 
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(8),
        #     nn.Conv2d(8, 8, 3, stride=1, padding=0),     # [B, 8, 6, 6] 
        # )  

        # # 解码器  
        # self.decoder = nn.Sequential(  
        #     nn.ConvTranspose2d(8, 128, 3, stride=1, padding=0),  
        #     nn.ReLU(True),  
        #     nn.BatchNorm2d(128),
        #     nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  
        #     nn.ReLU(True),  
        #     nn.BatchNorm2d(64),
        #     nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), 
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(32),  
        #     nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(16),  
        #     nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(8),  
        #     nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
        # )

    def forward(self, x):
        x = self.encoder(x)
        latent_vector = x

        x = self.decoder(x)
        return x, latent_vector

    def encode(self, x):
        x = x.float()
        x = self.encoder(x)

        return x

    def decode(self, vectors):
        x = self.decoder(vectors)

        return x


class VM_AE_Net(nn.Module):
    def __init__(self, config):
        super(VM_AE_Net, self).__init__()
        self.AE = AE_Net(config)

    def forward(self, x):
        x, latent_vector = self.AE(x)
        return x, latent_vector
    
    def encode(self, x):
        x = x.float()
        x = self.AE.encoder(x)

        return x

    def decode(self, vectors):
        x = self.AE.decoder(vectors)

        return x
    

class AM_AE_Net(nn.Module):
    def __init__(self, config):
        super(AM_AE_Net, self).__init__()
        self.AE = AE_Net(config)

    def forward(self, x):
        x, latent_vector = self.AE(x)
        return x, latent_vector
    
    def encode(self, x):
        x = x.float().to("cuda")
        x = self.AE.encoder(x)

        return x

    def decode(self, vectors):
        x = self.AE.decoder(vectors)

        return x
    

# class Edge_AE_Net(nn.Module):
#     def __init__(self, config):
#         super(Edge_AE_Net, self).__init__()
#         self.AE = AE_Net(config)

#     def forward(self, x):
#         x, latent_vector = self.AE(x)
#         return x, latent_vector