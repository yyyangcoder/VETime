import torch
from transformers import CLIPVisionModel
from model.Vision_encoder.models_mae import MaskedAutoencoderViT,MAE_ARCH
from torch import nn

from huggingface_hub import snapshot_download
import os
from pathlib import Path

# MAE_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/mae/visualize/"
# 使用绝对路径，避免相对路径问题
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
vision_PATH = os.path.join(BASE_DIR, 'checkpoints', 'weight_v')

class MAETS_AD(nn.Module):

    def __init__(self, ckpt_path='mae_visualize_base.pth'):
        super(MAETS_AD, self).__init__()
        config = MAE_ARCH[ckpt_path.split('/')[-1]]
        self.config = config
        self.vision_model = MaskedAutoencoderViT(**config)
        self.hidden_size = self.config['embed_dim']

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.vision_model.load_state_dict(checkpoint['model'], strict=True)
           
    def forward(self, x):
        # Forecasting using visual model.
        # x: look-back window, size: [bs x context_len x nvars]
        # fp64=True can avoid math overflow in some benchmark, like Bitcoin.
        # return: forecasting window, size: [bs x pred_len x nvars]

        latent, pred, mask = self.vision_model(x, mask_ratio=0.0)  

        return latent
        

class VitTS_AD(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.encode_image=CLIPVisionModel.from_pretrained(model_path)
        self.config = self.encode_image.config
    def forward(self, x,pos=None):
        output = self.encode_image(x,pos=pos).last_hidden_state
        return output #recovered,output,patch_embed