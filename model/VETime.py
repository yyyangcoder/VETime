import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Union, Optional
from dataclasses import dataclass
from loss.loss import win_Contrastive_Loss
from model.TS_encoder.ts_encoder import TimeSeriesEncoder
from model.TS_encoder.ts_model import TS_Model
from model.VTS_module import V_Attention, VTS_Alignment, M_moe


class VETIME(TS_Model):
    """Model for time series pretraining with masked reconstruction and anomaly detection."""
    
    def __init__(self, config_v, vision_model,config_t,ts_model, model_name=None, **kwargs):
        super().__init__(config_t, **kwargs)
        
        # vison setting
        self.vit_encoder = vision_model
        v_dim=vision_model.hidden_size
        t_dim=config_t.d_model
        self.name=model_name
        self.MAX_L=vision_model.MAX_L
        
        t_dim2 = int(t_dim*2)
        self.mlp_i = nn.Sequential(
            nn.Linear(v_dim, t_dim2),
            nn.GELU(),
            nn.Linear(t_dim2, t_dim),
            nn.LayerNorm(t_dim),
        )
        self.pos_emb_v = nn.Parameter(torch.zeros(1, self.MAX_L, v_dim))
        nn.init.normal_(self.pos_emb_v, std=0.02)
        self.I_att = V_Attention(t_dim)

        # ts setting
        self.ts_encoder = ts_model
        self.patch_size =self.ts_encoder.patch_size
        self.projection_layer = self.ts_encoder.ts_encoder.projection_layer
        self.reconstruction_head = ts_model.reconstruction_head
        self.anomaly_head = ts_model.anomaly_head
        self.d_proj=ts_model.d_proj

        # fusion setting
        self.fusion = VTS_Alignment(v_dim,t_dim)
        self.mm_w = M_moe(t_dim)
        # loss setting
        self.cl_loss=win_Contrastive_Loss(t_dim,temperature=0.1)

    def forward(self, hidden_states: torch.Tensor,time_series: torch.Tensor, # grid_thw: torch.Tensor,size,
                att_mask: Optional[torch.Tensor] = None,init_img_size=None,labels=None):

        TS_embeddings0,local_embeddings0,patch_mask=self.ts_encoder(time_series,att_mask)
        B, seq_len, num_features = time_series.size()
        
        image_features,_=self.vit_encoder(hidden_states)
        I_embeddings = self.vit_encoder.unfold_image(image_features,init_img_size)
        I_embeddings =self.mlp_i(I_embeddings+self.pos_emb_v[:, :I_embeddings.size(1), :])
        I_embeddings0=self.I_att(I_embeddings, patch_mask)

        I_embeddings,TS_embeddings = self.fusion(I_embeddings0,TS_embeddings0,patch_mask)
        loss_sc=self.compute_cl(I_embeddings,TS_embeddings,labels)
        mix_out0 = torch.cat([TS_embeddings,I_embeddings],dim=-1)
        mix_out,m_w1 = self.mm_w(mix_out0,TS_embeddings0,I_embeddings0,mix_out0)
        mix_out2,m_w2 = self.mm_w(mix_out0,TS_embeddings0,I_embeddings0,mix_out0,mix_out0)
        m_w1 = m_w1+m_w2

        patch_proj = self.projection_layer(mix_out)
        local_embeddings = patch_proj.view(B, num_features, seq_len//self.patch_size, self.patch_size, self.d_proj)
        local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).contiguous()  # (B, num_patches, patch_size, num_features, d_proj)
        local_embeddings1 = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]  # (B, seq_len, num_features, d_proj)
        
        patch_proj2 = self.projection_layer(mix_out2)
        local_embeddings = patch_proj2.view(B, num_features, seq_len//self.patch_size, self.patch_size, self.d_proj)
        local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).contiguous()  # (B, num_patches, patch_size, num_features, d_proj)
        local_embeddings2 = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]  # (B, 
        
        return local_embeddings1,m_w1,loss_sc,local_embeddings2
    
    def compute_cl(self, I_emb, TS_emb, labels,num_features=1):
        if not self.training:
            return 0.0  
        else:
            return self.cl_loss(I_emb, TS_emb, labels,num_features)

    def split_data(self,images, time_series, att_mask, labels):
        """
        Split batched time-series data into chunks of length <= max_len along the time dimension (dim=1).
        
        Args:
            images:       [B, T, ...]
            time_series:  [B, T, F]
            att_mask:     [B, T]
            labels:       [B, T] or [B, T, 1]
            max_len:      int, maximum sequence length per chunk

        Returns:
            List of tuples: [(img_chunk, ts_chunk, mask_chunk, label_chunk), ...]
        """
        B, T,_ = time_series.shape
        if T != labels.shape[1]:
            raise ValueError("Data and labels must have the same length in the first dimension.")
        
        if T % self.patch_size != 0:
            raise ValueError(f"Total length T={T} is not divisible by patch_size={self.patch_size}.")
        
        if self.MAX_L < self.patch_size:
            raise ValueError(f"MAX_length ({self.MAX_L}) must be >= patch_size ({self.patch_size}).")

        # Work in "patch units"
        num_patches = T // self.patch_size
        max_patches_per_chunk = self.MAX_L // self.patch_size  # floor division

        if max_patches_per_chunk == 0:
            raise ValueError("MAX_length is too small to fit even one patch.")

        # Minimum number of chunks needed
        min_splits = (num_patches + max_patches_per_chunk - 1) // max_patches_per_chunk

        # Now distribute num_patches into min_splits chunks as evenly as possible
        base_patches = num_patches // min_splits
        remainder = num_patches % min_splits

        chunks = []
        start_time = 0

        for i in range(min_splits):
            patches_in_this_chunk = base_patches + (1 if i < remainder else 0)
            chunk_length = patches_in_this_chunk * self.patch_size  # back to time steps
            end_time = start_time + chunk_length

            
            # Slice all tensors along time dimension (dim=1)
            img_chunk = images[:, :,:,start_time:end_time]          # [B, L, ...]
            ts_chunk = time_series[:, start_time:end_time,:]      # [B, L, F]
            mask_chunk = att_mask[:, start_time:end_time]       # [B, L]
            label_chunk = labels[:, start_time:end_time]        # [B, L] or [B, L, 1]
            
            chunks.append((img_chunk, ts_chunk, mask_chunk, label_chunk))
            start_time = end_time
        return chunks

