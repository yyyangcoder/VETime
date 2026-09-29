import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


class win_ContrastiveLoss_init(nn.Module):
    """
    N-level (token/time-step level) contrastive loss.
    Assumes feat_modality1 and feat_modality2 are aligned position-wise:
        feat_modality1[b, i] <-> feat_modality2[b, i]  (positive pair)
    All other pairs in the batch are treated as negatives.
    
    Inputs:
        feat_modality1: (B, N, D)
        feat_modality2: (B, N, D)
    Output:
        scalar loss (mean of symmetric InfoNCE)
    """
    def __init__(self, dim=512,temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.mlp1=nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim//2),
        )
        self.mlp2=nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim//2),
        )
    def _find_segments(self, labels: torch.Tensor):
        """
        labels: (B, N) or (N,) of 0/1, where 1 = foreground / event
        Returns: list of list of (start, end) tuples for each batch
        """
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        segments = []
        for lbl in labels:
            segs = []
            i = 0
            while i < len(lbl):
                if lbl[i] == 1:
                    start = i
                    while i < len(lbl) and lbl[i] == 1:
                        i += 1
                    segs.append((start, i))
                else:
                    i += 1
            segments.append(segs)
        return segments

    def intra_loss(self, z1, z2, label,start,end):
        N, D = z1.shape
        L = end - start

        # === Anchor ===
        anchor = z2[start:end].mean(dim=0)  # (D,)

        # === Negative (same positions, from z1) ===
        positive = z1[start:end].mean(dim=0)     # (D,)

        # === Positive candidates ===
        Neg_candidates = []

        # (1) Left adjacent single point
        if start - 1 >= 0 and label[start - 1] == 0:
            Neg_candidates.append(z1[start - 1])

        # (2) Right adjacent single point
        if end < N and label[end] == 0:
            Neg_candidates.append(z1[end])

        # (3) Left window: [start - L, start)
        left_win_start = max(0, start - L)
        left_win_end = start
        if left_win_end > left_win_start:
            Neg_candidates.append(z1[left_win_start:left_win_end].mean(dim=0))

        # (4) Right window: [end, end + L)
        right_win_start = end
        right_win_end = min(N, end + L)
        if right_win_end > right_win_start:
            Neg_candidates.append(z1[right_win_start:right_win_end].mean(dim=0))
        
        
        # Stack positives: (K, D)
        negative = torch.stack(Neg_candidates)  # (K, D)
        K = negative.shape[0]

        # Compute similarities
        neg_sims = torch.matmul(negative, anchor.unsqueeze(-1)).squeeze(-1)  # (K,)
        pos_sim = torch.dot(anchor, positive).unsqueeze(0)                     # (1,)

        # All logits: [pos_sims..., neg_sim]
        logits = torch.cat([pos_sim, neg_sims], dim=0) / self.temperature      # (K+1,)

        # Numerator: sum of exp(sim_pos / T)
        numerator = torch.exp(pos_sim / self.temperature).sum()

        # Denominator: sum of exp(all / T)
        denominator = torch.exp(logits).sum()

        # Avoid log(0)
        loss = -torch.log(numerator / (denominator + 1e-8) + 1e-8)

        return loss,left_win_start,right_win_end

    def inter_loss(self, z1, z2, start,end, Neg_candidates_start):
        N, D = z1.shape
        L = end - start

        # === Anchor ===
        anchor = z2[start:end].mean(dim=0)  # (D,)

        # === Negative (same positions, from z1) ===
        positive = z1[start:end].mean(dim=0)     # (D,)

        # === Positive candidates ===
        Neg_candidates = [z1[i:i+L].mean(dim=0) for i in Neg_candidates_start]
        if len(Neg_candidates) == 0:
            return torch.tensor(0.0, device=z1.device, requires_grad=True)

        # Stack positives: (K, D)
        negative = torch.stack(Neg_candidates)  # (K, D)
        K = negative.shape[0]

        # Compute similarities
        neg_sims = torch.matmul(negative, anchor.unsqueeze(-1)).squeeze(-1)  # (K,)
        pos_sim = torch.dot(anchor, positive).unsqueeze(0)                     # (1,)

        # All logits: [pos_sims..., neg_sim]
        logits = torch.cat([pos_sim, neg_sims], dim=0) / self.temperature      # (K+1,)

        # Numerator: sum of exp(sim_pos / T)
        numerator = torch.exp(pos_sim / self.temperature).sum()

        # Denominator: sum of exp(all / T)
        denominator = torch.exp(logits).sum()

        # Avoid log(0)
        loss = -torch.log(numerator / (denominator + 1e-8) + 1e-8)

        return loss

        
    def _sample_bg_windows_fast(self, label, start, end, max_samples=10, max_attempts=50):
        N=label.shape[0]
        L=end-start

        Neg_candidates = []
        attempts = 0

        left_start, left_end = 0, max(0, start - L + 1)          # [0, start - L + 1)
        right_start, right_end = end, max(end, N - L + 1)       # [end, N - L + 1)

        total_left = left_end - left_start
        total_right = right_end - right_start

        if total_left + total_right == 0:
            return []

        while len(Neg_candidates) < max_samples and attempts < max_attempts:
            if total_left > 0 and (total_right == 0 or random.random() < total_left / (total_left + total_right)):
                i = random.randint(left_start, left_end - 1)
            else:
                i = random.randint(right_start, right_end - 1)
            if not (label[i:i+L] == 0).all():
                attempts += 1
                continue
            if any(abs(i - j) < L for j in Neg_candidates):
                attempts += 1
                continue
            Neg_candidates.append(i)
            attempts += 1

        return Neg_candidates

class win_Contrastive_Loss(win_ContrastiveLoss_init):
    """
    N-level (token/time-step level) contrastive loss.
    Assumes feat_modality1 and feat_modality2 are aligned position-wise:
        feat_modality1[b, i] <-> feat_modality2[b, i]  (positive pair)
    All other pairs in the batch are treated as negatives.
    
    Inputs:
        feat_modality1: (B, N, D)
        feat_modality2: (B, N, D)
    Output:
        InfoNCE loss
    """
    def __init__(self, dim=512,temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.mlp1=nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim//2),
        )
        self.mlp2=nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim//2),
        )
    def forward(self, f1: torch.Tensor, f2: torch.Tensor,labels0,num_f=1) -> torch.Tensor:
        """
        feat_modality1: (B, N, D) 为ts
        feat_modality2: (B, N, D) 为vision
        """
        B, N, D = f1.shape
        assert f2.shape == (B, N, D)

        mask = labels0.view(B, N//num_f, -1)
        labels = mask.sum(dim=-1) > 0  # (B, num_patches)

        assert labels.shape == (B, N//num_f)

        # Project features
        z10 = self.mlp1(f1)  # (B, N, D//4)
        z20 = self.mlp2(f2)  # (B, N, D//4)

        # L2 normalize
        z1s = F.normalize(z10, dim=-1)
        z2s = F.normalize(z20, dim=-1)

        total_loss = 0.0
        loss_intra = 0.0

        segments_per_batch = self._find_segments(labels)
        num_segments = 0
        for b in range(B):
            z1 =z1s[b]
            z2 = z2s[b]
            label = labels[b]
            for (start, end) in segments_per_batch[b]:
                L = end - start
                if L <= 0:
                    continue
                intra1,win_start,win_end = self.intra_loss(z1, z2, label,start,end)
                intra2,_,_ = self.intra_loss(z2, z1, label,start,end)
            
                inter1 =0
                inter2 =0

                if L > 1:
                    cand_strat=self._sample_bg_windows_fast(label, win_start,win_end, 10)
                    inter1 = self.inter_loss(z1, z2, win_start,win_end,cand_strat)
                    inter2 = self.inter_loss(z2, z1, win_start,win_end,cand_strat)
                num_segments += 1
                total_loss += intra1 + inter2

        if num_segments == 0:
            return torch.zeros((), device=f1.device)

        return total_loss/num_segments
def load_balance_loss(probs,top_k=3):
    """
    Standard MoE load balancing loss.
    probs: (B, C) or (B, T, C)
    """
    # Flatten to (N_total_tokens, C)
    if not isinstance(probs, torch.Tensor):
        return 0
    num_experts = probs.size(-1)
    probs = probs.view(-1, probs.size(-1))  # [N, C]

    N=probs.size(0)
    
    # Fraction of tokens assigned to each expert (importance)
    importance = probs.mean(dim=0)  # [C]
    _, topk_ids = torch.topk(probs, top_k, dim=-1)
    # Gate output probability per expert (load)
    load = torch.zeros(num_experts, device=probs.device, dtype=torch.float32)
    expert_indices = topk_ids.view(-1)          # [N * k]
    ones = torch.ones_like(expert_indices, dtype=torch.float32)
    load.scatter_add_(0, expert_indices, ones)
    load = load / (N * top_k)  
    # CV (coefficient of variation) or dot product
    loss = (importance * load).sum()  # multiply by C
    # loss += -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
    return loss