"""Multi-label classification and global-prototype contrastive losses."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import *

class FocalDiceLoss(nn.Module):
    """Multi-label Focal-Dice loss. Input should be logits."""
    def __init__(self, p_pos=FOCAL_P_POS, p_neg=FOCAL_P_NEG, clip_pos=FOCAL_CLIP_POS, clip_neg=FOCAL_CLIP_NEG,
                 pos_weight=FOCAL_POS_WEIGHT, reduction=FOCAL_REDUCTION):
        super().__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, target):
        assert logits.shape == target.shape
        predict = torch.sigmoid(logits).view(logits.shape[0], -1)
        target = target.view(target.shape[0], -1)

        xs_pos = predict
        p_pos = predict
        if self.clip_pos is not None and self.clip_pos >= 0:
            m_pos = (xs_pos + self.clip_pos).clamp(max=1)
            p_pos = torch.mul(m_pos, xs_pos)
        num_pos = torch.sum(torch.mul(p_pos, target), dim=1)
        den_pos = torch.sum(p_pos.pow(self.p_pos) + target.pow(self.p_pos), dim=1)
        loss_pos = 1 - (2 * num_pos) / (den_pos + 1e-8)

        xs_neg = 1 - predict
        p_neg = 1 - predict
        if self.clip_neg is not None and self.clip_neg >= 0:
            m_neg = (xs_neg + self.clip_neg).clamp(max=1)
            p_neg = torch.mul(m_neg, xs_neg)
        num_neg = torch.sum(torch.mul(p_neg, (1 - target)), dim=1)
        den_neg = torch.sum(p_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)
        loss_neg = 1 - (2 * num_neg) / (den_neg + 1e-8)

        loss = loss_pos * self.pos_weight + loss_neg * (1 - self.pos_weight)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

class LabelSeqContrastiveLoss(nn.Module):
    """
    MultiPep-DLCL-style label-sequence contrastive loss.

    This version uses GLOBAL label prototypes, not sample-dynamic label tokens.

    S: (B, D)      sample/table representation, e.g. FT [CLS] feature
    L: (C, D)      global label prototypes, usually derived from label_score.weight
    Y: (B, C)      multi-hot labels
    """
    def __init__(self, temperature=0.7):
        super().__init__()
        self.temperature = temperature

    def forward(self, S, L, Y):
        assert S.ndim == 2, f"S should be (B, D), got {tuple(S.shape)}"
        assert L.ndim == 2, f"L should be (C, D), got {tuple(L.shape)}"
        assert Y.ndim == 2, f"Y should be (B, C), got {tuple(Y.shape)}"
        assert S.shape[1] == L.shape[1], "S and L must have the same embedding dimension"
        assert Y.shape[1] == L.shape[0], "Y columns must match number of label prototypes"

        S = F.normalize(S, dim=-1)
        L = F.normalize(L, dim=-1)

        logits = S @ L.T / self.temperature  # (B, C)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        exp_logits = torch.exp(logits)
        pos_mask = (Y > 0.5).float()
        valid = pos_mask.sum(dim=1) > 0
        if valid.sum() == 0:
            return torch.zeros((), device=S.device)

        exp_pos = (exp_logits * pos_mask).sum(dim=1)
        exp_all = exp_logits.sum(dim=1)
        loss = -torch.log((exp_pos[valid] + 1e-12) / (exp_all[valid] + 1e-12))
        return loss.mean()

def get_current_eta_cl(epoch_1based: int) -> float:
    """Linearly warm up contrastive loss after CL_WARMUP_START_EPOCH."""
    if ETA_CL <= 0:
        return 0.0
    if epoch_1based <= CL_WARMUP_START_EPOCH:
        return 0.0
    if CL_WARMUP_EPOCHS <= 0:
        return ETA_CL
    progress = (epoch_1based - CL_WARMUP_START_EPOCH) / float(CL_WARMUP_EPOCHS)
    return ETA_CL * float(np.clip(progress, 0.0, 1.0))

def compute_pos_weight(Y):
    pos = Y.sum(axis=0).astype(np.float64)
    neg = Y.shape[0] - pos
    raw = neg / np.maximum(pos, 1.0)
    pos_weight = np.sqrt(raw)
    pos_weight = np.clip(pos_weight, POS_WEIGHT_MIN, POS_WEIGHT_MAX)
    return pos_weight.astype(np.float32)

def _compute_total_loss(logits_tab, logits_lab, S_cl, label_proto, y,
                        focal_dice_fn, contrastive_fn, pos_weight_t,
                        eta_cl=ETA_CL):
    L_sce = F.binary_cross_entropy_with_logits(
        logits_tab, y, pos_weight=pos_weight_t
    )
    L_lce = F.binary_cross_entropy_with_logits(
        logits_lab, y, pos_weight=pos_weight_t
    )
    L_sml = focal_dice_fn(logits_tab, y)

    if eta_cl > 0:
        if CONTRAST_INCLUDE_BENIGN:
            L_cl = contrastive_fn(S_cl, label_proto, y)
        else:
            L_cl = contrastive_fn(S_cl, label_proto[1:], y[:, 1:])
    else:
        L_cl = torch.zeros((), device=y.device)

    L = ALPHA_CE * (L_sce + L_lce) + BETA_FOCAL * L_sml + eta_cl * L_cl
    return L, {
        "sce": L_sce.item(),
        "lce": L_lce.item(),
        "sml": L_sml.item(),
        "cl": L_cl.item(),
        "eta_cl": float(eta_cl),
    }
