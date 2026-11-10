"""Neural-network architecture for SynPhen-CL.

Class definitions are kept structurally aligned with the supplied 08 model script to preserve checkpoint compatibility.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import *

class NumericalFeatureTokenizer(nn.Module):
    """
    Numerical tokenizer from FT-Transformer.

    For every numerical feature j:
        T_j = x_j * W_j + b_j
    A trainable [CLS] token is prepended.

    Input:  x_num  (B, d_in)
    Output: tokens (B, d_in + 1, d_token)
    """
    def __init__(self, d_numerical: int, d_token: int, bias: bool = True):
        super().__init__()
        self.d_numerical = d_numerical
        self.d_token = d_token

        self.weight = nn.Parameter(torch.empty(d_numerical, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.bias = nn.Parameter(torch.empty(d_numerical, d_token)) if bias else None
        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.kaiming_uniform_(self.cls, a=math.sqrt(5))

    @property
    def n_tokens(self):
        return self.d_numerical + 1

    def forward(self, x_num):
        assert x_num.ndim == 2
        x = x_num.unsqueeze(-1) * self.weight.unsqueeze(0)
        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)
        cls = self.cls.expand(x.shape[0], -1, -1)
        return torch.cat([cls, x], dim=1)

class GEGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

class FTTransformerBlock(nn.Module):
    """PreNorm Transformer block for feature tokens."""
    def __init__(self, d_token, n_heads, d_ffn_factor,
                 attention_dropout, ffn_dropout, residual_dropout,
                 skip_first_norm=False):
        super().__init__()
        self.skip_first_norm = skip_first_norm
        self.norm_attn = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            d_token, n_heads, dropout=attention_dropout, batch_first=True
        )
        #互相attention
        self.norm_ffn = nn.LayerNorm(d_token)
        d_hidden = int(d_token * d_ffn_factor)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_hidden * 2),
            GEGLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_hidden, d_token),
        )
        self.residual_dropout = residual_dropout

    def _drop(self, x):
        if self.residual_dropout > 0:
            return F.dropout(x, self.residual_dropout, self.training)
        return x

    def forward(self, x):
        # FT-Transformer uses PreNorm; in the original implementation, the first
        # normalization of the first layer is removed for better optimization.
        #为了训练更稳定
        x_res = x if self.skip_first_norm else self.norm_attn(x)
        x_res, _ = self.attn(x_res, x_res, x_res, need_weights=False)
        x = x + self._drop(x_res)

        x_res = self.ffn(self.norm_ffn(x))
        x = x + self._drop(x_res)
        return x

class FTTransformerBackbone(nn.Module):
    """
    FT-Transformer encoder for numerical tabular data.

    Returns:
        tokens:     (B, d_in + 1, D), including [CLS]
        cls_feature:(B, D)
        logits:     (B, n_labels)
    """
    def __init__(self, d_in, n_labels,
                 d_token=128, n_layers=2, n_heads=8,
                 d_ffn_factor=2.0,
                 attention_dropout=0.15,
                 ffn_dropout=0.10,
                 residual_dropout=0.0,
                 token_bias=True):
        super().__init__()
        self.tokenizer = NumericalFeatureTokenizer(d_in, d_token, bias=token_bias)
        self.blocks = nn.ModuleList([
            FTTransformerBlock(
                d_token=d_token,
                n_heads=n_heads,
                d_ffn_factor=d_ffn_factor,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                residual_dropout=residual_dropout,
                skip_first_norm=(i == 0),
            )
            for i in range(n_layers)
        ])
        self.last_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_labels)

    def forward(self, x):
        tokens = self.tokenizer(x)
        for block in self.blocks:
            tokens = block(tokens)
        cls_feature = F.relu(self.last_norm(tokens[:, 0]))
        logits = self.head(cls_feature)
        return tokens, cls_feature, logits

class LabelWiseNormedLinear(nn.Module):
    """
    Label-wise cosine classifier.

    Input:  x      (B, C, D)
    Output: logits (B, C)
    """
    def __init__(self, n_labels, d_model, s=10.0, use_bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_labels, d_model))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.bias = nn.Parameter(torch.zeros(n_labels)) if use_bias else None

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        logits = (x * w.unsqueeze(0)).sum(dim=-1) * self.s
        if self.bias is not None:
            logits = logits + self.bias.unsqueeze(0)
        return logits

class LabelTableFusionTransformer(nn.Module):
    """
    Label-table Fusion Transformer.

    Unlike the previous TabM version, K/V are FT feature tokens, not model logits.

    feature_tokens: (B, n_features + 1, D), including [CLS]
    L0:             (C, D)
    return L:       (B, C, D)
    """
    def __init__(self, d_model=128, n_heads=8, dim_ff=512,
                 dropout=0.1, n_layers=1):
        super().__init__()
        self.n_layers = n_layers
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_ff, d_model),
            )
            for _ in range(n_layers)
        ])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, feature_tokens, L0):
        B = feature_tokens.shape[0]
        L = L0.unsqueeze(0).expand(B, -1, -1).contiguous()

        for i in range(self.n_layers):
            L2, _ = self.self_attn_layers[i](L, L, L, need_weights=False)
            L = self.norm1[i](L + self.drop(L2))

            L2, _ = self.cross_attn_layers[i](
                query=L,
                key=feature_tokens,
                value=feature_tokens,
                need_weights=False,
            )
            L = self.norm2[i](L + self.drop(L2))

            L2 = self.ffn_layers[i](L)
            L = self.norm3[i](L + self.drop(L2))
        return L

class V2P_FTTransformer_DLCL(nn.Module):
    """
    Overall model:
        FT-Transformer tabular encoder + label-table fusion + label-wise head.
    """
    def __init__(self, d_in, n_labels=24,
                 ft_d_token=128, ft_n_layers=2, ft_n_heads=8,
                 ft_d_ffn_factor=2.0,
                 ft_attention_dropout=0.15,
                 ft_ffn_dropout=0.10,
                 ft_residual_dropout=0.0,
                 ft_token_bias=True,
                 d_model=128, n_heads=8, dim_ff=512,
                 trans_dropout=0.1, n_label_fusion_layers=1,
                 label_prior=None):
        super().__init__()
        assert d_model == ft_d_token, "Keep D_MODEL == FT_D_TOKEN for direct token fusion."
        self.n_labels = n_labels
        self.d_model = d_model

        self.encoder = FTTransformerBackbone(
            d_in=d_in,
            n_labels=n_labels,
            d_token=ft_d_token,
            n_layers=ft_n_layers,
            n_heads=ft_n_heads,
            d_ffn_factor=ft_d_ffn_factor,
            attention_dropout=ft_attention_dropout,
            ffn_dropout=ft_ffn_dropout,
            residual_dropout=ft_residual_dropout,
            token_bias=ft_token_bias,
        )

        # Always use external CODER+cui2vec label semantic priors.
        # The prior matrix is stored as a trainable parameter.
        assert label_prior is not None, "label_prior must be provided in this always-prior version."
        assert label_prior.shape[0] == n_labels
        in_dim = label_prior.shape[1]
        self.label_proj = nn.Linear(in_dim, d_model)
        nn.init.xavier_uniform_(self.label_proj.weight)
        nn.init.zeros_(self.label_proj.bias)

        label_prior_tensor = torch.tensor(label_prior, dtype=torch.float32)
        self.label_prior = nn.Parameter(label_prior_tensor, requires_grad=True)

        self.label_fusion = LabelTableFusionTransformer(
            d_model=d_model,
            n_heads=n_heads,
            dim_ff=dim_ff,
            dropout=trans_dropout,
            n_layers=n_label_fusion_layers,
        )

        self.label_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(trans_dropout),
            nn.Linear(d_model // 2, d_model),
        )
        self.label_score = LabelWiseNormedLinear(
            n_labels=n_labels,
            d_model=d_model,
            s=10.0,
            use_bias=True,
        )
        # For contrastive learning, keep the sample feature close to the
        # actual feature used by the tabular classifier, like MultiPep-DLCL.
        self.sample_cl_proj = nn.Identity()

        # MultiPep-DLCL derives label CL prototypes from the label classifier
        # weights. Here we do the same with the label-wise classifier weights.
        self.label_proto_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def _get_label_emb(self):
        return self.label_proj(self.label_prior)

    def forward(self, x):
        feature_tokens, cls_feature, logits_tab = self.encoder(x)
        L0 = self._get_label_emb()

        L = self.label_fusion(feature_tokens, L0)       # (B, C, D)
        L_head = self.label_head(L)                    # (B, C, D)
        logits_lab = self.label_score(L_head)          # (B, C)

        S_cl = self.sample_cl_proj(cls_feature)        # (B, D)
        label_proto = self.label_proto_proj(self.label_score.weight)  # (C, D)
        return logits_tab, logits_lab, S_cl, label_proto

def _build_model(d_in, label_prior, n_outputs, device=DEVICE):
    model = V2P_FTTransformer_DLCL(
        d_in=d_in,
        n_labels=n_outputs,
        ft_d_token=FT_D_TOKEN,
        ft_n_layers=FT_N_LAYERS,
        ft_n_heads=FT_N_HEADS,
        ft_d_ffn_factor=FT_D_FFN_FACTOR,
        ft_attention_dropout=FT_ATTN_DROPOUT,
        ft_ffn_dropout=FT_FFN_DROPOUT,
        ft_residual_dropout=FT_RESIDUAL_DROPOUT,
        ft_token_bias=FT_TOKEN_BIAS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        dim_ff=DIM_FF,
        trans_dropout=TRANS_DROPOUT,
        n_label_fusion_layers=N_LABEL_FUSION_LAYERS,
        label_prior=label_prior,
    ).to(device)
    return model
