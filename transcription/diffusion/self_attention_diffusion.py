import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor

from .natten_diffusion import AdaLayerNorm

class SelfAttention2D_diffusion(nn.Module):
    """
    Self Attention 2D Module for diffusion
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        bias: bool = True,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        diffusion_step: int = 100,
        timestep_type: str = "adalayernorm_abs",
        use_style_enc: bool = False,
        use_chord_enc: bool = False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.timestep_type = timestep_type
        self.use_style_enc = use_style_enc
        self.use_chord_enc = use_chord_enc

        # Timestep conditioning
        if timestep_type is not None:
            self.ln = AdaLayerNorm(i_dim=dim, diffusion_step=diffusion_step, o_dim=None, emb_type=timestep_type)
            if self.use_style_enc:
                self.ln_style = AdaLayerNorm(i_dim=1024, diffusion_step=None, o_dim=2*dim, emb_type='style')

            if self.use_chord_enc:
                self.ln_chord = AdaLayerNorm(i_dim=2048, diffusion_step=None, o_dim=2*dim, emb_type='chord')

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, cond: Tensor, t: Tensor, style_emb: Tensor = None, chord_emb: Tensor = None) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"SelfAttention2D expected a rank-4 input tensor; got {x.dim()=}.")

        B, H, W, C = x.shape
        x = x.reshape(B, H*W, C)  # Flatten for attention computation
        
        # Timestep conditioning
        if self.timestep_type is not None:
            x = self.ln(x, t)

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3 x B x num_heads x (H*W) x head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, H*W, H*W)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute attention output
        x = (attn @ v).transpose(1, 2).reshape(B, H*W, C)

        # LayerNorm
        if self.timestep_type is not None:
            if style_emb is not None and self.use_style_enc:
                x = self.ln_style(x, style_emb=style_emb)
            if chord_emb is not None and self.use_chord_enc:
                x = self.ln_chord(x, chord_emb=chord_emb)

        # Reshape back to 2D
        x = x.reshape(B, H, W, C)
        return self.proj_drop(self.proj(x)), None, t
