#################################################################################################
# Copyright (c) 2022-2024 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################
from typing import Optional

import math

import torch
from torch import nn, Tensor
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_

from natten.functional import na2d_av, na2d_qk_with_bias, na1d_av, na1d_qk_with_bias

class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AdaLayerNorm(nn.Module):
    def __init__(self, i_dim, diffusion_step, o_dim=None, emb_type="adalayernorm_abs"):
        super().__init__()
        self.emb_type = emb_type
        if emb_type == 'style':
            self.emb = None
            self.layernorm = nn.LayerNorm(o_dim // 2, elementwise_affine=False)
        elif emb_type == 'chord':
            self.emb = None
            self.layernorm = nn.LayerNorm(o_dim // 2, elementwise_affine=False)
        else:
            if "abs" in emb_type:
                self.emb = SinusoidalPosEmb(diffusion_step, i_dim)
            else:
                self.emb = nn.Embedding(diffusion_step, i_dim)
            assert o_dim == None
            o_dim = 2*i_dim
            self.layernorm = nn.LayerNorm(i_dim, elementwise_affine=False)

        self.silu = nn.SiLU()
        self.linear = nn.Linear(i_dim, o_dim)

    def forward(self, x, timestep=None, style_emb=None, chord_emb=None): # TODO : check if valid
        if self.emb_type == 'style':
            assert style_emb != None and timestep == None
            emb = self.linear(self.silu(style_emb)).unsqueeze(1)
        elif self.emb_type == 'chord':
            assert chord_emb != None and timestep == None
            emb = self.linear(self.silu(chord_emb)).unsqueeze(1)
        else:
            assert timestep != None and style_emb == None
            emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
            
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x

class NeighborhoodAttention2D_diffusion(nn.Module):
    """
    Neighborhood Attention 2D Module for diffusion 
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        dilation: int = 1,
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
        self.scale = qk_scale or self.head_dim**-0.5
        self.timestep_type = timestep_type
        self.use_style_enc = use_style_enc
        self.use_chord_enc = use_chord_enc
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation
        if timestep_type != None:
            self.ln = AdaLayerNorm(i_dim=dim, diffusion_step=diffusion_step, o_dim=None, emb_type=timestep_type)
            if self.use_style_enc:
                self.ln_style = AdaLayerNorm(i_dim=1024, diffusion_step=None, o_dim=2*dim, emb_type='style')

            if self.use_chord_enc:
                self.ln_chord = AdaLayerNorm(i_dim=2048, diffusion_step=None, o_dim=2*dim, emb_type='chord')

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, cond:Tensor, t:Tensor, style_emb:Tensor = None, chord_emb: Tensor=None) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"NeighborhoodAttention2D expected a rank-4 input tensor; got {x.dim()=}."
            )

        B, H, W, C = x.shape
        # Pad if the input is small than the minimum supported size
        H_padded, W_padded = H, W
        padding_h = padding_w = 0
        if H < self.window_size or W < self.window_size:
            padding_h = max(0, self.window_size - H_padded)
            padding_w = max(0, self.window_size - W_padded)
            x = pad(x, (0, 0, 0, padding_w, 0, padding_h))
            _, H_padded, W_padded, _ = x.shape
        
        # conditioning t
        if self.timestep_type != None:
            x = x.reshape(B, H*W, C)
            x = self.ln(x, t)
            x = x.reshape(B, H, W, C)

        qkv = (
            self.qkv(x)
            .reshape(B, H_padded, W_padded, 3, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5)
        ) # 3 x B x num_heads x H_padded x W_padded x head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = na2d_qk_with_bias(q, k, self.rpb, self.kernel_size, self.dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = na2d_av(attn, v, self.kernel_size, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H_padded, W_padded, C)

        # Remove padding, if added any
        if padding_h or padding_w:
            x = x[:, :H, :W, :]
        
        # layernorm
        if self.timestep_type != None:
            x = x.reshape(B, H*W, C)
            if style_emb != None and self.use_style_enc:
                x = self.ln_style(x, style_emb=style_emb)
            if chord_emb != None and self.use_chord_enc:
                x = self.ln_chord(x, chord_emb=chord_emb)
            x = x.reshape(B, H, W, C)

        return self.proj_drop(self.proj(x)), None, t

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"has_bias={self.rpb is not None}"
        )


class NeighborhoodCrossAttention2D_diffusion(nn.Module):
    """
    Neighborhood Cross Attention 2D Module for diffusion conditioning
    1 Block contains [AdaLayerNorm, Self Attention, Residual, AdaLayerNorm, Cross Attention, Residual, AdaLayerNorm, MLP, Residual]
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        diffusion_step: int = 100,
        timestep_type: str = "adalayernorm_abs"
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.timestep_type = timestep_type
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation
        if timestep_type != None:
            self.ln1 = AdaLayerNorm(dim=dim, diffusion_step=diffusion_step, emb_type=timestep_type)
            self.ln1_1 = AdaLayerNorm(dim=dim, diffusion_step=diffusion_step, emb_type=timestep_type)
        self.ln2 = nn.LayerNorm(dim)

        self.self_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.cross_kv = nn.Linear(128, dim * 2, bias=qkv_bias)
        self.cross_q = nn.Linear(dim, dim, bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)

        # MLP instead of projection
        self.mlp = nn.Sequential(
                # nn.Linear(dim, dim * 2),
                # nn.Linear(dim * 2, dim),
                nn.Linear(dim, dim),
                nn.Dropout(proj_drop)
        )
    
    def attention(self, q: Tensor, k: Tensor, v: Tensor, B, H_padded, W_padded, C) -> Tensor:
        attn = na2d_qk_with_bias(q, k, self.rpb, self.kernel_size, self.dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = na2d_av(attn, v, self.kernel_size, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H_padded, W_padded, C)
        return x

    def forward(self, x: Tensor, cond:Tensor, t:Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"NeighborhoodAttention2D expected a rank-4 input tensor; got {x.dim()=}."
            )

        B, H, W, C = x.shape
        # Pad if the input is small than the minimum supported size
        H_padded, W_padded = H, W
        padding_h = padding_w = 0
        if H < self.window_size or W < self.window_size:
            padding_h = max(0, self.window_size - H_padded)
            padding_w = max(0, self.window_size - W_padded)
            x = pad(x, (0, 0, 0, padding_w, 0, padding_h))
            _, H_padded, W_padded, _ = x.shape

        # conditioning t
        res = x
        x = x.reshape(B, H*W, C)
        if self.timestep_type != None:
            x = self.ln1(x, t)
        x = x.reshape(B, H, W, C)

        # self attention
        qkv = (
            self.self_qkv(x)
            .reshape(B, H_padded, W_padded, 3, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5)
        ) # 3 x B x num_heads x H_padded x W_padded x head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        x = self.attention(q, k, v, B, H_padded, W_padded, C)

        # residual connection
        x = x + res

        # conditioning t
        res = x
        x = x.reshape(B, H*W, C)
        if self.timestep_type != None:
            x = self.ln1_1(x, t)
        x = x.reshape(B, H, W, C)

        # cross attention
        kv = (
            self.cross_kv(cond)
            .reshape(B, H_padded, W_padded, 2, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5)
        ) # 2 x B x num_heads x H_padded x W_padded x head_dim
        q = (
            self.cross_q(x)
            .reshape(B, H_padded, W_padded, 1, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5)
        ) # 1 x B x num_heads x H_padded x W_padded x head_dim
        q, k, v = q[0], kv[0], kv[1] # TODO: check if this makes sense
        q = q * self.scale
        x = self.attention(q, k, v, B, H_padded, W_padded, C)

        # residual connection
        x = x + res


        # Remove padding, if added any
        if padding_h or padding_w:
            x = x[:, :H, :W, :]

        # layernorm
        x = x.reshape(B, H*W, C)
        x = self.ln2(x)
        x = x.reshape(B, H, W, C)

        return self.mlp(x), cond, t

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"has_bias={self.rpb is not None}"
        )


class NeighborhoodAttention1D_diffusion(nn.Module):
    """
    Neighborhood Attention 1D Module for diffusion 
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        spatial_size: list = [313, 88],
        diffusion_step: int = 100,
        timestep_type: str = "adalayernorm_abs"
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation
        self.ln = AdaLayerNorm(dim=dim, diffusion_step=diffusion_step, emb_type=timestep_type)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, t:Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"NeighborhoodAttention1D expected a rank-3 input tensor; got {x.dim()=}."
            )

        B, L, C = x.shape
        H, W = self.spatial_size[0], self.spatial_size[1]
        # Pad if the input is small than the minimum supported size
        L_padded = L
        padding = 0
        if L_padded < self.window_size:
            padding = max(0, self.window_size - L_padded)
            x = pad(x, (0, 0, 0, padding))
            _, L_padded, _ = x.shape
            assert L_padded == L + padding
        
        # conditioning t
        if L == H: # B, H, C
            x = x.reshape(-1, W, H, C).permute(0,2,1,3).reshape(-1, H*W, C)
            x = self.ln(x, t)
            x = x.reshape(-1, H, W, C).permute(0,2,1,3).reshape(B, H, C)
        if L == W: # B, W, C
            x = x.reshape(-1, H, W, C).reshape(-1, H*W, C)
            x = self.ln(x, t)
            x = x.reshape(-1, H, W, C).reshape(B, W, C)
        
        qkv = (
            self.qkv(x)
            .reshape(B, L_padded, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = na1d_qk_with_bias(q, k, self.rpb, self.kernel_size, self.dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = na1d_av(attn, v, self.kernel_size, self.dilation)
        x = x.permute(0, 2, 1, 3).reshape(B, L_padded, C)

        # Remove padding, if added any
        if padding:
            x = x[:, :L, :]

        return self.proj_drop(self.proj(x)), t

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"has_bias={self.rpb is not None}"
        )




class NeighborhoodAttention2D_diffusion_encoder(nn.Module):
    """
    Neighborhood Attention 2D Module for diffusion 
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        diffusion_step: int = 100,
        timestep_type: str = "adalayernorm_abs"
    ):
        super().__init__()
        self.num_heads = num_heads
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        # if dim == 1:
        #     self.qkv = nn.linear(dim, 48*3, bias=qkv_bias)
        #     dim = 48
        # else:
        self.ln = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.dim = dim
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"NeighborhoodAttention2D expected a rank-4 input tensor; got {x.dim()=}."
            )

        B, H, W, C = x.shape
        C = self.dim
        # Pad if the input is small than the minimum supported size
        H_padded, W_padded = H, W
        padding_h = padding_w = 0
        if H < self.window_size or W < self.window_size:
            padding_h = max(0, self.window_size - H_padded)
            padding_w = max(0, self.window_size - W_padded)
            x = pad(x, (0, 0, 0, padding_w, 0, padding_h))
            _, H_padded, W_padded, _ = x.shape

        # conditioning t
        x = x.reshape(B, H*W, C)
        x = self.ln(x)
        x = x.reshape(B, H, W, C)
        
        qkv = (
            self.qkv(x)
            .reshape(B, H_padded, W_padded, 3, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5)
        ) # 3 x B x num_heads x H_padded x W_padded x head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = na2d_qk_with_bias(q, k, self.rpb, self.kernel_size, self.dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = na2d_av(attn, v, self.kernel_size, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H_padded, W_padded, C)

        # Remove padding, if added any
        if padding_h or padding_w:
            x = x[:, :H, :W, :]

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"has_bias={self.rpb is not None}"
        )