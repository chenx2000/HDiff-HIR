"""
UNet denoising network for HDiff-HIR (Real CASSI reconstruction).

Differs from the simulation version in that it includes `initial_x()`:
    converts the 2D CASSI measurement to a 3D HSI cube (shift-back)
    inside the network forward pass.

Components:
    - LGS_MSA: Local-Global Spectral Multi-head Self-Attention
    - LocalGlobalAttentionBlock: Single attention block (DWConv + LGS_MSA + FFN)
    - UNet: Denoising U-Net with MCGM condition generation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum


# =============================================================================
# Building Blocks
# =============================================================================

class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps.

    Args:
        dim: Embedding dimension (must be even).
        scale: Linear scale applied to timesteps. Default: 1.0.
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample(nn.Module):
    """2x spatial downsampling via strided 4x4 convolution.

    Channels are doubled: in_channels -> in_channels * 2.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.Downsample = nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1, bias=False)

    def forward(self, x, time_emb, y):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")
        return self.Downsample(x)


class Upsample(nn.Module):
    """2x spatial upsampling via transposed convolution.

    Channels are halved: in_channels -> in_channels // 2.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, stride=2, kernel_size=2,
            padding=0, output_padding=0,
        )

    def forward(self, x, time_emb, y):
        return self.upsample(x)


class PreNorm(nn.Module):
    """Layer normalization wrapper applied before a given function."""

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class FeedForward(nn.Module):
    """Feed-forward network with depthwise separable convolutions.

    Args:
        dim: Input/output feature dimension.
        mult: Expansion ratio. Default: 4.
    """

    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """x: (B, H, W, C) -> (B, H, W, C)"""
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


# =============================================================================
# Local-Global Spectral Multi-head Self-Attention (LGS_MSA)
# =============================================================================

class LGS_MSA(nn.Module):
    """Local-Global Spectral Multi-head Self-Attention (Fig. 4).

    When only_local_branch=True, uses only local window attention (LSAB).
    When only_local_branch=False, uses both local and global branches (LGSAB).

    Args:
        dim: Input feature dimension.
        window_size1: Local window size. Default: (8, 8).
        window_size2: Global window size. Default: (16, 16).
        dim_head: Dimension per attention head. Default: 28.
        heads: Number of attention heads. Default: 2.
        only_local_branch: If True, use only local attention. Default: False.
        time_emb_dim: Timestep embedding dimension. Default: None.
        con: If True, this is a condition branch (no time bias). Default: False.
    """

    def __init__(
            self,
            dim,
            window_size1=(8, 8),
            window_size2=(16, 16),
            dim_head=28,
            heads=2,
            only_local_branch=False,
            time_emb_dim=None,
            con=False,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size1 = window_size1
        self.window_size2 = window_size2
        self.only_local_branch = only_local_branch
        self.con = con

        inner_dim = dim_head * heads
        self.inner_dim = inner_dim

        # Time bias projections
        self.time_bias_1 = nn.Linear(time_emb_dim, inner_dim) if time_emb_dim is not None else None
        self.time_bias_2 = nn.Linear(time_emb_dim, inner_dim) if time_emb_dim is not None else None
        self.time_bias_3 = nn.Linear(time_emb_dim, inner_dim) if time_emb_dim is not None else None
        self.time_bias_1_local = nn.Linear(time_emb_dim, inner_dim) if time_emb_dim is not None else None
        self.time_bias_2_local = nn.Linear(time_emb_dim, inner_dim) if time_emb_dim is not None else None
        self.time_bias_3_local = nn.Linear(time_emb_dim, inner_dim) if time_emb_dim is not None else None

        # QKV projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        # Non-local branch: atrous (dilated) convolutions
        self.atrous_conv = nn.Conv2d(
            inner_dim // 2, inner_dim // 2, kernel_size=3, dilation=3, stride=1, padding=3,
        )
        self.atrous_conv_2 = nn.Conv2d(
            inner_dim // 2, inner_dim // 2, kernel_size=3, dilation=3, stride=1, padding=3,
        )

        # Local branch: depthwise convolutions (half channels)
        self.dwconv_q = nn.Conv2d(inner_dim // 2, inner_dim // 2, 3, 1, 1, bias=False, groups=inner_dim // 2)
        self.dwconv_k = nn.Conv2d(inner_dim // 2, inner_dim // 2, 3, 1, 1, bias=False, groups=inner_dim // 2)
        self.dwconv_v = nn.Conv2d(inner_dim // 2, inner_dim // 2, 3, 1, 1, bias=False, groups=inner_dim // 2)

        # Local-only branch: depthwise convolutions (full channels)
        self.dwconv_local_q = nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, bias=False, groups=inner_dim)
        self.dwconv_local_k = nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, bias=False, groups=inner_dim)
        self.dwconv_local_v = nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, bias=False, groups=inner_dim)

    def _forward_local(self, x, time_emb):
        """Local-only attention path (LSAB)."""
        b, h, w, c_in = x.shape
        w_size1 = self.window_size1

        # Dynamic padding for attention window partition
        pad_l = pad_t = 0
        pad_r = (w_size1[1] - w % w_size1[1]) % w_size1[1]
        pad_b = (w_size1[0] - h % w_size1[0]) % w_size1[0]
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        
        # Update h, w after padding
        _, hp, wp, _ = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Add time bias (skip for condition branches)
        if not self.con:
            time_q = self.time_bias_1_local(F.silu(time_emb))[:, :, None, None].permute(0, 2, 3, 1)
            time_k = self.time_bias_2_local(F.silu(time_emb))[:, :, None, None].permute(0, 2, 3, 1)
            time_v = self.time_bias_3_local(F.silu(time_emb))[:, :, None, None].permute(0, 2, 3, 1)
            q = q + time_q
            k = k + time_k
            v = v + time_v

        # Depthwise conv for QKV
        q = self.dwconv_local_q(q.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        k = self.dwconv_local_k(k.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        v = self.dwconv_local_v(v.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # Window partition and attention
        q, k, v = map(
            lambda t: rearrange(t, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c',
                                b0=w_size1[0], b1=w_size1[1]),
            (q, k, v),
        )
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
            (q, k, v),
        )
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = rearrange(out, '(b h w) (b0 b1) c -> b (h b0) (w b1) c',
                        h=hp // w_size1[0], w=wp // w_size1[1], b0=w_size1[0])

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            out = out[:, :h, :w, :].contiguous()
            
        return out

    def _forward_local_global(self, x, time_emb):
        """Local + Global attention path (LGSAB)."""
        b, h, w, c_in = x.shape
        w_size1 = self.window_size1
        w_size2 = self.window_size2

        # Dynamic padding for attention window partition (need divisible by w_size2 which is larger)
        pad_l = pad_t = 0
        pad_r = (w_size2[1] - w % w_size2[1]) % w_size2[1]
        pad_b = (w_size2[0] - h % w_size2[0]) % w_size2[0]
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        
        # Update h, w after padding
        _, hp, wp, _ = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Add time bias
        time_q = self.time_bias_1(F.silu(time_emb))[:, :, None, None].permute(0, 2, 3, 1)
        time_k = self.time_bias_2(F.silu(time_emb))[:, :, None, None].permute(0, 2, 3, 1)
        time_v = self.time_bias_3(F.silu(time_emb))[:, :, None, None].permute(0, 2, 3, 1)
        q = q + time_q
        k = k + time_k
        v = v + time_v

        # Split channels: local branch (first half) + global branch (second half)
        _, _, _, c = q.shape
        q1, q2 = q[:, :, :, :c // 2], q[:, :, :, c // 2:]
        k1, k2 = k[:, :, :, :c // 2], k[:, :, :, c // 2:]
        v1, v2 = v[:, :, :, :c // 2], v[:, :, :, c // 2:]

        # --- Local branch ---
        q1 = self.dwconv_q(q1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        k1 = self.dwconv_k(k1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        v1 = self.dwconv_v(v1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        q1, k1, v1 = map(
            lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                b0=w_size1[0], b1=w_size1[1]),
            (q1, k1, v1),
        )
        q1, k1, v1 = map(
            lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads),
            (q1, k1, v1),
        )
        q1 = q1 * self.scale
        sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1)
        attn1 = sim1.softmax(dim=-1)
        out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
        out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')
        out1 = rearrange(out1, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c',
                         h=hp // w_size1[0], w=wp // w_size1[1], b0=w_size1[0])

        # --- Global (non-local) branch ---
        k2 = self.atrous_conv(k2.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        v2 = self.atrous_conv_2(v2.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        q2, k2, v2 = map(
            lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                b0=w_size2[0], b1=w_size2[1]),
            (q2, k2, v2),
        )
        q2, k2, v2 = map(
            lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads),
            (q2, k2, v2),
        )
        q2 = q2 * self.scale
        sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
        attn2 = sim2.softmax(dim=-1)
        out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)
        out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')
        out2 = rearrange(out2, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c',
                         h=hp // w_size2[0], w=wp // w_size2[1], b0=w_size2[0])

        # Concatenate local and global outputs
        out = torch.cat([out1, out2], dim=-1).contiguous()
        out = self.to_out(out)
        
        # Remove padding
        if pad_r > 0 or pad_b > 0:
            out = out[:, :h, :w, :].contiguous()
            
        return out

    def forward(self, x, time_emb=None, y=None):
        """x: (B, H, W, C) -> (B, H, W, C)"""
        # Dynamic padding inside features is handled in _forward_local and _forward_local_global

        if self.only_local_branch:
            return self._forward_local(x, time_emb)
        else:
            return self._forward_local_global(x, time_emb)


# =============================================================================
# Attention Block and Stacked Blocks
# =============================================================================

class LocalGlobalAttentionBlock(nn.Module):
    """Single Local-Global Attention Block (LSAB or LGSAB, Fig. 3c/d).

    Consists of: DWConv -> LGS_MSA -> FFN, with residual connections.

    Args:
        dim: Feature dimension.
        heads: Number of attention heads (heads=1 -> local only / LSAB).
        time_emb_dim: Timestep embedding dimension.
        con: If True, this is a condition branch. Default: False.
    """

    def __init__(self, dim, heads=8, time_emb_dim=None, con=False):
        super().__init__()
        self.attn = PreNorm(dim, LGS_MSA(
            dim=dim, dim_head=dim, heads=heads,
            only_local_branch=(heads == 1),
            time_emb_dim=time_emb_dim, con=con,
        ))
        self.ffn = PreNorm(dim, FeedForward(dim=dim))
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)

    def forward(self, x, time_emb=None, y=None):
        """x: (B, H, W, C) -> (B, H, W, C)"""
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x) + x
        x = x.permute(0, 2, 3, 1)
        x = self.attn(x, time_emb) + x
        x = self.ffn(x) + x
        return x


class LocalGlobalAttentionBlocks(nn.Module):
    """Stacked Local-Global Attention Blocks.

    Args:
        dim: Feature dimension.
        heads: Number of attention heads.
        num_blocks: Number of stacked blocks.
        time_emb_dim: Timestep embedding dimension.
        con: If True, this is a condition branch. Default: False.
    """

    def __init__(self, dim, heads=4, num_blocks=1, time_emb_dim=None, con=False):
        super().__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(LocalGlobalAttentionBlock(
                heads=heads, dim=dim,
                time_emb_dim=time_emb_dim, con=con,
            ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, time_emb=None, y=None):
        """x: (B, C, H, W) -> (B, C, H, W)"""
        x = x.permute(0, 2, 3, 1)
        for block in self.blocks:
            x = block(x, time_emb)
        x = x.permute(0, 3, 1, 2)
        return x


# =============================================================================
# UNet Denoising Network (Real CASSI)
# =============================================================================

class UNet(nn.Module):
    """Denoising U-Net for HDiff-HIR Real CASSI reconstruction (Fig. 3a).

    Unlike the simulation version, this UNet contains `initial_x()` which
    converts the 2D CASSI measurement to a 3D HSI cube (shift-back operation)
    inside the forward pass.

    Architecture:
        - Encoder: LSAB (×N1) → Downsample → LGSAB (×N2) → Downsample
        - Bottleneck: LGSAB (×N3)
        - Decoder: Upsample → LGSAB (×N2) → Upsample → LSAB (×N1)
        - Skip connections between encoder and decoder

    Model variants (Table I in paper):
        - HDiff-HIR-S: num_blocks=(1, 1, 2), condition_blocks=(2, 2), ~4.86M params
        - HDiff-HIR-M: num_blocks=(2, 2, 2), condition_blocks=(2, 2), ~5.50M params
        - HDiff-HIR-L: num_blocks=(2, 4, 6), condition_blocks=(2, 2), ~13.80M params

    Args:
        img_channels: Number of spectral bands. Default: 28.
        base_channels: Base channel width. Default: 32.
        channel_mults: Channel multipliers for each encoder level. Default: (2, 4).
        num_blocks: Tuple (N1, N2, N3) — number of attention blocks at each
            encoder level and bottleneck. Default: (2, 4, 6) (L variant).
        condition_blocks: Tuple (L1, L2) — number of attention blocks in
            the MCGM condition branches. Default: (2, 2).
        time_emb_scale: Scale for timestep embedding. Default: 1.0.
        num_classes: Number of classes for class conditioning, or None. Default: None.
        initial_pad: Padding applied to input if H/W is not power of 2. Default: 0.
    """

    def __init__(
            self,
            img_channels=28,
            base_channels=32,
            channel_mults=(2, 4),
            num_blocks=(2, 4, 6),
            condition_blocks=(2, 2),
            time_emb_scale=1.0,
            num_classes=None,
            initial_pad=0,
    ):
        super().__init__()

        self.initial_pad = initial_pad
        self.num_classes = num_classes

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels),
        )

        # --- MCGM: Condition Generation Module (Fig. 3b) ---
        # For real data: condition = initial_x(measurement), no mask concatenation
        self.fution = nn.Conv2d(img_channels, img_channels, 3, padding=1)

        # Condition 1: full-resolution path
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)
        self.condition_1 = LocalGlobalAttentionBlocks(
            dim=base_channels, heads=1, time_emb_dim=base_channels,
            num_blocks=condition_blocks[0], con=True,
        )

        # Condition 2: downsampled path
        self.init_conv_3 = nn.Conv2d(img_channels, base_channels, 3, padding=1)
        self.condition_2 = LocalGlobalAttentionBlocks(
            dim=base_channels, heads=1, time_emb_dim=base_channels,
            num_blocks=condition_blocks[1], con=True,
        )
        self.down_sample = Downsample(in_channels=base_channels)

        # --- Denoising Network input ---
        self.init_conv_2 = nn.Conv2d(img_channels, base_channels, 3, padding=1)
        self.first_conv = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)

        # Condition 2 fusion (at second encoder level)
        self.fution_2 = nn.Conv2d(base_channels * 4, base_channels * 2, 1, 1, 0, bias=False)

        # --- Encoder ---
        self.downs = nn.ModuleList()
        now_channels = base_channels

        for i, mult in enumerate(channel_mults):
            self.downs.append(LocalGlobalAttentionBlocks(
                dim=now_channels, heads=now_channels // base_channels,
                time_emb_dim=base_channels, num_blocks=num_blocks[i],
            ))
            self.downs.append(Downsample(in_channels=now_channels))
            now_channels *= 2

        # --- Bottleneck ---
        self.mid = nn.ModuleList([
            LocalGlobalAttentionBlocks(
                dim=now_channels, heads=now_channels // base_channels,
                time_emb_dim=base_channels, num_blocks=num_blocks[-1],
            ),
        ])

        # --- Decoder ---
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            self.ups.append(Upsample(now_channels))
            now_channels //= 2
            self.ups.append(LocalGlobalAttentionBlocks(
                dim=now_channels, heads=now_channels // base_channels,
                time_emb_dim=base_channels, num_blocks=num_blocks[i],
            ))

        # --- Output ---
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)

    def initial_x(self, y):
        """Convert 2D CASSI measurement to 3D HSI cube via shift-back.

        This is a key difference from the simulation version: in real CASSI,
        the raw 2D measurement needs to be converted to a 3D cube inside
        the network (not in data preprocessing).

        Args:
            y: 2D measurement tensor of shape (B, H, W_disp) where
               W_disp = W + (nC-1)*step.

        Returns:
            3D HSI cube of shape (B, nC, H, W).
        """
        nC, step = 28, 2
        bs, row, col = y.shape
        x = torch.zeros(bs, nC, row, row).cuda().float()
        for i in range(nC):
            x[:, i, :, :] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        return x

    def forward(self, xt, time=None, condition=None, y=None, mask=None):
        """
        Args:
            xt: Noisy HSI tensor of shape (B, N, H, W), where N is spectral bands.
            time: Diffusion timestep tensor of shape (B,).
            condition: For real data: 2D measurement of shape (B, H, W_disp).
                       initial_x() is called internally to convert to 3D cube.
            y: Class label (unused). Default: None.
            mask: Physical mask of shape (B, N, H, W), or None.
        Returns:
            Predicted clean HSI tensor of shape (B, N, H, W).
        """
        ip = self.initial_pad
        if ip != 0:
            xt = F.pad(xt, (ip,) * 4)

        if self.time_mlp is not None:
            if time is None:
                raise ValueError("time conditioning was specified but time is not passed")
            time_emb = self.time_mlp(time)
        else:
            time_emb = None

        if self.num_classes is not None and y is None:
            raise ValueError("class conditioning was specified but y is not passed")

        if condition is not None:
            # Real CASSI: convert 2D measurement to 3D cube
            condition = self.initial_x(condition)
            condition = self.fution(condition)

            # Condition 1: full-resolution path
            condition_1 = self.init_conv(condition)
            condition_1 = self.condition_1(condition_1, time_emb, y)

            # Condition 2: downsampled path
            condition_2 = self.init_conv_3(condition)
            condition_2 = self.condition_2(condition_2, time_emb, y)
            condition_2 = self.down_sample(condition_2, time_emb, y)

            xt = self.init_conv_2(xt)
            x = self.first_conv(torch.cat((xt, condition_1), dim=1))

        # --- Encoder with skip connections ---
        skips = []
        for i, layer in enumerate(self.downs):
            if i == 2:
                x = self.fution_2(torch.cat([x, condition_2], dim=1))
            x = layer(x, time_emb, y)
            if isinstance(layer, LocalGlobalAttentionBlocks):
                skips.append(x)

        # --- Bottleneck ---
        for layer in self.mid:
            x = layer(x, time_emb, y)

        # --- Decoder with skip connections ---
        for layer in self.ups:
            if isinstance(layer, LocalGlobalAttentionBlocks):
                x = x + skips.pop()
            x = layer(x, time_emb, y)

        # --- Output ---
        x = self.out_conv(x)

        if self.initial_pad != 0:
            return x[:, :, ip:-ip, ip:-ip]
        else:
            return x
