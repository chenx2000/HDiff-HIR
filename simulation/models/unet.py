"""
Denoising Network (UNet) for HDiff-HIR.

The network consists of:
    - UNet: The overall denoising U-Net with MCGM condition integration
    - LGS_MSA: Local-Global Spectral-enhanced Multi-head Self-Attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum


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
        """
        Args:
            x: Timestep tensor of shape (N,).
        Returns:
            Positional embedding of shape (N, dim).
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample(nn.Module):
    """Downsample feature maps by 2x using strided convolution (Conv 4×4, stride 2).

    Doubles the channel count: in_channels → in_channels * 2.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1, bias=False)

    def forward(self, x, time_emb=None, y=None):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")
        return self.downsample(x)


class Upsample(nn.Module):
    """Upsample feature maps by 2x using transposed convolution (DeConv 2×2).

    Halves the channel count: in_channels → in_channels // 2.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2, padding=0
        )

    def forward(self, x, time_emb=None, y=None):
        return self.upsample(x)


class PreNorm(nn.Module):
    """Apply LayerNorm before a given module."""

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class FeedForward(nn.Module):
    """Feed-Forward Network (FFN) used in attention blocks.

    Structure: Conv1×1 → GELU → DwConv3×3 → GELU → Conv1×1.

    Args:
        dim: Input/output channel dimension.
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
        """
        Args:
            x: Tensor of shape (B, H, W, C).
        Returns:
            Tensor of shape (B, H, W, C).
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


# =============================================================================
# Attention Modules
# =============================================================================

class LGS_MSA(nn.Module):
    """Local-Global Spectral-enhanced Multi-head Self-Attention (Fig. 4).

    When only_local_branch=True, this becomes LS-MSA (Fig. 4b) used in LSAB.
    When only_local_branch=False, this becomes LGS-MSA (Fig. 4a) used in LGSAB,
    which splits channels into local (DwConv) and global (dilated conv) branches.

    Args:
        dim: Input channel dimension.
        window_size1: Window size for local branch. Default: (8, 8).
        window_size2: Window size for global branch. Default: (16, 16).
        dim_head: Dimension per attention head.
        heads: Number of attention heads.
        only_local_branch: If True, use only local branch (LS-MSA).
        time_emb_dim: Dimension of time embedding, or None for condition modules.
        con: If True, skip time embedding (used in condition modules).
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

        # Positional embeddings
        if only_local_branch:
            seq_l = window_size1[0] * window_size1[1]
            self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
            nn.init.trunc_normal_(self.pos_emb)
        else:
            seq_l1 = window_size1[0] * window_size1[1]
            self.pos_emb1 = nn.Parameter(torch.Tensor(1, 1, heads, seq_l1, seq_l1))
            seq_l2 = window_size2[0] * window_size2[1]
            self.pos_emb2 = nn.Parameter(torch.Tensor(1, 1, heads, seq_l2, seq_l2))
            nn.init.trunc_normal_(self.pos_emb1)
            nn.init.trunc_normal_(self.pos_emb2)

        # Time embedding projections
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

        # Global branch: dilated convolutions (DiConv 3×3)
        self.atrous_conv = nn.Conv2d(
            inner_dim // 2, inner_dim // 2, kernel_size=3, dilation=3, stride=1, padding=3
        )
        self.atrous_conv_2 = nn.Conv2d(
            inner_dim // 2, inner_dim // 2, kernel_size=3, dilation=3, stride=1, padding=3
        )

        # Local branch for LGS-MSA: depthwise convolutions (DwConv 3×3)
        self.dwconv_q = nn.Conv2d(inner_dim // 2, inner_dim // 2, 3, 1, 1, bias=False, groups=inner_dim // 2)
        self.dwconv_k = nn.Conv2d(inner_dim // 2, inner_dim // 2, 3, 1, 1, bias=False, groups=inner_dim // 2)
        self.dwconv_v = nn.Conv2d(inner_dim // 2, inner_dim // 2, 3, 1, 1, bias=False, groups=inner_dim // 2)

        # Local branch for LS-MSA: depthwise convolutions (DwConv 3×3)
        self.dwconv_local_q = nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, bias=False, groups=inner_dim)
        self.dwconv_local_k = nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, bias=False, groups=inner_dim)
        self.dwconv_local_v = nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, bias=False, groups=inner_dim)

    def forward(self, x, time_emb=None, y=None):
        """
        Args:
            x: Input tensor of shape (B, H, W, C).
            time_emb: Time embedding of shape (B, time_emb_dim), or None.
            y: Unused (kept for interface consistency).
        Returns:
            Output tensor of shape (B, H, W, C).
        """
        b, h, w, _ = x.shape
        w_size1 = self.window_size1
        w_size2 = self.window_size2
        assert h % w_size1[0] == 0 and w % w_size1[1] == 0, \
            'feature map dimensions must be divisible by the window size'

        if self.only_local_branch:
            # LS-MSA path (Fig. 4b)
            return self._forward_local(x, time_emb, h, w, w_size1)
        else:
            # LGS-MSA path (Fig. 4a)
            return self._forward_local_global(x, time_emb, h, w, w_size1, w_size2)

    def _forward_local(self, x, time_emb, h, w, w_size):
        """LS-MSA: Local spectral-enhanced MSA (Fig. 4b)."""
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Add time embedding bias (skip for condition modules)
        if not self.con:
            time_q = self.time_bias_1_local(F.silu(time_emb))[:, :, None, None].permute(0, 2, 3, 1)
            time_k = self.time_bias_2_local(F.silu(time_emb))[:, :, None, None].permute(0, 2, 3, 1)
            time_v = self.time_bias_3_local(F.silu(time_emb))[:, :, None, None].permute(0, 2, 3, 1)
            q = q + time_q
            k = k + time_k
            v = v + time_v

        # Depthwise convolution for spatial mixing
        q = self.dwconv_local_q(q.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        k = self.dwconv_local_k(k.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        v = self.dwconv_local_v(v.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # Window partition → multi-head attention
        q, k, v = map(
            lambda t: rearrange(t, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1]),
            (q, k, v)
        )
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
            (q, k, v)
        )
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = rearrange(out, '(b h w) (b0 b1) c -> b (h b0) (w b1) c',
                        h=h // w_size[0], w=w // w_size[1], b0=w_size[0])
        return out

    def _forward_local_global(self, x, time_emb, h, w, w_size1, w_size2):
        """LGS-MSA: Local-global spectral-enhanced MSA (Fig. 4a)."""
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Add time embedding bias
        time_q = self.time_bias_1(F.silu(time_emb))[:, :, None, None].permute(0, 2, 3, 1)
        time_k = self.time_bias_2(F.silu(time_emb))[:, :, None, None].permute(0, 2, 3, 1)
        time_v = self.time_bias_3(F.silu(time_emb))[:, :, None, None].permute(0, 2, 3, 1)
        q = q + time_q
        k = k + time_k
        v = v + time_v

        # Split channels into local and global halves
        _, _, _, c = q.shape
        q1, q2 = q[:, :, :, :c // 2], q[:, :, :, c // 2:]
        k1, k2 = k[:, :, :, :c // 2], k[:, :, :, c // 2:]
        v1, v2 = v[:, :, :, :c // 2], v[:, :, :, c // 2:]

        # --- Local branch: DwConv 3×3 ---
        q1 = self.dwconv_q(q1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        k1 = self.dwconv_k(k1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        v1 = self.dwconv_v(v1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        q1, k1, v1 = map(
            lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c', b0=w_size1[0], b1=w_size1[1]),
            (q1, k1, v1)
        )
        q1, k1, v1 = map(
            lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads),
            (q1, k1, v1)
        )
        q1 = q1 * self.scale
        sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1)
        sim1 = sim1 + self.pos_emb1
        attn1 = sim1.softmax(dim=-1)
        out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
        out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')
        out1 = rearrange(out1, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c',
                         h=h // w_size1[0], w=w // w_size1[1], b0=w_size1[0])

        # --- Global branch: Dilated Conv 3×3 ---
        k2 = self.atrous_conv(k2.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        v2 = self.atrous_conv_2(v2.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        q2, k2, v2 = map(
            lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c', b0=w_size2[0], b1=w_size2[1]),
            (q2, k2, v2)
        )
        q2, k2, v2 = map(
            lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads),
            (q2, k2, v2)
        )
        q2 = q2 * self.scale
        sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
        sim2 = sim2 + self.pos_emb2
        attn2 = sim2.softmax(dim=-1)
        out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)
        out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')
        out2 = rearrange(out2, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c',
                         h=h // w_size2[0], w=w // w_size2[1], b0=w_size2[0])

        # Concatenate local and global branches → linear projection
        out = torch.cat([out1, out2], dim=-1).contiguous()
        out = self.to_out(out)
        return out


# =============================================================================
# LSAB / LGSAB
# =============================================================================

class LocalGlobalAttentionBlock(nn.Module):
    """Local-Global Attention Block (Fig. 3c/d).

    When heads=1, this acts as LSAB (Local Spectral Attention Block, Fig. 3d).
    When heads>1, this acts as LGSAB (Local-Global Spectral Attention Block, Fig. 3c).

    Structure: DwConv 3×3 → LayerNorm → LGS-MSA/LS-MSA → LayerNorm → FFN

    Args:
        dim: Channel dimension.
        heads: Number of attention heads. heads=1 yields LS-MSA (local only).
        time_emb_dim: Dimension of time embedding, or None.
        con: If True, skip time embedding (for condition modules).
    """

    def __init__(self, dim, heads=8, time_emb_dim=None, con=False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)
        self.attn = PreNorm(dim, LGS_MSA(
            dim=dim, dim_head=dim, heads=heads,
            only_local_branch=(heads == 1),
            time_emb_dim=time_emb_dim, con=con,
        ))
        self.ffn = PreNorm(dim, FeedForward(dim=dim))

    def forward(self, x, time_emb=None, y=None):
        """
        Args:
            x: Input tensor of shape (B, H, W, C) in channel-last format.
            time_emb: Time embedding of shape (B, time_emb_dim), or None.
            y: Unused (kept for interface consistency).
        Returns:
            Output tensor of shape (B, H, W, C).
        """
        # DwConv 3×3 with residual
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x) + x
        x = x.permute(0, 2, 3, 1)

        # Attention with residual
        x = self.attn(x, time_emb) + x

        # FFN with residual
        x = self.ffn(x) + x

        return x


class LocalGlobalAttentionBlocks(nn.Module):
    """Stacked Local-Global Attention Blocks.

    Stacks multiple LocalGlobalAttentionBlock modules sequentially.
    Used as LSAB (×N) or LGSAB (×N) in the denoising network (Fig. 3a).

    Args:
        dim: Channel dimension.
        heads: Number of attention heads per block.
        num_blocks: Number of stacked blocks.
        time_emb_dim: Dimension of time embedding.
        con: If True, skip time embedding (for condition modules).
    """

    def __init__(self, dim, heads=4, num_blocks=1, time_emb_dim=None, con=False):
        super().__init__()
        blocks = [
            LocalGlobalAttentionBlock(
                dim=dim, heads=heads, time_emb_dim=time_emb_dim, con=con
            )
            for _ in range(num_blocks)
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, time_emb=None, y=None):
        """
        Args:
            x: Input tensor of shape (B, C, H, W) in channel-first format.
            time_emb: Time embedding of shape (B, time_emb_dim), or None.
            y: Unused (kept for interface consistency).
        Returns:
            Output tensor of shape (B, C, H, W).
        """
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)
        for block in self.blocks:
            x = block(x, time_emb)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
        return x


# =============================================================================
# UNet Denoising Network
# =============================================================================

class UNet(nn.Module):
    """Denoising U-Net for HDiff-HIR (Fig. 3a).

    The network takes noisy HSI x_t as input and estimates the noise,
    conditioned on the shifted measurement and mask via MCGM (Fig. 3b).

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

        # --- MCGM: Mask-integrated Condition Generation Module (Fig. 3b) ---
        # Fusion of physical mask (M) and shifted measurement (Y)
        self.fution = nn.Conv2d(img_channels * 2, img_channels, 3, padding=1)

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

    def forward(self, xt, time=None, condition=None, y=None, mask=None, lasttime=None):
        """
        Args:
            xt: Noisy HSI tensor of shape (B, N, H, W), where N is spectral bands.
            time: Diffusion timestep tensor of shape (B,).
            condition: Shifted measurement of shape (B, N, H, W), or None.
            y: Class label (unused). Default: None.
            mask: Physical mask of shape (B, N, H, W), or None.
            lasttime: Unused. Kept for interface compatibility.
        Returns:
            Predicted noise tensor of shape (B, N, H, W).
        """
        ip = self.initial_pad
        if ip != 0:
            xt = F.pad(xt, (ip,) * 4)

        # Time embedding
        if self.time_mlp is not None:
            if time is None:
                raise ValueError("time conditioning was specified but time is not passed")
            time_emb = self.time_mlp(time)
        else:
            time_emb = None

        if self.num_classes is not None and y is None:
            raise ValueError("class conditioning was specified but y is not passed")

        # --- MCGM: generate conditions from mask and measurement ---
        if condition is not None:
            condition = self.fution(torch.cat([condition, mask], dim=1))

            # Condition 1: full resolution
            condition_1 = self.init_conv(condition)
            condition_1 = self.condition_1(condition_1, time_emb, y)

            # Condition 2: downsampled
            condition_2 = self.init_conv_3(condition)
            condition_2 = self.condition_2(condition_2, time_emb, y)
            condition_2 = self.down_sample(condition_2, time_emb, y)

            # Fuse noisy input with condition 1
            xt = self.init_conv_2(xt)
            x = self.first_conv(torch.cat((xt, condition_1), dim=1))

        # --- Encoder with skip connections ---
        skips = []
        for i, layer in enumerate(self.downs):
            if i == 2:
                # Fuse condition 2 at the second encoder level
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

        # --- Output projection ---
        x = self.out_conv(x)

        if self.initial_pad != 0:
            return x[:, :, ip:-ip, ip:-ip]
        else:
            return x