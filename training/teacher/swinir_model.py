"""SwinIR Neural Network Model for Image Super-Resolution.

Author: Antigravity
Purpose: Production implementation of SwinIR (Swin Transformer for Image Restoration)
         for Classical Image Super-Resolution x4.
Reference: "SwinIR: Image Restoration Using Swin Transformer" (Liang et al., ICCV 2021)
"""

import math
from typing import Sequence, Optional, Type, Tuple, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic depth per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """Multilayer Perceptron (MLP) block for Swin Transformer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition input tensor into non-overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size (int): Window size.

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse windows back to feature map tensor.

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size.
        H (int): Height of image.
        W (int): Width of image.

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.

    Supports both shifted and non-shifted window attention.
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define relative position bias table: (2*Wh-1 * 2*Ww-1, num_heads)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Get pair-wise relative position index for each token inside window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for WindowAttention.

        Args:
            x: Input features of shape (num_windows*B, N, C) where N = Wh*Ww.
            mask (Optional[torch.Tensor]): Attention mask of shape (num_windows, Wh*Ww, Wh*Ww) or None.

        Returns:
            torch.Tensor: Output features of shape (num_windows*B, N, C).
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (B_, num_heads, N, head_dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block (STB)."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0..window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size: Tuple[int, int]) -> torch.Tensor:
        """Calculate attention mask for shifted window attention."""
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        """Forward pass for SwinTransformerBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H*W, C).
            x_size (Tuple[int, int]): Current spatial resolution (H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, H*W, C).
        """
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, f"Input feature size ({L}) does not match H*W ({H*W})"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # Calculate dynamic mask if spatial resolution changed
            if x_size != self.input_resolution:
                attn_mask = self.calculate_mask(x_size).to(x.device)
            else:
                attn_mask = self.attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B, H, W, C

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class ResidualGroup(nn.Module):
    """Residual Swin Transformer Block (RSTB)."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Sequence[float] = (),
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        resi_connection: str = "1conv",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # 3 consecutive convs for deeper residual extraction
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )
        else:
            self.conv = nn.Identity()

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        """Forward pass for ResidualGroup.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H*W, C).
            x_size (Tuple[int, int]): Spatial resolution (H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, H*W, C).
        """
        H, W = x_size
        shortcut = x
        for block in self.blocks:
            x = block(x, x_size)

        # Convert to B, C, H, W for spatial convolution
        B, L, C = x.shape
        x_spatial = x.transpose(1, 2).view(B, C, H, W)
        x_spatial = self.conv(x_spatial)
        x_spatial = x_spatial.flatten(2).transpose(1, 2)

        return shortcut + x_spatial


class Upsample(nn.Sequential):
    """Sub-pixel Convolution Upsample module for SwinIR."""

    def __init__(self, scale: int, num_feat: int) -> None:
        m = []
        if (scale & (scale - 1)) == 0:  # scale is power of 2
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported upscale factor scale={scale}")
        super().__init__(*m)


class SwinIR(nn.Module):
    """SwinIR model for Classical Image Super-Resolution.

    Reference Architecture Specs for Classical SR x4 (SwinIR-M):
    - upscale = 4
    - in_chans = 3
    - img_size = 64
    - window_size = 8
    - img_range = 1.0
    - depths = [6, 6, 6, 6, 6, 6]
    - embed_dim = 180
    - num_heads = [6, 6, 6, 6, 6, 6]
    - mlp_ratio = 2.0
    - upsampler = 'pixelshuffle'
    - resi_connection = '1conv'
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 180,
        depths: Sequence[int] = (6, 6, 6, 6, 6, 6),
        num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        patch_norm: bool = True,
        upscale: int = 4,
        img_range: float = 1.0,
        upsampler: str = "pixelshuffle",
        resi_connection: str = "1conv",
        num_feat: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.img_range = img_range
        if in_chans == 3:
            # Mean RGB normalization values used in DIV2K training
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.register_buffer("mean", torch.tensor(rgb_mean).view(1, 3, 1, 1))
        else:
            self.register_buffer("mean", torch.zeros(1, in_chans, 1, 1))

        self.upscale = upscale
        self.window_size = window_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # 1. Shallow Feature Extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 2. Deep Feature Extraction (Sequential RSTB blocks)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                input_resolution=(self.patches_resolution[0], self.patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)

        self.norm = norm_layer(embed_dim)

        # Build feature fusion conv after body
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )
        else:
            self.conv_after_body = nn.Identity()

        # 3. High-Resolution Reconstruction
        if upsampler == "pixelshuffle":
            # Classical SR upsampling backend
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, in_chans, 3, 1, 1)
        elif upsampler == "pixelshuffledirect":
            # Lightweight SR upsampling backend
            self.conv_up = nn.Conv2d(embed_dim, in_chans * (upscale ** 2), 3, 1, 1)
            self.upsample = nn.PixelShuffle(upscale)
            self.conv_last = nn.Conv2d(in_chans, in_chans, 3, 1, 1)
        else:
            raise ValueError(f"Unsupported upsampler type: {upsampler}")

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize layer weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Deep feature extraction pass across all RSTB blocks."""
        H, W = x.shape[2], x.shape[3]
        x_first = self.conv_first(x)
        res = x_first  # B, embed_dim, H, W

        # Flatten spatial dimensions for Transformer blocks
        x = x_first.flatten(2).transpose(1, 2)  # B, H*W, embed_dim

        for layer in self.layers:
            x = layer(x, (H, W))

        x = self.norm(x)  # B, H*W, embed_dim
        x = x.transpose(1, 2).view(-1, self.embed_dim, H, W)  # B, embed_dim, H, W

        x = self.conv_after_body(x)
        return x + res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for SwinIR.

        Args:
            x (torch.Tensor): Input LR image tensor of shape (B, in_chans, H, W).

        Returns:
            torch.Tensor: High-Resolution SR image tensor of shape (B, in_chans, H*scale, W*scale).
        """
        # Range normalization
        if self.img_range == 255.0:
            x = x / 255.0
        x = (x - self.mean) * self.img_range

        if hasattr(self, "conv_before_upsample"):
            x = self.forward_features(x)
            x = self.conv_before_upsample(x)
            x = self.upsample(x)
            x = self.conv_last(x)
        elif hasattr(self, "conv_up"):
            x = self.forward_features(x)
            x = self.conv_up(x)
            x = self.upsample(x)
            x = self.conv_last(x)

        # Reverse range normalization
        x = (x / self.img_range) + self.mean
        if self.img_range == 255.0:
            x = x * 255.0

        return x
