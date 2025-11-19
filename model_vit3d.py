"""
RigidRegViT: A 3D Vision Transformer for rigid parameter regression from concatenated CT+MRI.

- Input:  (B, 2, D, H, W)  float32 (values in [0,1] recommended)
- Output: (B, 6)  [rx, ry, rz, tx, ty, tz]

Design:
- 3D Patch Embedding (Conv3d with kernel=stride=patch_size) -> tokens
- 3D sine-cos positional encoding (shape-agnostic per token grid)
- [CLS] token pooled through Transformer encoder -> regression head

Notes:
- D/H/W should be divisible by patch_size. With target_size=(256,256,256), patch_size=(32,32,32) works well.
- embed_dim must be divisible by 6 (for 3D sin-cos PE).
"""
from __future__ import annotations
from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Patch Embedding (3D)
# ---------------------------
class PatchEmbed3D(nn.Module):
    """3D patchify via Conv3d (kernel=stride=patch_size) -> tokens (B, N, E)."""
    def __init__(self, in_chans: int = 2, embed_dim: int = 192,
                 patch_size: Tuple[int, int, int] = (32, 32, 32)) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Args: x: (B, C=2, D, H, W)
        Returns:
            tokens: (B, N=D'*H'*W', E)
            grid: (D', H', W') token grid size
        """
        x = self.proj(x)                  # (B, E, D', H', W')
        B, E, Dp, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x, (Dp, Hp, Wp)


# ---------------------------
# Positional Encoding (3D sin-cos)
# ---------------------------
@torch.no_grad()
def get_1d_sincos_pos_embed(dim: int, length: int, device: torch.device) -> torch.Tensor:
    """Return (length, dim) with standard sin-cos positional embeddings. dim must be even."""
    if dim % 2 != 0:
        raise ValueError(f"1D sincos dim must be even; got {dim}")
    pe = torch.zeros(length, dim, device=device)
    position = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)  # (L,1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (L,dim)


@torch.no_grad()
def get_3d_sincos_pos_embed(embed_dim: int, D: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """3D separable sin-cos positional embeddings of shape (D*H*W, embed_dim).
    Requires embed_dim % 6 == 0 so that each axis gets an even dim for sin/cos.
    """
    if embed_dim % 6 != 0:
        raise ValueError(f"embed_dim must be divisible by 6 for 3D sin-cos; got {embed_dim}")
    dim_per_axis = embed_dim // 3
    if dim_per_axis % 2 != 0:
        raise ValueError(f"embed_dim//3 must be even (for sin/cos pairs); got {dim_per_axis}")

    pe_z = get_1d_sincos_pos_embed(dim_per_axis, D, device)  # (D, dim/3)
    pe_y = get_1d_sincos_pos_embed(dim_per_axis, H, device)  # (H, dim/3)
    pe_x = get_1d_sincos_pos_embed(dim_per_axis, W, device)  # (W, dim/3)

    # Combine via concat along channel for each grid location
    grid_z = pe_z[:, None, None, :]  # (D,1,1,Cz)
    grid_y = pe_y[None, :, None, :]  # (1,H,1,Cy)
    grid_x = pe_x[None, None, :, :]  # (1,1,W,Cx)
    grid = torch.cat([
        grid_z.expand(D, H, W, -1),
        grid_y.expand(D, H, W, -1),
        grid_x.expand(D, H, W, -1)
    ], dim=-1)  # (D,H,W, embed_dim)
    grid = grid.reshape(D * H * W, embed_dim)
    return grid  # (N, E)


# ---------------------------
# Transformer Blocks
# ---------------------------
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, E)
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------
# RigidRegViT (3D ViT)
# ---------------------------
class RigidRegViT(nn.Module):
    def __init__(
        self,
        in_chans: int = 2,
        embed_dim: int = 192,    # must be divisible by 6
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        patch_size: Tuple[int, int, int] = (32, 32, 32),
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % 6 != 0:
            raise ValueError(f"embed_dim must be divisible by 6; got {embed_dim}")

        self.patch_embed = PatchEmbed3D(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop, attn_drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 6)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, D, H, W) concatenated [CT, MRI]
        Returns:
            (B, 6) rigid params [rx, ry, rz, tx, ty, tz]
        """
        if x.ndim != 5:
            raise ValueError(f"Input must be 5D (B,2,D,H,W); got {tuple(x.shape)}")
        if x.shape[1] != 2:
            raise ValueError(f"Expected 2 channels (CT,MRI); got C={x.shape[1]}")

        B = x.shape[0]

        # Patchify -> tokens
        tok, (Dp, Hp, Wp) = self.patch_embed(x)     # (B, N, E)
        # Positional encoding per grid
        pos = get_3d_sincos_pos_embed(tok.shape[-1], Dp, Hp, Wp, tok.device)  # (N, E)
        pos = pos.unsqueeze(0).expand(B, -1, -1)  # (B, N, E)

        # Prepend CLS token and run encoder
        cls = self.cls_token.expand(B, -1, -1)    # (B,1,E)
        x = torch.cat([cls, tok + pos], dim=1)    # (B, 1+N, E)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Regress from CLS
        cls_out = x[:, 0]                         # (B, E)
        params = self.head(cls_out)               # (B, 6)
        return params


if __name__ == "__main__":
    # Quick self-test: two shapes
    model = RigidRegViT(embed_dim=192, depth=4, num_heads=6, patch_size=(32,32,32))
    for shp in [(1,2,256,256,256), (2,2,192,224,256)]:
        x = torch.randn(*shp)
        with torch.no_grad():
            y = model(x)
        print(f"Input {shp} -> output {tuple(y.shape)}")
