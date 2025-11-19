"""Cross-modal rigid regressor with self- and cross-attention plus attention map exposure."""
from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """3D patch embedding for single-modality volumes."""

    def __init__(self, in_chans: int = 1, embed_dim: int = 128,
                 patch_size: Tuple[int, int, int] = (16, 16, 16)) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D tensor (B,C,D,H,W); got {x.shape}")
        x = self.proj(x)
        B, E, Dp, Hp, Wp = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return tokens, (Dp, Hp, Wp)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttentionWithMaps(nn.Module):
    """Multi-head self-attention that returns the per-head attention maps."""

    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_maps = self.attn_drop(attn)
        out = attn_maps @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn


class CrossAttention(nn.Module):
    """Cross-attention (queries from one modality, keys/values from the other)."""

    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, queries: torch.Tensor, kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Nq, C = queries.shape
        Nk = kv.shape[1]
        q = self.q_proj(queries).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kv).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kv).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_maps = self.attn_drop(attn)
        out = attn_maps @ v
        out = out.transpose(1, 2).reshape(B, Nq, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn


class SelfAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttentionWithMaps(dim, num_heads, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        delta, attn = self.attn(self.norm1(x))
        x = x + delta
        x = x + self.mlp(self.norm2(x))
        return x, attn


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        self.ct_norm = nn.LayerNorm(dim)
        self.mr_norm = nn.LayerNorm(dim)
        self.ct_cross = CrossAttention(dim, num_heads, attn_drop=drop, proj_drop=drop)
        self.mr_cross = CrossAttention(dim, num_heads, attn_drop=drop, proj_drop=drop)
        self.ct_ffn_norm = nn.LayerNorm(dim)
        self.mr_ffn_norm = nn.LayerNorm(dim)
        self.ct_ffn = MLP(dim, mlp_ratio, drop)
        self.mr_ffn = MLP(dim, mlp_ratio, drop)

    def forward(self, ct_tokens: torch.Tensor, mr_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        ct_delta, ct_attn = self.ct_cross(self.ct_norm(ct_tokens), self.mr_norm(mr_tokens))
        mr_delta, mr_attn = self.mr_cross(self.mr_norm(mr_tokens), self.ct_norm(ct_tokens))
        ct_tokens = ct_tokens + ct_delta
        mr_tokens = mr_tokens + mr_delta
        ct_tokens = ct_tokens + self.ct_ffn(self.ct_ffn_norm(ct_tokens))
        mr_tokens = mr_tokens + self.mr_ffn(self.mr_ffn_norm(mr_tokens))
        return ct_tokens, mr_tokens, (ct_attn, mr_attn)


class CrossModalAttnRigidRegressor(nn.Module):
    """Rigid regressor that keeps separate CT/MR encoders and exposes attention maps."""

    def __init__(
        self,
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        cross_heads: int = 4,
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.ct_embed = PatchEmbed3D(1, embed_dim, patch_size)
        self.mr_embed = PatchEmbed3D(1, embed_dim, patch_size)
        self.ct_blocks = nn.ModuleList([SelfAttnBlock(embed_dim, num_heads, drop=drop) for _ in range(depth)])
        self.mr_blocks = nn.ModuleList([SelfAttnBlock(embed_dim, num_heads, drop=drop) for _ in range(depth)])
        self.cross_block = CrossAttentionBlock(embed_dim, cross_heads, drop=drop)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 6),
        )
        self.token_grid: Optional[Tuple[int, int, int]] = None

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass returning rigid params and attention maps.

        Args:
            x: (B, 2, D, H, W) where channel 0 is CT and 1 is MR.
        Returns:
            dict with ``rigid_params`` plus attention maps and token grid shape for mapping
            attention indices back to spatial patches (Dp, Hp, Wp).
        """
        if x.ndim != 5 or x.shape[1] != 2:
            raise ValueError(f"Expected input (B,2,D,H,W); got {x.shape}")
        ct, mr = x[:, :1], x[:, 1:2]
        ct_tokens, grid = self.ct_embed(ct)
        mr_tokens, grid_mr = self.mr_embed(mr)
        if grid != grid_mr:
            raise ValueError("CT and MR token grids must match")
        self.token_grid = grid  # (Dp, Hp, Wp) so attention map indices map back to patch grid

        ct_maps: List[torch.Tensor] = []
        mr_maps: List[torch.Tensor] = []
        for blk in self.ct_blocks:
            ct_tokens, attn = blk(ct_tokens)
            ct_maps.append(attn)
        for blk in self.mr_blocks:
            mr_tokens, attn = blk(mr_tokens)
            mr_maps.append(attn)

        ct_tokens, mr_tokens, cross_maps = self.cross_block(ct_tokens, mr_tokens)
        fused = torch.cat([ct_tokens.mean(dim=1), mr_tokens.mean(dim=1)], dim=-1)
        params = self.head(fused)
        return {
            "rigid_params": params,
            "ct_attn_maps": ct_maps,
            "mr_attn_maps": mr_maps,
            "cross_attn_maps": {"ct_to_mr": cross_maps[0], "mr_to_ct": cross_maps[1]},
            "token_grid": grid,
        }


if __name__ == "__main__":
    model = CrossModalAttnRigidRegressor()
    x = torch.randn(2, 2, 64, 64, 64)
    out = model(x)
    print("Rigid params:", out["rigid_params"].shape)
    print("CT attn layers:", len(out["ct_attn_maps"]))
    print("Token grid:", out["token_grid"])
