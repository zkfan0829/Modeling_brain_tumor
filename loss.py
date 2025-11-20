
"""
Differentiable registration losses for rigid CT–MRI.

Implemented losses:
- soft_mutual_information_loss: differentiable MI via kernel-based soft histograms.
- mind_loss: MIND-SSC descriptor L1 distance (3D).
- DiceLoss: binary Dice loss for tumor supervision.

Utilities:
- params_to_affine and warp_image: convert 6 rigid params -> grid_sample warp.

Conventions:
- Predicted params are ordered [rx, ry, rz, tx, ty, tz].
- Rotations are in degrees; translations can be in millimeters or voxels.
- Use `spacing=(sx, sy, sz)` to convert mm -> vox; default assumes 1.0 mm voxels.
- Images are expected float tensors in [0,1], shaped (B,1,D,H,W) and same size.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _deg2rad(x: torch.Tensor) -> torch.Tensor:
    return x * math.pi / 180.0


def params_to_affine(
    params: torch.Tensor,
    vol_shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """
    Convert rigid params to an N×3×4 affine for F.affine_grid in NORMALIZED coords.

    Args:
        params: (B,6) tensor [rx, ry, rz, tx, ty, tz]; rx/ry/rz in degrees.
        vol_shape: (D,H,W) of the input (moving) volume.
        spacing: voxel spacing in mm for (sz, sy, sx). Used to interpret translations
                 given in millimeters. If your translations are already in voxels,
                 pass spacing=(1,1,1) or pre-divide.
    Returns:
        theta: (B, 3, 4) affine mapping normalized output grid -> normalized input coords.
    """
    if params.ndim != 2 or params.shape[1] != 6:
        raise ValueError(f"params must be (B,6), got {tuple(params.shape)}")

    B = params.shape[0]
    device = params.device
    D, H, W = vol_shape

    rx, ry, rz, tx, ty, tz = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4], params[:, 5]

    rx = _deg2rad(rx)
    ry = _deg2rad(ry)
    rz = _deg2rad(rz)

    # Convert mm -> vox (if spacing != 1); then vox -> normalized [-1,1]
    sx, sy, sz = spacing[2], spacing[1], spacing[0]  # map (sz,sy,sx) to xyz order
    tx_vox = tx / (sx if sx != 0 else 1.0)
    ty_vox = ty / (sy if sy != 0 else 1.0)
    tz_vox = tz / (sz if sz != 0 else 1.0)

    tx_n = 2.0 * tx_vox / max(W - 1, 1)
    ty_n = 2.0 * ty_vox / max(H - 1, 1)
    tz_n = 2.0 * tz_vox / max(D - 1, 1)

    cx, sx_ = torch.cos(rx), torch.sin(rx)
    cy, sy_ = torch.cos(ry), torch.sin(ry)
    cz, sz_ = torch.cos(rz), torch.sin(rz)

    # Rotation matrices around x,y,z (right-handed), applied z then y then x
    Rx = torch.stack([
        torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx),
        torch.zeros_like(cx), cx, -sx_,
        torch.zeros_like(cx), sx_,  cx
    ], dim=1).reshape(B, 3, 3)

    Ry = torch.stack([
         cy, torch.zeros_like(cy), sy_,
        torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy),
        -sy_, torch.zeros_like(cy), cy
    ], dim=1).reshape(B, 3, 3)

    Rz = torch.stack([
        cz, -sz_, torch.zeros_like(cz),
        sz_,  cz,  torch.zeros_like(cz),
        torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)
    ], dim=1).reshape(B, 3, 3)

    R = Rz @ (Ry @ Rx)  # (B,3,3)

    t = torch.stack([tx_n, ty_n, tz_n], dim=1).unsqueeze(-1)  # (B,3,1)
    theta = torch.cat([R, t], dim=2)  # (B,3,4)
    return theta


def warp_image(
    moving: torch.Tensor,
    params: torch.Tensor,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Warp `moving` by the rigid transform specified in `params`.

    Args:
        moving: (B,1,D,H,W) float in [0,1].
        params: (B,6) [rx,ry,rz,tx,ty,tz].
        spacing: (sz, sy, sx) mm per voxel to interpret translations; defaults to 1.0mm.
        mode: grid_sample interpolation mode ("bilinear" = trilinear) or "nearest".
        padding_mode: how to sample outside the image ("zeros", "border", "reflection").
        align_corners: standard PyTorch flag; keeping True is typical for registration.
    Returns:
        warped: (B,1,D,H,W)
    """
    if moving.ndim != 5 or moving.shape[1] != 1:
        raise ValueError(f"moving must be (B,1,D,H,W); got {tuple(moving.shape)}")

    B, _, D, H, W = moving.shape
    theta = params_to_affine(params, (D, H, W), spacing=spacing)  # (B,3,4)
    grid = F.affine_grid(theta, size=moving.shape, align_corners=align_corners)
    warped = F.grid_sample(moving, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return warped


def invert_affine(theta: torch.Tensor) -> torch.Tensor:
    """Return the inverse of a batch of 3x4 affine matrices in homogeneous coords."""
    if theta.ndim != 3 or theta.shape[1:] != (3, 4):
        raise ValueError(f"theta must be (B,3,4); got {theta.shape}")
    B = theta.shape[0]
    pad = torch.tensor([0, 0, 0, 1], device=theta.device, dtype=theta.dtype).view(1, 1, 4).expand(B, -1, -1)
    homo = torch.cat([theta, pad], dim=1)  # (B,4,4)
    inv = torch.inverse(homo)
    return inv[:, :3, :4]


def warp_image_with_theta(
    moving: torch.Tensor,
    theta: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Warp `moving` using a precomputed normalized affine ``theta`` (B,3,4)."""
    if moving.ndim != 5 or moving.shape[1] != 1:
        raise ValueError(f"moving must be (B,1,D,H,W); got {tuple(moving.shape)}")
    if theta.ndim != 3 or theta.shape[1:] != (3, 4):
        raise ValueError(f"theta must be (B,3,4); got {theta.shape}")
    grid = F.affine_grid(theta, size=moving.shape, align_corners=align_corners)
    return F.grid_sample(moving, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


class DiceLoss(nn.Module):
    """Binary Dice loss. Inputs expected as probabilities/masks in [0,1].

    The formulation mirrors ``eval.py``'s DSC (2|A∩B|/(|A|+|B|)) while keeping a
    small ``smooth`` term for numerical stability and differentiability. Masks
    are treated as soft probabilities so gradients can flow through the warp
    used for tumor supervision.
    """
class DiceLoss(nn.Module):
    """Binary Dice loss. Inputs expected as probabilities/masks in [0,1]."""

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if preds.shape != targets.shape:
            raise ValueError(f"DiceLoss expects matching shapes, got {preds.shape} vs {targets.shape}")
        B = preds.shape[0]
        preds = preds.reshape(B, -1)
        targets = targets.reshape(B, -1)
        intersection = (preds * targets).sum(dim=1)
        denom = preds.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


def soft_mutual_information_loss(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    params: torch.Tensor,
    *,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_bins: int = 64,
    max_samples: int = 65536,
    sigma: float = 0.02,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable (negative) mutual information between fixed and warped(moving).

    Steps:
      1) Warp moving with `params` using trilinear grid_sample.
      2) Sample up to `max_samples` voxels (optionally masked).
      3) Build soft histograms via Gaussian kernel around bin centers (in [0,1]).
      4) Compute MI = sum p(x,y) * log(p(x,y)/(p(x)p(y))). Return -MI to minimize.

    Args:
        fixed:  (B,1,D,H,W) in [0,1]
        moving: (B,1,D,H,W) in [0,1]
        params: (B,6)
        spacing: voxel spacing (sz,sy,sx) in mm
        num_bins: histogram bins
        max_samples: cap sampled voxels per batch for memory/speed
        sigma: Gaussian width for soft binning (0.01–0.05 works for [0,1] data)
        mask: optional (B,1,D,H,W) boolean/float mask (1=use); broadcast supported
        eps: numerical stability
    Returns:
        loss: scalar tensor (negative MI)
    """
    if fixed.shape != moving.shape:
        raise ValueError(f"fixed and moving must have same shape, got {fixed.shape} vs {moving.shape}")
    if fixed.ndim != 5 or fixed.shape[1] != 1:
        raise ValueError(f"fixed/moving must be (B,1,D,H,W); got {fixed.shape}")

    B, _, D, H, W = fixed.shape
    warped = warp_image(moving, params, spacing=spacing, mode="bilinear", padding_mode="zeros")

    # Flatten and sample
    f = fixed.reshape(B, -1)
    w = warped.reshape(B, -1)

    if mask is not None:
        m = (mask > 0).reshape(B, -1)
    else:
        m = torch.ones_like(f, dtype=torch.bool)

    # Prepare bin centers [0,1]
    device = fixed.device
    bin_centers = torch.linspace(0.0, 1.0, steps=num_bins, device=device).view(1, 1, num_bins)  # (1,1,K)

    losses = []
    for b in range(B):
        f_b = f[b]
        w_b = w[b]
        m_b = m[b]
        idx = torch.nonzero(m_b, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        if idx.numel() > max_samples:
            # uniform random subsample for speed
            perm = torch.randperm(idx.numel(), device=device)[:max_samples]
            idx = idx[perm]
        fx = f_b[idx].unsqueeze(-1).unsqueeze(-1)  # (N,1,1)
        wx = w_b[idx].unsqueeze(-1).unsqueeze(-1)  # (N,1,1)

        # Soft-assign to bins via Gaussian kernel
        # A_x: (N, K) soft assignments for fixed; A_y: (N, K) for warped
        A_x = torch.exp(-0.5 * ((fx - bin_centers) / (sigma + eps)) ** 2)  # (N,1,K)
        A_y = torch.exp(-0.5 * ((wx - bin_centers) / (sigma + eps)) ** 2)  # (N,1,K)
        A_x = A_x.squeeze(1)
        A_y = A_y.squeeze(1)

        # Normalize along bins to form probabilities per voxel
        A_x = A_x / (A_x.sum(dim=1, keepdim=True) + eps)
        A_y = A_y / (A_y.sum(dim=1, keepdim=True) + eps)

        # Joint soft-histogram: (KxK) = (KxN) @ (NxK)
        # Pxy[i,j] = sum_n A_x[n,i] * A_y[n,j] / N
        Pxy = (A_x.transpose(0, 1) @ A_y)  # (K,K)
        Pxy = Pxy / (Pxy.sum() + eps)

        Px = Pxy.sum(dim=1, keepdim=True)  # (K,1)
        Py = Pxy.sum(dim=0, keepdim=True)  # (1,K)

        # MI = sum Pxy * log(Pxy/(Px*Py))
        denom = Px @ Py  # (K,K)
        mi = (Pxy * torch.log((Pxy + eps) / (denom + eps))).sum()
        losses.append(-mi)  # negative MI for minimization

    if len(losses) == 0:
        # No valid voxels; return 0 to avoid NaN
        return torch.zeros((), device=fixed.device)

    return torch.stack(losses).mean()


def _avg_pool3d(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    return F.avg_pool3d(x, kernel_size=k, stride=1, padding=k // 2)


def mind_descriptor(
    img: torch.Tensor,
    patch_size: int = 3,
    offsets: Tuple[Tuple[int, int, int], ...] = (
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
    ),
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute a simplified 3D MIND-SSC descriptor with 6 offsets.

    Args:
        img: (B,1,D,H,W) in [0,1]
        patch_size: odd int; local patch extent for SSD (default 3)
        offsets: tuple of 3D neighbor offsets
        eps: stability
    Returns:
        desc: (B, len(offsets), D, H, W)
    """
    if img.ndim != 5 or img.shape[1] != 1:
        raise ValueError(f"img must be (B,1,D,H,W); got {tuple(img.shape)}")

    B, _, D, H, W = img.shape
    # Local variance estimate
    mu = _avg_pool3d(img, k=patch_size)
    mu2 = _avg_pool3d(img * img, k=patch_size)
    var = torch.clamp(mu2 - mu * mu, min=eps)

    descs = []
    for dz, dy, dx in offsets:
        shifted = torch.roll(img, shifts=(dz, dy, dx), dims=(2, 3, 4))
        ssd = _avg_pool3d((img - shifted) ** 2, k=patch_size)
        d = torch.exp(-ssd / (var + eps))
        descs.append(d)
    desc = torch.cat(descs, dim=1)  # (B, O, D,H,W)
    return desc


def mind_loss(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    params: torch.Tensor,
    *,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    downsample: int = 2,
    patch_size: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    L1 distance between MIND descriptors of fixed and warped(moving).

    Args:
        fixed:  (B,1,D,H,W) in [0,1]
        moving: (B,1,D,H,W) in [0,1]
        params: (B,6)
        spacing: passed to warp_image
        downsample: optional integer factor to reduce compute (>=1). If >1, both
                    images are avg-pooled by `downsample` before MIND.
        patch_size: MIND patch size
        eps: stability
    Returns:
        loss: scalar tensor
    """
    if fixed.shape != moving.shape:
        raise ValueError(f"fixed and moving must have same shape, got {fixed.shape} vs {moving.shape}")
    if fixed.ndim != 5 or fixed.shape[1] != 1:
        raise ValueError(f"fixed/moving must be (B,1,D,H,W); got {fixed.shape}")

    warped = warp_image(moving, params, spacing=spacing, mode="bilinear", padding_mode="zeros")

    x = fixed
    y = warped
    if downsample > 1:
        x = F.avg_pool3d(x, kernel_size=downsample, stride=downsample)
        y = F.avg_pool3d(y, kernel_size=downsample, stride=downsample)

    Dx = mind_descriptor(x, patch_size=patch_size, eps=eps)
    Dy = mind_descriptor(y, patch_size=patch_size, eps=eps)

    return (Dx - Dy).abs().mean()
