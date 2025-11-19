"""
Training script for rigid CT–MRI registration with 3 losses: MSE (on params), MI, and MIND.

Updates per your requests:
- Uses your dataset-building prelude (ROOT/CSV_*/ranges/SEED/INTERP) and calls
  `build_or_load_params` to construct `params_df`, then instantiates `BrainRigidDataset`.
- Model & losses handle arbitrary input sizes; no fixed (256,256,256) asserts.
- 7:1:2 split, clear epoch logs, checkpoint on best validation.

Key entry points for navigation:
- Training orchestration lives here in ``main`` (dataset -> loaders -> loops).
- Rigid models: ``RigidRegCNN`` (model_cnn.py), ``RigidRegViT`` (model_vit3d.py),
  and ``CrossModalAttnRigidRegressor`` (model_cross_attn.py).
- Differentiable losses: registration losses plus ``DiceLoss`` for tumor masks in loss.py.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import SimpleITK as sitk

# --- Local modules ---
from model_cnn import RigidRegCNN
from model_vit3d import RigidRegViT
from model_cross_attn import CrossModalAttnRigidRegressor

from loss import soft_mutual_information_loss, mind_loss, warp_image, DiceLoss

from utils import *
from utils import _unpack_sample , _normalize_shapes

# -----------------
# Your config prelude
# -----------------
ROOT = Path("/orange/xujie/data/2025_Brain_images/Preprocessed_minmax/")
CSV_SUMMARY = ROOT / "summary.csv"
CSV_PARAMS  = ROOT / "rigid_params.csv"     # where we persist parameters

# How many augmented samples per case
NUM_SAMPLES_PER_CASE = 5



# Reproducibility
SEED = 42

# Interp for MRI (labels would use NearestNeighbor)
INTERP = sitk.sitkLinear



# -----------------
# Training / Eval
# -----------------

def _extract_tensor(batch: Any, key: str):
    if isinstance(batch, dict) and key in batch:
        val = batch[key]
        if torch.is_tensor(val):
            return val
    return None


def _forward_rigid(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Support models that either return Tensor or dict with ``rigid_params``."""
    out = model(x)
    if isinstance(out, dict):
        if "rigid_params" not in out:
            raise ValueError("Model dict output must contain 'rigid_params'.")
        return out["rigid_params"]
    return out


def compute_losses(
    model: nn.Module,
    batch: Any,
    device: torch.device,
    *,
    mi_bins: int,
    mi_sigma: float,
    mi_samples: int,
    mind_down: int,
    spacing: Tuple[float, float, float],
    weights: Tuple[float, float, float],  # (w_mse, w_mi, w_mind)
    dice_loss_fn: Optional[nn.Module] = None,
    tumor_weight: float = 0.0,
):
    ct, mr, gt = _unpack_sample(batch)
    ct = ct.to(device)   # (B,1,D,H,W)
    mr = mr.to(device)
    gt = gt.to(device)   # (B,6)

    # (B,2,D,H,W)
    x = torch.cat([ct, mr], dim=1)
    pred = _forward_rigid(model, x)  # (B,6)

    # MSE on params
    mse = F.mse_loss(pred, gt)

    mi  = soft_mutual_information_loss(ct, mr, pred, num_bins=mi_bins, sigma=mi_sigma,
                                       max_samples=mi_samples, spacing=spacing)
    mnd = mind_loss(ct, mr, pred, spacing=spacing, downsample=mind_down)

    total = weights[0] * mse + weights[1] * mi + weights[2] * mnd

    metrics = {
        "mse": float(mse.detach().cpu()),
        "mi": float(mi.detach().cpu()),
        "mind": float(mnd.detach().cpu()),
    }

    tumor_loss = None
    tumor_dice = None
    if dice_loss_fn is not None and tumor_weight > 0.0:
        tumor_fixed = _extract_tensor(batch, "tumor")
        tumor_moving = _extract_tensor(batch, "tumor_moving")
        tumor_flags = _extract_tensor(batch, "tumor_available")

        if tumor_fixed is not None and tumor_moving is not None:
            tumor_fixed = tumor_fixed.to(device)
            tumor_moving = tumor_moving.to(device)
            if tumor_flags is not None and torch.is_tensor(tumor_flags):
                flags = tumor_flags.view(-1)
                valid_idx = torch.nonzero(flags > 0, as_tuple=False).squeeze(1)
            else:
                valid_idx = torch.arange(tumor_fixed.shape[0], device=device)

            if valid_idx.numel() > 0:
                tumor_fixed = tumor_fixed.index_select(0, valid_idx.to(device))
                tumor_moving = tumor_moving.index_select(0, valid_idx.to(device))
                params_sel = pred.index_select(0, valid_idx.to(device))
                warped = warp_image(tumor_moving, params_sel, spacing=spacing, mode="nearest")
                tumor_loss = dice_loss_fn(warped, tumor_fixed)
                total = total + tumor_weight * tumor_loss
                tumor_dice = 1.0 - tumor_loss
                metrics["tumor_loss"] = float(tumor_loss.detach().cpu())
                metrics["tumor_dice"] = float(tumor_dice.detach().cpu())

    metrics["total"] = float(total.detach().cpu())
    return total, metrics


def _reduce_metrics(agg):
    n = max(agg["n"], 1)
    out = {
        "mi": agg["mi"] / n,
        "mse": agg["mse"] / n,
        "mind": agg["mind"] / n,
        "total": agg["total"] / n,
    }
    if agg["tumor_n"] > 0:
        out["tumor_loss"] = agg["tumor_loss"] / agg["tumor_n"]
        out["tumor_dice"] = agg["tumor_dice"] / agg["tumor_n"]
    return out


def train_one_epoch(model, loader, opt, device, cfg):
    model.train()
    agg = {"mi": 0.0, "mse": 0.0, "mind": 0.0, "total": 0.0, "n": 0,
           "tumor_loss": 0.0, "tumor_dice": 0.0, "tumor_n": 0}

    for batch in loader:
        opt.zero_grad(set_to_none=True)
        loss, m = compute_losses(
            model, batch, device,
            mi_bins=cfg.mi_bins, mi_sigma=cfg.mi_sigma, mi_samples=cfg.mi_samples,
            mind_down=cfg.mind_down, spacing=cfg.spacing,
            weights=(cfg.w_mse, cfg.w_mi, cfg.w_mind),
            dice_loss_fn=cfg.dice_loss, tumor_weight=cfg.w_tumor,
        )
        loss.backward()
        opt.step()

        agg["mi"] += m["mi"]; agg["mse"] += m["mse"]; agg["mind"] += m["mind"]; agg["total"] += m["total"]; agg["n"] += 1
        if "tumor_loss" in m:
            agg["tumor_loss"] += m["tumor_loss"]
            agg["tumor_dice"] += m.get("tumor_dice", 0.0)
            agg["tumor_n"] += 1

    return _reduce_metrics(agg)


def eval_epoch(model, loader, device, cfg):
    model.eval()
    agg = {"mi": 0.0, "mse": 0.0, "mind": 0.0, "total": 0.0, "n": 0,
           "tumor_loss": 0.0, "tumor_dice": 0.0, "tumor_n": 0}
    with torch.inference_mode():
        for batch in loader:
            loss, m = compute_losses(
                model, batch, device,
                mi_bins=cfg.mi_bins, mi_sigma=cfg.mi_sigma, mi_samples=cfg.mi_samples,
                mind_down=cfg.mind_down, spacing=cfg.spacing,
                weights=(cfg.w_mse, cfg.w_mi, cfg.w_mind),
                dice_loss_fn=cfg.dice_loss, tumor_weight=cfg.w_tumor,
            )
            agg["mi"] += m["mi"]; agg["mse"] += m["mse"]; agg["mind"] += m["mind"]; agg["total"] += m["total"]; agg["n"] += 1
            if "tumor_loss" in m:
                agg["tumor_loss"] += m["tumor_loss"]
                agg["tumor_dice"] += m.get("tumor_dice", 0.0)
                agg["tumor_n"] += 1
    return _reduce_metrics(agg)

def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return ckpt
# -----------------
# Main
# -----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=69)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--model", type=str, choices=("cnn", "vit", "cross_attn"), default="cnn",
                   help="Backbone used for rigid regression.")

    # Loss hyperparams
    p.add_argument("--mi_bins", type=int, default=64)
    p.add_argument("--mi_sigma", type=float, default=0.02)
    p.add_argument("--mi_samples", type=int, default=65536)
    p.add_argument("--mind_down", type=int, default=2)
    p.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0),
                   help="Spacing (sz, sy, sx) mm/voxel; used to interpret translations in mm.")

    # Loss weights
    p.add_argument("--w_mse", type=float, default=1.0)
    p.add_argument("--w_mi", type=float, default=1.0)
    p.add_argument("--w_mind", type=float, default=1.0)
    p.add_argument("--w_tumor", type=float, default=0.0,
                   help="Weight for Dice loss between warped tumor masks.")

    args = p.parse_args()

    # Seed first so the param table & splits are reproducible
    set_seed(SEED)

    # Ensure required dataset utilities exist
    if BrainRigidDataset is None or build_or_load_params is None:
        raise ImportError(
            "Expected 'dataset.py' to define BrainRigidDataset and build_or_load_params. "
            "Please provide them or edit imports above."
        )

    # 1) Prepare (or load) rigid_params.csv
    params_df = build_or_load_params(CSV_SUMMARY, CSV_PARAMS, NUM_SAMPLES_PER_CASE)

    # 2) Instantiate dataset (you can pass additional kwargs your Dataset supports)
    dataset = BrainRigidDataset(
        params_df,
        target_size=(256, 256, 256)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloaders with 7:1:2 split
    train_loader, val_loader, test_loader = make_loaders(dataset, args.batch_size, SEED, args.num_workers)

    # Model
    if args.model == "cnn":
        model = RigidRegCNN().to(device)
    elif args.model == "vit":
        model = RigidRegViT(
            embed_dim=192,
            depth=12,
            num_heads=12,
            patch_size=(16, 16, 16),
        ).to(device)
    else:
        model = CrossModalAttnRigidRegressor().to(device)
    ckpt_path = Path(args.checkpoint_dir) / "best.pt"
    if ckpt_path.exists():
        ckpt = load_checkpoint(model, ckpt_path, device)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"No checkpoint found at {ckpt_path}; starting from scratch.")

    # Sanity check forward pass with one batch
    sample_batch = next(iter(train_loader))
    with torch.no_grad():
        ct, mr, _ = _unpack_sample(sample_batch)
        x = torch.cat([ct, mr], dim=1).to(device)  # (B,2,D,H,W)
        y = model(x)
        if isinstance(y, dict):
            y = y.get("rigid_params")
        if y is None or y.shape[-1] != 6:
            raise RuntimeError(f"Model forward sanity check failed: expected last dim 6, got {tuple(y.shape)}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_total = float("inf")
    best_path = Path(args.checkpoint_dir) / "best.pt"

    dice_loss_fn = DiceLoss().to(device)

    class Cfg:  # pack loss cfg
        mi_bins = args.mi_bins; mi_sigma = args.mi_sigma; mi_samples = args.mi_samples
        mind_down = args.mind_down; spacing = tuple(args.spacing)
        w_mse = args.w_mse; w_mi = args.w_mi; w_mind = args.w_mind; w_tumor = args.w_tumor
        dice_loss = dice_loss_fn

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, opt, device, Cfg)
        val_metrics = eval_epoch(model, val_loader, device, Cfg)

        # Total validation loss with same weights as training
        val_total = (
            args.w_mse * val_metrics["mse"]
            + args.w_mi * val_metrics["mi"]
            + args.w_mind * val_metrics["mind"]
            + args.w_tumor * val_metrics.get("tumor_loss", 0.0)
        )
        scheduler.step(val_total)

        def _fmt(metrics, split):
            base = (f"{split} MI: {metrics['mi']:.4f}, "
                    f"{split} MSE: {metrics['mse']:.4f}, "
                    f"{split} MIND: {metrics['mind']:.4f}")
            if "tumor_loss" in metrics:
                base += f", {split} TumorDiceLoss: {metrics['tumor_loss']:.4f} (Dice={metrics['tumor_dice']:.4f})"
            return base

        print(f"Epoch {epoch}: {_fmt(train_metrics, 'Train')} | {_fmt(val_metrics, 'Val')}")

        # Save on improvement
        if val_total < best_val_total:
            best_val_total = val_total
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "val_total": val_total,
                "cfg": {
                    "ROOT": str(ROOT),
                    "CSV_SUMMARY": str(CSV_SUMMARY),
                    "CSV_PARAMS": str(CSV_PARAMS),
                    "NUM_SAMPLES_PER_CASE": NUM_SAMPLES_PER_CASE,
                    "ROT_DEG_RANGE": ROT_DEG_RANGE,
                    "TRANS_MM_RANGE": TRANS_MM_RANGE,
                    "SEED": SEED,
                    "INTERP": int(INTERP),
                    "args": vars(args),
                },
            }, best_path)
            print(f"  ✔ Saved new best checkpoint to {best_path} (val_total={val_total:.4f})")

    # Final test evaluation
    test_metrics = eval_epoch(model, test_loader, device, Cfg)
    test_msg = (
        f"TEST: MI Loss: {test_metrics['mi']:.4f}, MSE: {test_metrics['mse']:.4f}, "
        f"MIND: {test_metrics['mind']:.4f}"
    )
    if "tumor_loss" in test_metrics:
        test_msg += f", TumorDiceLoss: {test_metrics['tumor_loss']:.4f} (Dice={test_metrics['tumor_dice']:.4f})"
    print(test_msg)


if __name__ == "__main__":
    main()

