"""
Training script for rigid CT–MRI registration with 3 losses: MSE (on params), MI, and MIND.

Updates per your requests:
- Uses your dataset-building prelude (ROOT/CSV_*/ranges/SEED/INTERP) and calls
  `build_or_load_params` to construct `params_df`, then instantiates `BrainRigidDataset`.
- Model & losses handle arbitrary input sizes; no fixed (256,256,256) asserts.
- 7:1:2 split, clear epoch logs, checkpoint on best validation.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import SimpleITK as sitk

# --- Local modules ---
from model_cnn import RigidRegCNN
from model_vit3d import RigidRegViT

from loss import soft_mutual_information_loss, mind_loss

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
):
    ct, mr, gt = _unpack_sample(batch)
    ct = ct.to(device)   # (B,1,D,H,W)
    mr = mr.to(device)
    gt = gt.to(device)   # (B,6)

    # (B,2,D,H,W)
    x = torch.cat([ct, mr], dim=1)
    pred = model(x)  # (B,6)

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
        "total": float(total.detach().cpu()),
    }
    return total, metrics


def train_one_epoch(model, loader, opt, device, cfg):
    model.train()
    agg = {"mi": 0.0, "mse": 0.0, "mind": 0.0, "n": 0}

    for batch in loader:
        opt.zero_grad(set_to_none=True)
        loss, m = compute_losses(
            model, batch, device,
            mi_bins=cfg.mi_bins, mi_sigma=cfg.mi_sigma, mi_samples=cfg.mi_samples,
            mind_down=cfg.mind_down, spacing=cfg.spacing, weights=(cfg.w_mse, cfg.w_mi, cfg.w_mind)
        )
        loss.backward()
        opt.step()

        agg["mi"] += m["mi"]; agg["mse"] += m["mse"]; agg["mind"] += m["mind"]; agg["n"] += 1

    n = max(agg["n"], 1)
    return agg["mi"]/n, agg["mse"]/n, agg["mind"]/n


def eval_epoch(model, loader, device, cfg):
    model.eval()
    agg = {"mi": 0.0, "mse": 0.0, "mind": 0.0, "n": 0}
    with torch.inference_mode():
        for batch in loader:
            loss, m = compute_losses(
                model, batch, device,
                mi_bins=cfg.mi_bins, mi_sigma=cfg.mi_sigma, mi_samples=cfg.mi_samples,
                mind_down=cfg.mind_down, spacing=cfg.spacing, weights=(cfg.w_mse, cfg.w_mi, cfg.w_mind)
            )
            agg["mi"] += m["mi"]; agg["mse"] += m["mse"]; agg["mind"] += m["mind"]; agg["n"] += 1
    n = max(agg["n"], 1)
    return agg["mi"]/n, agg["mse"]/n, agg["mind"]/n

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
    model = RigidRegCNN().to(device)
    #model = RigidRegViT(
    #    embed_dim=192,      # divisible by 6
    #    depth=12,           # ← use 12 like ViT-B
    #    num_heads=12,        
    #    patch_size=(16,16,16)
    #    ).to(device)
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
        if y.shape[-1] != 6:
            raise RuntimeError(f"Model forward sanity check failed: expected last dim 6, got {tuple(y.shape)}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_total = float("inf")
    best_path = Path(args.checkpoint_dir) / "best.pt"

    class Cfg:  # pack loss cfg
        mi_bins = args.mi_bins; mi_sigma = args.mi_sigma; mi_samples = args.mi_samples
        mind_down = args.mind_down; spacing = tuple(args.spacing)
        w_mse = args.w_mse; w_mi = args.w_mi; w_mind = args.w_mind

    for epoch in range(1, args.epochs + 1):
        train_mi, train_mse, train_mind = train_one_epoch(model, train_loader, opt, device, Cfg)
        val_mi, val_mse, val_mind = eval_epoch(model, val_loader, device, Cfg)

        # Total validation loss with same weights as training
        val_total = args.w_mse * val_mse + args.w_mi * val_mi + args.w_mind * val_mind
        scheduler.step(val_total)

        print(
            f"Epoch {epoch}: "
            f"Train MI Loss: {train_mi:.4f}, Train MSE: {train_mse:.4f}, Train MIND: {train_mind:.4f}, "
            f"Val MI Loss: {val_mi:.4f}, Val MSE: {val_mse:.4f}, Val MIND: {val_mind:.4f}"
        )

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
    test_mi, test_mse, test_mind = eval_epoch(model, test_loader, device, Cfg)
    print(
        f"TEST: MI Loss: {test_mi:.4f}, MSE: {test_mse:.4f}, MIND: {test_mind:.4f}"
    )


if __name__ == "__main__":
    main()

