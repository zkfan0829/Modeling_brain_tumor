#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation for rigid CT–MRI registration.

Loads the best model checkpoint, reconstructs the dataset and 7:1:2 split,
and evaluates alignment quality on CT brain masks by:
  (1) Applying the ground-truth transform T to the original mask
  (2) Applying the model's predicted transform (Pred) on top of that
Then computes overlap/surface metrics vs the original mask:
  - Dice (DSC)
  - Hausdorff Distance (HD max)
  - HD95 (95th percentile symmetric surface distance)

Since different pipelines train the network to predict either T or T^{-1},
we report BOTH compositions:
  A)  Mask -> T -> Pred            (label: after_T_then_pred)
  B)  Mask -> T -> Pred^{-1}       (label: after_T_then_pred_inv)

Whichever yields better metrics is the one matching your training label convention.

Dependencies: torch, numpy, pandas, SimpleITK
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import SimpleITK as sitk

# --- Local modules (must exist, as in training) ---
from model_cnn import RigidRegCNN
from model_vit3d import RigidRegViT
from utils import (
    set_seed,
    build_or_load_params,
    BrainRigidDataset,
    make_loaders,
)
try:
    # Optional helper if available; otherwise we fallback to path construction
    from utils import case_paths as _maybe_case_paths
except Exception:
    _maybe_case_paths = None

# -----------------------------
# Helpers: geometry & transforms
# -----------------------------

def _image_center_physical(img: sitk.Image) -> Tuple[float, float, float]:
    """Return physical center of a SimpleITK image."""
    size = np.array(list(img.GetSize()), dtype=np.float64)
    return img.TransformContinuousIndexToPhysicalPoint((size - 1.0) / 2.0)

def params_to_euler3d(
    img_ref: sitk.Image,
    params: np.ndarray,
    *,
    degrees: bool = True
) -> sitk.Euler3DTransform:
    """
    Build a SimpleITK Euler3DTransform from 6 params with center at img_ref center.

    Expected order: [rx, ry, rz, tx, ty, tz]
      - rx, ry, rz in degrees if degrees=True (converted to radians internally)
      - tx, ty, tz in millimeters
    """
    assert params.shape[-1] == 6, f"Expect 6 params, got shape {params.shape}"
    rx, ry, rz, tx, ty, tz = [float(x) for x in params]
    if degrees:
        d2r = np.pi / 180.0
        rx, ry, rz = rx * d2r, ry * d2r, rz * d2r
    T = sitk.Euler3DTransform()
    T.SetRotation(rx, ry, rz)
    T.SetTranslation((tx, ty, tz))
    T.SetCenter(_image_center_physical(img_ref))
    return T

def resample_mask(
    mask: sitk.Image,
    transform: sitk.Transform,
    *,
    default_value: int = 0
) -> sitk.Image:
    """
    Resample a binary mask to its own geometry with a given transform.
    Nearest-neighbor interpolation, preserves spacing/direction/origin.
    """
    return sitk.Resample(
        mask,
        mask,  # reference geometry
        transform,
        sitk.sitkNearestNeighbor,
        default_value,
        mask.GetPixelID(),
    )

# --- Add this helper (top-level, above evaluate) ---
def mask_batch_to_sitk(mask_obj, spacing_xyz: tuple[float, float, float]) -> sitk.Image:
    """
    Convert a mask from the batch (torch.Tensor | np.ndarray | sitk.Image)
    into a binary SimpleITK image with spacing (x,y,z) in mm.
    """
    if isinstance(mask_obj, sitk.Image):
        return sitk.Cast(mask_obj > 0, sitk.sitkUInt8)

    import numpy as np
    try:
        import torch
        if isinstance(mask_obj, torch.Tensor):
            arr = mask_obj.detach().cpu().numpy()
        else:
            arr = np.asarray(mask_obj)
    except Exception:
        arr = np.asarray(mask_obj)

    # Reduce to (D,H,W)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[None, ...]

    arr_bin = (arr > 0.5).astype(np.uint8)
    img = sitk.GetImageFromArray(arr_bin)  # expects [z, y, x]
    sx, sy, sz = spacing_xyz
    img.SetSpacing((sx, sy, sz))
    return img

def composite_transform_3d(transforms: list[sitk.Transform]) -> sitk.Transform:
    """
    Compose multiple 3D transforms in given order: result = Tn ∘ ... ∘ T2 ∘ T1.
    """
    comp = sitk.CompositeTransform(3)
    for T in transforms:
        comp.AddTransform(T)
    return comp

# -----------------------------
# Helpers: metrics (DSC, HD, HD95)
# -----------------------------

def dice_coef(A: sitk.Image, B: sitk.Image) -> float:
    """Dice for binary masks (label > 0). Robust to empty sets."""
    a = sitk.GetArrayFromImage(A) > 0
    b = sitk.GetArrayFromImage(B) > 0
    inter = (a & b).sum()
    sa = a.sum()
    sb = b.sum()
    if sa == 0 and sb == 0:
        return 1.0
    if sa + sb == 0:
        return 0.0
    return 2.0 * inter / (sa + sb)

def _surface_distances_mm(ref: sitk.Image, test: sitk.Image) -> np.ndarray:
    """
    Symmetric surface distances between ref and test (both binary),
    using image spacing via SimpleITK distance maps.
    Returns an array of distances in mm (both directions concatenated).
    """
    # Generate surfaces (contours)
    ref_surf = sitk.LabelContour(ref > 0)
    test_surf = sitk.LabelContour(test > 0)

    # Distance maps (use spacing)
    # SignedMaurerDistanceMap with useImageSpacing=True gives distances in mm.
    ref_dm = sitk.SignedMaurerDistanceMap(ref > 0, insideIsPositive=False,
                                          squaredDistance=False, useImageSpacing=True)
    test_dm = sitk.SignedMaurerDistanceMap(test > 0, insideIsPositive=False,
                                           squaredDistance=False, useImageSpacing=True)

    ref_dm_np = sitk.GetArrayFromImage(ref_dm)
    test_dm_np = sitk.GetArrayFromImage(test_dm)
    ref_surf_np = sitk.GetArrayFromImage(ref_surf) > 0
    test_surf_np = sitk.GetArrayFromImage(test_surf) > 0

    # Distances from ref surface to test object
    d_ref_to_test = np.abs(test_dm_np[ref_surf_np])
    # Distances from test surface to ref object
    d_test_to_ref = np.abs(ref_dm_np[test_surf_np])

    # Handle empty surfaces gracefully
    d1 = d_ref_to_test if d_ref_to_test.size > 0 else np.array([np.inf])
    d2 = d_test_to_ref if d_test_to_ref.size > 0 else np.array([np.inf])
    return np.concatenate([d1, d2], axis=0)

def hd_and_hd95_mm(A: sitk.Image, B: sitk.Image) -> Tuple[float, float]:
    """
    Return (HD_max, HD95) in millimeters between two binary masks.
    """
    dists = _surface_distances_mm(A, B)
    # If both are empty -> zero distances
    if np.isinf(dists).all():
        return 0.0, 0.0
    dists = dists[np.isfinite(dists)]
    if dists.size == 0:
        return 0.0, 0.0
    hd = float(np.max(dists))
    hd95 = float(np.percentile(dists, 95.0))
    return hd, hd95

# -----------------------------
# Main evaluation
# -----------------------------

def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return ckpt

def _safe_case_paths(root: Path, case_id: str) -> Tuple[Optional[Path], Optional[Path], Path]:
    """
    Best-effort retrieval of (ct_path, mr_path, mask_path). If utils.case_paths
    is available, use it; else construct conventional paths.
    """
    if _maybe_case_paths is not None:
        try:
            return _maybe_case_paths(case_id)  # expected to return (ct, mr, mask)
        except Exception:
            pass
    ct = root / "preprocessed" / case_id / "ct_norm.nii.gz"
    mr = root / "preprocessed" / case_id / "mr_norm.nii.gz"
    mask = root / "preprocessed" / case_id / "brain_mask_ct.nii.gz"
    return ct, mr, mask

def _extract_params_row(df_row: pd.Series) -> np.ndarray:
    """
    Try a few common column namings to pull rx,ry,rz,tx,ty,tz (in deg/mm).
    Adjust here if your CSV uses different names.
    """
    candidates = [
        ("rx_deg","ry_deg","rz_deg","tx_mm","ty_mm","tz_mm"),
        ("rot_x_deg","rot_y_deg","rot_z_deg","trans_x_mm","trans_y_mm","trans_z_mm"),
        ("rx","ry","rz","tx","ty","tz"),
    ]
    for cols in candidates:
        if all(c in df_row for c in cols):
            return np.array([df_row[c] for c in cols], dtype=float)
    raise KeyError("Could not find transform columns in params_df row. "
                   "Expected one of sets like rx_deg/ry_deg/... and tx_mm/ty_mm/...")
    
# --- Replace your evaluate(...) with this ---
def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    out_csv = Path(args.out_csv)

    # ---------------- Load model & training cfg ----------------
    model_type = args.model
    if model_type == 'CNN':
        model = RigidRegCNN().to(device)
    elif model_type == 'VIT':
        model = RigidRegViT(
            embed_dim=192,      # divisible by 6
            depth=12,           # ← use 12 like ViT-B
            num_heads=12,        
            patch_size=(16,16,16)
            ).to(device)
    ckpt = load_checkpoint(model, ckpt_path, device)
    model.eval()

    # Pull dataset config from checkpoint if present
    cfg = ckpt.get("cfg", {})
    ROOT = Path(cfg.get("ROOT", args.root))
    CSV_SUMMARY = Path(cfg.get("CSV_SUMMARY", ROOT / "summary.csv"))
    CSV_PARAMS = Path(cfg.get("CSV_PARAMS", ROOT / "rigid_params.csv"))
    NUM_SAMPLES_PER_CASE = int(cfg.get("NUM_SAMPLES_PER_CASE", 5))
    SEED = int(cfg.get("SEED", 42))

    # ---------------- Rebuild dataset & split ----------------
    set_seed(SEED)
    params_df = build_or_load_params(CSV_SUMMARY, CSV_PARAMS, NUM_SAMPLES_PER_CASE)
    dataset = BrainRigidDataset(params_df, target_size=(256, 256, 256))  # or None if you want native sizes

    train_loader, val_loader, test_loader = make_loaders(
        dataset, batch_size=1, seed=SEED, num_workers=args.num_workers
    )
    if hasattr(test_loader, "dataset") and hasattr(test_loader.dataset, "indices"):
        test_indices = list(test_loader.dataset.indices)
    else:
        n = len(dataset)
        test_indices = list(range(int(round(0.8 * n)), n))

    # ---------------- Loop over test samples ----------------
    rows = []
    with torch.inference_mode():
        for idx in test_indices:
            # 1) Fetch sample dict directly from dataset
            batch = dataset[idx]  # {'ct','mr','mask','tumor'(opt),'six_params','meta'}
            ct_t   = batch["ct"]                 # [1,D,H,W] float32
            mr_t   = batch["mr"]                 # [1,D,H,W] float32 (moved by T)
            mask_t = batch["mask"]               # [1,D,H,W] float32 {0,1}
            tumor_t = batch.get("tumor", None)   # [1,D,H,W] float32 {0,1} or None
            six    = batch["six_params"]         # torch [6]
            meta   = batch.get("meta", {})

            # 2) Build model input and predict
            x = torch.cat([ct_t, mr_t], dim=0).unsqueeze(0).to(device)   # (1,2,D,H,W)
            pred = model(x).detach().cpu().numpy().reshape(-1)           # (6,)

            # 3) IDs, spacing, GT params, MASK (and TUMOR if present) from batch (no filesystem I/O)
            case_id = str(meta.get("case_id", idx))
            ct_spacing_xyz = tuple(meta.get("ct_spacing", (1.0, 1.0, 1.0)))  # (x,y,z) mm

            gt_params = six.detach().cpu().numpy().astype(float)             # (6,)
            mask_ct_bin = mask_batch_to_sitk(mask_t, ct_spacing_xyz)

            tumor_ct_bin = None
            if tumor_t is not None:
                tumor_ct_bin = mask_batch_to_sitk(tumor_t, ct_spacing_xyz)

            # 4) Build transforms w.r.t. CT geometry center
            T_gt   = params_to_euler3d(mask_ct_bin, gt_params, degrees=True)
            T_pred = params_to_euler3d(mask_ct_bin, pred, degrees=True)
            T_pred_inv = sitk.Transform(T_pred).GetInverse()

            # 5) Compose & resample: original -> T -> Pred (and Pred^{-1})
            T_then_Pred = composite_transform_3d([T_gt, T_pred])
            mask_after_T = resample_mask(mask_ct_bin, T_gt)
            mask_after_T_then_pred = resample_mask(mask_ct_bin, T_then_Pred)

            T_then_PredInv = composite_transform_3d([T_gt, T_pred_inv])
            mask_after_T_then_pred_inv = resample_mask(mask_ct_bin, T_then_PredInv)

            # ---- Tumor paths (only if tumor exists) ----
            if tumor_ct_bin is not None:
                tumor_after_T = resample_mask(tumor_ct_bin, T_gt)
                tumor_after_T_then_pred = resample_mask(tumor_ct_bin, T_then_Pred)
                tumor_after_T_then_pred_inv = resample_mask(tumor_ct_bin, T_then_PredInv)
            else:
                tumor_after_T = None
                tumor_after_T_then_pred = None
                tumor_after_T_then_pred_inv = None

            # 6) Metrics (mask)
            dsc_T = dice_coef(mask_ct_bin, mask_after_T)
            hd_T, hd95_T = hd_and_hd95_mm(mask_ct_bin, mask_after_T)

            dsc_TP = dice_coef(mask_ct_bin, mask_after_T_then_pred)
            hd_TP, hd95_TP = hd_and_hd95_mm(mask_ct_bin, mask_after_T_then_pred)

            dsc_TPinv = dice_coef(mask_ct_bin, mask_after_T_then_pred_inv)
            hd_TPinv, hd95_TPinv = hd_and_hd95_mm(mask_ct_bin, mask_after_T_then_pred_inv)

            # 6b) Metrics (tumor) — NaN if tumor missing
            import numpy as np
            if tumor_ct_bin is not None:
                dsc_T_tumor = dice_coef(tumor_ct_bin, tumor_after_T)
                hd_T_tumor, hd95_T_tumor = hd_and_hd95_mm(tumor_ct_bin, tumor_after_T)

                dsc_TP_tumor = dice_coef(tumor_ct_bin, tumor_after_T_then_pred)
                hd_TP_tumor, hd95_TP_tumor = hd_and_hd95_mm(tumor_ct_bin, tumor_after_T_then_pred)

                dsc_TPinv_tumor = dice_coef(tumor_ct_bin, tumor_after_T_then_pred_inv)
                hd_TPinv_tumor, hd95_TPinv_tumor = hd_and_hd95_mm(tumor_ct_bin, tumor_after_T_then_pred_inv)
            else:
                dsc_T_tumor = np.nan; hd_T_tumor = np.nan; hd95_T_tumor = np.nan
                dsc_TP_tumor = np.nan; hd_TP_tumor = np.nan; hd95_TP_tumor = np.nan
                dsc_TPinv_tumor = np.nan; hd_TPinv_tumor = np.nan; hd95_TPinv_tumor = np.nan

            rows.append({
                "case_id": case_id,
                # GT and prediction for sanity/debug
                "gt_rx_deg": gt_params[0], "gt_ry_deg": gt_params[1], "gt_rz_deg": gt_params[2],
                "gt_tx_mm": gt_params[3], "gt_ty_mm": gt_params[4], "gt_tz_mm": gt_params[5],
                "pred_rx_deg": pred[0], "pred_ry_deg": pred[1], "pred_rz_deg": pred[2],
                "pred_tx_mm": pred[3], "pred_ty_mm": pred[4], "pred_tz_mm": pred[5],
                # Metrics after applying only T — mask
                "dsc_after_T": dsc_T,
                "hd_after_T_mm": hd_T, "hd95_after_T_mm": hd95_T,
                # Metrics after T then Pred — mask
                "dsc_after_T_then_pred": dsc_TP,
                "hd_after_T_then_pred_mm": hd_TP, "hd95_after_T_then_pred_mm": hd95_TP,
                # Metrics after T then Pred^{-1} — mask
                "dsc_after_T_then_pred_inv": dsc_TPinv,
                "hd_after_T_then_pred_inv_mm": hd_TPinv, "hd95_after_T_then_pred_inv_mm": hd95_TPinv,
                # ---- Tumor metrics (may be NaN) ----
                "dsc_after_T_tumor": dsc_T_tumor,
                "hd_after_T_tumor_mm": hd_T_tumor, "hd95_after_T_tumor_mm": hd95_T_tumor,
                "dsc_after_T_then_pred_tumor": dsc_TP_tumor,
                "hd_after_T_then_pred_tumor_mm": hd_TP_tumor, "hd95_after_T_then_pred_tumor_mm": hd95_TP_tumor,
                "dsc_after_T_then_pred_inv_tumor": dsc_TPinv_tumor,
                "hd_after_T_then_pred_inv_tumor_mm": hd_TPinv_tumor, "hd95_after_T_then_pred_inv_tumor_mm": hd95_TPinv_tumor,
            })

    # ---------------- Save & summarize ----------------
    df = pd.DataFrame(rows).sort_values("case_id")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    def _summ(col):
        return f"{df[col].mean():.4f} ± {df[col].std():.4f} (n={df[col].notna().sum()})"

    print("\n=== Evaluation (mask alignment) ===")
    print(f"Saved per-case metrics → {out_csv}")
    print("\nAfter applying T only (misalignment baseline):")
    print(f"  DSC:   {_summ('dsc_after_T')}")
    print(f"  HD:    {_summ('hd_after_T_mm')} mm")
    print(f"  HD95:  {_summ('hd95_after_T_mm')} mm")

    print("\nAfter T then Pred (assumes Pred ≈ T or corrective in same direction):")
    print(f"  DSC:   {_summ('dsc_after_T_then_pred')}")
    print(f"  HD:    {_summ('hd_after_T_then_pred_mm')} mm")
    print(f"  HD95:  {_summ('hd95_after_T_then_pred_mm')} mm")

    print("\nAfter T then Pred^{-1} (assumes Pred ≈ T^{-1}):")
    print(f"  DSC:   {_summ('dsc_after_T_then_pred_inv')}")
    print(f"  HD:    {_summ('hd_after_T_then_pred_inv_mm')} mm")
    print(f"  HD95:  {_summ('hd95_after_T_then_pred_inv_mm')} mm")

    # ---- Optional tumor summaries (only if any tumors present) ----
    has_any_tumor = df["dsc_after_T_tumor"].notna().any()
    if has_any_tumor:
        print("\n=== Tumor-only Evaluation (subset with tumor present) ===")
        print("\nAfter applying T only:")
        print(f"  DSC:   {_summ('dsc_after_T_tumor')}")
        print(f"  HD:    {_summ('hd_after_T_tumor_mm')} mm")
        print(f"  HD95:  {_summ('hd95_after_T_tumor_mm')} mm")

        print("\nAfter T then Pred:")
        print(f"  DSC:   {_summ('dsc_after_T_then_pred_tumor')}")
        print(f"  HD:    {_summ('hd_after_T_then_pred_tumor_mm')} mm")
        print(f"  HD95:  {_summ('hd95_after_T_then_pred_tumor_mm')} mm")

        print("\nAfter T then Pred^{-1}:")
        print(f"  DSC:   {_summ('dsc_after_T_then_pred_inv_tumor')}")
        print(f"  HD:    {_summ('hd_after_T_then_pred_inv_tumor_mm')} mm")
        print(f"  HD95:  {_summ('hd95_after_T_then_pred_inv_tumor_mm')} mm")

# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="CNN",
                    help="Pick the model [CNN/VIT]")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                    help="Path to best model checkpoint (.pt) from training.")
    ap.add_argument("--root", type=str, default="/orange/xujie/data/2025_Brain_images/Preprocessed_minmax/",
                    help="Dataset root (used only if not embedded in checkpoint cfg).")
    ap.add_argument("--out_csv", type=str, default="checkpoints/eval_metrics.csv",
                    help="Path to write per-case evaluation metrics (CSV).")
    ap.add_argument("--num_workers", type=int, default=2,
                    help="Dataloader workers (used during split rebuild).")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
