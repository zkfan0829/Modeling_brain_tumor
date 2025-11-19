from pathlib import Path
import math, random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict, Tuple

import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset, DataLoader
ROOT = Path("/orange/xujie/data/2025_Brain_images/Preprocessed_minmax/")
# Rotation (deg) and translation (mm) ranges
ROT_DEG_RANGE   = (-10.0, 10.0)
TRANS_MM_RANGE  = (-5.0, 5.0)


# ======================
# Utils
# ======================
def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def case_paths(case_id: str):
    case_dir = ROOT / "preprocessed" / case_id
    ct_path   = case_dir / "ct_norm.nii.gz"
    mr_path   = case_dir / "mr_norm.nii.gz"
    mask_path = case_dir / "brain_mask_ct.nii.gz"
    tumor_path = case_dir / "tumor_mask.nii.gz"
    return case_dir, ct_path, mr_path, mask_path, tumor_path

def img_center_phys(img: sitk.Image):
    size = np.array(list(img.GetSize()), dtype=np.float64)    # (x,y,z) in idx
    center_idx = (size - 1) / 2.0
    return img.TransformContinuousIndexToPhysicalPoint(center_idx.tolist())

def make_euler3d(center_xyz, rx_deg, ry_deg, rz_deg, tx_mm, ty_mm, tz_mm):
    rx = math.radians(rx_deg); ry = math.radians(ry_deg); rz = math.radians(rz_deg)
    T = sitk.Euler3DTransform()
    T.SetCenter(center_xyz)                # physical (mm)
    T.SetRotation(rx, ry, rz)              # radians
    T.SetTranslation((tx_mm, ty_mm, tz_mm))
    return T

def transform_to_4x4_with_center(T: sitk.Transform):
    """Effective homogeneous 4×4 in physical space, including rotation about center."""
    M = np.eye(4, dtype=np.float64)
    m = T.GetMatrix()  # 9 values row-major 3x3
    M[0,0:3] = m[0:3]
    M[1,0:3] = m[3:6]
    M[2,0:3] = m[6:9]
    tx, ty, tz = T.GetTranslation()
    c = np.array(T.GetCenter(), dtype=np.float64)
    M[:3, 3] = M[:3,:3] @ (-c) + np.array([tx, ty, tz]) + c
    return M

def resample_mr_to_ct(mr_img: sitk.Image, ct_ref: sitk.Image, T: sitk.Transform, default_val=0.0, interp=sitk.sitkLinear):
    """Apply T to MRI and resample onto CT geometry. OOB filled with 0."""
    moved = sitk.Resample(
        mr_img,
        ct_ref,             # geometry (size/spacing/origin/direction)
        T,
        interp,
        default_val,
        mr_img.GetPixelID()
    )
    return moved

def sitk_to_torch(img: sitk.Image):
    """Return torch tensor with shape [1, D, H, W] (float32)."""
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (z,y,x) = (D,H,W)
    t = torch.from_numpy(arr)[None, ...]                  # add channel=1
    return t

def sitk_to_torch_mask(mask_img: sitk.Image):
    """Return mask as float32 {0,1} tensor [1, D, H, W]."""
    arr = sitk.GetArrayFromImage(mask_img)
    arr = (arr > 0).astype(np.float32)
    return torch.from_numpy(arr)[None, ...]

def load_and_binarize_mask(mask_path: Path) -> sitk.Image:
    """Load mask, binarize (>0 -> 1), cast to UInt8."""
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    m = sitk.ReadImage(str(mask_path))
    m = sitk.Cast(m > 0, sitk.sitkUInt8)
    return m

# --- helper: make target reference with same physical FOV but new size ---
def make_target_ref_like(ct_img: sitk.Image, target_size=(256, 256, 256)) -> sitk.Image:
    size_xyz = np.array(ct_img.GetSize(), dtype=np.float64)       # (x,y,z)
    spacing_xyz = np.array(ct_img.GetSpacing(), dtype=np.float64) # (x,y,z)
    phys_len = size_xyz * spacing_xyz                              # mm extent per axis

    target_size = np.array(target_size, dtype=np.int32)            # (x,y,z)
    target_spacing = phys_len / np.maximum(target_size, 1)

    ref = sitk.Image(int(target_size[0]), int(target_size[1]), int(target_size[2]), ct_img.GetPixelID())
    ref.SetOrigin(ct_img.GetOrigin())
    ref.SetDirection(ct_img.GetDirection())
    ref.SetSpacing(tuple(target_spacing.tolist()))
    return ref

# -----------------
# Utilities
# -----------------

def set_seed(seed: int) -> None:
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _normalize_shapes(ct: torch.Tensor, mr: torch.Tensor, params: torch.Tensor):
    """Ensure batched shapes: ct,mr -> (B,1,D,H,W); params -> (B,6)."""
    if ct.ndim == 3:  # (D,H,W)
        ct = ct.unsqueeze(0).unsqueeze(0)
    elif ct.ndim == 4:  # (1,D,H,W) or (B,D,H,W)
        if ct.shape[0] != 1:  # assume (B,D,H,W)
            ct = ct.unsqueeze(1)
        else:
            ct = ct.unsqueeze(0)  # -> (1,1,D,H,W)
    # else: already (B,1,D,H,W)

    if mr.ndim == 3:
        mr = mr.unsqueeze(0).unsqueeze(0)
    elif mr.ndim == 4:
        if mr.shape[0] != 1:
            mr = mr.unsqueeze(1)
        else:
            mr = mr.unsqueeze(0)

    if params.ndim == 1:  # (6,)
        params = params.unsqueeze(0)
    return ct.float(), mr.float(), params.float()


def _unpack_sample(sample: Any):
    """Accept dict or tuple and return (ct,mr,params) as batched tensors.
    Outputs:
      ct: (B,1,D,H,W), mr: (B,1,D,H,W), params: (B,6)
    """
    if isinstance(sample, dict):
        keys = sample.keys()
        for k in ("ct", "mr", "six_params"):
            if k not in keys:
                raise ValueError(f"Dataset dict must contain keys ct, mr, params; got keys {keys}")
        ct = sample["ct"]; mr = sample["mr"]; params = sample["six_params"]
    elif isinstance(sample, (list, tuple)) and len(sample) == 3:
        ct, mr, params = sample
    else:
        raise ValueError("Dataset sample must be dict(ct,mr,params) or (ct, mr, params) tuple.")

    if not (torch.is_tensor(ct) and torch.is_tensor(mr) and torch.is_tensor(params)):
        raise ValueError("ct, mr, params must be torch.Tensors")

    return _normalize_shapes(ct, mr, params)


def make_loaders(dataset, batch_size: int, seed: int, num_workers: int):
    N = len(dataset)
    if N < 10:
        raise ValueError(f"Dataset too small (N={N}). Need >=10 for 7:1:2 split.")

    n_train = int(round(0.7 * N))
    n_val   = int(round(0.1 * N))
    n_test  = N - n_train - n_val
    if n_test <= 0:
        n_test = 1; n_val = max(n_val - 1, 1); n_train = N - n_val - n_test

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader


# ======================
# Parameter generation / persistence
# ======================
def sample_params():
    rx = random.uniform(*ROT_DEG_RANGE)
    ry = random.uniform(*ROT_DEG_RANGE)
    rz = random.uniform(*ROT_DEG_RANGE)
    tx = random.uniform(*TRANS_MM_RANGE)
    ty = random.uniform(*TRANS_MM_RANGE)
    tz = random.uniform(*TRANS_MM_RANGE)
    return rx, ry, rz, tx, ty, tz

def build_or_load_params(csv_summary="", csv_params="", num_per_case=5):
    """
    If csv_params exists, load it.
    Else, create it by enumerating case_id from summary.csv and sampling N rows per case.
    """
    if csv_params.exists():
        params = pd.read_csv(csv_params)
        needed = {"case_id","sample_idx","rx_deg","ry_deg","rz_deg","tx_mm","ty_mm","tz_mm"}
        missing = needed - set(params.columns)
        if missing:
            raise ValueError(f"{csv_params} missing columns: {missing}")
        return params

    df = pd.read_csv(csv_summary, low_memory=False)
    assert 'case_id' in df.columns, "summary.csv must contain 'case_id'."

    set_seed(SEED)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Enumerate cases"):
        cid = str(r['case_id'])
        _, ct_p, mr_p, mask_p, tumor_p = case_paths(cid)
        if not (ct_p.exists() and mr_p.exists() and mask_p.exists()):
            continue
        for k in range(num_per_case):
            rx, ry, rz, tx, ty, tz = sample_params()
            rows.append({
                "case_id": cid,
                "sample_idx": k,
                "rx_deg": rx, "ry_deg": ry, "rz_deg": rz,
                "tx_mm": tx, "ty_mm": ty, "tz_mm": tz
            })
    params = pd.DataFrame(rows)
    params.to_csv(csv_params, index=False)
    print(f"[INFO] Wrote parameter table -> {csv_params}  ({len(params)} rows)")
    return params


# ======================
# Dataset
# ======================
class BrainRigidDataset(Dataset):
    """
    Serves CT, moved-MR (transformed on-the-fly), CT brain mask, tumor masks, and rigid params.

    Tumor supervision:
      * ``tumor`` is always on the fixed CT grid.
      * ``tumor_moving`` is the same mask warped with the synthetic transform applied to MRI.
        Warping it back during training makes it easy to compute Dice loss in CT space.
    """
    def __init__(self, params_df: pd.DataFrame, target_size=(256, 256, 256), interp=sitk.sitkLinear):
        self.df = params_df.reset_index(drop=True)
        self.target_size = target_size  # set None to disable resizing
        self.interp = interp

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        # preserve zero-padded IDs if needed; fallback to int->str is fine if your folder names are numeric
        cid = str(int(row["case_id"])) #if row["case_id"].isdigit() else str(row["case_id"])
        _, ct_path, mr_path, mask_path, tumor_path = case_paths(cid)

        # Load aligned originals
        ct   = sitk.ReadImage(str(ct_path))
        mr   = sitk.ReadImage(str(mr_path))
        mask = load_and_binarize_mask(mask_path)  # UInt8 {0,1}
        
        # --- Optional tumor mask (may not exist) ---
        tumor = None
        try:
            if tumor_path is not None:
                tumor = load_and_binarize_mask(tumor_path)  # UInt8 {0,1}
        except Exception:
            tumor = None  # keep None if not found or unreadable


        # Build transform around CT center (physical space)
        center = img_center_phys(ct)
        T = make_euler3d(
            center_xyz=center,
            rx_deg=float(row["rx_deg"]),
            ry_deg=float(row["ry_deg"]),
            rz_deg=float(row["rz_deg"]),
            tx_mm=float(row["tx_mm"]),
            ty_mm=float(row["ty_mm"]),
            tz_mm=float(row["tz_mm"]),
        )

        # --- Resampling logic ---
        if self.target_size is not None:
            # resample to target reference preserving CT's physical FOV
            ref = make_target_ref_like(ct, target_size=self.target_size)

            ct_res   = sitk.Resample(ct,   ref, sitk.Transform(), self.interp,                     0.0, ct.GetPixelID())
            mr_res   = sitk.Resample(mr,   ref, T,                 self.interp,                     0.0, mr.GetPixelID())

            # IMPORTANT: masks must use nearest-neighbor and keep label type
            mask_res = sitk.Resample(mask, ref, sitk.Transform(), sitk.sitkNearestNeighbor,         0,  sitk.sitkUInt8)
            # re-binarize defensively in case of any stray vals (shouldn't happen with NN)
            mask_res = sitk.Cast(mask_res > 0, sitk.sitkUInt8)
            
            # Tumor mask if available
            if tumor is not None:
                tumor_ct = sitk.Resample(tumor, ref, sitk.Transform(), sitk.sitkNearestNeighbor,   0,  sitk.sitkUInt8)
                tumor_ct = sitk.Cast(tumor_ct > 0, sitk.sitkUInt8)
                tumor_moving = sitk.Resample(tumor, ref, T, sitk.sitkNearestNeighbor,               0,  sitk.sitkUInt8)
                tumor_moving = sitk.Cast(tumor_moving > 0, sitk.sitkUInt8)
            else:
                tumor_ct = None
                tumor_moving = None

            ct, moved_mr, mask_out, tumor_out, tumor_move_out = ct_res, mr_res, mask_res, tumor_ct, tumor_moving
        else:
            # keep original CT grid; move MR onto CT grid; ensure mask is on CT grid too
            mr_moved = sitk.Resample(mr, ct, T, self.interp, 0.0, mr.GetPixelID())

            # If mask isn't already identical geometry to CT, resample with identity
            same_geom = (
                mask.GetSize()      == ct.GetSize() and
                mask.GetSpacing()   == ct.GetSpacing() and
                mask.GetOrigin()    == ct.GetOrigin() and
                mask.GetDirection() == ct.GetDirection()
            )
            if not same_geom:
                mask_ct = sitk.Resample(mask, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            else:
                mask_ct = mask
            mask_ct = sitk.Cast(mask_ct > 0, sitk.sitkUInt8)
            
            # Tumor to CT grid if available
            if tumor is not None:
                same_geom_t = (
                    tumor.GetSize()      == ct.GetSize() and
                    tumor.GetSpacing()   == ct.GetSpacing() and
                    tumor.GetOrigin()    == ct.GetOrigin() and
                    tumor.GetDirection() == ct.GetDirection()
                )
                if not same_geom_t:
                    tumor_ct = sitk.Resample(tumor, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
                else:
                    tumor_ct = tumor
                tumor_ct = sitk.Cast(tumor_ct > 0, sitk.sitkUInt8)
                tumor_moving = sitk.Resample(tumor_ct, ct, T, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
                tumor_moving = sitk.Cast(tumor_moving > 0, sitk.sitkUInt8)
            else:
                tumor_ct = None
                tumor_moving = None

            ct, moved_mr, mask_out, tumor_out, tumor_move_out = ct, mr_moved, mask_ct, tumor_ct, tumor_moving

        # Convert to tensors
        ct_t   = sitk_to_torch(ct)              # [1, D, H, W], float32
        mr_t   = sitk_to_torch(moved_mr)        # [1, D, H, W], float32
        mask_t = sitk_to_torch_mask(mask_out)   # [1, D, H, W], float32 {0,1}
        if tumor_out is not None and tumor_move_out is not None:
            tumor_t = sitk_to_torch_mask(tumor_out)
            tumor_moving_t = sitk_to_torch_mask(tumor_move_out)
            has_tumor = True
        else:
            tumor_t = torch.zeros_like(ct_t)
            tumor_moving_t = torch.zeros_like(ct_t)
            has_tumor = False

        # Physical metadata (for logging or model use)
        meta = {
            "case_id": cid,
            "ct_spacing": ct.GetSpacing(),        # (x,y,z)
            "ct_origin":  ct.GetOrigin(),
            "ct_direction": ct.GetDirection(),
            "mr_spacing": mr.GetSpacing(),
            "mr_origin":  mr.GetOrigin(),
            "mr_direction": mr.GetDirection(),
        }

        # 6-parameter vector
        six = torch.tensor([
            float(row["rx_deg"]), float(row["ry_deg"]), float(row["rz_deg"]),
            float(row["tx_mm"]),  float(row["ty_mm"]),  float(row["tz_mm"]),
        ], dtype=torch.float32)

        sample = {
            "ct": ct_t,                 # torch [1, D, H, W], float32
            "mr": mr_t,                 # torch [1, D, H, W], float32 (moved)
            "mask": mask_t,             # torch [1, D, H, W], float32 {0,1} (CT brain mask on same grid as ct/mr)
            "tumor": tumor_t,           # torch [1, D, H, W], float32 {0,1}
            "tumor_moving": tumor_moving_t,  # same shape, warped with MR transform
            "tumor_available": torch.tensor(1 if has_tumor else 0, dtype=torch.uint8),
            "six_params": six,          # torch [6]
            "meta": meta
        }
        return sample

    
    
