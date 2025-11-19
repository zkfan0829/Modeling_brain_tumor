# classical_rigid.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import SimpleITK as sitk
import itk
import os
def _tensor_or_array_to_sitk(
    vol, spacing_xyz: Tuple[float, float, float]
) -> sitk.Image:
    """
    vol: torch.Tensor | np.ndarray | sitk.Image
         shape [1,D,H,W] or [D,H,W]  (float)
    returns a float32 sitk.Image with spacing=(x,y,z)
    """
    if isinstance(vol, sitk.Image):
        img = sitk.Cast(vol, sitk.sitkFloat32)
        img.SetSpacing(tuple(float(x) for x in spacing_xyz))
        return img

    try:
        import torch
        if isinstance(vol, torch.Tensor):
            arr = vol.detach().cpu().numpy()
        else:
            arr = np.asarray(vol)
    except Exception:
        arr = np.asarray(vol)

    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[None, ...]
    img = sitk.GetImageFromArray(arr.astype(np.float32))  # expects [z,y,x]
    img.SetSpacing(tuple(float(x) for x in spacing_xyz))
    return img


# ---------------------------
# ELASTIX (preferred, if available)
# ---------------------------
# add anywhere above estimate_rigid_elastix(...)
def _to_itk_image(vol, spacing_xyz, is_mask: bool = False):
    """
    Accepts sitk.Image | torch.Tensor | np.ndarray shaped [1,D,H,W] or [D,H,W].
    Returns an ITK image with correct spacing (x,y,z). Masks -> uint8.
    """
    if isinstance(vol, sitk.Image):
        arr = sitk.GetArrayFromImage(vol)  # [z,y,x]
        sp  = vol.GetSpacing()
        ori = vol.GetOrigin()
        dire = vol.GetDirection()
    else:
        try:
            import torch
            if isinstance(vol, torch.Tensor):
                arr = vol.detach().cpu().numpy()
            else:
                arr = np.asarray(vol)
        except Exception:
            arr = np.asarray(vol)
        while arr.ndim > 3:
            arr = arr[0]
        if arr.ndim == 2:
            arr = arr[None, ...]
        sp = spacing_xyz
        ori = (0.0, 0.0, 0.0)
        dire = (1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0)

    if is_mask:
        arr = (arr > 0.5).astype(np.uint8)
        img = itk.image_from_array(arr)
        img = itk.cast_image_filter(img, ttype=[type(img), itk.Image[itk.UC, 3]])
    else:
        img = itk.image_from_array(arr.astype(np.float32))

    img.SetSpacing(tuple(float(x) for x in sp))
    img.SetOrigin(tuple(float(x) for x in ori))
    # Only pass direction if original was SITK (otherwise keep identity)
    # direction (only if we came from a sitk.Image)
    if dire is not None:
        try:
            dir_arr = np.asarray(dire, dtype=float).reshape(3, 3)
            dir_mat = itk.matrix_from_array(dir_arr)   # <- key change
            img.SetDirection(dir_mat)
        except Exception:
            # Fallback to identity if anything goes wrong
            # (registration will still work; geometry is consistent across inputs)
            pass

    return img

# add anywhere above estimate_rigid_elastix(...)
def _sitk_euler_from_elx_params(pmap: "itk.ParameterMap"):
    """
    Builds a SimpleITK Euler3DTransform from an elastix parameter map and
    returns (transform, params_deg_mm[6]).
    """
    rx, ry, rz, tx, ty, tz = [float(x) for x in pmap["TransformParameters"]]
    cx, cy, cz = [float(x) for x in pmap["CenterOfRotationPoint"]]

    T = sitk.Euler3DTransform()
    T.SetCenter((cx, cy, cz))
    T.SetRotation(rx, ry, rz)  # radians
    T.SetTranslation((tx, ty, tz))

    params_deg_mm = np.array([np.degrees(rx), np.degrees(ry), np.degrees(rz),
                              tx, ty, tz], dtype=float)
    return T, params_deg_mm

def estimate_rigid_elastix(
    fixed_img: sitk.Image,
    moving_img: sitk.Image,
    fixed_mask: Optional[sitk.Image] = None,
    *,
    output_directory: Optional[str] = None,
    log_to_console: bool = False,
    log_to_file: bool = True,
    write_result_image: bool = False,
    max_iters: Optional[int] = 1024,
) -> tuple[sitk.Transform, Optional[np.ndarray]]:
    # Convert to ITK (your _to_itk_image already fixed SetDirection)
    f_itk = _to_itk_image(fixed_img, fixed_img.GetSpacing(), is_mask=False)
    m_itk = _to_itk_image(moving_img, moving_img.GetSpacing(), is_mask=False)
    fm_itk = _to_itk_image(fixed_mask, fixed_img.GetSpacing(), is_mask=True) if fixed_mask is not None else None

    # Parameter map
    param_obj = itk.ParameterObject.New()
    pmap = param_obj.GetDefaultParameterMap("rigid")
    pmap["AutomaticTransformInitialization"] = ["true"]
    pmap["AutomaticScalesEstimation"] = ["true"]
    pmap["UseDirectionCosines"] = ["true"]
    if max_iters is not None:
        pmap["MaximumNumberOfIterations"] = [str(int(max_iters))]
    # Avoid writing huge result volumes
    pmap["WriteResultImage"] = ["true" if write_result_image else "false"]
    param_obj.AddParameterMap(pmap)

    # Output/log folder
    if output_directory is None:
        output_directory = tempfile.mkdtemp(prefix="elx_")
    os.makedirs(output_directory, exist_ok=True)

    # Run
    result_img, result_tpo = itk.elastix_registration_method(
        fixed_image=f_itk,
        moving_image=m_itk,
        fixed_mask=fm_itk,
        parameter_object=param_obj,
        output_directory=output_directory,
        log_to_console=bool(log_to_console),
        log_to_file=bool(log_to_file),
    )

    # Last stage params -> SITK Euler3D
    last_idx = result_tpo.GetNumberOfParameterMaps() - 1
    last_map = result_tpo.GetParameterMap(last_idx)
    T_sitk, params_deg_mm = _sitk_euler_from_elx_params(last_map)
    return T_sitk, params_deg_mm

# ---------------------------
# ANTs (antspyx) option
# ---------------------------
 
        
def estimate_rigid_ants(
    fixed_img: sitk.Image,
    moving_img: sitk.Image,
) -> tuple[sitk.Transform, Optional[np.ndarray]]:
    """
    3D rigid registration using ANTs (antspyx).

    Returns:
        (T_sitk, None)

    T_sitk is a SimpleITK transform mapping moving -> fixed, defined in the SAME
    physical coordinate frame as the input SimpleITK images.

    Implementation notes:
    - SimpleITK: GetArrayFromImage -> [z, y, x]
    - ANTs: ants.from_numpy expects [x, y, z]
    - We transpose [z,y,x] -> [x,y,z] and KEEP spacing/origin/direction consistent.
    - Because of that, the affine parameters reported by ANTs are directly usable
      in SimpleITK without extra axis permutations.
    """
    try:
        import ants
    except Exception as e:
        raise RuntimeError("antspyx is not installed. Please `pip install antspyx`.") from e

    def _sitk_to_ants(img: sitk.Image) -> "ants.ANTsImage":
        """
        Convert SimpleITK image -> ANTsImage with matching physical space.

        - Reorders array from [z,y,x] (SITK) to [x,y,z] (ANTs)
        - Copies spacing, origin, and direction so that both toolkits operate
          in the same (x,y,z) physical coordinate system.
        """
        arr_zyx = sitk.GetArrayFromImage(img).astype(np.float32)  # [z,y,x]
        arr_xyz = np.transpose(arr_zyx, (2, 1, 0))                # -> [x,y,z]

        aimg = ants.from_numpy(arr_xyz)

        # Spacing / origin in (x,y,z)
        sx, sy, sz = img.GetSpacing()
        ox, oy, oz = img.GetOrigin()
        aimg.set_spacing((sx, sy, sz))
        aimg.set_origin((ox, oy, oz))

        # Direction: SITK stores 3x3 row-major as flat tuple
        try:
            dire = img.GetDirection()
            if len(dire) == 9:
                aimg.set_direction(dire)
        except Exception:
            pass

        return aimg

    # --- Build ANTs images with consistent geometry ---
    fixed_ants = _sitk_to_ants(fixed_img)
    moving_ants = _sitk_to_ants(moving_img)

    # --- Run ANTs registration (rigid) ---
    reg = ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform="Rigid",   # rigid: rotations + translations
    )

    # Forward transform: moving -> fixed
    tx_path = reg["fwdtransforms"][0]
    atx = ants.read_transform(tx_path)

    p = np.array(atx.parameters, dtype=float)
    fp = np.array(atx.fixed_parameters, dtype=float) if len(atx.fixed_parameters) >= 3 else np.zeros(3, dtype=float)

    # Two common cases:
    # - 12 params: 3x3 matrix + 3 trans (AffineTransform in 3D)
    # - 6 params: 3 angles + 3 trans (Euler-style rigid)
    if p.size == 12:
        # Affine (general rigid from ANTs)
        M = p[:9].reshape(3, 3)     # row-major
        t = p[9:12]

        aff = sitk.AffineTransform(3)
        aff.SetMatrix(M.ravel().tolist())
        aff.SetTranslation(t.tolist())
        aff.SetCenter(fp[:3].tolist())
        return aff, None

    elif p.size == 6:
        
        # (rx, ry, rz, tx, ty, tz) in *radians* + mm, about center fp
        rx, ry, rz, tx, ty, tz = p.tolist()

        T = sitk.Euler3DTransform()
        T.SetCenter(fp[:3].tolist())
        T.SetRotation(rx, ry, rz)          # radians
        T.SetTranslation((tx, ty, tz))
        return T, None

    else:
        raise RuntimeError(
            f"Unexpected ANTs rigid parameter length {p.size}; "
            f"expected 12 (affine) or 6 (rigid). Got: {p}"
        )


# ---------------------------
# Fallback: pure SimpleITK rigid
# ---------------------------
def estimate_rigid_sitk(
    fixed_img: sitk.Image,
    moving_img: sitk.Image,
    fixed_mask: Optional[sitk.Image] = None,
) -> tuple[sitk.Transform, Optional[np.ndarray]]:
    """
    No Elastix/ANTs? Use SimpleITK's ImageRegistrationMethod (Euler3D + MI).
    """
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.2)
    if fixed_mask is not None:
        R.SetMetricFixedMask(sitk.Cast(fixed_mask > 0, sitk.sitkUInt8))

    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=500, convergenceMinimumValue=1e-6, convergenceWindowSize=20)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    init = sitk.CenteredTransformInitializer(
        fixed_img, moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    R.SetInitialTransform(init, inPlace=False)
    outT = R.Execute(fixed_img, moving_img)
    return outT, None


# ---------------------------
# Public convenience
# ---------------------------
def classical_rigid_predict(
    ct_vol, mr_vol, spacing_xyz,
    *,
    fixed_mask: Optional[sitk.Image] = None,
    method: str = "elastix",   # 'elastix' | 'ants' | 'sitk'
    **kw,                     # <- forward debug/tuning knobs
):
    """
    Converts tensors/arrays -> SITK, runs chosen method, returns:
        (T_pred (SITK Transform moving->fixed), params_deg_mm_or_None)
    """
    ct_img = _tensor_or_array_to_sitk(ct_vol, spacing_xyz)
    mr_img = _tensor_or_array_to_sitk(mr_vol, spacing_xyz)

    if method.lower() == "elastix":
        return estimate_rigid_elastix(ct_img, mr_img, fixed_mask=fixed_mask, **kw)

    if method.lower() == "ants":
        return estimate_rigid_ants(ct_img, mr_img)

    # default fallback
    return estimate_rigid_sitk(ct_img, mr_img, fixed_mask=fixed_mask)


