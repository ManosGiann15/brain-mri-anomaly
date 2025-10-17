#!/usr/bin/env python3
"""
Export BraTS-style per-subject folders to two outputs:

1) Per-modality PNGs:
   out_modalities/
     train/{t1,t2,t1ce,flair}/*.png
     val/{t1,t2,t1ce,flair}/*.png

2) Fusion PNGs (RGB of 3 modalities):
   out_fusion/
     train/*.png
     val/*.png

Input layout (example):
brats20/
  Train/python scripts/export_patchcore_from_brats.py \python scripts/export_patchcore_from_brats.py \
  --src-train brats20/Train \
  --src-val brats20/Val \
  --out-modalities patchcore_modalities \
  --out-fusion patchcore_fusion \
  --fusion t1ce,t2,flair \
  --export-modalities flair t1 t1ce t2 \
  --image-size 256 256
  --src-train brats20/Train \
  --src-val brats20/Val \
  --out-modalities patchcore_modalities \
  --out-fusion patchcore_fusion \
  --fusion t1ce,t2,flair \
  --export-modalities flair t1 t1ce t2 \
  --image-size 256 256
    BraTS20_Training_001/
      BraTS20_Training_001_flair.nii.gzpython scripts/export_patchcore_from_brats.py \
  --src-train brats20/Train \
  --src-val brats20/Val \
  --out-modalities patchcore_modalities \
  --out-fusion patchcore_fusion \
  --fusion t1ce,t2,flair \
  --export-modalities flair t1 t1ce t2 \
  --image-size 256 256. ;/.
      BraTS20_Training_001_t1.nii.gz
      BraTS20_Training_001_t1ce.nii.gz
      BraTS20_Training_001_t2.nii.gz
    BraTS20_Training_002/
    ...
  Val/
    BraTS20_Validation_001/   # Name doesn't matter; script uses directory name

Usage:
------
python scripts/export_patchcore_from_brats.py \
  --src-train brats20/Train \
  --src-val brats20/Val \
  --out-modalities patchcore_modalities \
  --out-fusion patchcore_fusion \
  --fusion t1ce,t2,flair \
  --export-modalities flair t1 t1ce t2 \
  --image-size 256 256
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Optional NIfTI support
try:
    import nibabel as nib  # for .nii/.nii.gz
    _HAS_NIB = True
except Exception:
    _HAS_NIB = False


# ----------------------------
# IO & image helpers
# ----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_modality(path: Path) -> np.ndarray:
    """
    Load a 2D/3D volume from .nii/.nii.gz (float32) or .npy (float32).
    Returns array shaped (H, W, S) for 3D or (H, W) for 2D.
    """
    if path.suffix == ".npy":
        arr = np.load(path)
        return arr.astype(np.float32, copy=False)
    else:
        if not _HAS_NIB:
            raise ImportError(
                "nibabel is required to read NIfTI (.nii/.nii.gz). pip install nibabel"
            )
        img = nib.load(str(path))
        arr = img.get_fdata(dtype=np.float32)
        if arr.ndim == 3:
            return arr
        elif arr.ndim == 2:
            return arr
        else:
            raise ValueError(f"Unexpected NIfTI dims for {path}: {arr.shape}")


def is_3d(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and min(arr.shape) > 1


def robust_clip_and_norm(
    x: np.ndarray, p_lo: float = 0.5, p_hi: float = 99.5
) -> np.ndarray:
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    x = np.clip(x, lo, hi)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def to_uint8(img01: np.ndarray) -> np.ndarray:
    return (np.clip(img01, 0, 1) * 255.0).round().astype(np.uint8)


def is_empty_slice(img: np.ndarray, thresh: float = 0.005) -> bool:
    """Consider slice empty if non-zero fraction < thresh."""
    if img.size == 0:
        return True
    return (np.count_nonzero(img) / img.size) < thresh


def resize_uint8(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize HxW or HxWx3 uint8 with bilinear resampling (PIL)."""
    if img.ndim == 2:
        pil = Image.fromarray(img, mode="L").resize(size, resample=Image.BILINEAR)
        return np.array(pil)
    elif img.ndim == 3 and img.shape[2] == 3:
        pil = Image.fromarray(img, mode="RGB").resize(size, resample=Image.BILINEAR)
        return np.array(pil)
    else:
        raise ValueError(f"Unexpected image shape for resize: {img.shape}")


def save_rgb(img3: np.ndarray, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    Image.fromarray(img3, mode="RGB").save(out_path)


# ----------------------------
# Subject discovery (BraTS-style)
# ----------------------------
MOD_ALIASES = {
    "flair": ("flair",),
    "t1": ("t1",),
    "t1ce": ("t1ce", "t1ce"),
    "t2": ("t2",),
}

NIFTI_SUFFIXES = (".nii", ".npy")  # handle .nii and .npy; accept .nii.gz via suffixes logic


def find_modality_file(subj_dir: Path, subj_name: str, modality: str) -> Optional[Path]:
    """
    Find file inside subject folder that matches the modality.
    Tries:
      {subj_name}_{modality}.nii / .nii.gz / .npy
      '*{modality}*' as fallback
    """
    # exact pattern first (.nii and .npy)
    cand_nii = subj_dir / f"{subj_name}_{modality}.nii"
    cand_npy = subj_dir / f"{subj_name}_{modality}.npy"
    cand_niigz = subj_dir / f"{subj_name}_{modality}.nii.gz"
    for cand in (cand_niigz, cand_nii, cand_npy):
        if cand.exists():
            return cand

    # fallback: any file containing modality token
    tokens = MOD_ALIASES.get(modality, (modality,))
    for tok in tokens:
        for h in sorted(subj_dir.glob(f"*{tok}*")):
            # accept .npy, .nii, .nii.gz
            if h.suffix == ".npy":
                return h
            if h.suffix == ".nii":
                return h
            if "".join(h.suffixes).endswith(".nii.gz"):
                return h
    return None


def list_subjects(src_split_dir: Path) -> List[Path]:
    """Return subject directories under src_split_dir."""
    if not src_split_dir.exists():
        return []
    return sorted([p for p in src_split_dir.iterdir() if p.is_dir()])


def subject_name_from_dir(d: Path) -> str:
    """Use the directory name as subject id."""
    return d.name


# ----------------------------
# Export logic
# ----------------------------
def export_subject(
    subj_dir: Path,
    subj_name: str,
    out_modalities_root: Optional[Path],
    out_fusion_root: Optional[Path],
    split: str,  # "train" | "val"
    image_size: Tuple[int, int],
    export_modalities: List[str],
    fusion_triplet: Optional[Tuple[str, str, str]],
    empty_thresh: float,
) -> int:
    """
    Load requested modalities for a subject and export slices.
    Returns number of images written.
    """
    required = set(export_modalities)
    if fusion_triplet:
        required.update(fusion_triplet)

    # Load volumes
    volumes: Dict[str, np.ndarray] = {}
    for m in required:
        f = find_modality_file(subj_dir, subj_name, m)
        if f is None:
            raise FileNotFoundError(
                f"Missing modality {m} for {subj_name} in {subj_dir}"
            )
        vol = load_modality(f)
        volumes[m] = vol

    # Slices in common
    def ns(a: np.ndarray) -> int:
        return a.shape[2] if is_3d(a) else 1

    max_slices = min(ns(v) for v in volumes.values())
    written = 0

    for s in range(max_slices):
        # normalize each modality slice to [0,1], skip empty
        norm_slices: Dict[str, Optional[np.ndarray]] = {}
        for m, vol in volumes.items():
            sl = vol[:, :, s] if is_3d(vol) else vol
            if is_empty_slice(sl, thresh=empty_thresh):
                norm_slices[m] = None
                continue
            norm_slices[m] = robust_clip_and_norm(sl)

        # Fusion export (RGB)
        if out_fusion_root is not None and fusion_triplet:
            r, g, b = fusion_triplet
            if (
                norm_slices.get(r) is not None
                and norm_slices.get(g) is not None
                and norm_slices.get(b) is not None
            ):
                R = to_uint8(norm_slices[r])
                G = to_uint8(norm_slices[g])
                B = to_uint8(norm_slices[b])
                rgb = np.dstack([R, G, B])
                rgb = resize_uint8(rgb, image_size)
                out_file = out_fusion_root / split / f"{subj_name}_s{str(s).zfill(3)}_FUSION.png"
                save_rgb(rgb, out_file)
                written += 1

        # Per-modality export (as 3-channel grayscale)
        if out_modalities_root is not None:
            for m in export_modalities:
                sl01 = norm_slices.get(m)
                if sl01 is None:
                    continue
                g = to_uint8(sl01)
                g = resize_uint8(g, image_size)
                rgb = np.dstack([g, g, g])
                out_file = out_modalities_root / split / m / f"{subj_name}_s{str(s).zfill(3)}_{m}.png"
                save_rgb(rgb, out_file)
                written += 1

    return written


def export_split(
    split_name: str,
    src_dir: Path,
    out_modalities_root: Optional[Path],
    out_fusion_root: Optional[Path],
    image_size: Tuple[int, int],
    export_modalities: List[str],
    fusion_triplet: Optional[Tuple[str, str, str]],
    empty_thresh: float,
) -> int:
    """Process one split directory (Train or Val)."""
    subj_dirs = list_subjects(src_dir)
    if not subj_dirs:
        print(f"[WARN] No subject folders found under {src_dir} (split={split_name})")
        return 0

    total = 0
    for d in tqdm(subj_dirs, desc=f"Exporting {split_name} subjects"):
        subj = subject_name_from_dir(d)
        total += export_subject(
            subj_dir=d,
            subj_name=subj,
            out_modalities_root=out_modalities_root,
            out_fusion_root=out_fusion_root,
            split=split_name,
            image_size=image_size,
            export_modalities=export_modalities,
            fusion_triplet=fusion_triplet,
            empty_thresh=empty_thresh,
        )
    return total


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export BraTS-style per-subject folders to per-modality and fusion PNG datasets."
    )
    p.add_argument(
        "--src-train",
        type=Path,
        required=True,
        help="Path to the 'Train' folder (e.g., brats20/Train).",
    )
    p.add_argument(
        "--src-val",
        type=Path,
        required=True,
        help="Path to the 'Val' folder (e.g., brats20/Val).",
    )
    p.add_argument(
        "--out-modalities",
        type=Path,
        required=True,
        help="Output root for per-modality PNGs.",
    )
    p.add_argument(
        "--out-fusion",
        type=Path,
        required=True,
        help="Output root for fusion PNGs.",
    )
    p.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(256, 256),
        metavar=("W", "H"),
        help="Output size, e.g., 256 256",
    )
    p.add_argument(
        "--export-modalities",
        nargs="*",
        default=["flair", "t1", "t1ce", "t2"],
        help="Modalities to export as grayscale→RGB copies.",
    )
    p.add_argument(
        "--fusion",
        type=str,
        default=None,
        help="3-modality fusion triplet, e.g., 't1ce,t2,flair' (R,G,B). Required for fusion output.",
    )
    p.add_argument(
        "--empty-thresh",
        type=float,
        default=0.005,
        help="Fraction of non-zero pixels below which a slice is considered empty.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Prepare outputs
    out_modalities_root = args.out_modalities
    out_fusion_root = args.out_fusion
    ensure_dir(out_modalities_root)
    ensure_dir(out_fusion_root)

    # Parse fusion
    fusion_triplet: Optional[Tuple[str, str, str]] = None
    if args.fusion:
        parts = [x.strip().lower() for x in args.fusion.split(",")]
        if len(parts) != 3:
            raise ValueError("--fusion expects exactly 3 modalities, e.g. 't1ce,t2,flair'")
        for p in parts:
            if p not in {"flair", "t1", "t1ce", "t2"}:
                raise ValueError(f"Unknown modality in fusion: {p}")
        fusion_triplet = (parts[0], parts[1], parts[2])
        print(f"[INFO] Fusion RGB = (R={fusion_triplet[0]}, G={fusion_triplet[1]}, B={fusion_triplet[2]})")
    else:
        print("[INFO] No fusion specified; fusion dataset will be empty unless --fusion is provided.")

    # Normalize modality list
    export_modalities = [m for m in args.export_modalities if m in {"flair", "t1", "t1ce", "t2"}]
    if not export_modalities and fusion_triplet is None:
        raise ValueError("Nothing to export. Provide --export-modalities and/or --fusion.")

    # Export both splits
    total_train = export_split(
        split_name="train",
        src_dir=args.src_train,
        out_modalities_root=out_modalities_root,
        out_fusion_root=out_fusion_root,
        image_size=(args.image_size[0], args.image_size[1]),
        export_modalities=export_modalities,
        fusion_triplet=fusion_triplet,
        empty_thresh=args.empty_thresh,
    )

    total_val = export_split(
        split_name="val",
        src_dir=args.src_val,
        out_modalities_root=out_modalities_root,
        out_fusion_root=out_fusion_root,
        image_size=(args.image_size[0], args.image_size[1]),
        export_modalities=export_modalities,
        fusion_triplet=fusion_triplet,
        empty_thresh=args.empty_thresh,
    )

    print(f"[DONE] Wrote {total_train + total_val} images total.")
    print(f"  Per-modality root: {out_modalities_root}")
    print(f"  Fusion root:       {out_fusion_root}")


if __name__ == "__main__":
    main()
