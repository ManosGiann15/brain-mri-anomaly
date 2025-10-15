#!/usr/bin/env python3
"""
Export BraTS-style per-subject folders to PatchCore/anomalib Folder dataset.

Input layout (your screenshot):
brats20/
  Train/
    BraTS20_Training_001/
      BraTS20_Training_001_flair.nii.gz
      BraTS20_Training_001_t1.nii.gz
      BraTS20_Training_001_t1ce.nii.gz
      BraTS20_Training_001_t2.nii.gz
      # (or .nii or .npy per modality)
    BraTS20_Training_002/
    ...

Output (anomalib Folder):
patchcore/
  train/
    normal/            *_FUSION.png  (if --fusion used)
    normal/flair/      *_flair.png
    normal/t1/         *_t1.png
    normal/t1ce/       *_t1ce.png
    normal/t2/         *_t2.png
  val/
    anomalous/ ...
  test/
    anomalous/ ...

Usage:
------
python scripts/export_patchcore_from_brats.py \
  --src-train brats20/Train \
  --out-root patchcore \
  --fusion t1ce,t2,flair \
  --export-modalities flair t1 t1ce t2 \
  --image-size 256 256 \
  --splits-csv config/splits.csv
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import nibabel as nib  # for .nii/.nii.gz

    _HAS_NIB = True
except Exception:
    _HAS_NIB = False

from tqdm import tqdm


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
        # Ensure (H, W, S)
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
    Image.fromarray(img3, mode="RGB").save(out_path)


# ----------------------------
# Splits
# ----------------------------
@dataclass
class SubjectMeta:
    split: str  # train | val | test
    label: str  # normal | anomalous


def load_splits_csv(csv_path: Optional[Path]) -> Dict[str, SubjectMeta]:
    """
    CSV columns: subject,split,label
    Example:
        BraTS20_Training_001,train,normal
        BraTS20_Training_002,val,anomalous
    If missing, default: train/normal.
    """
    mapping: Dict[str, SubjectMeta] = {}
    if csv_path is None or not csv_path.exists():
        return mapping
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj = row["subject"].strip()
            split = row["split"].strip().lower()
            label = row["label"].strip().lower()
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid split for {subj}: {split}")
            if label not in {"normal", "anomalous"}:
                raise ValueError(f"Invalid label for {subj}: {label}")
            mapping[subj] = SubjectMeta(split=split, label=label)
    return mapping


# ----------------------------
# Subject discovery (BraTS-style)
# ----------------------------
MOD_ALIASES = {
    "flair": ("flair",),
    "t1": ("t1",),
    "t1ce": ("t1ce", "t1ce"),  # keep as is (alias slot for safety)
    "t2": ("t2",),
}

NIFTI_SUFFIXES = (".nii", ".nii.gz", ".npy")


def find_modality_file(subj_dir: Path, subj_name: str, modality: str) -> Optional[Path]:
    """
    Find file inside subject folder that matches the modality.
    Tries:
      {subj_name}_{modality} with .nii/.nii.gz/.npy
      '*{modality}*.nii*' as fallback
    """
    # exact pattern first
    for suf in NIFTI_SUFFIXES:
        cand = subj_dir / f"{subj_name}_{modality}{suf}"
        if cand.exists():
            return cand

    # fallback: any file containing modality token
    tokens = MOD_ALIASES.get(modality, (modality,))
    for tok in tokens:
        hits = sorted(subj_dir.glob(f"*{tok}*"))
        for h in hits:
            if h.suffix in NIFTI_SUFFIXES or "".join(h.suffixes) in [".nii.gz"]:
                return h
    return None


def list_subjects(src_train: Path) -> List[Path]:
    """Return subject directories under src_train (BraTS20_Training_XXX)."""
    return sorted([p for p in src_train.iterdir() if p.is_dir()])


def subject_name_from_dir(d: Path) -> str:
    """Use the directory name as subject id."""
    return d.name


# ----------------------------
# Export logic
# ----------------------------
def make_out_dir(
    out_root: Path, split: str, label: str, subfolder: Optional[str] = None
) -> Path:
    base = out_root / split / label
    if subfolder:
        base = base / subfolder
    ensure_dir(base)
    return base


def export_subject(
    subj_dir: Path,
    subj_name: str,
    out_root: Path,
    meta: SubjectMeta,
    image_size: Tuple[int, int],
    export_modalities: List[str],
    fusion_triplet: Optional[Tuple[str, str, str]],
    empty_thresh: float,
) -> int:
    """
    Load requested modalities for a subject and export slices.
    Returns number of images written.
    """
    # Load volumes for required modalities
    required = set(export_modalities)
    if fusion_triplet:
        required.update(fusion_triplet)

    volumes: Dict[str, np.ndarray] = {}
    for m in required:
        f = find_modality_file(subj_dir, subj_name, m)
        if f is None:
            raise FileNotFoundError(
                f"Missing modality {m} for {subj_name} in {subj_dir}"
            )
        vol = load_modality(f)
        volumes[m] = vol

    # Determine number of slices available in common
    def ns(a: np.ndarray) -> int:
        return a.shape[2] if is_3d(a) else 1

    max_slices = min(ns(v) for v in volumes.values())
    written = 0

    for s in range(max_slices):
        # normalize each modality slice to [0,1], skip empty
        norm_slices: Dict[str, Optional[np.ndarray]] = {}
        empty_any = False
        for m, vol in volumes.items():
            sl = vol[:, :, s] if is_3d(vol) else vol
            if is_empty_slice(sl, thresh=empty_thresh):
                norm_slices[m] = None
                continue
            norm_slices[m] = robust_clip_and_norm(sl)

        # Fusion export
        if fusion_triplet:
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
                odir = make_out_dir(out_root, meta.split, meta.label)
                oname = f"{subj_name}_s{str(s).zfill(3)}_FUSION.png"
                save_rgb(rgb, odir / oname)
                written += 1

        # Per-modality export (as 3-channel grayscale)
        for m in export_modalities:
            sl01 = norm_slices.get(m)
            if sl01 is None:
                continue
            g = to_uint8(sl01)
            g = resize_uint8(g, image_size)
            rgb = np.dstack([g, g, g])
            odir = make_out_dir(out_root, meta.split, meta.label, subfolder=m)
            oname = f"{subj_name}_s{str(s).zfill(3)}_{m}.png"
            save_rgb(rgb, odir / oname)
            written += 1

    return written


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export BraTS-style per-subject folders to PatchCore/anomalib Folder."
    )
    p.add_argument(
        "--src-train",
        type=Path,
        required=True,
        help="Path to the 'Train' folder (e.g., brats20/Train).",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root for PatchCore Folder layout (e.g., patchcore).",
    )
    p.add_argument(
        "--splits-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns: subject,split,label.",
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
        help="3-modality fusion triplet, e.g., 't1ce,t2,flair' (R,G,B).",
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

    src_train: Path = args.src_train
    out_root: Path = args.out_root
    ensure_dir(out_root)

    # Fusion parsing
    fusion_triplet: Optional[Tuple[str, str, str]] = None
    if args.fusion:
        parts = [x.strip().lower() for x in args.fusion.split(",")]
        if len(parts) != 3:
            raise ValueError(
                "--fusion expects exactly 3 modalities, e.g. 't1ce,t2,flair'"
            )
        for p in parts:
            if p not in {"flair", "t1", "t1ce", "t2"}:
                raise ValueError(f"Unknown modality in fusion: {p}")
        fusion_triplet = (parts[0], parts[1], parts[2])
        print(
            f"[INFO] Fusion RGB = (R={fusion_triplet[0]}, G={fusion_triplet[1]}, B={fusion_triplet[2]})"
        )

    # Discover subjects
    subj_dirs = list_subjects(src_train)
    if not subj_dirs:
        raise RuntimeError(f"No subject folders found under {src_train}")

    # Load splits map
    split_map = load_splits_csv(args.splits_csv)
    default_meta = SubjectMeta(split="train", label="normal")

    total = 0
    for d in tqdm(subj_dirs, desc="Exporting subjects"):
        subj = subject_name_from_dir(d)
        meta = split_map.get(subj, default_meta)
        total += export_subject(
            subj_dir=d,
            subj_name=subj,
            out_root=out_root,
            meta=meta,
            image_size=(args.image_size[0], args.image_size[1]),
            export_modalities=[
                m for m in args.export_modalities if m in {"flair", "t1", "t1ce", "t2"}
            ],
            fusion_triplet=fusion_triplet,
            empty_thresh=args.empty_thresh,
        )

    print(f"[DONE] Wrote {total} images to {out_root}")


if __name__ == "__main__":
    main()
