from pathlib import Path
import numpy as np
import nibabel as nib
import tqdm

# ------------------------------------------------------------------------------
# 1. Load volume safely (float32, minimal memory)
# ------------------------------------------------------------------------------

def load_volume(path: Path) -> np.ndarray:
    """
    Load a NIfTI MRI volume as a float32 NumPy array.
    Using float32 halves memory footprint compared to float64.
    """
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32)


# ------------------------------------------------------------------------------
# 2. Mask creation
# ------------------------------------------------------------------------------

def create_brain_mask(volume: np.ndarray) -> np.ndarray:
    """
    Generate a binary brain mask where non-zero voxels correspond to tissue.
    """
    brain_mask = volume > 0
    return brain_mask.astype(np.uint8)


# ------------------------------------------------------------------------------
# 3. Safe mean/std computation to avoid TB allocations
# ------------------------------------------------------------------------------

def safe_mean_std(volume: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """
    Compute mean and std of masked voxels without flattening the entire array.
    Works slice-by-slice to prevent large memory allocations.
    """
    total_sum = 0.0
    total_sq = 0.0
    total_count = 0

    # iterate over slices along last axis (axial view)
    for i in range(volume.shape[2]):
        v_slice = volume[:, :, i]
        m_slice = mask[:, :, i].astype(bool)

        if np.any(m_slice):
            vals = v_slice[m_slice]
            total_sum += float(vals.sum())
            total_sq += float(np.square(vals).sum())
            total_count += vals.size

    mean = total_sum / total_count if total_count > 0 else 0.0
    var = (total_sq / total_count - mean**2) if total_count > 0 else 0.0
    std = np.sqrt(var)
    return mean, std


# ------------------------------------------------------------------------------
# 4. Volume normalization (z-score or min-max)
# ------------------------------------------------------------------------------

def normalize_volume(volume: np.ndarray, mask: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize intensity values within the brain region.
    Avoids huge temporary arrays by using slice-wise computation.
    """
    if method == "zscore":
        mean, std = safe_mean_std(volume, mask)
        if std > 0:
            volume = (volume - mean) / std

    elif method == "minmax":
        # compute min/max only within mask safely
        masked_vals = volume[mask.astype(bool)]
        min_val = masked_vals.min()
        max_val = masked_vals.max()
        if max_val > min_val:
            volume = (volume - min_val) / (max_val - min_val)

    return volume


# ------------------------------------------------------------------------------
# 5. Slice extraction
# ------------------------------------------------------------------------------

def extract_slices(volume: np.ndarray, mask: np.ndarray, axis: int = 2) -> list[np.ndarray]:
    """
    Extract 2D slices along the chosen axis, discarding empty ones.
    """
    slices = []
    for i in range(volume.shape[axis]):
        if axis == 0:
            slice_ = volume[i, :, :]
            mask_slice = mask[i, :, :]
        elif axis == 1:
            slice_ = volume[:, i, :]
            mask_slice = mask[:, i, :]
        else:
            slice_ = volume[:, :, i]
            mask_slice = mask[:, :, i]

        if np.any(mask_slice):
            slices.append(slice_)
    return slices


# ------------------------------------------------------------------------------
# 6. Full preprocessing pipeline
# ------------------------------------------------------------------------------

def preprocess_patient(patient_dir: Path, modalities: list[str], save_dir: Path, force: bool = False) -> None:
    """
    Full preprocessing for one patient:
      load → mask → normalize → slice → save.

    If `force=False`, skip any modality that already has a .done marker file.
    """
    patient_id = patient_dir.name
    patient_save_dir = save_dir / patient_id
    patient_save_dir.mkdir(parents=True, exist_ok=True)

    for mod in modalities:
        mod_path = patient_dir / f"{patient_id}_{mod}.nii"
        if not mod_path.exists():
            print(f"[WARN] Missing modality: {mod_path}")
            continue

        mod_save_dir = patient_save_dir / mod
        done_marker = mod_save_dir / ".done"

        # --- Skip if already processed ---
        if not force and done_marker.exists():
            print(f"[SKIP] {patient_id}/{mod} already processed (found {done_marker.name})")
            continue

        # Ensure output dir exists
        mod_save_dir.mkdir(parents=True, exist_ok=True)

        # Load & process
        volume = load_volume(mod_path)
        mask = create_brain_mask(volume)
        norm_volume = normalize_volume(volume, mask, method="zscore")
        slices = extract_slices(norm_volume, mask, axis=2)

        # Save slices; skip ones that already exist
        saved = 0
        for idx, slice_ in enumerate(slices):
            slice_path = mod_save_dir / f"slice_{idx:03d}.npy"
            if slice_path.exists():
                continue
            np.save(slice_path, slice_)
            saved += 1

        # Write a marker only if at least one slice exists in folder
        # (in case everything already existed, we still confirm completion)
        if any(mod_save_dir.glob("slice_*.npy")):
            done_marker.touch()

        # Free memory
        del volume, norm_volume, mask, slices

        print(f"[OK] {patient_id}/{mod}: saved {saved} new slices" + ("" if saved else " (all existed)"))


if __name__ == "__main__":
    RAW_DIR = Path("../../data/brats20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
    PROCESSED_DIR = Path("../../data/processed")
    MODALITIES = ["t1", "t1ce", "t2", "flair"]

    FORCE = False  # set True to reprocess even if .done exists

    patient_folders = [p for p in RAW_DIR.iterdir() if p.is_dir()]
    for patient_folder in tqdm.tqdm(patient_folders, desc="Processing patients", total=len(patient_folders)):
        preprocess_patient(patient_folder, MODALITIES, PROCESSED_DIR, force=FORCE)
