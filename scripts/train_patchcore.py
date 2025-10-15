#!/usr/bin/env python3
from __future__ import annotations
import argparse
import inspect
from pathlib import Path

# --------------------------- Windows symlink patch --------------------------- #
def _patch_symlink_for_windows():
    """On Windows without symlink privilege, attempt a junction (/J) for dirs
    or a hardlink (/H) for files; as last resort, copy. This intercepts calls
    from libraries (e.g., anomalib Engine) that try to create symlinks.
    """
    import os, shutil, subprocess, sys

    if os.name != "nt":
        return  # Not Windows -> nothing to do

    _orig_symlink = getattr(os, "symlink", None)
    if _orig_symlink is None:
        # Very old Python on Windows may not have os.symlink
        def _dummy_symlink(src, dst, target_is_directory=False):
            # Fallback directly to copy/junction behavior
            _win_make_link_or_copy(src, dst, target_is_directory)
        os.symlink = _dummy_symlink  # type: ignore[attr-defined]
    else:
        def _safe_symlink(src, dst, target_is_directory=False):
            try:
                return _orig_symlink(src, dst, target_is_directory=target_is_directory)  # try normal symlink
            except OSError:
                _win_make_link_or_copy(src, dst, target_is_directory)

        os.symlink = _safe_symlink  # type: ignore[attr-defined]

    # Patch pathlib.Path.symlink_to to delegate to our os.symlink
    from pathlib import Path as _P

    def _safe_symlink_to(self, target, target_is_directory=False):
        return os.symlink(str(target), str(self), target_is_directory=target_is_directory)

    _P.symlink_to = _safe_symlink_to  # type: ignore[assignment]

def _win_make_link_or_copy(src, dst, target_is_directory):
    """Create junction/hardlink on Windows without elevation; fallback to copy."""
    import os, shutil, subprocess
    from pathlib import Path as _P

    src_p, dst_p = _P(src), _P(dst)

    # Remove existing dst if present
    if dst_p.exists() or dst_p.is_symlink():
        if dst_p.is_dir():
            shutil.rmtree(dst_p, ignore_errors=True)
        else:
            try:
                dst_p.unlink()
            except Exception:
                pass

    # Prefer junctions for directories (works without admin)
    try:
        if target_is_directory or src_p.is_dir():
            cmd = ["cmd", "/c", "mklink", "/J", str(dst_p), str(src_p)]
        else:
            # Hard link for files; if different volumes, this will fail and we’ll copy.
            cmd = ["cmd", "/c", "mklink", "/H", str(dst_p), str(src_p)]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return
    except Exception:
        # Fallback: copy
        if src_p.is_dir():
            shutil.copytree(src_p, dst_p)
        else:
            shutil.copy2(src_p, dst_p)
        return
# --------------------------------------------------------------------------- #


def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--layers", type=str, default="layer2,layer3")
    parser.add_argument("--coreset", type=float, default=0.10)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--precision", type=str, default="32-true")

    # NEW: Lightning/Trainer tuning flags (these were “unrecognized” before)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--enable-progress-bar", type=str2bool, default=True)
    parser.add_argument("--profiler", type=str, default=None)
    parser.add_argument("--persistent-workers", type=str2bool, default=False)

    args = parser.parse_args()

    return args


def build_folder_datamodule(args):
    """
    Builds a Folder datamodule compatible with both old and new anomalib versions.
    Older versions (<=1.1) need name + normal_dir, etc.
    """
    from anomalib.data import Folder
    W, H = args.image_size

    sig = inspect.signature(Folder.__init__)
    params = set(sig.parameters.keys())

    def has(x): return x in params

    kwargs = {}

    # Universal
    if has("name"):
        kwargs["name"] = "brain_mri_patchcore"
    if has("root"):
        kwargs["root"] = str(args.data_root)
    if has("image_size"):
        kwargs["image_size"] = (W, H)
    if has("train_batch_size"):
        kwargs["train_batch_size"] = args.batch_size
    if has("eval_batch_size"):
        kwargs["eval_batch_size"] = args.batch_size
    if has("num_workers"):
        kwargs["num_workers"] = args.num_workers

    # GPU/CPU
    if has("pin_memory"):
        kwargs["pin_memory"] = True
    if has("persistent_workers"):
        kwargs["persistent_workers"] = args.num_workers > 0
    if has("prefetch_factor") and args.num_workers > 0:
        kwargs["prefetch_factor"] = 2
        
    # Explicit folders (v1.x style)
    if has("normal_dir"):
        kwargs["normal_dir"] = str(Path(args.data_root) / ".." / "train" / "normal")
    if has("normal_test_dir"):
        kwargs["normal_test_dir"] = str(Path(args.data_root) / ".."  / "val" / "normal")

    print("[INFO] Building Folder datamodule with args:")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")

    return Folder(**kwargs)


def str2bool(s: str) -> bool:
    return s.lower() in {"1", "true", "yes", "y", "t"}


def main():
    # Patch symlinks BEFORE importing/creating Engine so any internal symlink calls are intercepted
    _patch_symlink_for_windows()

    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Build data module
    try:
        datamodule = build_folder_datamodule(args)
    except Exception as e:
        raise SystemExit(f"Failed to build Folder datamodule. Error: {e}")

    # Train with anomalib v2 Engine
    try:
        from anomalib.engine import Engine
        from anomalib.models import Patchcore

        model_v2 = Patchcore(
            backbone=args.backbone,
            layers=[s.strip() for s in args.layers.split(",") if s.strip()],
            coreset_sampling_ratio=args.coreset,
        )

        engine = Engine(
            # this becomes Trainer(**kwargs) behind the scenes
            default_root_dir=args.results_dir,
            max_epochs=args.max_epochs,
            precision=args.precision,
            enable_progress_bar=args.enable_progress_bar,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
            profiler=(None if (args.profiler in (None, "", "none")) else args.profiler),
            logger=False,  # optional: disable Lightning loggers
        )
        # You can pass precision/max_epochs via trainer kwargs if your anomalib supports it
        engine.fit(model=model_v2, datamodule=datamodule)
        
        if args.predict:
            engine.predict(datamodule=datamodule, model=model_v2)

        print("[OK] Trained with anomalib v2 Engine (symlink-safe on Windows).")
        return

    except Exception as e:
        print(f"[INFO] Error Occurred during training: ({e})")
        raise


if __name__ == "__main__":
    main()
