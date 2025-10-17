#!/usr/bin/env python3
from __future__ import annotations
import argparse
import inspect
import os
from pathlib import Path




from lightning.pytorch.callbacks import Callback
import torch
try:
    from torchvision import tv_tensors
except Exception:
    tv_tensors = None

def _to_plain_tensor(x):
    if tv_tensors is not None and isinstance(x, tv_tensors.Image):
        return x.as_subclass(torch.Tensor)
    return x

class CastInputsToHalf(Callback):
    def on_before_batch_transfer(self, trainer, pl_module, batch, dataloader_idx: int):
        # anomalib batches often have .image attribute
        if hasattr(batch, "image"):
            batch.image = _to_plain_tensor(batch.image).half()
        elif isinstance(batch, dict) and "image" in batch:
            batch["image"] = _to_plain_tensor(batch["image"]).half()
        return batch
    
    
from lightning.pytorch.callbacks import Callback
import torch, gc

class VRAMGuard(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (batch_idx + 1) % 500 == 0:
            print(f"[VRAMGuard] cleared cache at batch {batch_idx}")


# --------------------------- Windows symlink patch --------------------------- #
def _patch_symlink_for_windows():
    """On Windows without symlink privilege, attempt a junction (/J) for dirs
    or a hardlink (/H) for files; as last resort, copy. This intercepts calls
    from libraries (e.g., anomalib Engine) that try to create symlinks.
    """
    import os, shutil, subprocess

    if os.name != "nt":
        return  # Not Windows -> nothing to do

    _orig_symlink = getattr(os, "symlink", None)
    if _orig_symlink is None:
        # Very old Python on Windows may not have os.symlink
        def _dummy_symlink(src, dst, target_is_directory=False):
            _win_make_link_or_copy(src, dst, target_is_directory)
        os.symlink = _dummy_symlink  # type: ignore[attr-defined]
    else:
        def _safe_symlink(src, dst, target_is_directory=False):
            try:
                return _orig_symlink(src, dst, target_is_directory=target_is_directory)
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
    import shutil, subprocess
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


def str2bool(s: str) -> bool:
    return s.lower() in {"1", "true", "yes", "y", "t"}


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
    parser.add_argument("--precision", type=str, default="32-true")  # e.g. "16-mixed" for AMP

    # NEW: CUDA/CPU selection
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device. 'auto' picks CUDA if available, else CPU.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="1",
        help="Number of devices or a comma list of GPU indices. Examples: '1', '0', '0,1', 'auto'.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES mask, e.g. '0' or '0,1'.",
    )

    # NEW: predict flag (it was referenced but not defined)
    parser.add_argument("--predict", type=str2bool, default=False)

    # NEW: Lightning/Trainer tuning flags
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--enable-progress-bar", type=str2bool, default=True)
    parser.add_argument("--profiler", type=str, default=None)
    parser.add_argument("--persistent-workers", type=str2bool, default=False)

    return parser.parse_args()


def _resolve_accelerator_and_devices(device_arg: str, devices_arg: str):
    """
    Map CLI args to Lightning Trainer arguments (accelerator, devices).
    """
    cuda_ok = _torch_cuda_available()

    if device_arg == "auto":
        accelerator = "gpu" if cuda_ok else "cpu"
    elif device_arg == "cuda":
        accelerator = "gpu" if cuda_ok else "cpu"
        if device_arg == "cuda" and not cuda_ok:
            print("[WARN] --device=cuda requested but CUDA not available. Falling back to CPU.")
    else:
        accelerator = "cpu"

    # devices parsing:
    # - "auto": pass through
    # - "1": integer
    # - "0,1": list of indices (Lightning accepts "0,1" as str too)
    if devices_arg.strip().lower() == "auto":
        devices = "auto"
    else:
        if "," in devices_arg:
            # keep as string for Lightning ("0,1")
            devices = devices_arg
        else:
            # single integer as int
            try:
                devices = int(devices_arg)
            except ValueError:
                # fallback to string (e.g., "0")
                devices = devices_arg

    return accelerator, devices


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

    # GPU/CPU dataloader niceties
    if has("pin_memory"):
        kwargs["pin_memory"] = (_torch_cuda_available() and args.device != "cpu")
    if has("persistent_workers"):
        kwargs["persistent_workers"] = bool(args.persistent_workers) and args.num_workers > 0
    if has("prefetch_factor") and args.num_workers > 0:
        kwargs["prefetch_factor"] = 2

    # Explicit folders (v1.x style)
    if has("normal_dir"):
        kwargs["normal_dir"] = str(Path(args.data_root) / ".." / "train")
    if has("normal_test_dir"):
        kwargs["normal_test_dir"] = str(Path(args.data_root) / ".." / "val")

    print("[INFO] Building Folder datamodule with args:")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")

    return Folder(**kwargs)


def main():
    # Patch symlinks BEFORE importing/creating Engine so any internal symlink calls are intercepted
    _patch_symlink_for_windows()

    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Optional: set CUDA_VISIBLE_DEVICES mask
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # Resolve accelerator/devices for Lightning
    accelerator, devices = _resolve_accelerator_and_devices(args.device, args.devices)

    # Optional PyTorch CUDA niceties
    try:
        import torch
        if accelerator == "gpu" and torch.cuda.is_available():
            # Helps matmul kernels on Ampere+; safe on 11.x toolkits
            torch.set_float32_matmul_precision("high")
            print(f"[INFO] CUDA detected: {torch.version.cuda}, device_count={torch.cuda.device_count()}")
            print(f"[INFO] Using accelerator='{accelerator}', devices='{devices}', precision='{args.precision}'")
        else:
            print(f"[INFO] Using CPU. precision='{args.precision}'")
    except Exception:
        pass

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
        
        
        # Disable FAISS GPU usage (forces CPU)
        os.environ["ANOMALIB_FAISS_ON_GPU"] = "0"

        # Make sure PatchCore stores features on CPU
        if hasattr(model_v2, "model"):
            if hasattr(model_v2.model, "store_device"):
                model_v2.model.store_device = "cpu"
            if hasattr(model_v2.model, "nn_method"):
                model_v2.model.nn_method = {"name": "faiss", "on_gpu": False}


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
            callbacks=[VRAMGuard()],  
            # NEW: device selection
            accelerator=accelerator,   # "gpu" or "cpu"
            devices=devices,           # e.g. 1, "0", or "0,1", or "auto"
        )
        
        # Empty cache before training to reduce OOM issues
        if accelerator == "gpu":
            torch.cuda.empty_cache()
            torch.cuda.memory_summary(device=None, abbreviated=False)

        engine.fit(model=model_v2, datamodule=datamodule)

        if args.predict:
            engine.predict(datamodule=datamodule, model=model_v2)

        print("[OK] Trained with anomalib v2 Engine (CUDA/CPU selectable).")
        return

    except Exception as e:
        print(f"[INFO] Error Occurred during training: ({e})")
        raise


if __name__ == "__main__":
    main()
