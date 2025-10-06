#!/usr/bin/env python3
import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "Missing dependency: pyyaml. Install with: pip install pyyaml", file=sys.stderr
    )
    sys.exit(1)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CFG = ROOT / "config" / "datasets.yaml"


def run(cmd, check=True):
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    ret = proc.wait()
    if check and ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)
    return ret


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def file_checksum(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: Path, checksum_spec: str) -> bool:
    """
    checksum_spec looks like: 'sha256:abcdef...' or 'md5:1234...'
    """
    if not checksum_spec:
        return True
    try:
        algo, expected = checksum_spec.split(":", 1)
    except ValueError:
        print(f"[WARN] Invalid checksum format: {checksum_spec}")
        return True
    algo = algo.lower().strip()
    expected = expected.strip()
    if not expected:
        return True
    print(f"[INFO] Verifying {algo} checksum for {path.name} …")
    actual = file_checksum(path, algo)
    ok = actual == expected
    print(
        f"[INFO] Expected: {expected}\n[INFO] Actual  : {actual}\n[INFO] Match   : {ok}"
    )
    return ok


def extract_archive(archive_path: Path, dest_dir: Path):
    suffixes = "".join(archive_path.suffixes)
    print(f"[INFO] Extracting {archive_path.name} -> {dest_dir} (type: {suffixes})")

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif (
        suffixes.endswith(".tar.gz")
        or suffixes.endswith(".tgz")
        or suffixes.endswith(".tar")
    ):
        mode = "r:gz" if ".gz" in suffixes or ".tgz" in suffixes else "r:"
        with tarfile.open(archive_path, mode) as tf:
            tf.extractall(dest_dir)
    else:
        print(
            f"[WARN] Unknown archive type for {archive_path.name}. Skipping extraction."
        )


def have_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def ensure_tools():
    missing = []
    # curl or wget, at least one
    if not (have_cmd("curl") or have_cmd("wget")):
        missing.append("curl or wget")
    return missing


def download_kaggle(ref: str, outpath: Path):
    if not have_cmd("kaggle"):
        print("[ERROR] Kaggle CLI not found. Install with: pip install kaggle")
        sys.exit(2)
    # verify kaggle.json exists
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(
            "[ERROR] ~/.kaggle/kaggle.json not found. Put your Kaggle API key there and chmod 600."
        )
        sys.exit(2)

    # kaggle datasets download -d owner/dataset -p <dir> -f <filename> (we’ll just save as outpath)
    outdir = outpath.parent
    ensure_dir(outdir)
    # We cannot force Kaggle CLI to write to a specific filename; it uses dataset name.
    # So we download to the directory, then rename to desired archive name if needed.
    cmd = ["kaggle", "datasets", "download", "-d", ref, "-p", str(outdir)]
    run(cmd)

    # find the most recent zip in outdir that matches ref last token
    last = ref.split("/")[-1]
    candidates = list(outdir.glob(f"{last}*.zip"))
    if not candidates:
        # sometimes datasets produce arbitrary filename; fallback: most recent zip
        candidates = sorted(
            outdir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True
        )

    if not candidates:
        print(f"[ERROR] Could not find a downloaded zip for {ref} in {outdir}")
        sys.exit(2)

    downloaded = candidates[0]
    if downloaded != outpath:
        if outpath.exists():
            outpath.unlink()
        downloaded.rename(outpath)
    print(f"[INFO] Saved Kaggle archive to {outpath}")


def download_url(url: str, outpath: Path):
    ensure_dir(outpath.parent)
    if have_cmd("curl"):
        run(["curl", "-L", "-o", str(outpath), url])
    elif have_cmd("wget"):
        run(["wget", "-O", str(outpath), url])
    else:
        print("[ERROR] Need curl or wget to download URLs.")
        sys.exit(2)


def download_gdrive(file_id: str, outpath: Path):
    if not have_cmd("gdown"):
        print("[ERROR] gdown not found. Install with: pip install gdown")
        sys.exit(2)
    ensure_dir(outpath.parent)
    run(["gdown", "--id", file_id, "--output", str(outpath)])


def dataset_done(dest: Path) -> bool:
    # Heuristic: destination exists and is non-empty
    return dest.exists() and any(dest.iterdir())


def main():
    ap = argparse.ArgumentParser(
        description="Download datasets defined in a YAML config."
    )
    ap.add_argument(
        "--config", type=Path, default=DEFAULT_CFG, help="Path to datasets.yaml"
    )
    ap.add_argument(
        "--force", action="store_true", help="Re-download even if dest exists"
    )
    args = ap.parse_args()

    tools_missing = ensure_tools()
    if tools_missing:
        print(f"[ERROR] Missing tools: {', '.join(tools_missing)}")
        sys.exit(2)

    if not args.config.exists():
        print(f"[ERROR] Config not found: {args.config}")
        sys.exit(1)

    with args.config.open("r") as f:
        cfg = yaml.safe_load(f)

    datasets = cfg.get("datasets", [])
    if not datasets:
        print("[WARN] No datasets defined.")
        return

    for ds in datasets:
        name = ds.get("name", "unnamed")
        dtype = ds.get("type")
        dest = ROOT / ds.get("dest", f"data/{name}")
        archive_name = ds.get("archive", f"{name}.zip")
        archive_path = ROOT / "data" / "_archives" / archive_name
        extract = bool(ds.get("extract", True))
        checksum = ds.get("checksum", "")

        print(f"\n=== {name} ({dtype}) ===")
        print(f"dest={dest} | archive={archive_path}")

        if dataset_done(dest) and not args.force:
            print(
                "[INFO] Destination already exists and is non-empty. Skipping. Use --force to re-download."
            )
            continue

        # ensure archive dir
        ensure_dir(archive_path.parent)
        # download
        if dtype == "kaggle":
            ref = ds["ref"]
            download_kaggle(ref, archive_path)
        elif dtype == "url":
            url = ds["url"]
            download_url(url, archive_path)
        elif dtype == "gdrive":
            fid = ds["gdrive_id"]
            download_gdrive(fid, archive_path)
        else:
            print(f"[WARN] Unknown dataset type: {dtype}. Skipping.")
            continue

        # checksum
        if checksum:
            if not verify_checksum(archive_path, checksum):
                print(
                    f"[ERROR] Checksum failed for {archive_path.name}. Delete the file and retry."
                )
                sys.exit(2)

        # prepare dest
        ensure_dir(dest)
        # extract
        if extract:
            extract_archive(archive_path, dest)
            print(f"[INFO] Extracted to {dest}")
        else:
            # If not extracting, at least move archive next to dest (optional)
            pass

    print("\n[OK] All datasets processed.")


if __name__ == "__main__":
    main()
