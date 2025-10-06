"""Utilities for downloading datasets declared in a YAML configuration file.

This module focuses on Kaggle datasets because they require a couple of extra
steps (API credentials, handling large downloads, etc.) and the default Kaggle
CLI can hang on Windows when it prompts for interactive confirmation. Using the
Python API avoids that prompt and keeps the workflow identical across
platforms.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from kaggle.api.kaggle_api_extended import KaggleApi

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Dataset:
    """Configuration for a single dataset entry."""

    name: str
    provider: str
    source: str
    destination: Path
    archive: Path
    unzip: bool = True
    force: bool = False

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, object]) -> "Dataset":
        """Create a :class:`Dataset` from a configuration mapping."""

        try:
            provider = str(data["provider"])
            source = str(data["source"])
            destination = Path(data["destination"])
            archive = Path(data["archive"])
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise ValueError(f"Dataset '{name}' is missing required key: {exc.args[0]}") from exc

        unzip = bool(data.get("unzip", True))
        force = bool(data.get("force", False))
        return cls(
            name=name,
            provider=provider,
            source=source,
            destination=destination,
            archive=archive,
            unzip=unzip,
            force=force,
        )


def load_datasets(config_path: Path) -> list[Dataset]:
    """Load dataset declarations from ``config_path``."""

    if not config_path.exists():
        raise FileNotFoundError(f"Dataset configuration file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_datasets = config.get("datasets")
    if not isinstance(raw_datasets, dict) or not raw_datasets:
        raise ValueError(
            "The dataset configuration must define a non-empty 'datasets' mapping."
        )

    datasets: list[Dataset] = []
    for name, data in raw_datasets.items():
        if not isinstance(data, dict):
            raise ValueError(f"Dataset '{name}' must be defined using a mapping")
        datasets.append(Dataset.from_mapping(name, data))
    return datasets


class KaggleDownloader:
    """Download helper that wraps :class:`KaggleApi`."""

    def __init__(self) -> None:
        self._api = KaggleApi()
        try:
            self._api.authenticate()
        except Exception as exc:  # pragma: no cover - depends on local setup
            raise RuntimeError(
                "Unable to authenticate with Kaggle. Ensure that you have created a kaggle.json "
                "API token and placed it under ~/.kaggle/kaggle.json (Linux/macOS) or "
                "%USERPROFILE%/.kaggle/kaggle.json (Windows)."
            ) from exc

    def download(self, dataset: Dataset, *, force: bool) -> None:
        """Download ``dataset`` to the configured locations."""

        if dataset.provider != "kaggle":
            raise ValueError(f"Unsupported provider '{dataset.provider}' for dataset {dataset.name}")

        destination = dataset.destination
        archive_path = dataset.archive
        archive_dir = archive_path.parent
        archive_dir.mkdir(parents=True, exist_ok=True)

        destination.mkdir(parents=True, exist_ok=True)

        # Kaggle saves the zip file using the dataset slug. We rename it to the configured path.
        slug = dataset.source.rsplit("/", maxsplit=1)[-1]
        downloaded_archive = archive_dir / f"{slug}.zip"

        if archive_path.exists() and archive_path != downloaded_archive and force:
            LOGGER.debug("Removing stale archive %s", archive_path)
            archive_path.unlink()

        should_download = force or not archive_path.exists()
        if should_download:
            LOGGER.info("Downloading %s from Kaggle ...", dataset.name)
            self._api.dataset_download_files(
                dataset.source,
                path=str(archive_dir),
                force=True,
                quiet=False,
                unzip=False,
            )
            if not downloaded_archive.exists():  # pragma: no cover - sanity guard
                raise RuntimeError(
                    f"Kaggle download for {dataset.name} did not produce {downloaded_archive}" 
                )
            if downloaded_archive != archive_path:
                downloaded_archive.replace(archive_path)
        else:
            LOGGER.info("Skipping download for %s; archive already present.", dataset.name)

        if dataset.unzip:
            if not archive_path.exists():  # pragma: no cover - sanity check
                raise FileNotFoundError(
                    f"Archive for {dataset.name} not found at {archive_path}."
                )
            destination_contents = list(destination.iterdir())
            if force and destination_contents:
                LOGGER.debug("Clearing destination directory %s", destination)
                shutil.rmtree(destination)
                destination.mkdir(parents=True, exist_ok=True)
                destination_contents = []

            if destination_contents:
                LOGGER.info(
                    "Skipping extraction for %s; destination already contains files.",
                    dataset.name,
                )
            else:
                LOGGER.info("Extracting %s to %s", dataset.name, destination)
                shutil.unpack_archive(str(archive_path), str(destination))

        LOGGER.info("Finished processing %s", dataset.name)


def ensure_supported_providers(datasets: Iterable[Dataset]) -> None:
    """Validate that every dataset uses a supported provider."""

    for dataset in datasets:
        if dataset.provider not in {"kaggle"}:
            raise ValueError(f"Unsupported provider '{dataset.provider}' for dataset {dataset.name}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets.yaml"),
        help="Path to the dataset configuration file (default: configs/datasets.yaml).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional list of dataset names to download (defaults to all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-downloading archives and re-extracting datasets.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON summary of the processed datasets.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity (only warnings and errors).",
    )
    return parser.parse_args(argv)


def configure_logging(*, quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    configure_logging(quiet=args.quiet)

    try:
        datasets = load_datasets(args.config)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1

    ensure_supported_providers(datasets)

    if args.only:
        requested = set(args.only)
        datasets = [dataset for dataset in datasets if dataset.name in requested]
        missing = requested.difference({dataset.name for dataset in datasets})
        if missing:
            LOGGER.error("Unknown dataset(s): %s", ", ".join(sorted(missing)))
            return 1

    downloader = KaggleDownloader()

    processed: list[dict[str, str]] = []
    for dataset in datasets:
        LOGGER.info("=== %s ===", dataset.name)
        downloader.download(dataset, force=args.force or dataset.force)
        processed.append(
            {
                "name": dataset.name,
                "provider": dataset.provider,
                "source": dataset.source,
                "destination": str(dataset.destination),
                "archive": str(dataset.archive),
            }
        )

    if args.json:
        print(json.dumps({"processed": processed}, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
