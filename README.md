# Brain MRI Anomaly Detection

This repository provides utilities for training and evaluating autoencoder models for brain MRI anomaly detection.

## Prerequisites

- Python 3.9 or later
- [pip](https://pip.pypa.io/)
- [GNU Make](https://www.gnu.org/software/make/)
- (Optional) [PowerShell](https://learn.microsoft.com/powershell/) for running `make.ps1`

## Environment setup

The project uses a local virtual environment stored in `.venv`. The Makefile now works on both Linux/macOS and Windows. Every task that installs dependencies automatically creates the appropriate virtual environment layout (`.venv/bin` on Unix and `.venv/Scripts` on Windows).

### 1. Create the virtual environment

```bash
make venv
```

This command will:

1. Create a virtual environment under `.venv`.
2. Upgrade `pip` inside the environment.
3. Install Python dependencies from `requirements.txt`.
4. Install the `pre-commit` hooks into the repository.

Once `make venv` has been run you can optionally activate the environment:

- **Linux/macOS**
  ```bash
  source .venv/bin/activate
  ```
- **Windows (PowerShell or Command Prompt)**
  ```powershell
  .\.venv\Scripts\activate
  ```

> **Tip:** If you are using Windows, install GNU Make through [chocolatey](https://community.chocolatey.org/packages/make) (`choco install make`) or [scoop](https://scoop.sh/) (`scoop install make`) and run the `make` commands from PowerShell, Command Prompt, or Git Bash.

### 2. Common development commands

All developer workflows run the tools inside the virtual environment created above:

```bash
make lint   # Run ruff, black, isort, and mypy
make fmt    # Format the codebase using ruff, black, and isort
make test   # Execute pytest
make train  # Train the autoencoder with configs/train_ae.yaml
make eval   # Evaluate the trained autoencoder
make app    # Launch the Gradio demo application
```

### Alternative: PowerShell helper

If you prefer not to install GNU Make on Windows, you can run the equivalent tasks through the provided PowerShell script:

```powershell
.\make.ps1 install
.\make.ps1 lint
```

The PowerShell script assumes you have already created and activated a virtual environment (for example by running `python -m venv .venv` followed by `.\.venv\Scripts\activate`).

## Downloading datasets

The project ships with a helper that uses the Kaggle Python API (instead of the CLI) so it works the same on Linux, macOS, and Windows without getting stuck on interactive prompts.

1. Create a Kaggle API token by visiting <https://www.kaggle.com/settings/account> and clicking **Create New Token**. Save the downloaded `kaggle.json` to:
   - **Linux/macOS:** `~/.kaggle/kaggle.json`
   - **Windows:** `%USERPROFILE%\.kaggle\kaggle.json`
2. Run the dataset download task:
   ```bash
   make data  # or: python scripts/download_datasets.py --config configs/datasets.yaml
   ```
   The helper reads `configs/datasets.yaml`, downloads each archive into `data/_archives/`, and then extracts it into `data/`.
3. If you only need specific datasets, pass their names:
   ```bash
   python scripts/download_datasets.py --only brats20
   ```
4. Use `--force` to re-download and re-extract a dataset if the local copy becomes corrupted.

## Additional notes

- `make install` installs the project dependencies into whichever Python environment is currently active. This is useful if you manage environments manually.
- When running `make` targets, you can override the Python interpreter that is used to create the virtual environment by providing the `PY` variable, e.g. `make PY=python3.11 venv`.
