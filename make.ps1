param([string]$Task = "help")

function Install-Deps {
  python -m pip install -U pip
  pip install -r requirements.txt
  pre-commit install
}

switch ($Task) {
  "install" { Install-Deps }
  "lint"    { ruff check src tests; black --check src tests; isort --check-only src tests; mypy src }
  "fmt"     { ruff check --fix src tests; black src tests; isort src tests }
  "test"    { pytest -q --maxfail=1 --disable-warnings }
  "train"   { python -m brain_mri_anomaly.train_ae --config configs/train_ae.yaml }
  "eval"    { python -m brain_mri_anomaly.eval_ae --config configs/train_ae.yaml }
  "app"     { python -m brain_mri_anomaly.app_gradio }
  "data"    { python scripts/download_datasets.py --config configs/datasets.yaml }
  default   {
    Write-Host "Usage: .\make.ps1 <install|lint|fmt|test|train|eval|app|data>"
  }
}
