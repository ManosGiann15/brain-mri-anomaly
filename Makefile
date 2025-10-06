.PHONY: help install venv reenv precommit lint fmt test train eval app clean
.RECIPEPREFIX := >

# ---- Config ---------------------------------------------------------------
PY ?= python
VENV_DIR := .venv

# Resolve venv executables per OS
ifeq ($(OS),Windows_NT)
VENV_PY := $(VENV_DIR)/Scripts/python.exe
else
VENV_PY := $(VENV_DIR)/bin/python
endif

# Dev tools you rely on in targets below
DEV_PKGS := pre-commit ruff black isort mypy pytest pytest-cov

# ---- Default --------------------------------------------------------------
help:
> @echo "Targets:"
> @echo "  make venv       -> create venv, install deps + dev tools, install git hooks"
> @echo "  make reenv      -> recreate venv from scratch"
> @echo "  make install    -> install deps into the ACTIVE Python (no venv implied)"
> @echo "  make precommit  -> install pre-commit and git hooks in the venv"
> @echo "  make lint       -> ruff/black/isort/mypy"
> @echo "  make fmt        -> ruff --fix + black + isort"
> @echo "  make test       -> pytest"
> @echo "  make train      -> run training"
> @echo "  make eval       -> run evaluation"
> @echo "  make app        -> launch Gradio app"
> @echo "  make clean      -> remove caches/build artifacts"

# ---- Bootstrap venv (idempotent) -----------------------------------------
$(VENV_PY): requirements.txt
> $(PY) -m venv $(VENV_DIR)
> $(VENV_PY) -m pip install -U pip setuptools wheel
> $(VENV_PY) -m pip install -r requirements.txt
> $(VENV_PY) -m pip install $(DEV_PKGS)
> -$(VENV_PY) -m pre_commit install

venv: $(VENV_PY)

# Force fresh venv (handy when env drifts)
reenv:
> @echo "Recreating virtual environment..."
> rmdir /S /Q $(VENV_DIR) 2> NUL || rm -rf $(VENV_DIR) || true
> $(MAKE) venv

# ---- Install into system / current interpreter (optional) -----------------
install:
> $(PY) -m pip install -U pip setuptools wheel
> $(PY) -m pip install -r requirements.txt
> $(PY) -m pip install $(DEV_PKGS)
> -$(PY) -m pre_commit install

# ---- Dev tooling ----------------------------------------------------------
precommit: venv
> $(VENV_PY) -m pip install -U pre-commit
> -$(VENV_PY) -m pre_commit install
> @echo "Pre-commit installed. Run: $(VENV_PY) -m pre_commit run --all-files"

lint: venv
> $(VENV_PY) -m ruff check src tests
> $(VENV_PY) -m black --check src tests
> $(VENV_PY) -m isort --check-only src tests
> $(VENV_PY) -m mypy src

fmt: venv
> $(VENV_PY) -m ruff check --fix src tests
> $(VENV_PY) -m black src tests
> $(VENV_PY) -m isort src tests

test: venv
> $(VENV_PY) -m pytest -q --maxfail=1 --disable-warnings

# ---- Project entrypoints --------------------------------------------------
train: venv
> $(VENV_PY) -m brain_mri_anomaly.train_ae --config configs/train_ae.yaml

eval: venv
> $(VENV_PY) -m brain_mri_anomaly.eval_ae --config configs/train_ae.yaml

app: venv
> $(VENV_PY) -m brain_mri_anomaly.app_gradio

# ---- Hygiene --------------------------------------------------------------
clean:
> @echo "Cleaning caches..."
> rmdir /S /Q .pytest_cache 2> NUL || rm -rf .pytest_cache || true
> rmdir /S /Q __pycache__ 2> NUL || rm -rf __pycache__ || true
> rmdir /S /Q .ruff_cache 2> NUL || rm -rf .ruff_cache || true
> rmdir /S /Q build 2> NUL || rm -rf build || true
> rmdir /S /Q dist 2> NUL || rm -rf dist || true
