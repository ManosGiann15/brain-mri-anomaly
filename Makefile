.PHONY: install venv lint fmt test train eval app
.RECIPEPREFIX := >

PY ?= python
VENV_DIR := .venv

ifeq ($(OS),Windows_NT)
VENV_PY := $(VENV_DIR)/Scripts/python.exe
VENV_PIP := $(VENV_DIR)/Scripts/pip.exe
else
VENV_PY := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
endif

$(VENV_PY): requirements.txt
>$(PY) -m venv $(VENV_DIR)
>$(VENV_PY) -m pip install -U pip
>$(VENV_PIP) install -r requirements.txt
>$(VENV_PY) -m pre_commit install

venv: $(VENV_PY)

install:
>$(PY) -m pip install -U pip
>$(PY) -m pip install -r requirements.txt
>$(PY) -m pre_commit install

lint: $(VENV_PY)
>$(VENV_PY) -m ruff check src tests
>$(VENV_PY) -m black --check src tests
>$(VENV_PY) -m isort --check-only src tests
>$(VENV_PY) -m mypy src

fmt: $(VENV_PY)
>$(VENV_PY) -m ruff check --fix src tests
>$(VENV_PY) -m black src tests
>$(VENV_PY) -m isort src tests

test: $(VENV_PY)
>$(VENV_PY) -m pytest -q --maxfail=1 --disable-warnings

train: $(VENV_PY)
>$(VENV_PY) -m brain_mri_anomaly.train_ae --config configs/train_ae.yaml

eval: $(VENV_PY)
>$(VENV_PY) -m brain_mri_anomaly.eval_ae --config configs/train_ae.yaml

app: $(VENV_PY)
>$(VENV_PY) -m brain_mri_anomaly.app_gradio
