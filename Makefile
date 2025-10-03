.PHONY: install venv lint fmt test train eval app

PY=python

venv:
	$(PY) -m venv .venv
	. .venv/bin/activate; pip install -U pip; pip install -r requirements.txt
	. .venv/bin/activate; pre-commit install

install:
	pip install -U pip
	pip install -r requirements.txt
	pre-commit install

lint:
	ruff check src tests
	black --check src tests
	isort --check-only src tests
	mypy src

fmt:
	ruff check --fix src tests
	black src tests
	isort src tests

test:
	pytest -q --maxfail=1 --disable-warnings

train:
	$(PY) -m brain_mri_anomaly.train_ae --config configs/train_ae.yaml

eval:
	$(PY) -m brain_mri_anomaly.eval_ae --config configs/train_ae.yaml

app:
	$(PY) -m brain_mri_anomaly.app_gradio
