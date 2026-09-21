# Ally — task contract (CLAUDE.md "Commands"). Works from Git Bash, or via .\tasks.ps1 on PowerShell.
# Python lives in a 3.11 venv; override with: make test PY=python
PY          ?= .venv/Scripts/python.exe
UV          ?= uv
# CUDA wheel index for torch. Confirm the tag on https://pytorch.org/get-started/locally/ before first install.
TORCH_INDEX ?= https://download.pytorch.org/whl/cu126
SESSION     ?=

.PHONY: help venv install install-dev lock test lint fmt replay scenarios run bench \
        train-fall eval-fall train-fer eval-fer precommit clean

help:
	@echo "targets: venv install install-dev lock test lint fmt replay SESSION=<name> scenarios run bench"
	@echo "         train-fall eval-fall train-fer eval-fer precommit clean"

# ---- environment -----------------------------------------------------------
venv:            ## create .venv on Python 3.11 with uv (no admin needed)
	$(UV) python install 3.11
	$(UV) venv --python 3.11 .venv

lock:            ## freeze requirements*.in -> requirements*.txt (exact pins, 3.11, all platforms: CI is Linux)
	$(UV) pip compile requirements-dev.in -o requirements-dev.txt --python-version 3.11 --universal
	$(UV) pip compile requirements.in -o requirements.txt --python-version 3.11 --universal \
	    --extra-index-url $(TORCH_INDEX) --index-strategy unsafe-best-match

install-dev:     ## deps for unit/replay tests + lint (CPU only)
	$(UV) pip install --python $(PY) -r $(if $(wildcard requirements-dev.txt),requirements-dev.txt,requirements-dev.in)
	$(UV) pip install --python $(PY) -e . --no-deps

install: install-dev  ## full runtime incl. CUDA torch (large download)
	$(UV) pip install --python $(PY) -r $(if $(wildcard requirements.txt),requirements.txt,requirements.in) \
	    --extra-index-url $(TORCH_INDEX) --index-strategy unsafe-best-match

precommit:
	$(PY) -m pre_commit install
	$(PY) -m pre_commit run --all-files

# ---- quality gates ---------------------------------------------------------
test:
	$(PY) -m pytest tests/unit tests/replay

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

# ---- pipeline --------------------------------------------------------------
replay:          ## make replay SESSION=<name>  (tests/replay/<name>/ or data/sessions/<name>/)
	@test -n "$(SESSION)" || (echo "usage: make replay SESSION=<name>" && exit 1)
	$(PY) -m ally.main --replay $(SESSION)

scenarios:       ## 30 scripted dialogue scenarios (needs ANTHROPIC_API_KEY)
	$(PY) scripts/run_scenarios.py --repeat 1

run:
	$(PY) -m ally.main

bench:           ## FPS / VRAM profile of the pose model on the webcam
	$(PY) scripts/benchmark_fps.py

# ---- experiments (fixed seeds, write to results/) --------------------------
train-fall:
	$(PY) scripts/train_fall.py
eval-fall:
	$(PY) scripts/eval_fall.py
train-fer:
	$(PY) scripts/train_fer.py
eval-fer:
	$(PY) scripts/eval_fer.py

clean:
	$(PY) -c "import shutil,glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True)+['.pytest_cache','.ruff_cache']]"
