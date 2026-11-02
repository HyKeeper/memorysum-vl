.PHONY: help install test lint fmt run models docker clean

PY := .venv/bin/python
PIP := .venv/bin/pip

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## create .venv and install the package with dev extras
	python3 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -e '.[dev]'

test:  ## run the end-to-end tests (no GPU, no network)
	.venv/bin/pytest -q

lint:  ## ruff check
	.venv/bin/ruff check .

fmt:  ## ruff format
	.venv/bin/ruff format .

run:  ## run the API in the foreground on :8003
	.venv/bin/uvicorn memorysum_vl.main:app --app-dir src --host 127.0.0.1 --port 8003

models:  ## download the three GGUFs into $$MODEL_DIR (default ~/models)
	./deploy/llama/download-models.sh

docker:  ## build the API image
	docker build -f deploy/Dockerfile -t memorysum-vl:latest .

clean:  ## remove caches (keeps .venv and data/)
	rm -rf .pytest_cache .ruff_cache build dist
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
