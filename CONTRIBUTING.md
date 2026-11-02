# Contributing

Thanks for looking. This project is small on purpose: one process, one SQLite
file, three model servers. Please keep it that way.

## Getting set up

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env
```

`make test` runs the full suite — **it never calls a model**. `tests/conftest.py`
provides in-process fakes for the captioner/embedder/reranker, so the whole
Add→Search path (including degradation) runs in well under a second. If your change
needs a real model to be meaningful, add a script under `scripts/` instead of a
test that requires a GPU.

```bash
make lint   # ruff check
make fmt    # ruff format
make test   # pytest -q
```

CI runs exactly these three on Python 3.12 and 3.13, plus a Docker build. Run
`ruff format .` before opening a PR or CI will fail on the format check.

## Where things go

| You want to… | Touch |
|---|---|
| add/change an HTTP field or endpoint | `schemas/`, `api/routes/`, `api/mappers.py` (all under `src/memorysum_vl/`) |
| change orchestration | `src/memorysum_vl/services/` (no I/O, no storage details) |
| change retrieval quality | `src/memorysum_vl/adapters/stores/sqlite_store.py` (routes, fusion, rerank) |
| change persistence | `src/memorysum_vl/adapters/stores/repository.py` (SQL lives here and nowhere else) |
| support another backend | implement `src/memorysum_vl/domain/ports.py` protocols + a factory in `src/memorysum_vl/adapters/stores/factory.py` |
| change a model client | `src/memorysum_vl/adapters/llm/*` — keep failures as `RemoteModelError` so degradation still works |
| add a knob | `src/memorysum_vl/core/config.py` (defaults + validation) **and** `.env.example` **and** `docs/deployment.md` |

## House rules

* **The domain layer stays pure.** `src/memorysum_vl/domain/` imports nothing from FastAPI, SQLite or
  httpx. Everything crosses a port.
* **Degrade, don't fail.** A missing model server must not fail a request; mark the
  unit (`caption_state` / `embedding_state`) and log a warning. Tests cover this.
* **No hidden I/O.** Writes are synchronous by contract: when `Add` returns
  `success`, the unit is searchable. Do not move work into a background task.
* **Logs are for humans.** One line per event, request-id correlated, bodies
  truncated. Never log a full data-URI payload.
* **Chinese and English both live here.** Code comments and commit messages may be
  either; user-facing docs are English (`README.md`) with a Chinese mirror
  (`README.zh-CN.md`).

## Pull requests

1. One concern per PR; describe *why*, not only *what*.
2. `make lint && make test` green, plus `ruff format .`.
3. If behaviour or configuration changed, update `docs/` and `.env.example` in the
   same PR — a stale README is a bug.
4. If you touched retrieval, say what you measured (query, `top_k`, latency,
   recall before/after). "Feels better" is not a benchmark.

## Reporting bugs

Include: what you called, the `x-request-id` from the response, and the matching
log lines (`grep <request-id> info.log`). The middleware logs the request body,
the response size and the retrieval route statistics for exactly this reason.
