"""`python -m memorysum_vl` / `memorysum-vl` — run the API server.

Kept separate from `main.py` so importing the app never starts uvicorn (tests and
the Docker `HEALTHCHECK` import it directly).
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="memorysum-vl",
        description="Run the memorysum-vl memory service (needs the three llama.cpp servers).",
    )
    parser.add_argument("--host", default=os.getenv("MEMORYSUM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MEMORYSUM_PORT", "8003")))
    parser.add_argument(
        "--reload", action="store_true", help="reload on code changes (development)"
    )
    args = parser.parse_args()

    import uvicorn

    # log_config=None keeps our own formatter/handlers instead of uvicorn's defaults
    uvicorn.run(
        "memorysum_vl.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        access_log=False,
        log_config=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
