#!/usr/bin/env python3
"""Minimal client for memorysum-vl — standard library only.

    python examples/client.py                      # text demo against :8003
    python examples/client.py --image desk.jpg     # add + search with an image
    python examples/client.py --query "where are the keys?"

The token is read from $MEMORYSUM_TOKEN or .secrets/api_token.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import pathlib
import urllib.request

DEFAULT_BASE_URL = os.getenv("MEMORYSUM_BASE_URL", "http://127.0.0.1:8003")
DEFAULT_TOKEN_FILE = os.getenv("MEMORYSUM_TOKEN_FILE", ".secrets/api_token")


class MemoryClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, token: str = "", timeout: float = 300.0):
        if not token:
            raise SystemExit(
                "no token: pass --token, set MEMORYSUM_TOKEN, or create .secrets/api_token"
            )
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

    # -- plumbing ---------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self._base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:  # noqa: PERF203 - surface the envelope
            detail = exc.read().decode("utf-8", "replace")
            raise SystemExit(f"{path} → HTTP {exc.code}: {detail}") from exc

    # -- API --------------------------------------------------------------
    def health(self) -> dict:
        with urllib.request.urlopen(f"{self._base}/healthz", timeout=10) as response:
            return json.loads(response.read())

    def add(self, *, request_id: str, user_id: str, session_id: str, messages: list[dict]) -> dict:
        return self._post(
            "/v1/add",
            {
                "request_id": request_id,
                "user_id": user_id,
                "session_id": session_id,
                "messages": messages,
            },
        )

    def search(
        self,
        *,
        user_id: str,
        query: str | list[dict],
        top_k: int = 10,
        options: list[str] | None = None,
    ) -> list[dict]:
        payload: dict = {"user_id": user_id, "query": query, "top_k": top_k}
        if options:
            payload["options"] = options
        return self._post("/v1/search", payload)["data"]


def inline_image(path: str) -> dict:
    """Encode a local file as the data-URI content part the API expects."""
    data = pathlib.Path(path).read_bytes()
    media_type = mimetypes.guess_type(path)[0] or "image/jpeg"
    if media_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise SystemExit(f"unsupported image type {media_type!r}; use jpeg/png/webp")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{base64.b64encode(data).decode()}"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--token", default=os.getenv("MEMORYSUM_TOKEN", ""))
    parser.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--image", default="", help="add this image and search for it")
    parser.add_argument("--query", default="", help="search query (text)")
    parser.add_argument("--user-id", default="demo-user")
    args = parser.parse_args()

    token = args.token
    if not token and pathlib.Path(args.token_file).is_file():
        token = pathlib.Path(args.token_file).read_text(encoding="utf-8").strip()

    client = MemoryClient(args.base_url, token)
    print("healthz:", client.health())

    if args.image:
        print("\nadd (text + image):")
        print(
            client.add(
                request_id="client-demo-image",
                user_id=args.user_id,
                session_id="client-demo-session",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Here is my desk right now."},
                            inline_image(args.image),
                        ],
                    }
                ],
            )
        )
        query_text = args.query or "what is on my desk?"
        print(f"\nsearch (text + image) — {query_text!r}:")
        for hit in client.search(
            user_id=args.user_id,
            query=[{"type": "text", "text": query_text}, inline_image(args.image)],
            top_k=5,
        ):
            print(f"  {hit['score']:.4f}  {hit['id']}")

    print("\nadd (text):")
    print(
        client.add(
            request_id="client-demo-text",
            user_id=args.user_id,
            session_id="client-demo-session",
            messages=[
                {"role": "user", "content": "The spare key lives in the blue kitchen drawer."},
                {"role": "assistant", "content": "Noted."},
            ],
        )
    )

    query_text = args.query or "where are the keys?"
    print(f"\nsearch — {query_text!r}:")
    for hit in client.search(user_id=args.user_id, query=query_text, top_k=5):
        text = hit["content"][0]["text"] if isinstance(hit["content"], list) else hit["content"]
        print(f"  {hit['score']:.4f}  {hit['id']}  {text[:90]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
