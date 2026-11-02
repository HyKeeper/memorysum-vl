#!/usr/bin/env python3
"""对本机服务做一次真实冒烟：写入文本 + 图片，再按文字/图片各搜一次。

```
.venv/bin/python scripts/smoke_api.py
.venv/bin/python scripts/smoke_api.py --images a.jpg b.png --query "这张图里是什么？"
```

不给 `--images` 时会用 Pillow 现生成两张 PNG，所以不依赖任何本地样本数据。
需要服务已启动，且 `.secrets/api_token` 存在（或用 `--token-file` 指定）。
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import pathlib
import tempfile
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_TEXT = "Bob: I'm flying to Tokyo on Tuesday for a conference."
DEFAULT_QUERY = "Which newspaper announced the moon landing?"


def synth_images(directory: pathlib.Path) -> list[pathlib.Path]:
    """没有现成图片时现生成两张：只为了跑通 add/search 链路，不校验检索质量。"""
    from PIL import Image, ImageDraw

    created: list[pathlib.Path] = []
    for index, (color, label) in enumerate([("#1e88e5", "smoke A"), ("#f4511e", "smoke B")]):
        image = Image.new("RGB", (480, 320), color)
        draw = ImageDraw.Draw(image)
        draw.rectangle([40, 40, 440, 280], outline="white", width=6)
        draw.text((70, 150), label, fill="white")
        path = directory / f"smoke-{index}.png"
        image.save(path)
        created.append(path)
    return created


def call(base: str, token: str, path: str, payload: dict) -> tuple[dict, float]:
    request = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read()), time.perf_counter() - started


def data_uri(path: pathlib.Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{payload}"


def render(hit: dict) -> str:
    content = hit["content"]
    if isinstance(content, str):
        return content
    return " | ".join(part["text"] for part in content if part["type"] == "text")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8003")
    parser.add_argument("--token-file", default=str(ROOT / ".secrets/api_token"))
    parser.add_argument("--user", default="smoke-user")
    parser.add_argument(
        "--images", nargs="*", default=None, help="图片路径；不给则自动生成两张 PNG"
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    token = pathlib.Path(args.token_file).read_text(encoding="utf-8").splitlines()[0].strip()
    images = [pathlib.Path(item) for item in (args.images or []) if pathlib.Path(item).exists()]
    if not images:
        images = synth_images(pathlib.Path(tempfile.mkdtemp(prefix="memorysum-smoke-")))
        print(f"未提供可用图片，已生成 {len(images)} 张临时 PNG")

    messages: list[dict] = [{"role": "user", "timestamp": 1_704_067_200_000, "content": args.text}]
    for index, image in enumerate(images):
        messages.append(
            {
                "role": "user",
                "timestamp": 1_704_067_200_000 + (index + 1) * 60_000,
                "content": [
                    {"type": "text", "text": f"reference image {index}"},
                    {"type": "image_url", "image_url": {"url": data_uri(image)}},
                ],
            }
        )

    response, elapsed = call(
        args.base,
        token,
        "/v1/add",
        {
            "request_id": f"smoke:{int(time.time())}",
            "user_id": args.user,
            "session_id": "smoke:session",
            "messages": messages,
        },
    )
    print(f"add  {elapsed * 1000:6.0f} ms  {response}")

    response, elapsed = call(
        args.base,
        token,
        "/v1/search",
        {
            "query": [{"type": "text", "text": args.query}],
            "user_id": args.user,
            "top_k": args.top_k,
        },
    )
    print(f"search(text)  {elapsed * 1000:6.0f} ms")
    for hit in response["data"]:
        print(f"  [{hit['score']:.4f}] {render(hit)[:200]}")

    response, elapsed = call(
        args.base,
        token,
        "/v1/search",
        {
            "query": [
                {"type": "image_url", "image_url": {"url": data_uri(images[-1])}},
                {"type": "text", "text": "what is this?"},
            ],
            "user_id": args.user,
            "top_k": args.top_k,
        },
    )
    print(f"search(image)  {elapsed * 1000:6.0f} ms")
    for hit in response["data"][:2]:
        print(f"  [{hit['score']:.4f}] {render(hit)[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
