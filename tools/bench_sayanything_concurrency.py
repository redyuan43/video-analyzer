#!/usr/bin/env python3
"""Benchmark SayAnything Gateway chat-completion concurrency."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://100.91.42.28:18080/v1"
DEFAULT_MODEL = "say_anything_v0.3.3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--requests", type=int, default=4, help="Requests per benchmark round")
    parser.add_argument(
        "--concurrency",
        default="1,2,4",
        help="Comma-separated concurrency levels. 1 is serial baseline.",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--session-prefix", default="video-analyzer-bench")
    parser.add_argument("--codex", choices=["auto", "force", "deny"], default="deny")
    parser.add_argument("--with-system", action="store_true", help="Include a system message; default matches GenericOpenAIAPIClient user-only requests")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def bypass_proxy_environment() -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def post_chat_completion(
    endpoint: str,
    model: str,
    batch: str,
    index: int,
    max_tokens: int,
    timeout: float,
    session_prefix: str,
    codex: str,
    with_system: bool,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if with_system:
        payload["messages"].append({"role": "system", "content": "/no_think 直接输出最终答案，不要解释。"})
    payload["messages"].append({"role": "user", "content": f"/no_think\n并发测试 {batch}-{index}：请只回答“OK {batch}-{index}”。"})
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-SayAnything-Session": f"{session_prefix}-{batch}-{index}-{int(time.time() * 1000)}",
            "X-SayAnything-Codex": codex,
        },
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.perf_counter()
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        elapsed = time.perf_counter() - started
        body = json.loads(raw)
        content = (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        return {
            "index": index,
            "ok": True,
            "seconds": round(elapsed, 3),
            "model": body.get("model"),
            "content": content[:200],
        }
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - started
        try:
            detail = exc.read().decode("utf-8")[:500]
        except Exception:
            detail = ""
        return {
            "index": index,
            "ok": False,
            "seconds": round(elapsed, 3),
            "error": f"HTTP {exc.code} {exc.reason}",
            "detail": detail,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {"index": index, "ok": False, "seconds": round(elapsed, 3), "error": repr(exc)}


def run_round(args: argparse.Namespace, concurrency: int) -> dict[str, Any]:
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    started = time.perf_counter()
    batch = f"c{concurrency}"
    rows: list[dict[str, Any]] = []
    if concurrency == 1:
        for index in range(1, args.requests + 1):
            rows.append(
                post_chat_completion(
                    endpoint,
                    args.model,
                    batch,
                    index,
                    args.max_tokens,
                    args.timeout,
                    args.session_prefix,
                    args.codex,
                    args.with_system,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    post_chat_completion,
                    endpoint,
                    args.model,
                    batch,
                    index,
                    args.max_tokens,
                    args.timeout,
                    args.session_prefix,
                    args.codex,
                    args.with_system,
                )
                for index in range(1, args.requests + 1)
            ]
            for future in as_completed(futures):
                rows.append(future.result())
        rows.sort(key=lambda item: int(item["index"]))
    wall_seconds = round(time.perf_counter() - started, 3)
    ok_rows = [row for row in rows if row.get("ok")]
    row_seconds = [float(row.get("seconds") or 0) for row in rows]
    return {
        "concurrency": concurrency,
        "requests": args.requests,
        "wall_seconds": wall_seconds,
        "ok": len(ok_rows),
        "failed": len(rows) - len(ok_rows),
        "throughput_rps": round(len(ok_rows) / wall_seconds, 3) if wall_seconds > 0 else 0,
        "min_request_seconds": round(min(row_seconds), 3) if row_seconds else 0,
        "max_request_seconds": round(max(row_seconds), 3) if row_seconds else 0,
        "rows": rows,
    }


def parse_concurrency(value: str) -> list[int]:
    levels = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        levels.append(max(1, int(part)))
    return levels or [1]


def main() -> int:
    args = parse_args()
    bypass_proxy_environment()
    payload = {
        "base_url": args.base_url,
        "model": args.model,
        "requests_per_round": args.requests,
        "rounds": [run_round(args, level) for level in parse_concurrency(args.concurrency)],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
