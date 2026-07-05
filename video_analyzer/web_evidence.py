"""Collect external web evidence for existing evidence gaps."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests

from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient
from video_analyzer.config import Config, build_openai_extra_body, resolve_api_key, resolve_temperature


DEFAULT_TEXT_MODEL = "redhatai_qwen3.6-35b-a3b-nvfp4"
DEFAULT_TIMEOUT_SECONDS = 12
DEFAULT_MAX_GAPS = 12
DEFAULT_MAX_RESULTS_PER_GAP = 4
DEFAULT_MAX_PAGE_CHARS = 6000
VIDEO_ONLY_CATEGORIES = {
    "frames_manifest_missing",
    "frame_missing",
    "asr_empty",
    "ocr_empty",
    "ocr_failed",
    "ocr_text_empty",
    "vl_empty",
    "vl_failed",
    "vl_response_empty",
}


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    text: str = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect external web evidence for video-analysis evidence gaps")
    parser.add_argument("run_dir")
    parser.add_argument("--config", default="config")
    parser.add_argument("--profile")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--text-model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-gaps", type=int, default=DEFAULT_MAX_GAPS)
    parser.add_argument("--max-results-per-gap", type=int, default=DEFAULT_MAX_RESULTS_PER_GAP)
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--no-network", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config(args.config)
    profile = config.get_runtime_profile(args.profile)
    default_base_url = (config.get("endpoints") or {}).get("services", {}).get("amd_fast_base_url")
    llm_base_url = args.llm_base_url or profile.get("llm_base_url") or default_base_url
    text_model = args.text_model or profile.get("text_model") or DEFAULT_TEXT_MODEL
    client = None
    if llm_base_url and text_model:
        client = GenericOpenAIAPIClient(
            resolve_api_key(api_key_env=profile.get("text_api_key_env") or profile.get("api_key_env"), api_url=llm_base_url),
            llm_base_url,
            extra_body=build_openai_extra_body(profile, llm_base_url),
        )
    result = build_web_evidence(
        Path(args.run_dir),
        client=client,
        text_model=text_model,
        temperature=args.temperature if args.temperature is not None else resolve_temperature(profile, 0.2),
        max_gaps=args.max_gaps,
        max_results_per_gap=args.max_results_per_gap,
        timeout_sec=args.timeout_sec,
        no_network=args.no_network,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


def build_web_evidence(
    run_dir: Path,
    *,
    client: Any | None = None,
    text_model: str = DEFAULT_TEXT_MODEL,
    temperature: float = 0.2,
    max_gaps: int = DEFAULT_MAX_GAPS,
    max_results_per_gap: int = DEFAULT_MAX_RESULTS_PER_GAP,
    timeout_sec: float = DEFAULT_TIMEOUT_SECONDS,
    no_network: bool = False,
    search_fn: Any | None = None,
    fetch_fn: Any | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    gaps = read_json(run_dir / "evidence_gaps.json") or {"items": []}
    context = load_context(run_dir)
    items = []
    diagnostics: list[str] = []
    search = search_fn or search_web
    fetch = fetch_fn or fetch_page_text

    for gap in (gaps.get("items") or [])[: max(0, max_gaps)]:
        item = collect_gap_evidence(
            gap,
            context=context,
            client=client,
            text_model=text_model,
            temperature=temperature,
            max_results=max_results_per_gap,
            timeout_sec=timeout_sec,
            no_network=no_network,
            search_fn=search,
            fetch_fn=fetch,
        )
        items.append(item)
        diagnostics.extend(item.get("diagnostics") or [])

    summary = summarize_items(items, total_gaps=len(gaps.get("items") or []))
    payload = {
        "version": 1,
        "run_dir": str(run_dir),
        "summary": summary,
        "items": items,
        "diagnostics": diagnostics[:20],
    }
    write_json(run_dir / "web_evidence.json", payload)
    (run_dir / "web_evidence.md").write_text(render_web_evidence_markdown(payload), encoding="utf-8")
    return {"summary": summary, "web_evidence": payload}


def collect_gap_evidence(
    gap: dict[str, Any],
    *,
    context: dict[str, str],
    client: Any | None,
    text_model: str,
    temperature: float,
    max_results: int,
    timeout_sec: float,
    no_network: bool,
    search_fn: Any,
    fetch_fn: Any,
) -> dict[str, Any]:
    category = str(gap.get("category") or "")
    query = build_search_query(gap, context)
    item: dict[str, Any] = {
        "gap_id": gap.get("id"),
        "category": category,
        "severity": gap.get("severity"),
        "message": gap.get("message"),
        "query": query,
        "status": "unresolved",
        "sources": [],
        "uncertainty_note": "未找到可用于补强该缺口的外部证据，回答时应继续标注需复核。",
        "diagnostics": [],
    }
    if category in VIDEO_ONLY_CATEGORIES:
        item["status"] = "video_only_gap"
        item["uncertainty_note"] = "该缺口属于视频内证据缺失，外部网页不能替代 OCR/VL/截图/Transcript。"
        return item
    if no_network:
        item["diagnostics"].append("network disabled")
        return item
    try:
        results = search_fn(query, max_results=max_results, timeout_sec=timeout_sec)
    except Exception as exc:
        item["diagnostics"].append(f"search failed: {exc}")
        return item
    enriched: list[SearchResult] = []
    for result in results[:max_results]:
        try:
            result.text = fetch_fn(result.url, timeout_sec=timeout_sec)
        except Exception as exc:
            item["diagnostics"].append(f"fetch failed: {result.url}: {exc}")
        enriched.append(result)
    item["sources"] = [source_to_dict(result) for result in enriched]
    if client and enriched:
        review = review_gap_with_llm(gap, context, enriched, client, text_model, temperature)
        item.update(review)
    elif enriched:
        item["status"] = "partial_external_support"
        item["uncertainty_note"] = "已找到外部候选来源，但未经过模型复核，只能作为背景线索。"
    return item


def build_search_query(gap: dict[str, Any], context: dict[str, str]) -> str:
    terms = [
        extract_title(context.get("page_context") or "") or extract_title(context.get("operation_manual") or ""),
        str(gap.get("message") or ""),
        str(gap.get("category") or ""),
    ]
    compact = " ".join(clean_text(term) for term in terms if clean_text(term))
    compact = re.sub(r"\b(gap|missing|warning|error|info)\b", " ", compact, flags=re.I)
    words = compact.split()
    return " ".join(words[:16]) or "video tutorial evidence"


def search_web(query: str, *, max_results: int, timeout_sec: float) -> list[SearchResult]:
    response = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": "Mozilla/5.0 video-analyzer evidence collector"},
        timeout=timeout_sec,
    )
    response.raise_for_status()
    return parse_duckduckgo_html(response.text, max_results=max_results)


def parse_duckduckgo_html(text: str, *, max_results: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
        r'(?:<a[^>]+class="result__snippet"[^>]*>|<div[^>]+class="result__snippet"[^>]*>)(?P<snippet>.*?)(?:</a>|</div>)',
        flags=re.S | re.I,
    )
    for match in pattern.finditer(text):
        url = clean_result_url(html.unescape(match.group("href")))
        title = clean_html(match.group("title"))
        snippet = clean_html(match.group("snippet"))
        if url and title:
            results.append(SearchResult(title=title, url=url, snippet=snippet))
        if len(results) >= max_results:
            break
    return results


def clean_result_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        uddg = parse_qs(parsed.query).get("uddg")
        if uddg:
            return unquote(uddg[0])
    return url


def fetch_page_text(url: str, *, timeout_sec: float) -> str:
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 video-analyzer evidence collector"},
        timeout=timeout_sec,
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type") or ""
    if "text" not in content_type and "html" not in content_type and "json" not in content_type:
        return ""
    return trim(clean_html(response.text), DEFAULT_MAX_PAGE_CHARS)


def review_gap_with_llm(
    gap: dict[str, Any],
    context: dict[str, str],
    results: list[SearchResult],
    client: Any,
    text_model: str,
    temperature: float,
) -> dict[str, Any]:
    prompt = build_review_prompt(gap, context, results)
    try:
        response = client.generate(prompt=prompt, model=text_model, temperature=temperature, num_predict=2500)
        parsed = parse_json_object(response.get("response") or "")
    except Exception as exc:
        return {
            "status": "partial_external_support",
            "uncertainty_note": f"外部候选来源存在，但模型复核失败：{exc}",
            "diagnostics": [f"llm review failed: {exc}"],
        }
    status = parsed.get("status") if parsed.get("status") in {"resolved_by_external", "partial_external_support", "unresolved"} else "partial_external_support"
    sources = parsed.get("sources") if isinstance(parsed.get("sources"), list) else []
    return {
        "status": status,
        "used_for": parsed.get("used_for") or "",
        "uncertainty_note": parsed.get("uncertainty_note") or "外部证据只能作为补充，不能替代视频内证据。",
        "sources": merge_review_sources(results, sources),
    }


def build_review_prompt(gap: dict[str, Any], context: dict[str, str], results: list[SearchResult]) -> str:
    sources = []
    for index, result in enumerate(results, start=1):
        sources.append(
            f"[{index}] {result.title}\nURL: {result.url}\n摘要: {result.snippet}\n正文摘录: {trim(result.text, 1800)}"
        )
    return f"""
你是证据复核助手。请判断外部网页是否能补强视频分析中的证据缺口。

规则：
- 外部网页只能补背景、官方说明、版本、项目链接、发布状态等外部事实。
- 不能用外部网页替代视频里的 UI 操作、按钮点击、画面状态、OCR/VL/Transcript。
- 如果只能部分补强，status 用 partial_external_support。
- 如果不能补强，status 用 unresolved。
- 只输出 JSON，不要输出 Markdown。

视频上下文标题/摘要：
{trim(context.get("page_context") or context.get("operation_manual") or "", 2500)}

证据缺口：
{json.dumps(gap, ensure_ascii=False, indent=2)}

外部候选来源：
{chr(10).join(sources)}

输出格式：
{{
  "status": "resolved_by_external | partial_external_support | unresolved",
  "used_for": "这些来源能补强什么",
  "uncertainty_note": "仍需向用户说明的不确定性",
  "sources": [
    {{"url": "来源 URL", "source_confidence": "high|medium|low", "used_for": "用途"}}
  ]
}}
""".strip()


def merge_review_sources(results: list[SearchResult], reviewed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_url = {item.get("url"): item for item in reviewed if isinstance(item, dict)}
    merged = []
    for result in results:
        item = source_to_dict(result)
        review = by_url.get(result.url) or {}
        if review:
            item.update(
                {
                    "source_confidence": review.get("source_confidence") or item["source_confidence"],
                    "used_for": review.get("used_for") or "",
                }
            )
        merged.append(item)
    return merged


def source_to_dict(result: SearchResult) -> dict[str, Any]:
    return {
        "title": result.title,
        "url": result.url,
        "snippet": result.snippet,
        "text_excerpt": trim(result.text, 800),
        "source_confidence": infer_source_confidence(result.url),
    }


def infer_source_confidence(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host.endswith(("github.com", "gitlab.com")) or any(part in host for part in ("docs.", "developer.", "learn.")):
        return "high"
    if host.endswith((".edu", ".gov", ".org")):
        return "medium"
    return "low"


def summarize_items(items: list[dict[str, Any]], *, total_gaps: int) -> dict[str, Any]:
    counts = {
        "total_gaps": total_gaps,
        "processed_gaps": len(items),
        "resolved_by_external": 0,
        "partial_external_support": 0,
        "unresolved": 0,
        "video_only_gap": 0,
        "source_count": 0,
    }
    for item in items:
        status = item.get("status") or "unresolved"
        if status in counts:
            counts[status] += 1
        counts["source_count"] += len(item.get("sources") or [])
    return counts


def render_web_evidence_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    lines = [
        "# 外部联网证据补全",
        "",
        f"- 已处理缺口：{summary.get('processed_gaps', 0)} / {summary.get('total_gaps', 0)}",
        f"- 外部补强：{summary.get('resolved_by_external', 0)}",
        f"- 部分补强：{summary.get('partial_external_support', 0)}",
        f"- 仍未解决：{summary.get('unresolved', 0)}",
        f"- 视频内证据缺口：{summary.get('video_only_gap', 0)}",
        "",
    ]
    for item in payload.get("items") or []:
        lines.extend(
            [
                f"## {item.get('gap_id') or '-'} · {item.get('category') or '-'}",
                "",
                f"- 状态：{item.get('status') or 'unresolved'}",
                f"- 缺口：{item.get('message') or '-'}",
                f"- 不确定性：{item.get('uncertainty_note') or '-'}",
                "",
            ]
        )
        for source in item.get("sources") or []:
            lines.append(f"- [{source.get('title') or source.get('url')}]({source.get('url')}) · {source.get('source_confidence')}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_context(run_dir: Path) -> dict[str, str]:
    return {
        "page_context": read_text(run_dir / "orin" / "page_context.md") or read_text(run_dir.parent / "page_context.md") or read_text(run_dir / "input_page_context.md"),
        "operation_manual": read_text(run_dir / "operation_manual.md"),
        "manual_evidence": read_text(run_dir / "manual_evidence.md"),
    }


def extract_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip("# -*\t ")
        if line:
            return trim(line, 120)
    return ""


def clean_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return clean_text(html.unescape(text))


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def trim(text: str, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if not match:
        match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("model did not return a JSON object")
    return json.loads(match.group(1 if match.lastindex else 0))


def read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
