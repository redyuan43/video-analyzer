"""Collect external web evidence and audit factual claims from video artifacts."""

from __future__ import annotations

import argparse
import html
import json
import re
import shlex
import subprocess
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
DEFAULT_MAX_CLAIMS = 20
DEFAULT_MAX_RESULTS_PER_GAP = 4
DEFAULT_MAX_PAGE_CHARS = 6000
DEFAULT_SEARCH_PROVIDER = "duckduckgo"
AMD_BRAVE_SEARCH_PROVIDER = "amd_brave_ssh"
DEFAULT_AMD_SSH_TARGET = "AMD"
DEFAULT_AMD_BRAVE_ENV_FILE = "~/.lmstudio/credentials/brave-search.env"
FACT_VERDICTS = {"supported", "contradicted", "not_enough_evidence", "not_applicable"}
SOURCE_TIERS = {"primary", "authoritative", "secondary", "unknown"}
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

# The script is sent over stdin to AMD. It reads the remote private credential
# file there, so BRAVE_API_KEY never appears in nx2 arguments, logs, or output.
REMOTE_BRAVE_BRIDGE = r"""
set -eu
ENV_FILE="$1"
ACTION="$2"
REQUEST_VALUE="$3"
case "$ENV_FILE" in
  "~/"*) ENV_FILE="$HOME/${ENV_FILE:2}" ;;
esac
if [ ! -r "$ENV_FILE" ]; then
  echo '{"error":"Brave credential file is not readable"}' >&2
  exit 2
fi
set -a
. "$ENV_FILE"
set +a
export ACTION REQUEST_VALUE
python3 - <<'PY'
import html
import json
import os
import re
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

action = os.environ.get("ACTION", "")
value = os.environ.get("REQUEST_VALUE", "")
api_key = os.environ.get("BRAVE_API_KEY", "")
if not api_key:
    raise SystemExit("BRAVE_API_KEY is not set")

headers = {"User-Agent": "Mozilla/5.0 video-analyzer evidence collector"}
if action == "search":
    params = urlencode({"q": value, "count": 8, "country": "US"})
    request = Request(
        "https://api.search.brave.com/res/v1/web/search?" + params,
        headers={**headers, "Accept": "application/json", "X-Subscription-Token": api_key},
    )
    with urlopen(request, timeout=30) as response:
        print(response.read().decode("utf-8", "replace"))
elif action == "fetch":
    request = Request(value, headers=headers)
    with urlopen(request, timeout=30) as response:
        content_type = response.headers.get("content-type", "")
        body = response.read(120000).decode("utf-8", "replace")
    if "text" not in content_type and "html" not in content_type and "json" not in content_type:
        body = ""
    body = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", body)
    body = re.sub(r"(?s)<[^>]+>", " ", body)
    body = re.sub(r"\s+", " ", html.unescape(body)).strip()
    print(json.dumps({"text": body[:6000]}, ensure_ascii=False))
else:
    raise SystemExit("unsupported Brave bridge action")
PY
""".lstrip()


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    text: str = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and audit external web evidence for video-analysis artifacts")
    parser.add_argument("run_dir")
    parser.add_argument("--config", default="config")
    parser.add_argument("--profile")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--text-model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-gaps", type=int, default=DEFAULT_MAX_GAPS)
    parser.add_argument("--max-claims", type=int, default=DEFAULT_MAX_CLAIMS)
    parser.add_argument("--max-results-per-gap", type=int, default=DEFAULT_MAX_RESULTS_PER_GAP)
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--search-provider", choices=(DEFAULT_SEARCH_PROVIDER, AMD_BRAVE_SEARCH_PROVIDER))
    parser.add_argument("--brave-ssh-target")
    parser.add_argument("--brave-remote-env-file")
    parser.add_argument("--no-network", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config(args.config)
    profile = config.get_runtime_profile(args.profile)
    web_settings = config.get("web_evidence") or {}
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
    search_provider = (
        args.search_provider
        or profile.get("web_evidence_search_provider")
        or web_settings.get("search_provider")
        or DEFAULT_SEARCH_PROVIDER
    )
    result = build_web_evidence(
        Path(args.run_dir),
        client=client,
        text_model=text_model,
        temperature=args.temperature if args.temperature is not None else resolve_temperature(profile, 0.2),
        max_gaps=args.max_gaps,
        max_claims=args.max_claims,
        max_results_per_gap=args.max_results_per_gap,
        timeout_sec=args.timeout_sec,
        no_network=args.no_network,
        search_provider=search_provider,
        brave_ssh_target=(
            args.brave_ssh_target
            or profile.get("brave_ssh_target")
            or web_settings.get("brave_ssh_target")
            or DEFAULT_AMD_SSH_TARGET
        ),
        brave_remote_env_file=(
            args.brave_remote_env_file
            or profile.get("brave_remote_env_file")
            or web_settings.get("brave_remote_env_file")
            or DEFAULT_AMD_BRAVE_ENV_FILE
        ),
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
    max_claims: int = DEFAULT_MAX_CLAIMS,
    max_results_per_gap: int = DEFAULT_MAX_RESULTS_PER_GAP,
    timeout_sec: float = DEFAULT_TIMEOUT_SECONDS,
    no_network: bool = False,
    search_provider: str = DEFAULT_SEARCH_PROVIDER,
    brave_ssh_target: str = DEFAULT_AMD_SSH_TARGET,
    brave_remote_env_file: str = DEFAULT_AMD_BRAVE_ENV_FILE,
    search_fn: Any | None = None,
    fetch_fn: Any | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    gaps = read_json(run_dir / "evidence_gaps.json") or {"items": []}
    triage = read_json(run_dir / "evidence_triage.json") or {"items": []}
    triage_by_gap_id = {item.get("gap_id"): item for item in triage.get("items") or [] if isinstance(item, dict)}
    context = load_context(run_dir)
    search, fetch = build_network_functions(
        search_provider,
        brave_ssh_target=brave_ssh_target,
        brave_remote_env_file=brave_remote_env_file,
        search_fn=search_fn,
        fetch_fn=fetch_fn,
    )
    items = []
    diagnostics: list[str] = []

    for gap in (gaps.get("items") or [])[: max(0, max_gaps)]:
        item = collect_gap_evidence(
            gap,
            triage_item=triage_by_gap_id.get(gap.get("id")),
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

    claims = extract_review_claims(context, max_claims=max_claims)
    audited_claims = []
    for claim in claims:
        item = audit_fact_claim(
            claim,
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
        audited_claims.append(item)
        diagnostics.extend(item.get("diagnostics") or [])

    summary = summarize_items(
        items,
        total_gaps=len(gaps.get("items") or []),
        claims=audited_claims,
    )
    payload = {
        "version": 2,
        "run_dir": str(run_dir),
        "search_provider": search_provider,
        "summary": summary,
        "items": items,
        "claims": audited_claims,
        "diagnostics": diagnostics[:40],
    }
    write_json(run_dir / "web_evidence.json", payload)
    (run_dir / "web_evidence.md").write_text(render_web_evidence_markdown(payload), encoding="utf-8")
    return {"summary": summary, "web_evidence": payload}


def build_network_functions(
    search_provider: str,
    *,
    brave_ssh_target: str,
    brave_remote_env_file: str,
    search_fn: Any | None,
    fetch_fn: Any | None,
) -> tuple[Any, Any]:
    if search_fn or fetch_fn:
        return search_fn or search_web, fetch_fn or fetch_page_text
    if search_provider == AMD_BRAVE_SEARCH_PROVIDER:
        return (
            lambda query, max_results, timeout_sec: search_amd_brave(
                query,
                max_results=max_results,
                timeout_sec=timeout_sec,
                ssh_target=brave_ssh_target,
                remote_env_file=brave_remote_env_file,
            ),
            lambda url, timeout_sec: fetch_amd_brave_page(
                url,
                timeout_sec=timeout_sec,
                ssh_target=brave_ssh_target,
                remote_env_file=brave_remote_env_file,
            ),
        )
    return search_web, fetch_page_text


def collect_gap_evidence(
    gap: dict[str, Any],
    *,
    triage_item: dict[str, Any] | None = None,
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
    if triage_item and triage_item.get("resolution_route") != "web_search":
        item["status"] = "not_applicable"
        item["uncertainty_note"] = triage_item.get("recommendation") or "证据分诊认为该缺口不应通过联网补证据解决。"
        item["triage"] = {
            "resolution_route": triage_item.get("resolution_route"),
            "evidence_class": triage_item.get("evidence_class"),
            "publish_impact": triage_item.get("publish_impact"),
        }
        return item
    if no_network:
        item["diagnostics"].append("network disabled")
        return item
    try:
        results = search_fn(query, max_results=max_results, timeout_sec=timeout_sec)
    except Exception as exc:
        item["diagnostics"].append(f"search failed: {exc}")
        return item
    enriched = enrich_results(results, max_results=max_results, timeout_sec=timeout_sec, fetch_fn=fetch_fn, diagnostics=item["diagnostics"])
    item["sources"] = [source_to_dict(result) for result in enriched]
    if client and enriched:
        review = review_gap_with_llm(gap, context, enriched, client, text_model, temperature)
        item.update(review)
    elif enriched:
        item["status"] = "partial_external_support"
        item["uncertainty_note"] = "已找到外部候选来源，但未经过模型复核，只能作为背景线索。"
    return item


def audit_fact_claim(
    claim: dict[str, Any],
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
    item = {
        **claim,
        "query": build_claim_query(claim, context),
        "verdict": "not_enough_evidence",
        "confidence": "low",
        "conclusion": "尚未获得足以判断该视频断言的外部事实依据。",
        "recommended_wording": "保留为“视频中称”，并标注待复核。",
        "sources": [],
        "source_ids": [],
        "diagnostics": [],
    }
    if no_network:
        item["diagnostics"].append("network disabled")
        return item
    try:
        results = search_fn(item["query"], max_results=max_results, timeout_sec=timeout_sec)
    except Exception as exc:
        item["diagnostics"].append(f"search failed: {exc}")
        return item
    enriched = enrich_results(results, max_results=max_results, timeout_sec=timeout_sec, fetch_fn=fetch_fn, diagnostics=item["diagnostics"])
    item["sources"] = [source_to_dict(result, index=index) for index, result in enumerate(enriched, start=1)]
    if not enriched:
        item["conclusion"] = "未找到可核验该断言的来源。"
        return item
    if not client:
        item["conclusion"] = "已找到候选来源，但尚未经过模型事实裁决。"
        return item
    item.update(review_claim_with_llm(item, context, enriched, client, text_model, temperature))
    return item


def extract_review_claims(context: dict[str, str], *, max_claims: int) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    sources = (
        ("operation_manual.md", context.get("operation_manual") or ""),
        ("operation_manual_review.md", context.get("operation_manual_review") or ""),
        ("manual_evidence.md", context.get("manual_evidence") or ""),
    )
    seen: set[str] = set()
    for source_file, text in sources:
        for excerpt in review_section_items(text):
            normalized = normalize_claim(excerpt)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            claims.append(
                {
                    "id": f"claim_{len(claims) + 1:04d}",
                    "claim": normalized,
                    "source_file": source_file,
                    "source_excerpt": trim(excerpt, 500),
                }
            )
            if len(claims) >= max(0, max_claims):
                return claims
    return claims


def review_section_items(text: str) -> list[str]:
    section = extract_review_section(text)
    if not section:
        return []
    items: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped or re.fullmatch(r"\|?[\s:|-]+\|?", stripped):
            continue
        if "|" in stripped:
            cells = [clean_text(cell) for cell in stripped.strip("|").split("|")]
            if len(cells) >= 2 and cells[0] not in {"序号", "内容"} and not cells[0].isdigit():
                continue
            if len(cells) >= 2 and cells[1] not in {"内容", "证据来源", "复核建议"}:
                items.append(cells[1])
            continue
        bullet = re.sub(r"^(?:[-*+]\s+|\d+[.)、]\s*)", "", stripped)
        if bullet != stripped:
            items.append(bullet)
    return items


def extract_review_section(text: str) -> str:
    if not text:
        return ""
    heading = re.search(r"(?im)^#{1,6}\s*(?:\d+[.、]\s*)?需复核(?:项|内容)?\s*$", text)
    if not heading:
        heading = re.search(r"(?im)^#{1,6}\s*.*需复核.*$", text)
    if not heading:
        return ""
    section = text[heading.end() :]
    next_heading = re.search(r"(?m)^#{1,6}\s+\S", section)
    if next_heading:
        section = section[: next_heading.start()]
    return section.strip()


def normalize_claim(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"^(?:播客|视频|讲者|作者)(?:中)?(?:称|说|提到)[：: ]*", "", value)
    return value.strip("：:。；; ")


def build_claim_query(claim: dict[str, Any], context: dict[str, str]) -> str:
    title = extract_title(context.get("page_context") or "") or extract_title(context.get("operation_manual") or "")
    compact = " ".join(part for part in (claim.get("claim"), title) if clean_text(part))
    return " ".join(compact.split()[:24]) or "video factual claim verification"


def enrich_results(
    results: list[SearchResult],
    *,
    max_results: int,
    timeout_sec: float,
    fetch_fn: Any,
    diagnostics: list[str],
) -> list[SearchResult]:
    enriched: list[SearchResult] = []
    for result in sort_results_by_source_tier(results[:max_results]):
        try:
            result.text = fetch_fn(result.url, timeout_sec=timeout_sec)
        except Exception as exc:
            diagnostics.append(f"fetch failed: {result.url}: {exc}")
        enriched.append(result)
    return enriched


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


def search_amd_brave(
    query: str,
    *,
    max_results: int,
    timeout_sec: float,
    ssh_target: str,
    remote_env_file: str,
) -> list[SearchResult]:
    payload = run_amd_brave_bridge(
        "search",
        query,
        timeout_sec=timeout_sec,
        ssh_target=ssh_target,
        remote_env_file=remote_env_file,
    )
    results = (payload.get("web") or {}).get("results") or []
    parsed = []
    for item in results[:max_results]:
        if not isinstance(item, dict):
            continue
        url = clean_text(item.get("url"))
        title = clean_text(item.get("title"))
        if url and title:
            parsed.append(SearchResult(title=title, url=url, snippet=clean_text(item.get("description"))))
    return parsed


def fetch_amd_brave_page(
    url: str,
    *,
    timeout_sec: float,
    ssh_target: str,
    remote_env_file: str,
) -> str:
    payload = run_amd_brave_bridge(
        "fetch",
        url,
        timeout_sec=timeout_sec,
        ssh_target=ssh_target,
        remote_env_file=remote_env_file,
    )
    return trim(clean_text(payload.get("text")), DEFAULT_MAX_PAGE_CHARS)


def run_amd_brave_bridge(
    action: str,
    value: str,
    *,
    timeout_sec: float,
    ssh_target: str,
    remote_env_file: str,
) -> dict[str, Any]:
    if action not in {"search", "fetch"}:
        raise ValueError(f"unsupported Brave bridge action: {action}")
    remote_command = "bash -s -- " + " ".join(
        shlex.quote(part) for part in (remote_env_file, action, value)
    )
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        ssh_target,
        remote_command,
    ]
    try:
        result = subprocess.run(
            command,
            input=REMOTE_BRAVE_BRIDGE,
            text=True,
            capture_output=True,
            timeout=max(15, int(timeout_sec) + 20),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("AMD Brave SSH request timed out") from exc
    if result.returncode != 0:
        message = redact_secret(trim(result.stderr or result.stdout or "AMD Brave SSH request failed", 500))
        raise RuntimeError(message)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("AMD Brave bridge returned invalid JSON") from exc


def redact_secret(value: str) -> str:
    return re.sub(r"(BRAVE_API_KEY=)[^\s]+", r"\1[REDACTED]", str(value or ""))


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


def review_claim_with_llm(
    claim: dict[str, Any],
    context: dict[str, str],
    results: list[SearchResult],
    client: Any,
    text_model: str,
    temperature: float,
) -> dict[str, Any]:
    prompt = build_claim_review_prompt(claim, context, results)
    try:
        response = client.generate(prompt=prompt, model=text_model, temperature=temperature, num_predict=1800)
        parsed = parse_json_object(response.get("response") or "")
    except Exception as exc:
        return {
            "verdict": "not_enough_evidence",
            "confidence": "low",
            "conclusion": "候选来源存在，但模型事实裁决失败。",
            "recommended_wording": "保留为“视频中称”，并标注待复核。",
            "source_ids": [],
            "diagnostics": [f"claim review failed: {exc}"],
        }
    source_ids = [str(item) for item in parsed.get("source_ids") or [] if re.fullmatch(r"source_\d{2}", str(item))]
    available_ids = {f"source_{index:02d}" for index in range(1, len(results) + 1)}
    source_ids = [item for item in source_ids if item in available_ids]
    verdict = str(parsed.get("verdict") or "")
    if verdict not in FACT_VERDICTS or (verdict in {"supported", "contradicted"} and not source_ids):
        verdict = "not_enough_evidence"
    confidence = str(parsed.get("confidence") or "").lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    if verdict == "not_enough_evidence":
        confidence = "low" if not source_ids else confidence
    return {
        "verdict": verdict,
        "confidence": confidence,
        "conclusion": sentence_safe_trim(clean_text(parsed.get("conclusion")) or "证据不足，不能确认该断言。", 400),
        "recommended_wording": sentence_safe_trim(
            clean_text(parsed.get("recommended_wording")) or "保留为“视频中称”，并标注待复核。",
            300,
        ),
        "source_ids": source_ids,
    }


def build_review_prompt(gap: dict[str, Any], context: dict[str, str], results: list[SearchResult]) -> str:
    sources = format_sources_for_prompt(results)
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
{sources}

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


def build_claim_review_prompt(claim: dict[str, Any], context: dict[str, str], results: list[SearchResult]) -> str:
    return f"""
你是严格的视频事实核验员。请判断“视频/音频中的说法”是否由候选来源直接支持。

规则：
- 先区分“视频说了什么”和“外部事实是否成立”；不要改写或否认视频原始记录。
- 优先官方公告、厂商文档、财报、监管机构、标准组织、技术白皮书；没有直接一手依据时只能降低置信度。
- 只有来源明确支持或直接冲突时才用 supported / contradicted。
- 搜索摘要、间接提及、无法覆盖数字/日期/范围的网页不能确认断言。
- “没找到”不是“反驳”，应为 not_enough_evidence。
- 只能引用下面给出的 source_id；不得编造 URL、来源或事实。
- 只输出 JSON。

视频原始断言：
{json.dumps({key: claim.get(key) for key in ("claim", "source_file", "source_excerpt")}, ensure_ascii=False)}

视频上下文：
{trim(context.get("page_context") or context.get("operation_manual") or "", 1800)}

候选来源：
{format_sources_for_prompt(results)}

输出格式：
{{
  "verdict": "supported|contradicted|not_enough_evidence|not_applicable",
  "confidence": "high|medium|low",
  "conclusion": "基于来源的事实判断，包含数字/范围/时间的边界",
  "recommended_wording": "面向最终读者的安全表述",
  "source_ids": ["source_01"]
}}
""".strip()


def format_sources_for_prompt(results: list[SearchResult]) -> str:
    sources = []
    for index, result in enumerate(results, start=1):
        sources.append(
            f"[source_{index:02d}] {result.title}\n"
            f"URL: {result.url}\n"
            f"来源等级: {infer_source_tier(result.url)}\n"
            f"摘要: {result.snippet}\n"
            f"正文摘录: {trim(result.text, 1800)}"
        )
    return "\n\n".join(sources)


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


def source_to_dict(result: SearchResult, *, index: int | None = None) -> dict[str, Any]:
    payload = {
        "title": result.title,
        "url": result.url,
        "snippet": result.snippet,
        "text_excerpt": trim(result.text, 800),
        "source_confidence": infer_source_confidence(result.url),
        "source_tier": infer_source_tier(result.url),
    }
    if index is not None:
        payload["source_id"] = f"source_{index:02d}"
    return payload


def infer_source_tier(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host.endswith((".gov", ".mil", ".int")) or any(part in host for part in ("standards.", "itu.", "iso.", "iec.")):
        return "authoritative"
    if any(part in host for part in ("docs.", "developer.", "learn.", "investor.", "investors.", "ir.")):
        return "primary"
    if host.endswith(("github.com", "gitlab.com")):
        return "primary"
    if host.endswith((".edu", ".org")):
        return "authoritative"
    if host:
        return "secondary"
    return "unknown"


def infer_source_confidence(url: str) -> str:
    tier = infer_source_tier(url)
    return "high" if tier in {"primary", "authoritative"} else "medium" if tier == "secondary" else "low"


def sort_results_by_source_tier(results: list[SearchResult]) -> list[SearchResult]:
    order = {"primary": 0, "authoritative": 1, "secondary": 2, "unknown": 3}
    return sorted(results, key=lambda result: (order[infer_source_tier(result.url)], result.url))


def summarize_items(
    items: list[dict[str, Any]],
    *,
    total_gaps: int,
    claims: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    counts = {
        "total_gaps": total_gaps,
        "processed_gaps": len(items),
        "resolved_by_external": 0,
        "partial_external_support": 0,
        "unresolved": 0,
        "video_only_gap": 0,
        "not_applicable": 0,
        "source_count": 0,
        "claim_count": len(claims or []),
        "supported": 0,
        "contradicted": 0,
        "not_enough_evidence": 0,
        "claim_not_applicable": 0,
    }
    for item in items:
        status = item.get("status") or "unresolved"
        if status in counts:
            counts[status] += 1
        counts["source_count"] += len(item.get("sources") or [])
    for claim in claims or []:
        verdict = claim.get("verdict") or "not_enough_evidence"
        if verdict in {"supported", "contradicted", "not_enough_evidence"}:
            counts[verdict] += 1
        elif verdict == "not_applicable":
            counts["claim_not_applicable"] += 1
        counts["source_count"] += len(claim.get("sources") or [])
    return counts


def render_web_evidence_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    lines = [
        "# 联网事实核验",
        "",
        f"- 检索提供方：{payload.get('search_provider') or DEFAULT_SEARCH_PROVIDER}",
        f"- 已处理缺口：{summary.get('processed_gaps', 0)} / {summary.get('total_gaps', 0)}",
        f"- 外部补强：{summary.get('resolved_by_external', 0)}",
        f"- 部分补强：{summary.get('partial_external_support', 0)}",
        f"- 仍未解决：{summary.get('unresolved', 0)}",
        "",
        "## 视频断言事实审计",
        "",
        f"- 断言数：{summary.get('claim_count', 0)}",
        f"- 有直接支持：{summary.get('supported', 0)}",
        f"- 有直接冲突：{summary.get('contradicted', 0)}",
        f"- 证据不足：{summary.get('not_enough_evidence', 0)}",
        "",
    ]
    for claim in payload.get("claims") or []:
        lines.extend(
            [
                f"### {claim.get('id') or '-'} · {claim.get('verdict') or 'not_enough_evidence'}",
                "",
                f"- 视频说法：{claim.get('claim') or '-'}",
                f"- 视频来源：{claim.get('source_file') or '-'}",
                f"- 核验结论：{claim.get('conclusion') or '-'}",
                f"- 置信度：{claim.get('confidence') or 'low'}",
                f"- 建议表述：{claim.get('recommended_wording') or '-'}",
                "",
            ]
        )
        for source in claim.get("sources") or []:
            marker = "*" if source.get("source_id") in (claim.get("source_ids") or []) else "-"
            lines.append(
                f"{marker} [{source.get('title') or source.get('url')}]({source.get('url')})"
                f" · {source.get('source_tier') or 'unknown'}"
            )
        lines.append("")
    if payload.get("items"):
        lines.extend(["## 证据缺口补强", ""])
    for item in payload.get("items") or []:
        lines.extend(
            [
                f"### {item.get('gap_id') or '-'} · {item.get('category') or '-'}",
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
        "operation_manual_review": read_text(run_dir / "docs_analysis" / "operation_manual_review.md"),
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


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def sentence_safe_trim(text: str, limit: int) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    boundary = max(text.rfind(mark, 0, limit) for mark in ("。", "；", ".", ";"))
    return text[:boundary].rstrip() if boundary >= max(20, limit // 2) else trim(text, limit)


def trim(text: Any, limit: int) -> str:
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
