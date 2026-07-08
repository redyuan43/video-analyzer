#!/usr/bin/env python3
"""Build structured study artifacts from an operation-manual run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient
from video_analyzer.config import Config, build_openai_extra_body, resolve_api_key, resolve_temperature
from video_analyzer.multidoc import parse_chapters, timestamp_to_seconds


CORE_SOURCE_TYPES = {"asr", "subtitle", "ocr", "vl", "manual"}
LEARNING_SOURCE_TYPES = {"asr", "subtitle", "ocr", "vl"}
DEFAULT_TEXT_MODEL = "redhatai_qwen3.6-35b-a3b-nvfp4"
DEFAULT_STUDY_CARD_TEMPERATURE = 0.1
STUDY_CARD_MAX_TRANSCRIPT_CHARS = 7000
STUDY_CARD_MAX_CONTEXT_CHARS = 1800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate study guide, evidence gaps, and publish decision")
    parser.add_argument("run_dir")
    parser.add_argument("--config", default="config")
    parser.add_argument("--profile")
    parser.add_argument("--language", default="zh-CN")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--text-model")
    parser.add_argument("--study-card-llm-base-url")
    parser.add_argument("--study-card-model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--study-only", action="store_true", help="Only rebuild study guide/card artifacts; leave review and publish decision untouched")
    parser.add_argument("--skip-review", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Config(args.config)
    profile = config.get_runtime_profile(args.profile)
    study_cards_config = config.get("study_cards") or {}
    default_base_url = (config.get("endpoints") or {}).get("services", {}).get("amd_fast_base_url")
    llm_base_url = args.llm_base_url or profile.get("llm_base_url") or default_base_url
    study_card_llm_base_url = (
        args.study_card_llm_base_url
        or profile.get("study_card_llm_base_url")
        or study_cards_config.get("llm_base_url")
    )
    study_card_model = args.study_card_model or profile.get("study_card_model") or study_cards_config.get("model")
    study_card_temperature = float(
        profile.get(
            "study_card_temperature",
            study_cards_config.get("temperature", DEFAULT_STUDY_CARD_TEMPERATURE),
        )
    )
    result = build_study_artifacts(
        run_dir=Path(args.run_dir),
        language=args.language,
        llm_base_url=llm_base_url,
        text_model=args.text_model or profile.get("text_model"),
        temperature=args.temperature if args.temperature is not None else resolve_temperature(profile, 0.2),
        api_key_env=profile.get("text_api_key_env") or profile.get("api_key_env"),
        extra_body=build_openai_extra_body(profile, llm_base_url),
        study_card_llm_base_url=study_card_llm_base_url,
        study_card_model=study_card_model,
        study_card_temperature=study_card_temperature,
        study_card_api_key_env=(
            profile.get("study_card_api_key_env")
            or study_cards_config.get("api_key_env")
            or profile.get("api_key_env")
        ),
        study_card_extra_body=build_openai_extra_body(profile, study_card_llm_base_url, prefix="study_card_")
        if study_card_llm_base_url
        else None,
        study_only=args.study_only,
        skip_review=args.skip_review,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


def build_study_artifacts(
    run_dir: Path,
    language: str = "zh-CN",
    llm_base_url: str | None = None,
    text_model: str | None = None,
    temperature: float = 0.2,
    api_key_env: str | None = None,
    extra_body: dict[str, Any] | None = None,
    study_card_llm_base_url: str | None = None,
    study_card_model: str | None = None,
    study_card_temperature: float = DEFAULT_STUDY_CARD_TEMPERATURE,
    study_card_api_key_env: str | None = None,
    study_card_extra_body: dict[str, Any] | None = None,
    study_card_client: Any | None = None,
    study_card_fallback_client: Any | None = None,
    study_only: bool = False,
    skip_review: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    if not (run_dir / "analysis.json").is_file():
        raise FileNotFoundError(f"Missing analysis.json in run directory: {run_dir}")

    analysis = read_json(run_dir / "analysis.json")
    frames = load_frame_manifest(run_dir)
    transcript = load_transcript(run_dir, analysis)
    chapters = build_chapters(run_dir, transcript)
    evidence = build_evidence(run_dir, analysis, transcript, frames)
    chapter_packets = build_chapter_packets(run_dir, chapters, evidence, frames)
    synthesize_study_cards(
        chapter_packets,
        language=language,
        study_card_llm_base_url=study_card_llm_base_url,
        study_card_model=study_card_model,
        study_card_temperature=study_card_temperature,
        study_card_api_key_env=study_card_api_key_env,
        study_card_extra_body=study_card_extra_body,
        study_card_client=study_card_client,
        fallback_llm_base_url=llm_base_url,
        fallback_model=text_model,
        fallback_temperature=temperature,
        fallback_api_key_env=api_key_env,
        fallback_extra_body=extra_body,
        fallback_client=study_card_fallback_client,
    )
    gaps = detect_evidence_gaps(run_dir, analysis, transcript, frames, evidence, chapter_packets)
    guide = build_study_guide(run_dir, analysis, evidence, chapter_packets, gaps, language)

    write_json(run_dir / "study_guide.json", guide)
    write_json(run_dir / "evidence_gaps.json", gaps)
    write_chapter_packets(run_dir, chapter_packets)
    write_study_markdown(run_dir, guide, chapter_packets, gaps)
    if study_only:
        decision = read_json_if_exists(run_dir / "publish_decision.json") or {"status": "not_run", "reason": "study-only requested"}
        summary = build_study_summary(run_dir, chapter_packets, evidence, gaps, decision)
        return {
            "summary": summary,
            "study_guide": guide,
            "evidence_gaps": gaps,
            "evidence_review": read_json_if_exists(run_dir / "evidence_review.json") or {},
            "publish_decision": decision,
        }

    review = build_evidence_review(
        run_dir=run_dir,
        guide=guide,
        gaps=gaps,
        chapter_packets=chapter_packets,
        language=language,
        llm_base_url=llm_base_url,
        text_model=text_model or DEFAULT_TEXT_MODEL,
        temperature=temperature,
        api_key_env=api_key_env,
        extra_body=extra_body,
        skip_review=skip_review,
        client=client,
    )
    write_json(run_dir / "evidence_review.json", review)
    (run_dir / "review_notes.md").write_text(render_review_notes(review, gaps), encoding="utf-8")

    decision = build_publish_decision(gaps, review)
    write_json(run_dir / "publish_decision.json", decision)

    summary = build_study_summary(run_dir, chapter_packets, evidence, gaps, decision)
    return {"summary": summary, "study_guide": guide, "evidence_gaps": gaps, "evidence_review": review, "publish_decision": decision}


def build_study_summary(
    run_dir: Path,
    chapter_packets: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    gaps: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "study_guide": str(run_dir / "study_guide.json"),
        "evidence_gaps": str(run_dir / "evidence_gaps.json"),
        "evidence_review": str(run_dir / "evidence_review.json"),
        "publish_decision": str(run_dir / "publish_decision.json"),
        "study_overview": str(run_dir / "study_overview.md"),
        "study_cards": str(run_dir / "study_cards.md"),
        "evidence_index": str(run_dir / "evidence_index.md"),
        "chapters": len(chapter_packets),
        "evidence": len(evidence),
        "gaps": len(gaps.get("items") or []),
        "decision": decision.get("status"),
    }


def load_transcript(run_dir: Path, analysis: dict[str, Any]) -> dict[str, Any]:
    for path in (run_dir / "orin" / "transcript.json", run_dir / "transcript.json"):
        if path.is_file():
            data = read_json(path)
            if isinstance(data, dict):
                return normalize_transcript(data)
    transcript = analysis.get("transcript")
    return normalize_transcript(transcript) if isinstance(transcript, dict) else {}


def normalize_transcript(transcript: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(transcript)
    segments = []
    for segment in transcript.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        start = segment.get("start", segment.get("start_time", segment.get("Start")))
        end = segment.get("end", segment.get("end_time", segment.get("End")))
        text = segment.get("text", segment.get("Content", segment.get("content")))
        segments.append({**segment, "start": start, "end": end, "text": text})
    normalized["segments"] = segments
    return normalized


def load_frame_manifest(run_dir: Path) -> dict[int, dict[str, Any]]:
    manifest_path = run_dir / "frames_manifest.json"
    if not manifest_path.is_file():
        return {}
    payload = read_json(manifest_path)
    frames = payload.get("frames") if isinstance(payload, dict) else payload
    result: dict[int, dict[str, Any]] = {}
    for index, frame in enumerate(frames or []):
        if not isinstance(frame, dict):
            continue
        number = int(frame.get("frame_number", index))
        path = str(frame.get("path") or f"frames/frame_{number}.jpg")
        result[number] = {
            "frame_number": number,
            "path": path,
            "timestamp_sec": float_or_zero(frame.get("timestamp")),
            "timestamp_label": format_timestamp(frame.get("timestamp")),
            "exists": (run_dir / path).is_file(),
            "score": frame.get("score"),
        }
    return result


def is_audio_only_run(run_dir: Path) -> bool:
    manifest_path = run_dir / "frames_manifest.json"
    if not manifest_path.is_file():
        return False
    payload = read_json(manifest_path)
    return isinstance(payload, dict) and payload.get("source") == "audio_only"


def build_chapters(run_dir: Path, transcript: dict[str, Any]) -> list[dict[str, Any]]:
    page_context = read_text_if_exists(run_dir / "orin" / "page_context.md") or read_text_if_exists(run_dir.parent / "page_context.md")
    chapters = parse_chapters(page_context, transcript)
    return [
        {
            "chapter_id": f"chapter_{index:02d}",
            "index": index,
            "title": chapter.get("title") or f"章节 {index:02d}",
            "start": chapter.get("start") or "00:00:00",
            "end": chapter.get("end") or "",
            "start_sec": timestamp_to_seconds(chapter.get("start")),
            "end_sec": timestamp_to_seconds(chapter.get("end")),
        }
        for index, chapter in enumerate(chapters, start=1)
    ]


def build_evidence(
    run_dir: Path,
    analysis: dict[str, Any],
    transcript: dict[str, Any],
    frames: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    segments = transcript.get("segments") or []
    for index, segment in enumerate(segments):
        start = float_or_zero(segment.get("start", segment.get("start_time")))
        text = clean_text(segment.get("text"))
        if not text:
            continue
        evidence.append(
            evidence_item(
                source_type="asr",
                index=index,
                timestamp_sec=start,
                text=text,
                confidence=0.75,
                extra={"end_sec": float_or_zero(segment.get("end", segment.get("end_time")))},
            )
        )

    for index, event in enumerate(load_ocr_events(run_dir, analysis)):
        frame_number = int(event.get("frame_number", index))
        frame = frames.get(frame_number, {})
        timestamp = float_or_zero(event.get("timestamp", frame.get("timestamp_sec")))
        status = str(event.get("status") or "").lower()
        text = clean_text(event.get("text"))
        confidence = 0.9 if status in {"ok", "succeeded", "success"} and text else 0.35
        evidence.append(
            evidence_item(
                source_type="ocr",
                index=index,
                timestamp_sec=timestamp,
                text=text,
                confidence=confidence,
                frame=frame,
                extra={"frame_number": frame_number, "status": status or "unknown", "error": event.get("error")},
            )
        )

    for index, event in enumerate(load_frame_analyses(run_dir, analysis)):
        frame_number = int(event.get("frame_number", index))
        frame = frames.get(frame_number, {})
        timestamp = float_or_zero(event.get("timestamp", frame.get("timestamp_sec")))
        text = clean_text(event.get("response") or event.get("text"))
        has_error = bool(event.get("error")) or "error analyzing frame" in text.lower()
        evidence.append(
            evidence_item(
                source_type="vl",
                index=index,
                timestamp_sec=timestamp,
                text=text,
                confidence=0.85 if text and not has_error else 0.25,
                frame=frame,
                extra={"frame_number": frame_number, "status": "error" if has_error else "ok", "error": event.get("error")},
            )
        )

    manual = read_text_if_exists(run_dir / "operation_manual.md") or read_text_if_exists(run_dir / "operation_manual.quality_failed.md")
    if manual:
        evidence.append(
            evidence_item(
                source_type="manual",
                index=0,
                timestamp_sec=0.0,
                text=trim(manual, 1200),
                confidence=0.7,
                extra={"path": "operation_manual.md"},
            )
        )
    page_context = read_text_if_exists(run_dir / "input_page_context.md") or read_text_if_exists(run_dir.parent / "page_context.md")
    if page_context:
        evidence.append(
            evidence_item(
                source_type="page_context",
                index=0,
                timestamp_sec=0.0,
                text=trim(page_context, 1000),
                confidence=0.45,
                extra={"path": "input_page_context.md"},
            )
        )
    return sorted(evidence, key=lambda item: (item["timestamp_sec"], item["id"]))


def evidence_item(
    source_type: str,
    index: int,
    timestamp_sec: float,
    text: str,
    confidence: float,
    frame: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame = frame or {}
    item = {
        "id": f"{source_type}_{index:04d}",
        "source_type": source_type,
        "timestamp_sec": round(float_or_zero(timestamp_sec), 3),
        "timestamp_label": format_timestamp(timestamp_sec),
        "confidence": confidence,
        "text": text,
        "jump_target": {"timestamp_sec": round(float_or_zero(timestamp_sec), 3)},
    }
    if frame.get("path"):
        item["frame_path"] = frame["path"]
        item["frame_exists"] = bool(frame.get("exists"))
    if extra:
        item.update({key: value for key, value in extra.items() if value is not None})
    return item


def build_chapter_packets(
    run_dir: Path,
    chapters: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    frames: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    packets = []
    for chapter in chapters:
        start = chapter["start_sec"]
        end = chapter["end_sec"] if chapter["end_sec"] > start else float("inf")
        chapter_evidence = [item for item in evidence if start <= item["timestamp_sec"] < end or item["source_type"] == "manual"]
        core_evidence = [item for item in chapter_evidence if item["source_type"] in CORE_SOURCE_TYPES and item.get("text")]
        learning_evidence = chapter_learning_evidence(core_evidence)
        representative = choose_representative_frame(start, end, frames)
        packet = {
            **chapter,
            "summary": summarize_chapter(learning_evidence),
            "key_points": key_points_from_evidence(learning_evidence),
            "representative_frame": representative,
            "evidence_ids": [item["id"] for item in chapter_evidence[:30]],
            "evidence": chapter_evidence[:30],
            "_synthesis_evidence": chapter_evidence,
            "review_flags": [],
        }
        packets.append(packet)
    return packets


def synthesize_study_cards(
    packets: list[dict[str, Any]],
    language: str,
    study_card_llm_base_url: str | None = None,
    study_card_model: str | None = None,
    study_card_temperature: float = DEFAULT_STUDY_CARD_TEMPERATURE,
    study_card_api_key_env: str | None = None,
    study_card_extra_body: dict[str, Any] | None = None,
    study_card_client: Any | None = None,
    fallback_llm_base_url: str | None = None,
    fallback_model: str | None = None,
    fallback_temperature: float = 0.2,
    fallback_api_key_env: str | None = None,
    fallback_extra_body: dict[str, Any] | None = None,
    fallback_client: Any | None = None,
) -> None:
    primary_error = ""
    primary_client = None
    fallback_llm_client = None
    use_primary = bool(study_card_model and (study_card_client or study_card_llm_base_url))
    use_fallback = bool(
        fallback_model
        and (fallback_client or fallback_llm_base_url)
        and not same_model_endpoint(study_card_model, study_card_llm_base_url, fallback_model, fallback_llm_base_url)
    )

    if use_primary and not study_card_client:
        try:
            primary_client = build_llm_client(
                study_card_llm_base_url,
                api_key_env=study_card_api_key_env,
                extra_body=study_card_extra_body,
            )
        except Exception as exc:
            primary_error = str(exc)
            use_primary = False
    else:
        primary_client = study_card_client

    if use_fallback and not fallback_client:
        try:
            fallback_llm_client = build_llm_client(
                fallback_llm_base_url,
                api_key_env=fallback_api_key_env,
                extra_body=fallback_extra_body,
            )
        except Exception:
            fallback_llm_client = None
            use_fallback = False
    else:
        fallback_llm_client = fallback_client

    for packet in packets:
        packet["source_title"] = packet.get("source_title") or packet.get("title") or ""
        generated = None
        errors: list[str] = []
        if use_primary and primary_client and study_card_model:
            try:
                generated = synthesize_one_study_card(
                    packet,
                    client=primary_client,
                    model=study_card_model,
                    temperature=study_card_temperature,
                    language=language,
                )
                apply_generated_study_card(packet, generated, status="generated", model=study_card_model)
                continue
            except Exception as exc:
                errors.append(f"primary {study_card_model}: {exc}")
        elif primary_error:
            errors.append(f"primary unavailable: {primary_error}")

        if errors and use_fallback and fallback_llm_client and fallback_model:
            try:
                generated = synthesize_one_study_card(
                    packet,
                    client=fallback_llm_client,
                    model=fallback_model,
                    temperature=fallback_temperature,
                    language=language,
                )
                apply_generated_study_card(
                    packet,
                    generated,
                    status="fallback_model",
                    model=fallback_model,
                    reason="; ".join(errors),
                )
                continue
            except Exception as exc:
                errors.append(f"fallback {fallback_model}: {exc}")

        packet["card_synthesis"] = {
            "status": "deterministic",
            "model": None,
            "reason": "; ".join(errors) if errors else "study card model not configured",
        }


def build_llm_client(
    llm_base_url: str | None,
    api_key_env: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> GenericOpenAIAPIClient:
    if not llm_base_url:
        raise ValueError("llm base url unavailable")
    return GenericOpenAIAPIClient(
        resolve_api_key(api_key_env=api_key_env, api_url=llm_base_url),
        llm_base_url,
        max_retries=1,
        timeout_seconds=180,
        extra_body=extra_body,
    )


def same_model_endpoint(
    left_model: str | None,
    left_base_url: str | None,
    right_model: str | None,
    right_base_url: str | None,
) -> bool:
    return bool(left_model and right_model and left_base_url and right_base_url and left_model == right_model and left_base_url == right_base_url)


def synthesize_one_study_card(
    packet: dict[str, Any],
    client: Any,
    model: str,
    temperature: float,
    language: str,
) -> dict[str, Any]:
    prompt = build_study_card_prompt(packet, language)
    response = client.generate(prompt=prompt, model=model, temperature=temperature, num_predict=900)
    text = clean_text(response.get("response"))
    parsed = parse_json_object(text)
    if not parsed:
        raise ValueError("model did not return a JSON object")
    return normalize_generated_study_card(parsed, packet)


def build_study_card_prompt(packet: dict[str, Any], language: str) -> str:
    synthesis_evidence = packet.get("_synthesis_evidence") or packet.get("evidence") or []
    transcript_items = [
        item
        for item in synthesis_evidence
        if item.get("source_type") in {"asr", "subtitle"} and clean_text(item.get("text"))
    ]
    context_items = [
        item
        for item in synthesis_evidence
        if item.get("source_type") in {"ocr", "vl"} and clean_text(item.get("text"))
    ]
    transcript_text = "\n".join(
        f"[{item.get('timestamp_label')}] {clean_text(item.get('text'))}" for item in transcript_items
    )
    context_text = "\n".join(
        f"[{item.get('timestamp_label')}] {item.get('source_type')}: {clean_text(item.get('text'))}"
        for item in context_items
    )
    payload = {
        "chapter_id": packet.get("chapter_id"),
        "time": f"{packet.get('start', '')} - {packet.get('end', '')}",
        "source_title": packet.get("source_title") or packet.get("title"),
        "deterministic_summary": packet.get("summary"),
        "transcript": trim(transcript_text, STUDY_CARD_MAX_TRANSCRIPT_CHARS),
        "visual_context": trim(context_text, STUDY_CARD_MAX_CONTEXT_CHARS),
    }
    return f"""/no_think
你是视频访谈学习卡编辑。请用 {language} 基于整章转写生成一个章节学习卡。

只输出 JSON 对象，字段固定：
- title: string，整章主题短标题，8-22 个中文字符左右；不要复制开头句、语气词、时间残片或问句残片。
- summary: string，2-3 句自然摘要，概括整章讨论脉络和结论。
- key_points: string[]，3-5 条归纳后的学习要点，不要连续摘抄原文。

硬规则：
- 必须综合整章 ASR/字幕，不得只看第一句话。
- 标题不能是“他说”“呃”“过去九个月”“我会先写”这类低信息片段。
- 如果是访谈内容，优先概括讨论主题、观点变化和可复用启发。
- OCR/VL 只是补充上下文；没有时不要编造画面信息。

章节材料：
{json.dumps(payload, ensure_ascii=False)}
""".strip()


def normalize_generated_study_card(parsed: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    title = normalize_card_text(parsed.get("title"), max_chars=40)
    summary = normalize_card_text(parsed.get("summary"), max_chars=420)
    key_points = normalize_key_points(parsed.get("key_points"))
    if is_low_information_title(title, packet):
        raise ValueError(f"low-information title: {title}")
    if len(summary) < 16:
        raise ValueError("summary is too short")
    if len(key_points) < 2:
        raise ValueError("key_points must contain at least two useful items")
    return {
        "title": title,
        "summary": summary,
        "key_points": key_points[:5],
    }


def normalize_card_text(value: Any, max_chars: int) -> str:
    if isinstance(value, list):
        value = "；".join(str(item) for item in value if clean_text(item))
    text = clean_text(value).strip("`#* \t\r\n")
    text = re.sub(r"^\d{1,2}[.、]\s*", "", text).strip()
    return sentence_safe_trim(text, max_chars)


def normalize_key_points(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"\n+|[；;]", str(value or ""))
    points: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = normalize_card_text(item, max_chars=180)
        text = re.sub(r"^[-*•]\s*", "", text).strip()
        normalized = re.sub(r"\W+", "", text).lower()
        if len(text) < 6 or not normalized or normalized in seen:
            continue
        points.append(text)
        seen.add(normalized)
    return points


def is_low_information_title(title: str, packet: dict[str, Any]) -> bool:
    title = clean_text(title).strip(" ，,；;。.!！?？")
    if not title:
        return True
    compact = re.sub(r"\W+", "", title).lower()
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", title))
    if cjk_count <= 2 and len(compact) <= 6:
        return True
    if compact in {"嗯", "呃", "啊", "哦", "他说", "我说", "你说", "就是", "然后", "没有"}:
        return True
    vague_prefixes = ("过去", "现在", "明年", "本来", "那你", "那个", "这个", "就是", "然后", "我会先", "没有去")
    if len(title) <= 10 and title.startswith(vague_prefixes):
        return True
    source_title = clean_text(packet.get("source_title"))
    if source_title and title == source_title and len(title) <= 12 and title.startswith(vague_prefixes):
        return True
    return False


def apply_generated_study_card(
    packet: dict[str, Any],
    generated: dict[str, Any],
    status: str,
    model: str,
    reason: str | None = None,
) -> None:
    packet["title"] = generated["title"]
    packet["summary"] = generated["summary"]
    packet["key_points"] = generated["key_points"]
    packet["card_synthesis"] = {
        "status": status,
        "model": model,
    }
    if reason:
        packet["card_synthesis"]["reason"] = reason


def choose_representative_frame(start: float, end: float, frames: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    available = [frame for frame in frames.values() if frame.get("exists") and start <= frame.get("timestamp_sec", 0) < end]
    if not available and frames:
        available = [frame for frame in frames.values() if frame.get("exists")]
    if not available:
        return None
    target = start if end == float("inf") else (start + end) / 2
    return min(available, key=lambda frame: abs(frame.get("timestamp_sec", 0) - target))


def detect_evidence_gaps(
    run_dir: Path,
    analysis: dict[str, Any],
    transcript: dict[str, Any],
    frames: dict[int, dict[str, Any]],
    evidence: list[dict[str, Any]],
    chapter_packets: list[dict[str, Any]],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    audio_only = is_audio_only_run(run_dir)

    if not frames and not audio_only:
        add_gap(items, "frames_manifest_missing", "error", "frames_manifest.json 不存在或为空", None)
    for frame in frames.values():
        if not frame.get("exists"):
            add_gap(items, "frame_missing", "error", f"帧文件不存在：{frame.get('path')}", frame)

    transcript_segments = transcript.get("segments") or []
    transcript_md = read_text_if_exists(run_dir / "transcript.md") or read_text_if_exists(run_dir / "orin" / "transcript.md")
    if not transcript_segments and not clean_text(transcript_md):
        add_gap(items, "asr_empty", "error", "ASR transcript 为空或缺少 segments", None)

    ocr_events = load_ocr_events(run_dir, analysis)
    if not ocr_events and not audio_only:
        add_gap(items, "ocr_empty", "warning", "OCR events 为空", None)
    for index, event in enumerate(ocr_events):
        text = clean_text(event.get("text"))
        status = str(event.get("status") or "").lower()
        frame = frames.get(int(event.get("frame_number", index)))
        if status and status not in {"ok", "success", "succeeded"}:
            add_gap(items, "ocr_failed", "warning", f"OCR 失败：{event.get('error') or status}", frame, evidence_id=f"ocr_{index:04d}")
        elif not text:
            add_gap(items, "ocr_text_empty", "info", "OCR 文本为空", frame, evidence_id=f"ocr_{index:04d}")

    frame_analyses = load_frame_analyses(run_dir, analysis)
    if not frame_analyses and not audio_only:
        add_gap(items, "vl_empty", "warning", "VL frame analyses 为空", None)
    for index, event in enumerate(frame_analyses):
        text = clean_text(event.get("response") or event.get("text"))
        frame = frames.get(int(event.get("frame_number", index)))
        if event.get("error") or "error analyzing frame" in text.lower():
            add_gap(items, "vl_failed", "error", f"VL 失败：{event.get('error') or trim(text, 240)}", frame, evidence_id=f"vl_{index:04d}")
        elif not text:
            add_gap(items, "vl_response_empty", "warning", "VL response 为空", frame, evidence_id=f"vl_{index:04d}")

    for packet in chapter_packets:
        has_core = any(item.get("source_type") in CORE_SOURCE_TYPES and clean_text(item.get("text")) for item in packet.get("evidence") or [])
        if not has_core:
            add_gap(items, "chapter_core_evidence_missing", "error", f"章节无核心证据：{packet['title']}", None, chapter_id=packet["chapter_id"])

    for review_path in (
        run_dir / "docs_analysis_chapters" / "deep_report_v2.review.md",
        run_dir / "docs_analysis" / "orin" / "round_04_review.md",
        run_dir / "docs_analysis" / "operation_manual_review.md",
    ):
        review_text = read_text_if_exists(review_path)
        if re.search(r"(不能发布|不建议直接发布|不建议发布|停止发布|阻止发布)", review_text):
            add_gap(
                items,
                "prior_review_blocks_publish",
                "error",
                f"既有复核文档建议阻止发布：{review_path.relative_to(run_dir)}",
                None,
            )
            break

    return {
        "version": 1,
        "items": items,
        "summary": {
            "total": len(items),
            "errors": sum(1 for item in items if item["severity"] == "error"),
            "warnings": sum(1 for item in items if item["severity"] == "warning"),
            "infos": sum(1 for item in items if item["severity"] == "info"),
        },
    }


def add_gap(
    items: list[dict[str, Any]],
    category: str,
    severity: str,
    message: str,
    frame: dict[str, Any] | None,
    evidence_id: str | None = None,
    chapter_id: str | None = None,
) -> None:
    gap = {
        "id": f"gap_{len(items) + 1:04d}",
        "category": category,
        "severity": severity,
        "message": message,
    }
    if frame:
        gap["timestamp_sec"] = frame.get("timestamp_sec")
        gap["timestamp_label"] = frame.get("timestamp_label")
        gap["frame_path"] = frame.get("path")
    if evidence_id:
        gap["evidence_id"] = evidence_id
    if chapter_id:
        gap["chapter_id"] = chapter_id
    items.append(gap)


def build_study_guide(
    run_dir: Path,
    analysis: dict[str, Any],
    evidence: list[dict[str, Any]],
    chapter_packets: list[dict[str, Any]],
    gaps: dict[str, Any],
    language: str,
) -> dict[str, Any]:
    metadata = analysis.get("metadata") or {}
    title = extract_title(read_text_if_exists(run_dir.parent / "page_context.md") or read_text_if_exists(run_dir / "input_page_context.md"))
    return {
        "version": 1,
        "language": language,
        "title": title or run_dir.parent.name,
        "run_dir": str(run_dir),
        "duration_sec": metadata.get("duration_processed"),
        "overview": {
            "summary": summarize_video(chapter_packets),
            "chapter_count": len(chapter_packets),
            "evidence_count": len(evidence),
            "gap_count": gaps.get("summary", {}).get("total", 0),
        },
        "chapters": public_chapter_packets(chapter_packets),
        "evidence": evidence,
        "evidence_gaps": gaps,
    }


def build_evidence_review(
    run_dir: Path,
    guide: dict[str, Any],
    gaps: dict[str, Any],
    chapter_packets: list[dict[str, Any]],
    language: str,
    llm_base_url: str | None,
    text_model: str,
    temperature: float,
    api_key_env: str | None,
    extra_body: dict[str, Any] | None,
    skip_review: bool,
    client: Any | None,
) -> dict[str, Any]:
    if skip_review:
        return {"status": "skipped", "reason": "skip-review requested", "risk_level": deterministic_risk(gaps)}
    if not llm_base_url:
        return {"status": "skipped", "reason": "llm base url unavailable", "risk_level": deterministic_risk(gaps)}
    try:
        client = client or GenericOpenAIAPIClient(
            resolve_api_key(api_key_env=api_key_env, api_url=llm_base_url),
            llm_base_url,
            extra_body=extra_body,
        )
        prompt = build_review_prompt(guide, gaps, chapter_packets, language)
        response = client.generate(prompt=prompt, model=text_model, temperature=temperature, num_predict=3000)
        text = clean_text(response.get("response"))
        parsed = parse_json_object(text)
        if parsed:
            parsed.setdefault("status", "reviewed")
            parsed.setdefault("raw_text", text)
            return parsed
        return {"status": "reviewed", "risk_level": deterministic_risk(gaps), "raw_text": text}
    except Exception as exc:
        return {"status": "failed", "reason": str(exc), "risk_level": deterministic_risk(gaps)}


def build_review_prompt(
    guide: dict[str, Any],
    gaps: dict[str, Any],
    chapter_packets: list[dict[str, Any]],
    language: str,
) -> str:
    compact_chapters = [
        {
            "chapter_id": packet["chapter_id"],
            "title": packet["title"],
            "time": f"{packet['start']} - {packet['end']}",
            "summary": packet["summary"],
            "evidence_ids": packet["evidence_ids"][:12],
        }
        for packet in chapter_packets
    ]
    run_dir = Path(guide.get("run_dir") or "")
    draft_docs = {}
    if run_dir:
        for name, rel_path in {
            "operation_manual": "operation_manual.md",
            "knowledge_notes_v2": "docs_analysis_chapters/knowledge_notes_v2.md",
            "deep_report_v2": "docs_analysis_chapters/deep_report_v2.md",
            "deep_report_review": "docs_analysis_chapters/deep_report_v2.review.md",
            "operation_manual_review": "docs_analysis/operation_manual_review.md",
        }.items():
            draft_docs[name] = trim(read_text_if_exists(run_dir / rel_path), 1800)
    return f"""/no_think
你是视频学习资料的证据复核员。请用 {language} 复核证据缺口对学习资料和最终发布的影响。

只输出 JSON，字段固定：
- risk_level: low | medium | high
- publish_recommendation: publishable | publish_with_warnings | blocked
- affected_chapters: string[]
- conclusion_downgrades: string[]
- recommended_actions: string[]
- notes: string

硬规则事实：
{json.dumps(gaps, ensure_ascii=False)[:9000]}

章节摘要：
{json.dumps(compact_chapters, ensure_ascii=False)[:9000]}

草稿与既有复核摘录：
{json.dumps(draft_docs, ensure_ascii=False)[:9000]}

全局概览：
{json.dumps(guide.get('overview') or {}, ensure_ascii=False)}
""".strip()


def build_publish_decision(gaps: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    errors = [item for item in gaps.get("items") or [] if item.get("severity") == "error"]
    warnings = [item for item in gaps.get("items") or [] if item.get("severity") == "warning"]
    hard_block_categories = {"asr_empty", "frame_missing", "vl_failed", "chapter_core_evidence_missing", "prior_review_blocks_publish"}
    hard_block = any(item.get("category") in hard_block_categories for item in errors)
    model_recommendation = review.get("publish_recommendation")
    if hard_block:
        status = "blocked"
        reason = "硬规则发现关键证据缺口"
    elif model_recommendation in {"blocked", "publish_with_warnings", "publishable"}:
        status = model_recommendation
        reason = "模型复核建议"
    elif errors:
        status = "blocked"
        reason = "存在 error 级证据缺口"
    elif warnings:
        status = "publish_with_warnings"
        reason = "存在 warning 级证据缺口"
    else:
        status = "publishable"
        reason = "未发现阻断级证据缺口"
    return {
        "version": 1,
        "status": status,
        "reason": reason,
        "hard_rule_priority": True,
        "gap_summary": gaps.get("summary") or {},
        "review_status": review.get("status"),
        "risk_level": review.get("risk_level") or deterministic_risk(gaps),
        "blocked_by": [item["id"] for item in errors if item.get("category") in hard_block_categories],
    }


def write_chapter_packets(run_dir: Path, packets: list[dict[str, Any]]) -> None:
    chapter_dir = run_dir / "study_chapters"
    chapter_dir.mkdir(exist_ok=True)
    for packet in packets:
        write_json(chapter_dir / f"{packet['chapter_id']}.json", public_chapter_packet(packet))


def write_study_markdown(run_dir: Path, guide: dict[str, Any], packets: list[dict[str, Any]], gaps: dict[str, Any]) -> None:
    (run_dir / "study_overview.md").write_text(render_study_overview(guide, packets, gaps), encoding="utf-8")
    (run_dir / "study_cards.md").write_text(render_study_cards(packets), encoding="utf-8")
    (run_dir / "evidence_index.md").write_text(render_evidence_index(guide.get("evidence") or []), encoding="utf-8")


def public_chapter_packets(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [public_chapter_packet(packet) for packet in packets]


def public_chapter_packet(packet: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in packet.items() if not key.startswith("_")}


def render_study_overview(guide: dict[str, Any], packets: list[dict[str, Any]], gaps: dict[str, Any]) -> str:
    lines = [
        f"# {guide.get('title') or '视频学习总览'}",
        "",
        guide.get("overview", {}).get("summary") or "本学习总览基于视频转写、OCR、视觉分析和手册证据生成。",
        "",
        "## 全片结构",
        "",
        "```mermaid",
        "flowchart TD",
    ]
    for packet in packets:
        lines.append(f"    {packet['chapter_id']}[{packet['index']:02d}. {safe_mermaid_label(packet['title'])}]")
    for left, right in zip(packets, packets[1:]):
        lines.append(f"    {left['chapter_id']} --> {right['chapter_id']}")
    lines.extend(["```", "", "## 章节地图", ""])
    for packet in packets:
        lines.extend(
            [
                f"### {packet['index']:02d}. {packet['title']}",
                "",
                f"- 时间：{packet['start']} - {packet['end']}",
                f"- 摘要：{packet['summary']}",
                f"- 可跳转时间：{packet['start']}",
                "",
            ]
        )
    lines.extend(["## 证据缺口", ""])
    if gaps.get("items"):
        for item in gaps["items"][:30]:
            lines.append(f"- `{item['severity']}` {item['message']}")
    else:
        lines.append("- 未发现证据缺口。")
    return "\n".join(lines).rstrip() + "\n"


def render_study_cards(packets: list[dict[str, Any]]) -> str:
    lines = ["# 章节学习卡片", ""]
    for packet in packets:
        lines.extend(
            [
                f"## {packet['index']:02d}. {packet['title']}",
                "",
                f"- 时间：{packet['start']} - {packet['end']}",
                f"- 摘要：{packet['summary']}",
                "- 要点：",
            ]
        )
        for point in packet.get("key_points") or []:
            lines.append(f"  - {point}")
        frame = packet.get("representative_frame")
        if frame:
            lines.extend(["", f"![{packet['title']}]({frame['path']})"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_evidence_index(evidence: list[dict[str, Any]]) -> str:
    lines = ["# 证据索引", "", "| 时间 | 类型 | 置信度 | 证据 |", "| --- | --- | --- | --- |"]
    for item in evidence:
        text = clean_text(item.get("text")).replace("|", "\\|")
        lines.append(f"| {item['timestamp_label']} | {item['source_type']} | {item['confidence']} | {trim(text, 160)} |")
    return "\n".join(lines).rstrip() + "\n"


def render_review_notes(review: dict[str, Any], gaps: dict[str, Any]) -> str:
    lines = ["# 证据复核说明", "", f"- 复核状态：{review.get('status')}", f"- 风险等级：{review.get('risk_level') or deterministic_risk(gaps)}", ""]
    if review.get("notes"):
        lines.extend(["## 复核摘要", "", str(review["notes"]).strip(), ""])
    for key, title in (("conclusion_downgrades", "需降级结论"), ("recommended_actions", "建议补采/修正")):
        values = review.get(key) or []
        if values:
            lines.extend([f"## {title}", ""])
            lines.extend(f"- {value}" for value in values)
            lines.append("")
    lines.extend(["## 硬规则缺口", ""])
    for item in gaps.get("items") or []:
        lines.append(f"- `{item['severity']}` {item['message']}")
    return "\n".join(lines).rstrip() + "\n"


def summarize_video(chapter_packets: list[dict[str, Any]]) -> str:
    if not chapter_packets:
        return "未能生成章节结构。"
    titles = "、".join(packet["title"] for packet in chapter_packets[:6])
    return f"全片分为 {len(chapter_packets)} 个学习章节，核心路径包括：{titles}。"


def chapter_learning_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timed = [item for item in evidence if item.get("source_type") in LEARNING_SOURCE_TYPES and clean_text(item.get("text"))]
    if timed:
        return sorted(timed, key=learning_evidence_sort_key)
    return [item for item in evidence if clean_text(item.get("text"))]


def learning_evidence_sort_key(item: dict[str, Any]) -> tuple[int, float, str]:
    priority = {"asr": 0, "subtitle": 0, "ocr": 1, "vl": 2}
    return (priority.get(str(item.get("source_type")), 9), float_or_zero(item.get("timestamp_sec")), str(item.get("id") or ""))


def summarize_chapter(evidence: list[dict[str, Any]]) -> str:
    units = learning_units_from_evidence(evidence, max_units=3, max_chars=420)
    if units:
        return "；".join(units)
    return "本章暂无可用核心证据。"


def key_points_from_evidence(evidence: list[dict[str, Any]]) -> list[str]:
    return learning_units_from_evidence(evidence, max_units=5, max_chars=760)


def learning_units_from_evidence(evidence: list[dict[str, Any]], max_units: int, max_chars: int) -> list[str]:
    units: list[str] = []
    seen: set[str] = set()
    used_chars = 0
    for item in evidence:
        for unit in split_learning_units(clean_text(item.get("text"))):
            normalized = re.sub(r"\W+", "", unit).lower()
            if not normalized or normalized in seen:
                continue
            remaining = max_chars - used_chars
            if remaining <= 0:
                return units
            clipped = sentence_safe_trim(unit, min(220, remaining))
            if not clipped:
                continue
            units.append(clipped)
            seen.add(normalized)
            used_chars += len(clipped)
            if len(units) >= max_units:
                return units
    return units


def split_learning_units(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = [part.strip(" ，,；;。.!！?？") for part in re.split(r"(?<=[。！？!?；;])\s*|\n+", text) if part.strip()]
    if len(parts) <= 1 and len(text) > 180:
        parts = [part.strip(" ，,；;。.!！?？") for part in re.split(r"[，,]\s*", text) if part.strip()]
    return [part for part in parts if len(part) >= 4]


def sentence_safe_trim(text: str, max_chars: int) -> str:
    text = clean_text(text).strip(" ，,；;。.!！?？")
    if len(text) <= max_chars:
        return text
    boundary = max(text.rfind(mark, 0, max_chars) for mark in ("。", "！", "？", "；", ";", ".", "!", "?"))
    if boundary >= max(24, max_chars // 2):
        return text[: boundary + 1].strip(" ，,；;。.!！?？")
    comma = max(text.rfind(mark, 0, max_chars) for mark in ("，", ",", "、"))
    if comma >= max(24, max_chars // 2):
        return text[:comma].strip(" ，,；;。.!！?？")
    return text[:max_chars].rstrip()


def deterministic_risk(gaps: dict[str, Any]) -> str:
    summary = gaps.get("summary") or {}
    if summary.get("errors"):
        return "high"
    if summary.get("warnings"):
        return "medium"
    return "low"


def load_ocr_events(run_dir: Path, analysis: dict[str, Any]) -> list[dict[str, Any]]:
    data = read_json_if_exists(run_dir / "orin" / "ocr_events.json")
    return data if isinstance(data, list) else list(analysis.get("ocr_events") or [])


def load_frame_analyses(run_dir: Path, analysis: dict[str, Any]) -> list[dict[str, Any]]:
    data = read_json_if_exists(run_dir / "orin" / "frame_analyses.json")
    return data if isinstance(data, list) else list(analysis.get("frame_analyses") or [])


def extract_title(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip().lstrip("#").strip()
        if line and not line.startswith(("URL", "http")):
            return line[:120]
    return ""


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fence:
        text = fence.group(1)
    else:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            text = match.group(0)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_if_exists(path: Path) -> Any:
    if not path.is_file():
        return None
    return read_json(path)


def read_text_if_exists(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def trim(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 12].rstrip() + "..."


def float_or_zero(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def format_timestamp(value: Any) -> str:
    seconds = int(float_or_zero(value))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def safe_mermaid_label(value: str) -> str:
    return re.sub(r"[\[\]{}()|]", " ", str(value or "")).strip()[:80]


if __name__ == "__main__":
    raise SystemExit(main())
