"""Question answering over generated video-analysis documents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient
from video_analyzer.config import Config, build_openai_extra_body, resolve_api_key, resolve_temperature

DEFAULT_MAX_CONTEXT_CHARS = 60000
DEFAULT_TEMPERATURE = 0.2

DOC_SOURCES = [
    ("operation_manual", "operation_manual.md", "final manual; medium confidence, use for user-facing procedure summary"),
    ("transcript", "transcript.md", "timestamped ASR transcript; high value for spoken claims"),
    ("manual_evidence", "manual_evidence.md", "frame/OCR/VL evidence; highest confidence for visible UI and screenshots"),
    (
        "analysis_json",
        "analysis.json",
        "canonical structured analysis artifact; fallback when markdown sidecars are missing",
    ),
    ("page_context", "orin/page_context.md", "page metadata, description, subtitle diagnostics, selected comments"),
    ("comments", "orin/comments.md", "low-confidence community comments; FAQ/supplement only"),
    ("knowledge_notes", "docs_analysis/knowledge_notes.md", "derived multi-round notes; medium confidence"),
    ("deep_report", "docs_analysis/deep_report.md", "derived multi-round report; medium confidence"),
    ("manual_review", "docs_analysis/operation_manual_review.md", "derived manual review and missing-item hints"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions over an operation-manual run directory")
    parser.add_argument("run_dir", help="operation-manual run directory")
    parser.add_argument("question", nargs="?", help="Single question. Omit for interactive chat mode")
    parser.add_argument("--config", default="config", help="Configuration directory containing optional config.json")
    parser.add_argument("--profile", help="Runtime profile from config/default_config.json or config.json")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--text-model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config(args.config)
    profile = config.get_runtime_profile(args.profile)
    default_base_url = (config.get("endpoints") or {}).get("services", {}).get("amd_fast_base_url")
    base_url = args.llm_base_url or profile.get("llm_base_url") or default_base_url
    model = args.text_model or profile.get("text_model")
    if not model:
        raise ValueError("Runtime profile must provide text_model, or pass --text-model")
    client = GenericOpenAIAPIClient(
        resolve_api_key(
            api_key_env=profile.get("text_api_key_env") or profile.get("api_key_env"),
            api_url=base_url,
        ),
        base_url,
        extra_body=build_openai_extra_body(profile, base_url),
    )
    run_dir = Path(args.run_dir).expanduser().resolve()
    temperature = args.temperature if args.temperature is not None else resolve_temperature(profile, DEFAULT_TEMPERATURE)

    if args.question:
        print(ask_video_docs(run_dir, args.question, client, model, temperature, args.max_context_chars))
        return 0

    chat_loop(run_dir, client, model, temperature, args.max_context_chars)
    return 0


def chat_loop(run_dir: Path, client: Any, model: str, temperature: float, max_context_chars: int) -> None:
    print(f"Video docs chat: {run_dir}")
    print("Type your question. Use /exit or Ctrl-D to quit.\n")
    history: list[dict[str, str]] = []
    while True:
        try:
            question = input("你> ").strip()
        except EOFError:
            print()
            return
        if not question:
            continue
        if question in {"/exit", "/quit", "退出", "q"}:
            return
        answer = ask_video_docs(run_dir, question, client, model, temperature, max_context_chars, history)
        history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
        history[:] = history[-8:]
        print(f"\nAI> {answer}\n")


def ask_video_docs(
    run_dir: Path,
    question: str,
    client: Any,
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    history: list[dict[str, str]] | None = None,
) -> str:
    bundle = load_video_docs(run_dir, max_context_chars)
    prompt = build_doc_chat_prompt(bundle, question, history or [])
    response = client.generate(prompt=prompt, model=model, temperature=temperature, num_predict=5000)
    return (response.get("response") or "").strip()


def load_video_docs(run_dir: Path, max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> dict[str, Any]:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    sources = []
    used = 0
    budget_per_doc = max(2000, max_context_chars // max(len(DOC_SOURCES), 1))
    for name, rel_path, note in DOC_SOURCES:
        path = run_dir / rel_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        remaining = max_context_chars - used
        if remaining <= 0:
            break
        limit = min(budget_per_doc, remaining)
        sources.append(
            {
                "name": name,
                "path": str(path),
                "note": note,
                "text": trim(text, limit),
                "original_chars": len(text),
            }
        )
        used += len(sources[-1]["text"])
    if not sources:
        raise FileNotFoundError(f"No supported video-analysis documents found in: {run_dir}")
    return {"run_dir": str(run_dir), "sources": sources, "context_chars": used}


def build_doc_chat_prompt(bundle: dict[str, Any], question: str, history: list[dict[str, str]]) -> str:
    source_text = "\n\n".join(
        f"## Source: {source['name']}\nPath: {source['path']}\nEvidence note: {source['note']}\n\n{source['text']}"
        for source in bundle["sources"]
    )
    history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history[-8:])
    return f"""
你是视频资料问答助手。请只基于给定运行目录中的文档回答问题。

证据权重：
1. manual_evidence / OCR / VL / 截图证据最高。
2. transcript 带时间戳转写用于 spoken claims。
3. analysis.json 是结构化聚合产物，可用于兜底和交叉验证，但原始 manual_evidence / transcript 优先于它。
4. operation_manual 和 docs_analysis 是派生总结，可用于组织答案。
5. page_context 的简介/metadata 可补充背景。
6. comments/comment-only 信息只能作为社区补充或 FAQ，不能单独形成确定结论。

回答要求：
- 用中文回答，除非用户要求其他语言。
- 能引用来源时写出文件名或时间戳，例如 transcript.md 03:20、manual_evidence.md frame_012。
- 不确定、资料冲突或证据不足时明确说“需复核”。
- 不要凭空补充视频资料里没有的信息。
- 如果用户问操作步骤，优先使用 OCR/VL/手册证据，不要只靠评论。

运行目录：{bundle['run_dir']}

历史对话：
{history_text or '(none)'}

资料包：
{source_text}

用户问题：{question}
""".strip()


def trim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[truncated]"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
