#!/usr/bin/env python3
"""Discover compatible GPUs that are free after managed services are unloaded."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    memory_total_mib: int
    memory_free_mib: int


def _run_nvidia_smi(query: str) -> str:
    binary = os.environ.get("NVIDIA_SMI_BIN", "nvidia-smi")
    result = subprocess.run(
        [
            binary,
            f"--query-{query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def parse_gpus(output: str) -> list[GPU]:
    gpus: list[GPU] = []
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        fields = [item.strip() for item in raw_line.split(",", 4)]
        if len(fields) != 5:
            raise ValueError(f"unexpected nvidia-smi GPU row: {raw_line}")
        gpus.append(
            GPU(
                index=int(fields[0]),
                uuid=fields[1],
                name=fields[2],
                memory_total_mib=int(fields[3]),
                memory_free_mib=int(fields[4]),
            )
        )
    return gpus


def parse_compute_processes(output: str) -> dict[str, list[dict[str, object]]]:
    by_uuid: dict[str, list[dict[str, object]]] = {}
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        fields = [item.strip() for item in raw_line.split(",", 3)]
        if len(fields) != 4:
            raise ValueError(f"unexpected nvidia-smi process row: {raw_line}")
        used_memory = None if fields[3] in {"", "N/A", "[N/A]"} else int(fields[3])
        by_uuid.setdefault(fields[0], []).append(
            {
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_memory_mib": used_memory,
            }
        )
    return by_uuid


def select_gpus(
    gpus: list[GPU],
    processes: dict[str, list[dict[str, object]]],
    *,
    allowed_names: list[str],
    min_total_mib: int,
    min_free_mib: int,
    max_count: int | None,
) -> dict[str, object]:
    selected: list[GPU] = []
    skipped: list[dict[str, object]] = []
    allowed = [item.casefold() for item in allowed_names]
    for gpu in sorted(gpus, key=lambda item: item.index):
        reasons = []
        gpu_processes = processes.get(gpu.uuid, [])
        if allowed and not any(item in gpu.name.casefold() for item in allowed):
            reasons.append("incompatible_model")
        if gpu.memory_total_mib < min_total_mib:
            reasons.append("insufficient_total_memory")
        if gpu_processes:
            reasons.append("compute_process_present")
        if gpu.memory_free_mib < min_free_mib:
            reasons.append("insufficient_free_memory")
        if reasons:
            skipped.append(
                {
                    **asdict(gpu),
                    "reasons": reasons,
                    "processes": gpu_processes,
                }
            )
            continue
        if max_count is not None and len(selected) >= max_count:
            skipped.append({**asdict(gpu), "reasons": ["worker_limit"], "processes": []})
            continue
        selected.append(gpu)
    return {
        "selected": [asdict(gpu) for gpu in selected],
        "selected_gpu_ids": [gpu.index for gpu in selected],
        "worker_count": len(selected),
        "skipped": skipped,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allowed-name",
        action="append",
        dest="allowed_names",
        help="Case-insensitive GPU model substring. Repeat for multiple models.",
    )
    parser.add_argument("--min-total-mib", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, required=True)
    parser.add_argument("--max-count", default="auto")
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=float(os.environ.get("VIDEO_ANALYZER_GPU_SETTLE_SECONDS", "2")),
    )
    parser.add_argument("--format", choices=["json", "csv", "count"], default="json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    allowed_names = args.allowed_names or ["Tesla P40", "Tesla V100"]
    max_count = None if str(args.max_count).lower() == "auto" else int(args.max_count)
    try:
        time.sleep(max(0.0, args.settle_seconds))
        gpus = parse_gpus(
            _run_nvidia_smi("gpu=index,uuid,name,memory.total,memory.free")
        )
        try:
            process_output = _run_nvidia_smi(
                "compute-apps=gpu_uuid,pid,process_name,used_memory"
            )
        except subprocess.CalledProcessError as exc:
            process_output = exc.stdout or ""
        payload = select_gpus(
            gpus,
            parse_compute_processes(process_output),
            allowed_names=allowed_names,
            min_total_mib=max(0, args.min_total_mib),
            min_free_mib=max(0, args.min_free_mib),
            max_count=max_count,
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"GPU discovery failed: {exc}", file=sys.stderr)
        return 1

    selected_ids = payload["selected_gpu_ids"]
    if args.format == "csv":
        print(",".join(str(item) for item in selected_ids))
    elif args.format == "count":
        print(payload["worker_count"])
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if selected_ids else 3


if __name__ == "__main__":
    raise SystemExit(main())
