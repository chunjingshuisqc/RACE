#!/usr/bin/env python3
"""Run TrickCatcher and RACE-Vote on the same fixed paper subjects."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from race_vote_core import (
    GLMClient, ROOT, apply_k, atomic_json, baseline_run, evaluation_tasks,
    process_task,
)
from tc_core import read_text, safe_id

BASELINE_CONFIG = ROOT / "Ablation-TrickCatcher" / "config.json"
RACE_CONFIG = ROOT / "RACE-Vote" / "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paper-protocol TrickCatcher/RACE-Vote batch"
    )
    parser.add_argument("--k", type=int, choices=(6, 10), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _status(root: Path, task_id: str) -> dict[str, Any]:
    path = root / "raw" / safe_id(task_id) / "status.json"
    return json.loads(read_text(path)) if path.is_file() else {"status": "not_run"}


def main() -> int:
    args = parse_args()
    baseline_cfg = apply_k(baseline_run.load_config(BASELINE_CONFIG), args.k)
    race_cfg = apply_k(baseline_run.load_config(RACE_CONFIG), args.k)
    if Path(baseline_cfg["dataset_path"]) != Path(race_cfg["dataset_path"]):
        raise ValueError("Both methods must use the same dataset_path")
    subjects = evaluation_tasks(baseline_cfg)
    if args.discover_only:
        print(json.dumps({
            **baseline_run.dataset_summary(Path(baseline_cfg["dataset_path"])),
            "evaluation_puts": len(subjects), "k": args.k,
        }, ensure_ascii=False, indent=2))
        return 0

    reports = {
        "TrickCatcher": baseline_run.preflight(baseline_cfg),
        "RACE-Vote": baseline_run.preflight(race_cfg),
    }
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    if args.preflight:
        return 0 if all(report["ready"] for report in reports.values()) else 2
    if not all(report["ready"] for report in reports.values()):
        return 2
    if args.limit is None or args.limit <= 0:
        raise SystemExit("A positive --limit is required to prevent an accidental full run.")

    tasks = subjects[args.start:args.start + args.limit]
    baseline_root = Path(baseline_cfg["output_dir"])
    race_root = Path(race_cfg["output_dir"])
    baseline_root.mkdir(parents=True, exist_ok=True)
    race_root.mkdir(parents=True, exist_ok=True)
    baseline_run.create_manifest(baseline_root, baseline_cfg, reports["TrickCatcher"])
    baseline_run.create_manifest(race_root, race_cfg, reports["RACE-Vote"])
    subject_file = baseline_root.parent / "subjects.json"
    atomic_json(subject_file, {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": "all Python PUTs in TrickyBugs",
        "k": args.k, "subject_count": len(subjects),
        "task_ids": [task.task_id for task in subjects],
    })

    for phase, cfg, output_root in (
        ("TrickCatcher", baseline_cfg, baseline_root),
        ("RACE-Vote", race_cfg, race_root),
    ):
        print(f"=== {phase}, K={args.k} ({len(tasks)} PUTs) ===", flush=True)
        model = GLMClient(cfg)
        for index, task in enumerate(tasks, 1):
            print(f"[{index}/{len(tasks)}] {task.task_id}", flush=True)
            try:
                print("  " + process_task(
                    task, model, output_root, cfg, args.overwrite
                ), flush=True)
            except Exception as exc:
                print(f"  error: {type(exc).__name__}: {exc}", flush=True)

    print(json.dumps({
        "k": args.k, "batch_start": args.start, "batch_size": len(tasks),
        "evaluation_puts": len(subjects), "subject_file": str(subject_file),
        "batch_status": {
            method: {
                status: sum(
                    _status(root, task.task_id).get("status") == status for task in tasks
                )
                for status in ("complete", "generator_failed", "error", "not_run")
            }
            for method, root in (("TrickCatcher", baseline_root), ("RACE-Vote", race_root))
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
