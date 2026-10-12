"""Run one RACE-Vote or ablation profile."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from race_vote_core import (
    GLMClient, apply_k, atomic_json, baseline_run, evaluation_tasks, process_task,
    resolve_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a RACE-Vote experiment profile")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--k", type=int, choices=(6, 10), default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = apply_k(baseline_run.load_config(Path(args.config)), args.k)
    features = resolve_features(cfg)
    if cfg["experiment_name"] in {"Ablation-No-RC", "Ablation-No-CE", "Ablation-No-RV"} and args.k != 10:
        raise SystemExit("The paper evaluates component ablations only at K=10.")
    if args.discover_only:
        result = baseline_run.dataset_summary(Path(cfg["dataset_path"]))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    report = baseline_run.preflight(cfg)
    report["execution_population"] = {
        "mode": "fixed_common_subjects",
        "available_puts": len(evaluation_tasks(cfg)),
    }
    output_root = Path(cfg["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "experiment_manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": cfg["experiment_name"],
        "features": features,
        "shared_core": str(Path(__file__).resolve().parent),
        "config": cfg,
        "preflight": report,
        "evaluation_population": "fixed common subject set; failed generations remain FNs",
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.preflight:
        return 0 if report["ready"] else 2
    if not report["ready"]:
        return 2
    if args.limit is None or args.limit <= 0:
        raise SystemExit("A positive --limit is required to prevent an accidental full run.")
    subjects = evaluation_tasks(cfg)
    tasks = subjects[args.start:args.start + args.limit]
    model = GLMClient(cfg)
    for index, task in enumerate(tasks, 1):
        print(f"[{index}/{len(tasks)}] {task.task_id}", flush=True)
        try:
            print("  " + process_task(
                task, model, output_root, cfg, args.overwrite
            ), flush=True)
        except Exception as exc:
            print(f"  error: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
