"""Prepare review data or analyze a RACE-Vote experiment profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from race_vote_core import (
    analyze_profile, apply_k, baseline_analysis, baseline_run, evaluation_task_ids,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a RACE-Vote profile")
    parser.add_argument("command", choices=("prepare-review", "analyze"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--k", type=int, choices=(6, 10), default=10)
    args = parser.parse_args()
    cfg = apply_k(baseline_run.load_config(Path(args.config)), args.k)
    if cfg["experiment_name"] in {"Ablation-No-RC", "Ablation-No-CE", "Ablation-No-RV"} and args.k != 10:
        raise SystemExit("The paper evaluates component ablations only at K=10.")
    root = Path(cfg["output_dir"])
    task_ids = evaluation_task_ids(cfg)
    result = (
        baseline_analysis.prepare_review(root, task_ids)
        if args.command == "prepare-review"
        else analyze_profile(cfg, task_ids)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
