"""Build the four result tables reported by the paper."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from race_vote_core import ROOT

MAIN_METHODS = ("Ablation-TrickCatcher", "RACE-Vote")
ABLATIONS = ("RACE-Vote", "Ablation-No-RC", "Ablation-No-CE", "Ablation-No-RV")


def _summary(baseline: Path, k: int, method: str) -> dict[str, Any] | None:
    path = baseline / f"K{k}" / method / "analysis_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _metric(summary: dict[str, Any] | None) -> dict[str, Any]:
    return summary.get("bug_detection_effectiveness", {}) if summary else {}


def _status(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "missing"
    return "ready" if summary.get("final_metrics_ready") else "provisional"


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def _detection_row(method: str, k: int, summary: dict[str, Any] | None) -> dict[str, Any]:
    metric = _metric(summary)
    scope = summary.get("experiment_scope", {}) if summary else {}
    return {
        "method": method, "k_cap": k, "status": _status(summary),
        "evaluation_puts": scope.get("evaluation_puts", ""),
        "actual_k_min": metric.get("actual_k_min", ""),
        "actual_k_max": metric.get("actual_k_max", ""),
        "actual_k_mean": metric.get("actual_k_mean", ""),
        "tp_puts": metric.get("tp_puts", ""), "fn_puts": metric.get("fn_puts", ""),
        "tp_records": metric.get("tp_records", ""), "fp_records": metric.get("fp_records", ""),
        "precision": metric.get("precision", ""), "recall": metric.get("recall", ""),
        "f1": metric.get("f1", ""),
    }


def _plot_scores(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    ready = [row for row in rows if row["status"] != "missing"]
    if not ready:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x = list(range(len(ready))); width = 0.24
    fig, ax = plt.subplots(figsize=(max(7, len(ready) * 1.5), 4.6))
    for offset, metric in ((-width, "precision"), (0, "recall"), (width, "f1")):
        ax.bar([i + offset for i in x], [float(row.get(metric) or 0) for row in ready],
               width, label=metric.title())
    ax.set(title=title, ylabel="Score", ylim=(0, 1.05), xticks=x,
           xticklabels=[f"{row['method']}\nK={row['k_cap']}" for row in ready])
    ax.tick_params(axis="x", rotation=12); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=180); plt.close(fig)


def _plot_counts(path: Path, rows: list[dict[str, Any]], fields: tuple[str, str],
                 labels: tuple[str, str], title: str) -> None:
    ready = [row for row in rows if row["status"] != "missing"]
    if not ready:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x = list(range(len(ready))); width = 0.34
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for offset, field, label in ((-width / 2, fields[0], labels[0]),
                                  (width / 2, fields[1], labels[1])):
        ax.bar([i + offset for i in x], [float(row.get(field) or 0) for row in ready],
               width, label=label)
    ax.set(title=title, ylabel="Variants", xticks=x,
           xticklabels=[f"K={row['k_cap']}" for row in ready])
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def build_comparisons(baseline: Path) -> dict[str, Any]:
    output = baseline / "RACEVote-Comparison"
    output.mkdir(parents=True, exist_ok=True)

    rq1 = [_detection_row(method, k, _summary(baseline, k, method))
           for k in (6, 10) for method in MAIN_METHODS]
    rq2 = [_detection_row(method, 10, _summary(baseline, 10, method))
           for method in ABLATIONS]
    rq3 = []
    rq4 = []
    for k in (6, 10):
        summary = _summary(baseline, k, "RACE-Vote")
        quality = summary.get("variant_quality", {}) if summary else {}
        useful = summary.get("useful_buggy", {}) if summary else {}
        rq3.append({
            "method": "RACE-Vote", "k_cap": k, "status": _status(summary),
            "assessed_variants": quality.get("accepted_variants", ""),
            "observed_buggy_variants": quality.get("observed_buggy_variants", ""),
            "observed_buggy_ratio": quality.get("observed_buggy_ratio", ""),
        })
        rq4.append({
            "method": "RACE-Vote", "k_cap": k, "status": _status(summary),
            "useful_variants": useful.get("useful_variants", ""),
            "useful_buggy_variants": useful.get("useful_buggy_variants", ""),
            "useful_buggy_ratio": useful.get("useful_buggy_ratio", ""),
        })

    for name, rows in (("rq1_detection_effectiveness.csv", rq1),
                       ("rq2_component_ablation.csv", rq2),
                       ("rq3_variant_quality.csv", rq3),
                       ("rq4_useful_buggy_variants.csv", rq4)):
        _write(output / name, rows)
    _plot_scores(output / "rq1_detection_effectiveness.png", rq1,
                 "RQ1: Detection Effectiveness")
    _plot_scores(output / "rq2_component_ablation.png", rq2,
                 "RQ2: Component Ablation (K=10)")
    _plot_counts(output / "rq3_variant_quality.png", rq3,
                 ("assessed_variants", "observed_buggy_variants"),
                 ("Assessed", "Observed buggy"), "RQ3: Variant Quality")
    _plot_counts(output / "rq4_useful_buggy_variants.png", rq4,
                 ("useful_variants", "useful_buggy_variants"),
                 ("Useful", "Useful and buggy"), "RQ4: Useful Buggy Variants")

    result = {
        "rq1": "TrickCatcher vs RACE-Vote at K=6 and K=10",
        "rq2": "RACE-Vote and three component ablations at K=10",
        "rq3": "RACE-Vote retained variants on evidence-producing PUTs",
        "rq4": "observed-buggy variants that support verified TP evidence",
        "output_dir": str(output),
    }
    (output / "comparison_scope.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(build_comparisons(ROOT / "Baseline"), ensure_ascii=False, indent=2))
