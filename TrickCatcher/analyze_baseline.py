#!/usr/bin/env python3
"""Prepare manual validity review and analyze the TrickCatcher baseline."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tc_core import atomic_json, normal_output, read_jsonl, read_text


KS = (10,)


def configure_csv_field_limit() -> None:
    """Allow review CSV cells to contain very large generated inputs."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


configure_csv_field_limit()


def reuse_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "reuse_manifest.json"
    return json.loads(read_text(path)) if path.is_file() else {}


def analysis_data_directory(directory: Path) -> Path:
    source = reuse_manifest(directory).get("analysis_source_task_dir")
    return Path(str(source)) if source else directory


def generator_eligibility(directory: Path) -> dict[str, Any]:
    data_directory = analysis_data_directory(directory)
    status_path = directory / "status.json"
    status = json.loads(read_text(status_path)) if status_path.is_file() else {}
    generator_status = status.get("generator_status", {})
    detail_path = data_directory / "input_generator_status.json"
    if detail_path.is_file():
        generator_status = json.loads(read_text(detail_path))
    records: list[dict[str, Any]] = []
    records_path = data_directory / "input_generators.json"
    if records_path.is_file():
        value = json.loads(read_text(records_path))
        records = value if isinstance(value, list) else []
    executable_attempts = int(generator_status.get(
        "executable_attempts",
        sum(item.get("execution", {}).get("status") == "ok" for item in records),
    ))
    usable_attempts = int(generator_status.get(
        "usable_attempts",
        sum(
            item.get("execution", {}).get("status") == "ok"
            and int(item.get("parsed_inputs", 0)) > 0
            for item in records
        ),
    ))
    generated_inputs = int(generator_status.get(
        "generated_unique_inputs", status.get("generated_inputs", 0)
    ))
    explicit = status.get("generator_eligible", generator_status.get("generator_eligible"))
    eligible = bool(explicit) if explicit is not None else (
        status.get("status") == "complete" and usable_attempts > 0 and generated_inputs > 0
    )
    if status.get("status") == "generator_failed":
        eligible = False
    variants_path = data_directory / "accepted_variants.json"
    accepted_variants = int(status.get("accepted_variants", 0))
    if variants_path.is_file():
        value = json.loads(read_text(variants_path))
        accepted_variants = len(value) if isinstance(value, list) else accepted_variants
    required_variants = int(status.get("required_variants", 10))
    return {
        "task_id": status.get("task_id", directory.name),
        "directory": str(directory),
        "status": status.get("status", "missing"),
        "generator_eligible": eligible,
        "accepted_variants": accepted_variants,
        "required_variants": required_variants,
        "generator_attempts": int(generator_status.get("attempts", len(records))),
        "executable_attempts": executable_attempts,
        "usable_attempts": usable_attempts,
        "generated_inputs": generated_inputs,
        "input_yield": generator_status.get("input_yield", ""),
        "exclusion_reason": status.get("exclusion_reason", ""),
    }


def experiment_scope(
    root: Path, included_task_ids: set[str] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [generator_eligibility(path) for path in sorted((root / "raw").glob("*"))]
    if included_task_ids is not None:
        rows = [row for row in rows if row["task_id"] in included_task_ids]
        present = {str(row["task_id"]) for row in rows}
        rows.extend({
            "task_id": task_id, "directory": "", "status": "not_run",
            "generator_eligible": False, "accepted_variants": 0,
            "required_variants": 0, "generator_attempts": 0,
            "executable_attempts": 0, "usable_attempts": 0,
            "generated_inputs": 0, "input_yield": "", "exclusion_reason": "",
        } for task_id in sorted(included_task_ids - present))
    included = [
        row for row in rows
        if row["status"] == "complete" and row["generator_eligible"]
    ]
    excluded = [
        row for row in rows
        if row["status"] == "generator_failed"
        or (row["status"] == "complete" and not row["generator_eligible"])
    ]
    errors = [row for row in rows if row["status"] == "error"]
    summary = {
        "evaluation_puts": len(included_task_ids) if included_task_ids is not None else len(rows),
        "attempted_puts": sum(row["status"] != "not_run" for row in rows),
        "completed_puts": sum(row["status"] in {"complete", "generator_failed"} for row in rows),
        "included_generator_eligible_puts": len(included),
        "excluded_generator_puts": len(excluded),
        "not_run_puts": sum(row["status"] == "not_run" for row in rows),
        "error_puts": len(errors),
        "generator_success_rate": len(included) / len(rows) if rows else 0.0,
        "population_filter": "fixed common evaluation set; generation failures remain false negatives",
        "fixed_evaluation_set": included_task_ids is not None,
    }
    return rows, summary


def task_dirs(root: Path, included_task_ids: set[str] | None = None) -> list[Path]:
    paths: list[Path] = []
    for path in sorted((root / "raw").glob("*")):
        status = path / "status.json"
        eligibility = generator_eligibility(path)
        if (status.is_file()
                and json.loads(read_text(status)).get("status") == "complete"
                and (included_task_ids is None
                     or eligibility["task_id"] in included_task_ids)):
            paths.append(path)
    return paths


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_label(value: str | None) -> bool | None:
    normalized = (value or "").strip().lower()
    if normalized in {"1", "true", "yes", "y", "valid", "是", "合法"}:
        return True
    if normalized in {"0", "false", "no", "n", "invalid", "否", "非法"}:
        return False
    return None


def normal_variant_outputs(execution: dict[str, Any]) -> dict[int, str]:
    values: dict[int, str] = {}
    for item in execution.get("variant_results", []):
        output = normal_output(item.get("result", {}))
        if output is not None:
            values[int(item["variant_index"])] = output
    return values


def candidate_for(execution: dict[str, Any], subset: tuple[int, ...]) -> dict[str, Any] | None:
    put = normal_output(execution.get("put_result", {}))
    if put is None:
        return None
    outputs = normal_variant_outputs(execution)
    different = [outputs[index] for index in subset if index in outputs and outputs[index] != put]
    if not different:
        return None
    counts = Counter(different)
    oracle, support = counts.most_common(1)[0]
    return {"oracle": oracle, "support": support,
            "different_normal_variants": len(different), "put_output": put}


def paper_candidates_for(
    execution: dict[str, Any], subset: tuple[int, ...]
) -> list[dict[str, Any]]:
    """Return every maximum-support oracle, including ties, as in the paper code."""
    put = normal_output(execution.get("put_result", {}))
    if put is None:
        return []
    outputs = normal_variant_outputs(execution)
    counts = Counter(
        outputs[index]
        for index in subset
        if index in outputs and outputs[index] != put
    )
    if not counts:
        return []
    maximum = max(counts.values())
    return [
        {
            "oracle": output,
            "support": support,
            "different_normal_variants": sum(counts.values()),
            "put_output": put,
        }
        for output, support in counts.items()
        if support == maximum
    ]


def any_candidate(execution: dict[str, Any]) -> bool:
    put = normal_output(execution.get("put_result", {}))
    if put is None:
        return False
    return any(output != put for output in normal_variant_outputs(execution).values())


def prepare_review(
    root: Path, included_task_ids: set[str] | None = None
) -> dict[str, int]:
    review_path = root / "manual" / "input_validity.csv"
    old: dict[tuple[str, str], dict[str, str]] = {}
    if review_path.is_file():
        with review_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                old[(row["task_id"], row["input_id"])] = row
    rows: list[dict[str, Any]] = []
    source_by_stdin: dict[tuple[str, str], dict[str, str]] = {}
    loaded_source_reviews: set[Path] = set()
    for directory in task_dirs(root, included_task_ids):
        manifest = reuse_manifest(directory)
        source_dirs = {
            Path(str(value)) for key, value in manifest.items()
            if key in {"analysis_source_task_dir", "reused_stage1_source_task_dir"}
            and value
        }
        for source_dir in source_dirs:
            source_profile = source_dir.parent.parent
            source_reviews = [
                source_profile / "manual" / "input_validity.csv",
                source_profile.parent / "Ablation-TrickCatcher" / "manual" / "input_validity.csv",
            ]
            for source_review in source_reviews:
                if source_review in loaded_source_reviews or not source_review.is_file():
                    continue
                loaded_source_reviews.add(source_review)
                with source_review.open(encoding="utf-8-sig", newline="") as handle:
                    for source_row in csv.DictReader(handle):
                        try:
                            stdin = str(json.loads(source_row.get("stdin_json", '""')))
                        except json.JSONDecodeError:
                            continue
                        source_by_stdin[(source_row["task_id"], stdin)] = source_row
        data_directory = analysis_data_directory(directory)
        metadata = json.loads(read_text(data_directory / "task.json"))
        for execution in read_jsonl(data_directory / "executions.jsonl"):
            if not any_candidate(execution):
                continue
            key = (metadata["task_id"], execution["input_id"])
            previous = old.get(
                key,
                source_by_stdin.get((metadata["task_id"], str(execution["stdin"])), {}),
            )
            rows.append({
                "task_id": metadata["task_id"], "problem_id": metadata["problem_id"],
                "problem_description": str(Path(metadata["source_path"]).parents[2] / "problem_description.txt"),
                "input_id": execution["input_id"],
                "stdin_json": json.dumps(execution["stdin"], ensure_ascii=False),
                "reference_status": execution.get("reference_status"),
                "reference_output_json": json.dumps(execution.get("reference_output"), ensure_ascii=False),
                "input_valid": previous.get("input_valid", ""),
                "review_note": previous.get("review_note", ""),
            })
    fields = ["task_id", "problem_id", "problem_description", "input_id", "stdin_json", "reference_status",
              "reference_output_json", "input_valid", "review_note"]
    write_csv(review_path, rows, fields)
    labeled = sum(parse_label(str(row["input_valid"])) is not None for row in rows)
    return {"rows": len(rows), "labeled": labeled, "pending": len(rows) - labeled}


def load_labels(root: Path) -> dict[tuple[str, str], bool | None]:
    path = root / "manual" / "input_validity.csv"
    labels: dict[tuple[str, str], bool | None] = {}
    if path.is_file():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                labels[(row["task_id"], row["input_id"])] = parse_label(row.get("input_valid"))
    return labels


def classification(execution: dict[str, Any], candidate: dict[str, Any],
                   label: bool | None) -> str:
    if label is None:
        return "pending"
    if not label:
        return "fp_invalid_input"
    reference = execution.get("reference_output")
    if reference is None:
        return "unresolved_reference"
    if candidate["oracle"] != reference:
        return "fp_wrong_oracle"
    if candidate["put_output"] != reference:
        return "tp"
    return "right_oracle_no_bug"


def load_task_bundle(directory: Path) -> dict[str, Any]:
    data_directory = analysis_data_directory(directory)
    return {
        "directory": directory,
        "data_directory": data_directory,
        "metadata": json.loads(read_text(data_directory / "task.json")),
        "variants": json.loads(read_text(data_directory / "accepted_variants.json")),
        "executions": read_jsonl(data_directory / "executions.jsonl"),
    }


def paper_protocol_analysis(
    bundles: list[dict[str, Any]],
    labels: dict[tuple[str, str], bool | None],
    k: int,
    evaluation_task_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Compute record-level evidence precision and PUT-level recall."""
    evidence_rows: list[dict[str, Any]] = []
    put_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        metadata = bundle["metadata"]
        task_id = metadata["task_id"]
        indices = sorted(int(item["accepted_index"]) for item in bundle["variants"])
        subset = tuple(indices)
        counts: Counter[str] = Counter()
        for execution in bundle["executions"]:
            label = labels.get((task_id, execution["input_id"]))
            for candidate in paper_candidates_for(execution, subset):
                category = classification(execution, candidate, label)
                counts[category] += 1
                evidence_rows.append({
                    "task_id": task_id, "problem_id": metadata["problem_id"],
                    "input_id": execution["input_id"], "k_cap": k,
                    "actual_k": len(indices), "candidate_oracle": candidate["oracle"],
                    "put_output": candidate["put_output"],
                    "reference_output": execution.get("reference_output"),
                    "raw_support": candidate["support"],
                    "weighted_support": candidate.get("weighted_support", ""),
                    "classification": category,
                })
        tp_rows = counts["tp"]
        fp_rows = sum(count for name, count in counts.items() if name.startswith("fp_"))
        precision: float | str = tp_rows / (tp_rows + fp_rows) if tp_rows + fp_rows else ""
        put_rows.append({
            "task_id": task_id,
            "problem_id": metadata["problem_id"],
            "requested_k": k,
            "actual_k": len(indices),
            "available_variants": len(indices),
            "candidate_oracle_rows": sum(counts.values()),
            "tp_rows": tp_rows,
            "fp_rows": fp_rows,
            "precision": precision,
            "detected": int(tp_rows > 0),
            "recall_class": "tp" if tp_rows else "fn",
            "pending_rows": counts["pending"],
            "unresolved_reference_rows": counts["unresolved_reference"],
        })

    completed_ids = {str(row["task_id"]) for row in put_rows}
    for task_id in sorted((evaluation_task_ids or set()) - completed_ids):
        parts = task_id.split("/")
        put_rows.append({
            "task_id": task_id, "problem_id": parts[1] if len(parts) > 1 else "",
            "requested_k": k, "actual_k": 0, "available_variants": 0,
            "candidate_oracle_rows": 0, "tp_rows": 0, "fp_rows": 0,
            "precision": "", "detected": 0, "recall_class": "fn",
            "pending_rows": 0, "unresolved_reference_rows": 0,
        })

    precision_values: list[float] = [
        float(row["precision"]) for row in put_rows if row["precision"] != ""
    ]
    precision = statistics.fmean(precision_values) if precision_values else 0.0
    tp_puts = sum(int(row["detected"]) for row in put_rows)
    fn_puts = len(put_rows) - tp_puts
    recall = tp_puts / (tp_puts + fn_puts) if tp_puts + fn_puts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    actual_ks = [int(row["actual_k"]) for row in put_rows]
    actual_k_counts = Counter(actual_ks)
    summary: dict[str, Any] = {
        "k": k,
        "requested_k_cap": k,
        "actual_k_policy": "use the retained pool from this independent K run",
        "actual_k_min": min(actual_ks) if actual_ks else 0,
        "actual_k_max": max(actual_ks) if actual_ks else 0,
        "actual_k_mean": statistics.fmean(actual_ks) if actual_ks else 0.0,
        "actual_k_distribution": {
            str(actual_k): count for actual_k, count in sorted(actual_k_counts.items())
        },
        "metric_protocol": "candidate-level macro precision and PUT-level recall TP/(TP+FN)",
        "put_count": len(put_rows),
        "buggy_puts": len(put_rows),
        "tp_puts": tp_puts,
        "fn_puts": fn_puts,
        "tn_puts": 0,
        "puts_with_defined_precision": len(precision_values),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_std_across_puts": (
            statistics.pstdev(precision_values) if len(precision_values) > 1 else 0.0
        ),
        "put_detection_rate": recall,
        "tp_records": sum(int(row["tp_rows"]) for row in put_rows),
        "fp_records": sum(int(row["fp_rows"]) for row in put_rows),
        "pending_rows": sum(int(row["pending_rows"]) for row in put_rows),
        "unresolved_reference": sum(
            int(row["unresolved_reference_rows"]) for row in put_rows
        ),
    }
    summary["provisional"] = bool(
        summary["pending_rows"]
    )
    return evidence_rows, put_rows, summary


def task_outcome_actual_k(bundle: dict[str, Any], labels: dict[tuple[str, str], bool | None]) -> dict[str, Any]:
    task_id = bundle["metadata"]["task_id"]
    subset = tuple(sorted(int(item["accepted_index"]) for item in bundle["variants"])[:10])
    if not subset:
        return {"eligible": False, "detected": False, "tp_cases": 0, "fp_cases": 0,
                "pending": 0, "produced_candidate_evidence": False,
                "useful": {}, "observed_buggy": {}}
    useful = {index: 0 for index in subset}
    observed_buggy = {index: False for index in subset}
    detected = produced_candidate_evidence = False
    tp_cases = fp_cases = pending = 0
    for execution in bundle["executions"]:
        label = labels.get((task_id, execution["input_id"]))
        reference = execution.get("reference_output")
        outputs = normal_variant_outputs(execution)
        if label is True and reference is not None:
            for index in observed_buggy:
                if outputs.get(index) != reference:
                    observed_buggy[index] = True
        candidates = paper_candidates_for(execution, subset)
        produced_candidate_evidence = produced_candidate_evidence or bool(candidates)
        for candidate in candidates:
            category = classification(execution, candidate, label)
            if category == "tp":
                detected = True
                tp_cases += 1
                for index, output in outputs.items():
                    if output == reference and index in useful:
                        useful[index] += 1
            elif category.startswith("fp_"):
                fp_cases += 1
            elif category == "pending":
                pending += 1
    return {"eligible": True, "detected": detected, "tp_cases": tp_cases,
            "fp_cases": fp_cases, "pending": pending,
            "produced_candidate_evidence": produced_candidate_evidence,
            "useful": useful, "observed_buggy": observed_buggy}


def variant_analysis(bundles: list[dict[str, Any]], outcomes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        meta = bundle["metadata"]
        outcome = outcomes[meta["task_id"]]
        if not outcome.get("produced_candidate_evidence"):
            continue
        for variant in bundle["variants"]:
            index = int(variant["accepted_index"])
            base = variant.get("base_tests", {})
            extra = variant.get("extra_tests", {})
            observed = bool(extra.get("total") and not extra.get("all_passed", False))
            observed = observed or outcome.get("observed_buggy", {}).get(index, False)
            contributions = int(outcome.get("useful", {}).get(index, 0))
            rows.append({
                "task_id": meta["task_id"], "problem_id": meta["problem_id"],
                "variant_index": index,
                "base_total": base.get("total", 0), "base_passed": base.get("passed", 0),
                "base_pass_rate": (base.get("passed", 0) / base.get("total", 1)) if base.get("total") else "",
                "extra_total": extra.get("total", 0), "extra_passed": extra.get("passed", 0),
                "extra_pass_rate": (extra.get("passed", 0) / extra.get("total", 1)) if extra.get("total") else "",
                "observed_buggy": int(observed), "useful": int(contributions > 0),
                "useful_contributions": contributions,
                "useful_buggy": int(observed and contributions > 0),
            })
    return rows


def _mean_numeric(values: Any) -> float:
    numbers = [float(value) for value in values if value not in (None, "")]
    return statistics.fmean(numbers) if numbers else 0.0


def make_figures(
    root: Path,
    variant_cardinality: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    useful_summary: dict[str, Any],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    plt.style.use("seaborn-v0_8-whitegrid")
    detection_effectiveness = variant_cardinality[0] if variant_cardinality else None
    if detection_effectiveness:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        names = ["Precision", "Recall", "F1"]
        values = [
            detection_effectiveness["precision"],
            detection_effectiveness["recall"],
            detection_effectiveness["f1"],
        ]
        bars = ax.bar(names, values, color=["#4C78A8", "#F58518", "#54A24B"])
        ax.set(title=f"Bug Detection Effectiveness (K={detection_effectiveness['k']})",
               ylabel="Score", ylim=(0, 1.12))
        ax.bar_label(bars, fmt="%.3f", padding=3)
        fig.tight_layout()
        path = figure_dir / "bug_detection_effectiveness.png"
        fig.savefig(path, dpi=180); plt.close(fig); created.append(str(path))
    if variants:
        syntax_accepted = len(variants)
        observed = sum(row["observed_buggy"] for row in variants)
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        bars = ax.bar(["Accepted variants", "Observed buggy"], [syntax_accepted, observed],
                      color=["#54A24B", "#E45756"])
        ax.set(title="Program Variant Quality", ylabel="Number of variants")
        ax.bar_label(bars, padding=3)
        fig.tight_layout()
        path = figure_dir / "variant_quality.png"
        fig.savefig(path, dpi=180); plt.close(fig); created.append(str(path))
    if useful_summary.get("useful_variants", 0):
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        bars = ax.bar(["Useful", "Useful & buggy"],
                      [useful_summary["useful_variants"], useful_summary["useful_buggy_variants"]],
                      color=["#72B7B2", "#B279A2"])
        ax.set(title="Usefulness of Buggy Program Variants", ylabel="Number of variants")
        ax.bar_label(bars, padding=3)
        fig.tight_layout()
        path = figure_dir / "useful_buggy_variants.png"
        fig.savefig(path, dpi=180); plt.close(fig); created.append(str(path))
    return created


def analyze(
    root: Path,
    include_variant_diagnostics: bool = True,
    included_task_ids: set[str] | None = None,
    k_values: tuple[int, ...] = KS,
) -> dict[str, Any]:
    scope_rows, scope_summary = experiment_scope(root, included_task_ids)
    directories = task_dirs(root, included_task_ids)
    bundles = [load_task_bundle(path) for path in directories]
    labels = load_labels(root)
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "generator_eligibility.csv", scope_rows)
    atomic_json(tables / "experiment_scope.json", scope_summary)
    evidence_rows: list[dict[str, Any]] = []
    put_metric_rows: list[dict[str, Any]] = []
    variant_cardinality_rows: list[dict[str, Any]] = []
    for k in k_values:
        details, per_put, summary = paper_protocol_analysis(
            bundles, labels, k, included_task_ids
        )
        evidence_rows.extend(details)
        put_metric_rows.extend(per_put)
        variant_cardinality_rows.append(summary)
    write_csv(tables / "candidate_evidence_records.csv", evidence_rows)
    write_csv(tables / "put_metrics.csv", put_metric_rows)
    detection_effectiveness = dict(variant_cardinality_rows[0]) if variant_cardinality_rows else {}
    atomic_json(
        tables / "bug_detection_effectiveness.json", detection_effectiveness
    )
    write_csv(
        tables / "bug_detection_effectiveness.csv",
        [detection_effectiveness] if detection_effectiveness else [],
    )
    variants: list[dict[str, Any]] = []
    variant_quality_summary: dict[str, Any] = {}
    useful_summary: dict[str, Any] = {}
    if include_variant_diagnostics:
        outcomes = {bundle["metadata"]["task_id"]: task_outcome_actual_k(bundle, labels)
                    for bundle in bundles}
        variants = variant_analysis(bundles, outcomes)
        write_csv(tables / "variant_quality.csv", variants)
        observed_buggy = sum(int(row["observed_buggy"]) for row in variants)
        variant_quality_summary = {
            "accepted_variants": len(variants),
            "observed_buggy_variants": observed_buggy,
            "observed_buggy_ratio": observed_buggy / len(variants) if variants else 0.0,
            "mean_base_pass_rate": _mean_numeric(row["base_pass_rate"] for row in variants),
            "mean_extra_pass_rate": _mean_numeric(row["extra_pass_rate"] for row in variants),
        }
        atomic_json(tables / "variant_quality_summary.json", variant_quality_summary)
        useful = [row for row in variants if row["useful"]]
        useful_buggy = [row for row in variants if row["useful_buggy"]]
        useful_summary = {
            "accepted_variants": len(variants), "useful_variants": len(useful),
            "useful_buggy_variants": len(useful_buggy),
            "useful_buggy_ratio": len(useful_buggy) / len(useful) if useful else 0.0,
            "provisional": any(outcome["pending"] for outcome in outcomes.values()),
        }
        write_csv(tables / "useful_buggy_variants.csv", useful_buggy)
        atomic_json(tables / "useful_buggy_summary.json", useful_summary)
    figures = make_figures(
        root,
        variant_cardinality_rows,
        variants,
        useful_summary,
    )
    review_total = len(labels)
    labeled = sum(value is not None for value in labels.values())
    summary = {
        "completed_tasks": len(bundles), "review_rows": review_total,
        "review_labeled": labeled, "review_pending": review_total - labeled,
        "bug_detection_effectiveness": detection_effectiveness,
        "variant_cardinality_sensitivity": variant_cardinality_rows,
        "metric_protocol": "candidate-level macro precision and PUT-level recall TP/(TP+FN)",
        "evaluation_population": scope_summary["population_filter"],
        "experiment_scope": scope_summary,
        "analysis_sections": [
            "bug_detection_effectiveness",
        ],
        "figures": figures,
        "final_metrics_ready": scope_summary["not_run_puts"] == 0
                               and labeled == review_total
                               and scope_summary["error_puts"] == 0
                               and not detection_effectiveness.get("provisional", True),
    }
    if include_variant_diagnostics:
        summary["analysis_sections"].extend([
            "program_variant_quality", "useful_buggy_variants",
        ])
        summary.update({
            "variant_quality_rows": len(variants),
            "variant_quality": variant_quality_summary,
            "useful_buggy": useful_summary,
        })
    atomic_json(root / "analysis_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TrickCatcher baseline outputs")
    parser.add_argument("command", choices=("prepare-review", "analyze"))
    parser.add_argument("--baseline", default="D:/RACEVoteLab/Baseline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.baseline)
    if args.command == "prepare-review":
        result = prepare_review(root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Review file: {root / 'manual' / 'input_validity.csv'}")
        return 0
    summary = analyze(root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["final_metrics_ready"]:
        print("Metrics are provisional until every review row is labeled and references resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
