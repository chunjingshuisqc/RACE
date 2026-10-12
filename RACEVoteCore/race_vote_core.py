"""Shared implementation for RACE-Vote and its ablation profiles."""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRICKCATCHER = ROOT / "TrickCatcher"
if str(TRICKCATCHER) not in sys.path:
    sys.path.insert(0, str(TRICKCATCHER))

import analyze_baseline as baseline_analysis  # noqa: E402
import run_baseline as baseline_run  # noqa: E402
from tc_core import (  # noqa: E402
    GLMClient,
    atomic_json,
    discover_trickybugs,
    normal_output,
    read_jsonl,
    read_text,
    safe_id,
    task_metadata,
)


PROFILE_FEATURES: dict[str, dict[str, bool]] = {
    "RACE-Vote": {"rc_weight": True, "ce_coevo": True, "llm_revote": True},
    "Ablation-TrickCatcher": {
        "rc_weight": False, "ce_coevo": False, "llm_revote": False,
    },
    "Ablation-No-RC": {"rc_weight": False, "ce_coevo": True, "llm_revote": True},
    "Ablation-No-CE": {"rc_weight": True, "ce_coevo": False, "llm_revote": True},
    "Ablation-No-RV": {"rc_weight": True, "ce_coevo": True, "llm_revote": False},
}
PAPER_K_VALUES = (6, 10)


def apply_k(cfg: dict[str, Any], k: int) -> dict[str, Any]:
    """Place each independent K run in its own output tree."""
    if k not in PAPER_K_VALUES:
        raise ValueError(f"Paper protocol supports K={PAPER_K_VALUES}, received {k}")
    value = dict(cfg)
    value["variant_pool_size"] = k
    value["k_values"] = [k]
    output = Path(str(value["output_dir"]))
    value["output_dir"] = str(output.parent / f"K{k}" / output.name)
    if value.get("reuse_source_output"):
        source = Path(str(value["reuse_source_output"]))
        value["reuse_source_output"] = str(source.parent / f"K{k}" / source.name)
    return value


def evaluation_tasks(cfg: dict[str, Any]) -> list[Any]:
    """Return the configured deterministic common PUT set."""
    expected = int(cfg.get("subject_count", 846))
    tasks = discover_trickybugs(Path(str(cfg["dataset_path"])), expected)
    if len(tasks) != expected:
        raise ValueError(f"Experiment requires {expected} PUTs, found {len(tasks)}")
    return tasks


def evaluation_task_ids(cfg: dict[str, Any]) -> set[str]:
    return {task.task_id for task in evaluation_tasks(cfg)}


def resolve_features(cfg: dict[str, Any]) -> dict[str, bool]:
    profile = str(cfg.get("experiment_name", ""))
    if profile not in PROFILE_FEATURES:
        raise ValueError(f"Unknown experiment profile: {profile!r}")
    expected = PROFILE_FEATURES[profile]
    configured = cfg.get("features", {})
    resolved = {name: bool(configured.get(name, value)) for name, value in expected.items()}
    if resolved != expected:
        raise ValueError(
            f"Feature flags for {profile} must be {expected}, received {resolved}"
        )
    return resolved


def _normal_variant_outputs(execution: dict[str, Any]) -> dict[int, str]:
    outputs: dict[int, str] = {}
    for item in execution.get("variant_results", []):
        output = normal_output(item.get("result", {}))
        if output is not None:
            outputs[int(item["variant_index"])] = output
    return outputs


def compute_rc_weights(
    variants: list[dict[str, Any]], executions: list[dict[str, Any]]
) -> dict[int, dict[str, float]]:
    """Reliability weighting with behavioral-correlation de-duplication."""
    indices = [int(item["accepted_index"]) for item in variants]
    signatures: dict[int, list[str | None]] = {index: [] for index in indices}
    ok_counts = Counter()
    for execution in executions:
        outputs = _normal_variant_outputs(execution)
        for index in indices:
            signatures[index].append(outputs.get(index))
            ok_counts[index] += int(index in outputs)

    raw: dict[int, dict[str, float]] = {}
    total = max(1, len(executions))
    for index in indices:
        reliability = 0.5 + 0.5 * ok_counts[index] / total
        correlations: list[float] = []
        for other in indices:
            if other == index:
                continue
            comparable = [
                (left, right)
                for left, right in zip(signatures[index], signatures[other])
                if left is not None and right is not None
            ]
            if comparable:
                correlations.append(
                    sum(left == right for left, right in comparable) / len(comparable)
                )
        correlation = sum(correlations) / len(correlations) if correlations else 0.0
        score = reliability / (1.0 + correlation)
        raw[index] = {
            "reliability": reliability,
            "mean_behavioral_correlation": correlation,
            "raw_weight": score,
        }
    mean_score = (
        sum(item["raw_weight"] for item in raw.values()) / len(raw) if raw else 1.0
    )
    for item in raw.values():
        item["weight"] = item["raw_weight"] / mean_score if mean_score else 1.0
    return raw


def _counterexample_context(executions: list[dict[str, Any]], limit: int = 8) -> str:
    records: list[dict[str, Any]] = []
    for execution in executions:
        put = normal_output(execution.get("put_result", {}))
        if put is None:
            continue
        outputs = _normal_variant_outputs(execution)
        groups = Counter(output for output in outputs.values() if output != put)
        if not groups:
            continue
        records.append({
            "stdin": execution.get("stdin", ""),
            "put_output": put,
            "variant_outputs": {str(index): output for index, output in outputs.items()},
            "alternative_output_support": dict(groups),
        })
        if len(records) >= limit:
            break
    if not records:
        return "No output disagreement was observed. Emphasize boundary and corner cases."
    return json.dumps(records, ensure_ascii=False, indent=2)


def _reindex_variants(records: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        digest = str(record.get("source_sha1", record.get("path", "")))
        if digest in seen:
            continue
        seen.add(digest)
        value = dict(record)
        value["accepted_index"] = len(selected)
        selected.append(value)
        if len(selected) >= target:
            break
    return selected


def _reindex_cases(groups: list[list[dict[str, Any]]], target: int) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    seen: set[str] = set()
    for group in groups:
        for case in group:
            stdin = str(case["stdin"])
            if stdin in seen:
                continue
            seen.add(stdin)
            cases.append({"input_id": f"input_{len(cases):03d}", "stdin": stdin})
            if len(cases) >= target:
                return cases
    return cases


def run_llm_revote(
    task: Any,
    variants: list[dict[str, Any]],
    executions: list[dict[str, Any]],
    rc_weights: dict[int, dict[str, float]],
    model: GLMClient,
    seed: int,
) -> dict[str, Any]:
    context = _counterexample_context(executions, limit=6)
    rows = []
    for item in variants:
        index = int(item["accepted_index"])
        weight_info = rc_weights.get(index, {"weight": 1.0})
        source_excerpt = ""
        path = Path(str(item.get("path", "")))
        if path.is_file():
            source_excerpt = read_text(path)[:800]
        rows.append({
            "variant_index": index,
            "initial_weight": weight_info.get("weight", 1.0),
            "reliability": weight_info.get("reliability", 1.0),
            "behavioral_correlation": weight_info.get(
                "mean_behavioral_correlation", 0.0
            ),
            "code_excerpt": source_excerpt,
        })
    prompt = f"""You are recalibrating votes from program variants in differential testing.
Use only the specification, initial weights, and observed disagreements. Do not infer or emit
the correct answer. Down-weight variants likely to share the same correlated fault and up-weight
independent, stable evidence. Return one JSON object only:
{{"multipliers": {{"0": 1.0}}, "reason": "short explanation"}}
Every multiplier must be between 0.25 and 4.0.

SPECIFICATION:
{task.specification}

VARIANTS:
{json.dumps(rows, ensure_ascii=False)}

OBSERVED DISAGREEMENTS:
{context}
"""
    generation = model.generate(prompt, seed, use_turbo=True)
    text = generation.get("text", "")
    match = re.search(r"\{.*\}", text, flags=re.S)
    multipliers = {int(item["accepted_index"]): 1.0 for item in variants}
    status = "fallback"
    reason = "No valid JSON vote recalibration was returned."
    if match:
        try:
            value = json.loads(match.group(0))
            supplied = value.get("multipliers", {})
            for index in multipliers:
                if str(index) in supplied:
                    multipliers[index] = min(4.0, max(0.25, float(supplied[str(index)])))
            status = "ok"
            reason = str(value.get("reason", ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return {
        "status": status,
        "multipliers": multipliers,
        "reason": reason,
        "generation": generation,
    }


def _uniform_weights(variants: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    return {
        int(item["accepted_index"]): {
            "reliability": 1.0,
            "mean_behavioral_correlation": 0.0,
            "raw_weight": 1.0,
            "weight": 1.0,
        }
        for item in variants
    }


def _reuse_source_task_dir(task: Any, cfg: dict[str, Any]) -> Path:
    source_root = Path(str(cfg.get(
        "reuse_source_output", ROOT / "Baseline" / "RACE-Vote"
    )))
    source_dir = source_root / "raw" / safe_id(task.task_id)
    status_path = source_dir / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"RACE-Vote source status is missing: {status_path}")
    status = json.loads(read_text(status_path))
    if status.get("status") != "complete":
        raise ValueError(f"RACE-Vote source is not reusable: {task.task_id}")
    return source_dir


def _load_reused_final(source_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    variants_path = source_dir / "accepted_variants.json"
    executions_path = source_dir / "executions.jsonl"
    if not variants_path.is_file() or not executions_path.is_file():
        raise FileNotFoundError(f"RACE-Vote final artifacts are incomplete: {source_dir}")
    variants = json.loads(read_text(variants_path))
    executions = read_jsonl(executions_path)
    status = json.loads(read_text(source_dir / "status.json"))
    return variants, executions, status


def _write_reuse_shell(
    task: Any,
    task_dir: Path,
    cfg: dict[str, Any],
    source_dir: Path,
    variants: list[dict[str, Any]],
    executions: list[dict[str, Any]],
    source_status: dict[str, Any],
    weights: dict[int, dict[str, float]],
    revote: dict[str, Any],
    mode: str,
) -> str:
    atomic_json(task_dir / "task.json", task_metadata(task))
    atomic_json(task_dir / "reuse_manifest.json", {
        "mode": mode,
        "analysis_source_task_dir": str(source_dir),
        "reused": ["task", "accepted_variants", "inputs", "executions"],
    })
    atomic_json(task_dir / "rc_weights.json", {
        "enabled": bool(resolve_features(cfg)["rc_weight"]),
        "weights": weights,
    })
    atomic_json(task_dir / "llm_revote.json", revote)
    atomic_json(task_dir / "status.json", {
        "status": "complete",
        "task_id": task.task_id,
        "experiment_name": cfg["experiment_name"],
        "features": resolve_features(cfg),
        "generator_eligible": bool(source_status.get("generator_eligible", False)),
        "accepted_variants": len(variants),
        "actual_k": len(variants),
        "actual_k_policy": (
            f"retained pool from independent K={int(cfg.get('variant_pool_size', 10))} run"
        ),
        "generated_inputs": int(source_status.get("generated_inputs", len(executions))),
        "executed_inputs": len(executions),
        "generator_status": source_status.get("generator_status", {}),
        "reuse_mode": mode,
        "reuse_source_task_dir": str(source_dir),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    return f"complete ({mode}; reused RACE-Vote artifacts)"


def _process_full_reuse(
    task: Any,
    model: GLMClient,
    task_dir: Path,
    cfg: dict[str, Any],
    mode: str,
) -> str:
    source_dir = _reuse_source_task_dir(task, cfg)
    variants, executions, source_status = _load_reused_final(source_dir)
    if mode == "reuse_full_disable_revote":
        source_weights = json.loads(read_text(source_dir / "rc_weights.json"))
        weights = {
            int(index): data for index, data in source_weights.get("weights", {}).items()
        }
        revote = {
            "enabled": False,
            "status": "disabled_by_ablation",
            "multipliers": {index: 1.0 for index in weights},
            "source_revote_ignored": str(source_dir / "llm_revote.json"),
        }
    elif mode == "reuse_full_revote_only":
        weights = _uniform_weights(variants) if executions else {}
        revote = (
            run_llm_revote(
                task, variants, executions, weights, model,
                int(cfg.get("seed", 42)) + 30_000,
            ) if executions else {
                "status": "skipped_empty_pool", "multipliers": {}, "reason": "",
            }
        )
        revote["enabled"] = bool(executions)
    else:
        raise ValueError(f"Unsupported full reuse mode: {mode}")
    return _write_reuse_shell(
        task, task_dir, cfg, source_dir, variants, executions, source_status,
        weights, revote, mode,
    )


def _process_no_ce_reuse(
    task: Any,
    model: GLMClient,
    task_dir: Path,
    output_root: Path,
    cfg: dict[str, Any],
) -> str:
    source_dir = _reuse_source_task_dir(task, cfg)
    source_status = json.loads(read_text(source_dir / "status.json"))
    stage1 = source_dir / "stage_1_exploration"
    variants_path = stage1 / "accepted_variants.json"
    inputs_path = stage1 / "inputs.jsonl"
    if not variants_path.is_file() or not inputs_path.is_file():
        raise FileNotFoundError(f"RACE-Vote stage 1 is incomplete: {stage1}")
    initial_variants = json.loads(read_text(variants_path))
    initial_cases = read_jsonl(inputs_path)
    target_variants = int(cfg.get("variant_pool_size", 10))
    target_inputs = int(cfg.get("inputs_per_task", 100))
    continuation = task_dir / "stage_2_without_counterexample_feedback"

    remaining_variants = max(0, target_variants - len(initial_variants))
    added_variants: list[dict[str, Any]] = []
    variant_attempts: list[dict[str, Any]] = []
    if remaining_variants:
        variant_cfg = dict(cfg)
        variant_cfg.update({
            "variant_pool_size": remaining_variants,
            "max_variant_attempts": int(cfg.get("no_ce_variant_attempts", 22)),
            "seed": int(cfg.get("seed", 42)) + 10_000,
        })
        added_variants, variant_attempts = baseline_run.generate_variants(
            task, model, continuation, variant_cfg
        )
    variants = _reindex_variants(initial_variants + added_variants, target_variants)

    remaining_inputs = max(0, target_inputs - len(initial_cases))
    added_cases: list[dict[str, Any]] = []
    added_generator_status: dict[str, Any] = {}
    if remaining_inputs:
        case_cfg = dict(cfg)
        case_cfg.update({
            "inputs_per_task": remaining_inputs,
            "generator_attempts": int(cfg.get("no_ce_generator_attempts", 2)),
            "seed": int(cfg.get("seed", 42)) + 20_000,
        })
        added_cases, added_generator_status = baseline_run.generate_cases(
            task, model, continuation, case_cfg
        )
    cases = _reindex_cases([initial_cases, added_cases], target_inputs)
    atomic_json(task_dir / "task.json", task_metadata(task))
    atomic_json(task_dir / "accepted_variants.json", variants)
    baseline_run.write_jsonl(task_dir / "inputs.jsonl", cases)
    baseline_run.write_jsonl(task_dir / "variant_attempts.jsonl", variant_attempts)
    atomic_json(task_dir / "coevolution.json", {
        "enabled": False,
        "ablation": "counterexample feedback removed",
        "reused_stage1_source_task_dir": str(stage1),
        "reused_stage1_variants": len(initial_variants),
        "reused_stage1_inputs": len(initial_cases),
        "generic_stage2_variants": len(added_variants),
        "generic_stage2_inputs": len(added_cases),
    })
    atomic_json(task_dir / "reuse_manifest.json", {
        "mode": "reuse_stage1_without_feedback",
        "reused_stage1_source_task_dir": str(source_dir),
        "reused": ["stage_1_variants", "stage_1_inputs"],
        "regenerated": ["stage_2_variants_without_feedback", "stage_2_inputs_without_feedback"],
    })

    executions = (
        baseline_run.execute_cases(task, variants, cases, task_dir, output_root, cfg)
        if variants and cases else []
    )
    if not executions:
        baseline_run.write_jsonl(task_dir / "executions.jsonl", [])
    weights = compute_rc_weights(variants, executions) if executions else {}
    atomic_json(task_dir / "rc_weights.json", {"enabled": True, "weights": weights})
    revote = (
        run_llm_revote(
            task, variants, executions, weights, model,
            int(cfg.get("seed", 42)) + 30_000,
        ) if executions else {
            "status": "skipped_empty_pool", "multipliers": {}, "reason": "",
        }
    )
    revote["enabled"] = bool(executions)
    atomic_json(task_dir / "llm_revote.json", revote)
    generator_status = dict(source_status.get("generator_status", {}))
    generator_status.update({
        "generator_eligible": bool(cases),
        "reused_stage1": True,
        "generic_stage2": added_generator_status,
        "generated_unique_inputs": len(cases),
        "target_inputs": target_inputs,
        "input_yield": len(cases) / target_inputs if target_inputs else 1.0,
    })
    atomic_json(task_dir / "status.json", {
        "status": "complete",
        "task_id": task.task_id,
        "experiment_name": cfg["experiment_name"],
        "features": resolve_features(cfg),
        "generator_eligible": bool(cases),
        "accepted_variants": len(variants),
        "actual_k": len(variants),
        "actual_k_policy": f"retained pool from independent K={target_variants} run",
        "generated_inputs": len(cases),
        "executed_inputs": len(executions),
        "generator_status": generator_status,
        "reuse_mode": "reuse_stage1_without_feedback",
        "reuse_source_task_dir": str(source_dir),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    return "complete (reused RACE-Vote stage 1; regenerated generic stage 2)"


def process_task(
    task: Any,
    model: GLMClient,
    output_root: Path,
    cfg: dict[str, Any],
    overwrite: bool,
) -> str:
    features = resolve_features(cfg)
    task_dir = output_root / "raw" / safe_id(task.task_id)
    status_path = task_dir / "status.json"
    if status_path.is_file() and not overwrite:
        status = json.loads(read_text(status_path))
        if status.get("status") == "variant_failed":
            executions_path = task_dir / "executions.jsonl"
            if executions_path.is_file():
                status.update({
                    "status": "complete",
                    "actual_k": int(status.get("accepted_variants", 0)),
                    "actual_k_policy": "continue with every accepted variant, capped at 10",
                    "restored_at": datetime.now(timezone.utc).isoformat(),
                })
                status.pop("exclusion_reason", None)
                status.pop("required_variants", None)
                status.pop("legacy_status_migrated_at", None)
                atomic_json(status_path, status)
                return "skipped (complete; restored actual-k result)"
            variants_path = task_dir / "accepted_variants.json"
            inputs_path = task_dir / "inputs.jsonl"
            if (not features["ce_coevo"] and variants_path.is_file()
                    and inputs_path.is_file()):
                try:
                    variants = json.loads(read_text(variants_path))
                    cases = read_jsonl(inputs_path)
                    executions = baseline_run.execute_cases(
                        task, variants, cases, task_dir, output_root, cfg
                    )
                    weights = _uniform_weights(variants)
                    atomic_json(task_dir / "rc_weights.json", {
                        "enabled": False, "weights": weights,
                    })
                    atomic_json(task_dir / "llm_revote.json", {
                        "enabled": False, "status": "disabled",
                        "multipliers": {index: 1.0 for index in weights},
                    })
                    status.update({
                        "status": "complete",
                        "actual_k": len(variants),
                        "actual_k_policy": (
                            "continue with every accepted variant, capped at 10"
                        ),
                        "accepted_variants": len(variants),
                        "generated_inputs": len(cases),
                        "executed_inputs": len(executions),
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    })
                    status.pop("exclusion_reason", None)
                    status.pop("required_variants", None)
                    atomic_json(status_path, status)
                    return "complete (resumed with actual k)"
                except Exception as exc:
                    atomic_json(status_path, {
                        "status": "error", "task_id": task.task_id,
                        "error_type": type(exc).__name__, "error": str(exc),
                    })
                    raise
        if status.get("status") in {"complete", "generator_failed"}:
            return f"skipped ({status['status']})"
    task_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(task_dir / "task.json", task_metadata(task))
    try:
        reuse_mode = str(cfg.get("reuse_mode", ""))
        if reuse_mode in {"reuse_full_disable_revote", "reuse_full_revote_only"}:
            return _process_full_reuse(task, model, task_dir, cfg, reuse_mode)
        if reuse_mode == "reuse_stage1_without_feedback":
            return _process_no_ce_reuse(task, model, task_dir, output_root, cfg)
        if reuse_mode:
            raise ValueError(f"Unsupported reuse mode: {reuse_mode}")
        target_variants = int(cfg.get("variant_pool_size", 10))
        target_inputs = int(cfg.get("inputs_per_task", 100))
        if features["ce_coevo"]:
            initial_variant_count = max(1, target_variants // 2)
            initial_input_count = max(1, target_inputs // 2)
        else:
            initial_variant_count = target_variants
            initial_input_count = target_inputs

        initial_dir = task_dir / "stage_1_exploration"
        initial_case_cfg = dict(cfg)
        initial_case_cfg.update({
            "inputs_per_task": initial_input_count,
            "generator_attempts": int(cfg.get("initial_generator_attempts", 2)),
        })
        initial_cases, generator_status = baseline_run.generate_cases(
            task, model, initial_dir, initial_case_cfg
        )
        if (bool(cfg.get("require_usable_generator", True))
                and not generator_status["generator_eligible"]
                and not features["ce_coevo"]):
            status = {
                "status": "generator_failed",
                "task_id": task.task_id,
                "generator_eligible": False,
                "features": features,
                "exclusion_reason": "no executable generator produced parseable inputs",
                "generated_inputs": len(initial_cases),
                "generator_status": generator_status,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(status_path, status)
            return "generator_failed (retained as FN in the evaluation set)"

        initial_variant_cfg = dict(cfg)
        initial_variant_cfg.update({
            "variant_pool_size": initial_variant_count,
            "max_variant_attempts": int(cfg.get(
                "initial_variant_attempts", cfg.get("max_variant_attempts", 20)
            )),
        })
        initial_variants, initial_attempts = baseline_run.generate_variants(
            task, model, initial_dir, initial_variant_cfg
        )
        exploration = (
            baseline_run.execute_cases(
                task, initial_variants, initial_cases, initial_dir,
                output_root, cfg,
            )
            if features["ce_coevo"] and initial_variants and initial_cases else []
        )

        all_variants = list(initial_variants)
        all_cases = [initial_cases]
        evolution: dict[str, Any] = {"enabled": features["ce_coevo"]}
        evolved_attempts: list[dict[str, Any]] = []
        if features["ce_coevo"]:
            context = _counterexample_context(exploration)
            evolved_task = replace(
                task,
                specification=(
                    task.specification
                    + "\n\nCOUNTEREXAMPLE FEEDBACK FROM EXPLORATION:\n"
                    + context
                    + "\nGenerate candidates that test or repair the implicated boundaries."
                ),
            )
            evolved_dir = task_dir / "stage_2_counterexample_evolution"
            remaining_variant_count = max(0, target_variants - len(initial_variants))
            evolved_variants: list[dict[str, Any]] = []
            if remaining_variant_count:
                evolved_variant_cfg = dict(cfg)
                evolved_variant_cfg.update({
                    "variant_pool_size": remaining_variant_count,
                    "max_variant_attempts": int(cfg.get("evolved_variant_attempts", 25)),
                    "seed": int(cfg.get("seed", 42)) + 10_000,
                })
                evolved_variants, evolved_attempts = baseline_run.generate_variants(
                    evolved_task, model, evolved_dir, evolved_variant_cfg
                )
            all_variants.extend(evolved_variants)
            evolution.update({
                "counterexample_context": context,
                "remaining_variant_target": remaining_variant_count,
                "evolved_variants": len(evolved_variants),
            })

        variants = _reindex_variants(all_variants, target_variants)
        atomic_json(task_dir / "accepted_variants.json", variants)
        atomic_json(task_dir / "coevolution.json", evolution)
        baseline_run.write_jsonl(
            task_dir / "variant_attempts.jsonl", initial_attempts + evolved_attempts
        )
        if features["ce_coevo"]:
            targeted_case_cfg = dict(cfg)
            targeted_case_cfg.update({
                "inputs_per_task": target_inputs - len(initial_cases),
                "generator_attempts": int(cfg.get("evolved_generator_attempts", 2)),
                "seed": int(cfg.get("seed", 42)) + 20_000,
            })
            targeted_cases, targeted_status = baseline_run.generate_cases(
                evolved_task, model, evolved_dir, targeted_case_cfg
            )
            all_cases.append(targeted_cases)
            evolution.update({
                "targeted_generator_status": targeted_status,
                "targeted_inputs": len(targeted_cases),
            })
            atomic_json(task_dir / "coevolution.json", evolution)

        cases = _reindex_cases(all_cases, target_inputs)
        baseline_run.write_jsonl(task_dir / "inputs.jsonl", cases)
        executions = (
            baseline_run.execute_cases(task, variants, cases, task_dir, output_root, cfg)
            if variants and cases else []
        )
        if not executions:
            baseline_run.write_jsonl(task_dir / "executions.jsonl", [])

        weights = (
            compute_rc_weights(variants, executions)
            if features["rc_weight"] and executions else
            _uniform_weights(variants) if executions else {}
        )
        atomic_json(task_dir / "rc_weights.json", {
            "enabled": features["rc_weight"],
            "weights": weights,
        })
        revote: dict[str, Any] = {
            "enabled": features["llm_revote"],
            "status": "disabled",
            "multipliers": {index: 1.0 for index in weights},
        }
        if features["llm_revote"] and executions:
            revote = run_llm_revote(
                task, variants, executions, weights, model,
                int(cfg.get("seed", 42)) + 30_000,
            )
            revote["enabled"] = True
        atomic_json(task_dir / "llm_revote.json", revote)
        status = {
            "status": "complete",
            "task_id": task.task_id,
            "experiment_name": cfg["experiment_name"],
            "features": features,
            "generator_eligible": bool(cases),
            "accepted_variants": len(variants),
            "actual_k": len(variants),
            "actual_k_policy": f"retained pool from independent K={target_variants} run",
            "generated_inputs": len(cases),
            "executed_inputs": len(executions),
            "generator_status": generator_status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(status_path, status)
        return "complete"
    except Exception as exc:
        atomic_json(status_path, {
            "status": "error", "task_id": task.task_id,
            "error_type": type(exc).__name__, "error": str(exc),
        })
        raise


def weighted_candidates_for(
    execution: dict[str, Any], subset: tuple[int, ...]
) -> list[dict[str, Any]]:
    put = normal_output(execution.get("put_result", {}))
    if put is None:
        return []
    outputs = _normal_variant_outputs(execution)
    weights = {
        int(index): float(value)
        for index, value in execution.get("_race_weights", {}).items()
    }
    multipliers = {
        int(index): float(value)
        for index, value in execution.get("_race_multipliers", {}).items()
    }
    scores: Counter[str] = Counter()
    support: Counter[str] = Counter()
    for index in subset:
        output = outputs.get(index)
        if output is None or output == put:
            continue
        scores[output] += weights.get(index, 1.0) * multipliers.get(index, 1.0)
        support[output] += 1
    if not scores:
        return []
    maximum = max(scores.values())
    tolerance = max(1e-12, abs(maximum) * 1e-9)
    return [
        {
            "oracle": output,
            "support": support[output],
            "weighted_support": score,
            "different_normal_variants": sum(support.values()),
            "put_output": put,
        }
        for output, score in scores.items()
        if math.isclose(score, maximum, rel_tol=1e-9, abs_tol=tolerance)
    ]


def analyze_profile(
    cfg: dict[str, Any], included_task_ids: set[str] | None = None
) -> dict[str, Any]:
    output_root = Path(cfg["output_dir"])
    features = resolve_features(cfg)
    if included_task_ids is None:
        included_task_ids = evaluation_task_ids(cfg)
    original_loader = baseline_analysis.load_task_bundle
    original_candidates = baseline_analysis.paper_candidates_for

    def load_bundle(directory: Path) -> dict[str, Any]:
        bundle = original_loader(directory)
        weight_data = json.loads(read_text(directory / "rc_weights.json"))
        revote_data = json.loads(read_text(directory / "llm_revote.json"))
        weights = {
            int(index): float(data["weight"])
            for index, data in weight_data.get("weights", {}).items()
        }
        multipliers = {
            int(index): float(value)
            for index, value in revote_data.get("multipliers", {}).items()
        }
        for execution in bundle["executions"]:
            execution["_race_weights"] = weights if features["rc_weight"] else {}
            execution["_race_multipliers"] = (
                multipliers if features["llm_revote"] else {}
            )
        return bundle

    try:
        baseline_analysis.load_task_bundle = load_bundle
        baseline_analysis.paper_candidates_for = weighted_candidates_for
        summary = baseline_analysis.analyze(
            output_root,
            include_variant_diagnostics=bool(
                cfg.get("include_variant_diagnostics", True)
            ),
            included_task_ids=included_task_ids,
            k_values=tuple(int(k) for k in cfg.get("k_values", baseline_analysis.KS)),
        )
    finally:
        baseline_analysis.load_task_bundle = original_loader
        baseline_analysis.paper_candidates_for = original_candidates
    summary["experiment_name"] = cfg["experiment_name"]
    summary["features"] = features
    summary["oracle_protocol"] = (
        "weighted differential voting" if features["rc_weight"] or features["llm_revote"]
        else "TrickCatcher diversity-driven voting"
    )
    summary["evaluation_set"] = {
        "fixed": included_task_ids is not None,
        "subjects": len(included_task_ids) if included_task_ids is not None else None,
    }
    atomic_json(output_root / "analysis_summary.json", summary)
    return summary
