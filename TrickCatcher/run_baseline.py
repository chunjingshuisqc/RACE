#!/usr/bin/env python3
"""Generate and execute the four-part TrickCatcher baseline experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tc_core import (
    GLMClient, atomic_json, check_syntax, compile_references,
    discover_trickybugs, evaluate_tests, extract_code, generator_prompt,
    glm_api_status, normal_output, read_text, run_input_generator, run_python, run_references,
    read_jsonl, safe_id, task_metadata, variant_prompt, write_jsonl,
)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(read_text(path))


def dataset_summary(dataset: Path) -> dict[str, int]:
    paths = sorted((dataset / "problems").glob("*/buggy_programs/python/*.py"))
    base = 0
    for path in paths:
        problem = path.parents[2]
        base += (problem / "original_test_cases").is_dir()
    return {"python_puts": len(paths), "puts_with_base_tests": base}


def preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset = Path(cfg["dataset_path"])
    data = dataset_summary(dataset) if dataset.is_dir() else {
        "python_puts": 0, "puts_with_base_tests": 0,
    }
    report = {
        "dataset_ready": dataset.is_dir(), **data,
        "glm_api": glm_api_status(),
        "compiler": shutil.which(cfg.get("cpp_compiler", "g++")),
        "python": sys.version,
    }
    report["ready"] = bool(report["dataset_ready"] and report["python_puts"]
                           and report["glm_api"]["ready"] and report["compiler"])
    return report


def create_manifest(output_root: Path, cfg: dict[str, Any], report: dict[str, Any]) -> None:
    atomic_json(output_root / "experiment_manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "TrickCatcher baseline on TrickyBugs Python",
        "metric_protocol": {
            "name": "candidate-level macro precision and PUT-level recall TP/(TP+FN)",
            "population_filter": (
                "include only PUTs for which at least one generator exits successfully "
                "and yields parseable unique inputs"
            ),
            "paper_deviation": (
                "generator-ineligible PUTs are excluded; each remaining PUT uses all "
                "accepted variants available up to the configured cap of 10"
            ),
            "precision": "mean of per-PUT TP_mass/(TP_mass+FP_mass)",
            "recall": "buggy PUTs detected by at least one TP / all included buggy PUTs",
            "fn": "included buggy PUTs for which no TP is produced",
            "f1": "harmonic mean of final precision and recall",
            "tie_handling": "fractional credit across maximum-support oracle ties",
        },
        "replications": [
            "bug_detection_effectiveness",
            "variant_cardinality_k_6_and_k_10",
            "variant_pass_rate",
            "useful_buggy_variants",
        ],
        "config": cfg, "preflight": report,
        "manual_step": "Fill Baseline/manual/input_validity.csv before final analysis.",
    })


def generate_variants(task: Any, model: GLMClient, task_dir: Path,
                      cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted_target = int(cfg.get("variant_pool_size", 10))
    max_attempts = int(cfg.get("max_variant_attempts", 30))
    timeout = int(cfg.get("execution_timeout_seconds", 5))
    base_seed = int(cfg.get("seed", 42))
    variants_dir = task_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for attempt_index in range(max_attempts):
        if len(accepted) >= accepted_target:
            break
        generation = model.generate(variant_prompt(task), base_seed + attempt_index)
        source = extract_code(generation["text"])
        syntax_ok, syntax_error = check_syntax(source)
        record: dict[str, Any] = {
            "attempt": attempt_index, "generation": generation,
            "syntax_ok": syntax_ok, "syntax_error": syntax_error,
            "accepted": False, "rejection": None,
        }
        if not syntax_ok:
            record["rejection"] = "syntax"
            attempts.append(record)
            continue
        digest = __import__("hashlib").sha1(source.encode()).hexdigest()
        if digest in hashes:
            record["rejection"] = "duplicate"
            attempts.append(record)
            continue
        candidate = variants_dir / f"candidate_{attempt_index:02d}.py"
        candidate.write_text(source, encoding="utf-8")
        base_result = evaluate_tests(candidate, task.base_tests, timeout, stop_on_failure=True)
        record["base_filter"] = base_result
        if task.base_tests and not base_result["all_passed"]:
            record["rejection"] = "base_tests"
            attempts.append(record)
            continue
        accepted_index = len(accepted)
        final_path = variants_dir / f"variant_{accepted_index:02d}.py"
        candidate.replace(final_path)
        hashes.add(digest)
        full_base = evaluate_tests(final_path, task.base_tests, timeout)
        extra = evaluate_tests(final_path, task.extra_tests, timeout)
        record.update({"accepted": True, "accepted_index": accepted_index,
                       "path": str(final_path), "base_tests": full_base,
                       "extra_tests": extra, "source_sha1": digest})
        accepted.append(record)
        attempts.append(record)
    write_jsonl(task_dir / "variant_attempts.jsonl", attempts)
    atomic_json(task_dir / "accepted_variants.json", accepted)
    return accepted, attempts


def generate_cases(task: Any, model: GLMClient, task_dir: Path,
                   cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_dir.mkdir(parents=True, exist_ok=True)
    target = int(cfg.get("inputs_per_task", 100))
    attempts = int(cfg.get("generator_attempts", 3))
    timeout = int(cfg.get("execution_timeout_seconds", 5))
    seed = int(cfg.get("seed", 42)) + 100_000
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for index in range(attempts):
        if len(cases) >= target:
            break
        generation = model.generate(generator_prompt(task, target - len(cases)), seed + index)
        source = extract_code(generation["text"])
        generated, execution = run_input_generator(source, target - len(cases), timeout)
        generator_file = task_dir / f"input_generator_{index:02d}.py"
        generator_file.write_text(source, encoding="utf-8")
        records.append({"attempt": index, "generation": generation,
                        "path": str(generator_file), "execution": execution,
                        "parsed_inputs": len(generated)})
        for case in generated:
            if case["stdin"] not in seen:
                seen.add(case["stdin"])
                cases.append({"input_id": f"input_{len(cases):03d}", "stdin": case["stdin"]})
                if len(cases) >= target:
                    break
    atomic_json(task_dir / "input_generators.json", records)
    write_jsonl(task_dir / "inputs.jsonl", cases)
    executable_attempts = sum(
        item["execution"].get("status") == "ok" for item in records
    )
    usable_attempts = sum(
        item["execution"].get("status") == "ok" and item["parsed_inputs"] > 0
        for item in records
    )
    generator_status = {
        "generator_eligible": usable_attempts > 0 and len(cases) > 0,
        "definition": "at least one generator exits successfully and yields parseable unique inputs",
        "attempts": len(records),
        "executable_attempts": executable_attempts,
        "usable_attempts": usable_attempts,
        "generated_unique_inputs": len(cases),
        "target_inputs": target,
        "input_yield": len(cases) / target if target else 0.0,
    }
    atomic_json(task_dir / "input_generator_status.json", generator_status)
    return cases, generator_status


def execute_cases(task: Any, accepted: list[dict[str, Any]], cases: list[dict[str, Any]],
                  task_dir: Path, output_root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    timeout = int(cfg.get("execution_timeout_seconds", 5))
    reference_count = int(cfg.get("reference_consensus_size", 3))
    put_path = task_dir / "put.py"
    put_path.write_text(task.source, encoding="utf-8")
    compiled = compile_references(
        task, output_root / "cache" / "reference_bin", reference_count,
        cfg.get("cpp_compiler", "g++"),
    )
    atomic_json(task_dir / "reference_programs.json", compiled)
    rows: list[dict[str, Any]] = []
    for case in cases:
        stdin = case["stdin"]
        reference_results, reference_output, reference_status = run_references(
            compiled, stdin, timeout,
        )
        variant_results = []
        for variant in accepted:
            result = run_python(Path(variant["path"]), stdin, timeout)
            variant_results.append({"variant_index": variant["accepted_index"], "result": result})
        rows.append({
            "input_id": case["input_id"], "stdin": stdin,
            "put_result": run_python(put_path, stdin, timeout),
            "reference_status": reference_status, "reference_output": reference_output,
            "reference_results": reference_results, "variant_results": variant_results,
        })
    write_jsonl(task_dir / "executions.jsonl", rows)
    return rows


def process_task(task: Any, model: GLMClient, output_root: Path,
                 cfg: dict[str, Any], overwrite: bool) -> str:
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
            if variants_path.is_file() and inputs_path.is_file():
                try:
                    accepted = json.loads(read_text(variants_path))
                    cases = read_jsonl(inputs_path)
                    executions = execute_cases(
                        task, accepted, cases, task_dir, output_root, cfg
                    )
                    status.update({
                        "status": "complete",
                        "actual_k": len(accepted),
                        "actual_k_policy": (
                            "continue with every accepted variant, capped at 10"
                        ),
                        "accepted_variants": len(accepted),
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
        cases, generator_status = generate_cases(task, model, task_dir, cfg)
        if bool(cfg.get("require_usable_generator", True)) and not generator_status["generator_eligible"]:
            status = {
                "status": "generator_failed",
                "task_id": task.task_id,
                "generator_eligible": False,
                "exclusion_reason": "no executable generator produced parseable inputs",
                "accepted_variants": 0,
                "variant_attempts": 0,
                "generated_inputs": len(cases),
                "generator_status": generator_status,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(status_path, status)
            return "generator_failed (excluded)"
        accepted, attempts = generate_variants(task, model, task_dir, cfg)
        executions = execute_cases(task, accepted, cases, task_dir, output_root, cfg)
        status = {
            "status": "complete", "task_id": task.task_id,
            "generator_eligible": True,
            "accepted_variants": len(accepted), "variant_attempts": len(attempts),
            "actual_k": len(accepted),
            "actual_k_policy": "continue with every accepted variant, capped at 10",
            "generated_inputs": len(cases), "executed_inputs": len(executions),
            "generator_status": generator_status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(status_path, status)
        return "complete"
    except Exception as exc:
        atomic_json(status_path, {"status": "error", "task_id": task.task_id,
                                  "error_type": type(exc).__name__, "error": str(exc)})
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TrickCatcher baseline replication")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--preflight", action="store_true", help="Check data/GLM configuration/compiler only")
    parser.add_argument("--discover-only", action="store_true", help="Count PUTs without model loading")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config))
    dataset = Path(cfg["dataset_path"])
    if args.discover_only:
        summary = dataset_summary(dataset) if dataset.is_dir() else {
            "python_puts": 0, "puts_with_base_tests": 0,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["python_puts"] else 2
    report = preflight(cfg)
    output_root = Path(cfg["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    create_manifest(output_root, cfg, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.preflight:
        return 0 if report["ready"] else 2
    if not report["ready"]:
        print("Preflight failed; run with --preflight for details.", file=sys.stderr)
        return 2
    tasks = discover_trickybugs(dataset, args.limit, args.start)
    print(f"TrickyBugs Python PUTs selected: {len(tasks)} / {report['python_puts']}")
    print(f"Selected with base tests: {sum(bool(t.base_tests) for t in tasks)}")
    model = GLMClient(cfg)
    for index, task in enumerate(tasks, 1):
        print(f"[{index}/{len(tasks)}] {task.task_id}", flush=True)
        try:
            print("  " + process_task(task, model, output_root, cfg, args.overwrite), flush=True)
        except Exception as exc:
            print(f"  error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    print("Generation/execution finished. Next: analyze_baseline.py prepare-review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
