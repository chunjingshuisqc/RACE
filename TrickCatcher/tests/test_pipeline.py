"""No-model integration test for review, metrics, and visualizations."""

from __future__ import annotations

import csv
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1]))

from analyze_baseline import (  # noqa: E402
    analyze,
    load_labels,
    load_task_bundle,
    paper_candidates_for,
    paper_protocol_analysis,
    prepare_review,
)
from tc_core import Task, atomic_json, write_jsonl  # noqa: E402
from run_baseline import generate_cases, process_task  # noqa: E402


def result(output: str) -> dict:
    return {"status": "ok", "output": output, "stderr": "", "returncode": 0}


def make_task(root: Path, number: int) -> None:
    directory = root / "raw" / f"task_{number}"
    directory.mkdir(parents=True)
    task_id = f"problems/p{number:05d}/buggy_programs/python/sol_0.py"
    atomic_json(directory / "task.json", {
        "dataset": "TrickyBugs", "problem_id": f"p{number:05d}",
        "task_id": task_id, "source_path": f"D:/fake/problems/p{number:05d}/buggy_programs/python/sol_0.py",
        "base_test_count": 2, "extra_test_count": 2,
    })
    variants = []
    for index in range(10):
        variants.append({
            "accepted_index": index,
            "base_tests": {"total": 2, "passed": 2, "all_passed": True},
            "extra_tests": {"total": 2, "passed": 1 if index == 0 else 2,
                            "all_passed": index != 0},
        })
    atomic_json(directory / "accepted_variants.json", variants)
    executions = [
        {"input_id": "input_000", "stdin": "valid\n", "put_result": result("wrong"),
         "reference_status": "consensus", "reference_output": "right",
         "variant_results": [{"variant_index": i, "result": result("right" if i < 6 else "wrong")}
                             for i in range(10)]},
        {"input_id": "input_001", "stdin": "invalid\n", "put_result": result("x"),
         "reference_status": "consensus", "reference_output": "z",
         "variant_results": [{"variant_index": i, "result": result("y" if i < 6 else "x")}
                             for i in range(10)]},
        {"input_id": "input_002", "stdin": "same\n", "put_result": result("same"),
         "reference_status": "consensus", "reference_output": "same",
         "variant_results": [{"variant_index": i, "result": result("same")}
                             for i in range(10)]},
    ]
    write_jsonl(directory / "executions.jsonl", executions)
    atomic_json(directory / "status.json", {
        "status": "complete", "task_id": task_id,
        "generator_eligible": True, "generated_inputs": 3, "executed_inputs": 3,
        "generator_status": {
            "generator_eligible": True, "attempts": 1,
            "executable_attempts": 1, "usable_attempts": 1,
            "generated_unique_inputs": 3, "target_inputs": 3, "input_yield": 1.0,
        },
    })


def make_generator_failed(root: Path, number: int) -> None:
    directory = root / "raw" / f"task_{number}"
    directory.mkdir(parents=True)
    task_id = f"problems/p{number:05d}/buggy_programs/python/sol_0.py"
    atomic_json(directory / "task.json", {
        "dataset": "TrickyBugs", "problem_id": f"p{number:05d}",
        "task_id": task_id,
    })
    atomic_json(directory / "status.json", {
        "status": "generator_failed", "task_id": task_id,
        "generator_eligible": False, "generated_inputs": 0,
        "exclusion_reason": "no executable generator produced parseable inputs",
        "generator_status": {
            "generator_eligible": False, "attempts": 3,
            "executable_attempts": 0, "usable_attempts": 0,
            "generated_unique_inputs": 0, "target_inputs": 100, "input_yield": 0.0,
        },
    })


def main() -> None:
    persistent = os.environ.get("TC_TEST_OUTPUT")
    context = (contextlib.nullcontext(persistent) if persistent
               else tempfile.TemporaryDirectory(prefix="tc_analysis_test_"))
    with context as temp:
        root = Path(temp)
        root.mkdir(parents=True, exist_ok=True)
        generator_task = SimpleNamespace(specification="Read one integer.")

        class SequenceModel:
            def __init__(self, outputs: list[str]) -> None:
                self.outputs = outputs
                self.index = 0

            def generate(self, prompt: str, seed: int) -> dict:
                output = self.outputs[min(self.index, len(self.outputs) - 1)]
                self.index += 1
                return {"text": output, "prompt_tokens": 1,
                        "completion_tokens": 1, "seed": seed}

        valid_source = """import json
for i in range(3):
    print(json.dumps({"stdin": f"{i}\\n"}))
"""
        eligible_dir = root / "generator_gate_eligible"
        cases, generator_status = generate_cases(
            generator_task,
            SequenceModel(["not valid python !!!", valid_source]),
            eligible_dir,
            {"inputs_per_task": 3, "generator_attempts": 3,
             "execution_timeout_seconds": 2, "seed": 42},
        )
        assert len(cases) == 3, cases
        assert eligible_dir.is_dir()
        assert generator_status["generator_eligible"], generator_status
        assert generator_status["executable_attempts"] == 1, generator_status
        failed_dir = root / "generator_gate_failed"
        failed_dir.mkdir()
        failed_cases, failed_status = generate_cases(
            generator_task,
            SequenceModel(["not valid python !!!"]),
            failed_dir,
            {"inputs_per_task": 3, "generator_attempts": 3,
             "execution_timeout_seconds": 2, "seed": 42},
        )
        assert failed_cases == [], failed_cases
        assert not failed_status["generator_eligible"], failed_status
        process_root = root / "process_gate"
        gate_task = Task(
            dataset="TrickyBugs", problem_id="p99999",
            task_id="problems/p99999/buggy_programs/python/sol_0.py",
            source_path="D:/fake/sol_0.py", source="print(0)",
            specification="Read one integer.",
            base_tests=[], extra_tests=[], reference_programs=[],
        )
        process_result = process_task(
            gate_task, SequenceModel(["not valid python !!!"]), process_root,
            {"inputs_per_task": 3, "generator_attempts": 1,
             "execution_timeout_seconds": 2, "seed": 42,
             "require_usable_generator": True}, False,
        )
        assert process_result == "generator_failed (excluded)", process_result
        process_task_dir = next((process_root / "raw").iterdir())
        process_status = json.loads((process_task_dir / "status.json").read_text())
        assert process_status["status"] == "generator_failed", process_status
        assert not (process_task_dir / "accepted_variants.json").exists()
        variant_gate_root = root / "variant_gate"
        variant_result = process_task(
            gate_task, SequenceModel([valid_source, "not valid python !!!"]),
            variant_gate_root,
            {"inputs_per_task": 3, "generator_attempts": 1,
             "variant_pool_size": 1, "max_variant_attempts": 1,
             "execution_timeout_seconds": 2, "seed": 42,
             "require_usable_generator": True}, False,
        )
        assert variant_result == "complete", variant_result
        variant_task_dir = next((variant_gate_root / "raw").iterdir())
        variant_status = json.loads((variant_task_dir / "status.json").read_text())
        assert variant_status["status"] == "complete", variant_status
        assert variant_status["accepted_variants"] == 0, variant_status
        assert (variant_task_dir / "executions.jsonl").exists()
        variant_status["status"] = "variant_failed"
        (variant_task_dir / "status.json").write_text(
            json.dumps(variant_status), encoding="utf-8"
        )
        restored = process_task(
            gate_task, SequenceModel(["should not be requested"]),
            variant_gate_root,
            {"inputs_per_task": 3, "generator_attempts": 1,
             "variant_pool_size": 1, "max_variant_attempts": 1,
             "execution_timeout_seconds": 2, "seed": 42,
             "require_usable_generator": True}, False,
        )
        assert restored == "skipped (complete; restored actual-k result)", restored
        make_task(root, 1)
        make_task(root, 2)
        make_generator_failed(root, 3)
        review = prepare_review(root)
        assert review == {"rows": 4, "labeled": 0, "pending": 4}, review
        path = root / "manual" / "input_validity.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        for row in rows:
            row["input_valid"] = "1" if row["input_id"] == "input_000" else "0"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        summary = analyze(root)
        assert summary["final_metrics_ready"], summary
        detection = summary["bug_detection_effectiveness"]
        assert detection["precision"] == 0.5, detection
        assert detection["recall"] == 1.0, detection
        assert detection["f1"] == 2 / 3, detection
        assert detection["tp_puts"] == 2, detection
        assert detection["fn_puts"] == 0, detection
        assert detection["put_detection_rate"] == 1.0, detection
        assert summary["metric_protocol"] == (
            "candidate-level macro precision and PUT-level recall TP/(TP+FN)"
        )
        assert [row["k"] for row in summary["variant_cardinality_sensitivity"]] == [10]
        assert summary["variant_cardinality_sensitivity"][0]["recall"] == 1.0
        assert (root / "tables" / "candidate_evidence_records.csv").is_file()
        assert not (root / "tables" / "variant_cardinality_combination_metrics.csv").exists()
        scope = summary["experiment_scope"]
        assert scope["attempted_puts"] == 3, scope
        assert scope["included_generator_eligible_puts"] == 2, scope
        assert scope["excluded_generator_puts"] == 1, scope
        assert summary["completed_tasks"] == 2, summary
        tie_execution = {
            "put_result": result("put"),
            "variant_results": [
                {"variant_index": index, "result": result("right" if index < 5 else "other")}
                for index in range(10)
            ],
        }
        tie_candidates = paper_candidates_for(tie_execution, tuple(range(10)))
        assert {item["oracle"] for item in tie_candidates} == {"right", "other"}
        assert {item["support"] for item in tie_candidates} == {5}
        limited = load_task_bundle(root / "raw" / "task_1")
        limited["variants"] = limited["variants"][:6]
        for execution in limited["executions"]:
            execution["variant_results"] = [
                item for item in execution["variant_results"]
                if item["variant_index"] < 6
            ]
        _, limited_puts, limited_summary = paper_protocol_analysis(
            [limited], load_labels(root), 10
        )
        assert limited_puts[0]["actual_k"] == 6, limited_puts[0]
        assert limited_puts[0]["candidate_oracle_rows"] > 0, limited_puts[0]
        assert limited_summary["put_count"] == 1, limited_summary
        assert limited_summary["actual_k_distribution"] == {"6": 1}, limited_summary
        assert summary["variant_quality"]["accepted_variants"] == 20
        assert summary["variant_quality"]["observed_buggy_variants"] == 10
        assert summary["useful_buggy"]["useful_buggy_variants"] == 2
        reduced = analyze(root, include_variant_diagnostics=False)
        assert reduced["analysis_sections"] == [
            "bug_detection_effectiveness",
        ]
        assert "variant_quality" not in reduced and "useful_buggy" not in reduced
        assert len(reduced["figures"]) == 1, reduced["figures"]
        assert len(summary["figures"]) == 3, summary["figures"]
        for figure in summary["figures"]:
            assert Path(figure).is_file() and Path(figure).stat().st_size > 1000
        paired_id = "problems/p00001/buggy_programs/python/sol_0.py"
        paired_review = prepare_review(root, {paired_id})
        assert paired_review == {"rows": 2, "labeled": 2, "pending": 0}, paired_review
        paired = analyze(root, included_task_ids={paired_id})
        assert paired["completed_tasks"] == 1, paired
        assert paired["experiment_scope"]["fixed_evaluation_set"], paired
        assert paired["experiment_scope"]["evaluation_puts"] == 1, paired
        assert paired["experiment_scope"]["included_generator_eligible_puts"] == 1, paired
        print("baseline analysis integration: PASS")
        print("output=", root)
        print(json.dumps({
            "bug_detection_effectiveness": detection,
            "figures": len(summary["figures"]),
        }, indent=2))


if __name__ == "__main__":
    main()
