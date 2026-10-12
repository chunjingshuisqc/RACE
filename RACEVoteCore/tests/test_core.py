"""No-model tests for feature profiles and weighted voting."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

CORE = Path(__file__).resolve().parents[1]
ROOT = CORE.parent
sys.path.insert(0, str(CORE))

from race_vote_core import (  # noqa: E402
    PROFILE_FEATURES,
    baseline_run,
    compute_rc_weights,
    process_task,
    resolve_features,
    run_llm_revote,
    weighted_candidates_for,
)
from tc_core import Task  # noqa: E402


def result(output: str, status: str = "ok") -> dict:
    return {"status": status, "output": output}


def main() -> None:
    for name, expected in PROFILE_FEATURES.items():
        config = json.loads((ROOT / name / "config.json").read_text(encoding="utf-8"))
        assert resolve_features(config) == expected
        assert Path(config["output_dir"]).name == name
        assert config["include_variant_diagnostics"] == (
            name == "RACE-Vote"
        )

    variants = [{"accepted_index": index} for index in range(3)]
    executions = [
        {"variant_results": [
            {"variant_index": 0, "result": result("A")},
            {"variant_index": 1, "result": result("A")},
            {"variant_index": 2, "result": result("B")},
        ]}
        for _ in range(4)
    ]
    weights = compute_rc_weights(variants, executions)
    assert weights[2]["weight"] > weights[0]["weight"], weights
    assert weights[0]["mean_behavioral_correlation"] > weights[2][
        "mean_behavioral_correlation"
    ], weights

    execution = {
        "put_result": result("PUT"),
        "variant_results": [
            {"variant_index": 0, "result": result("A")},
            {"variant_index": 1, "result": result("A")},
            {"variant_index": 2, "result": result("B")},
        ],
        "_race_weights": {"0": 0.4, "1": 0.4, "2": 1.5},
        "_race_multipliers": {},
    }
    candidates = weighted_candidates_for(execution, (0, 1, 2))
    assert [candidate["oracle"] for candidate in candidates] == ["B"], candidates

    class MockModel:
        def generate(self, prompt: str, seed: int, *, use_turbo: bool = False) -> dict:
            assert use_turbo
            return {
                "text": '{"multipliers":{"0":0.1,"1":8,"2":1.5},"reason":"test"}',
                "seed": seed,
            }

    revote = run_llm_revote(
        SimpleNamespace(specification="spec"), variants, executions, weights,
        MockModel(), 7,
    )
    assert revote["status"] == "ok", revote
    assert revote["multipliers"] == {0: 0.25, 1: 4.0, 2: 1.5}, revote

    original_cases = baseline_run.generate_cases
    original_variants = baseline_run.generate_variants
    original_execute = baseline_run.execute_cases
    import race_vote_core as module
    original_revote = module.run_llm_revote
    counters = {"cases": 0, "variants": 0, "execute": 0}
    variant_targets: list[int] = []

    def fake_cases(task, model, directory, cfg):
        counters["cases"] += 1
        return ([{"input_id": "input_000", "stdin": "1\n"}], {
            "generator_eligible": True, "attempts": 1,
            "executable_attempts": 1, "usable_attempts": 1,
            "generated_unique_inputs": 1, "target_inputs": 1, "input_yield": 1.0,
        })

    def fake_variants(task, model, directory, cfg):
        counters["variants"] += 1
        variant_targets.append(int(cfg.get("variant_pool_size", 2)))
        values = [
            {"accepted_index": index, "path": str(directory / f"v{index}.py"),
             "source_sha1": f"{counters['variants']}-{index}",
             "base_tests": {"total": 0, "passed": 0, "all_passed": True},
             "extra_tests": {"total": 0, "passed": 0, "all_passed": True}}
            for index in range(int(cfg.get("variant_pool_size", 2)))
        ]
        return values, list(values)

    def fake_execute(task, variants, cases, directory, output_root, cfg):
        counters["execute"] += 1
        return [{
            "input_id": "input_000", "stdin": "1\n",
            "put_result": result("PUT"),
            "reference_status": "consensus", "reference_output": "A",
            "variant_results": [
                {"variant_index": item["accepted_index"], "result": result("A")}
                for item in variants
            ],
        }]

    def fake_revote(task, variants, executions, weights, model, seed):
        return {"status": "ok", "multipliers": {index: 1.0 for index in weights}}

    baseline_run.generate_cases = fake_cases
    baseline_run.generate_variants = fake_variants
    baseline_run.execute_cases = fake_execute
    module.run_llm_revote = fake_revote
    try:
        task = Task(
            dataset="TrickyBugs", problem_id="p00001", task_id="task/put.py",
            source_path="D:/fake/put.py", source="print(0)", specification="spec",
            base_tests=[], extra_tests=[], reference_programs=[],
        )
        with tempfile.TemporaryDirectory(prefix="race_vote_core_test_") as temp:
            for name, expected in PROFILE_FEATURES.items():
                counters.update({"cases": 0, "variants": 0, "execute": 0})
                variant_targets.clear()
                config = json.loads((ROOT / name / "config.json").read_text())
                config.update({"output_dir": str(Path(temp) / name),
                               "variant_pool_size": 2, "inputs_per_task": 2})
                config.pop("reuse_mode", None)
                config.pop("reuse_source_output", None)
                assert process_task(task, MockModel(), Path(config["output_dir"]),
                                    config, False) == "complete"
                status_file = next((Path(config["output_dir"]) / "raw").glob("*/status.json"))
                status = json.loads(status_file.read_text())
                assert status["features"] == expected, status
                stages = 2 if expected["ce_coevo"] else 1
                assert counters == {"cases": stages, "variants": stages,
                                    "execute": stages}, (name, counters)
                expected_targets = [1, 1] if expected["ce_coevo"] else [2]
                assert variant_targets == expected_targets, (name, variant_targets)

            source_root = Path(temp) / "reuse-source"
            source_task_dir = source_root / "raw" / module.safe_id(task.task_id)
            source_stage1 = source_task_dir / "stage_1_exploration"
            source_stage1.mkdir(parents=True)
            source_variants = [
                {"accepted_index": index, "path": str(source_task_dir / f"v{index}.py"),
                 "source_sha1": f"source-{index}",
                 "base_tests": {"total": 0, "passed": 0, "all_passed": True},
                 "extra_tests": {"total": 0, "passed": 0, "all_passed": True}}
                for index in range(2)
            ]
            source_executions = [{
                "input_id": "input_000", "stdin": "1\n",
                "put_result": result("PUT"),
                "reference_status": "consensus", "reference_output": "A",
                "variant_results": [
                    {"variant_index": item["accepted_index"], "result": result("A")}
                    for item in source_variants
                ],
            }]
            (source_task_dir / "task.json").write_text(json.dumps({
                "task_id": task.task_id, "problem_id": task.problem_id,
                "source_path": task.source_path,
            }), encoding="utf-8")
            (source_task_dir / "accepted_variants.json").write_text(
                json.dumps(source_variants), encoding="utf-8"
            )
            baseline_run.write_jsonl(source_task_dir / "executions.jsonl", source_executions)
            baseline_run.write_jsonl(source_task_dir / "inputs.jsonl", [{
                "input_id": "input_000", "stdin": "1\n",
            }])
            (source_task_dir / "status.json").write_text(json.dumps({
                "status": "complete", "task_id": task.task_id,
                "generator_eligible": True, "accepted_variants": 2,
                "generated_inputs": 1, "generator_status": {
                    "generator_eligible": True, "generated_unique_inputs": 1,
                },
            }), encoding="utf-8")
            source_weights = compute_rc_weights(source_variants, source_executions)
            (source_task_dir / "rc_weights.json").write_text(
                json.dumps({"enabled": True, "weights": source_weights}), encoding="utf-8"
            )
            (source_stage1 / "accepted_variants.json").write_text(
                json.dumps(source_variants[:1]), encoding="utf-8"
            )
            baseline_run.write_jsonl(source_stage1 / "inputs.jsonl", [{
                "input_id": "input_000", "stdin": "1\n",
            }])

            for name, mode in (
                ("Ablation-No-RV", "reuse_full_disable_revote"),
                ("Ablation-No-RC", "reuse_full_revote_only"),
            ):
                counters.update({"cases": 0, "variants": 0, "execute": 0})
                config = json.loads((ROOT / name / "config.json").read_text())
                config.update({
                    "output_dir": str(Path(temp) / f"reuse-{name}"),
                    "reuse_source_output": str(source_root),
                    "reuse_mode": mode,
                    "variant_pool_size": 2, "inputs_per_task": 2,
                })
                reuse_result = process_task(
                    task, MockModel(), Path(config["output_dir"]), config, False
                )
                assert "reused RACE-Vote artifacts" in reuse_result, reuse_result
                assert counters == {"cases": 0, "variants": 0, "execute": 0}, counters
                derived_dir = next((Path(config["output_dir"]) / "raw").iterdir())
                manifest = json.loads((derived_dir / "reuse_manifest.json").read_text())
                assert manifest["analysis_source_task_dir"] == str(source_task_dir), manifest
                bundle = module.baseline_analysis.load_task_bundle(derived_dir)
                assert bundle["data_directory"] == source_task_dir, bundle

            counters.update({"cases": 0, "variants": 0, "execute": 0})
            config = json.loads((ROOT / "Ablation-No-CE" / "config.json").read_text())
            config.update({
                "output_dir": str(Path(temp) / "reuse-Ablation-No-CE"),
                "reuse_source_output": str(source_root),
                "reuse_mode": "reuse_stage1_without_feedback",
                "variant_pool_size": 2, "inputs_per_task": 2,
            })
            no_ce_result = process_task(
                task, MockModel(), Path(config["output_dir"]), config, False
            )
            assert "regenerated generic stage 2" in no_ce_result, no_ce_result
            assert counters == {"cases": 1, "variants": 1, "execute": 1}, counters

            def insufficient_variants(task, model, directory, cfg):
                target = int(cfg.get("variant_pool_size", 2))
                values = fake_variants(task, model, directory, cfg)[0][:-1]
                return values, list(values)

            counters.update({"cases": 0, "variants": 0, "execute": 0})
            baseline_run.generate_variants = insufficient_variants
            config = json.loads((ROOT / "Ablation-TrickCatcher" / "config.json").read_text())
            config.update({"output_dir": str(Path(temp) / "actual-k-baseline"),
                           "variant_pool_size": 2, "inputs_per_task": 2})
            result_text = process_task(
                task, MockModel(), Path(config["output_dir"]), config, False
            )
            assert result_text == "complete", result_text
            actual_status_file = next(
                (Path(config["output_dir"]) / "raw").glob("*/status.json")
            )
            actual_status = json.loads(actual_status_file.read_text())
            assert actual_status["status"] == "complete", actual_status
            assert actual_status["accepted_variants"] == 1, actual_status
            assert counters["execute"] == 1, counters
            actual_status["status"] = "variant_failed"
            actual_status_file.write_text(json.dumps(actual_status), encoding="utf-8")
            counters.update({"cases": 0, "variants": 0, "execute": 0})
            resumed = process_task(
                task, MockModel(), Path(config["output_dir"]), config, False
            )
            assert resumed == "complete (resumed with actual k)", resumed
            assert counters == {"cases": 0, "variants": 0, "execute": 1}, counters
            resumed_status = json.loads(actual_status_file.read_text())
            assert resumed_status["status"] == "complete", resumed_status

            counters.update({"cases": 0, "variants": 0, "execute": 0})
            config = json.loads((ROOT / "RACE-Vote" / "config.json").read_text())
            config.update({"output_dir": str(Path(temp) / "actual-k-race"),
                           "variant_pool_size": 2, "inputs_per_task": 2})
            result_text = process_task(
                task, MockModel(), Path(config["output_dir"]), config, False
            )
            assert result_text == "complete", result_text
            actual_status_file = next(
                (Path(config["output_dir"]) / "raw").glob("*/status.json")
            )
            actual_status = json.loads(actual_status_file.read_text())
            assert actual_status["status"] == "complete", actual_status
            assert actual_status["accepted_variants"] == 1, actual_status
            assert counters == {"cases": 2, "variants": 2, "execute": 1}, counters
            assert (actual_status_file.parent / "llm_revote.json").exists()
    finally:
        baseline_run.generate_cases = original_cases
        baseline_run.generate_variants = original_variants
        baseline_run.execute_cases = original_execute
        module.run_llm_revote = original_revote
    print("RACE-Vote core tests: PASS")


if __name__ == "__main__":
    main()
