"""Shared data, model, and execution utilities for the TrickCatcher baseline."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class TestCase:
    name: str
    stdin: str
    expected: str


@dataclass
class Task:
    dataset: str
    problem_id: str
    task_id: str
    source_path: str
    source: str
    specification: str
    base_tests: list[TestCase]
    extra_tests: list[TestCase]
    reference_programs: list[str]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def normalized_output(value: str) -> str:
    return value.replace("\r\n", "\n").strip()


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:160] or hashlib.sha1(value.encode()).hexdigest()[:12]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_test_dir(path: Path) -> list[TestCase]:
    tests: list[TestCase] = []
    if not path.is_dir():
        return tests
    for inp in sorted(path.glob("*.in")):
        out = inp.with_suffix(".out")
        if out.is_file():
            tests.append(TestCase(inp.name, read_text(inp), normalized_output(read_text(out))))
    return tests


def discover_trickybugs(root: Path, limit: int | None = None,
                        start: int = 0) -> list[Task]:
    paths = sorted((root / "problems").glob("*/buggy_programs/python/*.py"))
    paths = paths[start: None if limit is None else start + limit]
    cache: dict[Path, dict[str, Any]] = {}
    tasks: list[Task] = []
    for path in paths:
        problem_dir = path.parents[2]
        if problem_dir not in cache:
            description = problem_dir / "problem_description.txt"
            cache[problem_dir] = {
                "specification": read_text(description) if description.is_file() else "",
                "base_tests": load_test_dir(problem_dir / "original_test_cases"),
                "extra_tests": load_test_dir(problem_dir / "additional_test_cases"),
                "references": [str(p) for p in sorted((problem_dir / "reference_programs").glob("*.cpp"))],
            }
        common = cache[problem_dir]
        task_id = path.relative_to(root).as_posix()
        tasks.append(Task(
            dataset="TrickyBugs", problem_id=problem_dir.name, task_id=task_id,
            source_path=str(path), source=read_text(path),
            specification=common["specification"],
            base_tests=common["base_tests"], extra_tests=common["extra_tests"],
            reference_programs=common["references"],
        ))
    return tasks


def run_process(args: list[str], *, stdin: str = "", timeout: int = 5,
                cwd: str | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            args, input=stdin, text=True, capture_output=True, timeout=timeout,
            cwd=cwd, env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        return {
            "status": "ok" if result.returncode == 0 else "error",
            "output": normalized_output(result.stdout),
            "stderr": result.stderr[-2000:], "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired as exc:
        return {"status": "timeout", "output": "", "stderr": str(exc), "returncode": None}
    except OSError as exc:
        return {"status": "spawn_error", "output": "", "stderr": str(exc), "returncode": None}


def normal_output(result: dict[str, Any]) -> str | None:
    return result.get("output") if result.get("status") == "ok" else None


def run_python(path: Path, stdin: str, timeout: int) -> dict[str, Any]:
    return run_process([sys.executable, "-I", str(path)], stdin=stdin,
                       timeout=timeout, cwd=str(path.parent))


def check_syntax(source: str) -> tuple[bool, str]:
    try:
        ast.parse(source)
        return True, ""
    except SyntaxError as exc:
        return False, f"{exc.msg} (line {exc.lineno})"


def evaluate_tests(program: Path, tests: list[TestCase], timeout: int,
                   stop_on_failure: bool = False) -> dict[str, Any]:
    passed = 0
    failures: list[dict[str, Any]] = []
    for test in tests:
        result = run_python(program, test.stdin, timeout)
        ok = result["status"] == "ok" and normal_output(result) == test.expected
        if ok:
            passed += 1
        else:
            failures.append({"test": test.name, "expected": test.expected, "result": result})
            if stop_on_failure:
                break
    return {"total": len(tests), "passed": passed,
            "all_passed": passed == len(tests), "failures": failures[:20]}


def compile_references(task: Task, cache_root: Path, count: int,
                       compiler: str = "g++") -> list[dict[str, Any]]:
    target_dir = cache_root / task.problem_id
    target_dir.mkdir(parents=True, exist_ok=True)
    compiled: list[dict[str, Any]] = []
    successful = 0
    for source_name in task.reference_programs:
        if successful >= count:
            break
        source = Path(source_name)
        exe = target_dir / f"{source.stem}.exe"
        if not exe.is_file() or exe.stat().st_mtime < source.stat().st_mtime:
            result = run_process([compiler, "-std=gnu++17", "-O2", str(source), "-o", str(exe)],
                                 timeout=120, cwd=str(target_dir))
            if result["status"] != "ok":
                compiled.append({"source": str(source), "executable": None, "compile": result})
                continue
        compiled.append({"source": str(source), "executable": str(exe),
                         "compile": {"status": "ok"}})
        successful += 1
    return compiled


def run_references(compiled: list[dict[str, Any]], stdin: str,
                   timeout: int) -> tuple[list[dict[str, Any]], str | None, str]:
    results: list[dict[str, Any]] = []
    outputs: list[str] = []
    for item in compiled:
        if not item.get("executable"):
            results.append({"source": item["source"], "result": item["compile"]})
            continue
        result = run_process([item["executable"]], stdin=stdin, timeout=timeout,
                             cwd=str(Path(item["executable"]).parent))
        results.append({"source": item["source"], "result": result})
        output = normal_output(result)
        if output is not None:
            outputs.append(output)
    if any(item["result"].get("status") != "ok" for item in results):
        return results, None, "reference_failure"
    if not outputs:
        return results, None, "no_normal_reference"
    counts = Counter(outputs)
    output, support = counts.most_common(1)[0]
    status = "consensus" if support == len(outputs) else "disagreement"
    return results, output if status == "consensus" else None, status


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLM_ENV_KEYS = (
    "GLM_API_KEY",
    "GLM_BASE_URL",
    "GLM_MODEL",
    "GLM_TURBO_MODEL",
)


def load_glm_settings() -> dict[str, str]:
    """Load the shared GLM API settings without overriding shell variables."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Missing python-dotenv; install the project dependencies") from exc
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return {key: os.getenv(key, "").strip() for key in GLM_ENV_KEYS}


def _is_placeholder(value: str) -> bool:
    lowered = value.lower()
    return not value or "replace_with" in lowered or "your_" in lowered or "<" in value


def glm_api_status() -> dict[str, Any]:
    """Return a redacted, network-free readiness report for the GLM API."""
    settings = load_glm_settings()
    missing = [key for key, value in settings.items() if _is_placeholder(value)]
    return {
        "ready": not missing,
        "api_key_configured": not _is_placeholder(settings["GLM_API_KEY"]),
        "base_url": settings["GLM_BASE_URL"],
        "model": settings["GLM_MODEL"],
        "turbo_model": settings["GLM_TURBO_MODEL"],
        "missing_or_placeholder": missing,
        "env_file": str(PROJECT_ROOT / ".env"),
    }


class GLMClient:
    """OpenAI-compatible GLM chat-completions client."""

    def __init__(self, cfg: dict[str, Any]):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing openai; install the project dependencies") from exc
        settings = load_glm_settings()
        missing = [key for key, value in settings.items() if _is_placeholder(value)]
        if missing:
            raise RuntimeError(
                "Configure these values in the project-root .env: " + ", ".join(missing)
            )
        self.model = settings["GLM_MODEL"]
        self.turbo_model = settings["GLM_TURBO_MODEL"]
        self.client = OpenAI(
            api_key=settings["GLM_API_KEY"],
            base_url=settings["GLM_BASE_URL"],
            timeout=float(cfg.get("api_timeout_seconds", 120)),
            max_retries=int(cfg.get("api_max_retries", 3)),
        )
        self.max_new_tokens = int(cfg.get("max_new_tokens", 1024))
        self.temperature = float(cfg.get("temperature", 0.8))
        self.top_p = float(cfg.get("top_p", 0.95))
        self.thinking_mode = str(cfg.get("api_thinking_mode", "disabled")).lower()
        if self.thinking_mode not in {"enabled", "disabled"}:
            raise ValueError("api_thinking_mode must be 'enabled' or 'disabled'")

    def generate(self, prompt: str, seed: int, *, use_turbo: bool = False) -> dict[str, Any]:
        model_name = self.turbo_model if use_turbo else self.model
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=seed,
            extra_body={"thinking": {"type": self.thinking_mode}},
        )
        usage = response.usage
        return {
            "text": response.choices[0].message.content or "",
            "prompt_tokens": int(usage.prompt_tokens) if usage else None,
            "completion_tokens": int(usage.completion_tokens) if usage else None,
            "total_tokens": int(usage.total_tokens) if usage else None,
            "seed": seed,
            "model": model_name,
            "response_id": response.id,
            "system_fingerprint": response.system_fingerprint,
            "finish_reason": response.choices[0].finish_reason,
            "thinking_mode": self.thinking_mode,
        }


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.I | re.S)
    code = max(blocks, key=len) if blocks else text
    code = re.split(r"\n(?:Explanation|说明|分析)\s*:", code, maxsplit=1)[0]
    return code.strip()


def variant_prompt(task: Task) -> str:
    return f"""You are a professional programming-contest participant skilled at finding bugs and logic flaws.
Given the problem description and Python program under test, repair any bug you find.
Reply with ONLY the complete repaired Python code, without Markdown or explanation.

PROBLEM DESCRIPTION:
{task.specification}

PROGRAM UNDER TEST:
{task.source}
"""


def generator_prompt(task: Task, count: int) -> str:
    return f"""Write a standalone Python 3 random input generator for the problem below.
Generate valid inputs satisfying every constraint, emphasizing boundaries and corner cases.
Use only the Python standard library and call random.seed(42).
Print exactly {count} JSON objects, one per line, each shaped as
{{"stdin":"the complete stdin text\\n"}}. Do not print expected outputs.
Reply with ONLY complete Python code, without Markdown or explanation.

PROBLEM DESCRIPTION:
{task.specification}
"""


def run_input_generator(source: str, count: int, timeout: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    syntax_ok, error = check_syntax(source)
    if not syntax_ok:
        return [], {"status": "syntax_error", "stderr": error}
    with tempfile.TemporaryDirectory(prefix="tc_generator_") as temp:
        script = Path(temp) / "generator.py"
        script.write_text(source, encoding="utf-8")
        result = run_python(script, "", max(timeout, 30))
    if result["status"] != "ok":
        return [], result
    cases: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in result["output"].splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        stdin = value.get("stdin") if isinstance(value, dict) else None
        if isinstance(stdin, str) and stdin not in seen:
            seen.add(stdin)
            cases.append({"stdin": stdin})
            if len(cases) >= count:
                break
    return cases, result


def task_metadata(task: Task) -> dict[str, Any]:
    value = asdict(task)
    value.pop("source", None)
    value["base_test_count"] = len(task.base_tests)
    value["extra_test_count"] = len(task.extra_tests)
    value.pop("base_tests", None)
    value.pop("extra_tests", None)
    return value
