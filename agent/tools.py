"""
agent/tools.py — tool definitions and executor for the scheduling research agent.

Four tools: read, write_scheduler, run_experiment, run_python.  Each returns a
plain string (what the LLM sees as the tool result).  run_experiment caches the
full per-job / per-site logs of every run; run_python is the general-purpose way
to look at them, so there are no canned report tools to learn.
"""
from __future__ import annotations

import ast
import concurrent.futures
import importlib
import importlib.util
import inspect
import json
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── root of the project ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AGENT_SCHEDULER_PATH = ROOT / "schedulers" / "agent_scheduler.py"
ANALYSIS_DIR = ROOT / "analysis"
RUNS_DIR = ANALYSIS_DIR / "runs"
_ANALYSIS_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

# Hard real-time ceiling on every simulation run — not a per-call option.  A
# policy that cannot decide inside it is failed rather than waited on, so a
# scheduler doing heavy per-decision work (an MILP solve per batch) is out of
# scope by construction, not by accident.
WALL_TIMEOUT_S = 180.0

_CONFIG_MAP = {
    "environment": str(ROOT / "config" / "environment.yaml"),
    "small_scale":  str(ROOT / "config" / "small_scale.yaml"),
    "middle_scale": str(ROOT / "config" / "middle_scale.yaml"),
    "large_scale":  str(ROOT / "config" / "large_scale.yaml"),
}
_BASELINE_MAP = {
    "MinT": ("schedulers.min_t", "MinTScheduler"),
    "MinE": ("schedulers.min_e", "MinEScheduler"),
    "MCT": ("schedulers.min_ct", "MCTScheduler"),
    "DIANA": ("schedulers.diana", "DIANAScheduler"),
    "LeastLoaded": ("schedulers.least_loaded", "LeastLoadedScheduler"),
    "Random": ("schedulers.random_sched", "RandomScheduler"),
}
_ALLOWED_READ_PATHS: List[Path] = [ROOT / "AGENT_INSTRUCTIONS.md"]
_ALLOWED_READ_DIRS:  List[Path] = [
    ROOT / "schedulers",
    ROOT / "config",
    ROOT / "env",
    ANALYSIS_DIR,
]


# ── tool schema definitions (Anthropic / canonical format) ───────────────────

TOOL_DEFS: List[dict] = [
    {
        "name": "read",
        "description": (
            "Read a file, or list a directory. Available: AGENT_INSTRUCTIONS.md "
            "and anything under schedulers/, env/, config/, analysis/."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the project root"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_scheduler",
        "description": (
            "Save a scheduler to schedulers/agent_scheduler.py, to be benchmarked "
            "as scheduler='agent'. The code must define one concrete "
            "SchedulerInterface subclass; it is syntax- and import-checked here."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python source (imports + class definition)",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "run_experiment",
        "description": (
            "Run the simulation and return its summary metrics. The full per-job "
            "and per-site logs are kept for run_python, keyed 'scheduler/config'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scheduler": {
                    "type": "string",
                    "enum": ["agent", "MinT", "MinE", "MCT", "DIANA", "LeastLoaded", "Random"],
                    "description": "'agent' is your last write_scheduler; the rest are baselines",
                },
                "config": {
                    "type": "string",
                    "enum": ["small_scale", "middle_scale", "large_scale", "environment"],
                    "default": "small_scale",
                    "description": (
                        "Grid to run on. small_scale is the default working "
                        "grid; large_scale is ~5x the jobs and sites"
                    ),
                },
                "until_s": {
                    "type": "number",
                    "default": 7200.0,
                    "description": "Simulation end time in seconds",
                },
                "defer_retry_s": {
                    "type": "number",
                    "description": (
                        "Sim-seconds between retries of a job you deferred by "
                        "returning None. Default 1e-6 livelocks a batching "
                        "scheduler; pass ~1.0 if schedule() ever returns None"
                    ),
                },
            },
            "required": ["scheduler"],
        },
    },
    {
        "name": "run_python",
        "description": (
            "Run Python in the project and return its stdout. Pre-bound: "
            "runs — {'scheduler/config': {'summary': dict, 'job_stats': [per-job "
            "dict], 'site_stats': [per-site dict]}} for every experiment run so "
            "far; summary / job_stats / site_stats — the most recent run. "
            "json, math, statistics, collections are imported, and the project "
            "is importable (e.g. `from env import HEPSimulator`). Use print()."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source. Use print() for output."},
                "name": {
                    "type": "string",
                    "description": "Optional basename; the script is kept at analysis/{name}.py",
                },
                "timeout_s": {"type": "number", "default": 120},
            },
            "required": ["code"],
        },
    },
]


# ── ToolCall / ChatResult data classes (used by loop.py) ─────────────────────

@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ChatResult:
    text: Optional[str]
    tool_calls: List[ToolCall]
    stop_reason: str
    raw_content: Any   # backend-native object stored for message-building


# ── Executor ─────────────────────────────────────────────────────────────────

class ToolExecutor:
    """Executes tool calls and caches simulation results."""

    _MAX_OUTPUT = 6000  # chars — keeps individual tool results from flooding context

    def __init__(self) -> None:
        # (scheduler, config) → {"summary": ..., "job_stats": ..., "site_stats": ...}
        self._cache: Dict[Tuple[str, str], dict] = {}
        self._agent_class_name: Optional[str] = None
        self._last_key: Optional[Tuple[str, str]] = None
        self._snippet_count = 0
        # Drop any runs left on disk by an earlier session, so `runs` inside
        # run_python only ever holds experiments from this one.
        if RUNS_DIR.is_dir():
            for stale in RUNS_DIR.iterdir():
                if stale.is_file():
                    stale.unlink()

    # ------------------------------------------------------------------ #
    def execute(self, name: str, args: dict) -> str:
        try:
            if   name == "read":            return self._truncate(self._read(**args))
            elif name == "write_scheduler": return self._write_scheduler(**args)
            elif name == "run_experiment":  return self._run_experiment(**args)
            elif name == "run_python":      return self._truncate(self._run_python(**args))
            else:
                return f"ERROR: unknown tool '{name}'"
        except TypeError as exc:
            # Almost always a bad argument name rather than a bug in the tool —
            # say which ones exist instead of returning a traceback.
            accepted = next(
                (list(t["input_schema"]["properties"]) for t in TOOL_DEFS if t["name"] == name),
                [],
            )
            return (
                f"ERROR: bad arguments for {name}: {exc}\n"
                f"Accepted parameters: {accepted}"
            )
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"

    def _truncate(self, text: str) -> str:
        if len(text) <= self._MAX_OUTPUT:
            return text
        return text[: self._MAX_OUTPUT] + f"\n[... truncated at {self._MAX_OUTPUT} chars]"

    # ------------------------------------------------------------------ #
    # read — a file, or a directory listing
    # ------------------------------------------------------------------ #
    def _read(self, path: str) -> str:
        p = (ROOT / path).resolve()
        allowed = any(p == ap for ap in _ALLOWED_READ_PATHS) or any(
            _is_under(p, d) or p == d.resolve() for d in _ALLOWED_READ_DIRS
        )
        if not allowed:
            return (
                f"ERROR: '{path}' is outside the readable paths "
                "(AGENT_INSTRUCTIONS.md, schedulers/, env/, config/, analysis/)."
            )
        if p.is_dir():
            files = sorted(f.relative_to(ROOT) for f in p.rglob("*") if f.is_file())
            return "\n".join(str(f) for f in files) or "(empty directory)"
        if not p.exists():
            return f"ERROR: not found: {path}"
        return p.read_text()

    # ------------------------------------------------------------------ #
    # write_scheduler
    # ------------------------------------------------------------------ #
    def _write_scheduler(self, code: str, class_name: Optional[str] = None) -> str:
        try:
            ast.parse(code)
        except SyntaxError as e:
            return f"ERROR: SyntaxError at line {e.lineno}: {e.msg}"

        AGENT_SCHEDULER_PATH.write_text(code)

        # Import check — catches bad imports, missing base class, etc. — and, when
        # no class_name was given, tells us which class to instantiate later.
        _purge_agent_module()
        try:
            mod = _load_module(AGENT_SCHEDULER_PATH, "_agent_check")
        except Exception as e:
            return f"ERROR during import: {type(e).__name__}: {e}\n{traceback.format_exc(limit=6)}"

        if class_name is not None:
            if getattr(mod, class_name, None) is None:
                return f"ERROR: class '{class_name}' not found in the code."
            resolved = class_name
        else:
            found = _concrete_schedulers(mod)
            if not found:
                return (
                    "ERROR: no concrete SchedulerInterface subclass found. Define "
                    "one class inheriting from SchedulerInterface (imported from "
                    "env.interfaces) that implements schedule()."
                )
            resolved = found[-1]

        self._agent_class_name = resolved
        return (
            f"OK — '{resolved}' saved to schedulers/agent_scheduler.py. "
            f"Benchmark it with run_experiment(scheduler='agent')."
        )

    # ------------------------------------------------------------------ #
    # run_experiment
    # ------------------------------------------------------------------ #
    def _run_experiment(
        self,
        scheduler: str,
        config: str = "small_scale",
        until_s: float = 7200.0,
        defer_retry_s: Optional[float] = None,
        wall_timeout_s: Optional[float] = None,
    ) -> str:
        import logging
        logging.disable(logging.WARNING)  # suppress simulator INFO logs during bench

        timeout = WALL_TIMEOUT_S if wall_timeout_s is None else float(wall_timeout_s)

        try:
            sched_obj = self._load_scheduler(scheduler)
        except Exception as e:
            return f"ERROR loading scheduler '{scheduler}': {e}"

        config_path = _CONFIG_MAP.get(config)
        if config_path is None:
            return f"ERROR: unknown config '{config}'. Choose from: {list(_CONFIG_MAP)}"

        from env import HEPSimulator
        try:
            sim = HEPSimulator.from_yaml(config_path, scheduler=sched_obj)
            if defer_retry_s is not None:
                # Batching schedulers hold jobs by returning None; each deferral
                # costs one event, so they need a coarser retry than the default.
                sim.defer_retry_s = float(defer_retry_s)
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = pool.submit(sim.run, until_s)
            pool.shutdown(wait=False)
            try:
                future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
                return (
                    f"ERROR: simulation exceeded the {timeout:.0f}s wall-clock "
                    f"limit. Either schedule() loops forever, or it does too much "
                    f"work per decision. Make the policy cheaper per job, or pass "
                    f"a larger wall_timeout_s for a bigger config."
                )
        except Exception as e:
            return f"ERROR during simulation: {type(e).__name__}: {e}\n{traceback.format_exc(limit=10)}"
        finally:
            logging.disable(logging.NOTSET)

        data = {
            "summary":    sim.summary(),
            "job_stats":  sim.job_stats(),
            "site_stats": sim.site_stats(),
        }
        self._cache[(scheduler, config)] = data
        self._last_key = (scheduler, config)
        self._dump_run(scheduler, config, data)

        return json.dumps(data["summary"], indent=2, default=_json_default)

    # ------------------------------------------------------------------ #
    # run_python
    # ------------------------------------------------------------------ #
    def _run_python(
        self,
        code: str,
        name: Optional[str] = None,
        timeout_s: float = 120,
    ) -> str:
        try:
            ast.parse(code)
        except SyntaxError as e:
            return f"ERROR: SyntaxError at line {e.lineno}: {e.msg}"

        if name is None:
            self._snippet_count += 1
            name = f"snippet_{self._snippet_count:03d}"
        elif not _ANALYSIS_NAME_RE.match(name):
            return f"ERROR: invalid name '{name}'. Use alphanumeric, underscore or hyphen."

        ANALYSIS_DIR.mkdir(exist_ok=True)
        path = ANALYSIS_DIR / f"{name}.py"
        path.write_text(_SCAFFOLD + _FUTURE_IMPORT_RE.sub("", code))

        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(ROOT),
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: analysis/{name}.py timed out after {timeout_s}s."

        out = proc.stdout
        if proc.returncode != 0:
            out += f"\n[exit={proc.returncode}] [stderr]\n{proc.stderr}"
        elif proc.stderr:
            out += f"\n[stderr]\n{proc.stderr}"
        return out or "(no output — did you print() anything?)"

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _dump_run(self, scheduler: str, config: str, data: dict) -> None:
        """Persist a run's logs so run_python subprocesses can load them."""
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        # A scheduler may be named by path, which contains separators and '.py'.
        key = f"{_slug(scheduler)}__{config}"
        (RUNS_DIR / f"{key}.json").write_text(
            json.dumps(data, default=_json_default)
        )
        (RUNS_DIR / "latest").write_text(key)

    def _load_scheduler(self, name: str):
        if name in _BASELINE_MAP:
            mod_name, cls_name = _BASELINE_MAP[name]
            # Force fresh import in case scheduler package was re-imported
            mod = importlib.import_module(mod_name)
            return getattr(mod, cls_name)()
        elif name == "agent":
            if not AGENT_SCHEDULER_PATH.exists():
                raise FileNotFoundError(
                    "No agent scheduler found. Use write_scheduler first."
                )
            _purge_agent_module()
            mod = _load_module(AGENT_SCHEDULER_PATH, "agent_scheduler")
            cls_name = self._agent_class_name
            if cls_name is None:
                found = _concrete_schedulers(mod)
                if not found:
                    raise ValueError(
                        "schedulers/agent_scheduler.py defines no concrete "
                        "SchedulerInterface subclass. Use write_scheduler first."
                    )
                cls_name = found[-1]
            cls = getattr(mod, cls_name)
            return cls()
        elif name.endswith(".py"):
            return _load_scheduler_file(name)
        else:
            raise ValueError(
                f"Unknown scheduler '{name}'. Valid: agent, "
                f"{', '.join(_BASELINE_MAP)}, a path to a .py file under "
                f"schedulers/ or runs/"
            )


# ── helpers ──────────────────────────────────────────────────────────────────

# `from __future__ import ...` is only legal as a file's first statement.
# _SCAFFOLD always precedes the model's snippet below, so if the model
# reflexively copies this line from the scheduler-writing prompts (where it
# IS required), it would land second and make the whole script a guaranteed
# SyntaxError — not something a retry can fix.  Harmless to drop: a flat
# analysis snippet has no forward-referenced type hints that need it.
_FUTURE_IMPORT_RE = re.compile(r"^[ \t]*from[ \t]+__future__[ \t]+import[^\n]*\n?", re.MULTILINE)

# Prepended to every run_python snippet: makes the project importable and binds
# the logs of every experiment run so far.  Written as source (not injected data)
# so a run's logs are loaded from disk rather than inlined into the script.
_SCAFFOLD = '''\
import collections, json, math, statistics, sys
from pathlib import Path

sys.path.insert(0, {root!r})
_runs_dir = Path({runs!r})
runs = {{
    p.stem.replace("__", "/"): json.loads(p.read_text())
    for p in sorted(_runs_dir.glob("*.json"))
}}
_latest = _runs_dir / "latest"
_key = _latest.read_text().strip().replace("__", "/") if _latest.exists() else None
_last = runs.get(_key, {{}})
summary    = _last.get("summary", {{}})
job_stats  = _last.get("job_stats", [])
site_stats = _last.get("site_stats", [])

# ── model-supplied code ─────────────────────────────────────────────────────
'''.format(root=str(ROOT), runs=str(RUNS_DIR))


def _concrete_schedulers(mod) -> List[str]:
    """Names of instantiable SchedulerInterface subclasses defined in `mod`.

    Ordered least- to most-derived so callers can take the last entry: a
    submission that subclasses one of its own helper policies should be run as
    the leaf, not the base.
    """
    from env.interfaces import SchedulerInterface

    found = [
        (name, obj)
        for name, obj in vars(mod).items()
        if isinstance(obj, type)
        and issubclass(obj, SchedulerInterface)
        and obj.__module__ == mod.__name__
        and not inspect.isabstract(obj)
    ]
    leaves = [
        name for name, obj in found
        if not any(other is not obj and issubclass(other, obj) for _, other in found)
    ]
    return leaves or [name for name, _ in found]


def _slug(name: str) -> str:
    """Filesystem-safe label for a scheduler name, which may be a path."""
    stem = Path(name).stem if name.endswith((".py", ".pt")) else name
    return re.sub(r"[^A-Za-z0-9_\-]", "_", stem)


def reset_workspace() -> List[str]:
    """Remove generated scheduler, analysis snippets and cached simulation logs."""
    removed: List[str] = []
    if ANALYSIS_DIR.is_dir():
        snippets = [p for p in ANALYSIS_DIR.glob("*.py")]
        for p in snippets:
            p.unlink()
        if snippets:
            removed.append(f"{len(snippets)} analysis snippet(s)")
    if RUNS_DIR.is_dir():
        logs = [p for p in RUNS_DIR.iterdir() if p.is_file()]
        for p in logs:
            p.unlink()
        if logs:
            removed.append(f"{len(logs)} cached experiment log(s)")
    if AGENT_SCHEDULER_PATH.exists():
        AGENT_SCHEDULER_PATH.unlink()
        removed.append("schedulers/agent_scheduler.py")
    return removed


def _is_under(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _purge_agent_module() -> None:
    for key in list(sys.modules):
        if "agent_scheduler" in key or key == "_agent_check":
            del sys.modules[key]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# Every candidate a search produces is archived at runs/<ts>/code/<name>.py, so
# a run's own artifacts are enough to re-benchmark it later — without this,
# comparing two searches means copying code back into the single
# schedulers/agent_scheduler.py scratch slot, which the next search overwrites.
_SCHEDULER_FILE_DIRS: List[Path] = [ROOT / "schedulers", ROOT / "runs"]


def _load_scheduler_file(path_str: str):
    """Instantiate the scheduler defined in a .py file given by path.

    Restricted to schedulers/ and runs/ for the same reason `read` is
    restricted: an agent naming this path should not be able to import
    arbitrary code from the filesystem.
    """
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not any(_is_under(path, d.resolve()) for d in _SCHEDULER_FILE_DIRS):
        raise ValueError(
            f"'{path_str}' is outside the loadable paths (schedulers/, runs/)."
        )
    if not path.is_file():
        raise FileNotFoundError(f"No such scheduler file: {path_str}")

    # Unique module name per path: two candidates from different runs can share
    # a class name, and reusing one module name would serve a stale class.
    mod_name = f"_sched_{abs(hash(str(path)))}"
    sys.modules.pop(mod_name, None)
    mod = _load_module(path, mod_name)
    found = _concrete_schedulers(mod)
    if not found:
        raise ValueError(
            f"'{path_str}' defines no concrete SchedulerInterface subclass."
        )
    return getattr(mod, found[-1])()


def _json_default(obj):
    try:
        return float(obj)
    except Exception:
        return str(obj)


