"""
agent/loop.py — backend-agnostic agentic research loop.

Two backends are provided:
  AnthropicBackend — uses claude-opus-4-7 via the Anthropic SDK (native tool use)
  OpenAIBackend    — uses any OpenAI-compatible server (vLLM, etc.) via
                     text-based tool calling; no --tool-call-parser required.

Text-based tool calling (OpenAI backend)
-----------------------------------------
Tool schemas are embedded in the system prompt.  The model is instructed to
emit tool calls as JSON inside <tool_call>…</tool_call> markers, which we
parse and execute.  Tool results are injected back as plain user messages.
This approach works with any model regardless of vLLM flag configuration.
"""
from __future__ import annotations

import json
import os
import re
import sys
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from .tools import TOOL_DEFS, ChatResult, ToolCall, ToolExecutor

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a scheduling research agent working on a simulated high-energy-physics
computing grid (ATLAS PanDA style: a few large Tier1 labs, mid-size Tier2
clusters, many small Tier3 sites).  Physics jobs arrive over time in DAG
workflows; each job needs input files that live on only some sites' disks.

YOUR TASK
Write the best scheduling policy you can: a SchedulerInterface subclass whose
schedule(job, env) returns the site index a job should run on.  That placement
is your only decision — hardware, bandwidth, file placement, workload, and the
disk eviction policy are all fixed, and the scheduler must not mutate env.

OBJECTIVE
Maximise throughput_charged_gflops_s.  That is the only score.  It is completed
GFLOPs divided by the span of every submitted job, charging each unfinished one
its elapsed wait — so finishing large work fast raises it, and stranding a job
can only lower it.  Every other metric is diagnostic.

HOW TO WORK
Read AGENT_INSTRUCTIONS.md first: it documents the interface, exactly what you
may read off env and job, and the metrics each run reports.  Then it is up to
you how to use the four tools — benchmark the baselines in schedulers/ for
reference, measure before and after each change, and use run_python to dig into
the per-job and per-site logs when a summary metric alone won't tell you why.

Site and job attributes are exactly those listed in AGENT_INSTRUCTIONS.md.  When
unsure of an attribute or a call signature, read the source under env/ rather
than guessing — a wrong guess surfaces as a traceback mid-simulation.  Numeric
ranges vary by config and seed, so measure rather than assume.
"""

_INITIAL_MESSAGE = (
    "Read AGENT_INSTRUCTIONS.md, then design, test and improve a scheduler."
)


# ── Backend abstraction ───────────────────────────────────────────────────────

class Backend(ABC):
    """Converts between the neutral ChatResult / ToolCall format and SDK-native types."""

    @abstractmethod
    def tools_for_api(self) -> list:
        """Return tool definitions in the format expected by this backend's API."""

    @abstractmethod
    def chat(self, messages: list) -> ChatResult:
        """Send messages to the model; return a ChatResult."""

    @abstractmethod
    def make_assistant_message(self, result: ChatResult) -> Any:
        """Serialize the assistant turn into a message dict (backend-specific)."""

    @abstractmethod
    def make_tool_results(self, results: List[Tuple[str, str, str]]) -> List[Any]:
        """
        Return a list of messages representing tool results.
        Each result is (tool_call_id, tool_name, output_str).
        """

# ── Anthropic backend ─────────────────────────────────────────────────────────

class AnthropicBackend(Backend):
    """Uses claude-opus-4-7 via the Anthropic Python SDK."""

    def __init__(self, model: str = "claude-opus-4-7", max_tokens: int = 8192) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens
        # Cache the system prompt + tool list so repeated calls are cheap
        self._system = [
            {"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
        ]

    def tools_for_api(self) -> list:
        return TOOL_DEFS   # already in Anthropic format

    def chat(self, messages: list) -> ChatResult:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            tools=self.tools_for_api(),
            messages=messages,
        )
        text = next(
            (b.text for b in response.content if hasattr(b, "text") and b.type == "text"),
            None,
        )
        tool_calls = [
            ToolCall(id=b.id, name=b.name, args=b.input)
            for b in response.content
            if b.type == "tool_use"
        ]
        return ChatResult(
            text=text,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            raw_content=response.content,
        )

    def make_assistant_message(self, result: ChatResult) -> dict:
        return {"role": "assistant", "content": result.raw_content}

    def make_tool_results(self, results: List[Tuple[str, str, str]]) -> List[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": output,
                        "is_error": output.startswith("ERROR"),
                    }
                    for tid, _, output in results
                ],
            }
        ]

# ── OpenAI-compatible backend (text-based tool calling) ───────────────────────

def _format_tool(tool: dict) -> str:
    """Render one TOOL_DEFS entry as a signature plus its descriptions."""
    schema = tool.get("input_schema", {})
    props: dict = schema.get("properties", {})
    required = set(schema.get("required", []))

    params = []
    for pname, spec in props.items():
        pytype = {"string": "str", "number": "float", "integer": "int",
                  "boolean": "bool"}.get(spec.get("type", ""), "any")
        if pname in required:
            params.append(f"{pname}: {pytype}")
        else:
            default = spec.get("default", None)
            params.append(f"{pname}: {pytype} = {default!r}")

    lines = [f"{tool['name']}({', '.join(params)})", f"  {tool['description']}"]
    for pname, spec in props.items():
        desc = spec.get("description")
        enum = spec.get("enum")
        if desc or enum:
            detail = desc or ""
            if enum:
                detail = (detail + "  " if detail else "") + f"one of {enum}"
            lines.append(f"    {pname}: {detail}")
    return "\n".join(lines)


def _build_tool_system_prompt() -> str:
    tool_docs = "\n\n".join(_format_tool(t) for t in TOOL_DEFS)
    return f"""{_SYSTEM_PROMPT}
TOOLS
-----
Emit tool calls as JSON inside tags — one block per call, multiple blocks
allowed per reply:
  <tool_call>{{"name":"<tool>","arguments":{{...}}}}</tool_call>

{tool_docs}
"""


_TOOL_SYSTEM_PROMPT = _build_tool_system_prompt()
_TC_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TC_COUNTER = 0


def _next_tc_id() -> str:
    global _TC_COUNTER
    _TC_COUNTER += 1
    return f"tc_{_TC_COUNTER:04d}"


class OpenAIBackend(Backend):
    """
    Uses any OpenAI-compatible server (vLLM, llama.cpp, etc.).

    Tool calling is done via text-based parsing of <tool_call> blocks embedded
    in the model's plain-text response — no --tool-call-parser or
    --enable-auto-tool-choice flags are required on the vLLM server.

    Qwen3 thinking mode
    -------------------
    Pass ``enable_thinking=True`` to activate Qwen3's extended reasoning
    (<think> blocks).  This also forces temperature=0.6 as required by the
    model when thinking is on.  With ``enable_thinking=False`` (default) the
    model responds faster with temperature=0.0.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        enable_thinking: bool = False,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install openai") from exc
        self._client = OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))
        self._model = model or self._detect_model()
        self._max_tokens = max_tokens
        self._temperature = 0.6 if enable_thinking else temperature
        self._extra_body: dict = (
            {"chat_template_kwargs": {"enable_thinking": True}}
            if enable_thinking else {}
        )

    def _detect_model(self) -> str:
        models = self._client.models.list().data
        if not models:
            raise RuntimeError("vLLM server returned no models.")
        return models[0].id

    def tools_for_api(self) -> list:
        return []  # not used; schemas are embedded in the system prompt

    def chat(self, messages: list) -> ChatResult:
        full = [{"role": "system", "content": _TOOL_SYSTEM_PROMPT}] + messages
        kwargs: dict = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=full,
        )
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        response = self._client.chat.completions.create(**kwargs)
        text: str = response.choices[0].message.content or ""

        tool_calls: List[ToolCall] = []
        for m in _TC_RE.finditer(text):
            raw = m.group(1).strip()
            try:
                obj = json.loads(raw)
                name = obj.get("name", "")
                args = obj.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append(ToolCall(id=_next_tc_id(), name=name, args=args))
            except (json.JSONDecodeError, AttributeError):
                pass  # malformed block — skip

        stop = "tool_calls" if tool_calls else "end_turn"
        return ChatResult(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop,
            raw_content=text,
        )

    def make_assistant_message(self, result: ChatResult) -> dict:
        return {"role": "assistant", "content": result.raw_content}

    def make_tool_results(self, results: List[Tuple[str, str, str]]) -> List[dict]:
        parts = "\n\n".join(
            f"[TOOL RESULT: {name}]\n{output}"
            for _, name, output in results
        )
        return [{"role": "user", "content": parts}]

def run_research_agent(
    backend: Backend,
    max_turns: int = 30,
    verbose: bool = True,
) -> dict:
    """
    Run the scheduling research agent loop.

    Parameters
    ----------
    backend : Backend
        AnthropicBackend or OpenAIBackend instance.
    max_turns : int
        Hard cap on the number of model turns.
    verbose : bool
        Print tool calls and model responses to stdout.

    Returns
    -------
    dict
        Cache of all experiment results keyed by (scheduler, config).
    """
    executor = ToolExecutor()
    messages: list = [{"role": "user", "content": _INITIAL_MESSAGE}]

    def _log(tag: str, text: str) -> None:
        if verbose:
            border = "─" * 72
            print(f"\n{border}\n[{tag}]\n{border}")
            print(text[:4000] + ("…" if len(text) > 4000 else ""))

    for turn in range(1, max_turns + 1):
        _log(f"Turn {turn} — sending to model", f"messages in context: {len(messages)}")

        result = backend.chat(messages)

        if result.text:
            _log("Agent", result.text)

        # Record the assistant turn
        messages.append(backend.make_assistant_message(result))

        # No tool calls → agent is done
        if not result.tool_calls:
            _log("Loop", "Agent finished (no tool calls).")
            break

        # Execute every requested tool
        tool_results: List[Tuple[str, str, str]] = []
        for tc in result.tool_calls:
            _log(f"Tool → {tc.name}", json.dumps(tc.args, indent=2))
            output = executor.execute(tc.name, tc.args)
            _log(f"Result ← {tc.name}", output)
            tool_results.append((tc.id, tc.name, output))

        # Feed results back
        for msg in backend.make_tool_results(tool_results):
            messages.append(msg)

    else:
        _log("Loop", f"Reached max_turns={max_turns}.")

    return executor._cache
