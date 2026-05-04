"""
Agentic GRPO Reward Module - Raw Structured Trajectory Judge

This reward path sends raw structured gold and generated trajectories plus the
execution trace to a categorical LLM judge. The judge returns discrete labels,
and Python converts those labels into deterministic scalar rewards.

Reward Schema Version: 3.0.0
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None

logger = logging.getLogger(__name__)
_GLOBAL_TOOL_CATALOG_CACHE: Optional[List[Dict[str, Any]]] = None


# ============================================================================
# Configuration Constants
# ============================================================================

REWARD_SCHEMA_VERSION = "3.0.0"

TASK_OUTCOME_REWARDS = {
    "fail": 0.00,
    "partial": 0.25,
    "success": 0.45,
}

STATE_GROUNDING_REWARDS = {
    "broken": 0.00,
    "weak": 0.08,
    "good": 0.17,
    "strong": 0.25,
}

TOOL_USE_QUALITY_REWARDS = {
    "bad": 0.00,
    "mixed": 0.10,
    "good": 0.20,
}

FORMAT_VALIDITY_REWARDS = {
    "invalid": 0.00,
    "mostly_valid": 0.05,
    "valid": 0.10,
}

PENALTY_HALLUCINATED_ENTITY = 0.20
PENALTY_UNSUPPORTED_CLAIM = 0.20
PENALTY_REPEATED_LOOP = 0.10

INVALID_FORMAT_TERMINATION_REASONS = {
    "invalid_tool_call_json",
    "invalid_action",
    "invalid_args",
}

ALLOWED_TASK_OUTCOMES = frozenset(TASK_OUTCOME_REWARDS)
ALLOWED_STATE_GROUNDING = frozenset(STATE_GROUNDING_REWARDS)
ALLOWED_TOOL_USE_QUALITY = frozenset(TOOL_USE_QUALITY_REWARDS)
ALLOWED_FORMAT_VALIDITY = frozenset(FORMAT_VALIDITY_REWARDS)

JUDGE_REQUIRED_FIELDS = (
    "task_outcome",
    "state_grounding",
    "tool_use_quality",
    "format_validity",
    "alternative_valid_path",
    "hallucinated_critical_entity",
    "unsupported_final_claim",
    "repeated_useless_loop",
    "reasoning",
)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class CategoricalJudgment:
    """Categorical outputs from the LLM judge."""

    task_outcome: str
    state_grounding: str
    tool_use_quality: str
    format_validity: str
    alternative_valid_path: bool
    hallucinated_critical_entity: bool
    unsupported_final_claim: bool
    repeated_useless_loop: bool
    reasoning: str


@dataclass
class RewardResult:
    """Complete reward computation result."""

    judgment: CategoricalJudgment
    task_outcome_reward: float
    state_grounding_reward: float
    tool_use_quality_reward: float
    format_validity_reward: float
    hallucination_penalty: float
    unsupported_claim_penalty: float
    loop_penalty: float
    total_reward: float
    cache_hit: bool = False
    judge_error: Optional[str] = None
    overrides_applied: List[str] = field(default_factory=list)


# ============================================================================
# Judge Payload Validation
# ============================================================================


def _normalize_llm_response_content(content: Any) -> str:
    """Normalize LangChain/OpenAI response content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        normalized = "".join(parts).strip()
        if normalized:
            return normalized
    raise TypeError(f"Unsupported judge response content type: {type(content).__name__}")


def validate_judge_output_payload(payload: Any) -> Dict[str, Any]:
    """
    Validate judge output and return a canonical dict.

    Accepted inputs:
    - dict
    - JSON string containing an object
    - JSON string containing a stringified object
    """
    parsed = payload
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Judge returned non-JSON string: {parsed[:200]!r}") from exc

    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Judge returned a JSON string, but it did not decode to an object"
            ) from exc

    if not isinstance(parsed, dict):
        raise TypeError(
            f"Judge output must decode to a JSON object, got {type(parsed).__name__}"
        )

    missing_fields = [field for field in JUDGE_REQUIRED_FIELDS if field not in parsed]
    if missing_fields:
        raise ValueError(f"Judge output missing required fields: {missing_fields}")

    if parsed["task_outcome"] not in ALLOWED_TASK_OUTCOMES:
        raise ValueError(f"Invalid task_outcome: {parsed['task_outcome']!r}")
    if parsed["state_grounding"] not in ALLOWED_STATE_GROUNDING:
        raise ValueError(f"Invalid state_grounding: {parsed['state_grounding']!r}")
    if parsed["tool_use_quality"] not in ALLOWED_TOOL_USE_QUALITY:
        raise ValueError(f"Invalid tool_use_quality: {parsed['tool_use_quality']!r}")
    if parsed["format_validity"] not in ALLOWED_FORMAT_VALIDITY:
        raise ValueError(f"Invalid format_validity: {parsed['format_validity']!r}")

    for field_name in (
        "alternative_valid_path",
        "hallucinated_critical_entity",
        "unsupported_final_claim",
        "repeated_useless_loop",
    ):
        if type(parsed[field_name]) is not bool:
            raise ValueError(
                f"Judge field {field_name!r} must be a boolean, got {type(parsed[field_name]).__name__}"
            )

    if not isinstance(parsed["reasoning"], str):
        raise ValueError(
            f"Judge field 'reasoning' must be a string, got {type(parsed['reasoning']).__name__}"
        )

    return {
        "task_outcome": parsed["task_outcome"],
        "state_grounding": parsed["state_grounding"],
        "tool_use_quality": parsed["tool_use_quality"],
        "format_validity": parsed["format_validity"],
        "alternative_valid_path": parsed["alternative_valid_path"],
        "hallucinated_critical_entity": parsed["hallucinated_critical_entity"],
        "unsupported_final_claim": parsed["unsupported_final_claim"],
        "repeated_useless_loop": parsed["repeated_useless_loop"],
        "reasoning": parsed["reasoning"],
    }


# ============================================================================
# Judge Client (vLLM-hosted Qwen)
# ============================================================================


class LocalQwenJudge:
    """Wrapper for a local Qwen judge exposed through vLLM."""

    def __init__(
        self,
        api_base: str = "",
        api_key: str = "EMPTY",
        model_name: str = "",
        max_retries: int = 3,
        timeout: int = 30,
    ):
        if ChatOpenAI is None:
            raise ImportError(
                "langchain-openai is required for the reward judge. "
                "Install: pip install langchain-openai langchain-core"
            )

        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout

        self.llm = ChatOpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            model=self.model_name,
            temperature=0.0,
            max_retries=max_retries,
            request_timeout=timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        logger.info(
            "Initialized LocalQwenJudge: api_base=%s, model=%s",
            api_base,
            model_name,
        )

    def judge(self, prompt: str, max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Invoke the judge and return a validated payload.

        Raises RuntimeError if retries are exhausted or the response is invalid.
        """
        retries = 0
        last_error: Optional[Exception] = None

        while retries < self.max_retries:
            try:
                llm_with_format = self.llm.bind(
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                messages = [
                    SystemMessage(
                        content=(
                            "You are a precise evaluation judge. "
                            "Always respond with valid JSON only."
                        )
                    ),
                    HumanMessage(content=prompt),
                ]
                response = llm_with_format.invoke(messages)
                content = _normalize_llm_response_content(response.content)
                parsed = json.loads(content)
                return validate_judge_output_payload(parsed)
            except Exception as exc:
                last_error = exc
                retries += 1
                wait_time = 5 * retries
                logger.warning(
                    "Judge call failed (attempt %d/%d): %s",
                    retries,
                    self.max_retries,
                    exc,
                )
                if retries < self.max_retries:
                    logger.info("Retrying judge in %ss", wait_time)
                    time.sleep(wait_time)
        return {
            "task_outcome": "fail",
            "state_grounding": "broken",
            "tool_use_quality": "bad",
            "format_validity": "invalid",
            "alternative_valid_path": False,
            "hallucinated_critical_entity": False,
            "unsupported_final_claim": False,
            "repeated_useless_loop": False,
            "reasoning": f"Judge failed after {self.max_retries} attempts. Last error: {last_error}",
        }
        # raise RuntimeError(
        #     f"Judge failed after {self.max_retries} attempts. Last error: {last_error}"
        # )


# ============================================================================
# Reward Cache
# ============================================================================


class RewardCache:
    """Persistent cache for judge results."""

    def __init__(self, cache_path: Optional[str] = None):
        self.cache_path = cache_path
        self.cache: Dict[str, Dict[str, Any]] = {}
        if cache_path and os.path.exists(cache_path):
            self._load_cache()

    def _load_cache(self):
        try:
            loaded_count = 0
            version_mismatch_count = 0
            with open(self.cache_path, "r") as cache_file:
                for line in cache_file:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    cache_key = entry.get("cache_key")
                    schema_version = entry.get("schema_version", "1.0.0")
                    if not cache_key:
                        continue
                    if schema_version == REWARD_SCHEMA_VERSION:
                        self.cache[cache_key] = entry
                        loaded_count += 1
                    else:
                        version_mismatch_count += 1
            logger.info(
                "Loaded %d cached rewards from %s (skipped %d with version mismatch)",
                loaded_count,
                self.cache_path,
                version_mismatch_count,
            )
        except Exception as exc:
            logger.warning("Failed to load reward cache from %s: %s", self.cache_path, exc)

    def _save_entry(self, cache_key: str, result: Dict[str, Any]):
        if not self.cache_path:
            return
        try:
            entry = {
                "cache_key": cache_key,
                "schema_version": REWARD_SCHEMA_VERSION,
                **result,
            }
            with open(self.cache_path, "a") as cache_file:
                cache_file.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning("Failed to save reward cache entry: %s", exc)

    def get_cache_key(
        self,
        judge_model_id: str,
        available_tools_json: str,
        gold_raw_trajectory_json: str,
        generated_raw_trajectory_json: str,
        exec_trace_json: str,
    ) -> str:
        components = [
            f"schema={REWARD_SCHEMA_VERSION}",
            f"judge={judge_model_id}",
            f"tools={available_tools_json}",
            f"gold={gold_raw_trajectory_json}",
            f"gen={generated_raw_trajectory_json}",
            f"exec={exec_trace_json}",
        ]
        return hashlib.sha256("|||".join(components).encode()).hexdigest()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        return self.cache.get(cache_key)

    def put(self, cache_key: str, result: Dict[str, Any]):
        self.cache[cache_key] = result
        self._save_entry(cache_key, result)


# ============================================================================
# Serialization Helpers
# ============================================================================


def _pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True)


def _compact_tool_catalog_from_schema(tool_schema: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact_catalog: List[Dict[str, Any]] = []
    for tool_name in sorted(tool_schema):
        info = tool_schema.get(tool_name, {}) or {}
        args_schema = info.get("args_schema", {}) if isinstance(info, dict) else {}
        required_args: List[str] = []
        optional_args: List[str] = []
        if isinstance(args_schema, dict):
            for arg_name in sorted(args_schema):
                arg_info = args_schema.get(arg_name, {}) or {}
                if isinstance(arg_info, dict) and arg_info.get("required"):
                    required_args.append(arg_name)
                else:
                    optional_args.append(arg_name)
        compact_catalog.append(
            {
                "name": tool_name,
                "description": info.get("description", "") if isinstance(info, dict) else "",
                "required_args": required_args,
                "optional_args": optional_args,
            }
        )
    return compact_catalog


def _load_global_available_tools() -> List[Dict[str, Any]]:
    global _GLOBAL_TOOL_CATALOG_CACHE
    if _GLOBAL_TOOL_CATALOG_CACHE is not None:
        return _GLOBAL_TOOL_CATALOG_CACHE

    try:
        from enterprise_tool_environment import EnterpriseBenchToolEnvironment

        tool_schema = EnterpriseBenchToolEnvironment.get_tool_schema()
        _GLOBAL_TOOL_CATALOG_CACHE = _compact_tool_catalog_from_schema(tool_schema)
    except Exception as exc:
        logger.warning("Failed to load global available tool catalog: %s", exc)
        _GLOBAL_TOOL_CATALOG_CACHE = []

    return _GLOBAL_TOOL_CATALOG_CACHE


def serialize_available_tools(task: Optional[Dict[str, Any]] = None) -> str:
    """
    Serialize a compact available-tools catalog for the judge.

    Preference order:
    1. task['available_tools'] if already in compact-list form
    2. task['available_tool_schema'] if present
    3. global EnterpriseBench tool schema
    """
    if task:
        compact_tools = task.get("available_tools")
        if isinstance(compact_tools, list) and compact_tools:
            return _pretty_json(compact_tools)

        tool_schema = task.get("available_tool_schema")
        if isinstance(tool_schema, dict) and tool_schema:
            return _pretty_json(_compact_tool_catalog_from_schema(tool_schema))

    return _pretty_json(_load_global_available_tools())


def synthesize_gold_messages(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create a deterministic raw gold trajectory view when raw gold messages are absent.
    """
    synthesized: List[Dict[str, Any]] = []
    for step in task.get("gold_step_outputs", []) or []:
        tool_name = step.get("tool", "")
        inputs = step.get("inputs", {})
        rationale = step.get("rationale", "")
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": rationale if isinstance(rationale, str) else "",
            "tool_calls": [
                {
                    "id": f"gold_step_{step.get('step', len(synthesized) + 1)}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": inputs,
                    },
                }
            ],
        }
        synthesized.append(assistant_msg)

        expected_output = step.get("expected_output")
        if expected_output is not None:
            synthesized.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": expected_output,
                }
            )

    final_output = task.get("gold_final_output")
    if isinstance(final_output, str) and final_output.strip():
        synthesized.append(
            {
                "role": "assistant",
                "content": final_output,
            }
        )

    return synthesized


def serialize_gold_raw_trajectory(task: Dict[str, Any]) -> str:
    """
    Serialize the raw gold/reference trajectory.

    Uses gold_messages when present, otherwise falls back to synthesized
    messages built from gold_step_outputs and gold_final_output.
    """
    gold_messages = task.get("gold_messages")
    source = "gold_messages"

    if not isinstance(gold_messages, list) or not gold_messages:
        gold_messages = synthesize_gold_messages(task)
        source = "synthesized_fallback"

    payload = {
        "source": source,
        "messages": gold_messages,
    }
    return _pretty_json(payload)


def serialize_generated_raw_trajectory(trajectory: Any) -> str:
    """
    Serialize the generated trajectory directly from raw segments.
    """
    segments: List[Dict[str, Any]] = []
    for index, segment in enumerate(trajectory.segments):
        if segment.segment_type not in {"thought_and_action", "observation"}:
            continue
        segments.append(
            {
                "index": index,
                "segment_type": segment.segment_type,
                "text": segment.text,
                "is_trainable": segment.is_trainable,
            }
        )

    payload = {
        "query_id": trajectory.query_id,
        "termination_reason": trajectory.termination_reason,
        "num_tool_calls": trajectory.num_tool_calls,
        "final_answer": trajectory.get_final_answer(),
        "segments": segments,
    }
    return _pretty_json(payload)


def serialize_execution_trace(trajectory: Any) -> str:
    """Serialize executed tool calls into a compact authoritative trace."""
    trace = []
    for index, call in enumerate(getattr(trajectory, "executed_tool_calls", []) or []):
        trace.append(
            {
                "index": index,
                "tool": call.tool_name,
                "args": call.args,
                "status": call.status,
                "output": (call.output or "")[:500],
                "error_message": call.error_message,
            }
        )
    return _pretty_json(trace)


# ============================================================================
# Deterministic Checks and Overrides
# ============================================================================


def detect_repeated_loop(trajectory: Any) -> bool:
    """True if the same tool call repeats three times consecutively."""
    calls = getattr(trajectory, "executed_tool_calls", []) or []
    if len(calls) < 3:
        return False
    for index in range(len(calls) - 2):
        call1 = calls[index]
        call2 = calls[index + 1]
        call3 = calls[index + 2]
        if (
            call1.tool_name == call2.tool_name == call3.tool_name
            and call1.args == call2.args == call3.args
        ):
            return True
    return False


def apply_deterministic_overrides(
    judgment: CategoricalJudgment,
    trajectory: Any,
) -> List[str]:
    """Apply objective post-judge overrides."""
    overrides: List[str] = []

    if trajectory.termination_reason in INVALID_FORMAT_TERMINATION_REASONS:
        judgment.format_validity = "invalid"
        overrides.append(f"invalid_termination={trajectory.termination_reason}")

    if trajectory.termination_reason != "success" and judgment.task_outcome == "success":
        judgment.task_outcome = "partial"
        overrides.append("non_success_termination_caps_outcome")

    if detect_repeated_loop(trajectory):
        judgment.repeated_useless_loop = True
        overrides.append("detected_repeated_loop")

    return overrides


def compute_reward_from_judgment(
    judgment: CategoricalJudgment,
    overrides_applied: List[str],
) -> RewardResult:
    """Convert categorical labels into the final scalar reward."""
    task_outcome_reward = TASK_OUTCOME_REWARDS[judgment.task_outcome]
    state_grounding_reward = STATE_GROUNDING_REWARDS[judgment.state_grounding]
    tool_use_quality_reward = TOOL_USE_QUALITY_REWARDS[judgment.tool_use_quality]
    format_validity_reward = FORMAT_VALIDITY_REWARDS[judgment.format_validity]

    base_reward = (
        task_outcome_reward
        + state_grounding_reward
        + tool_use_quality_reward
        + format_validity_reward
    )

    hallucination_penalty = (
        PENALTY_HALLUCINATED_ENTITY if judgment.hallucinated_critical_entity else 0.0
    )
    unsupported_claim_penalty = (
        PENALTY_UNSUPPORTED_CLAIM if judgment.unsupported_final_claim else 0.0
    )
    loop_penalty = PENALTY_REPEATED_LOOP if judgment.repeated_useless_loop else 0.0

    total_penalty = hallucination_penalty + unsupported_claim_penalty + loop_penalty
    final_reward = max(0.0, min(1.0, base_reward - total_penalty))

    return RewardResult(
        judgment=judgment,
        task_outcome_reward=task_outcome_reward,
        state_grounding_reward=state_grounding_reward,
        tool_use_quality_reward=tool_use_quality_reward,
        format_validity_reward=format_validity_reward,
        hallucination_penalty=hallucination_penalty,
        unsupported_claim_penalty=unsupported_claim_penalty,
        loop_penalty=loop_penalty,
        total_reward=final_reward,
        overrides_applied=overrides_applied,
    )


def build_hard_zero_result(reason: str) -> RewardResult:
    """Return a deterministic zero reward for objective hard-fail cases."""
    judgment = CategoricalJudgment(
        task_outcome="fail",
        state_grounding="broken",
        tool_use_quality="bad",
        format_validity="invalid",
        alternative_valid_path=False,
        hallucinated_critical_entity=False,
        unsupported_final_claim=False,
        repeated_useless_loop=False,
        reasoning=reason,
    )
    return compute_reward_from_judgment(judgment, overrides_applied=[reason])


# ============================================================================
# Judge Prompt Template
# ============================================================================


CATEGORICAL_JUDGE_PROMPT = """You are evaluating an agent's task execution from raw structured trajectory data.

Base your evaluation ONLY on:
- the task instruction
- the available tools catalog
- the raw gold/reference trajectory
- the raw generated trajectory
- the execution trace of what actually ran

Important rules:
1. The gold/reference trajectory is ONE valid solution, not the only valid solution.
2. Different valid tool orders or alternative valid paths may receive full credit.
3. Do NOT reward verbosity, elegant wording, or reasoning style.
4. Focus on task completion, grounding, tool usage quality, and format validity.
5. The execution trace is authoritative when raw text and execution disagree.

---

Task Instruction:
{task_instruction}

Available Tools:
{available_tools}

Gold Raw Trajectory:
{gold_raw_trajectory}

Generated Raw Trajectory:
{generated_raw_trajectory}

Execution Trace:
{execution_trace}

---

Return ONLY valid JSON with this exact structure:

{{
  "task_outcome": "<fail|partial|success>",
  "state_grounding": "<broken|weak|good|strong>",
  "tool_use_quality": "<bad|mixed|good>",
  "format_validity": "<invalid|mostly_valid|valid>",
  "alternative_valid_path": <true|false>,
  "hallucinated_critical_entity": <true|false>,
  "unsupported_final_claim": <true|false>,
  "repeated_useless_loop": <true|false>,
  "reasoning": "<2-3 sentence explanation>"
}}
"""


# ============================================================================
# Main Reward Function
# ============================================================================


class AgenticGRPORewardFunction:
    """
    Hybrid reward function using a categorical LLM judge plus deterministic guards.
    """

    def __init__(
        self,
        tasks: List[Dict[str, Any]],
        judge_api_base: str = "",
        judge_model: str = "",
        judge_api_key: Optional[str] = None,
        cache_path: Optional[str] = None,
        verbose: bool = False,
    ):
        self.tasks = {task["id"]: task for task in tasks}
        self.judge = LocalQwenJudge(
            api_base=judge_api_base,
            model_name=judge_model,
            api_key=judge_api_key or "EMPTY",
        )
        self.cache = RewardCache(cache_path=cache_path)
        self.verbose = verbose
        self._logged_gold_fallback_tasks: Set[str] = set()

        logger.info(
            "Initialized AgenticGRPORewardFunction (v%s): %d tasks, cache=%s",
            REWARD_SCHEMA_VERSION,
            len(self.tasks),
            "enabled" if cache_path else "disabled",
        )

    def _serialize_gold_raw_trajectory(self, task_id: str, task: Dict[str, Any]) -> str:
        gold_messages = task.get("gold_messages")
        if not isinstance(gold_messages, list) or not gold_messages:
            if task_id not in self._logged_gold_fallback_tasks:
                logger.info(
                    "Task %s has no gold_messages; using synthesized gold trajectory fallback",
                    task_id,
                )
                self._logged_gold_fallback_tasks.add(task_id)
        return serialize_gold_raw_trajectory(task)

    def __call__(self, task_id: str, trajectory: Any) -> float:
        return self.compute_detailed_reward(task_id, trajectory).total_reward

    def compute_detailed_reward(self, task_id: str, trajectory: Any) -> RewardResult:
        task = self.tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found in task set")

        required_tools = task.get("required_tools", []) or []
        executed_calls = getattr(trajectory, "executed_tool_calls", []) or []
        if required_tools and not executed_calls:
            result = build_hard_zero_result("no_tools_executed_when_required")
            if self.verbose:
                logger.info(
                    "Task %s Reward: total=%.3f (hard zero: required tools but no execution)",
                    task_id,
                    result.total_reward,
                )
            return result

        gold_raw_trajectory_json = self._serialize_gold_raw_trajectory(task_id, task)
        available_tools_json = serialize_available_tools(task)
        generated_raw_trajectory_json = serialize_generated_raw_trajectory(trajectory)
        exec_trace_json = serialize_execution_trace(trajectory)

        cache_key = self.cache.get_cache_key(
            judge_model_id=self.judge.model_name,
            available_tools_json=available_tools_json,
            gold_raw_trajectory_json=gold_raw_trajectory_json,
            generated_raw_trajectory_json=generated_raw_trajectory_json,
            exec_trace_json=exec_trace_json,
        )

        cached = self.cache.get(cache_key)
        if cached:
            if self.verbose:
                logger.info("Task %s: cache hit", task_id)
            judgment = CategoricalJudgment(**cached["judgment"])
            return RewardResult(
                judgment=judgment,
                task_outcome_reward=cached["task_outcome_reward"],
                state_grounding_reward=cached["state_grounding_reward"],
                tool_use_quality_reward=cached["tool_use_quality_reward"],
                format_validity_reward=cached["format_validity_reward"],
                hallucination_penalty=cached["hallucination_penalty"],
                unsupported_claim_penalty=cached["unsupported_claim_penalty"],
                loop_penalty=cached["loop_penalty"],
                total_reward=cached["total_reward"],
                cache_hit=True,
                overrides_applied=cached.get("overrides_applied", []),
            )

        prompt = CATEGORICAL_JUDGE_PROMPT.format(
            task_instruction=task.get("user", ""),
            available_tools=available_tools_json,
            gold_raw_trajectory=gold_raw_trajectory_json,
            generated_raw_trajectory=generated_raw_trajectory_json,
            execution_trace=exec_trace_json,
        )

        try:
            validated_output = self.judge.judge(prompt, max_tokens=1024)
            judgment = CategoricalJudgment(**validated_output)
        except Exception as exc:
            logger.error("Judge failed for task %s: %s", task_id, exc)
            raise RuntimeError(f"Judge failed for task {task_id}: {exc}") from exc

        overrides_applied = apply_deterministic_overrides(judgment, trajectory)
        result = compute_reward_from_judgment(judgment, overrides_applied)

        cache_data = {
            "judgment": {
                "task_outcome": judgment.task_outcome,
                "state_grounding": judgment.state_grounding,
                "tool_use_quality": judgment.tool_use_quality,
                "format_validity": judgment.format_validity,
                "alternative_valid_path": judgment.alternative_valid_path,
                "hallucinated_critical_entity": judgment.hallucinated_critical_entity,
                "unsupported_final_claim": judgment.unsupported_final_claim,
                "repeated_useless_loop": judgment.repeated_useless_loop,
                "reasoning": judgment.reasoning,
            },
            "task_outcome_reward": result.task_outcome_reward,
            "state_grounding_reward": result.state_grounding_reward,
            "tool_use_quality_reward": result.tool_use_quality_reward,
            "format_validity_reward": result.format_validity_reward,
            "hallucination_penalty": result.hallucination_penalty,
            "unsupported_claim_penalty": result.unsupported_claim_penalty,
            "loop_penalty": result.loop_penalty,
            "total_reward": result.total_reward,
            "overrides_applied": overrides_applied,
        }
        self.cache.put(cache_key, cache_data)

        if self.verbose:
            logger.info(
                "Task %s Reward: total=%.3f (outcome=%.3f, state=%.3f, tool=%.3f, format=%.3f, penalties=%.3f)",
                task_id,
                result.total_reward,
                result.task_outcome_reward,
                result.state_grounding_reward,
                result.tool_use_quality_reward,
                result.format_validity_reward,
                result.hallucination_penalty
                + result.unsupported_claim_penalty
                + result.loop_penalty,
            )

        return result


# ============================================================================
# Factory Function
# ============================================================================


def create_agentic_grpo_reward_function(
    tasks: List[Dict[str, Any]],
    judge_api_base: str = "",
    judge_model: str = "",
    judge_api_key: Optional[str] = None,
    cache_path: Optional[str] = "./reward_cache.jsonl",
    verbose: bool = False,
    **kwargs,
) -> Callable[[str, Any], float]:
    """Factory for the agentic GRPO reward function."""
    _ = kwargs
    reward_computer = AgenticGRPORewardFunction(
        tasks=tasks,
        judge_api_base=judge_api_base,
        judge_model=judge_model,
        judge_api_key=judge_api_key,
        cache_path=cache_path,
        verbose=verbose,
    )
    return reward_computer
