"""
Checkpoint-grounded reward module for emulation-system style tasks.

This reward follows the user's workflow-centric design:
- r(Final): binary trajectory-completion reward based on complete ordered
  tool+required-argument coverage via longest common subsequence.
- r(Tool): ordered tool-name and required-argument LCS ratio, scaled to 0.5.
- r(State): LLM-judged retained state variables on required-argument
  LCS-aligned steps, scaled to 0.5.
- r(Format): binary 0/1 based on artist-format compliance.

Unlike the older reward path, this module assumes the task supervision is
primarily a tool-execution trajectory rather than a final-answer retrieval task.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None

logger = logging.getLogger(__name__)

REWARD_SCHEMA_VERSION = "checkpoint_reward_v1.5.0"
TOOL_MAX_REWARD = 0.5
STATE_MAX_REWARD = 0.5
FINAL_MAX_REWARD = 1.0
FORMAT_MAX_REWARD = 1.0

INVALID_FORMAT_TERMINATION_REASONS = {
    "invalid_tool_call_json",
    "invalid_action",
    "invalid_args",
}

DEFAULT_ENTERPRISE_TOOLS_SCHEMA_PATH = (
    "/home/fripl/vharsh/research/EnterpriseBench/Task_Generation/utils/tools.json"
)


@dataclass
class StateJudgeResult:
    retained_count: int
    total_count: int
    reasoning: str


@dataclass
class RewardBreakdown:
    tool_reward: float
    state_reward: float
    final_reward: float
    format_reward: float
    total_reward: float
    cache_hit: bool = False
    state_retained_count: int = 0
    state_total_count: int = 0
    required_lcs_length: int = 0
    full_lcs_length: int = 0
    gold_sequence_length: int = 0
    format_valid: bool = False
    judge_reasoning: str = ""


def _pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True)


def _normalize_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_jsonish(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize_jsonish(v) for v in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_normalize_jsonish(value), sort_keys=True, ensure_ascii=True, separators=(",", ":"))


@lru_cache(maxsize=1)
def _load_required_tool_arguments() -> Dict[str, Tuple[str, ...]]:
    schema_path = os.getenv("ENTERPRISE_TOOLS_SCHEMA_PATH", DEFAULT_ENTERPRISE_TOOLS_SCHEMA_PATH)
    try:
        with open(schema_path, "r", encoding="utf-8") as handle:
            tools = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not load required tool arguments from %s; "
            "tool reward will use all arguments: %s",
            schema_path,
            exc,
        )
        return {}

    required_by_tool: Dict[str, Tuple[str, ...]] = {}
    for tool in tools if isinstance(tools, list) else []:
        if not isinstance(tool, dict):
            continue
        tool_name = str(tool.get("name", "")).strip()
        args_schema = tool.get("args_schema", {})
        if not tool_name or not isinstance(args_schema, dict):
            continue
        required_by_tool[tool_name] = tuple(
            str(arg_name)
            for arg_name, definition in args_schema.items()
            if isinstance(definition, dict) and definition.get("required") is True
        )
    return required_by_tool


def _extract_path(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _normalize_llm_response_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        joined = "".join(parts).strip()
        if joined:
            return joined
    raise TypeError(f"Unsupported judge response content type: {type(content).__name__}")


class SimpleRewardCache:
    def __init__(self, cache_path: Optional[str] = None):
        self.cache_path = cache_path
        self.cache: Dict[str, Dict[str, Any]] = {}
        if cache_path and os.path.exists(cache_path):
            self._load()

    def _load(self) -> None:
        try:
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("schema_version") != REWARD_SCHEMA_VERSION:
                        continue
                    cache_key = entry.get("cache_key")
                    if cache_key:
                        self.cache[cache_key] = entry
        except Exception as exc:
            logger.warning("Failed to load reward cache from %s: %s", self.cache_path, exc)

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        return self.cache.get(cache_key)

    def put(self, cache_key: str, payload: Dict[str, Any]) -> None:
        entry = {
            "cache_key": cache_key,
            "schema_version": REWARD_SCHEMA_VERSION,
            **payload,
        }
        self.cache[cache_key] = entry
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning("Failed to append reward cache entry: %s", exc)


class StateRetentionJudge:
    def __init__(
        self,
        api_base: str,
        model_name: str,
        api_key: str = "EMPTY",
        max_retries: int = 3,
        timeout: int = 30,
    ):
        if ChatOpenAI is None:
            raise ImportError(
                "langchain-openai is required for state-judge reward evaluation. "
                "Install: pip install langchain-openai langchain-core"
            )
        self.model_name = model_name
        self.max_retries = max_retries
        self.llm = ChatOpenAI(
            base_url=api_base,
            api_key=api_key,
            model=model_name,
            temperature=0.0,
            max_retries=max_retries,
            request_timeout=timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    def judge(self, prompt: str, total_count: int) -> StateJudgeResult:
        retries = 0
        last_error: Optional[Exception] = None
        while retries < self.max_retries:
            try:
                llm_with_format = self.llm.bind(
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=512,
                )
                messages = [
                    SystemMessage(
                        content=(
                            "You are a strict state-tracking judge. "
                            "Return JSON only."
                        )
                    ),
                    HumanMessage(content=prompt),
                ]
                response = llm_with_format.invoke(messages)
                content = _normalize_llm_response_content(response.content)
                parsed = json.loads(content)
                retained_count = int(parsed.get("retained_count", 0))
                reasoning = str(parsed.get("reasoning", "")).strip()
                retained_count = max(0, min(total_count, retained_count))
                return StateJudgeResult(
                    retained_count=retained_count,
                    total_count=total_count,
                    reasoning=reasoning,
                )
            except Exception as exc:
                last_error = exc
                retries += 1
                wait_time = 3 * retries
                logger.warning(
                    "State judge failed (attempt %d/%d): %s",
                    retries,
                    self.max_retries,
                    exc,
                )
                if retries < self.max_retries:
                    time.sleep(wait_time)

        return StateJudgeResult(
            retained_count=0,
            total_count=total_count,
            reasoning=f"Judge failed after {self.max_retries} attempts. Last error: {last_error}",
        )


def _build_gold_tool_sequence(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    sequence: List[Dict[str, Any]] = []
    for checkpoint in task.get("reward_checkpoints", []) or []:
        if not isinstance(checkpoint, dict):
            continue
        tool_name = str(checkpoint.get("tool") or checkpoint.get("api_id") or "").strip()
        if not tool_name:
            continue
        sequence.append(
            {
                "tool": tool_name,
                "args": _normalize_jsonish(checkpoint.get("input_bindings", {}) or {}),
                "state_tracking": checkpoint.get("state_tracking", {}) or {},
                "step": checkpoint.get("step"),
                "expected_output_parsed": checkpoint.get("expected_output_parsed"),
            }
        )
    return sequence


def _build_executed_tool_sequence(trajectory: Any) -> List[Dict[str, Any]]:
    sequence: List[Dict[str, Any]] = []
    for call in getattr(trajectory, "executed_tool_calls", []) or []:
        sequence.append(
            {
                "tool": str(call.tool_name),
                "args": _normalize_jsonish(call.args or {}),
                "status": str(call.status),
            }
        )
    return sequence


def _signature(step: Dict[str, Any]) -> Tuple[str, str]:
    return str(step.get("tool", "")), _canonical_json(step.get("args", {}))


def _required_argument_signature(step: Dict[str, Any]) -> Tuple[str, str]:
    tool_name = str(step.get("tool", ""))
    args = step.get("args", {}) or {}
    required_by_tool = _load_required_tool_arguments()

    # Tools outside the EnterpriseBench schema retain the existing exact behavior.
    if tool_name not in required_by_tool or not isinstance(args, dict):
        return tool_name, _canonical_json(args)

    required_args = {
        key: args[key]
        for key in required_by_tool[tool_name]
        if key in args
    }
    return tool_name, _canonical_json(required_args)


def _lcs_length(seq_a: Sequence[Tuple[str, str]], seq_b: Sequence[Tuple[str, str]]) -> int:
    return len(_lcs_alignment(seq_a, seq_b))


def _lcs_alignment(seq_a: Sequence[Tuple[str, str]], seq_b: Sequence[Tuple[str, str]]) -> List[Tuple[int, int]]:
    if not seq_a or not seq_b:
        return []
    dp = [[0] * (len(seq_b) + 1) for _ in range(len(seq_a) + 1)]
    for i, left in enumerate(seq_a, start=1):
        for j, right in enumerate(seq_b, start=1):
            if left == right:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    alignment: List[Tuple[int, int]] = []
    i = len(seq_a)
    j = len(seq_b)
    while i > 0 and j > 0:
        if seq_a[i - 1] == seq_b[j - 1]:
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    alignment.reverse()
    return alignment


def _score_argument_match(expected_args: Dict[str, Any], actual_args: Dict[str, Any]) -> float:
    expected_args = expected_args or {}
    actual_args = actual_args or {}
    if not expected_args and not actual_args:
        return 1.0
    if not expected_args:
        return 1.0
    matched = 0
    for key, expected_value in expected_args.items():
        if key in actual_args and _normalize_jsonish(actual_args[key]) == _normalize_jsonish(expected_value):
            matched += 1
    return matched / max(len(expected_args), 1)


def _compute_tool_reward(
    gold_sequence: List[Dict[str, Any]],
    generated_sequence: List[Dict[str, Any]],
) -> float:
    if not gold_sequence or not generated_sequence:
        return 0.0

    gold_signatures = [_required_argument_signature(step) for step in gold_sequence]
    generated_signatures = [_required_argument_signature(step) for step in generated_sequence]
    matched_steps = len(_lcs_alignment(gold_signatures, generated_signatures))
    return TOOL_MAX_REWARD * (matched_steps / len(gold_sequence))


def _build_state_items(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    tracking = step.get("state_tracking", {}) or {}
    expected_output = step.get("expected_output_parsed")
    state_items: List[Dict[str, Any]] = []

    for item in tracking.get("state_must_remember", []) or []:
        if not isinstance(item, dict):
            continue
        state_items.append(
            {
                "field": str(item.get("field", "")),
                "value": item.get("value"),
                "entity_type": item.get("entity_type"),
                "llm_prompt": item.get("llm_prompt", ""),
            }
        )

    for item in tracking.get("state_to_retain", []) or []:
        if not isinstance(item, dict):
            continue
        value = None
        field_path = item.get("field_path")
        if field_path and expected_output is not None:
            value = _extract_path(expected_output, str(field_path))
        state_items.append(
            {
                "field": str(item.get("field", "")),
                "value": value,
                "entity_type": item.get("entity_type"),
                "llm_prompt": item.get("llm_prompt", ""),
            }
        )

    return [item for item in state_items if item.get("field")]


def _trainable_segments(trajectory: Any) -> List[Any]:
    return [segment for segment in getattr(trajectory, "segments", []) if getattr(segment, "segment_type", "") == "thought_and_action"]


def _segment_format_valid(text: str) -> bool:
    reasoning_matches = re.findall(r"<reasoning>\s*.*?\s*</reasoning>", text, re.DOTALL | re.IGNORECASE)
    tool_matches = re.findall(r"<tool>\s*.*?\s*</tool>", text, re.DOTALL | re.IGNORECASE)
    answer_matches = re.findall(r"<answer>\s*.*?\s*</answer>", text, re.DOTALL | re.IGNORECASE)

    if len(reasoning_matches) != 1:
        return False
    if tool_matches and answer_matches:
        return False
    if not tool_matches and not answer_matches:
        return False
    if len(tool_matches) > 1 or len(answer_matches) > 1:
        return False
    return True


def _compute_format_reward(trajectory: Any) -> float:
    if getattr(trajectory, "termination_reason", "") in INVALID_FORMAT_TERMINATION_REASONS:
        return 0.0
    segments = _trainable_segments(trajectory)
    if not segments:
        return 0.0
    return FORMAT_MAX_REWARD if all(_segment_format_valid(segment.text) for segment in segments) else 0.0


STATE_JUDGE_PROMPT = """You are judging whether an agent retained the required state variables at a workflow step.

Count how many required state items are correctly retained in the agent step.
Retention can be shown by:
- explicitly mentioning the entity/value in reasoning, or
- correctly using the value in the tool arguments for this step.

Be strict:
- Do not infer missing state from world knowledge.
- Count only the items actually evidenced in the step text or tool args.
- Return a retained_count integer between 0 and total_count.

Task instruction:
{task_instruction}

Step number:
{step_number}

Required state items:
{required_state_items}

Agent step text:
{agent_step_text}

Agent tool call:
{agent_tool_call}

Return JSON only:
{{
  "retained_count": <integer>,
  "reasoning": "<short explanation>"
}}
"""


def _compute_state_reward(
    judge: StateRetentionJudge,
    task: Dict[str, Any],
    gold_sequence: List[Dict[str, Any]],
    generated_sequence: List[Dict[str, Any]],
    trajectory: Any,
) -> Tuple[float, int, int, str]:
    trainable_segments = _trainable_segments(trajectory)
    gold_signatures = [_required_argument_signature(step) for step in gold_sequence]
    generated_signatures = [_required_argument_signature(step) for step in generated_sequence]
    aligned_generated_by_gold = {
        gold_index: generated_index
        for gold_index, generated_index in _lcs_alignment(gold_signatures, generated_signatures)
    }
    total_state_items = 0
    retained_state_items = 0
    judge_notes: List[str] = []

    for index, gold_step in enumerate(gold_sequence):
        state_items = _build_state_items(gold_step)
        if not state_items:
            continue

        total_state_items += len(state_items)
        generated_index = aligned_generated_by_gold.get(index)
        if generated_index is None:
            judge_notes.append(
                f"step_{gold_step.get('step', index + 1)}: "
                "no required-argument LCS-aligned generated tool call"
            )
            continue

        segment_text = trainable_segments[generated_index].text if generated_index < len(trainable_segments) else ""
        generated_step = generated_sequence[generated_index]

        prompt = STATE_JUDGE_PROMPT.format(
            task_instruction=task.get("user", ""),
            step_number=gold_step.get("step", index + 1),
            required_state_items=_pretty_json(state_items),
            agent_step_text=segment_text,
            agent_tool_call=_pretty_json(
                {
                    "tool": generated_step.get("tool", ""),
                    "args": generated_step.get("args", {}),
                }
            ),
        )
        result = judge.judge(prompt=prompt, total_count=len(state_items))
        retained_state_items += result.retained_count
        if result.reasoning:
            judge_notes.append(f"step_{gold_step.get('step', index + 1)}: {result.reasoning}")

    if total_state_items == 0:
        return 0.0, 0, 0, ""

    reward = STATE_MAX_REWARD * (retained_state_items / total_state_items)
    return reward, retained_state_items, total_state_items, " | ".join(judge_notes)


class CheckpointTrajectoryRewardFunction:
    def __init__(
        self,
        tasks: List[Dict[str, Any]],
        judge_api_base: str = "http://gpu04:8001/v1",
        judge_model: str = "/home/fripl/vharsh/research/models/models/Qwen3-32b",
        judge_api_key: Optional[str] = None,
        cache_path: Optional[str] = None,
        verbose: bool = False,
    ):
        self.tasks = {task["id"]: task for task in tasks}
        self.judge = StateRetentionJudge(
            api_base=judge_api_base,
            model_name=judge_model,
            api_key=judge_api_key or "EMPTY",
        )
        self.cache = SimpleRewardCache(cache_path=cache_path)
        self.verbose = verbose
        logger.info(
            "Initialized CheckpointTrajectoryRewardFunction (v%s): %d tasks, cache=%s",
            REWARD_SCHEMA_VERSION,
            len(self.tasks),
            "enabled" if cache_path else "disabled",
        )

    def __call__(self, task_id: str, trajectory: Any) -> float:
        return self.compute_detailed_reward(task_id, trajectory).total_reward

    @staticmethod
    def _attach_breakdown(trajectory: Any, breakdown: RewardBreakdown) -> None:
        """Expose reward components to trainer metrics without recomputing reward."""
        setattr(trajectory, "reward_breakdown", breakdown)
        setattr(trajectory, "base_reward", float(breakdown.total_reward))

    def _cache_key(
        self,
        task: Dict[str, Any],
        trajectory: Any,
        gold_sequence: List[Dict[str, Any]],
        generated_sequence: List[Dict[str, Any]],
    ) -> str:
        payload = {
            "schema": REWARD_SCHEMA_VERSION,
            "judge_model": self.judge.model_name,
            "task_id": task.get("id"),
            "reward_checkpoints": task.get("reward_checkpoints", []),
            "gold_sequence": gold_sequence,
            "generated_sequence": generated_sequence,
            "termination_reason": getattr(trajectory, "termination_reason", ""),
            "trainable_text": getattr(trajectory, "trainable_text", ""),
        }
        return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()

    def compute_detailed_reward(self, task_id: str, trajectory: Any) -> RewardBreakdown:
        task = self.tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found in task set")

        gold_sequence = _build_gold_tool_sequence(task)
        generated_sequence = _build_executed_tool_sequence(trajectory)
        cache_key = self._cache_key(task, trajectory, gold_sequence, generated_sequence)

        cached = self.cache.get(cache_key)
        if cached:
            breakdown = RewardBreakdown(**cached["breakdown"], cache_hit=True)
            self._attach_breakdown(trajectory, breakdown)
            return breakdown

        tool_reward = _compute_tool_reward(gold_sequence, generated_sequence)
        required_gold_signatures = [_required_argument_signature(step) for step in gold_sequence]
        required_generated_signatures = [_required_argument_signature(step) for step in generated_sequence]
        required_lcs_length = _lcs_length(required_gold_signatures, required_generated_signatures)
        full_lcs_length = _lcs_length(
            [_signature(step) for step in gold_sequence],
            [_signature(step) for step in generated_sequence],
        )
        final_reward = (
            FINAL_MAX_REWARD
            if required_gold_signatures and required_lcs_length == len(required_gold_signatures)
            else 0.0
        )
        format_reward = _compute_format_reward(trajectory)
        state_reward, retained_count, total_state_count, judge_reasoning = _compute_state_reward(
            judge=self.judge,
            task=task,
            gold_sequence=gold_sequence,
            generated_sequence=generated_sequence,
            trajectory=trajectory,
        )

        total_reward = tool_reward + state_reward + final_reward + format_reward
        breakdown = RewardBreakdown(
            tool_reward=tool_reward,
            state_reward=state_reward,
            final_reward=final_reward,
            format_reward=format_reward,
            total_reward=total_reward,
            state_retained_count=retained_count,
            state_total_count=total_state_count,
            required_lcs_length=required_lcs_length,
            full_lcs_length=full_lcs_length,
            gold_sequence_length=len(required_gold_signatures),
            format_valid=format_reward > 0.0,
            judge_reasoning=judge_reasoning,
        )
        self._attach_breakdown(trajectory, breakdown)

        self.cache.put(
            cache_key,
            {
                "breakdown": {
                    "tool_reward": breakdown.tool_reward,
                    "state_reward": breakdown.state_reward,
                    "final_reward": breakdown.final_reward,
                    "format_reward": breakdown.format_reward,
                    "total_reward": breakdown.total_reward,
                    "state_retained_count": breakdown.state_retained_count,
                    "state_total_count": breakdown.state_total_count,
                    "required_lcs_length": breakdown.required_lcs_length,
                    "full_lcs_length": breakdown.full_lcs_length,
                    "gold_sequence_length": breakdown.gold_sequence_length,
                    "format_valid": breakdown.format_valid,
                    "judge_reasoning": breakdown.judge_reasoning,
                }
            },
        )

        if self.verbose:
            logger.info(
                "Task %s RewardNew: total=%.3f "
                "(tool=%.3f, state=%.3f, final=%.3f, format=%.3f, "
                "required_lcs=%d/%d, full_lcs=%d/%d, state=%d/%d)",
                task_id,
                breakdown.total_reward,
                breakdown.tool_reward,
                breakdown.state_reward,
                breakdown.final_reward,
                breakdown.format_reward,
                breakdown.required_lcs_length,
                breakdown.gold_sequence_length,
                breakdown.full_lcs_length,
                breakdown.gold_sequence_length,
                breakdown.state_retained_count,
                breakdown.state_total_count,
            )

        return breakdown


def create_checkpoint_reward_function(
    tasks: List[Dict[str, Any]],
    judge_api_base: str = "http://gpu04:8001/v1",
    judge_model: str = "/home/fripl/vharsh/research/models/models/Qwen3-32b",
    judge_api_key: Optional[str] = None,
    cache_path: Optional[str] = "./reward_new_cache.jsonl",
    verbose: bool = False,
    **kwargs: Any,
) -> Callable[[str, Any], float]:
    _ = kwargs
    return CheckpointTrajectoryRewardFunction(
        tasks=tasks,
        judge_api_base=judge_api_base,
        judge_model=judge_model,
        judge_api_key=judge_api_key,
        cache_path=cache_path,
        verbose=verbose,
    )
