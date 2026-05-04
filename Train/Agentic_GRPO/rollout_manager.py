"""
AgenticRolloutManager — Multi-Turn LLM Calls + Deterministic Executor Loop

Trajectory format
─────────────────
Each assistant turn contains:
  1. Free-form reasoning prose.
  2. (Optional) A JSON array with a single tool call:
         [{"name": "toolName", "args": {"param": "value"}}]

Each observation is injected as a system message:
  ["Function Call {'name': ..., 'args': ...} Succeeded. Result: {...}"]
  or
  ["Function Call {'name': ..., 'args': ...} Failed during execution. Error: {...}. ..."]

Task completion is signalled by the model appending <TASK_FINISHED> to its
final assistant turn (no tool call in that same turn).

Termination conditions
──────────────────────
  <TASK_FINISHED> in turn      → "success"
  No tool call *and* no tag    → "no_tool_call"      (implicit finish; logged as warning)
  Malformed JSON array         → "invalid_tool_call_json"
  Missing / null tool name     → "invalid_action"
  args not a dict              → "invalid_args"
  max_turns reached            → "max_turns_reached"
  Generation exception         → "generation_error"
"""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import torch

from data_structures import (
    CompletedTrajectory,
    ExecutedToolCall,
    ToolExecutionResult,
    ToolExecutionStatus,
    TrajectorySegment,
)
from prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """{tool_descriptions}"""
# The full prompt is built by PromptBuilder.build_react_prompt(); we just
# insert it verbatim.  The template variable is kept for callers that pass
# a custom system prompt via custom_system_prompt.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_task_finished(text: str) -> bool:
    """Return True if the assistant turn signals task completion."""
    return "<TASK_FINISHED>" in text


def _extract_tool_call_array(text: str) -> Optional[str]:
    """
    Find the first top-level JSON array in the text that looks like a tool
    call list: [{"name": ..., "args": ...}].

    Returns the raw array string, or None if none found.
    """
    # Walk through the text looking for '[' that starts a top-level array.
    depth    = 0
    in_str   = False
    escape   = False
    start    = None

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue

        if ch == '[':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start: i + 1]
                # Quick heuristic: must contain "name"
                if '"name"' in candidate:
                    return candidate
                start = None   # reset and keep looking


    return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AgenticRolloutManager:
    """
    Generates multi-turn trajectories by interleaving LLM planner calls with
    deterministic tool execution.

    Each turn:
      1. Build a prompt from the growing conversation history.
      2. Call the LLM → get assistant text with optional tool-call array.
      3. Append the assistant turn as a TRAINABLE segment.
      4. If <TASK_FINISHED> → done.
      5. If a JSON tool-call array is present → execute → inject observation
         as a NON-TRAINABLE system segment → loop.
      6. Otherwise → terminate with "no_tool_call".
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        tool_env_factory: Callable,
        max_tool_output_tokens: int = 1024,
        tool_timeout_seconds:   float = 30.0,
        max_context_length:     int   = 16_814,
        max_new_tokens:         int   = 2048,
        temperature:            float = 1.0,
        device:                 str   = "cuda",
        max_turns:              int   = 10,
    ):
        self.model                  = model
        self.tokenizer              = tokenizer
        self.tool_env_factory       = tool_env_factory
        self.max_tool_output_tokens = max_tool_output_tokens
        self.tool_timeout_seconds   = tool_timeout_seconds
        self.max_context_length     = max_context_length
        self.max_new_tokens         = max_new_tokens
        self.temperature            = temperature
        self.device                 = device
        self.max_turns              = max_turns

        # Build tool descriptions once from a sample environment
        sample_env = tool_env_factory()
        if hasattr(sample_env, "get_tool_schema"):
            tools = sample_env.get_tool_schema()
        elif hasattr(sample_env, "tool_methods"):
            tools = sample_env.tool_methods
        else:
            tools = []
            logger.warning("Could not find tools in environment")

        self.prompt_builder      = PromptBuilder(tools)
        self._system_prompt_text = self.prompt_builder.build_react_prompt()
        logger.info(
            "AgenticRolloutManager initialised with %d tools, max_turns=%d",
            len(tools), max_turns,
        )

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------

    def _model_generate(self, prompt: str) -> str:
        """Single forward pass; returns only the newly generated tokens."""
        inputs       = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids    = inputs["input_ids"].to(self.device)
        attn_mask    = inputs.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        context_len = input_ids.shape[1]
        remaining   = self.max_context_length - context_len
        if remaining < 100:
            raise ValueError(
                f"Context too long before generation: {context_len} tokens "
                f"(max {self.max_context_length})"
            )

        max_gen = min(self.max_new_tokens, remaining - 10)

        was_gc = getattr(self.model, "is_gradient_checkpointing", False)
        if was_gc:
            self.model.gradient_checkpointing_disable()

        t0 = time.time()
        logger.info("Generating with %d input tokens …", context_len)

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=max_gen,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
        finally:
            if was_gc:
                self.model.gradient_checkpointing_enable()

        elapsed    = time.time() - t0
        new_tokens = output_ids[0][context_len:]
        n          = len(new_tokens)
        logger.info("Generated %d tokens in %.1fs (%.1f tok/s)", n, elapsed, n / elapsed)

        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _model_generate_batch(self, prompts: List[str]) -> List[str]:
        """Batched forward pass; returns newly generated text for each prompt."""
        if not prompts:
            return []

        original_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            )
        finally:
            self.tokenizer.padding_side = original_padding_side
        input_ids = inputs["input_ids"].to(self.device)
        attn_mask = inputs.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        prompt_lengths = (
            attn_mask.sum(dim=1).tolist()
            if attn_mask is not None
            else [input_ids.shape[1]] * input_ids.shape[0]
        )
        max_context_len = max(prompt_lengths)
        remaining = self.max_context_length - max_context_len
        if remaining < 100:
            raise ValueError(
                f"Context too long before generation: {max_context_len} tokens "
                f"(max {self.max_context_length})"
            )

        max_gen = min(self.max_new_tokens, remaining - 10)

        was_gc = getattr(self.model, "is_gradient_checkpointing", False)
        if was_gc:
            self.model.gradient_checkpointing_disable()

        t0 = time.time()
        logger.info(
            "Generating batch of %d prompts with up to %d input tokens …",
            len(prompts),
            max_context_len,
        )

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=max_gen,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
        finally:
            if was_gc:
                self.model.gradient_checkpointing_enable()

        elapsed = time.time() - t0
        logger.info(
            "Generated batch in %.1fs across %d prompts",
            elapsed,
            len(prompts),
        )

        responses: List[str] = []
        padded_width = input_ids.shape[1]
        for row_idx, prompt_len in enumerate(prompt_lengths):
            new_tokens = output_ids[row_idx][padded_width:]
            n = len(new_tokens)
            logger.info(
                "  Batch item %d generated %d tokens from %d prompt tokens (%.1f tok/s)",
                row_idx,
                n,
                prompt_len,
                n / elapsed if elapsed > 0 else 0.0,
            )
            responses.append(
                self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            )

        return responses

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _truncate_tool_output(self, text: str) -> str:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > self.max_tool_output_tokens:
            truncated = self.tokenizer.decode(
                tokens[: self.max_tool_output_tokens], skip_special_tokens=True
            )
            logger.warning(
                "Tool output truncated: %d → %d tokens", len(tokens), self.max_tool_output_tokens
            )
            return truncated + "\n[... output truncated ...]"
        return text

    def _execute_tool(
        self, tool_env: Any, tool_name: str, args: Dict[str, Any]
    ) -> ToolExecutionResult:
        try:
            result = tool_env.execute(tool_name, args)

            if isinstance(result, ToolExecutionResult):
                return result

            if isinstance(result, dict):
                return ToolExecutionResult(
                    status=(
                        ToolExecutionStatus.SUCCESS
                        if result.get("success")
                        else ToolExecutionStatus.RUNTIME_ERROR
                    ),
                    output=str(result.get("output", result)),
                    execution_time_ms=result.get("execution_time_ms", 0.0),
                    error_message=result.get("error_message"),
                )

            return ToolExecutionResult(
                status=ToolExecutionStatus.SUCCESS,
                output=str(result),
                execution_time_ms=0.0,
            )

        except Exception as exc:
            logger.error("Tool execution error for '%s': %s", tool_name, exc)
            return ToolExecutionResult(
                status=ToolExecutionStatus.RUNTIME_ERROR,
                output=f"Error: {exc}",
                execution_time_ms=0.0,
                error_message=str(exc),
            )

    # ------------------------------------------------------------------
    # Observation formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_observation(
        tool_name: str, args: Dict[str, Any], result: ToolExecutionResult
    ) -> str:
        """
        Produce the system-message observation string matching the trajectory format:

        Success:
          ["Function Call {'name': 'lockDoors', 'args': {...}} Succeeded. Result: {...}"]

        Failure:
          ["Function Call {'name': '...', 'args': {...}} Failed during execution.
           Error: {...}. Function calls after this will not be executed."]
        """
        call_repr = f"{{'name': '{tool_name}', 'args': {args}}}"

        if "success" in str(result.status).lower():
            body = f"Function Call {call_repr} Succeeded. Result: {result.output}"
        else:
            err  = result.error_message or result.output
            body = (
                f"Function Call {call_repr} Failed during execution. "
                f"Error: {err}. "
                "Function calls after this will not be executed."
            )

        return json.dumps([body])

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _build_prompt_from_history(self, history: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,   # Qwen3 compatibility
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True,
            )

    # ------------------------------------------------------------------
    # Single trajectory (multi-turn loop)
    # ------------------------------------------------------------------

    def _build_completed_trajectory(
        self,
        query_id: str,
        segments: List[TrajectorySegment],
        executed_calls: List[ExecutedToolCall],
        termination_reason: str,
        t_start: float,
    ) -> CompletedTrajectory:
        full_text = "".join(s.text for s in segments)
        return CompletedTrajectory(
            query_id=query_id,
            segments=segments,
            num_tool_calls=len(executed_calls),
            total_tokens=len(self.tokenizer.encode(full_text, add_special_tokens=False)),
            generation_time_ms=(time.time() - t_start) * 1000,
            termination_reason=termination_reason,
            executed_tool_calls=executed_calls,
        )

    # ------------------------------------------------------------------
    # Batch entry point
    # ------------------------------------------------------------------

    def generate_batch_trajectories(
        self,
        queries:    List[Dict[str, str]],
        group_size: int = 4,
    ) -> Dict[str, List[CompletedTrajectory]]:
        """
        Generate `group_size` independent trajectories for each query.

        Query dict format:
            {"id": "q1", "user": "What is the weather in Paris?"}
            {"id": "q2", "user": "…", "system": "<optional system prompt override>"}

        Each trajectory uses its own isolated tool environment.
        Generation is batched turn-by-turn across active trajectories within
        each query to improve throughput while keeping tool execution isolated.
        """
        logger.info(
            "Batch: %d queries × %d trajectories each", len(queries), group_size
        )
        results: Dict[str, List[CompletedTrajectory]] = {}

        for q_idx, query in enumerate(queries):
            qid = query["id"]
            logger.info("[%d/%d] Query: %s", q_idx + 1, len(queries), qid)

            environments = [self.tool_env_factory() for _ in range(group_size)]
            system_prompt = query.get("system") or self._system_prompt_text
            states: List[Dict[str, Any]] = []
            completed: Dict[int, CompletedTrajectory] = {}

            for g in range(group_size):
                logger.info("  Trajectory %d/%d …", g + 1, group_size)
                states.append({
                    "group_idx": g,
                    "query_id": f"{qid}_g{g}",
                    "history": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query["user"]},
                    ],
                    "segments": [
                        TrajectorySegment(
                            text=system_prompt,
                            is_trainable=False,
                            segment_type="system",
                        ),
                        TrajectorySegment(
                            text=query["user"],
                            is_trainable=False,
                            segment_type="user",
                        ),
                    ],
                    "executed_calls": [],
                    "termination_reason": "unknown",
                    "done": False,
                    "tool_env": environments[g],
                    "t_start": time.time(),
                })

            for turn in range(self.max_turns):
                active_states = [state for state in states if not state["done"]]
                if not active_states:
                    break

                for state in active_states:
                    logger.info(
                        "[%s] Turn %d/%d",
                        state["query_id"],
                        turn + 1,
                        self.max_turns,
                    )

                prompts: List[str] = []
                prompt_failed_states: List[Dict[str, Any]] = []
                for state in active_states:
                    try:
                        prompts.append(self._build_prompt_from_history(state["history"]))
                        prompt_failed_states.append(state)
                    except Exception as exc:
                        logger.error(
                            "[%s] Prompt construction failed on turn %d: %s",
                            state["query_id"],
                            turn + 1,
                            exc,
                        )
                        state["termination_reason"] = "generation_error"
                        state["done"] = True
                        completed[state["group_idx"]] = self._build_completed_trajectory(
                            query_id=state["query_id"],
                            segments=state["segments"],
                            executed_calls=state["executed_calls"],
                            termination_reason=state["termination_reason"],
                            t_start=state["t_start"],
                        )

                if not prompts:
                    continue

                try:
                    responses = self._model_generate_batch(prompts)
                except Exception as exc:
                    logger.error(
                        "[%s] Batched generation failed on turn %d for %d active trajectories: %s",
                        qid,
                        turn + 1,
                        len(prompts),
                        exc,
                    )
                    for state in prompt_failed_states:
                        state["termination_reason"] = "generation_error"
                        state["done"] = True
                        completed[state["group_idx"]] = self._build_completed_trajectory(
                            query_id=state["query_id"],
                            segments=state["segments"],
                            executed_calls=state["executed_calls"],
                            termination_reason=state["termination_reason"],
                            t_start=state["t_start"],
                        )
                    continue

                for state, response_text in zip(prompt_failed_states, responses):
                    query_id = state["query_id"]
                    history = state["history"]
                    segments = state["segments"]
                    executed_calls = state["executed_calls"]

                    logger.info(
                        "[%s] Turn %d (%d chars): %s…",
                        query_id,
                        turn + 1,
                        len(response_text),
                        response_text[:],
                    )

                    history.append({"role": "assistant", "content": response_text})
                    segments.append(TrajectorySegment(
                        text=response_text,
                        is_trainable=True,
                        segment_type="thought_and_action",
                    ))

                    if _extract_task_finished(response_text):
                        logger.info("[%s] <TASK_FINISHED> found on turn %d.", query_id, turn + 1)
                        state["termination_reason"] = "success"
                        state["done"] = True
                        completed[state["group_idx"]] = self._build_completed_trajectory(
                            query_id=query_id,
                            segments=segments,
                            executed_calls=executed_calls,
                            termination_reason=state["termination_reason"],
                            t_start=state["t_start"],
                        )
                        continue

                    tool_call_raw = _extract_tool_call_array(response_text)
                    if tool_call_raw is None:
                        logger.warning(
                            "[%s] Turn %d: no tool call and no <TASK_FINISHED>. Terminating.",
                            query_id,
                            turn + 1,
                        )
                        state["termination_reason"] = "no_tool_call"
                        state["done"] = True
                        completed[state["group_idx"]] = self._build_completed_trajectory(
                            query_id=query_id,
                            segments=segments,
                            executed_calls=executed_calls,
                            termination_reason=state["termination_reason"],
                            t_start=state["t_start"],
                        )
                        continue

                    try:
                        calls = json.loads(tool_call_raw)
                        if not isinstance(calls, list) or not calls:
                            raise ValueError("Tool call must be a non-empty JSON array.")
                        command = calls[0]
                    except (json.JSONDecodeError, ValueError) as exc:
                        logger.warning(
                            "[%s] Invalid tool-call JSON on turn %d: %s",
                            query_id,
                            turn + 1,
                            exc,
                        )
                        state["termination_reason"] = "invalid_tool_call_json"
                        state["done"] = True
                        completed[state["group_idx"]] = self._build_completed_trajectory(
                            query_id=query_id,
                            segments=segments,
                            executed_calls=executed_calls,
                            termination_reason=state["termination_reason"],
                            t_start=state["t_start"],
                        )
                        continue

                    tool_name = str(command.get("name", "")).strip()
                    tool_args = command.get("args", {})

                    if not tool_name or tool_name.lower() in ("none", "null"):
                        logger.warning("[%s] Missing/null tool name: '%s'", query_id, tool_name)
                        state["termination_reason"] = "invalid_action"
                        state["done"] = True
                        completed[state["group_idx"]] = self._build_completed_trajectory(
                            query_id=query_id,
                            segments=segments,
                            executed_calls=executed_calls,
                            termination_reason=state["termination_reason"],
                            t_start=state["t_start"],
                        )
                        continue

                    if not isinstance(tool_args, dict):
                        logger.warning(
                            "[%s] 'args' must be a dict, got %s",
                            query_id,
                            type(tool_args).__name__,
                        )
                        state["termination_reason"] = "invalid_args"
                        state["done"] = True
                        completed[state["group_idx"]] = self._build_completed_trajectory(
                            query_id=query_id,
                            segments=segments,
                            executed_calls=executed_calls,
                            termination_reason=state["termination_reason"],
                            t_start=state["t_start"],
                        )
                        continue

                    logger.info("[%s] Executing '%s' with args: %s", query_id, tool_name, tool_args)
                    result = self._execute_tool(state["tool_env"], tool_name, tool_args)

                    executed_calls.append(ExecutedToolCall(
                        tool_name=tool_name,
                        args=tool_args,
                        status=str(result.status),
                        output=result.output,
                        error_message=getattr(result, "error_message", None),
                        execution_time_ms=getattr(result, "execution_time_ms", 0.0),
                    ))

                    logger.info(
                        "[%s] Tool result — status=%s, output=%.120s…",
                        query_id, result.status, result.output,
                    )

                    observation_text = self._truncate_tool_output(
                        self._format_observation(tool_name, tool_args, result)
                    )
                    history.append({"role": "system", "content": observation_text})
                    segments.append(TrajectorySegment(
                        text=observation_text,
                        is_trainable=False,
                        segment_type="observation",
                    ))
            else:
                for state in states:
                    if state["done"]:
                        continue
                    logger.warning(
                        "[%s] max_turns=%d reached without <TASK_FINISHED>.",
                        state["query_id"],
                        self.max_turns,
                    )
                    state["termination_reason"] = "max_turns_reached"
                    state["done"] = True
                    completed[state["group_idx"]] = self._build_completed_trajectory(
                        query_id=state["query_id"],
                        segments=state["segments"],
                        executed_calls=state["executed_calls"],
                        termination_reason=state["termination_reason"],
                        t_start=state["t_start"],
                    )

            trajectories: List[CompletedTrajectory] = []
            for g in range(group_size):
                traj = completed.get(g)
                if traj is None:
                    state = states[g]
                    traj = self._build_completed_trajectory(
                        query_id=state["query_id"],
                        segments=state["segments"],
                        executed_calls=state["executed_calls"],
                        termination_reason=state["termination_reason"],
                        t_start=state["t_start"],
                    )
                trajectories.append(traj)
                logger.info(
                    "  g%d: turns=%d, reason=%s, time=%.0fms",
                    g,
                    traj.num_tool_calls,
                    traj.termination_reason,
                    traj.generation_time_ms,
                )

            results[qid] = trajectories

            for env in environments:
                try:
                    env.reset()
                except Exception as exc:
                    logger.warning("Env reset failed: %s", exc)

            torch.cuda.empty_cache()

        return results
