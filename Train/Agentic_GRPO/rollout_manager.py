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
import re
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


def _extract_artist_finished(text: str) -> bool:
    """Return True if the assistant turn signals ARTIST-style completion."""
    return bool(re.search(r"<answer>\s*.*?\s*</answer>", text, re.DOTALL | re.IGNORECASE))


def _truncate_artist_response(text: str) -> str:
    """
    Truncate artist-mode response at the first complete </tool> or </answer> tag.

    This prevents the model from generating multiple tool calls, fabricated tool results,
    or mixing tool and answer in the same turn.

    Returns:
        Truncated text ending at the first complete action boundary.
    """
    # Find the first </tool> tag (case-insensitive)
    tool_match = re.search(r"</tool>", text, re.IGNORECASE)
    # Find the first </answer> tag (case-insensitive)
    answer_match = re.search(r"</answer>", text, re.IGNORECASE)

    # Determine which comes first
    truncate_at = None
    if tool_match and answer_match:
        # Both exist, take the earlier one
        truncate_at = min(tool_match.end(), answer_match.end())
    elif tool_match:
        truncate_at = tool_match.end()
    elif answer_match:
        truncate_at = answer_match.end()

    # If we found a tag, truncate at that position
    if truncate_at is not None:
        truncated = text[:truncate_at]
        if len(truncated) < len(text):
            logger.debug(
                "Artist-mode response truncated from %d to %d chars at first complete tag",
                len(text),
                len(truncated),
            )
        return truncated

    # No tags found, return as-is
    return text


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
        prompt_mode:            str   = "default",
        prompt_template_path:   str | None = None,
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
        self.prompt_mode            = (prompt_mode or "default").strip().lower()
        self.prompt_template_path   = prompt_template_path
        self.rollout_observer       = None

        # Build tool descriptions once from a sample environment
        sample_env = tool_env_factory()
        if hasattr(sample_env, "get_tool_schema"):
            tools = sample_env.get_tool_schema()
        elif hasattr(sample_env, "tool_methods"):
            tools = sample_env.tool_methods
        else:
            tools = []
            logger.warning("Could not find tools in environment")

        # Store all tools for filtering (if enabled later)
        self.all_tools = tools
        self.enable_tool_filtering = False  # Can be set externally
        self.num_random_tools = 30  # Can be set externally

        self.prompt_builder      = PromptBuilder(tools)
        self._system_prompt_text = self.prompt_builder.build_prompt(
            prompt_mode=self.prompt_mode,
            prompt_template_path=self.prompt_template_path,
        )
        logger.info(
            "AgenticRolloutManager initialised with %d tools, max_turns=%d, prompt_mode=%s",
            len(tools), max_turns, self.prompt_mode,
        )

    def set_rollout_observer(self, observer: Any) -> None:
        """Attach an optional benchmark-agnostic rollout lifecycle observer."""
        self.rollout_observer = observer

    def _notify_rollout_observer(
        self,
        method_name: str,
        *,
        default: Any = None,
        **kwargs: Any,
    ) -> Any:
        observer = self.rollout_observer
        method = getattr(observer, method_name, None) if observer is not None else None
        if not callable(method):
            return default
        try:
            return method(manager=self, **kwargs)
        except Exception:
            if bool(getattr(observer, "strict", False)):
                raise
            logger.exception("Rollout observer hook %s failed", method_name)
            return default

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
    def _format_default_observation(
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

    @staticmethod
    def _format_artist_observation(
        tool_name: str, args: Dict[str, Any], result: ToolExecutionResult
    ) -> str:
        payload = {
            "tool": tool_name,
            "args": args,
            "status": str(result.status),
            "output": result.output,
            "error_message": result.error_message,
        }
        return f"<tool_result>{json.dumps(payload, sort_keys=True, ensure_ascii=True)}</tool_result>"

    def _format_observation(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: ToolExecutionResult,
    ) -> str:
        if self.prompt_mode == "artist":
            return self._format_artist_observation(tool_name, args, result)
        return self._format_default_observation(tool_name, args, result)

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

    def _parse_artist_turn(self, text: str) -> Dict[str, Any]:
        tool_matches = re.findall(r"<tool>\s*(.*?)\s*</tool>", text, re.DOTALL | re.IGNORECASE)
        has_answer = _extract_artist_finished(text)

        if tool_matches:
            if len(tool_matches) > 1:
                return {
                    "error_reason": "invalid_tool_call_json",
                    "error": "Multiple <tool> blocks found in one turn; exactly one tool call is allowed per sub-step.",
                }
            if has_answer:
                return {
                    "error_reason": "invalid_tool_call_json",
                    "error": "Turn contains both <tool> and <answer>; tool and final answer must be in separate turns.",
                }
            raw_payload = tool_matches[0].strip()
        else:
            if has_answer:
                return {"finished": True}
            return {"error_reason": "no_tool_call"}

        try:
            parsed = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            return {"error_reason": "invalid_tool_call_json", "error": str(exc)}

        if isinstance(parsed, list):
            if len(parsed) != 1:
                return {
                    "error_reason": "invalid_tool_call_json",
                    "error": "Tool payload list must contain exactly one tool call.",
                }
            if not parsed:
                return {"error_reason": "invalid_tool_call_json", "error": "Empty tool list"}
            parsed = parsed[0]

        if not isinstance(parsed, dict):
            return {"error_reason": "invalid_tool_call_json", "error": "Tool payload must be a JSON object"}

        tool_name = (
            parsed.get("name")
            or parsed.get("tool_name")
            or parsed.get("tool")
            or parsed.get("function", {}).get("name")
        )
        tool_args = (
            parsed.get("args")
            or parsed.get("arguments")
            or parsed.get("input")
            or parsed.get("function", {}).get("arguments")
            or {}
        )
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                pass

        tool_name = str(tool_name or "").strip()
        if not tool_name or tool_name.lower() in ("none", "null"):
            return {"error_reason": "invalid_action"}
        if not isinstance(tool_args, dict):
            return {"error_reason": "invalid_args"}
        return {
            "finished": False,
            "tool_name": tool_name,
            "tool_args": tool_args,
        }

    def _parse_assistant_turn(self, text: str) -> Dict[str, Any]:
        if self.prompt_mode == "artist":
            return self._parse_artist_turn(text)
        if _extract_task_finished(text):
            return {"finished": True}

        tool_call_raw = _extract_tool_call_array(text)
        if tool_call_raw is None:
            return {"error_reason": "no_tool_call"}

        try:
            calls = json.loads(tool_call_raw)
            if not isinstance(calls, list) or not calls:
                raise ValueError("Tool call must be a non-empty JSON array.")
            command = calls[0]
        except (json.JSONDecodeError, ValueError) as exc:
            return {"error_reason": "invalid_tool_call_json", "error": str(exc)}

        tool_name = str(command.get("name", "")).strip()
        tool_args = command.get("args", {})
        if not tool_name or tool_name.lower() in ("none", "null"):
            return {"error_reason": "invalid_action"}
        if not isinstance(tool_args, dict):
            return {"error_reason": "invalid_args"}
        return {
            "finished": False,
            "tool_name": tool_name,
            "tool_args": tool_args,
        }

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

            # Filter tools if enabled (for memory optimization during training)
            system_prompt = query.get("system")
            if system_prompt is None and self.enable_tool_filtering:
                # Filter tools: gold + random, shuffled per task
                from tool_filter_for_training import filter_tools_for_training
                filtered_tools = filter_tools_for_training(
                    all_tools=self.all_tools,
                    task_data=query,  # Expects 'required_tools' field
                    num_random_tools=self.num_random_tools,
                    random_seed=hash(qid)  # Same shuffle for all trajectories of this task
                )
                # Rebuild prompt with filtered tools
                prompt_builder_filtered = PromptBuilder(filtered_tools)
                system_prompt = prompt_builder_filtered.build_prompt(
                    prompt_mode=self.prompt_mode,
                    prompt_template_path=self.prompt_template_path,
                )
            elif system_prompt is None:
                system_prompt = self._system_prompt_text

            states: List[Dict[str, Any]] = []
            completed: Dict[int, CompletedTrajectory] = {}

            for g in range(group_size):
                logger.info("  Trajectory %d/%d …", g + 1, group_size)
                state = {
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
                }
                states.append(state)
                self._notify_rollout_observer(
                    "on_state_initialized",
                    query=query,
                    state=state,
                )

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

                    # Truncate artist-mode responses at first complete tag
                    if self.prompt_mode == "artist":
                        original_len = len(response_text)
                        response_text = _truncate_artist_response(response_text)
                        if len(response_text) < original_len:
                            logger.info(
                                "[%s] Artist-mode truncation: %d → %d chars",
                                query_id,
                                original_len,
                                len(response_text),
                            )

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

                    parsed_turn = self._parse_assistant_turn(response_text)

                    if parsed_turn.get("finished"):
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

                    if parsed_turn.get("error_reason") == "no_tool_call":
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

                    if parsed_turn.get("error_reason") == "invalid_tool_call_json":
                        logger.warning(
                            "[%s] Invalid tool-call JSON on turn %d: %s",
                            query_id,
                            turn + 1,
                            parsed_turn.get("error", "invalid tool-call JSON"),
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

                    if parsed_turn.get("error_reason") == "invalid_action":
                        logger.warning("[%s] Missing/null tool name in assistant turn", query_id)
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

                    if parsed_turn.get("error_reason") == "invalid_args":
                        logger.warning(
                            "[%s] 'args' must be a dict, got %s",
                            query_id,
                            type(parsed_turn.get("tool_args")).__name__,
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

                    tool_name = parsed_turn["tool_name"]
                    tool_args = parsed_turn["tool_args"]

                    logger.info("[%s] Executing '%s' with args: %s", query_id, tool_name, tool_args)
                    self._notify_rollout_observer(
                        "before_tool_execution",
                        query=query,
                        state=state,
                        turn_index=turn,
                        tool_name=tool_name,
                        tool_args=tool_args,
                    )
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
                    self._notify_rollout_observer(
                        "after_tool_execution",
                        query=query,
                        state=state,
                        turn_index=turn,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        result=result,
                        observation_text=observation_text,
                    )
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

            additional_trajectories = self._notify_rollout_observer(
                "augment_group",
                query=query,
                states=states,
                trajectories=trajectories,
                default=[],
            )
            if additional_trajectories:
                trajectories.extend(additional_trajectories)

            results[qid] = trajectories

            for env in environments:
                try:
                    env.reset()
                except Exception as exc:
                    logger.warning("Env reset failed: %s", exc)

            torch.cuda.empty_cache()

        return results
