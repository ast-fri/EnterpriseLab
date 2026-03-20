"""
Agentic Rollout Manager - COMPLETE & IMPROVED

Combines all fixes + full functionality:
- enable_thinking=False for Qwen3
- No dangerous post-truncation on "\nFinal Answer:"
- Optional HF StoppingCriteria (safe, doesn't delete markers)
- Sequential generation (no threading)
- Observations as system messages
- All helper methods included
"""

import time
import logging
from typing import Any, Callable, List, Dict, Optional
import torch
import json

from data_structures import (
    CompletedTrajectory,
    TrajectorySegment,
    ToolExecutionResult,
    ToolExecutionStatus
)
from react_parser import ReActParser
from prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class SubstringStoppingCriteria:
    """
    Custom stopping criteria that stops when any substring appears.
    Does NOT remove the substring (safe for parsing).
    """
    def __init__(self, tokenizer, stop_strings: List[str], start_length: int):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.start_length = start_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Decode only the newly generated portion
        generated_ids = input_ids[0, self.start_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Check if any stop string appears
        for stop_str in self.stop_strings:
            if stop_str in generated_text:
                return True
        return False


class AgenticRolloutManager:
    """
    Manages trajectory generation with real-time tool execution.
    
    COMPLETE implementation with all features.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        tool_env_factory: Callable,
        max_turns: int = 10,
        max_tool_output_tokens: int = 500,
        tool_timeout_seconds: float = 30.0,
        max_context_length: int = 4096,
        max_new_tokens_per_turn: int = 512,
        temperature: float = 0.7,
        stop_strings: Optional[List[str]] = None,
        device: str = "cuda",
        use_stopping_criteria: bool = False
    ):
        """
        Args:
            model: Language model for generation
            tokenizer: Tokenizer for the model
            tool_env_factory: Factory function that returns a new tool environment
            max_turns: Maximum number of ReAct turns per trajectory
            max_tool_output_tokens: Max tokens to keep from tool outputs
            tool_timeout_seconds: Timeout for individual tool executions
            max_context_length: Maximum context window length
            max_new_tokens_per_turn: Max tokens to generate per turn
            temperature: Sampling temperature
            stop_strings: Optional list of strings to stop generation (via StoppingCriteria)
            device: Device to use for generation
            use_stopping_criteria: Whether to use HF StoppingCriteria (safe stopping)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.tool_env_factory = tool_env_factory
        self.max_turns = max_turns
        self.max_tool_output_tokens = max_tool_output_tokens
        self.tool_timeout_seconds = tool_timeout_seconds
        self.max_context_length = max_context_length
        self.max_new_tokens_per_turn = max_new_tokens_per_turn
        self.temperature = temperature
        self.device = device

        # IMPORTANT: Stop strings are used via StoppingCriteria (doesn't truncate)
        # NOT via post-processing split (which would delete markers)
        self.stop_strings = stop_strings or []
        self.use_stopping_criteria = use_stopping_criteria

        # Create prompt builder from tool schema
        sample_env = tool_env_factory()
        
        # Try to get proper schema first, fallback to tool_methods
        if hasattr(sample_env, 'get_tool_schema'):
            tools = sample_env.get_tool_schema()
        elif hasattr(sample_env, 'tool_methods'):
            tools = sample_env.tool_methods
        else:
            tools = []
            logger.warning("Could not find tools in environment")

        self.prompt_builder = PromptBuilder(tools)

        if isinstance(tools, (dict, list)):
            logger.info(f"Initialized with {len(tools)} tools")
        else:
            logger.info(f"Initialized with tools")

    def _truncate_tool_output(self, output: str) -> str:
        """Truncate tool output to prevent context window overflow."""
        tokens = self.tokenizer.encode(output, add_special_tokens=False)

        if len(tokens) > self.max_tool_output_tokens:
            truncated_tokens = tokens[:self.max_tool_output_tokens]
            truncated_text = self.tokenizer.decode(
                truncated_tokens, 
                skip_special_tokens=True
            )
            logger.warning(
                f"Tool output truncated: {len(tokens)} -> {self.max_tool_output_tokens} tokens"
            )
            return truncated_text + "\n[... output truncated ...]"

        return output

    def _execute_tool_with_timeout(
        self,
        tool_env: Any,
        tool_name: str,
        args: Any
    ) -> ToolExecutionResult:
        """Execute tool with timeout protection."""
        start_time = time.time()

        if not isinstance(args, dict):
            return ToolExecutionResult(
                status=ToolExecutionStatus.INVALID_ARGS,
                output=f"Error: Expected dict for action input, got {type(args).__name__}",
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message="Arguments must be a JSON object"
            )

        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
            
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(tool_env.execute, tool_name, args)
            result = future.result(timeout=self.tool_timeout_seconds)
            result.execution_time_ms = (time.time() - start_time) * 1000
            executor.shutdown(wait=False)
            return result

        except FuturesTimeoutError:
            logger.error(f"Tool '{tool_name}' timed out after {self.tool_timeout_seconds}s")
            return ToolExecutionResult(
                status=ToolExecutionStatus.TIMEOUT,
                output=f"Error: Tool execution exceeded {self.tool_timeout_seconds}s timeout",
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message="Timeout"
            )

        except Exception as e:
            logger.error(f"Tool '{tool_name}' raised exception: {e}", exc_info=True)
            return ToolExecutionResult(
                status=ToolExecutionStatus.RUNTIME_ERROR,
                output=f"Error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )

    def _model_generate(
        self, 
        prompt: str,
        max_new_tokens: int
    ) -> str:
        """
        Generate text from the model.
        
        FIXED: Use tokenizer(...) instead of tokenizer.encode()
        FIXED: Optional StoppingCriteria (safe, doesn't truncate)
        """
        # Use proper tokenization API
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False  # Chat template already added special tokens
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Build stopping criteria if enabled
        stopping_criteria = None
        if self.use_stopping_criteria and self.stop_strings:
            try:
                from transformers import StoppingCriteriaList
                stopping_criteria = StoppingCriteriaList([
                    SubstringStoppingCriteria(
                        self.tokenizer, 
                        self.stop_strings, 
                        input_ids.shape[1]
                    )
                ])
            except Exception as e:
                logger.warning(f"Could not create stopping criteria: {e}")

        # Generation with proper attention mask
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
            )

        # Decode only the new tokens
        new_tokens = output_ids[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # IMPORTANT: Do NOT split/truncate on stop strings here!
        # StoppingCriteria already stopped generation, and parser needs the full text

        return generated_text.strip()

    def _generate_single_trajectory(
        self,
        query_id: str,
        system_prompt: str,
        user_query: str,
        tool_env: Any
    ) -> CompletedTrajectory:
        """
        Generate a single trajectory using the ReAct loop.

        FIXED: enable_thinking=False for Qwen3
        FIXED: Observations as system messages, not user
        """
        start_time = time.time()
        segments = []

        # Initialize messages list for chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        # Store segments for trajectory tracking
        segments.append(TrajectorySegment(
            text=system_prompt,
            is_trainable=False,
            segment_type='system'
        ))

        segments.append(TrajectorySegment(
            text=f"\n\nUser: {user_query}\n",
            is_trainable=False,
            segment_type='user'
        ))

        num_tool_calls = 0
        termination_reason = "unknown"
        assistant_response = ""  # Accumulate assistant's full response

        # ReAct loop
        for turn in range(self.max_turns):
            # FIXED: Apply chat template with enable_thinking=False
            try:
                current_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False  # CRITICAL FIX for Qwen3
                )
            except TypeError:
                # Fallback for tokenizers that don't support enable_thinking
                try:
                    current_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    logger.error(f"Chat template failed: {e}. Falling back to simple concatenation.")
                    current_prompt = system_prompt + "\n\nUser: " + user_query + "\n\nAssistant: " + assistant_response

            # Check context length
            context_tokens = len(self.tokenizer.encode(current_prompt, add_special_tokens=False))
            remaining_tokens = self.max_context_length - context_tokens

            if remaining_tokens < 100:
                logger.warning(
                    f"Query {query_id}: Context overflow at turn {turn} "
                    f"({context_tokens}/{self.max_context_length} tokens)"
                )
                termination_reason = "context_overflow"
                break

            # Generate next segment
            try:
                max_gen_tokens = min(self.max_new_tokens_per_turn, remaining_tokens - 10)
                generated_text = self._model_generate(
                    prompt=current_prompt,
                    max_new_tokens=max_gen_tokens
                )
            except Exception as e:
                logger.error(f"Query {query_id}: Generation failed at turn {turn}: {e}")
                termination_reason = "generation_error"
                break

            # Parse the generated text
            parsed = ReActParser.parse(generated_text)

            # Add thought segment (trainable) if present
            if parsed['thought']:
                thought_text = f"Thought: {parsed['thought']}\n"
                segments.append(TrajectorySegment(
                    text=thought_text,
                    is_trainable=True,
                    segment_type='thought'
                ))
                assistant_response += thought_text

            # Check for terminal state
            if parsed['is_terminal'] and parsed['final_answer']:
                final_text = f"Final Answer: {parsed['final_answer']}"
                segments.append(TrajectorySegment(
                    text=final_text,
                    is_trainable=True,
                    segment_type='final_answer'
                ))
                assistant_response += final_text

                # Add to messages for proper context tracking
                messages.append({"role": "assistant", "content": assistant_response})

                termination_reason = "success"  # FIXED: use "success" not "final_answer"
                break

            # Execute tool if action present
            if parsed['action']:
                # Add action segment (trainable)
                action_text = f"Action: {parsed['action']}\n"

                if parsed['action_input']:
                    if isinstance(parsed['action_input'], dict):
                        action_text += f"Action Input: {json.dumps(parsed['action_input'])}\n"
                    else:
                        action_text += f"Action Input: {parsed['action_input']}\n"

                segments.append(TrajectorySegment(
                    text=action_text,
                    is_trainable=True,
                    segment_type='action'
                ))
                assistant_response += action_text

                # Execute tool
                tool_result = self._execute_tool_with_timeout(
                    tool_env=tool_env,
                    tool_name=parsed['action'],
                    args=parsed['action_input'] if isinstance(parsed['action_input'], dict) else {}
                )
                num_tool_calls += 1

                # Truncate output
                observation_text = self._truncate_tool_output(tool_result.output)
                observation = f"Observation: {observation_text}\n"

                segments.append(TrajectorySegment(
                    text=observation,
                    is_trainable=False,  # NOT trainable - environment output
                    segment_type='observation'
                ))

                # FIXED: Add assistant response to messages, then observation as SYSTEM
                messages.append({"role": "assistant", "content": assistant_response})
                messages.append({"role": "system", "content": observation})  # FIXED: system not user

                # Reset assistant response for next turn
                assistant_response = ""

            else:
                # Model generated text but no valid action or final answer
                logger.warning(
                    f"Query {query_id}: No valid action or final answer at turn {turn}. "
                    f"Generated: {generated_text[:100]}"
                )
                termination_reason = "malformed_output"
                break

        # Check if we hit max turns
        if turn == self.max_turns - 1 and termination_reason == "unknown":
            termination_reason = "max_turns"

        # Build full text for trajectory
        full_text = "".join([seg.text for seg in segments])

        trajectory = CompletedTrajectory(
            query_id=query_id,
            segments=segments,
            num_tool_calls=num_tool_calls,
            total_tokens=len(self.tokenizer.encode(full_text, add_special_tokens=False)),
            generation_time_ms=(time.time() - start_time) * 1000,
            termination_reason=termination_reason
        )

        return trajectory

    def generate_batch_trajectories(
        self,
        queries: List[Dict[str, str]],
        group_size: int = 4
    ) -> Dict[str, List[CompletedTrajectory]]:
        """
        Generate G trajectories for each query in the batch.

        FIXED: Sequential generation per group (removed threading)

        Supports formats:
        1. {'id': ..., 'system': ..., 'user': ...}
        2. {'id': ..., 'user': ...}  (auto-builds system prompt)
        """
        logger.info(
            f"Starting batch generation: {len(queries)} queries x {group_size} trajectories"
        )

        results = {}

        for query_idx, query in enumerate(queries):
            logger.info(f"Processing query {query_idx+1}/{len(queries)}: {query['id']}")

            # Auto-build system prompt if not provided
            if 'system' in query:
                system_prompt = query['system']
            else:
                system_prompt = self.prompt_builder.build_react_prompt()
                logger.debug(f"Auto-built system prompt")

            # Create G isolated environments
            environments = [self.tool_env_factory() for _ in range(group_size)]

            # FIXED: Generate G trajectories SEQUENTIALLY (not threaded)
            trajectories = []

            for g in range(group_size):
                logger.info(f"  Generating trajectory {g+1}/{group_size}...")
                
                try:
                    trajectory = self._generate_single_trajectory(
                        query_id=f"{query['id']}_g{g}",
                        system_prompt=system_prompt,
                        user_query=query['user'],
                        tool_env=environments[g]
                    )
                    trajectories.append(trajectory)

                    logger.info(
                        f"  Group {g+1}/{group_size} completed: "
                        f"{trajectory.num_tool_calls} tools, "
                        f"{trajectory.generation_time_ms:.0f}ms, "
                        f"reason={trajectory.termination_reason}"
                    )
                except Exception as e:
                    logger.error(f"  Group {g+1} failed: {e}", exc_info=True)
                    failed_trajectory = CompletedTrajectory(
                        query_id=f"{query['id']}_g{g}",
                        segments=[],
                        termination_reason="exception",
                        generation_time_ms=0
                    )
                    trajectories.append(failed_trajectory)

            results[query['id']] = trajectories

            # Clean up environments
            for env in environments:
                try:
                    env.reset()
                except Exception as e:
                    logger.warning(f"Failed to reset environment: {e}")

        return results
