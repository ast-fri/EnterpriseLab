"""
Core data structures for Agentic GRPO training - COMPLETE & IMPROVED

This module contains all the fundamental data types used throughout the training pipeline.
Critical: The is_trainable flag is what prevents the model from learning to hallucinate tool outputs.

IMPROVEMENTS:
- Unified ToolExecutionResult with ToolExecutionStatus enum
- Improved loop detection using brace-matching JSON extraction
- Split validation into structure vs training checks
- Better error messages and debugging utilities
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import json
import re


class ToolExecutionStatus(Enum):
    """Status codes for tool execution results."""
    SUCCESS = "success"
    INVALID_ARGS = "invalid_args"
    TIMEOUT = "timeout"
    RUNTIME_ERROR = "runtime_error"
    TOOL_NOT_FOUND = "tool_not_found"


@dataclass
class ToolExecutionResult:
    """
    Result of a single tool execution.

    Critical Design: We separate status from output because we need to handle failures
    differently in the training loop (some failures are the model's fault, some aren't).
    """
    status: ToolExecutionStatus
    output: str
    execution_time_ms: float
    error_message: Optional[str] = None

    def should_penalize_model(self) -> bool:
        """
        Determine if this failure should count against the model's reward.

        Attribution Logic:
        - INVALID_ARGS: YES (model failed to generate valid JSON)
        - TOOL_NOT_FOUND: YES (model hallucinated a non-existent tool)
        - TIMEOUT: NO (environment issue, not model's fault)
        - RUNTIME_ERROR: NO (could be environment or edge case)
        """
        return self.status in [
            ToolExecutionStatus.INVALID_ARGS, 
            ToolExecutionStatus.TOOL_NOT_FOUND
        ]


@dataclass
class TrajectorySegment:
    """
    A single segment of a ReAct trajectory.

    Critical: The is_trainable flag determines which tokens get gradients.
    - is_trainable=True: Model-generated tokens (thoughts, actions)
    - is_trainable=False: Environment outputs (observations), user queries, system prompts

    If we train on tool outputs, the model learns to hallucinate results instead of calling tools.
    """
    text: str
    is_trainable: bool
    segment_type: str  # 'system', 'user', 'thought', 'action', 'observation', 'final_answer'
    token_count: int = 0  # Filled by tokenizer during collation

    def __post_init__(self):
        # Validation: Certain segment types must NEVER be trainable
        if self.segment_type in ['observation', 'user', 'system']:
            if self.is_trainable:
                raise ValueError(
                    f"{self.segment_type} segments must have is_trainable=False. "
                    f"Training on environment outputs causes hallucination."
                )


def _extract_json_with_brace_matching(text: str, start_pos: int) -> Tuple[str, int]:
    """
    Extract a complete JSON object using brace counting.
    
    IMPROVED: Handles nested JSON properly (regex cannot do this reliably).
    
    Args:
        text: Full text containing JSON
        start_pos: Index of the opening '{'
        
    Returns:
        (json_string, end_pos) or ("", -1) if parsing fails
    """
    if start_pos >= len(text) or text[start_pos] != '{':
        return "", -1
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i in range(start_pos, len(text)):
        char = text[i]
        
        # Handle string escaping
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        # Track if we're inside a string
        if char == '"':
            in_string = not in_string
            continue
        
        # Only count braces outside of strings
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                
                # Found matching closing brace
                if brace_count == 0:
                    json_str = text[start_pos:i+1]
                    return json_str, i+1
    
    # Incomplete JSON
    return "", -1


@dataclass
class CompletedTrajectory:
    """
    A complete ReAct trajectory with metadata for training.

    This object stores both the structured segments (for precise loss masking)
    and cached full text (for the LLM judge).
    """
    query_id: str
    segments: List[TrajectorySegment]
    reward: Optional[float] = None
    advantage: Optional[float] = None

    # Metadata for debugging and analysis
    num_tool_calls: int = 0
    total_tokens: int = 0
    generation_time_ms: float = 0.0
    termination_reason: str = "unknown"  # 'success', 'max_turns', 'timeout', 'error', 'context_overflow'

    # Cached properties (lazy evaluation to save memory)
    _full_text: Optional[str] = None
    _trainable_text: Optional[str] = None

    @property
    def full_text(self) -> str:
        """Get the complete trajectory as a single string."""
        if self._full_text is None:
            self._full_text = "".join([seg.text for seg in self.segments])
        return self._full_text

    @property
    def trainable_text(self) -> str:
        """Get only the portions the model should be trained on."""
        if self._trainable_text is None:
            self._trainable_text = "".join([
                seg.text for seg in self.segments if seg.is_trainable
            ])
        return self._trainable_text

    def _extract_action_info(self, action_segment: TrajectorySegment) -> Optional[Tuple[str, Dict]]:
        """
        IMPROVED: Extract tool name and arguments from action segment.
        Uses brace-matching instead of regex for nested JSON.
        
        Returns:
            (tool_name, args_dict) or None if parsing fails
        """
        text = action_segment.text.strip()
        
        # Extract tool name
        tool_match = re.search(r"Action:\s*([^\n\r]+)", text)
        if not tool_match:
            return None
        
        tool_name = tool_match.group(1).strip()
        # Remove trailing () if present
        if tool_name.endswith('()'):
            tool_name = tool_name[:-2].strip()
        
        # Extract arguments using brace-matching
        args_match = re.search(r"Action Input:\s*", text, re.IGNORECASE)
        if not args_match:
            return (tool_name, {})
        
        # Find first '{' after "Action Input:"
        start_search = args_match.end()
        first_brace = text.find('{', start_search)
        
        if first_brace == -1:
            return (tool_name, {})
        
        # Use brace-matching extraction
        json_str, _ = _extract_json_with_brace_matching(text, first_brace)
        
        if not json_str:
            return (tool_name, {})
        
        try:
            args_dict = json.loads(json_str)
            return (tool_name, args_dict)
        except json.JSONDecodeError:
            return (tool_name, {})

    def validate_structure(self) -> Tuple[bool, str]:
        """
        IMPROVED: Validate trajectory structure (before reward assignment).

        Checks:
        1. Must have at least one trainable segment
        2. Must not be empty
        3. Segment types must be valid
        4. Check for pathological patterns (infinite loops using brace-matched args)

        Returns:
            (is_valid, error_message)
        """
        # Check 1: Must have trainable content
        if not any(seg.is_trainable for seg in self.segments):
            return False, "No trainable segments in trajectory"

        # Check 2: Must not be empty
        if len(self.segments) == 0:
            return False, "Empty trajectory"

        # Check 3: Validate segment types
        valid_types = {'system', 'user', 'thought', 'action', 'observation', 'final_answer'}
        for seg in self.segments:
            if seg.segment_type not in valid_types:
                return False, f"Invalid segment_type: {seg.segment_type}"

        # Check 4: IMPROVED - Detect infinite loops by comparing (tool_name, args) using brace-matching
        if self.num_tool_calls > 3:
            action_segments = [seg for seg in self.segments if seg.segment_type == 'action']
            
            if len(action_segments) > 3:
                # Extract (tool_name, args) pairs using improved extraction
                action_infos = []
                for seg in action_segments:
                    info = self._extract_action_info(seg)
                    if info:
                        action_infos.append(info)
                
                # Check for repeated identical actions
                if len(action_infos) >= 3:
                    # Look for 3+ consecutive identical (tool, args) pairs
                    for i in range(len(action_infos) - 2):
                        if action_infos[i] == action_infos[i+1] == action_infos[i+2]:
                            tool_name, args = action_infos[i]
                            return False, f"Infinite loop detected: repeated action '{tool_name}' with same args (3 consecutive times)"

        return True, ""

    def validate_for_training(self) -> Tuple[bool, str]:
        """
        IMPROVED: Validate trajectory is ready for training (after reward assignment).

        Checks:
        1. Structure is valid
        2. Reward has been assigned

        Returns:
            (is_valid, error_message)
        """
        # First check structure
        is_valid, error_msg = self.validate_structure()
        if not is_valid:
            return False, error_msg

        # Check reward assignment
        if self.reward is None:
            return False, "Reward not assigned"

        return True, ""

    def validate(self) -> Tuple[bool, str]:
        """
        Default validation (for backward compatibility).
        
        Uses validate_for_training() which includes all checks.
        """
        return self.validate_for_training()

    def get_stats(self) -> Dict[str, Any]:
        """Get trajectory statistics for logging."""
        trainable_tokens = sum(seg.token_count for seg in self.segments if seg.is_trainable)
        total_tokens = sum(seg.token_count for seg in self.segments)

        # Count segment types
        segment_counts = {}
        for seg in self.segments:
            seg_type = seg.segment_type
            segment_counts[seg_type] = segment_counts.get(seg_type, 0) + 1

        return {
            'query_id': self.query_id,
            'num_segments': len(self.segments),
            'segment_counts': segment_counts,
            'num_tool_calls': self.num_tool_calls,
            'total_tokens': total_tokens,
            'trainable_tokens': trainable_tokens,
            'trainable_ratio': trainable_tokens / max(total_tokens, 1),
            'generation_time_ms': self.generation_time_ms,
            'termination_reason': self.termination_reason,
            'reward': self.reward,
            'advantage': self.advantage
        }

    def get_action_sequence(self) -> List[str]:
        """
        IMPROVED: Get sequence of tool names used in this trajectory.
        
        Returns:
            List of tool names in order
        """
        tools = []
        for seg in self.segments:
            if seg.segment_type == 'action':
                info = self._extract_action_info(seg)
                if info:
                    tool_name, _ = info
                    tools.append(tool_name)
        return tools

    def get_action_sequence_with_args(self) -> List[Tuple[str, Dict]]:
        """
        Get sequence of (tool_name, args) pairs used in this trajectory.
        
        Returns:
            List of (tool_name, args_dict) tuples
        """
        actions = []
        for seg in self.segments:
            if seg.segment_type == 'action':
                info = self._extract_action_info(seg)
                if info:
                    actions.append(info)
        return actions

    def count_segment_tokens(self, tokenizer: Any) -> None:
        """
        Fill in token_count for all segments using a tokenizer.
        
        Args:
            tokenizer: Tokenizer to use for counting
        """
        for seg in self.segments:
            if seg.token_count == 0:  # Only compute if not already set
                seg.token_count = len(tokenizer.encode(seg.text, add_special_tokens=False))

    def has_successful_completion(self) -> bool:
        """Check if trajectory completed successfully."""
        return self.termination_reason == "success"

    def has_final_answer(self) -> bool:
        """Check if trajectory contains a final answer segment."""
        return any(seg.segment_type == 'final_answer' for seg in self.segments)

    def get_final_answer(self) -> Optional[str]:
        """
        Extract the final answer text from trajectory.
        
        Returns:
            Final answer string or None if not found
        """
        for seg in self.segments:
            if seg.segment_type == 'final_answer':
                # Extract text after "Final Answer: "
                text = seg.text.strip()
                match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
                if match:
                    return match.group(1).strip()
                return text
        return None

    def summary(self) -> str:
        """
        Get a human-readable summary of the trajectory.
        
        Returns:
            Summary string for logging/debugging
        """
        action_seq = self.get_action_sequence()
        final_ans = self.get_final_answer()
        
        summary = f"Trajectory {self.query_id}:\n"
        summary += f"  Status: {self.termination_reason}\n"
        summary += f"  Tools used ({len(action_seq)}): {' -> '.join(action_seq) if action_seq else 'None'}\n"
        summary += f"  Final answer: {final_ans[:100] if final_ans else 'None'}...\n"
        summary += f"  Reward: {self.reward:.3f if self.reward is not None else 'Not assigned'}\n"
        summary += f"  Tokens: {self.total_tokens} ({sum(1 for s in self.segments if s.is_trainable)} trainable segments)\n"
        
        return summary
