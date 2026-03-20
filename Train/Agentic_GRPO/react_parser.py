"""
ReAct format parser for extracting thoughts, actions, and final answers.

FIXED: Handles common edge cases properly:
- FIXED: Nested JSON in action inputs (brace-counting extraction)
- FIXED: Tool names with special characters
- FIXED: Qwen3 <think> tag stripping
- Partial outputs (model stopped mid-generation)
- Multiple actions in one output (takes first)
- Missing required fields
"""

import re
import json
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ReActParser:
    """
    Parser for ReAct-formatted model outputs.

    Expected format:
        Thought: <reasoning>
        Action: <tool_name>
        Action Input: {"arg": "value"}

    Or for terminal:
        Thought: <reasoning>
        Final Answer: <answer>
    """

    # Regex patterns for extraction
    THOUGHT_PATTERN = r"Thought:\s*(.+?)(?=\n(?:Action:|Final Answer:|$))"
    ACTION_PATTERN = r"Action:\s*([^\n\r]+)"  # FIXED: Capture to EOL
    FINAL_ANSWER_PATTERN = r"Final Answer:\s*(.+?)(?=\n(?:Thought:|Action:|Observation:|$)|$)"  # FIXED: Less strict

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """
        Strip Qwen3-style <think>...</think> tags before parsing.
        
        FIXED: Defense-in-depth for Qwen3 thinking mode.
        """
        # Remove <think>...</think> blocks (case insensitive, multiline)
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _extract_json_object(text: str, start_pos: int) -> tuple[str, int]:
        """
        Extract a complete JSON object using brace counting.
        
        FIXED: Handles nested JSON properly (regex cannot do this).
        
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

    @staticmethod
    def parse(text: str) -> Dict[str, Any]:
        """
        Parse ReAct-formatted text into structured components.

        FIXED: Robust nested JSON extraction, <think> stripping

        Args:
            text: Raw model output

        Returns:
            Dictionary with keys:
            - thought: str or None
            - action: str or None
            - action_input: dict or str (str if malformed JSON)
            - final_answer: str or None
            - is_terminal: bool
        """
        # FIXED: Strip <think> tags first
        text = ReActParser._strip_thinking_tags(text)
        
        result = {
            'thought': None,
            'action': None,
            'action_input': None,
            'final_answer': None,
            'is_terminal': False
        }

        # Extract thought
        thought_match = re.search(
            ReActParser.THOUGHT_PATTERN, 
            text, 
            re.DOTALL | re.IGNORECASE
        )
        if thought_match:
            result['thought'] = thought_match.group(1).strip()

        # Check for final answer (terminal state)
        final_match = re.search(
            ReActParser.FINAL_ANSWER_PATTERN, 
            text, 
            re.DOTALL | re.IGNORECASE
        )
        if final_match:
            result['final_answer'] = final_match.group(1).strip()
            result['is_terminal'] = True
            return result

        # Extract action
        action_match = re.search(ReActParser.ACTION_PATTERN, text, re.IGNORECASE)
        if action_match:
            # FIXED: Strip and clean action name (remove trailing parens, whitespace)
            action_name = action_match.group(1).strip()
            # Remove trailing () if present
            if action_name.endswith('()'):
                action_name = action_name[:-2]
            result['action'] = action_name.strip()

        # FIXED: Extract action input using brace counting (handles nested JSON)
        action_input_marker = re.search(
            r"Action Input:\s*", 
            text, 
            re.IGNORECASE
        )
        
        if action_input_marker:
            # Find the first '{' after "Action Input:"
            start_search = action_input_marker.end()
            first_brace = text.find('{', start_search)
            
            if first_brace != -1:
                # Extract complete JSON object with brace matching
                json_str, end_pos = ReActParser._extract_json_object(text, first_brace)
                
                if json_str:
                    try:
                        result['action_input'] = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse action input JSON: {e}")
                        logger.warning(f"Raw input: {json_str[:200]}")
                        # Return the raw string so we can penalize the model
                        result['action_input'] = json_str
                else:
                    logger.warning("Could not extract complete JSON object from Action Input")
            else:
                # No opening brace found - might be malformed
                # Try to extract whatever is on the same line
                rest_of_line = text[start_search:].split('\n')[0].strip()
                if rest_of_line:
                    logger.warning(f"Action Input without JSON braces: {rest_of_line}")
                    result['action_input'] = rest_of_line

        return result

    @staticmethod
    def validate_parse(parsed: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that a parsed result is well-formed.

        FIXED: More lenient for training (allow malformed JSON through with warning)

        Returns:
            (is_valid, error_message)
        """
        # Terminal state is always valid
        if parsed['is_terminal']:
            if not parsed['final_answer']:
                return False, "Final Answer is empty"
            return True, ""

        # Non-terminal must have action
        if not parsed['action']:
            return False, "No action found in non-terminal output"

        # Action input should be a dict (but we're lenient now)
        if parsed['action_input'] is not None:
            if not isinstance(parsed['action_input'], dict):
                # CHANGED: Log warning but still mark as valid
                # The reward function will penalize invalid args
                logger.warning(f"Action input is not valid JSON dict: {type(parsed['action_input'])}")
                # Still valid - let training handle it
                return True, ""

        return True, ""
