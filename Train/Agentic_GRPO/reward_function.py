"""
Reward Functions for Agentic GRPO (Ground-Truth Based)

Implements a 3-category reward:
1) Tool presence (each required tool present => +1)
2) Tool order (required tools appear as a subsequence in trajectory actions)
3) Final answer correctness (fuzzy match)

Signature is compatible with GRPOTrainer:
    reward_fn(task_id: str, trajectory: CompletedTrajectory) -> float
"""

import re
import logging
from typing import List, Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)


def _normalize_text(s: str) -> str:
    s = str(s)
    s = s.strip().lower()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_tools_from_trajectory(trajectory) -> List[str]:
    """
    Extract tool names in order from trajectory action segments.
    Expected action segment format includes: "Action: <tool_name>"
    """
    tools = []
    for segment in getattr(trajectory, "segments", []):
        if getattr(segment, "segment_type", None) != "action":
            continue
        action_text = (segment.text or "").strip()
        match = re.search(r"Action:\s*([^\n\r]+)", action_text)
        if not match:
            continue
        tool_name = match.group(1).strip()
        if tool_name.endswith("()"):
            tool_name = tool_name[:-2].strip()
        tools.append(tool_name)
    return tools


def _extract_final_answer_from_trajectory(trajectory) -> Optional[str]:
    """
    Extract final answer from trajectory's final_answer segment.
    Expected format: "Final Answer: <text>"
    """
    for segment in getattr(trajectory, "segments", []):
        if getattr(segment, "segment_type", None) == "final_answer":
            answer_text = (segment.text or "").strip()
            m = re.search(r"Final Answer:\s*(.+)", answer_text, re.DOTALL)
            return (m.group(1).strip() if m else answer_text)
    return None


def _subsequence_match_fraction(required: List[str], executed: List[str]) -> float:
    """
    Order score: how much of `required` appears in `executed` as a subsequence.
    Extra tools in between are allowed.
    """
    if not required:
        return 1.0
    i = 0
    for t in executed:
        if i < len(required) and t == required[i]:
            i += 1
    return i / len(required)


def _presence_fraction(required: List[str], executed: List[str]) -> float:
    """
    Presence score: each required tool present at least once contributes +1.
    """
    if not required:
        return 1.0
    executed_set = set(executed)
    present = sum(1 for t in required if t in executed_set)
    return present / len(required)


def _final_answer_fuzzy_score(pred: str, gold: str) -> float:
    """
    Fuzzy scoring:
    - 1.0 for exact normalized match OR strong substring match
    - else token overlap heuristic for partial credit
    """
    if pred is None or gold is None:
        return 0.0

    pred_n = _normalize_text(pred)
    gold_n = _normalize_text(gold)

    if not pred_n or not gold_n:
        return 0.0

    # Exact / substring match
    if pred_n == gold_n:
        return 1.0
    if gold_n in pred_n or pred_n in gold_n:
        return 1.0

    # Fuzzy token overlap
    pred_tokens = set(re.findall(r"\b\w+\b", pred_n))
    gold_tokens = set(re.findall(r"\b\w+\b", gold_n))
    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = len(pred_tokens & gold_tokens)
    denom = max(len(pred_tokens), len(gold_tokens))
    ratio = overlap / denom

    # Map overlap ratio to score
    if ratio >= 0.6:
        return 0.7
    if ratio >= 0.4:
        return 0.4
    if ratio >= 0.2:
        return 0.2
    return 0.0


def create_ground_truth_reward_function_v2(
    tasks: List[Dict[str, Any]],
    w_presence: float = 0.2,
    w_order: float = 0.2,
    w_final: float = 0.6,
    no_final_answer_cap: float = 0.2,
    non_success_cap: Optional[float] = 0.6,
) -> Callable[[str, Any], float]:
    """
    Build a reward function from EnterpriseBench ground truth.

    Required task fields expected (from your loader):
    - task['id']
    - task['required_tools']
    - task['gold_final_output']

    Returns:
        reward_fn(task_id, trajectory) -> float in [0,1]
    """
    # Build lookup: task_id -> ground truth
    gt_map = {}
    for t in tasks:
        gt_map[t["id"]] = {
            "required_tools": t.get("required_tools", []) or [],
            "gold_final_output": t.get("gold_final_output", None),
        }

    # Normalize weights to sum to 1 (robust to user edits)
    w_sum = max(w_presence + w_order + w_final, 1e-8)
    w_presence_n = w_presence / w_sum
    w_order_n = w_order / w_sum
    w_final_n = w_final / w_sum

    def reward_fn(task_id: str, trajectory) -> float:
        gt = gt_map.get(task_id)
        if gt is None:
            # Fallback: completion-only
            return 0.5 if getattr(trajectory, "termination_reason", "") == "success" else 0.1

        required_tools = gt["required_tools"]
        gold_final = gt["gold_final_output"]

        executed_tools = _extract_tools_from_trajectory(trajectory)
        final_answer = _extract_final_answer_from_trajectory(trajectory)

        # --- 3 category scores in [0,1]
        presence_score = _presence_fraction(required_tools, executed_tools)
        order_score = _subsequence_match_fraction(required_tools, executed_tools)
        final_score = _final_answer_fuzzy_score(final_answer, gold_final)

        # Weighted sum
        reward = (
            w_presence_n * presence_score +
            w_order_n * order_score +
            w_final_n * final_score
        )

        # Hard caps to prevent reward hacking
        if final_answer is None:
            reward = min(reward, no_final_answer_cap)

        if non_success_cap is not None:
            if getattr(trajectory, "termination_reason", "") != "success":
                reward = min(reward, non_success_cap)

        # Clamp
        return max(0.0, min(1.0, reward))

    return reward_fn
