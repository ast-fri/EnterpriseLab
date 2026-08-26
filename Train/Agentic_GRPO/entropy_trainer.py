"""
Entropy-aware DAPO trainer for agentic GRPO trajectories.

This trainer keeps the user's task reward as the primary learning signal and adds
an auxiliary semantic-entropy shaping signal on top of DAPO.

Key ideas:
- Exact reward from reward_new.py remains unchanged.
- Trajectories for the same prompt are clustered by semantic equivalence.
- Equivalence uses:
  1. Exact tool-action sequence match.
  2. Optional LLM-judged workflow equivalence.
- Semantic entropy is computed over the cluster distribution.
- Reward shaping is conditional:
  - High-reward groups: reinforce consistent, high-reward semantic clusters.
  - Low-reward groups: penalize collapsed low-reward groups and mildly preserve
    exploration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from data_structures import CompletedTrajectory
from dapo_trainer import DAPOTrainer

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None

logger = logging.getLogger(__name__)


def _normalize_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_jsonish(inner)
            for key, inner in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_normalize_jsonish(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _normalize_jsonish(value),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _trajectory_base_id(query_id: str) -> str:
    return query_id.rsplit("_g", 1)[0] if "_g" in query_id else query_id


def _strip_artist_tags(text: str) -> str:
    text = re.sub(r"<tool>\s*.*?\s*</tool>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<answer>\s*.*?\s*</answer>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _trajectory_reasoning_snippets(traj: CompletedTrajectory, max_chars: int = 800) -> List[str]:
    snippets: List[str] = []
    remaining = max_chars
    for segment in traj.segments:
        if segment.segment_type != "thought_and_action":
            continue
        cleaned = _strip_artist_tags(segment.text)
        if not cleaned:
            continue
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue
        snippet = cleaned[:remaining]
        snippets.append(snippet)
        remaining -= len(snippet)
        if remaining <= 0:
            break
    return snippets


def _tool_action_signature(traj: CompletedTrajectory) -> Tuple[Tuple[str, str], ...]:
    return tuple(
        (
            str(call.tool_name).strip(),
            _canonical_json(call.args or {}),
        )
        for call in (traj.executed_tool_calls or [])
    )


def _workflow_summary(traj: CompletedTrajectory) -> str:
    lines = [
        f"termination_reason: {traj.termination_reason}",
        f"num_tool_calls: {len(traj.executed_tool_calls or [])}",
    ]
    if traj.executed_tool_calls:
        lines.append("executed_tool_calls:")
        for step_index, call in enumerate(traj.executed_tool_calls, start=1):
            lines.append(
                f"  {step_index}. tool={call.tool_name} "
                f"args={_canonical_json(call.args or {})} status={call.status}"
            )
    else:
        lines.append("executed_tool_calls: []")

    snippets = _trajectory_reasoning_snippets(traj)
    if snippets:
        lines.append("reasoning_snippets:")
        for step_index, snippet in enumerate(snippets, start=1):
            lines.append(f"  {step_index}. {snippet}")
    return "\n".join(lines)


@dataclass
class SemanticClusterInfo:
    cluster_assignments: List[int]
    cluster_sizes: Dict[int, int]
    entropy: float
    normalized_entropy: float
    num_clusters: int


class WorkflowEquivalenceJudge:
    """Optional LLM judge for workflow-level semantic equivalence."""

    def __init__(
        self,
        api_base: str,
        model_name: str,
        api_key: str = "EMPTY",
        max_retries: int = 2,
        timeout: int = 30,
        confidence_threshold: float = 0.75,
    ):
        if ChatOpenAI is None:
            raise ImportError(
                "langchain-openai is required for workflow equivalence judging. "
                "Install: pip install langchain-openai langchain-core"
            )
        self.max_retries = max_retries
        self.confidence_threshold = float(confidence_threshold)
        self.llm = ChatOpenAI(
            base_url=api_base,
            api_key=api_key,
            model=model_name,
            temperature=0.0,
            max_retries=max_retries,
            request_timeout=timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.cache: Dict[str, Tuple[bool, float, str]] = {}

    @staticmethod
    def _normalize_response_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts).strip()
        raise TypeError(f"Unsupported judge response content type: {type(content).__name__}")

    def are_equivalent(self, workflow_a: str, workflow_b: str) -> Tuple[bool, float, str]:
        cache_key = _hash_text(workflow_a + "\n===\n" + workflow_b)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        system_prompt = (
            "You are a strict workflow equivalence judge for tool-using agent trajectories. "
            "Return JSON only. "
            "Two trajectories are semantically equivalent if they achieve the same effective workflow "
            "for the same task, even if reasoning prose differs or there are harmless formatting differences. "
            "Treat them as NOT equivalent if they operate on different entities, make different substantive "
            "tool choices, cause different state changes, or follow materially different workflows."
        )
        user_prompt = (
            "Compare the following two trajectories for semantic workflow equivalence.\n\n"
            "Trajectory A:\n"
            f"{workflow_a}\n\n"
            "Trajectory B:\n"
            f"{workflow_b}\n\n"
            "Return a JSON object with keys:\n"
            '- "equivalent": true or false\n'
            '- "confidence": number between 0 and 1\n'
            '- "reasoning": short explanation\n'
        )

        retries = 0
        last_error: Optional[Exception] = None
        while retries < self.max_retries:
            try:
                llm_with_format = self.llm.bind(
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=256,
                )
                response = llm_with_format.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                )
                content = self._normalize_response_content(response.content)
                parsed = json.loads(content)
                equivalent = bool(parsed.get("equivalent", False))
                confidence = float(parsed.get("confidence", 0.0))
                reasoning = str(parsed.get("reasoning", "")).strip()
                result = (
                    equivalent and confidence >= self.confidence_threshold,
                    confidence,
                    reasoning,
                )
                self.cache[cache_key] = result
                return result
            except Exception as exc:
                last_error = exc
                retries += 1
                if retries < self.max_retries:
                    time.sleep(2 * retries)

        logger.warning("Workflow equivalence judge failed after retries: %s", last_error)
        result = (False, 0.0, f"judge_failed: {last_error}")
        self.cache[cache_key] = result
        return result


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


class EntropyTrainer(DAPOTrainer):
    """DAPO trainer augmented with conditional semantic-entropy shaping."""

    def __init__(
        self,
        model: Any,
        ref_model: Any,
        optimizer: Any,
        tokenizer: Any,
        reward_function: Any,
        beta: float = 0.0,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
        logprob_chunk_size: int = 128,
        use_wandb: bool = False,
        epsilon: float = 0.2,
        epsilon_high: float = 0.28,
        dynamic_sampling: bool = True,
        dynamic_sampling_min_std: float = 1e-6,
        filter_overlong: bool = False,
        semantic_exploit_reward_threshold: float = 1.0,
        semantic_consistency_weight: float = 0.25,
        semantic_exploration_weight: float = 0.10,
        semantic_collapse_weight: float = 0.10,
        semantic_adjustment_clip: float = 0.50,
        semantic_equivalence_confidence_threshold: float = 0.75,
        semantic_judge_api_base: Optional[str] = None,
        semantic_judge_model: Optional[str] = None,
        semantic_judge_api_key: str = "EMPTY",
    ):
        super().__init__(
            model=model,
            ref_model=ref_model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            reward_function=reward_function,
            beta=beta,
            max_grad_norm=max_grad_norm,
            device=device,
            logprob_chunk_size=logprob_chunk_size,
            use_wandb=use_wandb,
            epsilon=epsilon,
            epsilon_high=epsilon_high,
            dynamic_sampling=dynamic_sampling,
            dynamic_sampling_min_std=dynamic_sampling_min_std,
            filter_overlong=filter_overlong,
        )
        self.semantic_exploit_reward_threshold = float(semantic_exploit_reward_threshold)
        self.semantic_consistency_weight = float(semantic_consistency_weight)
        self.semantic_exploration_weight = float(semantic_exploration_weight)
        self.semantic_collapse_weight = float(semantic_collapse_weight)
        self.semantic_adjustment_clip = float(semantic_adjustment_clip)
        self._last_group_semantic_info: Dict[str, SemanticClusterInfo] = {}

        judge_api_base = semantic_judge_api_base or os.getenv("SEMANTIC_JUDGE_API_BASE")
        judge_model = semantic_judge_model or os.getenv("SEMANTIC_JUDGE_MODEL")
        judge_api_key = semantic_judge_api_key or os.getenv("SEMANTIC_JUDGE_API_KEY", "EMPTY")
        self.workflow_judge: Optional[WorkflowEquivalenceJudge] = None
        if judge_api_base and judge_model:
            try:
                self.workflow_judge = WorkflowEquivalenceJudge(
                    api_base=judge_api_base,
                    model_name=judge_model,
                    api_key=judge_api_key,
                    confidence_threshold=semantic_equivalence_confidence_threshold,
                )
                logger.info(
                    "EntropyTrainer workflow judge enabled: model=%s api_base=%s",
                    judge_model,
                    judge_api_base,
                )
            except Exception as exc:
                logger.warning("Failed to initialize workflow judge, exact-match only fallback: %s", exc)
        else:
            logger.info("EntropyTrainer workflow judge disabled; using exact tool-action similarity only")

    def _are_semantically_equivalent(
        self,
        traj_a: CompletedTrajectory,
        traj_b: CompletedTrajectory,
    ) -> Tuple[bool, str]:
        exact_a = _tool_action_signature(traj_a)
        exact_b = _tool_action_signature(traj_b)
        if exact_a == exact_b:
            return True, "exact_tool_action_match"

        if not exact_a and not exact_b:
            if traj_a.termination_reason == traj_b.termination_reason:
                return True, "same_empty_termination"

        if self.workflow_judge is None:
            return False, "no_judge_non_exact"

        workflow_a = _workflow_summary(traj_a)
        workflow_b = _workflow_summary(traj_b)
        equivalent, confidence, reason = self.workflow_judge.are_equivalent(workflow_a, workflow_b)
        if equivalent:
            return True, f"llm_equivalent@{confidence:.2f}"
        return False, reason or f"llm_not_equivalent@{confidence:.2f}"

    def _build_semantic_clusters(self, trajectories: Sequence[CompletedTrajectory]) -> SemanticClusterInfo:
        size = len(trajectories)
        if size == 0:
            return SemanticClusterInfo([], {}, 0.0, 0.0, 0)
        if size == 1:
            return SemanticClusterInfo([0], {0: 1}, 0.0, 0.0, 1)

        union_find = _UnionFind(size)
        for left in range(size):
            for right in range(left + 1, size):
                equivalent, reason = self._are_semantically_equivalent(
                    trajectories[left],
                    trajectories[right],
                )
                if equivalent:
                    union_find.union(left, right)
                logger.debug(
                    "Semantic equivalence %s vs %s => %s (%s)",
                    trajectories[left].query_id,
                    trajectories[right].query_id,
                    equivalent,
                    reason,
                )

        root_to_cluster: Dict[int, int] = {}
        assignments: List[int] = []
        cluster_sizes: Dict[int, int] = {}
        for index in range(size):
            root = union_find.find(index)
            cluster_id = root_to_cluster.setdefault(root, len(root_to_cluster))
            assignments.append(cluster_id)
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

        probabilities = np.array(
            [count / size for count in cluster_sizes.values()],
            dtype=np.float64,
        )
        entropy = float(-(probabilities * np.log(probabilities + 1e-10)).sum())
        num_clusters = len(cluster_sizes)
        if num_clusters <= 1:
            normalized_entropy = 0.0
        else:
            normalized_entropy = float(entropy / math.log(num_clusters))

        return SemanticClusterInfo(
            cluster_assignments=assignments,
            cluster_sizes=cluster_sizes,
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            num_clusters=num_clusters,
        )

    def _compute_semantic_entropy(
        self,
        trajectories: List[CompletedTrajectory],
        valid_mask: Any,
    ) -> float:
        if hasattr(valid_mask, "detach"):
            valid_list = valid_mask.detach().cpu().tolist()
        else:
            valid_list = list(valid_mask)

        groups: Dict[str, List[CompletedTrajectory]] = {}
        for is_valid, traj in zip(valid_list, trajectories):
            if not is_valid:
                continue
            groups.setdefault(_trajectory_base_id(traj.query_id), []).append(traj)

        if not groups:
            return 0.0

        entropies: List[float] = []
        for base_id, group_trajs in groups.items():
            info = self._build_semantic_clusters(group_trajs)
            self._last_group_semantic_info[base_id] = info
            entropies.append(info.normalized_entropy)
        return float(np.mean(entropies)) if entropies else 0.0

    def _assign_rewards_and_advantages(
        self,
        all_trajectories: List[CompletedTrajectory],
        initial_valid_mask: List[bool],
    ) -> Tuple[List[bool], int]:
        valid_mask = list(initial_valid_mask)
        for index, traj in enumerate(all_trajectories):
            if traj.reward is None:
                task_id = _trajectory_base_id(traj.query_id)
                try:
                    traj.reward = self.reward_function(task_id, traj)
                except Exception as exc:
                    logger.error("Reward computation failed for %s: %s", task_id, exc)
            if traj.reward is None:
                traj.reward = 0.0
                valid_mask[index] = False
            setattr(traj, "base_reward", float(traj.reward or 0.0))

        groups: Dict[str, List[int]] = {}
        for index, traj in enumerate(all_trajectories):
            groups.setdefault(_trajectory_base_id(traj.query_id), []).append(index)

        filtered_groups = 0
        self._last_group_semantic_info = {}
        for base_id, indices in groups.items():
            valid_indices = [i for i in indices if valid_mask[i]]
            if not valid_indices:
                for index in indices:
                    all_trajectories[index].advantage = 0.0
                continue

            group_trajectories = [all_trajectories[i] for i in valid_indices]
            semantic_info = self._build_semantic_clusters(group_trajectories)
            self._last_group_semantic_info[base_id] = semantic_info

            base_rewards = [float(getattr(all_trajectories[i], "base_reward", 0.0)) for i in valid_indices]
            group_max_reward = max(base_rewards) if base_rewards else 0.0
            if len(valid_indices) <= 1:
                logging_entropy = 0.0
            else:
                # Logging uses the fixed group size as the maximum number of modes.
                # Keep normalized_entropy unchanged for backward-compatible shaping.
                logging_entropy = float(
                    semantic_info.entropy / math.log(len(valid_indices))
                )
            high_reward_group = (
                group_max_reward >= self.semantic_exploit_reward_threshold
            )
            for global_index in valid_indices:
                setattr(
                    all_trajectories[global_index],
                    "semantic_group_logging_entropy",
                    logging_entropy,
                )
                setattr(
                    all_trajectories[global_index],
                    "semantic_group_high_reward",
                    high_reward_group,
                )
            shaped_rewards: List[float] = []

            cluster_reward_map: Dict[int, List[float]] = {}
            for local_index, cluster_id in enumerate(semantic_info.cluster_assignments):
                cluster_reward_map.setdefault(cluster_id, []).append(base_rewards[local_index])
            cluster_mean_reward = {
                cluster_id: float(np.mean(values)) if values else 0.0
                for cluster_id, values in cluster_reward_map.items()
            }

            valid_count = max(len(valid_indices), 1)
            for local_index, global_index in enumerate(valid_indices):
                traj = all_trajectories[global_index]
                base_reward = float(getattr(traj, "base_reward", 0.0))
                cluster_id = semantic_info.cluster_assignments[local_index]
                cluster_size = semantic_info.cluster_sizes[cluster_id]
                cluster_probability = cluster_size / valid_count
                adjustment = 0.0

                if group_max_reward >= self.semantic_exploit_reward_threshold:
                    adjustment += (
                        self.semantic_consistency_weight
                        * cluster_probability
                        * max(cluster_mean_reward.get(cluster_id, 0.0), 0.0)
                    )
                else:
                    reward_gap = max(
                        0.0,
                        self.semantic_exploit_reward_threshold - group_max_reward,
                    )
                    adjustment += (
                        self.semantic_exploration_weight
                        * reward_gap
                        * semantic_info.normalized_entropy
                        * (1.0 - cluster_probability)
                    )
                    adjustment -= (
                        self.semantic_collapse_weight
                        * reward_gap
                        * (1.0 - semantic_info.normalized_entropy)
                    )

                adjustment = float(
                    np.clip(
                        adjustment,
                        -self.semantic_adjustment_clip,
                        self.semantic_adjustment_clip,
                    )
                )
                shaped_reward = base_reward + adjustment
                setattr(traj, "semantic_reward_adjustment", adjustment)
                setattr(traj, "semantic_shaped_reward", shaped_reward)
                shaped_rewards.append(shaped_reward)

            mean_reward = float(np.mean(shaped_rewards))
            std_reward = float(np.std(shaped_rewards)) if len(shaped_rewards) > 1 else 0.0

            if self.dynamic_sampling and std_reward <= self.dynamic_sampling_min_std:
                filtered_groups += 1
                for index in indices:
                    valid_mask[index] = False

            denom = std_reward if std_reward > 0 else 1.0
            for local_index, global_index in enumerate(valid_indices):
                all_trajectories[global_index].advantage = (
                    (shaped_rewards[local_index] - mean_reward) / (denom + 1e-8)
                )

            for global_index in indices:
                if global_index not in valid_indices:
                    all_trajectories[global_index].advantage = 0.0

            logger.debug(
                "EntropyTrainer group=%s max_reward=%.3f shaped_mean=%.3f shaped_std=%.3f "
                "clusters=%d semantic_entropy=%.3f",
                base_id,
                group_max_reward,
                mean_reward,
                std_reward,
                semantic_info.num_clusters,
                semantic_info.normalized_entropy,
            )

        if self.filter_overlong:
            for index, traj in enumerate(all_trajectories):
                if traj.termination_reason in {"max_turns_reached", "context_overflow"}:
                    valid_mask[index] = False

        return valid_mask, filtered_groups
