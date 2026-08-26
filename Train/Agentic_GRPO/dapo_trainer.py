# """
# DAPO trainer for agentic GRPO trajectories.

# This ports the DAPO loss used in the newer training stack while keeping the
# original training loop in this directory intact.
# """

# from __future__ import annotations

# import logging
# from typing import Any, Dict, List, Tuple

# import numpy as np
# import torch

# from data_structures import CompletedTrajectory
# from grpo_trainer import GRPOTrainer

# logger = logging.getLogger(__name__)


# class DAPOTrainer(GRPOTrainer):
#     """Decoupled Clip and Dynamic Sampling Policy Optimization trainer."""

#     def __init__(
#         self,
#         model: Any,
#         ref_model: Any,
#         optimizer: Any,
#         tokenizer: Any,
#         reward_function: Any,
#         beta: float = 0.0,
#         max_grad_norm: float = 1.0,
#         device: str = "cuda",
#         logprob_chunk_size: int = 128,
#         training_microbatch_size: int = 1,
#         use_wandb: bool = False,
#         epsilon: float = 0.2,
#         epsilon_high: float = 0.28,
#         dynamic_sampling: bool = True,
#         dynamic_sampling_min_std: float = 1e-6,
#         filter_overlong: bool = False,
#     ):
#         super().__init__(
#             model=model,
#             ref_model=ref_model,
#             optimizer=optimizer,
#             tokenizer=tokenizer,
#             reward_function=reward_function,
#             beta=beta,
#             max_grad_norm=max_grad_norm,
#             device=device,
#             logprob_chunk_size=logprob_chunk_size,
#             use_wandb=use_wandb,
#         )
#         self.epsilon = float(epsilon)
#         self.epsilon_high = float(epsilon_high)
#         self.dynamic_sampling = bool(dynamic_sampling)
#         self.dynamic_sampling_min_std = float(dynamic_sampling_min_std)
#         self.filter_overlong = bool(filter_overlong)
#         self.training_microbatch_size = training_microbatch_size
#         logger.info(
#             "DAPOTrainer initialized: epsilon=%.3f, epsilon_high=%.3f, dynamic_sampling=%s, beta=%.4f",
#             self.epsilon,
#             self.epsilon_high,
#             self.dynamic_sampling,
#             self.beta,
#         )

#     def _compute_token_logprobs_and_kl(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         outputs_policy = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             use_cache=False,
#         )
#         logits_policy = outputs_policy.logits

#         if self.beta == 0.0 or self.ref_model is self.model:
#             shift_logits_policy = logits_policy[:, :-1, :]
#             shift_labels = input_ids[:, 1:].contiguous()
#             policy_token_logprobs = self._compute_token_logprobs_chunked(
#                 shift_logits_policy,
#                 shift_labels,
#             )
#             # policy_token_entropies = self._compute_token_entropies_chunked(
#             #     shift_logits_policy
#             # )
#             policy_token_entropies = torch.zeros_like(policy_token_logprobs)
#             ref_token_logprobs = torch.zeros_like(policy_token_logprobs)
#             token_kl = torch.zeros_like(policy_token_logprobs)
#             return policy_token_logprobs, ref_token_logprobs, token_kl, policy_token_entropies

#         with torch.no_grad():
#             ref_device = next(self.ref_model.parameters()).device
#             if ref_device.type == "cpu":
#                 input_ids_ref = input_ids.cpu()
#                 attention_mask_ref = attention_mask.cpu()
#             else:
#                 input_ids_ref = input_ids
#                 attention_mask_ref = attention_mask

#             outputs_ref = self.ref_model(
#                 input_ids=input_ids_ref,
#                 attention_mask=attention_mask_ref,
#                 use_cache=False,
#             )
#             logits_ref = outputs_ref.logits
#             if ref_device.type == "cpu":
#                 logits_ref = logits_ref.to(logits_policy.device)

#         shift_logits_policy = logits_policy[:, :-1, :].contiguous()
#         shift_logits_ref = logits_ref[:, :-1, :].contiguous()
#         shift_labels = input_ids[:, 1:].contiguous()

#         policy_token_logprobs = self._compute_token_logprobs_chunked(
#             shift_logits_policy,
#             shift_labels,
#         )
#         # policy_token_entropies = self._compute_token_entropies_chunked(
#         #     shift_logits_policy
#         # )
#         policy_token_entropies = torch.zeros_like(policy_token_logprobs)
#         with torch.no_grad():
#             ref_token_logprobs = self._compute_token_logprobs_chunked(
#                 shift_logits_ref,
#                 shift_labels,
#             )
#         token_kl = policy_token_logprobs - ref_token_logprobs
#         return policy_token_logprobs, ref_token_logprobs, token_kl, policy_token_entropies

#     @staticmethod
#     def _flatten_trajectories(
#         batch_trajectories: Dict[str, List[CompletedTrajectory]],
#     ) -> List[CompletedTrajectory]:
#         all_trajectories: List[CompletedTrajectory] = []
#         for _, trajs in batch_trajectories.items():
#             all_trajectories.extend(trajs)
#         return all_trajectories

#     def _assign_rewards_and_advantages(
#         self,
#         all_trajectories: List[CompletedTrajectory],
#         initial_valid_mask: List[bool],
#     ) -> Tuple[List[bool], int]:
#         valid_mask = list(initial_valid_mask)
#         for i, traj in enumerate(all_trajectories):
#             if traj.reward is None:
#                 task_id = traj.query_id.rsplit("_g", 1)[0] if "_g" in traj.query_id else traj.query_id
#                 try:
#                     traj.reward = self.reward_function(task_id, traj)
#                 except Exception as exc:
#                     logger.error("Reward computation failed for %s: %s", task_id, exc)
#             if traj.reward is None:
#                 traj.reward = 0.0
#                 valid_mask[i] = False

#         groups: Dict[str, List[int]] = {}
#         for index, traj in enumerate(all_trajectories):
#             base_id = traj.query_id.rsplit("_g", 1)[0] if "_g" in traj.query_id else traj.query_id
#             groups.setdefault(base_id, []).append(index)

#         filtered_groups = 0
#         for base_id, indices in groups.items():
#             rewards = [float(all_trajectories[i].reward or 0.0) for i in indices]
#             mean_reward = float(np.mean(rewards))
#             std_reward = float(np.std(rewards)) if len(rewards) > 1 else 0.0

#             if self.dynamic_sampling and std_reward <= self.dynamic_sampling_min_std:
#                 filtered_groups += 1
#                 for i in indices:
#                     valid_mask[i] = False

#             denom = std_reward if std_reward > 0 else 1.0
#             for i in indices:
#                 all_trajectories[i].advantage = (
#                     (float(all_trajectories[i].reward or 0.0) - mean_reward)
#                     / (denom + 1e-8)
#                 )
#             if std_reward <= self.dynamic_sampling_min_std:
#                 logger.debug("DAPO filtered zero-variance group %s with rewards=%s", base_id, rewards)

#         if self.filter_overlong:
#             for i, traj in enumerate(all_trajectories):
#                 if traj.termination_reason in {"max_turns_reached", "context_overflow"}:
#                     valid_mask[i] = False

#         return valid_mask, filtered_groups

#     def train_step(
#         self,
#         batch_trajectories: Dict[str, List[CompletedTrajectory]],
#         collated_batch: Dict[str, torch.Tensor],
#         loss_scale: float = 1.0,
#     ) -> Dict[str, float]:
#         input_ids = collated_batch["input_ids"].to(self.device)
#         attention_mask = collated_batch["attention_mask"].to(self.device)
#         loss_mask = collated_batch["loss_mask"].to(self.device)
#         batch_size = input_ids.shape[0]

#         all_trajectories = self._flatten_trajectories(batch_trajectories)
#         assert len(all_trajectories) == batch_size, (
#             f"Mismatch: {len(all_trajectories)} trajectories vs {batch_size} collated"
#         )

#         initial_valid_mask: List[bool] = []
#         for traj in all_trajectories:
#             is_valid, error_msg = traj.validate_structure()
#             if not is_valid:
#                 logger.warning("Invalid trajectory %s: %s", traj.query_id, error_msg)
#             initial_valid_mask.append(is_valid)

#         valid_mask_list, filtered_groups = self._assign_rewards_and_advantages(
#             all_trajectories,
#             initial_valid_mask,
#         )
#         valid_mask = torch.tensor(valid_mask_list, dtype=torch.bool, device=self.device)
#         quality_metrics = self._build_quality_metric_payload(
#             all_trajectories,
#             valid_mask,
#         )

#         rewards = [float(traj.reward or 0.0) for traj in all_trajectories]
#         if valid_mask.sum() == 0:
#             logger.warning("DAPO found no valid non-zero-variance trajectories in batch")
#             return {
#                 "loss": 0.0,
#                 "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
#                 "std_reward": float(np.std(rewards)) if rewards else 0.0,
#                 "avg_advantage": 0.0,
#                 "std_advantage": 0.0,
#                 "avg_kl": 0.0,
#                 "grad_norm": 0.0,
#                 "num_valid": 0,
#                 "num_total": batch_size,
#                 "num_success": int(sum(quality_metrics["_protocol_complete_values"])),
#                 "num_errors": sum(
#                     traj.termination_reason != "success"
#                     for traj in all_trajectories
#                 ),
#                 "num_dapo_filtered_groups": filtered_groups,
#                 "avg_token_entropy": 0.0,
#                 "max_token_entropy": 0.0,
#                 "min_token_entropy": 0.0,
#                 "semantic_entropy": 0.0,
#                 "_loss_weighted_sum": 0.0,
#                 "_reward_values": rewards,
#                 "_advantage_values": [0.0 for _ in all_trajectories],
#                 "_kl_values": [],
#                 "_policy_logprob_sum": 0.0,
#                 "_ref_logprob_sum": 0.0,
#                 "_token_entropy_weighted_sum": 0.0,
#                 "_active_token_count": 0.0,
#                 "_token_entropy_values": [],
#                 **quality_metrics,
#             }

#         policy_token_logprobs, ref_token_logprobs, token_kl, policy_token_entropies = self._compute_token_logprobs_and_kl(
#             input_ids,
#             attention_mask,
#         )

#         shift_loss_mask = loss_mask[:, 1:].contiguous().float()
#         active_token_mask = shift_loss_mask * valid_mask.float().unsqueeze(1)

#         old_policy_token_logprobs = policy_token_logprobs.detach()
#         ratios = torch.exp(policy_token_logprobs - old_policy_token_logprobs)
#         clipped_ratios = torch.clamp(
#             ratios,
#             min=1.0 - self.epsilon,
#             max=1.0 + self.epsilon_high,
#         )

#         advantages = torch.tensor(
#             [float(traj.advantage or 0.0) for traj in all_trajectories],
#             dtype=torch.float32,
#             device=self.device,
#         ).unsqueeze(1)

#         unclipped_objective = ratios * advantages
#         clipped_objective = clipped_ratios * advantages
#         token_pg_loss = -torch.minimum(unclipped_objective, clipped_objective)
#         token_loss = (token_pg_loss + self.beta * token_kl) * active_token_mask

#         num_active_tokens = active_token_mask.sum().clamp(min=1.0)
#         loss = token_loss.sum() / num_active_tokens
#         unscaled_loss = float(loss.item())

#         (loss / loss_scale).backward()

#         with torch.no_grad():
#             advantages_np = [float(traj.advantage or 0.0) for traj in all_trajectories]
#             masked_kl = token_kl * active_token_mask
#             token_entropy_sum = (policy_token_entropies * active_token_mask).sum()
#             kl_denominator = shift_loss_mask.sum(dim=1).clamp(min=1.0)
#             kl_per_traj = masked_kl.sum(dim=1) / kl_denominator
#             policy_traj_logprobs = (
#                 (policy_token_logprobs * shift_loss_mask).sum(dim=1)
#                 / kl_denominator
#             )
#             ref_traj_logprobs = (
#                 (ref_token_logprobs * shift_loss_mask).sum(dim=1)
#                 / kl_denominator
#             )
#             clipped_fraction = (
#                 ((ratios < 1.0 - self.epsilon) | (ratios > 1.0 + self.epsilon_high)).float()
#                 * active_token_mask
#             ).sum() / num_active_tokens

#             termination_reasons: Dict[str, int] = {}
#             for traj in all_trajectories:
#                 reason = traj.termination_reason
#                 termination_reasons[reason] = termination_reasons.get(reason, 0) + 1

#             valid_kl_values = kl_per_traj[valid_mask].detach().cpu().tolist()
#             token_entropy_per_traj = (
#                 (policy_token_entropies * active_token_mask).sum(dim=1)
#                 / active_token_mask.sum(dim=1).clamp(min=1.0)
#             )
#             valid_token_entropy_values = token_entropy_per_traj[valid_mask].detach().cpu().tolist()

#             # Compute semantic entropy
#             semantic_entropy = self._compute_semantic_entropy(all_trajectories, valid_mask)

#             metrics = {
#                 "loss": unscaled_loss,
#                 "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
#                 "std_reward": float(np.std(rewards)) if rewards else 0.0,
#                 "min_reward": float(np.min(rewards)) if rewards else 0.0,
#                 "max_reward": float(np.max(rewards)) if rewards else 0.0,
#                 "median_reward": float(np.median(rewards)) if rewards else 0.0,
#                 "avg_advantage": float(np.mean(advantages_np)) if advantages_np else 0.0,
#                 "std_advantage": float(np.std(advantages_np)) if advantages_np else 0.0,
#                 "avg_kl": float(masked_kl.sum().item() / num_active_tokens.item()),
#                 "max_kl": float(max(valid_kl_values)) if valid_kl_values else 0.0,
#                 "min_kl": float(min(valid_kl_values)) if valid_kl_values else 0.0,
#                 "avg_token_entropy": float(token_entropy_sum.item() / num_active_tokens.item()),
#                 "max_token_entropy": float(max(valid_token_entropy_values)) if valid_token_entropy_values else 0.0,
#                 "min_token_entropy": float(min(valid_token_entropy_values)) if valid_token_entropy_values else 0.0,
#                 "semantic_entropy": semantic_entropy,
#                 "num_valid": int(valid_mask.sum().item()),
#                 "num_total": batch_size,
#                 "num_success": int(sum(quality_metrics["_protocol_complete_values"])),
#                 "num_max_turns": termination_reasons.get("max_turns_reached", 0),
#                 "num_errors": sum(
#                     count
#                     for reason, count in termination_reasons.items()
#                     if reason != "success"
#                 ),
#                 "policy_logprob_mean": float(policy_traj_logprobs.mean().item()),
#                 "ref_logprob_mean": float(ref_traj_logprobs.mean().item()),
#                 "dapo_clip_fraction": float(clipped_fraction.item()),
#                 "dapo_active_tokens": float(num_active_tokens.item()),
#                 "num_dapo_filtered_groups": filtered_groups,
#                 "_loss_weighted_sum": unscaled_loss * int(valid_mask.sum().item()),
#                 "_reward_values": rewards,
#                 "_advantage_values": advantages_np,
#                 "_kl_values": valid_kl_values,
#                 "_policy_logprob_sum": float(policy_traj_logprobs.sum().item()),
#                 "_ref_logprob_sum": float(ref_traj_logprobs.sum().item()),
#                 "_token_entropy_weighted_sum": float(token_entropy_sum.item()),
#                 "_active_token_count": float(num_active_tokens.item()),
#                 "_token_entropy_values": valid_token_entropy_values,
#                 **quality_metrics,
#             }

#         return metrics


"""
DAPO trainer for agentic GRPO trajectories.

This ports the DAPO loss used in the newer training stack while keeping the
original training loop in this directory intact.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from data_structures import CompletedTrajectory
from grpo_trainer import GRPOTrainer

logger = logging.getLogger(__name__)


class DAPOTrainer(GRPOTrainer):
    """Decoupled Clip and Dynamic Sampling Policy Optimization trainer."""

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
        training_microbatch_size: int = 1,
        compute_token_entropy: bool = False,
        use_wandb: bool = False,
        epsilon: float = 0.2,
        epsilon_high: float = 0.28,
        dynamic_sampling: bool = True,
        dynamic_sampling_min_std: float = 1e-6,
        filter_overlong: bool = False,
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
            training_microbatch_size=training_microbatch_size,
            compute_token_entropy=compute_token_entropy,
            use_wandb=use_wandb,
        )
        self.epsilon = float(epsilon)
        self.epsilon_high = float(epsilon_high)
        self.dynamic_sampling = bool(dynamic_sampling)
        self.dynamic_sampling_min_std = float(dynamic_sampling_min_std)
        self.filter_overlong = bool(filter_overlong)
        self._warned_missing_old_policy_logprobs = False
        logger.info(
            "DAPOTrainer initialized: epsilon=%.3f, epsilon_high=%.3f, "
            "dynamic_sampling=%s, beta=%.4f, training_microbatch_size=%d",
            self.epsilon,
            self.epsilon_high,
            self.dynamic_sampling,
            self.beta,
            self.training_microbatch_size,
        )

    def _compute_token_logprobs_and_kl(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use the base class's true LM-head chunking implementation."""
        return self._compute_token_logprobs_and_kl_from_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    @staticmethod
    def _flatten_trajectories(
        batch_trajectories: Dict[str, List[CompletedTrajectory]],
    ) -> List[CompletedTrajectory]:
        all_trajectories: List[CompletedTrajectory] = []
        for _, trajs in batch_trajectories.items():
            all_trajectories.extend(trajs)
        return all_trajectories

    def _assign_rewards_and_advantages(
        self,
        all_trajectories: List[CompletedTrajectory],
        initial_valid_mask: List[bool],
    ) -> Tuple[List[bool], int]:
        valid_mask = list(initial_valid_mask)
        for i, traj in enumerate(all_trajectories):
            if traj.reward is None:
                task_id = traj.query_id.rsplit("_g", 1)[0] if "_g" in traj.query_id else traj.query_id
                try:
                    traj.reward = self.reward_function(task_id, traj)
                except Exception as exc:
                    logger.error("Reward computation failed for %s: %s", task_id, exc)
            if traj.reward is None:
                traj.reward = 0.0
                valid_mask[i] = False

        groups: Dict[str, List[int]] = {}
        for index, traj in enumerate(all_trajectories):
            base_id = traj.query_id.rsplit("_g", 1)[0] if "_g" in traj.query_id else traj.query_id
            groups.setdefault(base_id, []).append(index)

        filtered_groups = 0
        for base_id, indices in groups.items():
            rewards = [float(all_trajectories[i].reward or 0.0) for i in indices]
            mean_reward = float(np.mean(rewards))
            std_reward = float(np.std(rewards)) if len(rewards) > 1 else 0.0

            if self.dynamic_sampling and std_reward <= self.dynamic_sampling_min_std:
                filtered_groups += 1
                for i in indices:
                    valid_mask[i] = False

            denom = std_reward if std_reward > 0 else 1.0
            for i in indices:
                all_trajectories[i].advantage = (
                    (float(all_trajectories[i].reward or 0.0) - mean_reward)
                    / (denom + 1e-8)
                )
            if std_reward <= self.dynamic_sampling_min_std:
                logger.debug("DAPO filtered zero-variance group %s with rewards=%s", base_id, rewards)

        if self.filter_overlong:
            for i, traj in enumerate(all_trajectories):
                if traj.termination_reason in {"max_turns_reached", "context_overflow"}:
                    valid_mask[i] = False

        return valid_mask, filtered_groups

    def train_step(
        self,
        batch_trajectories: Dict[str, List[CompletedTrajectory]],
        collated_batch: Dict[str, torch.Tensor],
        loss_scale: float = 1.0,
    ) -> Dict[str, float]:
        """Run DAPO while forwarding only a trajectory microbatch at a time."""
        input_ids = collated_batch["input_ids"].to(self.device)
        attention_mask = collated_batch["attention_mask"].to(self.device)
        loss_mask = collated_batch["loss_mask"].to(self.device)
        batch_size = input_ids.shape[0]

        all_trajectories = self._flatten_trajectories(batch_trajectories)
        assert len(all_trajectories) == batch_size, (
            f"Mismatch: {len(all_trajectories)} trajectories vs {batch_size} collated"
        )

        initial_valid_mask: List[bool] = []
        for trajectory in all_trajectories:
            is_valid, error_message = trajectory.validate_structure()
            if not is_valid:
                logger.warning(
                    "Invalid trajectory %s: %s", trajectory.query_id, error_message
                )
            initial_valid_mask.append(is_valid)

        valid_mask_list, filtered_groups = self._assign_rewards_and_advantages(
            all_trajectories,
            initial_valid_mask,
        )
        valid_mask = torch.tensor(
            valid_mask_list, dtype=torch.bool, device=self.device
        )
        quality_metrics = self._build_quality_metric_payload(
            all_trajectories,
            valid_mask,
        )
        rewards = [float(trajectory.reward or 0.0) for trajectory in all_trajectories]

        if valid_mask.sum() == 0:
            logger.warning("DAPO found no valid non-zero-variance trajectories in batch")
            return {
                "loss": 0.0,
                "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
                "std_reward": float(np.std(rewards)) if rewards else 0.0,
                "avg_advantage": 0.0,
                "std_advantage": 0.0,
                "avg_kl": 0.0,
                "grad_norm": 0.0,
                "num_valid": 0,
                "num_total": batch_size,
                "num_success": int(sum(quality_metrics["_protocol_complete_values"])),
                "num_errors": sum(
                    trajectory.termination_reason != "success"
                    for trajectory in all_trajectories
                ),
                "num_dapo_filtered_groups": filtered_groups,
                "avg_token_entropy": 0.0,
                "max_token_entropy": 0.0,
                "min_token_entropy": 0.0,
                "semantic_entropy": 0.0,
                "_loss_weighted_sum": 0.0,
                "_reward_values": rewards,
                "_advantage_values": [0.0 for _ in all_trajectories],
                "_kl_values": [],
                "_policy_logprob_sum": 0.0,
                "_ref_logprob_sum": 0.0,
                "_token_entropy_weighted_sum": 0.0,
                "_active_token_count": 0.0,
                "_token_entropy_values": [],
                "_expected_zero_gradient": True,
                **quality_metrics,
            }

        advantages = torch.tensor(
            [float(trajectory.advantage or 0.0) for trajectory in all_trajectories],
            dtype=torch.float32,
            device=self.device,
        )
        full_shift_loss_mask = loss_mask[:, 1:].contiguous().float()
        full_active_token_mask = (
            full_shift_loss_mask * valid_mask.float().unsqueeze(1)
        )
        global_active_tokens = full_active_token_mask.sum().clamp(min=1.0)

        old_policy_logprobs = collated_batch.get("old_policy_logprobs")
        if old_policy_logprobs is not None:
            old_policy_logprobs = old_policy_logprobs.to(self.device)
            if old_policy_logprobs.ndim != 2:
                raise ValueError("old_policy_logprobs must have shape [B, S-1] or [B, S]")
            if old_policy_logprobs.shape[0] != batch_size:
                raise ValueError("old_policy_logprobs batch dimension does not match input_ids")
            if old_policy_logprobs.shape[1] == input_ids.shape[1]:
                old_policy_logprobs = old_policy_logprobs[:, 1:]
            elif old_policy_logprobs.shape[1] != input_ids.shape[1] - 1:
                raise ValueError(
                    "old_policy_logprobs sequence dimension must equal S-1 or S"
                )
        elif not self._warned_missing_old_policy_logprobs:
            logger.warning(
                "No old_policy_logprobs were provided by the collator. DAPO will "
                "use detached current-policy logprobs, so ratios start at exactly 1 "
                "and clipping has no effect for this one-pass update."
            )
            self._warned_missing_old_policy_logprobs = True

        policy_traj_values = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        ref_traj_values = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        kl_traj_values = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        entropy_per_traj_values = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )

        total_loss_sum = 0.0
        total_entropy_sum = 0.0
        total_clipped_tokens = 0.0

        for micro_start in range(0, batch_size, self.training_microbatch_size):
            micro_end = min(
                micro_start + self.training_microbatch_size,
                batch_size,
            )
            micro_valid_mask = valid_mask[micro_start:micro_end]
            if not bool(micro_valid_mask.any().item()):
                continue

            (
                micro_input_ids,
                micro_attention_mask,
                micro_loss_mask,
                first_column,
                last_column,
            ) = self._slice_and_trim_batch(
                input_ids,
                attention_mask,
                loss_mask,
                micro_start,
                micro_end,
            )

            (
                policy_token_logprobs,
                ref_token_logprobs,
                token_kl,
                policy_token_entropies,
            ) = self._compute_token_logprobs_and_kl(
                micro_input_ids,
                micro_attention_mask,
            )

            shift_loss_mask = micro_loss_mask[:, 1:].contiguous().float()
            active_token_mask = (
                shift_loss_mask * micro_valid_mask.float().unsqueeze(1)
            )
            micro_advantages = advantages[micro_start:micro_end].unsqueeze(1)

            if old_policy_logprobs is None:
                micro_old_logprobs = policy_token_logprobs.detach()
            else:
                # Full old-logprob index j predicts original token j+1. After
                # trimming input columns [first_column:last_column], keep the
                # corresponding prediction positions [first_column:last_column-1].
                micro_old_logprobs = old_policy_logprobs[
                    micro_start:micro_end,
                    first_column:last_column - 1,
                ].to(policy_token_logprobs.device)
                if micro_old_logprobs.shape != policy_token_logprobs.shape:
                    raise RuntimeError(
                        "Trimmed old_policy_logprobs do not match current logprobs: "
                        f"{tuple(micro_old_logprobs.shape)} vs "
                        f"{tuple(policy_token_logprobs.shape)}"
                    )

            ratios = torch.exp(policy_token_logprobs - micro_old_logprobs)
            clipped_ratios = torch.clamp(
                ratios,
                min=1.0 - self.epsilon,
                max=1.0 + self.epsilon_high,
            )
            unclipped_objective = ratios * micro_advantages
            clipped_objective = clipped_ratios * micro_advantages
            token_pg_loss = -torch.minimum(unclipped_objective, clipped_objective)
            token_loss = (
                token_pg_loss + self.beta * token_kl
            ) * active_token_mask
            micro_loss_sum = token_loss.sum()
            (
                micro_loss_sum
                / global_active_tokens
                / float(loss_scale)
            ).backward()

            with torch.no_grad():
                token_counts = active_token_mask.sum(dim=1).clamp(min=1.0)
                policy_traj_logprobs = (
                    policy_token_logprobs * active_token_mask
                ).sum(dim=1) / token_counts
                ref_traj_logprobs = (
                    ref_token_logprobs * active_token_mask
                ).sum(dim=1) / token_counts
                kl_per_traj = (
                    token_kl * active_token_mask
                ).sum(dim=1) / token_counts
                entropy_per_traj = (
                    policy_token_entropies * active_token_mask
                ).sum(dim=1) / token_counts

                policy_traj_values[micro_start:micro_end] = policy_traj_logprobs.detach().to(
                    self.device, dtype=torch.float32
                )
                ref_traj_values[micro_start:micro_end] = ref_traj_logprobs.detach().to(
                    self.device, dtype=torch.float32
                )
                kl_traj_values[micro_start:micro_end] = kl_per_traj.detach().to(
                    self.device, dtype=torch.float32
                )
                entropy_per_traj_values[micro_start:micro_end] = entropy_per_traj.detach().to(
                    self.device, dtype=torch.float32
                )

                total_loss_sum += float(micro_loss_sum.detach().item())
                total_entropy_sum += float(
                    (policy_token_entropies * active_token_mask).sum().item()
                )
                total_clipped_tokens += float(
                    (
                        (
                            (ratios < 1.0 - self.epsilon)
                            | (ratios > 1.0 + self.epsilon_high)
                        ).float()
                        * active_token_mask
                    ).sum().item()
                )

            del (
                policy_token_logprobs,
                ref_token_logprobs,
                token_kl,
                policy_token_entropies,
                ratios,
                clipped_ratios,
                token_loss,
                micro_loss_sum,
            )

        unscaled_loss = total_loss_sum / float(global_active_tokens.item())

        with torch.no_grad():
            advantages_np = [
                float(trajectory.advantage or 0.0) for trajectory in all_trajectories
            ]
            valid_kl_values = kl_traj_values[valid_mask].detach().cpu().tolist()
            valid_token_entropy_values = (
                entropy_per_traj_values[valid_mask].detach().cpu().tolist()
            )
            semantic_entropy = self._compute_semantic_entropy(
                all_trajectories,
                valid_mask,
            )

            termination_reasons: Dict[str, int] = {}
            for trajectory in all_trajectories:
                reason = trajectory.termination_reason
                termination_reasons[reason] = termination_reasons.get(reason, 0) + 1

            active_token_count = float(global_active_tokens.item())
            clipped_fraction = total_clipped_tokens / max(active_token_count, 1.0)
            expected_zero_gradient = (
                self.beta == 0.0
                and all(
                    abs(advantages_np[index]) <= 1e-12
                    for index, is_valid in enumerate(valid_mask_list)
                    if is_valid
                )
            )

            metrics = {
                "loss": unscaled_loss,
                "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
                "std_reward": float(np.std(rewards)) if rewards else 0.0,
                "min_reward": float(np.min(rewards)) if rewards else 0.0,
                "max_reward": float(np.max(rewards)) if rewards else 0.0,
                "median_reward": float(np.median(rewards)) if rewards else 0.0,
                "avg_advantage": float(np.mean(advantages_np)) if advantages_np else 0.0,
                "std_advantage": float(np.std(advantages_np)) if advantages_np else 0.0,
                "avg_kl": (
                    float((kl_traj_values[valid_mask]).mean().item())
                    if bool(valid_mask.any().item())
                    else 0.0
                ),
                "max_kl": float(max(valid_kl_values)) if valid_kl_values else 0.0,
                "min_kl": float(min(valid_kl_values)) if valid_kl_values else 0.0,
                "avg_token_entropy": total_entropy_sum / max(active_token_count, 1.0),
                "max_token_entropy": (
                    float(max(valid_token_entropy_values))
                    if valid_token_entropy_values
                    else 0.0
                ),
                "min_token_entropy": (
                    float(min(valid_token_entropy_values))
                    if valid_token_entropy_values
                    else 0.0
                ),
                "semantic_entropy": semantic_entropy,
                "num_valid": int(valid_mask.sum().item()),
                "num_total": batch_size,
                "num_success": int(sum(quality_metrics["_protocol_complete_values"])),
                "num_max_turns": termination_reasons.get("max_turns_reached", 0),
                "num_errors": sum(
                    count
                    for reason, count in termination_reasons.items()
                    if reason != "success"
                ),
                "policy_logprob_mean": float(policy_traj_values.mean().item()),
                "ref_logprob_mean": float(ref_traj_values.mean().item()),
                "dapo_clip_fraction": clipped_fraction,
                "dapo_active_tokens": active_token_count,
                "num_dapo_filtered_groups": filtered_groups,
                "_loss_weighted_sum": unscaled_loss * int(valid_mask.sum().item()),
                "_reward_values": rewards,
                "_advantage_values": advantages_np,
                "_kl_values": valid_kl_values,
                "_policy_logprob_sum": float(policy_traj_values.sum().item()),
                "_ref_logprob_sum": float(ref_traj_values.sum().item()),
                "_token_entropy_weighted_sum": total_entropy_sum,
                "_active_token_count": active_token_count,
                "_token_entropy_values": valid_token_entropy_values,
                "_expected_zero_gradient": expected_zero_gradient,
                **quality_metrics,
            }

        return metrics