# """
# GRPO Trainer for Agentic RL - PRODUCTION READY

# FIXES APPLIED:
# ✅ Token-level KL divergence (not trajectory-level heuristic)
# ✅ Memory-efficient chunked logprob computation
# ✅ Length-normalized trajectory logprobs for policy gradient
# ✅ Proper mask shift alignment for causal LM
# ✅ Identity check for reference model (not equality)
# ✅ All imports and dependencies resolved
# ✅ Proper checkpoint handling
# ✅ Compatible with TrajectoryCollator segment-based masking
# """

# import os
# import random
# import re
# import torch
# import torch.nn.functional as F
# import logging
# from typing import List, Dict, Any, Optional, Callable, Tuple
# import numpy as np
# from tqdm import tqdm
# from collections import defaultdict

# from data_structures import CompletedTrajectory

# logger = logging.getLogger(__name__)


# class GRPOTrainer:
#     """
#     Group Relative Policy Optimization trainer with proper token-level KL.
    
#     GRPO objective:
#     L = -E[A(τ) * log π(τ) - β * KL(π || π_ref)]
    
#     where:
#     - A(τ) is group-relative advantage
#     - log π(τ) is length-normalized trajectory log probability
#     - KL is computed at token level and averaged over trainable tokens
#     """

#     def __init__(
#         self,
#         model: Any,
#         ref_model: Any,
#         optimizer: Any,
#         tokenizer: Any,
#         reward_function: Callable,
#         beta: float = 0.01,
#         max_grad_norm: float = 1.0,
#         device: str = "cuda",
#         logprob_chunk_size: int = 128,
#         use_wandb: bool = False
#     ):
#         """
#         Args:
#             model: Policy model to train
#             ref_model: Reference model (frozen)
#             optimizer: Optimizer for policy model
#             tokenizer: Tokenizer (needed for checkpointing)
#             reward_function: Function (task_id, trajectory) -> float
#             beta: KL penalty coefficient
#             max_grad_norm: Gradient clipping norm
#             device: Device to use
#             logprob_chunk_size: Chunk size for memory-efficient logprob computation
#             use_wandb: Whether to use Weights & Biases logging
#         """
#         self.model = model
#         self.ref_model = ref_model
#         self.optimizer = optimizer
#         self.tokenizer = tokenizer
#         self.reward_function = reward_function
#         self.beta = beta
#         self.max_grad_norm = max_grad_norm
#         self.device = device
#         self.logprob_chunk_size = logprob_chunk_size
#         self.use_wandb = use_wandb

#         # Set models to appropriate modes
#         self.model.train()
#         self.ref_model.eval()
#         self.reward_ema = None
#         self.reward_ema_alpha = 0.1  # Smoothing factor
#         # =======================================
        
#         # Freeze reference model
#         for param in self.ref_model.parameters():
#             param.requires_grad = False
        
#         logger.info(f"GRPOTrainer initialized with beta={beta}, max_grad_norm={max_grad_norm}")

#     def _capture_rng_state(self) -> Dict[str, Any]:
#         state: Dict[str, Any] = {
#             "python_random_state": random.getstate(),
#             "numpy_random_state": np.random.get_state(),
#             "torch_random_state": torch.random.get_rng_state(),
#         }
#         if torch.cuda.is_available():
#             state["cuda_random_state_all"] = torch.cuda.get_rng_state_all()
#         return state

#     def _restore_rng_state(self, rng_state: Optional[Dict[str, Any]]) -> None:
#         if not rng_state:
#             return
#         if "python_random_state" in rng_state:
#             random.setstate(rng_state["python_random_state"])
#         if "numpy_random_state" in rng_state:
#             np.random.set_state(rng_state["numpy_random_state"])
#         if "torch_random_state" in rng_state:
#             torch.random.set_rng_state(rng_state["torch_random_state"])
#         if torch.cuda.is_available() and "cuda_random_state_all" in rng_state:
#             torch.cuda.set_rng_state_all(rng_state["cuda_random_state_all"])

#     def _save_checkpoint(
#         self,
#         checkpoint_path: str,
#         epoch: int,
#         next_batch_idx: int,
#         global_step: int,
#         update_step: int,
#         num_epochs: int,
#         batch_size: int,
#         group_size: int,
#         gradient_accumulation_steps: int,
#         shuffled_query_ids: List[str],
#         all_metrics: List[Dict[str, Any]],
#     ) -> None:
#         os.makedirs(checkpoint_path, exist_ok=True)
#         self.model.save_pretrained(checkpoint_path)
#         self.tokenizer.save_pretrained(checkpoint_path)
#         trainer_state = {
#             "epoch": epoch,
#             "next_batch_idx": next_batch_idx,
#             "global_step": global_step,
#             "update_step": update_step,
#             "reward_ema": self.reward_ema,
#             "optimizer_state_dict": self.optimizer.state_dict(),
#             "num_epochs": num_epochs,
#             "batch_size": batch_size,
#             "group_size": group_size,
#             "gradient_accumulation_steps": gradient_accumulation_steps,
#             "shuffled_query_ids": shuffled_query_ids,
#             "all_metrics": all_metrics,
#             "rng_state": self._capture_rng_state(),
#         }
#         torch.save(trainer_state, os.path.join(checkpoint_path, "trainer_state.pt"))

#     @staticmethod
#     def _trajectory_base_id(query_id: str) -> str:
#         return query_id.rsplit("_g", 1)[0] if "_g" in query_id else query_id

#     @staticmethod
#     def _breakdown_value(trajectory: CompletedTrajectory, name: str) -> Any:
#         breakdown = getattr(trajectory, "reward_breakdown", None)
#         if isinstance(breakdown, dict):
#             return breakdown.get(name)
#         return getattr(breakdown, name, None) if breakdown is not None else None

#     @classmethod
#     def _trajectory_prefix_length(cls, trajectory: CompletedTrajectory) -> Optional[int]:
#         gold_length = cls._breakdown_value(trajectory, "gold_sequence_length")
#         if gold_length is not None:
#             try:
#                 parsed_length = int(gold_length)
#                 if parsed_length > 0:
#                     return parsed_length
#             except (TypeError, ValueError):
#                 pass

#         match = re.search(r"_p(\d+)(?:_g\d+)?$", trajectory.query_id)
#         return int(match.group(1)) if match else None

#     @classmethod
#     def _build_quality_metric_payload(
#         cls,
#         trajectories: List[CompletedTrajectory],
#         valid_mask: Any,
#     ) -> Dict[str, Any]:
#         """Return raw values so accumulation can aggregate without ratio bias."""
#         if hasattr(valid_mask, "detach"):
#             valid_values = [bool(value) for value in valid_mask.detach().cpu().tolist()]
#         else:
#             valid_values = [bool(value) for value in valid_mask]

#         base_rewards_all: List[float] = []
#         shaped_rewards_active: List[float] = []
#         final_by_prefix: List[Tuple[int, float]] = []
#         required_lcs_fractions: List[float] = []
#         format_valid_values: List[float] = []
#         protocol_complete_values: List[float] = []
#         semantic_adjustments_active: List[float] = []
#         active_advantages: List[float] = []

#         semantic_high_by_group: Dict[str, float] = {}
#         semantic_low_by_group: Dict[str, float] = {}

#         for is_valid, trajectory in zip(valid_values, trajectories):
#             base_reward = float(
#                 getattr(
#                     trajectory,
#                     "base_reward",
#                     trajectory.reward if trajectory.reward is not None else 0.0,
#                 )
#             )
#             base_rewards_all.append(base_reward)
#             protocol_complete_values.append(
#                 1.0 if trajectory.termination_reason == "success" else 0.0
#             )

#             prefix_length = cls._trajectory_prefix_length(trajectory)
#             final_reward = cls._breakdown_value(trajectory, "final_reward")
#             if prefix_length is not None and final_reward is not None:
#                 final_by_prefix.append((prefix_length, float(final_reward) > 0.0))

#             required_lcs = cls._breakdown_value(trajectory, "required_lcs_length")
#             gold_length = cls._breakdown_value(trajectory, "gold_sequence_length")
#             if required_lcs is not None and gold_length:
#                 required_lcs_fractions.append(
#                     float(required_lcs) / max(float(gold_length), 1.0)
#                 )

#             format_valid = cls._breakdown_value(trajectory, "format_valid")
#             if format_valid is not None:
#                 format_valid_values.append(1.0 if bool(format_valid) else 0.0)

#             group_id = cls._trajectory_base_id(trajectory.query_id)
#             semantic_entropy = getattr(
#                 trajectory, "semantic_group_logging_entropy", None
#             )
#             semantic_high_reward = getattr(
#                 trajectory, "semantic_group_high_reward", None
#             )
#             if semantic_entropy is not None and semantic_high_reward is not None:
#                 target = (
#                     semantic_high_by_group
#                     if bool(semantic_high_reward)
#                     else semantic_low_by_group
#                 )
#                 target[group_id] = float(semantic_entropy)

#             if not is_valid:
#                 continue

#             shaped_rewards_active.append(
#                 float(getattr(trajectory, "semantic_shaped_reward", base_reward))
#             )
#             semantic_adjustments_active.append(
#                 float(getattr(trajectory, "semantic_reward_adjustment", 0.0))
#             )
#             active_advantages.append(float(trajectory.advantage or 0.0))

#         return {
#             "_base_reward_values_all": base_rewards_all,
#             "_shaped_reward_values_active": shaped_rewards_active,
#             "_final_by_prefix": final_by_prefix,
#             "_required_lcs_fraction_values": required_lcs_fractions,
#             "_format_valid_values": format_valid_values,
#             "_protocol_complete_values": protocol_complete_values,
#             "_semantic_adjustment_values_active": semantic_adjustments_active,
#             "_active_advantage_values": active_advantages,
#             "_semantic_high_values": list(semantic_high_by_group.values()),
#             "_semantic_low_values": list(semantic_low_by_group.values()),
#         }

#     @staticmethod
#     def _mean_or_nan(values: List[float]) -> float:
#         return float(np.mean(values)) if values else float("nan")

#     def _aggregate_accumulated_metrics(
#         self,
#         metrics_window: List[Dict[str, Any]]
#     ) -> Dict[str, float]:
#         """
#         Aggregate per-microbatch metrics into one update-step metrics dict.

#         During gradient accumulation, each train_step() returns metrics for a
#         single microbatch. Optimizer-step logging should reflect the full
#         accumulation window, not just the final microbatch.
#         """
#         if not metrics_window:
#             raise ValueError("metrics_window must not be empty")

#         reward_values: List[float] = []
#         base_reward_values_all: List[float] = []
#         shaped_reward_values_active: List[float] = []
#         advantage_values: List[float] = []
#         active_advantage_values: List[float] = []
#         kl_values: List[float] = []
#         final_by_prefix: List[Tuple[int, float]] = []
#         required_lcs_fraction_values: List[float] = []
#         format_valid_values: List[float] = []
#         protocol_complete_values: List[float] = []
#         semantic_adjustment_values_active: List[float] = []
#         token_entropy_values: List[float] = []
#         semantic_high_values: List[float] = []
#         semantic_low_values: List[float] = []

#         total_valid = 0
#         total_trajectories = 0
#         total_success = 0
#         total_errors = 0
#         loss_weighted_sum = 0.0
#         policy_logprob_sum = 0.0
#         ref_logprob_sum = 0.0
#         token_entropy_weighted_sum = 0.0
#         total_active_tokens = 0.0
#         max_token_entropy_values = []
#         min_token_entropy_values = []
#         semantic_entropy_values = []
#         expected_zero_gradient = True

#         for metrics in metrics_window:
#             num_valid = int(metrics.get("num_valid", 0))
#             num_total = int(metrics.get("num_total", 0))

#             total_valid += num_valid
#             total_trajectories += num_total
#             total_errors += int(metrics.get("num_errors", 0))

#             loss_weighted_sum += float(metrics.get("_loss_weighted_sum", 0.0))
#             policy_logprob_sum += float(metrics.get("_policy_logprob_sum", 0.0))
#             ref_logprob_sum += float(metrics.get("_ref_logprob_sum", 0.0))
#             token_entropy_weighted_sum += float(metrics.get("_token_entropy_weighted_sum", 0.0))
#             total_active_tokens += float(metrics.get("_active_token_count", 0.0))

#             reward_values.extend(float(x) for x in metrics.get("_reward_values", []))
#             base_reward_values_all.extend(
#                 float(x) for x in metrics.get("_base_reward_values_all", [])
#             )
#             shaped_reward_values_active.extend(
#                 float(x) for x in metrics.get("_shaped_reward_values_active", [])
#             )
#             advantage_values.extend(float(x) for x in metrics.get("_advantage_values", []))
#             active_advantage_values.extend(
#                 float(x) for x in metrics.get("_active_advantage_values", [])
#             )
#             kl_values.extend(float(x) for x in metrics.get("_kl_values", []))
#             final_by_prefix.extend(
#                 (int(prefix), float(value))
#                 for prefix, value in metrics.get("_final_by_prefix", [])
#             )
#             required_lcs_fraction_values.extend(
#                 float(x)
#                 for x in metrics.get("_required_lcs_fraction_values", [])
#             )
#             format_valid_values.extend(
#                 float(x) for x in metrics.get("_format_valid_values", [])
#             )
#             protocol_complete_values.extend(
#                 float(x) for x in metrics.get("_protocol_complete_values", [])
#             )
#             semantic_adjustment_values_active.extend(
#                 float(x)
#                 for x in metrics.get("_semantic_adjustment_values_active", [])
#             )
#             token_entropy_values.extend(
#                 float(x) for x in metrics.get("_token_entropy_values", [])
#             )
#             semantic_high_values.extend(
#                 float(x) for x in metrics.get("_semantic_high_values", [])
#             )
#             semantic_low_values.extend(
#                 float(x) for x in metrics.get("_semantic_low_values", [])
#             )

#             # Backward-compatible fallback for metrics created before raw values.
#             if not metrics.get("_token_entropy_values"):
#                 if metrics.get("num_valid", 0) and "max_token_entropy" in metrics:
#                     max_token_entropy_values.append(float(metrics["max_token_entropy"]))
#                 if metrics.get("num_valid", 0) and "min_token_entropy" in metrics:
#                     min_token_entropy_values.append(float(metrics["min_token_entropy"]))
#             if "semantic_entropy" in metrics:
#                 semantic_entropy_values.append(float(metrics["semantic_entropy"]))
#             if (
#                 int(metrics.get("num_valid", 0)) > 0
#                 and not bool(metrics.get("_expected_zero_gradient", False))
#             ):
#                 expected_zero_gradient = False

#         if not base_reward_values_all:
#             base_reward_values_all = reward_values

#         final_values = [value for _, value in final_by_prefix]
#         final_by_length = {
#             prefix: [value for item_prefix, value in final_by_prefix if item_prefix == prefix]
#             for prefix in (1, 2, 3)
#         }
#         total_success = int(sum(protocol_complete_values))

#         aggregated = {
#             "loss": loss_weighted_sum / max(total_valid, 1),
#             "avg_reward": self._mean_or_nan(base_reward_values_all),
#             "base_reward_all": self._mean_or_nan(base_reward_values_all),
#             "shaped_reward_active": self._mean_or_nan(shaped_reward_values_active),
#             "std_reward": float(np.std(base_reward_values_all)) if base_reward_values_all else float("nan"),
#             "min_reward": float(np.min(base_reward_values_all)) if base_reward_values_all else float("nan"),
#             "max_reward": float(np.max(base_reward_values_all)) if base_reward_values_all else float("nan"),
#             "median_reward": float(np.median(base_reward_values_all)) if base_reward_values_all else float("nan"),
#             "avg_advantage": float(np.mean(advantage_values)) if advantage_values else 0.0,
#             "std_advantage": float(np.std(advantage_values)) if advantage_values else 0.0,
#             "advantage_abs_active": (
#                 float(np.mean(np.abs(active_advantage_values)))
#                 if active_advantage_values
#                 else float("nan")
#             ),
#             "avg_kl": float(np.mean(kl_values)) if kl_values else 0.0,
#             "max_kl": float(np.max(kl_values)) if kl_values else 0.0,
#             "min_kl": float(np.min(kl_values)) if kl_values else 0.0,
#             "num_valid": total_valid,
#             "num_total": total_trajectories,
#             "num_success": total_success,
#             "num_errors": total_errors,
#             "protocol_complete_rate": self._mean_or_nan(protocol_complete_values),
#             "trainable_rate": total_valid / max(total_trajectories, 1),
#             "final_rate": self._mean_or_nan(final_values),
#             "final_rate_p1": self._mean_or_nan(final_by_length[1]),
#             "final_rate_p2": self._mean_or_nan(final_by_length[2]),
#             "final_rate_p3": self._mean_or_nan(final_by_length[3]),
#             "required_lcs_fraction": self._mean_or_nan(required_lcs_fraction_values),
#             "format_valid_rate": self._mean_or_nan(format_valid_values),
#             "semantic_entropy_high": self._mean_or_nan(semantic_high_values),
#             "semantic_entropy_low": self._mean_or_nan(semantic_low_values),
#             "semantic_adjustment_active": self._mean_or_nan(
#                 semantic_adjustment_values_active
#             ),
#             "policy_logprob_mean": policy_logprob_sum / max(total_trajectories, 1),
#             "ref_logprob_mean": ref_logprob_sum / max(total_trajectories, 1),
#             "avg_token_entropy": token_entropy_weighted_sum / max(total_active_tokens, 1.0),
#             "max_token_entropy": (
#                 float(np.max(token_entropy_values))
#                 if token_entropy_values
#                 else (
#                     float(np.max(max_token_entropy_values))
#                     if max_token_entropy_values
#                     else float("nan")
#                 )
#             ),
#             "min_token_entropy": (
#                 float(np.min(token_entropy_values))
#                 if token_entropy_values
#                 else (
#                     float(np.min(min_token_entropy_values))
#                     if min_token_entropy_values
#                     else float("nan")
#                 )
#             ),
#             "semantic_entropy": float(np.mean(semantic_entropy_values)) if semantic_entropy_values else 0.0,
#             "eval_final_p1": float("nan"),
#             "eval_final_p2": float("nan"),
#             "eval_final_p3": float("nan"),
#             "_expected_zero_gradient": expected_zero_gradient,
#         }
#         return aggregated

#     @staticmethod
#     def _format_number(value: Any, digits: int = 2, signed: bool = False) -> str:
#         try:
#             numeric = float(value)
#         except (TypeError, ValueError):
#             return "N/A"
#         if not np.isfinite(numeric):
#             return "N/A"
#         sign = "+" if signed else ""
#         return f"{numeric:{sign}.{digits}f}"

#     @classmethod
#     def _format_percent(cls, value: Any, digits: int = 0) -> str:
#         try:
#             numeric = float(value)
#         except (TypeError, ValueError):
#             return "N/A"
#         if not np.isfinite(numeric):
#             return "N/A"
#         return f"{100.0 * numeric:.{digits}f}%"

#     @classmethod
#     def _format_training_metrics(
#         cls,
#         step: int,
#         metrics: Dict[str, Any],
#     ) -> str:
#         return (
#             f"\nStep {step}\n"
#             f"BaseReward(all)={cls._format_number(metrics.get('base_reward_all'))}  "
#             f"ShapedReward(active)={cls._format_number(metrics.get('shaped_reward_active'))}  "
#             f"BaseEMA={cls._format_number(metrics.get('reward_ema'))}\n"
#             f"Final={cls._format_percent(metrics.get('final_rate'))} "
#             f"[p1={cls._format_percent(metrics.get('final_rate_p1'))}, "
#             f"p2={cls._format_percent(metrics.get('final_rate_p2'))}, "
#             f"p3={cls._format_percent(metrics.get('final_rate_p3'))}]  "
#             f"RequiredLCS={cls._format_percent(metrics.get('required_lcs_fraction'))}\n"
#             f"ProtocolComplete={cls._format_percent(metrics.get('protocol_complete_rate'))}  "
#             f"FormatValid={cls._format_percent(metrics.get('format_valid_rate'))}  "
#             f"Trainable={cls._format_percent(metrics.get('trainable_rate'))}\n"
#             f"SemanticH[high]={cls._format_number(metrics.get('semantic_entropy_high'))}  "
#             f"SemanticH[low]={cls._format_number(metrics.get('semantic_entropy_low'))}  "
#             f"Adjustment={cls._format_number(metrics.get('semantic_adjustment_active'), signed=True)}\n"
#             f"AdvAbs={cls._format_number(metrics.get('advantage_abs_active'))}  "
#             f"TokenH={cls._format_number(metrics.get('avg_token_entropy'))}  "
#             f"GradNorm={cls._format_number(metrics.get('grad_norm'), digits=6)}\n"
#             f"EvalFinal[p1/p2/p3]="
#             f"{cls._format_percent(metrics.get('eval_final_p1'))}/"
#             f"{cls._format_percent(metrics.get('eval_final_p2'))}/"
#             f"{cls._format_percent(metrics.get('eval_final_p3'))}"
#         )

#     def _compute_token_logprobs_chunked(
#         self,
#         logits: torch.Tensor,
#         labels: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Compute per-token log probabilities in chunks to avoid OOM.

#         FIXED: Removed unused loss_mask parameter for clarity.
#         FIXED: Handle CPU tensors by moving chunks to GPU for computation.

#         Args:
#             logits: [batch_size, seq_len, vocab_size] - can be on CPU or GPU
#             labels: [batch_size, seq_len] - should be on GPU

#         Returns:
#             token_logprobs: [batch_size, seq_len] - log prob of each token on GPU
#         """
#         batch_size, seq_len, vocab_size = logits.shape
#         logits_device = logits.device
#         target_device = labels.device  # GPU device for output

#         # Output tensor on GPU
#         token_logprobs = torch.zeros(batch_size, seq_len, device=target_device)

#         # Process in chunks along sequence dimension
#         num_chunks = (seq_len + self.logprob_chunk_size - 1) // self.logprob_chunk_size

#         for chunk_idx in range(num_chunks):
#             start_idx = chunk_idx * self.logprob_chunk_size
#             end_idx = min((chunk_idx + 1) * self.logprob_chunk_size, seq_len)

#             # Extract chunk
#             logits_chunk = logits[:, start_idx:end_idx, :]  # [B, chunk_size, V]
#             labels_chunk = labels[:, start_idx:end_idx]      # [B, chunk_size]

#             # Reference logits may live on CPU or another CUDA device.
#             if logits_chunk.device != target_device:
#                 logits_chunk = logits_chunk.contiguous().to(target_device)
#             else:
#                 logits_chunk = logits_chunk.contiguous()

#             # Compute log_softmax only for this chunk
#             log_probs_chunk = F.log_softmax(logits_chunk, dim=-1)  # [B, chunk_size, V]

#             # Gather log probs for actual tokens
#             labels_expanded = labels_chunk.unsqueeze(-1)  # [B, chunk_size, 1]
#             selected_logprobs = torch.gather(log_probs_chunk, dim=-1, index=labels_expanded)

#             # Store: [B, chunk_size]
#             token_logprobs[:, start_idx:end_idx] = selected_logprobs.squeeze(-1)

#         return token_logprobs

#     def _compute_token_entropies_chunked(self, logits: torch.Tensor) -> torch.Tensor:
#         """
#         Compute per-token entropies in chunks to avoid OOM.

#         Args:
#             logits: [batch_size, seq_len, vocab_size] - typically on GPU for policy model

#         Returns:
#             token_entropies: [batch_size, seq_len]
#         """
#         batch_size, seq_len, _ = logits.shape
#         device = logits.device
#         token_entropies = torch.zeros(batch_size, seq_len, device=device)

#         # Compute entropies without tracking gradients to save memory
#         with torch.no_grad():
#             for start_idx in range(0, seq_len, self.logprob_chunk_size):
#                 end_idx = min(start_idx + self.logprob_chunk_size, seq_len)
#                 chunk_logits = logits[:, start_idx:end_idx, :].detach()
#                 chunk_logprobs = F.log_softmax(chunk_logits, dim=-1)
#                 chunk_probs = chunk_logprobs.exp()
#                 chunk_entropy = -(chunk_probs * chunk_logprobs).sum(dim=-1)
#                 token_entropies[:, start_idx:end_idx] = chunk_entropy

#         return token_entropies

#     def _compute_semantic_entropy(
#         self,
#         trajectories: List[CompletedTrajectory],
#         valid_mask: torch.Tensor
#     ) -> float:
#         """
#         Compute semantic entropy by grouping trajectories by semantic similarity.

#         Semantic entropy measures uncertainty at the meaning level rather than token level.
#         We group trajectories by their outcomes/actions and compute entropy over these semantic clusters.

#         Args:
#             trajectories: List of completed trajectories
#             valid_mask: Boolean mask for valid trajectories

#         Returns:
#             Semantic entropy value (scalar)
#         """
#         if valid_mask.sum() == 0:
#             return 0.0

#         # Group trajectories by their semantic outcomes
#         # We use termination reason + success status as a proxy for semantic grouping
#         semantic_groups = defaultdict(list)

#         for idx, traj in enumerate(trajectories):
#             if not valid_mask[idx]:
#                 continue

#             # Create semantic signature based on:
#             # 1. Termination reason (success, max_turns, error)
#             # 2. Reward bucket (discretize reward into bins for clustering)
#             # 3. Number of turns (trajectory length)
#             reward_bucket = int(traj.reward * 10) / 10 if traj.reward is not None else 0.0
#             semantic_key = (
#                 traj.termination_reason,
#                 reward_bucket,
#                 len(traj.segments)
#             )
#             semantic_groups[semantic_key].append(idx)

#         # Compute probability distribution over semantic clusters
#         total_valid = valid_mask.sum().item()
#         cluster_probs = []

#         for cluster_indices in semantic_groups.values():
#             cluster_prob = len(cluster_indices) / total_valid
#             cluster_probs.append(cluster_prob)

#         # Compute entropy: H = -sum(p * log(p))
#         cluster_probs = np.array(cluster_probs)
#         semantic_entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))

#         return float(semantic_entropy)

#     def _compute_logprobs_and_kl(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: torch.Tensor,
#         loss_mask: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Compute trajectory logprobs and token-level KL divergence.
        
#         FIXED: Proper token-level KL computation.
        
#         Args:
#             input_ids: [batch_size, seq_len]
#             attention_mask: [batch_size, seq_len]
#             loss_mask: [batch_size, seq_len] - 1 for trainable tokens
            
#         Returns:
#             policy_traj_logprobs: [batch_size] - length-normalized policy logprobs
#             ref_traj_logprobs: [batch_size] - length-normalized ref logprobs
#             kl_per_traj: [batch_size] - token-level KL averaged per trajectory
#             policy_token_logprobs: [batch_size, seq_len-1] - for debugging
#             policy_token_entropies: [batch_size, seq_len-1] - token entropy per position
#         """
#         # ============================================================
#         # Step 1: Get logits from both models
#         # ============================================================
        
#         # Policy model (with gradients)
#         outputs_policy = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             use_cache=False
#         )
#         logits_policy = outputs_policy.logits  # [B, S, V]

#         # Reference model (frozen) - handle CPU offloading
#         with torch.no_grad():
#             # Detect if ref_model is on CPU
#             ref_device = next(self.ref_model.parameters()).device

#             if ref_device.type == "cpu":
#                 # Move inputs to CPU for reference model
#                 input_ids_ref = input_ids.cpu()
#                 attention_mask_ref = attention_mask.cpu()
#             else:
#                 input_ids_ref = input_ids
#                 attention_mask_ref = attention_mask

#             outputs_ref = self.ref_model(
#                 input_ids=input_ids_ref,
#                 attention_mask=attention_mask_ref,
#                 use_cache=False
#             )
#             logits_ref = outputs_ref.logits  # [B, S, V]

#             # Keep logits_ref on CPU if ref_model is on CPU for memory efficiency
#             # Will be moved to GPU in chunks during computation
        
#         # ============================================================
#         # Step 2: Shift for next-token prediction
#         # ============================================================
        
#         shift_logits_policy = logits_policy[:, :-1, :]  # [B, S-1, V] on GPU
#         # Keep shift_logits_ref on CPU if logits_ref is on CPU (don't trigger GPU allocation)
#         if logits_ref.device.type == "cpu":
#             shift_logits_ref = logits_ref[:, :-1, :]  # View on CPU, no .contiguous() yet
#         else:
#             shift_logits_ref = logits_ref[:, :-1, :].contiguous()  # [B, S-1, V] on GPU
#         shift_labels = input_ids[:, 1:].contiguous()                 # [B, S-1] on GPU
#         shift_loss_mask = loss_mask[:, 1:].contiguous()              # [B, S-1] on GPU
        
#         # ============================================================
#         # Step 3: Compute per-token log probabilities (chunked)
#         # ============================================================
        
#         policy_token_logprobs = self._compute_token_logprobs_chunked(
#             shift_logits_policy, shift_labels
#         )  # [B, S-1]
#         # policy_token_entropies = self._compute_token_entropies_chunked(
#         #     shift_logits_policy
#         # )  # [B, S-1]
#         policy_token_entropies = torch.zeros_like(policy_token_logprobs)
#         with torch.no_grad():
#             ref_token_logprobs = self._compute_token_logprobs_chunked(
#                 shift_logits_ref, shift_labels
#             )  # [B, S-1]
        
#         # ============================================================
#         # Step 4: Compute trajectory-level logprobs (length-normalized)
#         # ============================================================
        
#         # Apply mask to get only trainable tokens
#         masked_policy_logprobs = policy_token_logprobs * shift_loss_mask  # [B, S-1]
#         masked_ref_logprobs = ref_token_logprobs * shift_loss_mask        # [B, S-1]
        
#         # Normalize by number of trainable tokens
#         num_trainable = shift_loss_mask.sum(dim=1).clamp(min=1)  # [B]
        
#         policy_traj_logprobs = masked_policy_logprobs.sum(dim=1) / num_trainable  # [B]
#         ref_traj_logprobs = masked_ref_logprobs.sum(dim=1) / num_trainable        # [B]
        
#         # ============================================================
#         # Step 5: Compute token-level KL divergence
#         # ============================================================
        
#         # KL(π || π_ref) ≈ log π(x) - log π_ref(x)
#         # For discrete distributions, this is exact
#         token_kl = (policy_token_logprobs - ref_token_logprobs) * shift_loss_mask  # [B, S-1]
        
#         # Average KL over trainable tokens per trajectory
#         kl_per_traj = token_kl.sum(dim=1) / num_trainable  # [B]
        
#         return (
#             policy_traj_logprobs,
#             ref_traj_logprobs,
#             kl_per_traj,
#             policy_token_logprobs,
#             policy_token_entropies,
#             shift_loss_mask,
#         )

#     def train_step(
#         self,
#         batch_trajectories: Dict[str, List[CompletedTrajectory]],
#         collated_batch: Dict[str, torch.Tensor],
#         loss_scale: float = 1.0
#     ) -> Dict[str, float]:
#         """
#         Perform one GRPO training step.
        
#         FIXED: Token-level KL computation for proper GRPO objective
        
#         Args:
#             batch_trajectories: Dict mapping query_id -> list of G trajectories
#             collated_batch: Collated batch from TrajectoryCollator
            
#         Returns:
#             Dictionary of training metrics
#         """
#         # Extract collated data
#         input_ids = collated_batch['input_ids'].to(self.device)
#         attention_mask = collated_batch['attention_mask'].to(self.device)
#         loss_mask = collated_batch['loss_mask'].to(self.device)
        
#         batch_size = input_ids.shape[0]
        
#         # ============================================================
#         # Step 1: Flatten trajectories and assign rewards
#         # ============================================================
        
#         all_trajectories = []
#         for query_id, trajs in batch_trajectories.items():
#             for traj in trajs:
#                 all_trajectories.append(traj)
        
#         assert len(all_trajectories) == batch_size, \
#             f"Mismatch: {len(all_trajectories)} trajectories vs {batch_size} collated"
        
#         # Validate trajectories first (before expensive reward computation)
#         valid_mask = []
#         for traj in all_trajectories:
#             is_valid, error_msg = traj.validate_structure()
#             if not is_valid:
#                 logger.warning(f"Invalid trajectory {traj.query_id}: {error_msg}")
#             valid_mask.append(is_valid)

#         # Compute rewards for ALL trajectories (valid and invalid)
#         for i, traj in enumerate(all_trajectories):
#             if traj.reward is None:  # ← REMOVED valid_mask check
#                 # Extract base task_id (remove _g suffix)
#                 task_id = traj.query_id.rsplit('_g', 1)[0] if '_g' in traj.query_id else traj.query_id
#                 try:
#                     traj.reward = self.reward_function(task_id, traj)
#                 except Exception as e:
#                     logger.error(f"Reward computation failed for {task_id}: {e}")
#                     # raise
            
#             # ADDITIONAL CHECK: Ensure reward is never None
#             if traj.reward is None:
#                 logger.error(f"Trajectory {traj.query_id} has None reward after computation!")
#                 traj.reward = 0.0
#                 valid_mask[i] = False
#         # ========== ADD REWARD NORMALIZATION HERE ==========
        
#         # ===================================================
#         valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)

#         if valid_mask.sum() == 0:
#             logger.error("No valid trajectories in batch!")
#             quality_metrics = self._build_quality_metric_payload(
#                 all_trajectories,
#                 valid_mask,
#             )
#             return {
#                 'loss': 0.0,
#                 'avg_reward': float(np.mean([traj.reward or 0.0 for traj in all_trajectories])),
#                 'std_reward': float(np.std([traj.reward or 0.0 for traj in all_trajectories])),
#                 'avg_advantage': 0.0,
#                 'avg_kl': 0.0,
#                 'avg_token_entropy': 0.0,
#                 'max_token_entropy': 0.0,
#                 'min_token_entropy': 0.0,
#                 'semantic_entropy': 0.0,
#                 'grad_norm': 0.0,
#                 'num_valid': 0,
#                 'num_total': batch_size,
#                 'num_success': int(sum(quality_metrics["_protocol_complete_values"])),
#                 'num_errors': sum(
#                     traj.termination_reason != "success"
#                     for traj in all_trajectories
#                 ),
#                 '_loss_weighted_sum': 0.0,
#                 '_reward_values': [float(traj.reward or 0.0) for traj in all_trajectories],
#                 '_advantage_values': [0.0 for _ in all_trajectories],
#                 '_kl_values': [],
#                 '_policy_logprob_sum': 0.0,
#                 '_ref_logprob_sum': 0.0,
#                 '_token_entropy_weighted_sum': 0.0,
#                 '_active_token_count': 0.0,
#                 '_token_entropy_values': [],
#                 **quality_metrics,
#             }
        
#         # ============================================================
#         # Step 2: Compute group-relative advantages
#         # ============================================================
        
#         # Group trajectories by base query
#         groups = {}
#         for traj in all_trajectories:
#             base_id = traj.query_id.rsplit('_g', 1)[0] if '_g' in traj.query_id else traj.query_id
#             if base_id not in groups:
#                 groups[base_id] = []
#             groups[base_id].append(traj)
        
#         # Compute advantages within each group
#         for base_id, group_trajs in groups.items():
#             group_rewards = [t.reward for t in group_trajs]
#             # ← ADD THIS SAFETY CHECK
#             if any(r is None for r in group_rewards):
#                 logger.error(f"Group {base_id} has None rewards: {group_rewards}")
#                 # Replace None with 0.0
#                 group_rewards = [r if r is not None else 0.0 for r in group_rewards]
#             mean_reward = np.mean(group_rewards)
#             std_reward = np.std(group_rewards) if len(group_rewards) > 1 else 1.0
            
#             # Normalize advantages
#             for traj in group_trajs:
#                 traj.advantage = (traj.reward - mean_reward) / (std_reward + 1e-8)
        
#         # ============================================================
#         # Step 3: Compute logprobs and KL
#         # ============================================================
        
#         policy_traj_logprobs, ref_traj_logprobs, kl_per_traj, policy_token_logprobs, policy_token_entropies, shift_loss_mask = \
#             self._compute_logprobs_and_kl(input_ids, attention_mask, loss_mask)
        
#         # ============================================================
#         # Step 4: Compute GRPO loss
#         # ============================================================
        
#         # Extract advantages as tensor
#         advantages = torch.tensor(
#             [traj.advantage for traj in all_trajectories],
#             dtype=torch.float32,
#             device=self.device
#         )  # [B]
#         # advantages = advantages.detach()  # <--- ADD THIS
#         # GRPO objective: maximize E[A * log π(τ) - β * KL(π || π_ref)]
#         # Loss = -E[A * log π(τ) - β * KL]
#         loss_per_traj = -(advantages * policy_traj_logprobs - self.beta * kl_per_traj)  # [B]
        
#         # Apply valid mask
#         valid_mask_float = valid_mask.float()
#         loss_per_traj = loss_per_traj * valid_mask_float
        
#         # Average loss over valid trajectories
#         loss = loss_per_traj.sum() / valid_mask.sum()
        
#         # ============================================================
#         # Step 5: Backward pass and optimization
#         # ============================================================
#         # Store unscaled loss for logging
#         unscaled_loss = loss.item()  # <--- ADD THIS
#         loss = loss/loss_scale
#         # self.optimizer.zero_grad()
#         loss.backward()
        
#         # Gradient clipping
#         # grad_norm = torch.nn.utils.clip_grad_norm_(
#         #     self.model.parameters(),
#         #     self.max_grad_norm
#         # )
        
#         # self.optimizer.step()
        
#         # ============================================================
#         # Step 6: Logging metrics
#         # ============================================================
        
#         with torch.no_grad():
#             rewards = [traj.reward for traj in all_trajectories]
#             advantages_np = [traj.advantage for traj in all_trajectories]
#             active_token_mask = shift_loss_mask * valid_mask_float.unsqueeze(1)
#             num_active_tokens = active_token_mask.sum().clamp(min=1.0)
#             token_entropy_sum = (policy_token_entropies * active_token_mask).sum()
#             token_entropy_per_traj = (
#                 (policy_token_entropies * active_token_mask).sum(dim=1)
#                 / active_token_mask.sum(dim=1).clamp(min=1.0)
#             )
#             valid_token_entropy_values = token_entropy_per_traj[valid_mask].detach().cpu().tolist()

#             # Compute semantic entropy
#             semantic_entropy = self._compute_semantic_entropy(all_trajectories, valid_mask)
#             quality_metrics = self._build_quality_metric_payload(
#                 all_trajectories,
#                 valid_mask,
#             )

#             # Termination reason statistics
#             termination_reasons = {}
#             for traj in all_trajectories:
#                 reason = traj.termination_reason
#                 termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
            
#             metrics = {
#                 'loss': unscaled_loss,
#                 'avg_reward': np.mean(rewards),
#                 'std_reward': np.std(rewards),
#                 'min_reward': np.min(rewards),
#                 'max_reward': np.max(rewards),
#                 'median_reward': np.median(rewards),  # <--- ADD THIS LINE
#                 'avg_advantage': np.mean(advantages_np),
#                 'std_advantage': np.std(advantages_np),
#                 'avg_kl': kl_per_traj.mean().item(),
#                 'max_kl': kl_per_traj.max().item(),
#                 'min_kl': kl_per_traj.min().item(),
#                 'avg_token_entropy': float(token_entropy_sum.item() / num_active_tokens.item()),
#                 'max_token_entropy': float(max(valid_token_entropy_values)) if valid_token_entropy_values else 0.0,
#                 'min_token_entropy': float(min(valid_token_entropy_values)) if valid_token_entropy_values else 0.0,
#                 'semantic_entropy': semantic_entropy,
#                 # 'grad_norm': grad_norm.item(),
#                 'num_valid': valid_mask.sum().item(),
#                 'num_total': batch_size,
#                 'policy_logprob_mean': policy_traj_logprobs.mean().item(),
#                 'ref_logprob_mean': ref_traj_logprobs.mean().item(),
#                 'num_success': int(sum(quality_metrics["_protocol_complete_values"])),
#                 'num_max_turns': termination_reasons.get('max_turns', 0),
#                 'num_errors': sum(
#                     count
#                     for reason, count in termination_reasons.items()
#                     if reason != "success"
#                 ),
#                 '_loss_weighted_sum': unscaled_loss * valid_mask.sum().item(),
#                 '_reward_values': list(rewards),
#                 '_advantage_values': list(advantages_np),
#                 '_kl_values': kl_per_traj.detach().cpu().tolist(),
#                 '_policy_logprob_sum': policy_traj_logprobs.sum().item(),
#                 '_ref_logprob_sum': ref_traj_logprobs.sum().item(),
#                 '_token_entropy_weighted_sum': float(token_entropy_sum.item()),
#                 '_active_token_count': float(num_active_tokens.item()),
#                 '_token_entropy_values': valid_token_entropy_values,
#                 **quality_metrics,
#             }

#         return metrics

#     def train(
#         self,
#         rollout_manager: Any,
#         train_queries: List[Dict],
#         num_epochs: int = 1,
#         group_size: int = 4,
#         batch_size: int = 1,
#         gradient_accumulation_steps: int = 1,
#         collator: Any = None,
#         checkpoint_dir: Optional[str] = None,
#         checkpoint_every: int = 100,
#         log_callback: Optional[Callable] = None,
#         resume_state: Optional[Dict[str, Any]] = None,
#         evaluation_callback: Optional[Callable[[int], Dict[str, float]]] = None,
#     ) -> List[Dict]:
#         """
#         Full training loop.
        
#         Args:
#             rollout_manager: AgenticRolloutManager instance
#             train_queries: List of training queries (dicts with 'id' and 'user' keys)
#             num_epochs: Number of training epochs
#             group_size: Number of trajectories per query (G)
#             batch_size: Number of queries per batch
#             collator: TrajectoryCollator instance
#             checkpoint_dir: Directory to save checkpoints (optional)
#             checkpoint_every: Save checkpoint every N steps
#             log_callback: Optional callback for logging (e.g., wandb)
#             evaluation_callback: Optional fixed held-out evaluator. It must return
#                 eval_final_p1, eval_final_p2, and eval_final_p3 rates.
            
#         Returns:
#             List of all metrics dicts from training
#         """
#         logger.info(f"Starting GRPO training: {num_epochs} epochs, {len(train_queries)} queries")
#         logger.info(f"Group size: {group_size}, Batch size: {batch_size}")
#         logger.info(f"Beta (KL coeff): {self.beta}, Max grad norm: {self.max_grad_norm}")
#         if evaluation_callback is None:
#             logger.info(
#                 "No held-out evaluation callback configured; EvalFinal will be N/A"
#             )
        
#         # Create checkpoint directory if specified
#         if checkpoint_dir:
#             os.makedirs(checkpoint_dir, exist_ok=True)
#             logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
        
#         train_queries_by_id = {query["id"]: query for query in train_queries}
#         global_step = int(resume_state.get("global_step", 0)) if resume_state else 0
#         update_step = int(resume_state.get("update_step", 0)) if resume_state else 0
#         all_metrics = list(resume_state.get("all_metrics", [])) if resume_state else []
#         accumulation_metrics: List[Dict[str, Any]] = []
#         start_epoch = int(resume_state.get("epoch", 0)) if resume_state else 0
#         resume_batch_idx = int(resume_state.get("next_batch_idx", 0)) if resume_state else 0
#         resume_shuffled_ids = list(resume_state.get("shuffled_query_ids", [])) if resume_state else []
#         if resume_state:
#             self.reward_ema = resume_state.get("reward_ema")
#             self._restore_rng_state(resume_state.get("rng_state"))
#             logger.info(
#                 "Resuming training from epoch=%d, batch=%d, global_step=%d, update_step=%d",
#                 start_epoch + 1,
#                 resume_batch_idx + 1,
#                 global_step,
#                 update_step,
#             )
#         self.optimizer.zero_grad()
#         for epoch in range(start_epoch, num_epochs):
#             logger.info(f"\n{'='*80}")
#             logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
#             logger.info(f"{'='*80}\n")
            
#             # Shuffle queries
#             if epoch == start_epoch and resume_shuffled_ids:
#                 queries_shuffled = [
#                     train_queries_by_id[qid] for qid in resume_shuffled_ids
#                     if qid in train_queries_by_id
#                 ]
#                 if len(queries_shuffled) != len(train_queries):
#                     missing_ids = [
#                         query["id"] for query in train_queries
#                         if query["id"] not in {q["id"] for q in queries_shuffled}
#                     ]
#                     queries_shuffled.extend(train_queries_by_id[qid] for qid in missing_ids)
#                 batch_start_idx = resume_batch_idx
#             else:
#                 queries_shuffled = train_queries.copy()
#                 random.shuffle(queries_shuffled)
#                 batch_start_idx = 0
            
#             # Process in batches
#             num_batches = (len(queries_shuffled) + batch_size - 1) // batch_size
            
#             for batch_idx in tqdm(range(batch_start_idx, num_batches), desc=f"Epoch {epoch+1}"):
#                 start_idx = batch_idx * batch_size
#                 end_idx = min((batch_idx + 1) * batch_size, len(queries_shuffled))
#                 batch_queries = queries_shuffled[start_idx:end_idx]
                
#                 # Generate trajectories
#                 logger.info(f"\nBatch {batch_idx + 1}/{num_batches}: Generating trajectories...")
#                 batch_trajectories = rollout_manager.generate_batch_trajectories(
#                     queries=batch_queries,
#                     group_size=group_size
#                 )
                
#                 # Collate trajectories
#                 all_trajs = []
#                 for query_id, trajs in batch_trajectories.items():
#                     all_trajs.extend(trajs)
                
#                 if collator is None:
#                     logger.error("No collator provided!")
#                     continue
                
#                 if getattr(self, "requires_agentflow_collation", False):
#                     collated_batch = collator.collate_agentflow(all_trajs)
#                 else:
#                     collated_batch = collator.collate(all_trajs)
                
#                 # Training step
#                 metrics = self.train_step(batch_trajectories, collated_batch, gradient_accumulation_steps)
#                 accumulation_metrics.append(metrics)
                
#                 # 4. Optimizer Step (only every N batches)
#                 if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
#                     aggregated_metrics = self._aggregate_accumulated_metrics(accumulation_metrics)

#                     trainable_gradients = [
#                         (name, parameter)
#                         for name, parameter in self.model.named_parameters()
#                         if parameter.requires_grad and parameter.grad is not None
#                     ]
#                     expected_zero_gradient = bool(
#                         aggregated_metrics.get("_expected_zero_gradient", False)
#                     )
#                     if (
#                         aggregated_metrics.get("num_valid", 0) > 0
#                         and not trainable_gradients
#                         and not expected_zero_gradient
#                     ):
#                         raise RuntimeError(
#                             "Valid trajectories produced no trainable parameter "
#                             "gradients. Check LoRA target modules before continuing; "
#                             "the optimizer would otherwise perform a silent no-op."
#                         )
                    
#                     # Gradient clipping
#                     grad_norm = torch.nn.utils.clip_grad_norm_(
#                         self.model.parameters(), self.max_grad_norm
#                     )
#                     aggregated_metrics['grad_norm'] = grad_norm.item()
#                     if trainable_gradients and not torch.isfinite(grad_norm):
#                         raise RuntimeError(
#                             f"Non-finite gradient norm before optimizer step: {grad_norm.item()}"
#                         )
#                     if (
#                         aggregated_metrics.get("num_valid", 0) > 0
#                         and grad_norm.item() <= 0.0
#                         and not expected_zero_gradient
#                     ):
#                         raise RuntimeError(
#                             "Valid trajectories produced a zero gradient norm. "
#                             "Stopping before a silent no-op optimizer step."
#                         )
                    
#                     if expected_zero_gradient and grad_norm.item() <= 0.0:
#                         logger.warning(
#                             "Skipping optimizer step because all accumulated "
#                             "AgentFlow groups have zero advantage and zero KL signal"
#                         )
#                     else:
#                         self.optimizer.step()
#                     self.optimizer.zero_grad()
#                     update_step += 1
                    
#                     # Update exponential moving average of rewards
#                     current_reward = aggregated_metrics['avg_reward']
#                     if self.reward_ema is None:
#                         self.reward_ema = current_reward
#                     else:
#                         self.reward_ema = (
#                             self.reward_ema_alpha * current_reward + 
#                             (1 - self.reward_ema_alpha) * self.reward_ema
#                         )
#                     aggregated_metrics['reward_ema'] = self.reward_ema
#                     # =========================================
#                     # Log only on update steps
#                     global_step += 1 # Or use update_step

#                     if evaluation_callback is not None:
#                         try:
#                             evaluation_metrics = evaluation_callback(global_step) or {}
#                             for prefix in (1, 2, 3):
#                                 key = f"eval_final_p{prefix}"
#                                 if key in evaluation_metrics:
#                                     aggregated_metrics[key] = float(
#                                         evaluation_metrics[key]
#                                     )
#                         except Exception as exc:
#                             logger.warning(
#                                 "Held-out evaluation failed at step %d: %s",
#                                 global_step,
#                                 exc,
#                             )
                    
#                     # Add standard logging fields
#                     aggregated_metrics['epoch'] = epoch + 1
#                     aggregated_metrics['global_step'] = global_step
#                     all_metrics.append(aggregated_metrics)
                    
#                     logger.info(
#                         self._format_training_metrics(
#                             global_step,
#                             aggregated_metrics,
#                         )
#                     )
                    
#                     if log_callback:
#                         log_callback(aggregated_metrics, global_step)
                    
#                     # # Checkpointing
#                     if checkpoint_dir and global_step % checkpoint_every == 0:
#                         checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}")
#                         logger.info(f"Saving checkpoint to {checkpoint_path}")
#                         self._save_checkpoint(
#                             checkpoint_path=checkpoint_path,
#                             epoch=epoch,
#                             next_batch_idx=batch_idx + 1,
#                             global_step=global_step,
#                             update_step=update_step,
#                             num_epochs=num_epochs,
#                             batch_size=batch_size,
#                             group_size=group_size,
#                             gradient_accumulation_steps=gradient_accumulation_steps,
#                             shuffled_query_ids=[query["id"] for query in queries_shuffled],
#                             all_metrics=all_metrics,
#                         )
#                     accumulation_metrics = []
#             resume_batch_idx = 0
#             resume_shuffled_ids = []

#             if checkpoint_dir and global_step % checkpoint_every == 0:
#                 checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}")
#                 logger.info(f"Saving checkpoint to {checkpoint_path}")
#                 self._save_checkpoint(
#                     checkpoint_path=checkpoint_path,
#                     epoch=epoch,
#                     next_batch_idx=num_batches,
#                     global_step=global_step,
#                     update_step=update_step,
#                     num_epochs=num_epochs,
#                     batch_size=batch_size,
#                     group_size=group_size,
#                     gradient_accumulation_steps=gradient_accumulation_steps,
#                     shuffled_query_ids=[query["id"] for query in queries_shuffled],
#                     all_metrics=all_metrics,
#                 )
        
#         # Final checkpoint
#         if checkpoint_dir:
#             final_path = os.path.join(checkpoint_dir, "final_model")
#             logger.info(f"\nTraining complete! Saving final model to {final_path}")
#             self._save_checkpoint(
#                 checkpoint_path=final_path,
#                 epoch=num_epochs,
#                 next_batch_idx=0,
#                 global_step=global_step,
#                 update_step=update_step,
#                 num_epochs=num_epochs,
#                 batch_size=batch_size,
#                 group_size=group_size,
#                 gradient_accumulation_steps=gradient_accumulation_steps,
#                 shuffled_query_ids=[],
#                 all_metrics=all_metrics,
#             )
        
#         # Finish wandb if used
#         # if self.use_wandb:
#         #     try:
#         #         import wandb
#         #         wandb.finish()
#         #     except ImportError:
#         #         logger.warning("wandb not available but use_wandb=True")
        
#         logger.info(f"\n{'='*80}")
#         logger.info(f"TRAINING COMPLETE")
#         logger.info(f"Total steps: {global_step}")
#         logger.info(f"Final avg reward: {all_metrics[-1]['avg_reward']:.3f}")
#         logger.info(f"{'='*80}\n")
        
#         return all_metrics


"""
GRPO Trainer for Agentic RL - PRODUCTION READY

FIXES APPLIED:
✅ Token-level KL divergence (not trajectory-level heuristic)
✅ Memory-efficient chunked logprob computation
✅ Length-normalized trajectory logprobs for policy gradient
✅ Proper mask shift alignment for causal LM
✅ Identity check for reference model (not equality)
✅ All imports and dependencies resolved
✅ Proper checkpoint handling
✅ Compatible with TrajectoryCollator segment-based masking
"""

import os
import random
import re
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from data_structures import CompletedTrajectory

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer with proper token-level KL.
    
    GRPO objective:
    L = -E[A(τ) * log π(τ) - β * KL(π || π_ref)]
    
    where:
    - A(τ) is group-relative advantage
    - log π(τ) is length-normalized trajectory log probability
    - KL is computed at token level and averaged over trainable tokens
    """

    def __init__(
        self,
        model: Any,
        ref_model: Any,
        optimizer: Any,
        tokenizer: Any,
        reward_function: Callable,
        beta: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
        logprob_chunk_size: int = 128,
        training_microbatch_size: int = 1,
        compute_token_entropy: bool = False,
        use_wandb: bool = False,
    ):
        """Initialize the trainer.

        Args:
            model: Trainable policy model.
            ref_model: Frozen reference model. It may be the policy itself when
                ``beta == 0``; in that case the policy is not frozen.
            optimizer: Optimizer over trainable policy parameters.
            tokenizer: Tokenizer used by checkpointing and trajectory collation.
            reward_function: Callable ``(task_id, trajectory) -> reward``.
            beta: Sampled token-level KL penalty coefficient.
            max_grad_norm: Gradient clipping norm.
            device: Device on which collated inputs and losses are assembled.
            logprob_chunk_size: Number of sequence positions projected through
                the LM head at once. Full ``[B, S, V]`` logits are never built.
            training_microbatch_size: Number of trajectories forwarded at once
                after advantages have been computed over the complete GRPO group.
            compute_token_entropy: Whether to compute full-vocabulary token
                entropy for logging. Disabled by default because it is expensive.
            use_wandb: Whether Weights & Biases logging is enabled externally.
        """
        if int(logprob_chunk_size) <= 0:
            raise ValueError("logprob_chunk_size must be positive")
        if int(training_microbatch_size) <= 0:
            raise ValueError("training_microbatch_size must be positive")

        self.model = model
        self.ref_model = ref_model if ref_model is not None else model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.reward_function = reward_function
        self.beta = float(beta)
        self.max_grad_norm = float(max_grad_norm)
        self.device = device
        self.logprob_chunk_size = int(logprob_chunk_size)
        self.training_microbatch_size = int(training_microbatch_size)
        self.compute_token_entropy = bool(compute_token_entropy)
        self.use_wandb = use_wandb

        self.model.train()
        if self.ref_model is not self.model:
            self.ref_model.eval()
            for parameter in self.ref_model.parameters():
                parameter.requires_grad = False
        elif self.beta != 0.0:
            logger.warning(
                "ref_model is the policy model while beta is non-zero; sampled KL "
                "will be identically zero"
            )

        self.reward_ema = None
        self.reward_ema_alpha = 0.1

        logger.info(
            "GRPOTrainer initialized: beta=%.4f, max_grad_norm=%.3f, "
            "training_microbatch_size=%d, logprob_chunk_size=%d, token_entropy=%s",
            self.beta,
            self.max_grad_norm,
            self.training_microbatch_size,
            self.logprob_chunk_size,
            self.compute_token_entropy,
        )

    def _capture_rng_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_random_state_all"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, rng_state: Optional[Dict[str, Any]]) -> None:
        if not rng_state:
            return
        if "python_random_state" in rng_state:
            random.setstate(rng_state["python_random_state"])
        if "numpy_random_state" in rng_state:
            np.random.set_state(rng_state["numpy_random_state"])
        if "torch_random_state" in rng_state:
            torch.random.set_rng_state(rng_state["torch_random_state"])
        if torch.cuda.is_available() and "cuda_random_state_all" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda_random_state_all"])

    def _save_checkpoint(
        self,
        checkpoint_path: str,
        epoch: int,
        next_batch_idx: int,
        global_step: int,
        update_step: int,
        num_epochs: int,
        batch_size: int,
        group_size: int,
        gradient_accumulation_steps: int,
        shuffled_query_ids: List[str],
        all_metrics: List[Dict[str, Any]],
    ) -> None:
        os.makedirs(checkpoint_path, exist_ok=True)
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        trainer_state = {
            "epoch": epoch,
            "next_batch_idx": next_batch_idx,
            "global_step": global_step,
            "update_step": update_step,
            "reward_ema": self.reward_ema,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "group_size": group_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "shuffled_query_ids": shuffled_query_ids,
            "all_metrics": all_metrics,
            "rng_state": self._capture_rng_state(),
        }
        torch.save(trainer_state, os.path.join(checkpoint_path, "trainer_state.pt"))

    @staticmethod
    def _trajectory_base_id(query_id: str) -> str:
        return query_id.rsplit("_g", 1)[0] if "_g" in query_id else query_id

    @staticmethod
    def _breakdown_value(trajectory: CompletedTrajectory, name: str) -> Any:
        breakdown = getattr(trajectory, "reward_breakdown", None)
        if isinstance(breakdown, dict):
            return breakdown.get(name)
        return getattr(breakdown, name, None) if breakdown is not None else None

    @classmethod
    def _trajectory_prefix_length(cls, trajectory: CompletedTrajectory) -> Optional[int]:
        gold_length = cls._breakdown_value(trajectory, "gold_sequence_length")
        if gold_length is not None:
            try:
                parsed_length = int(gold_length)
                if parsed_length > 0:
                    return parsed_length
            except (TypeError, ValueError):
                pass

        match = re.search(r"_p(\d+)(?:_g\d+)?$", trajectory.query_id)
        return int(match.group(1)) if match else None

    @classmethod
    def _build_quality_metric_payload(
        cls,
        trajectories: List[CompletedTrajectory],
        valid_mask: Any,
    ) -> Dict[str, Any]:
        """Return raw values so accumulation can aggregate without ratio bias."""
        if hasattr(valid_mask, "detach"):
            valid_values = [bool(value) for value in valid_mask.detach().cpu().tolist()]
        else:
            valid_values = [bool(value) for value in valid_mask]

        base_rewards_all: List[float] = []
        shaped_rewards_active: List[float] = []
        final_by_prefix: List[Tuple[int, float]] = []
        required_lcs_fractions: List[float] = []
        format_valid_values: List[float] = []
        protocol_complete_values: List[float] = []
        semantic_adjustments_active: List[float] = []
        active_advantages: List[float] = []

        semantic_high_by_group: Dict[str, float] = {}
        semantic_low_by_group: Dict[str, float] = {}

        for is_valid, trajectory in zip(valid_values, trajectories):
            base_reward = float(
                getattr(
                    trajectory,
                    "base_reward",
                    trajectory.reward if trajectory.reward is not None else 0.0,
                )
            )
            base_rewards_all.append(base_reward)
            protocol_complete_values.append(
                1.0 if trajectory.termination_reason == "success" else 0.0
            )

            prefix_length = cls._trajectory_prefix_length(trajectory)
            final_reward = cls._breakdown_value(trajectory, "final_reward")
            if prefix_length is not None and final_reward is not None:
                final_by_prefix.append((prefix_length, float(final_reward) > 0.0))

            required_lcs = cls._breakdown_value(trajectory, "required_lcs_length")
            gold_length = cls._breakdown_value(trajectory, "gold_sequence_length")
            if required_lcs is not None and gold_length:
                required_lcs_fractions.append(
                    float(required_lcs) / max(float(gold_length), 1.0)
                )

            format_valid = cls._breakdown_value(trajectory, "format_valid")
            if format_valid is not None:
                format_valid_values.append(1.0 if bool(format_valid) else 0.0)

            group_id = cls._trajectory_base_id(trajectory.query_id)
            semantic_entropy = getattr(
                trajectory, "semantic_group_logging_entropy", None
            )
            semantic_high_reward = getattr(
                trajectory, "semantic_group_high_reward", None
            )
            if semantic_entropy is not None and semantic_high_reward is not None:
                target = (
                    semantic_high_by_group
                    if bool(semantic_high_reward)
                    else semantic_low_by_group
                )
                target[group_id] = float(semantic_entropy)

            if not is_valid:
                continue

            shaped_rewards_active.append(
                float(getattr(trajectory, "semantic_shaped_reward", base_reward))
            )
            semantic_adjustments_active.append(
                float(getattr(trajectory, "semantic_reward_adjustment", 0.0))
            )
            active_advantages.append(float(trajectory.advantage or 0.0))

        return {
            "_base_reward_values_all": base_rewards_all,
            "_shaped_reward_values_active": shaped_rewards_active,
            "_final_by_prefix": final_by_prefix,
            "_required_lcs_fraction_values": required_lcs_fractions,
            "_format_valid_values": format_valid_values,
            "_protocol_complete_values": protocol_complete_values,
            "_semantic_adjustment_values_active": semantic_adjustments_active,
            "_active_advantage_values": active_advantages,
            "_semantic_high_values": list(semantic_high_by_group.values()),
            "_semantic_low_values": list(semantic_low_by_group.values()),
        }

    @staticmethod
    def _mean_or_nan(values: List[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    def _aggregate_accumulated_metrics(
        self,
        metrics_window: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Aggregate per-microbatch metrics into one update-step metrics dict.

        During gradient accumulation, each train_step() returns metrics for a
        single microbatch. Optimizer-step logging should reflect the full
        accumulation window, not just the final microbatch.
        """
        if not metrics_window:
            raise ValueError("metrics_window must not be empty")

        reward_values: List[float] = []
        base_reward_values_all: List[float] = []
        shaped_reward_values_active: List[float] = []
        advantage_values: List[float] = []
        active_advantage_values: List[float] = []
        kl_values: List[float] = []
        final_by_prefix: List[Tuple[int, float]] = []
        required_lcs_fraction_values: List[float] = []
        format_valid_values: List[float] = []
        protocol_complete_values: List[float] = []
        semantic_adjustment_values_active: List[float] = []
        token_entropy_values: List[float] = []
        semantic_high_values: List[float] = []
        semantic_low_values: List[float] = []

        total_valid = 0
        total_trajectories = 0
        total_success = 0
        total_errors = 0
        loss_weighted_sum = 0.0
        policy_logprob_sum = 0.0
        ref_logprob_sum = 0.0
        token_entropy_weighted_sum = 0.0
        total_active_tokens = 0.0
        max_token_entropy_values = []
        min_token_entropy_values = []
        semantic_entropy_values = []
        expected_zero_gradient = True

        for metrics in metrics_window:
            num_valid = int(metrics.get("num_valid", 0))
            num_total = int(metrics.get("num_total", 0))

            total_valid += num_valid
            total_trajectories += num_total
            total_errors += int(metrics.get("num_errors", 0))

            loss_weighted_sum += float(metrics.get("_loss_weighted_sum", 0.0))
            policy_logprob_sum += float(metrics.get("_policy_logprob_sum", 0.0))
            ref_logprob_sum += float(metrics.get("_ref_logprob_sum", 0.0))
            token_entropy_weighted_sum += float(metrics.get("_token_entropy_weighted_sum", 0.0))
            total_active_tokens += float(metrics.get("_active_token_count", 0.0))

            reward_values.extend(float(x) for x in metrics.get("_reward_values", []))
            base_reward_values_all.extend(
                float(x) for x in metrics.get("_base_reward_values_all", [])
            )
            shaped_reward_values_active.extend(
                float(x) for x in metrics.get("_shaped_reward_values_active", [])
            )
            advantage_values.extend(float(x) for x in metrics.get("_advantage_values", []))
            active_advantage_values.extend(
                float(x) for x in metrics.get("_active_advantage_values", [])
            )
            kl_values.extend(float(x) for x in metrics.get("_kl_values", []))
            final_by_prefix.extend(
                (int(prefix), float(value))
                for prefix, value in metrics.get("_final_by_prefix", [])
            )
            required_lcs_fraction_values.extend(
                float(x)
                for x in metrics.get("_required_lcs_fraction_values", [])
            )
            format_valid_values.extend(
                float(x) for x in metrics.get("_format_valid_values", [])
            )
            protocol_complete_values.extend(
                float(x) for x in metrics.get("_protocol_complete_values", [])
            )
            semantic_adjustment_values_active.extend(
                float(x)
                for x in metrics.get("_semantic_adjustment_values_active", [])
            )
            token_entropy_values.extend(
                float(x) for x in metrics.get("_token_entropy_values", [])
            )
            semantic_high_values.extend(
                float(x) for x in metrics.get("_semantic_high_values", [])
            )
            semantic_low_values.extend(
                float(x) for x in metrics.get("_semantic_low_values", [])
            )

            # Backward-compatible fallback for metrics created before raw values.
            if not metrics.get("_token_entropy_values"):
                if metrics.get("num_valid", 0) and "max_token_entropy" in metrics:
                    max_token_entropy_values.append(float(metrics["max_token_entropy"]))
                if metrics.get("num_valid", 0) and "min_token_entropy" in metrics:
                    min_token_entropy_values.append(float(metrics["min_token_entropy"]))
            if "semantic_entropy" in metrics:
                semantic_entropy_values.append(float(metrics["semantic_entropy"]))
            if (
                int(metrics.get("num_valid", 0)) > 0
                and not bool(metrics.get("_expected_zero_gradient", False))
            ):
                expected_zero_gradient = False

        if not base_reward_values_all:
            base_reward_values_all = reward_values

        final_values = [value for _, value in final_by_prefix]
        final_by_length = {
            prefix: [value for item_prefix, value in final_by_prefix if item_prefix == prefix]
            for prefix in (1, 2, 3)
        }
        total_success = int(sum(protocol_complete_values))

        aggregated = {
            "loss": loss_weighted_sum / max(total_valid, 1),
            "avg_reward": self._mean_or_nan(base_reward_values_all),
            "base_reward_all": self._mean_or_nan(base_reward_values_all),
            "shaped_reward_active": self._mean_or_nan(shaped_reward_values_active),
            "std_reward": float(np.std(base_reward_values_all)) if base_reward_values_all else float("nan"),
            "min_reward": float(np.min(base_reward_values_all)) if base_reward_values_all else float("nan"),
            "max_reward": float(np.max(base_reward_values_all)) if base_reward_values_all else float("nan"),
            "median_reward": float(np.median(base_reward_values_all)) if base_reward_values_all else float("nan"),
            "avg_advantage": float(np.mean(advantage_values)) if advantage_values else 0.0,
            "std_advantage": float(np.std(advantage_values)) if advantage_values else 0.0,
            "advantage_abs_active": (
                float(np.mean(np.abs(active_advantage_values)))
                if active_advantage_values
                else float("nan")
            ),
            "avg_kl": float(np.mean(kl_values)) if kl_values else 0.0,
            "max_kl": float(np.max(kl_values)) if kl_values else 0.0,
            "min_kl": float(np.min(kl_values)) if kl_values else 0.0,
            "num_valid": total_valid,
            "num_total": total_trajectories,
            "num_success": total_success,
            "num_errors": total_errors,
            "protocol_complete_rate": self._mean_or_nan(protocol_complete_values),
            "trainable_rate": total_valid / max(total_trajectories, 1),
            "final_rate": self._mean_or_nan(final_values),
            "final_rate_p1": self._mean_or_nan(final_by_length[1]),
            "final_rate_p2": self._mean_or_nan(final_by_length[2]),
            "final_rate_p3": self._mean_or_nan(final_by_length[3]),
            "required_lcs_fraction": self._mean_or_nan(required_lcs_fraction_values),
            "format_valid_rate": self._mean_or_nan(format_valid_values),
            "semantic_entropy_high": self._mean_or_nan(semantic_high_values),
            "semantic_entropy_low": self._mean_or_nan(semantic_low_values),
            "semantic_adjustment_active": self._mean_or_nan(
                semantic_adjustment_values_active
            ),
            "policy_logprob_mean": policy_logprob_sum / max(total_trajectories, 1),
            "ref_logprob_mean": ref_logprob_sum / max(total_trajectories, 1),
            "avg_token_entropy": token_entropy_weighted_sum / max(total_active_tokens, 1.0),
            "max_token_entropy": (
                float(np.max(token_entropy_values))
                if token_entropy_values
                else (
                    float(np.max(max_token_entropy_values))
                    if max_token_entropy_values
                    else float("nan")
                )
            ),
            "min_token_entropy": (
                float(np.min(token_entropy_values))
                if token_entropy_values
                else (
                    float(np.min(min_token_entropy_values))
                    if min_token_entropy_values
                    else float("nan")
                )
            ),
            "semantic_entropy": float(np.mean(semantic_entropy_values)) if semantic_entropy_values else 0.0,
            "eval_final_p1": float("nan"),
            "eval_final_p2": float("nan"),
            "eval_final_p3": float("nan"),
            "_expected_zero_gradient": expected_zero_gradient,
        }
        return aggregated

    @staticmethod
    def _format_number(value: Any, digits: int = 2, signed: bool = False) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "N/A"
        if not np.isfinite(numeric):
            return "N/A"
        sign = "+" if signed else ""
        return f"{numeric:{sign}.{digits}f}"

    @classmethod
    def _format_percent(cls, value: Any, digits: int = 0) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "N/A"
        if not np.isfinite(numeric):
            return "N/A"
        return f"{100.0 * numeric:.{digits}f}%"

    @classmethod
    def _format_training_metrics(
        cls,
        step: int,
        metrics: Dict[str, Any],
    ) -> str:
        return (
            f"\nStep {step}\n"
            f"BaseReward(all)={cls._format_number(metrics.get('base_reward_all'))}  "
            f"ShapedReward(active)={cls._format_number(metrics.get('shaped_reward_active'))}  "
            f"BaseEMA={cls._format_number(metrics.get('reward_ema'))}\n"
            f"Final={cls._format_percent(metrics.get('final_rate'))} "
            f"[p1={cls._format_percent(metrics.get('final_rate_p1'))}, "
            f"p2={cls._format_percent(metrics.get('final_rate_p2'))}, "
            f"p3={cls._format_percent(metrics.get('final_rate_p3'))}]  "
            f"RequiredLCS={cls._format_percent(metrics.get('required_lcs_fraction'))}\n"
            f"ProtocolComplete={cls._format_percent(metrics.get('protocol_complete_rate'))}  "
            f"FormatValid={cls._format_percent(metrics.get('format_valid_rate'))}  "
            f"Trainable={cls._format_percent(metrics.get('trainable_rate'))}\n"
            f"SemanticH[high]={cls._format_number(metrics.get('semantic_entropy_high'))}  "
            f"SemanticH[low]={cls._format_number(metrics.get('semantic_entropy_low'))}  "
            f"Adjustment={cls._format_number(metrics.get('semantic_adjustment_active'), signed=True)}\n"
            f"AdvAbs={cls._format_number(metrics.get('advantage_abs_active'))}  "
            f"TokenH={cls._format_number(metrics.get('avg_token_entropy'))}  "
            f"GradNorm={cls._format_number(metrics.get('grad_norm'), digits=6)}\n"
            f"EvalFinal[p1/p2/p3]="
            f"{cls._format_percent(metrics.get('eval_final_p1'))}/"
            f"{cls._format_percent(metrics.get('eval_final_p2'))}/"
            f"{cls._format_percent(metrics.get('eval_final_p3'))}"
        )

    def _compute_token_logprobs_chunked(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities in chunks to avoid OOM.

        FIXED: Removed unused loss_mask parameter for clarity.
        FIXED: Handle CPU tensors by moving chunks to GPU for computation.

        Args:
            logits: [batch_size, seq_len, vocab_size] - can be on CPU or GPU
            labels: [batch_size, seq_len] - should be on GPU

        Returns:
            token_logprobs: [batch_size, seq_len] - log prob of each token on GPU
        """
        batch_size, seq_len, vocab_size = logits.shape
        logits_device = logits.device
        target_device = labels.device  # GPU device for output

        # Output tensor on GPU
        token_logprobs = torch.zeros(batch_size, seq_len, device=target_device)

        # Process in chunks along sequence dimension
        num_chunks = (seq_len + self.logprob_chunk_size - 1) // self.logprob_chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.logprob_chunk_size
            end_idx = min((chunk_idx + 1) * self.logprob_chunk_size, seq_len)

            # Extract chunk
            logits_chunk = logits[:, start_idx:end_idx, :]  # [B, chunk_size, V]
            labels_chunk = labels[:, start_idx:end_idx]      # [B, chunk_size]

            # Reference logits may live on CPU or another CUDA device.
            if logits_chunk.device != target_device:
                logits_chunk = logits_chunk.contiguous().to(target_device)
            else:
                logits_chunk = logits_chunk.contiguous()

            # Compute log_softmax only for this chunk
            log_probs_chunk = F.log_softmax(logits_chunk, dim=-1)  # [B, chunk_size, V]

            # Gather log probs for actual tokens
            labels_expanded = labels_chunk.unsqueeze(-1)  # [B, chunk_size, 1]
            selected_logprobs = torch.gather(log_probs_chunk, dim=-1, index=labels_expanded)

            # Store: [B, chunk_size]
            token_logprobs[:, start_idx:end_idx] = selected_logprobs.squeeze(-1)

        return token_logprobs

    def _compute_token_entropies_chunked(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token entropies in chunks to avoid OOM.

        Args:
            logits: [batch_size, seq_len, vocab_size] - typically on GPU for policy model

        Returns:
            token_entropies: [batch_size, seq_len]
        """
        batch_size, seq_len, _ = logits.shape
        device = logits.device
        token_entropies = torch.zeros(batch_size, seq_len, device=device)

        # Compute entropies without tracking gradients to save memory
        with torch.no_grad():
            for start_idx in range(0, seq_len, self.logprob_chunk_size):
                end_idx = min(start_idx + self.logprob_chunk_size, seq_len)
                chunk_logits = logits[:, start_idx:end_idx, :].detach()
                chunk_logprobs = F.log_softmax(chunk_logits, dim=-1)
                chunk_probs = chunk_logprobs.exp()
                chunk_entropy = -(chunk_probs * chunk_logprobs).sum(dim=-1)
                token_entropies[:, start_idx:end_idx] = chunk_entropy

        return token_entropies

    @staticmethod
    def _module_device(module: Any) -> torch.device:
        """Return a usable device for a module, including parameterless wrappers."""
        try:
            return next(module.parameters()).device
        except (StopIteration, AttributeError):
            try:
                return next(module.buffers()).device
            except (StopIteration, AttributeError):
                return torch.device("cpu")

    @staticmethod
    def _unwrap_model_for_parts(model: Any) -> Any:
        """Unwrap common PEFT and single-process wrapper layers.

        The returned module is used only to access the already-injected backbone
        and LM head. LoRA modules remain installed in the underlying backbone.

        Calling submodules directly is intended for a single-process model or a
        model sharded with a device map. A custom FSDP/pipeline wrapper may need
        an architecture-specific implementation instead.
        """
        current = model
        seen = set()

        while id(current) not in seen:
            seen.add(id(current))
            changed = False

            # DDP-like wrappers expose the wrapped module here. Parameters keep
            # their autograd hooks, but custom distributed wrappers may require
            # their own implementation.
            wrapped = getattr(current, "module", None)
            if wrapped is not None and wrapped is not current:
                current = wrapped
                changed = True
                continue

            get_base_model = getattr(current, "get_base_model", None)
            if callable(get_base_model):
                try:
                    base_model = get_base_model()
                except Exception:
                    base_model = None
                if base_model is not None and base_model is not current:
                    current = base_model
                    changed = True
                    continue

            if not changed:
                break

        return current

    def _get_causal_lm_parts(self, model: Any) -> Tuple[Any, Any, Any]:
        """Return ``(causal_lm, transformer_backbone, lm_head)``.

        Supports the common Hugging Face causal-LM layout used by Gemma, Llama,
        Qwen and PEFT wrappers. For multimodal wrappers, nested language-model
        modules are preferred over the outer conditional-generation module.
        """
        root = self._unwrap_model_for_parts(model)
        queue: List[Any] = []

        # Prefer nested language models when present.
        for candidate in (
            getattr(root, "language_model", None),
            getattr(getattr(root, "model", None), "language_model", None),
            getattr(root, "text_model", None),
            root,
        ):
            if candidate is not None:
                queue.append(candidate)

        seen = set()
        while queue:
            candidate = queue.pop(0)
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))

            get_output_embeddings = getattr(candidate, "get_output_embeddings", None)
            lm_head = None
            if callable(get_output_embeddings):
                try:
                    lm_head = get_output_embeddings()
                except Exception:
                    lm_head = None
            if lm_head is None:
                lm_head = getattr(candidate, "lm_head", None)

            backbone = None
            get_decoder = getattr(candidate, "get_decoder", None)
            if callable(get_decoder):
                try:
                    backbone = get_decoder()
                except Exception:
                    backbone = None
            if backbone is None:
                possible_backbone = getattr(candidate, "model", None)
                if possible_backbone is not candidate:
                    backbone = possible_backbone

            if lm_head is not None and backbone is not None:
                return candidate, backbone, lm_head

            for attribute in ("language_model", "text_model", "base_model", "model"):
                nested = getattr(candidate, attribute, None)
                if nested is not None and nested is not candidate:
                    queue.append(nested)

        raise RuntimeError(
            "Could not locate a causal-LM transformer backbone and output head. "
            f"Top-level model type: {type(model).__name__}. Add an "
            "architecture-specific branch to _get_causal_lm_parts()."
        )

    @staticmethod
    def _extract_last_hidden_state(outputs: Any) -> torch.Tensor:
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is not None:
            return hidden_states
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        raise RuntimeError(
            f"Backbone output {type(outputs).__name__} has no last_hidden_state"
        )

    def _run_backbone(
        self,
        backbone: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run only the transformer backbone, never the full vocabulary head."""
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": True,
        }
        try:
            outputs = backbone(**kwargs)
        except TypeError as exc:
            # A few custom backbones do not expose return_dict. Retry only for
            # that compatibility case; preserve unrelated TypeErrors.
            if "return_dict" not in str(exc):
                raise
            kwargs.pop("return_dict")
            outputs = backbone(**kwargs)
        return self._extract_last_hidden_state(outputs)

    @staticmethod
    def _apply_model_logit_transforms(causal_lm: Any, logits: torch.Tensor) -> torch.Tensor:
        """Apply model-specific transforms normally performed after ``lm_head``."""
        final_logits_bias = getattr(causal_lm, "final_logits_bias", None)
        if final_logits_bias is not None:
            logits = logits + final_logits_bias.to(logits.device, dtype=logits.dtype)

        config = getattr(causal_lm, "config", None)
        softcap = getattr(config, "final_logit_softcapping", None)
        if softcap is None:
            softcap = getattr(config, "final_logit_soft_cap", None)
        if softcap is not None:
            softcap_value = float(softcap)
            if softcap_value > 0.0:
                logits = torch.tanh(logits / softcap_value) * softcap_value

        return logits

    @staticmethod
    def _selected_logprobs_from_logits(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return log-probabilities for selected labels without a persistent log-softmax."""
        batch_size, sequence_length, vocab_size = logits.shape
        negative_log_likelihood = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            reduction="none",
        )
        return -negative_log_likelihood.view(batch_size, sequence_length)

    def _project_policy_hidden_states_chunked(
        self,
        causal_lm: Any,
        lm_head: Any,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        output_device: torch.device,
    ) -> torch.Tensor:
        """Project policy hidden states in sequence chunks with recomputation.

        Checkpointing each head chunk prevents every ``[B, chunk, V]`` tensor
        from being retained until backward. Only ``[B, S]`` selected token
        log-probabilities persist.
        """
        chunks: List[torch.Tensor] = []
        head_device = self._module_device(lm_head)
        sequence_length = hidden_states.shape[1]

        for start in range(0, sequence_length, self.logprob_chunk_size):
            end = min(start + self.logprob_chunk_size, sequence_length)
            hidden_chunk = hidden_states[:, start:end, :]
            label_chunk = labels[:, start:end]

            def project_and_select(
                chunk_hidden: torch.Tensor,
                labels_for_chunk: torch.Tensor = label_chunk,
            ) -> torch.Tensor:
                if chunk_hidden.device != head_device:
                    chunk_hidden = chunk_hidden.to(head_device)
                logits_chunk = lm_head(chunk_hidden)
                logits_chunk = self._apply_model_logit_transforms(causal_lm, logits_chunk)
                chunk_labels = labels_for_chunk.to(logits_chunk.device)
                selected = self._selected_logprobs_from_logits(logits_chunk, chunk_labels)
                return selected.to(output_device)

            if torch.is_grad_enabled() and hidden_chunk.requires_grad:
                selected_logprobs = checkpoint(
                    project_and_select,
                    hidden_chunk,
                    use_reentrant=False,
                )
            else:
                selected_logprobs = project_and_select(hidden_chunk)
            chunks.append(selected_logprobs)

        if not chunks:
            return torch.empty(
                hidden_states.shape[0],
                0,
                device=output_device,
                dtype=hidden_states.dtype,
            )
        return torch.cat(chunks, dim=1)

    def _project_reference_hidden_states_chunked(
        self,
        causal_lm: Any,
        lm_head: Any,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        output_device: torch.device,
    ) -> torch.Tensor:
        """Project frozen reference hidden states without retaining vocabulary logits."""
        chunks: List[torch.Tensor] = []
        head_device = self._module_device(lm_head)
        sequence_length = hidden_states.shape[1]

        with torch.no_grad():
            for start in range(0, sequence_length, self.logprob_chunk_size):
                end = min(start + self.logprob_chunk_size, sequence_length)
                hidden_chunk = hidden_states[:, start:end, :]
                if hidden_chunk.device != head_device:
                    hidden_chunk = hidden_chunk.to(head_device)
                logits_chunk = lm_head(hidden_chunk)
                logits_chunk = self._apply_model_logit_transforms(causal_lm, logits_chunk)
                label_chunk = labels[:, start:end].to(logits_chunk.device)
                selected = self._selected_logprobs_from_logits(logits_chunk, label_chunk)
                chunks.append(selected.to(output_device))
                del logits_chunk, selected

        if not chunks:
            return torch.empty(
                hidden_states.shape[0],
                0,
                device=output_device,
                dtype=hidden_states.dtype,
            )
        return torch.cat(chunks, dim=1)

    def _compute_entropies_from_hidden_states_chunked(
        self,
        causal_lm: Any,
        lm_head: Any,
        hidden_states: torch.Tensor,
        output_device: torch.device,
    ) -> torch.Tensor:
        """Compute token entropy in LM-head chunks for logging only."""
        batch_size, sequence_length, _ = hidden_states.shape
        if not self.compute_token_entropy:
            return torch.zeros(
                batch_size,
                sequence_length,
                device=output_device,
                dtype=torch.float32,
            )

        chunks: List[torch.Tensor] = []
        head_device = self._module_device(lm_head)
        with torch.no_grad():
            for start in range(0, sequence_length, self.logprob_chunk_size):
                end = min(start + self.logprob_chunk_size, sequence_length)
                hidden_chunk = hidden_states[:, start:end, :].detach()
                if hidden_chunk.device != head_device:
                    hidden_chunk = hidden_chunk.to(head_device)
                logits_chunk = lm_head(hidden_chunk)
                logits_chunk = self._apply_model_logit_transforms(causal_lm, logits_chunk)
                log_probs = F.log_softmax(logits_chunk.float(), dim=-1)
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
                chunks.append(entropy.to(output_device))
                del logits_chunk, log_probs, entropy

        return torch.cat(chunks, dim=1) if chunks else torch.empty(
            batch_size,
            0,
            device=output_device,
            dtype=torch.float32,
        )

    def _compute_token_logprobs_and_kl_from_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute token log-probabilities without allocating full-sequence logits."""
        output_device = input_ids.device
        policy_causal_lm, policy_backbone, policy_lm_head = self._get_causal_lm_parts(
            self.model
        )
        policy_hidden_states = self._run_backbone(
            policy_backbone,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        shifted_policy_hidden = policy_hidden_states[:, :-1, :]
        shifted_labels = input_ids[:, 1:].contiguous()

        policy_token_logprobs = self._project_policy_hidden_states_chunked(
            causal_lm=policy_causal_lm,
            lm_head=policy_lm_head,
            hidden_states=shifted_policy_hidden,
            labels=shifted_labels,
            output_device=output_device,
        )
        policy_token_entropies = self._compute_entropies_from_hidden_states_chunked(
            causal_lm=policy_causal_lm,
            lm_head=policy_lm_head,
            hidden_states=shifted_policy_hidden,
            output_device=output_device,
        )

        if self.beta == 0.0 or self.ref_model is self.model:
            ref_token_logprobs = torch.zeros_like(policy_token_logprobs)
            token_kl = torch.zeros_like(policy_token_logprobs)
            return (
                policy_token_logprobs,
                ref_token_logprobs,
                token_kl,
                policy_token_entropies,
            )

        ref_causal_lm, ref_backbone, ref_lm_head = self._get_causal_lm_parts(
            self.ref_model
        )
        ref_device = self._module_device(ref_backbone)
        with torch.no_grad():
            ref_hidden_states = self._run_backbone(
                ref_backbone,
                input_ids=input_ids.to(ref_device),
                attention_mask=attention_mask.to(ref_device),
            )
            shifted_ref_hidden = ref_hidden_states[:, :-1, :]
            ref_token_logprobs = self._project_reference_hidden_states_chunked(
                causal_lm=ref_causal_lm,
                lm_head=ref_lm_head,
                hidden_states=shifted_ref_hidden,
                labels=shifted_labels.to(ref_device),
                output_device=output_device,
            )

        token_kl = policy_token_logprobs - ref_token_logprobs
        return (
            policy_token_logprobs,
            ref_token_logprobs,
            token_kl,
            policy_token_entropies,
        )

    @staticmethod
    def _slice_and_trim_batch(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        start: int,
        end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Slice trajectories and remove columns that are padding for the microbatch."""
        micro_input_ids = input_ids[start:end]
        micro_attention_mask = attention_mask[start:end]
        micro_loss_mask = loss_mask[start:end]

        active_columns = (
            micro_attention_mask.bool().any(dim=0).nonzero(as_tuple=False).flatten()
        )
        if active_columns.numel() == 0:
            raise RuntimeError("Training microbatch contains no active input tokens")

        first_column = int(active_columns[0].item())
        last_column = int(active_columns[-1].item()) + 1
        if last_column - first_column < 2:
            raise RuntimeError("Training microbatch must contain at least two tokens")

        return (
            micro_input_ids[:, first_column:last_column],
            micro_attention_mask[:, first_column:last_column],
            micro_loss_mask[:, first_column:last_column],
            first_column,
            last_column,
        )

    def _compute_semantic_entropy(
        self,
        trajectories: List[CompletedTrajectory],
        valid_mask: torch.Tensor
    ) -> float:
        """
        Compute semantic entropy by grouping trajectories by semantic similarity.

        Semantic entropy measures uncertainty at the meaning level rather than token level.
        We group trajectories by their outcomes/actions and compute entropy over these semantic clusters.

        Args:
            trajectories: List of completed trajectories
            valid_mask: Boolean mask for valid trajectories

        Returns:
            Semantic entropy value (scalar)
        """
        if valid_mask.sum() == 0:
            return 0.0

        # Group trajectories by their semantic outcomes
        # We use termination reason + success status as a proxy for semantic grouping
        semantic_groups = defaultdict(list)

        for idx, traj in enumerate(trajectories):
            if not valid_mask[idx]:
                continue

            # Create semantic signature based on:
            # 1. Termination reason (success, max_turns, error)
            # 2. Reward bucket (discretize reward into bins for clustering)
            # 3. Number of turns (trajectory length)
            reward_bucket = int(traj.reward * 10) / 10 if traj.reward is not None else 0.0
            semantic_key = (
                traj.termination_reason,
                reward_bucket,
                len(traj.segments)
            )
            semantic_groups[semantic_key].append(idx)

        # Compute probability distribution over semantic clusters
        total_valid = valid_mask.sum().item()
        cluster_probs = []

        for cluster_indices in semantic_groups.values():
            cluster_prob = len(cluster_indices) / total_valid
            cluster_probs.append(cluster_prob)

        # Compute entropy: H = -sum(p * log(p))
        cluster_probs = np.array(cluster_probs)
        semantic_entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))

        return float(semantic_entropy)

    def _compute_logprobs_and_kl(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute trajectory statistics using chunked LM-head projection."""
        (
            policy_token_logprobs,
            ref_token_logprobs,
            token_kl,
            policy_token_entropies,
        ) = self._compute_token_logprobs_and_kl_from_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        shift_loss_mask = loss_mask[:, 1:].contiguous().float()
        masked_policy_logprobs = policy_token_logprobs * shift_loss_mask
        masked_ref_logprobs = ref_token_logprobs * shift_loss_mask
        num_trainable = shift_loss_mask.sum(dim=1).clamp(min=1.0)

        policy_traj_logprobs = masked_policy_logprobs.sum(dim=1) / num_trainable
        ref_traj_logprobs = masked_ref_logprobs.sum(dim=1) / num_trainable
        kl_per_traj = (token_kl * shift_loss_mask).sum(dim=1) / num_trainable

        return (
            policy_traj_logprobs,
            ref_traj_logprobs,
            kl_per_traj,
            policy_token_logprobs,
            policy_token_entropies,
            shift_loss_mask,
        )

    def train_step(
        self,
        batch_trajectories: Dict[str, List[CompletedTrajectory]],
        collated_batch: Dict[str, torch.Tensor],
        loss_scale: float = 1.0,
    ) -> Dict[str, float]:
        """Perform one GRPO step with trajectory-level training microbatches."""
        input_ids = collated_batch["input_ids"].to(self.device)
        attention_mask = collated_batch["attention_mask"].to(self.device)
        loss_mask = collated_batch["loss_mask"].to(self.device)
        batch_size = input_ids.shape[0]

        all_trajectories: List[CompletedTrajectory] = []
        for trajectories in batch_trajectories.values():
            all_trajectories.extend(trajectories)
        assert len(all_trajectories) == batch_size, (
            f"Mismatch: {len(all_trajectories)} trajectories vs {batch_size} collated"
        )

        valid_values: List[bool] = []
        for trajectory in all_trajectories:
            is_valid, error_message = trajectory.validate_structure()
            if not is_valid:
                logger.warning(
                    "Invalid trajectory %s: %s", trajectory.query_id, error_message
                )
            valid_values.append(is_valid)

        for index, trajectory in enumerate(all_trajectories):
            if trajectory.reward is None:
                task_id = self._trajectory_base_id(trajectory.query_id)
                try:
                    trajectory.reward = self.reward_function(task_id, trajectory)
                except Exception as exc:
                    logger.error("Reward computation failed for %s: %s", task_id, exc)
            if trajectory.reward is None:
                logger.error(
                    "Trajectory %s has None reward after computation", trajectory.query_id
                )
                trajectory.reward = 0.0
                valid_values[index] = False

        valid_mask = torch.tensor(valid_values, dtype=torch.bool, device=self.device)
        if valid_mask.sum() == 0:
            logger.error("No valid trajectories in batch")
            quality_metrics = self._build_quality_metric_payload(
                all_trajectories, valid_mask
            )
            rewards = [float(trajectory.reward or 0.0) for trajectory in all_trajectories]
            return {
                "loss": 0.0,
                "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
                "std_reward": float(np.std(rewards)) if rewards else 0.0,
                "avg_advantage": 0.0,
                "avg_kl": 0.0,
                "avg_token_entropy": 0.0,
                "max_token_entropy": 0.0,
                "min_token_entropy": 0.0,
                "semantic_entropy": 0.0,
                "grad_norm": 0.0,
                "num_valid": 0,
                "num_total": batch_size,
                "num_success": int(sum(quality_metrics["_protocol_complete_values"])),
                "num_errors": sum(
                    trajectory.termination_reason != "success"
                    for trajectory in all_trajectories
                ),
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

        groups: Dict[str, List[CompletedTrajectory]] = {}
        for trajectory in all_trajectories:
            groups.setdefault(
                self._trajectory_base_id(trajectory.query_id), []
            ).append(trajectory)

        for base_id, group_trajectories in groups.items():
            group_rewards = [
                float(trajectory.reward or 0.0) for trajectory in group_trajectories
            ]
            mean_reward = float(np.mean(group_rewards))
            std_reward = float(np.std(group_rewards)) if len(group_rewards) > 1 else 1.0
            for trajectory in group_trajectories:
                trajectory.advantage = (
                    float(trajectory.reward or 0.0) - mean_reward
                ) / (std_reward + 1e-8)

        advantages = torch.tensor(
            [float(trajectory.advantage or 0.0) for trajectory in all_trajectories],
            dtype=torch.float32,
            device=self.device,
        )
        valid_count = valid_mask.sum().clamp(min=1).float()

        policy_traj_values = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        ref_traj_values = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        kl_traj_values = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        entropy_per_traj_values = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )

        total_loss_numerator = 0.0
        total_entropy_sum = 0.0
        total_active_tokens = 0.0

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
                _,
                _,
            ) = self._slice_and_trim_batch(
                input_ids,
                attention_mask,
                loss_mask,
                micro_start,
                micro_end,
            )

            (
                policy_traj_logprobs,
                ref_traj_logprobs,
                kl_per_traj,
                _,
                policy_token_entropies,
                shift_loss_mask,
            ) = self._compute_logprobs_and_kl(
                micro_input_ids,
                micro_attention_mask,
                micro_loss_mask,
            )

            micro_advantages = advantages[micro_start:micro_end]
            valid_float = micro_valid_mask.float()
            loss_per_traj = -(
                micro_advantages * policy_traj_logprobs
                - self.beta * kl_per_traj
            ) * valid_float
            micro_loss_numerator = loss_per_traj.sum()
            (micro_loss_numerator / valid_count / float(loss_scale)).backward()

            with torch.no_grad():
                policy_traj_values[micro_start:micro_end] = policy_traj_logprobs.detach().to(
                    self.device, dtype=torch.float32
                )
                ref_traj_values[micro_start:micro_end] = ref_traj_logprobs.detach().to(
                    self.device, dtype=torch.float32
                )
                kl_traj_values[micro_start:micro_end] = kl_per_traj.detach().to(
                    self.device, dtype=torch.float32
                )

                active_token_mask = shift_loss_mask * valid_float.unsqueeze(1)
                active_counts = active_token_mask.sum(dim=1).clamp(min=1.0)
                entropy_per_traj = (
                    policy_token_entropies * active_token_mask
                ).sum(dim=1) / active_counts
                entropy_per_traj_values[micro_start:micro_end] = entropy_per_traj.detach().to(
                    self.device, dtype=torch.float32
                )
                total_entropy_sum += float(
                    (policy_token_entropies * active_token_mask).sum().item()
                )
                total_active_tokens += float(active_token_mask.sum().item())
                total_loss_numerator += float(micro_loss_numerator.detach().item())

            del (
                policy_traj_logprobs,
                ref_traj_logprobs,
                kl_per_traj,
                policy_token_entropies,
                micro_loss_numerator,
            )

        unscaled_loss = total_loss_numerator / float(valid_count.item())

        with torch.no_grad():
            rewards = [float(trajectory.reward or 0.0) for trajectory in all_trajectories]
            advantages_np = [
                float(trajectory.advantage or 0.0) for trajectory in all_trajectories
            ]
            valid_token_entropy_values = (
                entropy_per_traj_values[valid_mask].detach().cpu().tolist()
            )
            valid_kl_values = kl_traj_values[valid_mask].detach().cpu().tolist()
            semantic_entropy = self._compute_semantic_entropy(
                all_trajectories, valid_mask
            )
            quality_metrics = self._build_quality_metric_payload(
                all_trajectories, valid_mask
            )

            termination_reasons: Dict[str, int] = {}
            for trajectory in all_trajectories:
                reason = trajectory.termination_reason
                termination_reasons[reason] = termination_reasons.get(reason, 0) + 1

            expected_zero_gradient = (
                self.beta == 0.0
                and all(
                    abs(advantages_np[index]) <= 1e-12
                    for index, is_valid in enumerate(valid_values)
                    if is_valid
                )
            )

            metrics = {
                "loss": unscaled_loss,
                "avg_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "min_reward": float(np.min(rewards)),
                "max_reward": float(np.max(rewards)),
                "median_reward": float(np.median(rewards)),
                "avg_advantage": float(np.mean(advantages_np)),
                "std_advantage": float(np.std(advantages_np)),
                "avg_kl": float(np.mean(valid_kl_values)) if valid_kl_values else 0.0,
                "max_kl": float(np.max(valid_kl_values)) if valid_kl_values else 0.0,
                "min_kl": float(np.min(valid_kl_values)) if valid_kl_values else 0.0,
                "avg_token_entropy": total_entropy_sum / max(total_active_tokens, 1.0),
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
                "policy_logprob_mean": float(policy_traj_values.mean().item()),
                "ref_logprob_mean": float(ref_traj_values.mean().item()),
                "num_success": int(sum(quality_metrics["_protocol_complete_values"])),
                "num_max_turns": termination_reasons.get("max_turns_reached", 0),
                "num_errors": sum(
                    count
                    for reason, count in termination_reasons.items()
                    if reason != "success"
                ),
                "_loss_weighted_sum": unscaled_loss * int(valid_mask.sum().item()),
                "_reward_values": rewards,
                "_advantage_values": advantages_np,
                "_kl_values": valid_kl_values,
                "_policy_logprob_sum": float(policy_traj_values.sum().item()),
                "_ref_logprob_sum": float(ref_traj_values.sum().item()),
                "_token_entropy_weighted_sum": total_entropy_sum,
                "_active_token_count": total_active_tokens,
                "_token_entropy_values": valid_token_entropy_values,
                "_expected_zero_gradient": expected_zero_gradient,
                **quality_metrics,
            }

        return metrics

    def train(
        self,
        rollout_manager: Any,
        train_queries: List[Dict],
        num_epochs: int = 1,
        group_size: int = 4,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        collator: Any = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 100,
        log_callback: Optional[Callable] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        evaluation_callback: Optional[Callable[[int], Dict[str, float]]] = None,
    ) -> List[Dict]:
        """
        Full training loop.
        
        Args:
            rollout_manager: AgenticRolloutManager instance
            train_queries: List of training queries (dicts with 'id' and 'user' keys)
            num_epochs: Number of training epochs
            group_size: Number of trajectories per query (G)
            batch_size: Number of queries per batch
            collator: TrajectoryCollator instance
            checkpoint_dir: Directory to save checkpoints (optional)
            checkpoint_every: Save checkpoint every N steps
            log_callback: Optional callback for logging (e.g., wandb)
            evaluation_callback: Optional fixed held-out evaluator. It must return
                eval_final_p1, eval_final_p2, and eval_final_p3 rates.
            
        Returns:
            List of all metrics dicts from training
        """
        logger.info(f"Starting GRPO training: {num_epochs} epochs, {len(train_queries)} queries")
        logger.info(f"Group size: {group_size}, Batch size: {batch_size}")
        logger.info(f"Beta (KL coeff): {self.beta}, Max grad norm: {self.max_grad_norm}")
        if evaluation_callback is None:
            logger.info(
                "No held-out evaluation callback configured; EvalFinal will be N/A"
            )
        
        # Create checkpoint directory if specified
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
        
        train_queries_by_id = {query["id"]: query for query in train_queries}
        global_step = int(resume_state.get("global_step", 0)) if resume_state else 0
        update_step = int(resume_state.get("update_step", 0)) if resume_state else 0
        all_metrics = list(resume_state.get("all_metrics", [])) if resume_state else []
        accumulation_metrics: List[Dict[str, Any]] = []
        start_epoch = int(resume_state.get("epoch", 0)) if resume_state else 0
        resume_batch_idx = int(resume_state.get("next_batch_idx", 0)) if resume_state else 0
        resume_shuffled_ids = list(resume_state.get("shuffled_query_ids", [])) if resume_state else []
        if resume_state:
            self.reward_ema = resume_state.get("reward_ema")
            self._restore_rng_state(resume_state.get("rng_state"))
            logger.info(
                "Resuming training from epoch=%d, batch=%d, global_step=%d, update_step=%d",
                start_epoch + 1,
                resume_batch_idx + 1,
                global_step,
                update_step,
            )
        self.optimizer.zero_grad()
        for epoch in range(start_epoch, num_epochs):
            logger.info(f"\n{'='*80}")
            logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*80}\n")
            
            # Shuffle queries
            if epoch == start_epoch and resume_shuffled_ids:
                queries_shuffled = [
                    train_queries_by_id[qid] for qid in resume_shuffled_ids
                    if qid in train_queries_by_id
                ]
                if len(queries_shuffled) != len(train_queries):
                    missing_ids = [
                        query["id"] for query in train_queries
                        if query["id"] not in {q["id"] for q in queries_shuffled}
                    ]
                    queries_shuffled.extend(train_queries_by_id[qid] for qid in missing_ids)
                batch_start_idx = resume_batch_idx
            else:
                queries_shuffled = train_queries.copy()
                random.shuffle(queries_shuffled)
                batch_start_idx = 0
            
            # Process in batches
            num_batches = (len(queries_shuffled) + batch_size - 1) // batch_size
            
            for batch_idx in tqdm(range(batch_start_idx, num_batches), desc=f"Epoch {epoch+1}"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(queries_shuffled))
                batch_queries = queries_shuffled[start_idx:end_idx]
                
                # Generate trajectories
                logger.info(f"\nBatch {batch_idx + 1}/{num_batches}: Generating trajectories...")
                batch_trajectories = rollout_manager.generate_batch_trajectories(
                    queries=batch_queries,
                    group_size=group_size
                )
                
                # Collate trajectories
                all_trajs = []
                for query_id, trajs in batch_trajectories.items():
                    all_trajs.extend(trajs)
                
                if collator is None:
                    logger.error("No collator provided!")
                    continue
                
                if getattr(self, "requires_agentflow_collation", False):
                    collated_batch = collator.collate_agentflow(all_trajs)
                else:
                    collated_batch = collator.collate(all_trajs)
                
                # Training step
                metrics = self.train_step(batch_trajectories, collated_batch, gradient_accumulation_steps)
                accumulation_metrics.append(metrics)
                
                # 4. Optimizer Step (only every N batches)
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    aggregated_metrics = self._aggregate_accumulated_metrics(accumulation_metrics)

                    trainable_gradients = [
                        (name, parameter)
                        for name, parameter in self.model.named_parameters()
                        if parameter.requires_grad and parameter.grad is not None
                    ]
                    expected_zero_gradient = bool(
                        aggregated_metrics.get("_expected_zero_gradient", False)
                    )
                    if (
                        aggregated_metrics.get("num_valid", 0) > 0
                        and not trainable_gradients
                        and not expected_zero_gradient
                    ):
                        raise RuntimeError(
                            "Valid trajectories produced no trainable parameter "
                            "gradients. Check LoRA target modules before continuing; "
                            "the optimizer would otherwise perform a silent no-op."
                        )
                    
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    aggregated_metrics['grad_norm'] = grad_norm.item()
                    if trainable_gradients and not torch.isfinite(grad_norm):
                        raise RuntimeError(
                            f"Non-finite gradient norm before optimizer step: {grad_norm.item()}"
                        )
                    if (
                        aggregated_metrics.get("num_valid", 0) > 0
                        and grad_norm.item() <= 0.0
                        and not expected_zero_gradient
                    ):
                        raise RuntimeError(
                            "Valid trajectories produced a zero gradient norm. "
                            "Stopping before a silent no-op optimizer step."
                        )
                    
                    if expected_zero_gradient and grad_norm.item() <= 0.0:
                        logger.warning(
                            "Skipping optimizer step because all accumulated "
                            "AgentFlow groups have zero advantage and zero KL signal"
                        )
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    update_step += 1
                    
                    # Update exponential moving average of rewards
                    current_reward = aggregated_metrics['avg_reward']
                    if self.reward_ema is None:
                        self.reward_ema = current_reward
                    else:
                        self.reward_ema = (
                            self.reward_ema_alpha * current_reward + 
                            (1 - self.reward_ema_alpha) * self.reward_ema
                        )
                    aggregated_metrics['reward_ema'] = self.reward_ema
                    # =========================================
                    # Log only on update steps
                    global_step += 1 # Or use update_step

                    if evaluation_callback is not None:
                        try:
                            evaluation_metrics = evaluation_callback(global_step) or {}
                            for prefix in (1, 2, 3):
                                key = f"eval_final_p{prefix}"
                                if key in evaluation_metrics:
                                    aggregated_metrics[key] = float(
                                        evaluation_metrics[key]
                                    )
                        except Exception as exc:
                            logger.warning(
                                "Held-out evaluation failed at step %d: %s",
                                global_step,
                                exc,
                            )
                    
                    # Add standard logging fields
                    aggregated_metrics['epoch'] = epoch + 1
                    aggregated_metrics['global_step'] = global_step
                    all_metrics.append(aggregated_metrics)
                    
                    logger.info(
                        self._format_training_metrics(
                            global_step,
                            aggregated_metrics,
                        )
                    )
                    
                    if log_callback:
                        log_callback(aggregated_metrics, global_step)
                    
                    # # Checkpointing
                    if checkpoint_dir and global_step % checkpoint_every == 0:
                        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}")
                        logger.info(f"Saving checkpoint to {checkpoint_path}")
                        self._save_checkpoint(
                            checkpoint_path=checkpoint_path,
                            epoch=epoch,
                            next_batch_idx=batch_idx + 1,
                            global_step=global_step,
                            update_step=update_step,
                            num_epochs=num_epochs,
                            batch_size=batch_size,
                            group_size=group_size,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            shuffled_query_ids=[query["id"] for query in queries_shuffled],
                            all_metrics=all_metrics,
                        )
                    accumulation_metrics = []
            resume_batch_idx = 0
            resume_shuffled_ids = []

            if checkpoint_dir and global_step % checkpoint_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}")
                logger.info(f"Saving checkpoint to {checkpoint_path}")
                self._save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    epoch=epoch,
                    next_batch_idx=num_batches,
                    global_step=global_step,
                    update_step=update_step,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    group_size=group_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    shuffled_query_ids=[query["id"] for query in queries_shuffled],
                    all_metrics=all_metrics,
                )
        
        # Final checkpoint
        if checkpoint_dir:
            final_path = os.path.join(checkpoint_dir, "final_model")
            logger.info(f"\nTraining complete! Saving final model to {final_path}")
            self._save_checkpoint(
                checkpoint_path=final_path,
                epoch=num_epochs,
                next_batch_idx=0,
                global_step=global_step,
                update_step=update_step,
                num_epochs=num_epochs,
                batch_size=batch_size,
                group_size=group_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                shuffled_query_ids=[],
                all_metrics=all_metrics,
            )
        
        # Finish wandb if used
        # if self.use_wandb:
        #     try:
        #         import wandb
        #         wandb.finish()
        #     except ImportError:
        #         logger.warning("wandb not available but use_wandb=True")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING COMPLETE")
        logger.info(f"Total steps: {global_step}")
        logger.info(f"Final avg reward: {all_metrics[-1]['avg_reward']:.3f}")
        logger.info(f"{'='*80}\n")
        
        return all_metrics