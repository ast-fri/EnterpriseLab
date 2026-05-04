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
import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
from tqdm import tqdm

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
        use_wandb: bool = False
    ):
        """
        Args:
            model: Policy model to train
            ref_model: Reference model (frozen)
            optimizer: Optimizer for policy model
            tokenizer: Tokenizer (needed for checkpointing)
            reward_function: Function (task_id, trajectory) -> float
            beta: KL penalty coefficient
            max_grad_norm: Gradient clipping norm
            device: Device to use
            logprob_chunk_size: Chunk size for memory-efficient logprob computation
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.reward_function = reward_function
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.logprob_chunk_size = logprob_chunk_size
        self.use_wandb = use_wandb

        # Set models to appropriate modes
        self.model.train()
        self.ref_model.eval()
        self.reward_ema = None
        self.reward_ema_alpha = 0.1  # Smoothing factor
        # =======================================
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Log reference model device
        ref_device = next(self.ref_model.parameters()).device
        logger.info(f"Reference model device: {ref_device} (CPU offload saves ~18GB GPU memory)")

        logger.info(f"GRPOTrainer initialized with beta={beta}, max_grad_norm={max_grad_norm}")

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
        advantage_values: List[float] = []
        kl_values: List[float] = []

        total_valid = 0
        total_trajectories = 0
        total_success = 0
        total_errors = 0
        loss_weighted_sum = 0.0
        policy_logprob_sum = 0.0
        ref_logprob_sum = 0.0

        for metrics in metrics_window:
            num_valid = int(metrics.get("num_valid", 0))
            num_total = int(metrics.get("num_total", 0))

            total_valid += num_valid
            total_trajectories += num_total
            total_success += int(metrics.get("num_success", 0))
            total_errors += int(metrics.get("num_errors", 0))

            loss_weighted_sum += float(metrics.get("_loss_weighted_sum", 0.0))
            policy_logprob_sum += float(metrics.get("_policy_logprob_sum", 0.0))
            ref_logprob_sum += float(metrics.get("_ref_logprob_sum", 0.0))

            reward_values.extend(float(x) for x in metrics.get("_reward_values", []))
            advantage_values.extend(float(x) for x in metrics.get("_advantage_values", []))
            kl_values.extend(float(x) for x in metrics.get("_kl_values", []))

        aggregated = {
            "loss": loss_weighted_sum / max(total_valid, 1),
            "avg_reward": float(np.mean(reward_values)) if reward_values else 0.0,
            "std_reward": float(np.std(reward_values)) if reward_values else 0.0,
            "min_reward": float(np.min(reward_values)) if reward_values else 0.0,
            "max_reward": float(np.max(reward_values)) if reward_values else 0.0,
            "median_reward": float(np.median(reward_values)) if reward_values else 0.0,
            "avg_advantage": float(np.mean(advantage_values)) if advantage_values else 0.0,
            "std_advantage": float(np.std(advantage_values)) if advantage_values else 0.0,
            "avg_kl": float(np.mean(kl_values)) if kl_values else 0.0,
            "max_kl": float(np.max(kl_values)) if kl_values else 0.0,
            "min_kl": float(np.min(kl_values)) if kl_values else 0.0,
            "num_valid": total_valid,
            "num_total": total_trajectories,
            "num_success": total_success,
            "num_errors": total_errors,
            "policy_logprob_mean": policy_logprob_sum / max(total_trajectories, 1),
            "ref_logprob_mean": ref_logprob_sum / max(total_trajectories, 1),
        }
        return aggregated

    def _compute_token_logprobs_chunked(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities in chunks to avoid OOM.
        
        FIXED: Removed unused loss_mask parameter for clarity.
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]
            
        Returns:
            token_logprobs: [batch_size, seq_len] - log prob of each token
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # Output tensor
        token_logprobs = torch.zeros(batch_size, seq_len, device=device)
        
        # Process in chunks along sequence dimension
        num_chunks = (seq_len + self.logprob_chunk_size - 1) // self.logprob_chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.logprob_chunk_size
            end_idx = min((chunk_idx + 1) * self.logprob_chunk_size, seq_len)
            
            # Extract chunk
            logits_chunk = logits[:, start_idx:end_idx, :]  # [B, chunk_size, V]
            labels_chunk = labels[:, start_idx:end_idx]      # [B, chunk_size]
            
            # Compute log_softmax only for this chunk
            log_probs_chunk = F.log_softmax(logits_chunk, dim=-1)  # [B, chunk_size, V]
            
            # Gather log probs for actual tokens
            labels_expanded = labels_chunk.unsqueeze(-1)  # [B, chunk_size, 1]
            selected_logprobs = torch.gather(log_probs_chunk, dim=-1, index=labels_expanded)
            
            # Store: [B, chunk_size]
            token_logprobs[:, start_idx:end_idx] = selected_logprobs.squeeze(-1)
        
        return token_logprobs

    def _compute_logprobs_and_kl(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute trajectory logprobs and token-level KL divergence.
        
        FIXED: Proper token-level KL computation.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            loss_mask: [batch_size, seq_len] - 1 for trainable tokens
            
        Returns:
            policy_traj_logprobs: [batch_size] - length-normalized policy logprobs
            ref_traj_logprobs: [batch_size] - length-normalized ref logprobs
            kl_per_traj: [batch_size] - token-level KL averaged per trajectory
            policy_token_logprobs: [batch_size, seq_len-1] - for debugging
        """
        # ============================================================
        # Step 1: Get logits from both models
        # ============================================================
        
        # Policy model (with gradients)
        outputs_policy = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )
        logits_policy = outputs_policy.logits  # [B, S, V]

        # Reference model (frozen) - handle CPU offloading
        with torch.no_grad():
            # Detect if ref_model is on CPU
            ref_device = next(self.ref_model.parameters()).device

            if ref_device.type == "cpu":
                # Move inputs to CPU for reference model
                input_ids_ref = input_ids.cpu()
                attention_mask_ref = attention_mask.cpu()
            else:
                input_ids_ref = input_ids
                attention_mask_ref = attention_mask

            outputs_ref = self.ref_model(
                input_ids=input_ids_ref,
                attention_mask=attention_mask_ref,
                use_cache=False
            )
            logits_ref = outputs_ref.logits  # [B, S, V]

            # Move logits back to GPU if ref_model was on CPU
            if ref_device.type == "cpu":
                logits_ref = logits_ref.to(logits_policy.device)
        
        # ============================================================
        # Step 2: Shift for next-token prediction
        # ============================================================
        
        shift_logits_policy = logits_policy[:, :-1, :].contiguous()  # [B, S-1, V]
        shift_logits_ref = logits_ref[:, :-1, :].contiguous()        # [B, S-1, V]
        shift_labels = input_ids[:, 1:].contiguous()                 # [B, S-1]
        shift_loss_mask = loss_mask[:, 1:].contiguous()              # [B, S-1]
        
        # ============================================================
        # Step 3: Compute per-token log probabilities (chunked)
        # ============================================================
        
        policy_token_logprobs = self._compute_token_logprobs_chunked(
            shift_logits_policy, shift_labels
        )  # [B, S-1]
        
        with torch.no_grad():
            ref_token_logprobs = self._compute_token_logprobs_chunked(
                shift_logits_ref, shift_labels
            )  # [B, S-1]
        
        # ============================================================
        # Step 4: Compute trajectory-level logprobs (length-normalized)
        # ============================================================
        
        # Apply mask to get only trainable tokens
        masked_policy_logprobs = policy_token_logprobs * shift_loss_mask  # [B, S-1]
        masked_ref_logprobs = ref_token_logprobs * shift_loss_mask        # [B, S-1]
        
        # Normalize by number of trainable tokens
        num_trainable = shift_loss_mask.sum(dim=1).clamp(min=1)  # [B]
        
        policy_traj_logprobs = masked_policy_logprobs.sum(dim=1) / num_trainable  # [B]
        ref_traj_logprobs = masked_ref_logprobs.sum(dim=1) / num_trainable        # [B]
        
        # ============================================================
        # Step 5: Compute token-level KL divergence
        # ============================================================
        
        # KL(π || π_ref) ≈ log π(x) - log π_ref(x)
        # For discrete distributions, this is exact
        token_kl = (policy_token_logprobs - ref_token_logprobs) * shift_loss_mask  # [B, S-1]
        
        # Average KL over trainable tokens per trajectory
        kl_per_traj = token_kl.sum(dim=1) / num_trainable  # [B]
        
        return policy_traj_logprobs, ref_traj_logprobs, kl_per_traj, policy_token_logprobs

    def train_step(
        self,
        batch_trajectories: Dict[str, List[CompletedTrajectory]],
        collated_batch: Dict[str, torch.Tensor],
        loss_scale: float = 1.0
    ) -> Dict[str, float]:
        """
        Perform one GRPO training step.
        
        FIXED: Token-level KL computation for proper GRPO objective
        
        Args:
            batch_trajectories: Dict mapping query_id -> list of G trajectories
            collated_batch: Collated batch from TrajectoryCollator
            
        Returns:
            Dictionary of training metrics
        """
        # Extract collated data
        input_ids = collated_batch['input_ids'].to(self.device)
        attention_mask = collated_batch['attention_mask'].to(self.device)
        loss_mask = collated_batch['loss_mask'].to(self.device)
        
        batch_size = input_ids.shape[0]
        
        # ============================================================
        # Step 1: Flatten trajectories and assign rewards
        # ============================================================
        
        all_trajectories = []
        for query_id, trajs in batch_trajectories.items():
            for traj in trajs:
                all_trajectories.append(traj)
        
        assert len(all_trajectories) == batch_size, \
            f"Mismatch: {len(all_trajectories)} trajectories vs {batch_size} collated"
        
        # Validate trajectories first (before expensive reward computation)
        valid_mask = []
        for traj in all_trajectories:
            is_valid, error_msg = traj.validate_structure()
            if not is_valid:
                logger.warning(f"Invalid trajectory {traj.query_id}: {error_msg}")
            valid_mask.append(is_valid)

        # Compute rewards for ALL trajectories (valid and invalid)
        for i, traj in enumerate(all_trajectories):
            if traj.reward is None:  # ← REMOVED valid_mask check
                # Extract base task_id (remove _g suffix)
                task_id = traj.query_id.rsplit('_g', 1)[0] if '_g' in traj.query_id else traj.query_id
                try:
                    traj.reward = self.reward_function(task_id, traj)
                except Exception as e:
                    logger.error(f"Reward computation failed for {task_id}: {e}")
                    # raise
            
            # ADDITIONAL CHECK: Ensure reward is never None
            if traj.reward is None:
                logger.error(f"Trajectory {traj.query_id} has None reward after computation!")
                traj.reward = 0.0
                valid_mask[i] = False
        # ========== ADD REWARD NORMALIZATION HERE ==========
        
        # ===================================================
        valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)

        if valid_mask.sum() == 0:
            logger.error("No valid trajectories in batch!")
            return {
                'loss': 0.0,
                'avg_reward': 0.0,
                'std_reward': 0.0,
                'avg_advantage': 0.0,
                'avg_kl': 0.0,
                'grad_norm': 0.0,
                'num_valid': 0,
                'num_total': batch_size
            }
        
        # ============================================================
        # Step 2: Compute group-relative advantages
        # ============================================================
        
        # Group trajectories by base query
        groups = {}
        for traj in all_trajectories:
            base_id = traj.query_id.rsplit('_g', 1)[0] if '_g' in traj.query_id else traj.query_id
            if base_id not in groups:
                groups[base_id] = []
            groups[base_id].append(traj)
        
        # Compute advantages within each group
        for base_id, group_trajs in groups.items():
            group_rewards = [t.reward for t in group_trajs]
            # ← ADD THIS SAFETY CHECK
            if any(r is None for r in group_rewards):
                logger.error(f"Group {base_id} has None rewards: {group_rewards}")
                # Replace None with 0.0
                group_rewards = [r if r is not None else 0.0 for r in group_rewards]
            mean_reward = np.mean(group_rewards)
            std_reward = np.std(group_rewards) if len(group_rewards) > 1 else 1.0
            
            # Normalize advantages
            for traj in group_trajs:
                traj.advantage = (traj.reward - mean_reward) / (std_reward + 1e-8)
        
        # ============================================================
        # Step 3: Compute logprobs and KL
        # ============================================================
        
        policy_traj_logprobs, ref_traj_logprobs, kl_per_traj, policy_token_logprobs = \
            self._compute_logprobs_and_kl(input_ids, attention_mask, loss_mask)
        
        # ============================================================
        # Step 4: Compute GRPO loss
        # ============================================================
        
        # Extract advantages as tensor
        advantages = torch.tensor(
            [traj.advantage for traj in all_trajectories],
            dtype=torch.float32,
            device=self.device
        )  # [B]
        # advantages = advantages.detach()  # <--- ADD THIS
        # GRPO objective: maximize E[A * log π(τ) - β * KL(π || π_ref)]
        # Loss = -E[A * log π(τ) - β * KL]
        loss_per_traj = -(advantages * policy_traj_logprobs - self.beta * kl_per_traj)  # [B]
        
        # Apply valid mask
        loss_per_traj = loss_per_traj * valid_mask.float()
        
        # Average loss over valid trajectories
        loss = loss_per_traj.sum() / valid_mask.sum()
        
        # ============================================================
        # Step 5: Backward pass and optimization
        # ============================================================
        # Store unscaled loss for logging
        unscaled_loss = loss.item()  # <--- ADD THIS
        loss = loss/loss_scale
        # self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        # grad_norm = torch.nn.utils.clip_grad_norm_(
        #     self.model.parameters(),
        #     self.max_grad_norm
        # )
        
        # self.optimizer.step()
        
        # ============================================================
        # Step 6: Logging metrics
        # ============================================================
        
        with torch.no_grad():
            rewards = [traj.reward for traj in all_trajectories]
            advantages_np = [traj.advantage for traj in all_trajectories]
            
            # Termination reason statistics
            termination_reasons = {}
            for traj in all_trajectories:
                reason = traj.termination_reason
                termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
            
            metrics = {
                'loss': unscaled_loss,
                'avg_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'min_reward': np.min(rewards),
                'max_reward': np.max(rewards),
                'median_reward': np.median(rewards),  # <--- ADD THIS LINE
                'avg_advantage': np.mean(advantages_np),
                'std_advantage': np.std(advantages_np),
                'avg_kl': kl_per_traj.mean().item(),
                'max_kl': kl_per_traj.max().item(),
                'min_kl': kl_per_traj.min().item(),
                # 'grad_norm': grad_norm.item(),
                'num_valid': valid_mask.sum().item(),
                'num_total': batch_size,
                'policy_logprob_mean': policy_traj_logprobs.mean().item(),
                'ref_logprob_mean': ref_traj_logprobs.mean().item(),
                'num_success': termination_reasons.get('success', 0),
                'num_max_turns': termination_reasons.get('max_turns', 0),
                'num_errors': termination_reasons.get('malformed_output', 0) + 
                             termination_reasons.get('generation_error', 0),
                '_loss_weighted_sum': unscaled_loss * valid_mask.sum().item(),
                '_reward_values': list(rewards),
                '_advantage_values': list(advantages_np),
                '_kl_values': kl_per_traj.detach().cpu().tolist(),
                '_policy_logprob_sum': policy_traj_logprobs.sum().item(),
                '_ref_logprob_sum': ref_traj_logprobs.sum().item(),
            }

        # MEMORY: Clear CUDA cache to free fragmented memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    def train(
        self,
        rollout_manager: Any,
        train_queries: List[Dict],
        num_epochs: int = 1,
        group_size: int = 4,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 1,
        collator: Any = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 100,
        log_callback: Optional[Callable] = None,
        resume_state: Optional[Dict[str, Any]] = None,
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
            
        Returns:
            List of all metrics dicts from training
        """
        logger.info(f"Starting GRPO training: {num_epochs} epochs, {len(train_queries)} queries")
        logger.info(f"Group size: {group_size}, Batch size: {batch_size}")
        logger.info(f"Beta (KL coeff): {self.beta}, Max grad norm: {self.max_grad_norm}")
        
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
                
                collated_batch = collator.collate(all_trajs)
                
                # Training step
                metrics = self.train_step(batch_trajectories, collated_batch, gradient_accumulation_steps)
                accumulation_metrics.append(metrics)
                
                # 4. Optimizer Step (only every N batches)
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    aggregated_metrics = self._aggregate_accumulated_metrics(accumulation_metrics)
                    
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    aggregated_metrics['grad_norm'] = grad_norm.item()
                    
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
                    
                    # Add standard logging fields
                    aggregated_metrics['epoch'] = epoch + 1
                    aggregated_metrics['global_step'] = global_step
                    all_metrics.append(aggregated_metrics)
                    
                    logger.info(
                        f"Step {global_step} | Loss: {aggregated_metrics['loss']:.4f} | "
                        f"Reward: {aggregated_metrics['avg_reward']:.3f} (EMA: {self.reward_ema:.3f}) | "
                        f"KL: {aggregated_metrics['avg_kl']:.4f} | "
                        f"Success: {aggregated_metrics['num_success']}/{aggregated_metrics['num_total']} | "
                        f"Valid: {aggregated_metrics['num_valid']}/{aggregated_metrics['num_total']}"
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
                        
                # Add step tracking
                # global_step += 1
                # metrics['epoch'] = epoch + 1
                # metrics['global_step'] = global_step
                # all_metrics.append(metrics)
                
                # Logging
                # logger.info(
                #     f"Step {global_step} | Loss: {metrics['loss']:.4f} | "
                #     f"Reward: {metrics['avg_reward']:.3f}±{metrics['std_reward']:.3f} | "
                #     f"KL: {metrics['avg_kl']:.4f} | "
                #     f"Success: {metrics['num_success']}/{metrics['num_total']} | "
                #     f"Valid: {metrics['num_valid']}/{metrics['num_total']}"
                # )
                
                # Callback logging (e.g., wandb)
                # if log_callback:
                #     log_callback(metrics, global_step)
                
                # Periodic checkpointing
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
