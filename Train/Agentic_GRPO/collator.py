"""
Trajectory Collator for GRPO Training

FIXED:
- Mask alignment with shifted labels (causal LM)
- EOS only trainable for successful trajectories
- Proper padding and attention masks
"""

import torch
import logging
from typing import List, Dict, Any
from data_structures import CompletedTrajectory

logger = logging.getLogger(__name__)


class TrajectoryCollator:
    """
    Collates trajectories into batched tensors for training.
    
    FIXED: Proper mask alignment and selective EOS training
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 16000,
        padding_side: str = "right"
    ):
        """
        Args:
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            padding_side: Which side to pad on ("right" or "left")
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding_side = padding_side
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")

    def collate(self, trajectories: List[CompletedTrajectory]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of trajectories into training tensors.
        
        FIXED: 
        - Mask alignment for causal LM
        - EOS only trainable for successful trajectories
        
        Args:
            trajectories: List of CompletedTrajectory objects
            
        Returns:
            Dictionary containing:
            - input_ids: [batch_size, seq_len]
            - attention_mask: [batch_size, seq_len]
            - loss_mask: [batch_size, seq_len] - 1 for trainable tokens
            - turn_ids: [batch_size, seq_len] - planner turn index, or -1
        """
        batch_input_ids = []
        batch_loss_masks = []
        batch_turn_ids = []
        
        for traj in trajectories:
            # Build full text and mask from segments
            input_ids = []
            loss_mask = []
            turn_ids = []
            next_turn_id = 0
            
            for segment in traj.segments:
                # Tokenize segment
                seg_tokens = self.tokenizer.encode(
                    segment.text,
                    add_special_tokens=False  # We handle special tokens manually
                )
                
                # Create mask for this segment
                seg_mask = [1 if segment.is_trainable else 0] * len(seg_tokens)
                segment_turn_id = next_turn_id if segment.is_trainable else -1
                seg_turn_ids = [segment_turn_id] * len(seg_tokens)
                
                input_ids.extend(seg_tokens)
                loss_mask.extend(seg_mask)
                turn_ids.extend(seg_turn_ids)
                if segment.is_trainable:
                    next_turn_id += 1
            
            # FIXED: Add EOS token - but only trainable for successful trajectories
            eos_token_id = self.tokenizer.eos_token_id
            input_ids.append(eos_token_id)
            
            # CRITICAL FIX: EOS is trainable ONLY for successful completions
            if traj.termination_reason == "success":
                loss_mask.append(1)  # Train on EOS for successful trajectories
                turn_ids.append(next_turn_id - 1 if next_turn_id else -1)
            else:
                loss_mask.append(0)  # Don't train on EOS for failures
                turn_ids.append(-1)
            
            # Truncate if too long
            if len(input_ids) > self.max_length:
                logger.warning(
                    f"Trajectory {traj.query_id} truncated: "
                    f"{len(input_ids)} -> {self.max_length} tokens"
                )
                input_ids = input_ids[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
                turn_ids = turn_ids[:self.max_length]
            
            batch_input_ids.append(input_ids)
            batch_loss_masks.append(loss_mask)
            batch_turn_ids.append(turn_ids)
        
        # Pad sequences to same length
        max_len_in_batch = max(len(ids) for ids in batch_input_ids)
        max_len_in_batch = min(max_len_in_batch, self.max_length)
        
        padded_input_ids = []
        padded_loss_masks = []
        padded_turn_ids = []
        attention_masks = []
        
        pad_token_id = self.tokenizer.pad_token_id
        
        for input_ids, loss_mask, turn_ids in zip(
            batch_input_ids,
            batch_loss_masks,
            batch_turn_ids,
        ):
            seq_len = len(input_ids)
            padding_len = max_len_in_batch - seq_len
            
            if self.padding_side == "right":
                # Right padding
                padded_ids = input_ids + [pad_token_id] * padding_len
                padded_mask = loss_mask + [0] * padding_len  # Padding is not trainable
                padded_turns = turn_ids + [-1] * padding_len
                attention_mask = [1] * seq_len + [0] * padding_len
            else:
                # Left padding
                padded_ids = [pad_token_id] * padding_len + input_ids
                padded_mask = [0] * padding_len + loss_mask
                padded_turns = [-1] * padding_len + turn_ids
                attention_mask = [0] * padding_len + [1] * seq_len
            
            padded_input_ids.append(padded_ids)
            padded_loss_masks.append(padded_mask)
            padded_turn_ids.append(padded_turns)
            attention_masks.append(attention_mask)
        
        # Convert to tensors
        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
        loss_mask_tensor = torch.tensor(padded_loss_masks, dtype=torch.float32)
        turn_ids_tensor = torch.tensor(padded_turn_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long)
        
        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'loss_mask': loss_mask_tensor,
            'turn_ids': turn_ids_tensor,
        }

    def collate_with_chat_template(
        self,
        trajectories: List[CompletedTrajectory],
        messages_list: List[List[Dict[str, str]]]
    ) -> Dict[str, torch.Tensor]:
        """
        IMPROVED: Collate using chat template for proper format alignment.
        
        This version uses the same chat template formatting as rollout,
        ensuring training/inference consistency.
        
        Args:
            trajectories: List of CompletedTrajectory objects
            messages_list: List of message histories (one per trajectory)
                          Each is a list of {"role": ..., "content": ...}
            
        Returns:
            Same format as collate()
        """
        if len(trajectories) != len(messages_list):
            raise ValueError(
                f"Mismatch: {len(trajectories)} trajectories vs "
                f"{len(messages_list)} message lists"
            )
        
        batch_input_ids = []
        batch_loss_masks = []
        batch_turn_ids = []
        
        for traj, messages in zip(trajectories, messages_list):
            # Apply chat template to get formatted text
            try:
                formatted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,  # Don't add prompt, we have full conversation
                    enable_thinking=False
                )
            except TypeError:
                # Fallback if enable_thinking not supported
                formatted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            
            # Tokenize
            full_tokens = self.tokenizer.encode(
                formatted_text,
                add_special_tokens=True  # Chat template handles this
            )
            
            # Build loss mask based on roles
            # Only train on assistant messages
            loss_mask = self._build_mask_from_messages(messages, full_tokens)
            turn_ids = self._turn_ids_from_loss_mask(loss_mask)
            
            # FIXED: Add/modify EOS mask based on success
            if traj.termination_reason == "success":
                # Make sure EOS is trainable
                if loss_mask[-1] == 0:
                    loss_mask[-1] = 1
                    previous_turn_ids = [turn_id for turn_id in turn_ids[:-1] if turn_id >= 0]
                    turn_ids[-1] = previous_turn_ids[-1] if previous_turn_ids else -1
            else:
                # Make sure EOS is NOT trainable
                if loss_mask[-1] == 1:
                    loss_mask[-1] = 0
                turn_ids[-1] = -1
            
            # Truncate if needed
            if len(full_tokens) > self.max_length:
                logger.warning(
                    f"Trajectory {traj.query_id} truncated: "
                    f"{len(full_tokens)} -> {self.max_length} tokens"
                )
                full_tokens = full_tokens[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
                turn_ids = turn_ids[:self.max_length]
            
            batch_input_ids.append(full_tokens)
            batch_loss_masks.append(loss_mask)
            batch_turn_ids.append(turn_ids)
        
        # Pad sequences (same as before)
        max_len_in_batch = max(len(ids) for ids in batch_input_ids)
        max_len_in_batch = min(max_len_in_batch, self.max_length)
        
        padded_input_ids = []
        padded_loss_masks = []
        padded_turn_ids = []
        attention_masks = []
        
        pad_token_id = self.tokenizer.pad_token_id
        
        for input_ids, loss_mask, turn_ids in zip(
            batch_input_ids,
            batch_loss_masks,
            batch_turn_ids,
        ):
            seq_len = len(input_ids)
            padding_len = max_len_in_batch - seq_len
            
            if self.padding_side == "right":
                padded_ids = input_ids + [pad_token_id] * padding_len
                padded_mask = loss_mask + [0] * padding_len
                padded_turns = turn_ids + [-1] * padding_len
                attention_mask = [1] * seq_len + [0] * padding_len
            else:
                padded_ids = [pad_token_id] * padding_len + input_ids
                padded_mask = [0] * padding_len + loss_mask
                padded_turns = [-1] * padding_len + turn_ids
                attention_mask = [0] * padding_len + [1] * seq_len
            
            padded_input_ids.append(padded_ids)
            padded_loss_masks.append(padded_mask)
            padded_turn_ids.append(padded_turns)
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'loss_mask': torch.tensor(padded_loss_masks, dtype=torch.float32),
            'turn_ids': torch.tensor(padded_turn_ids, dtype=torch.long),
        }

    def collate_agentflow(
        self,
        trajectories: List[CompletedTrajectory],
    ) -> Dict[str, torch.Tensor]:
        """
        Flatten exact AgentFlow planner turns into prompt/response training rows.

        AgentFlow reconstructs the planner prompt from evolving memory each turn,
        so these turns cannot be represented as one concatenated conversation.
        """
        rows = []
        for trajectory_index, trajectory in enumerate(trajectories):
            for planner_turn in trajectory.planner_turns:
                prompt_ids = self.tokenizer.encode(
                    planner_turn.prompt,
                    add_special_tokens=False,
                )
                response_ids = self.tokenizer.encode(
                    planner_turn.response,
                    add_special_tokens=False,
                )
                response_ids.append(self.tokenizer.eos_token_id)

                if len(response_ids) >= self.max_length:
                    logger.warning(
                        "AgentFlow response truncated for %s turn %d: %d -> %d tokens",
                        trajectory.query_id,
                        planner_turn.turn_index,
                        len(response_ids),
                        self.max_length,
                    )
                    response_ids = response_ids[: self.max_length]
                    prompt_ids = []
                else:
                    max_prompt_tokens = self.max_length - len(response_ids)
                    if len(prompt_ids) > max_prompt_tokens:
                        logger.warning(
                            "AgentFlow prompt left-truncated for %s turn %d: %d -> %d tokens",
                            trajectory.query_id,
                            planner_turn.turn_index,
                            len(prompt_ids),
                            max_prompt_tokens,
                        )
                        prompt_ids = prompt_ids[-max_prompt_tokens:]

                input_ids = prompt_ids + response_ids
                rows.append(
                    {
                        "input_ids": input_ids,
                        "loss_mask": [0] * len(prompt_ids) + [1] * len(response_ids),
                        "turn_ids": (
                            [-1] * len(prompt_ids)
                            + [planner_turn.turn_index] * len(response_ids)
                        ),
                        "trajectory_index": trajectory_index,
                        "turn_index": planner_turn.turn_index,
                    }
                )

        if not rows:
            raise ValueError("AgentFlow collation requires at least one planner turn")

        max_row_length = max(len(row["input_ids"]) for row in rows)
        padded_input_ids = []
        padded_loss_masks = []
        padded_turn_ids = []
        attention_masks = []
        for row in rows:
            row_length = len(row["input_ids"])
            padding_length = max_row_length - row_length
            if self.padding_side == "right":
                padded_input_ids.append(
                    row["input_ids"] + [self.tokenizer.pad_token_id] * padding_length
                )
                padded_loss_masks.append(row["loss_mask"] + [0] * padding_length)
                padded_turn_ids.append(row["turn_ids"] + [-1] * padding_length)
                attention_masks.append([1] * row_length + [0] * padding_length)
            else:
                padded_input_ids.append(
                    [self.tokenizer.pad_token_id] * padding_length + row["input_ids"]
                )
                padded_loss_masks.append([0] * padding_length + row["loss_mask"])
                padded_turn_ids.append([-1] * padding_length + row["turn_ids"])
                attention_masks.append([0] * padding_length + [1] * row_length)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "loss_mask": torch.tensor(padded_loss_masks, dtype=torch.float32),
            "turn_ids": torch.tensor(padded_turn_ids, dtype=torch.long),
            "trajectory_indices": torch.tensor(
                [row["trajectory_index"] for row in rows],
                dtype=torch.long,
            ),
            "planner_turn_indices": torch.tensor(
                [row["turn_index"] for row in rows],
                dtype=torch.long,
            ),
        }

    @staticmethod
    def _turn_ids_from_loss_mask(loss_mask: List[int]) -> List[int]:
        """Assign a new planner turn whenever a trainable token span starts."""
        turn_ids = [-1] * len(loss_mask)
        current_turn = -1
        was_trainable = False
        for index, is_trainable in enumerate(loss_mask):
            if is_trainable and not was_trainable:
                current_turn += 1
            if is_trainable:
                turn_ids[index] = current_turn
            was_trainable = bool(is_trainable)
        return turn_ids

    def _build_mask_from_messages(
        self,
        messages: List[Dict[str, str]],
        full_tokens: List[int]
    ) -> List[int]:
        """
        Build loss mask from message roles.
        
        Only assistant messages are trainable.
        System and user messages are not trainable.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            full_tokens: Full tokenized sequence
            
        Returns:
            loss_mask: List of 0/1 indicating trainable positions
        """
        # This is a simplified version - proper implementation would
        # track exact token boundaries per message
        
        # For now, tokenize each message separately to find boundaries
        loss_mask = [0] * len(full_tokens)
        
        current_pos = 0
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            # Tokenize this message
            msg_tokens = self.tokenizer.encode(
                content,
                add_special_tokens=False
            )
            
            # Check if trainable (assistant messages only)
            is_trainable = (role == "assistant")
            
            # Mark positions as trainable
            end_pos = min(current_pos + len(msg_tokens), len(full_tokens))
            
            if is_trainable:
                for i in range(current_pos, end_pos):
                    loss_mask[i] = 1
            
            current_pos = end_pos
            
            if current_pos >= len(full_tokens):
                break
        
        return loss_mask


class SimpleCollator:
    """
    Simplified collator that directly uses trajectory segments.
    
    Use this if you don't have message histories stored.
    """
    
    def __init__(self, tokenizer: Any, max_length: int = 4096):
        self.base_collator = TrajectoryCollator(
            tokenizer=tokenizer,
            max_length=max_length
        )
    
    def collate(self, trajectories: List[CompletedTrajectory]) -> Dict[str, torch.Tensor]:
        """Forward to base collator."""
        return self.base_collator.collate(trajectories)
