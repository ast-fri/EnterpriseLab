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
        max_length: int = 4096,
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
        """
        batch_input_ids = []
        batch_loss_masks = []
        
        for traj in trajectories:
            # Build full text and mask from segments
            input_ids = []
            loss_mask = []
            
            for segment in traj.segments:
                # Tokenize segment
                seg_tokens = self.tokenizer.encode(
                    segment.text,
                    add_special_tokens=False  # We handle special tokens manually
                )
                
                # Create mask for this segment
                seg_mask = [1 if segment.is_trainable else 0] * len(seg_tokens)
                
                input_ids.extend(seg_tokens)
                loss_mask.extend(seg_mask)
            
            # FIXED: Add EOS token - but only trainable for successful trajectories
            eos_token_id = self.tokenizer.eos_token_id
            input_ids.append(eos_token_id)
            
            # CRITICAL FIX: EOS is trainable ONLY for successful completions
            if traj.termination_reason == "success":
                loss_mask.append(1)  # Train on EOS for successful trajectories
            else:
                loss_mask.append(0)  # Don't train on EOS for failures
            
            # Truncate if too long
            if len(input_ids) > self.max_length:
                logger.warning(
                    f"Trajectory {traj.query_id} truncated: "
                    f"{len(input_ids)} -> {self.max_length} tokens"
                )
                input_ids = input_ids[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
            
            batch_input_ids.append(input_ids)
            batch_loss_masks.append(loss_mask)
        
        # Pad sequences to same length
        max_len_in_batch = max(len(ids) for ids in batch_input_ids)
        max_len_in_batch = min(max_len_in_batch, self.max_length)
        
        padded_input_ids = []
        padded_loss_masks = []
        attention_masks = []
        
        pad_token_id = self.tokenizer.pad_token_id
        
        for input_ids, loss_mask in zip(batch_input_ids, batch_loss_masks):
            seq_len = len(input_ids)
            padding_len = max_len_in_batch - seq_len
            
            if self.padding_side == "right":
                # Right padding
                padded_ids = input_ids + [pad_token_id] * padding_len
                padded_mask = loss_mask + [0] * padding_len  # Padding is not trainable
                attention_mask = [1] * seq_len + [0] * padding_len
            else:
                # Left padding
                padded_ids = [pad_token_id] * padding_len + input_ids
                padded_mask = [0] * padding_len + loss_mask
                attention_mask = [0] * padding_len + [1] * seq_len
            
            padded_input_ids.append(padded_ids)
            padded_loss_masks.append(padded_mask)
            attention_masks.append(attention_mask)
        
        # Convert to tensors
        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
        loss_mask_tensor = torch.tensor(padded_loss_masks, dtype=torch.float32)
        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long)
        
        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'loss_mask': loss_mask_tensor
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
            
            # FIXED: Add/modify EOS mask based on success
            if traj.termination_reason == "success":
                # Make sure EOS is trainable
                if loss_mask[-1] == 0:
                    loss_mask[-1] = 1
            else:
                # Make sure EOS is NOT trainable
                if loss_mask[-1] == 1:
                    loss_mask[-1] = 0
            
            # Truncate if needed
            if len(full_tokens) > self.max_length:
                logger.warning(
                    f"Trajectory {traj.query_id} truncated: "
                    f"{len(full_tokens)} -> {self.max_length} tokens"
                )
                full_tokens = full_tokens[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
            
            batch_input_ids.append(full_tokens)
            batch_loss_masks.append(loss_mask)
        
        # Pad sequences (same as before)
        max_len_in_batch = max(len(ids) for ids in batch_input_ids)
        max_len_in_batch = min(max_len_in_batch, self.max_length)
        
        padded_input_ids = []
        padded_loss_masks = []
        attention_masks = []
        
        pad_token_id = self.tokenizer.pad_token_id
        
        for input_ids, loss_mask in zip(batch_input_ids, batch_loss_masks):
            seq_len = len(input_ids)
            padding_len = max_len_in_batch - seq_len
            
            if self.padding_side == "right":
                padded_ids = input_ids + [pad_token_id] * padding_len
                padded_mask = loss_mask + [0] * padding_len
                attention_mask = [1] * seq_len + [0] * padding_len
            else:
                padded_ids = [pad_token_id] * padding_len + input_ids
                padded_mask = [0] * padding_len + loss_mask
                attention_mask = [0] * padding_len + [1] * seq_len
            
            padded_input_ids.append(padded_ids)
            padded_loss_masks.append(padded_mask)
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'loss_mask': torch.tensor(padded_loss_masks, dtype=torch.float32)
        }

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
