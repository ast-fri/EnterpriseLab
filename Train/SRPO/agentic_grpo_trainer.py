# agentric_grpo_trainer.py
from dataclasses import dataclass

@dataclass
class TrainingBatch:
    """Encapsulate batch data - no distributed logic here."""
    prompts: List[str]
    ground_truth: List[List[Dict]]
    trajectories: List[List[Dict]]
    rewards: torch.Tensor
    token_ids: torch.Tensor
    token_mask: torch.Tensor


class SimplifiedAgenticGRPOTrainer(GRPOTrainer):
    """Simplified trainer - minimal overrides, composition-based."""
    
    def __init__(self, model, args, tokenizer, train_dataset, 
                 trajectory_generator, reward_judge, **kwargs):
        super().__init__(model=model, args=args, train_dataset=train_dataset, **kwargs)
        
        self.tokenizer = tokenizer
        self.trajectory_gen = trajectory_generator  # Pure component
        self.reward_judge = reward_judge  # Pure component
        self.num_generations = args.num_generations
    
    def generate_and_score_completions(self, inputs: List[Dict], **kwargs) -> Dict:
        """
        Main generation pipeline - orchestrates pure components.
        
        Key principle: Only main process generates & scores,
        then broadcasts results to all processes.
        """
        device = self.accelerator.device
        
        # Extract from batch
        prompts = [x['prompt'] for x in inputs]
        ground_truth_list = [x['ground_truth'] for x in inputs]
        
        # Synchronize before any work
        self.accelerator.wait_for_everyone()
        
        # MAIN PROCESS: Generate and score
        if self.accelerator.is_main_process:
            print(f"Generating {len(prompts) * self.num_generations} trajectories...")
            
            trajectories_list = []
            rewards_list = []
            
            for prompt, ground_truth in zip(prompts, ground_truth_list):
                for gen_idx in range(self.num_generations):
                    # Use pure components (no distributed logic)
                    trajectory = self.trajectory_gen.generate(
                        prompt,
                        temperature=0.7 + (gen_idx * 0.1)
                    )
                    trajectories_list.append(trajectory)
                    
                    # Score trajectory
                    reward = self.reward_judge.score(trajectory, ground_truth)
                    rewards_list.append(reward)
            
            # Convert to tensor
            rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32, device=device)
            
            # Tokenize (only on main process to avoid dtype mismatches)
            token_ids_list = []
            for traj in trajectories_list:
                traj_text = self._format_trajectory(traj)
                inputs_enc = self.tokenizer(traj_text, return_tensors='pt', padding=False)
                token_ids_list.append(inputs_enc['input_ids'].squeeze(0))
            
            token_ids = pad(token_ids_list, padding_value=self.tokenizer.pad_token_id)
            token_mask = (token_ids != self.tokenizer.pad_token_id).float()
            
        else:
            # Non-main processes: Create dummy tensors
            rewards_tensor = None
            token_ids = None
            token_mask = None
            trajectories_list = None
        
        # Broadcast to all processes
        self.accelerator.wait_for_everyone()
        
        broadcast_objects = [rewards_tensor, token_ids, token_mask, trajectories_list]
        broadcast_objects = broadcast_object_list(broadcast_objects, from_process=0)
        rewards_tensor, token_ids, token_mask, trajectories_list = broadcast_objects
        
        # Move to device
        token_ids = token_ids.to(device)
        token_mask = token_mask.to(device)
        rewards_tensor = rewards_tensor.to(device)
        
        # Synchronize final broadcasts
        self.accelerator.wait_for_everyone()
        
        # Compute advantages (same logic as before, but cleaner)
        rewards_reshaped = rewards_tensor.view(-1, self.num_generations)
        mean_rewards = rewards_reshaped.mean(dim=1).repeat_interleave(self.num_generations)
        advantages = rewards_tensor - mean_rewards
        
        return {
            'completion_ids': token_ids,
            'completion_mask': token_mask,
            'advantages': advantages,
            'rewards': rewards_tensor,
        }
    
    def _format_trajectory(self, traj: List[Dict]) -> str:
        """Simple trajectory formatting."""
        text = ""
        for step in traj:
            text += f"<thought>{step['thought']}</thought>"
            text += f"<action>{step['action']}</action>"
            text += f"<observation>{step['observation']}</observation>"
        return text
