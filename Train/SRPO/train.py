
import torch
import torch.nn as nn
import json
import re
import copy
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig


# ============================================================================
# DATASET
# ============================================================================
class AgenticDataset(Dataset):
    """Dataset for agentic trajectories"""
    
    def __init__(self, data_path: str):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item.get('messages', [])
        
        # Split into prompt and ground truth
        # Assuming format: [system, user, assistant1, user1, assistant2, ...]
        prompt = messages[:2]  # system + user
        ground_truth = messages[2:]  # rest is trajectory
        
        return {
            'prompt': prompt,
            'ground_truth': ground_truth,
            'query': messages[1].get('content', '')
        }


# ============================================================================
# TOOLS INTERFACE
# ============================================================================
from tools import Tools

class ToolExecutor:
    """Simple interface for tool execution"""
    
    def __init__(self):
        # Initialize your actual tools here
        self.tools = Tools()
    
    def execute(self, tool_name: str, tool_args: dict) -> str:
        """Execute a tool and return observation"""
        try:
            result = self.tools.get_tool_context([{
                "tool_name": tool_name, 
                "tool_arguments": tool_args
            }])
            return str(result)
            
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"


# ============================================================================
# SIMPLIFIED GRPO TRAINER
# ============================================================================
class SimpleAgenticGRPOTrainer(GRPOTrainer):
    """
    Simplified GRPO trainer for agentic environments.
    Works on single GPU first, can be extended to multi-GPU later.
    """
    
    def __init__(
        self,
        model,
        args: GRPOConfig,
        train_dataset,
        processing_class,
        tool_executor: ToolExecutor,
        max_trajectory_steps: int = 5,
        stop_strings: List[str] = None,
        **kwargs
    ):
        # Dummy reward function (we override _calculate_rewards)
        def dummy_reward(completions, **kwargs):
            return [0.0] * len(completions)
        
        super().__init__(
            model=model,
            reward_funcs=dummy_reward,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            **kwargs
        )
        
        self.tool_executor = tool_executor
        self.max_trajectory_steps = max_trajectory_steps
        self.stop_strings = stop_strings or ["final_answer", "done", "complete"]
        
        print(f"✅ Trainer initialized")
        print(f"   Device: {self.accelerator.device}")
        print(f"   Num processes: {self.accelerator.num_processes}")
        print(f"   Max trajectory steps: {self.max_trajectory_steps}")
    
    # ------------------------------------------------------------------------
    # TRAJECTORY GENERATION
    # ------------------------------------------------------------------------
    
    def generate_trajectory(
        self, 
        prompt_messages: List[Dict],
        temperature: float = 0.7
    ) -> List[Dict]:
        """
        Generate a trajectory of thought-action-observation steps.
        
        Args:
            prompt_messages: Initial conversation (system + user)
            temperature: Sampling temperature
        
        Returns:
            List of trajectory steps
        """
        trajectory = []
        messages = prompt_messages.copy()
        
        # Set model to eval mode for generation
        original_mode = self.model.training
        self.model.eval()
        
        with torch.no_grad():
            for step in range(self.max_trajectory_steps):
                # 1. Generate thought + action
                generated_text = self._generate_step(messages, temperature)
                
                # 2. Parse thought and action
                thought, action = self._parse_thought_action(generated_text)
                
                # 3. Execute tool to get observation
                observation = self._execute_tool_call(action)
                
                # 4. Store step
                step_data = {
                    'step': step,
                    'thought': thought,
                    'action': action,
                    'observation': observation
                }
                trajectory.append(step_data)
                
                # 5. Update conversation context
                messages.append({
                    'role': 'assistant',
                    'content': f"<thought>{thought}</thought>\n<action>{action}</action>"
                })
                messages.append({
                    'role': 'user',
                    'content': f"<observation>{observation}</observation>"
                })
                
                # 6. Check if done
                if self._is_trajectory_complete(action, observation):
                    print(f"   ✅ Trajectory complete at step {step}")
                    break
        
        # Restore training mode
        self.model.train(original_mode)
        
        return trajectory
    
    def _generate_step(self, messages: List[Dict], temperature: float) -> str:
        """Generate one step using the model"""
        # Apply chat template
        text = self.processing_class.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # Tokenize
        inputs = self.processing_class(
            text,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Store original input length for proper decoding
        input_length = inputs['input_ids'].shape[1]
        
        # Generate with explicit parameters to prevent repetition
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                min_new_tokens=10,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=self.processing_class.pad_token_id,
                eos_token_id=self.processing_class.eos_token_id,
                use_cache=True
            )
        
        # CRITICAL: Only decode the NEW tokens (exclude input)
        # This effectively implements return_input_text=False behavior
        if output_ids.shape[1] > input_length:
            generated_ids = output_ids[0, input_length:]
        else:
            # Fallback: model didn't generate anything new
            print("⚠️ Warning: No new tokens generated!")
            generated_ids = torch.tensor([], dtype=torch.long)
        
        # Decode only the generated portion
        generated_text = self.processing_class.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        input_ids_only = output_ids[0, :input_length]
        input_text = self.processing_class.decode(
            input_ids_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        generated_text = generated_text[len(input_text):].strip()
        # Debug: Check if generation is working
    
        print(f"   Input length: {input_length} tokens")
        print(f"   Output length: {output_ids.shape[1]} tokens")
        print(f"   Generated tokens: {output_ids.shape[1] - input_length}")
        
        print(f"📝 Generated ({len(generated_text)} chars): {generated_text}...")
        
        return generated_text
    
    def _parse_thought_action(self, text: str) -> Tuple[str, str]:
        """Parse thought and action from generated text"""
        thought = ""
        action = ""
        
        # Try to extract structured tags
        thought_match = re.search(r'<thought>(.*?)</thought>', text, re.DOTALL)
        action_match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
        
        if thought_match:
            thought = thought_match.group(1).strip()
        if action_match:
            action = action_match.group(1).strip()
        
        # Fallback: split text in half
        if not thought and not action:
            split_point = len(text) // 2
            thought = text[:split_point].strip()
            action = text[split_point:].strip()
        
        return thought, action
    
    def _execute_tool_call(self, action: str) -> str:
        """Execute tool call and return observation"""
        try:
            # Try to parse as JSON
            action_dict = json.loads(action)
            tool_name = action_dict.get('tool_name', '')
            tool_args = action_dict.get('tool_arguments', {})
            
            # Execute tool
            observation = self.tool_executor.execute(tool_name, tool_args)
            return observation
        
        except json.JSONDecodeError:
            # Not valid JSON, return as-is
            return f"Invalid action format: {action[:100]}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _is_trajectory_complete(self, action: str, observation: str) -> bool:
        """Check if trajectory should stop"""
        combined = f"{action} {observation}".lower()
        return any(stop in combined for stop in self.stop_strings)
    
    # ------------------------------------------------------------------------
    # REWARD CALCULATION
    # ------------------------------------------------------------------------
    
    def _calculate_rewards(
        self,
        inputs: List[Dict],
        original_prompts: List,
        completions: List,
        completion_ids: List = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Calculate rewards for generated trajectories.
        Uses rule-based heuristics (can be replaced with LLM judge later).
        
        Args:
            inputs: List of input dicts with 'prompt', 'ground_truth', 'query'
            original_prompts: Copy of original prompts
            completions: Generated trajectory texts (can be strings or token IDs)
            completion_ids: Token IDs (optional)
            **kwargs: Additional arguments
        """
        rewards = []
        
        for idx, (input_dict, completion) in enumerate(zip(inputs, completions)):
            ground_truth = input_dict.get('ground_truth', [])
            
            # Convert completion to text if it's not already
            if isinstance(completion, torch.Tensor):
                completion_text = self.processing_class.decode(
                    completion, 
                    skip_special_tokens=True
                )
            elif isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], int):
                completion_text = self.processing_class.decode(
                    completion, 
                    skip_special_tokens=True
                )
            elif isinstance(completion, str):
                completion_text = completion
            else:
                # Fallback: try to decode from completion_ids if available
                if completion_ids is not None and idx < len(completion_ids):
                    completion_text = self.processing_class.decode(
                        completion_ids[idx], 
                        skip_special_tokens=True
                    )
                else:
                    print(f"⚠️ Warning: Unknown completion type: {type(completion)}")
                    completion_text = str(completion)
            
            # Extract tools from generated trajectory
            gen_tools = self._extract_tools_from_text(completion_text)
            
            # Extract tools from ground truth
            ref_tools = self._extract_tools_from_messages(ground_truth)
            
            # Calculate reward
            reward = self._compute_trajectory_reward(gen_tools, ref_tools)
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.model.device)
    
    def _extract_tools_from_text(self, text: str) -> List[Dict]:
        """Extract tool calls from trajectory text"""
        tools = []
        
        # Safety check
        if not isinstance(text, str):
            print(f"⚠️ Warning: _extract_tools_from_text received {type(text)}, converting to string")
            text = str(text)
        
        # Find all <action> blocks
        try:
            actions = re.findall(r'<action>(.*?)</action>', text, re.DOTALL)
        except Exception as e:
            print(f"⚠️ Warning: regex failed: {e}")
            return tools
        
        for action in actions:
            try:
                action_dict = json.loads(action.strip())
                tools.append({
                    'name': action_dict.get('tool_name', ''),
                    'args': action_dict.get('tool_arguments', {})
                })
            except Exception as e:
                # Not valid JSON, skip
                continue
        
        return tools
    
    def _extract_tools_from_messages(self, messages: List[Dict]) -> List[Dict]:
        """Extract tool calls from ground truth messages"""
        tools = []
        
        for msg in messages:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Extract tools from this message
                message_tools = self._extract_tools_from_text(content)
                tools.extend(message_tools)
        
        return tools
    
    def _compute_trajectory_reward(
        self,
        gen_tools: List[Dict],
        ref_tools: List[Dict]
    ) -> float:
        """
        Compute reward based on tool matching.
        
        Scoring:
        - +1 for each correct tool name
        - +1 for each correct tool arguments
        - +2 bonus for correct sequence order
        """
        score = 0.0
        
        # 1. Tool name matching
        gen_names = [t['name'] for t in gen_tools]
        ref_names = [t['name'] for t in ref_tools]
        
        for gen_name in gen_names:
            if gen_name in ref_names:
                score += 1.0
        
        # 2. Tool arguments matching (for matching names)
        for gen_tool in gen_tools:
            for ref_tool in ref_tools:
                if gen_tool['name'] == ref_tool['name']:
                    # Check if arguments match
                    if self._args_match(gen_tool['args'], ref_tool['args']):
                        score += 1.0
                    break
        
        # 3. Sequence order bonus
        if len(gen_tools) == len(ref_tools):
            if all(g['name'] == r['name'] for g, r in zip(gen_tools, ref_tools)):
                score += 2.0
        
        # Scale reward
        final_reward = score * 10.0
        
        # Add small noise for exploration
        noise = torch.randn(1).item() * 2.0
        final_reward += noise
        
        return max(final_reward, 0.0)  # Ensure non-negative
    
    def _args_match(self, args1: dict, args2: dict) -> bool:
        """Check if tool arguments roughly match"""
        # Simple string matching (can be improved)
        str1 = json.dumps(args1, sort_keys=True)
        str2 = json.dumps(args2, sort_keys=True)
        return str1 == str2
    
    # ------------------------------------------------------------------------
    # FORMATTING
    # ------------------------------------------------------------------------
    
    def format_trajectory(self, trajectory: List[Dict]) -> str:
        """Convert trajectory to text format"""
        text = ""
        for step in trajectory:
            text += f"<thought>{step['thought']}</thought>\n"
            text += f"<action>{step['action']}</action>\n"
            text += f"<observation>{step['observation']}</observation>\n"
        return text.strip()
    
    # ------------------------------------------------------------------------
    # MAIN GENERATION OVERRIDE
    # ------------------------------------------------------------------------
    
    def _generate_and_score_completions(self, inputs: List):
        """
        Override the internal TRL method to use trajectory generation.
        This is called by _prepare_inputs during training.
        """
        print(f"\n{'='*60}")
        print(f"🔄 Generating trajectories for {len(inputs)} prompts")
        print(f"{'='*60}")
        
        # Extract prompts
        prompts = [x['prompt'] for x in inputs]
        original_prompts = copy.deepcopy(prompts)
        
        # Create repeated prompts for multiple generations
        prompts_repeated = []
        inputs_repeated = []
        
        for i in range(len(prompts)):
            for gen_idx in range(self.args.num_generations):
                prompts_repeated.append(prompts[i])
                inputs_repeated.append(inputs[i])
        
        print(f"Total generations: {len(prompts_repeated)}")
        
        # Generate all trajectories
        all_trajectories = []
        
        for idx, prompt in enumerate(prompts_repeated):
            gen_num = (idx % self.args.num_generations) + 1
            temp = 0.7 + (gen_num * 0.1)  # Vary temperature
            
            print(f"\n📝 Generating {idx+1}/{len(prompts_repeated)} (temp={temp:.2f})")
            
            trajectory = self.generate_trajectory(prompt, temperature=temp)
            all_trajectories.append(trajectory)
            
            print(f"   ✅ Generated {len(trajectory)} steps")
        
        # Convert trajectories to text
        completion_texts = [self.format_trajectory(t) for t in all_trajectories]
        
        # Calculate rewards
        print(f"\n💰 Computing rewards...")
        rewards = self._calculate_rewards(
            inputs_repeated,
            original_prompts,
            completion_texts
        )
        
        print(f"✅ Rewards: min={rewards.min():.2f}, max={rewards.max():.2f}, mean={rewards.mean():.2f}")
        
        # Tokenize trajectories
        print(f"\n🔤 Tokenizing...")
        all_trajectory_ids = []
        
        for traj_text in completion_texts:
            traj_inputs = self.processing_class(
                traj_text,
                return_tensors='pt',
                padding=False,
                add_special_tokens=False
            )
            all_trajectory_ids.append(traj_inputs['input_ids'].squeeze(0))
        
        # Pad trajectories
        from trl.trainer.utils import pad
        completion_ids = pad(
            all_trajectory_ids,
            padding_value=self.processing_class.pad_token_id,
            padding_side="right"
        )
        completion_ids = completion_ids.to(self.model.device)
        completion_mask = (completion_ids != self.processing_class.pad_token_id).float()
        
        print(f"✅ Tokenized: shape={completion_ids.shape}")
        
        # Return in the format expected by TRL
        print(f"{'='*60}\n")
        
        return {
            'completions': completion_texts,
            'completion_ids': [completion_ids[i] for i in range(len(completion_ids))],
            'rewards': rewards,
        }



# ============================================================================
# TRAINING SCRIPT
# ============================================================================
def main():
    # -------------------------------------------------------------------------
    # 1. CONFIGURATION
    # -------------------------------------------------------------------------
    
    model_name = "path/to/your/pretrained/model/checkpoint"  # Replace with your model checkpoint path
    train_data_path = "path/to/your/training/data.json"  # Replace with your training data path
    output_dir = "./grpo_output"
    
    # -------------------------------------------------------------------------
    # 2. LOAD MODEL & TOKENIZER
    # -------------------------------------------------------------------------
    
    print("="*60)
    print("LOADING MODEL")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Tokenizer loaded")
    
    # Load model in bfloat16 for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
        low_cpu_mem_usage=True,
    )
    
    print(f"✅ Model loaded")
    
    # -------------------------------------------------------------------------
    # 3. APPLY LORA
    # -------------------------------------------------------------------------
    
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()
    
    print(f"✅ LoRA applied")
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Trainable: {trainable_params:,} / {total_params:,}")
    print(f"📊 Percentage: {100 * trainable_params / total_params:.2f}%")
    
    if trainable_params == 0:
        raise RuntimeError("❌ No trainable parameters!")
    
    print("="*60 + "\n")
    
    # -------------------------------------------------------------------------
    # 4. LOAD DATASET
    # -------------------------------------------------------------------------
    
    print("="*60)
    print("LOADING DATASET")
    print("="*60)
    
    dataset = AgenticDataset(train_data_path)
    print(f"✅ Dataset loaded: {len(dataset)} examples")
    print("="*60 + "\n")
    
    # -------------------------------------------------------------------------
    # 5. INITIALIZE TOOLS
    # -------------------------------------------------------------------------
    
    tool_executor = ToolExecutor()
    # TODO: Add your actual tools here
    # tool_executor.tools['search'] = your_search_function
    # tool_executor.tools['calculate'] = your_calc_function
    
    # -------------------------------------------------------------------------
    # 6. CONFIGURE TRAINING
    # -------------------------------------------------------------------------
    
    training_args = GRPOConfig(
        output_dir=output_dir,
        
        # Training
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        
        # Precision
        bf16=True,
        fp16=False,
        
        # GRPO specific
        num_generations=4,
        max_completion_length=512,
        max_prompt_length=512,
        beta=0.04,
        temperature=0.7,
        
        # Optimization
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        
        # Logging
        logging_steps=5,
        logging_first_step=True,
        
        # Saving
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        
        # Other
        seed=42,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Num generations: {training_args.num_generations}")
    print(f"Mixed precision: bf16={training_args.bf16}")
    print("="*60 + "\n")
    
    # -------------------------------------------------------------------------
    # 7. INITIALIZE TRAINER
    # -------------------------------------------------------------------------
    
    trainer = SimpleAgenticGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        tool_executor=tool_executor,
        max_trajectory_steps=5,
        stop_strings=["final_answer", "done", "complete"],
    )
    
    # -------------------------------------------------------------------------
    # 8. TRAIN
    # -------------------------------------------------------------------------
    
    print("="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    trainer.train()
    
    # -------------------------------------------------------------------------
    # 9. SAVE MODEL
    # -------------------------------------------------------------------------
    
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Model saved to {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()