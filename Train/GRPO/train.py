import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import torch
import torch.nn.functional as F
import json
from typing import List, Dict, Any
import re
from transformers import pipeline, TextStreamer
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from FineTuning.GRPO.reward import Reward
class GRPODataset(Dataset):
    def __init__(self, raw_data):
        self.data = raw_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item.get('messages', [])
        
        system_prompt = f"""You are an Enterprise agent.
           # Instructions:
           -  You always respond with a JSON object
           - Generate a thought based on the user query
           - Based on the thought select the most appropriate tool call from the list of available tools and output the tool_name
           - Provide the tool_arguments of the tool extracted from the user query
           - Scrictly output the thought, tool and final_answer if any in the provided DICT Format only
           - If you think you have the answer to the user query, or the task is executed, provide final answer, else keep final answer empty
           # Output Format:
           PROVIDE ONLY THE THOUGHT AND TOOLS, NOTHING ELSE
           {{
               "thought": ......,
               "tool": {{
                   "tool_name": ......,
                   "tool_arguments": .......
               }}
            
               "final_answer": ....
           }}
           """
        
        # Extract only system and user messages as prompt (input to model)
        prompt_messages = []
        query = ""
        ground_truth = []
        for msg in messages:
            if msg.get('role') in ['system', 'user']:
                if msg.get('role') == 'system':
                    # Prepend system_prompt to existing system message content
                    modified_msg = msg.copy()
                    modified_msg['content'] = system_prompt + "\n\n" + msg.get('content', '')
                    prompt_messages.append(modified_msg)
                else:
                    # Keep user messages as-is
                    query = msg.get("content")
                    prompt_messages.append(msg)
            else:
                ground_truth.append(msg)
                
        # Keep full messages as ground truth for reward computation
        return {
            'prompt': prompt_messages,
            'ground_truth': ground_truth,
            'query': query
        }

def custom_collate_fn(batch):
    # batch is a list of dicts with keys 'prompt' and 'ground_truth'
    prompts = [item['prompt'] for item in batch]
    ground_truths = [item['ground_truth'] for item in batch]
    query = [item['query'] for item in batch]
    
    return {
        'prompt': prompts,
        'ground_truth': ground_truths,
        'query': query
    }
class CaptureStreamer(TextStreamer):
    """
    Custom TextStreamer that captures output instead of printing.
    Synchronous - no threading overhead.
    
    IMPORTANT: Must create a NEW instance for each generation, or call reset() properly.
    """
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True, **kwargs):
        super().__init__(
            tokenizer, 
            skip_prompt=skip_prompt,
            skip_special_tokens=skip_special_tokens,
            **kwargs
        )
        self.generated_text = ""
        self._is_first_generation = True
    
    def on_finalized_text(self, text: str, stream_end: bool = False):
        """
        Called by TextStreamer when text chunk is finalized.
        Override to capture instead of print.
        """
        self.generated_text += text
        # Optional: Uncomment to see generation in real-time
        # print(text, end='', flush=True)
    
    def reset(self):
        """
        Reset captured text for reuse.
        CRITICAL: Also reset internal TextStreamer state.
        """
        self.generated_text = ""
        # Reset internal state of TextStreamer
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True  # CRITICAL: Reset prompt skip flag
    
    def end(self):
        """Called at the end of generation"""
        pass
class ARTIST_ExGRPO_Trainer:
    def __init__(self, model, tokenizer, dataset, tools, group_size=8, max_rollout_steps=10, device='cuda', 
                 gt_mix_ratio=0.5, use_policy_shaping=True):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.tools = tools
        self.group_size = group_size
        self.max_rollout_steps = max_rollout_steps
        self.device = device
        
        # ExGRPO specific parameters
        self.gt_mix_ratio = gt_mix_ratio  # 0.5 = 50% GT, 50% model rollouts
        self.use_policy_shaping = use_policy_shaping
        self.policy_shaping_beta = 0.1  # For importance sampling correction
        
        self.generation_config = {
            'max_new_tokens': 1024,
            'do_sample': True,
            'temperature': 0.6,
            'top_p': 0.95,
            'top_k': 20,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        self.streamer = CaptureStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        # Optimizer & scheduler setup could be added here
    def generate_text(self, prompt_messages: List[Dict], use_streaming: bool = True) -> str:
        """
        Generate text using the exact pattern: 
        model.generate(**tokenizer(text, return_tensors='pt').to("cuda"), streamer=...)
        
        Args:
            prompt_messages: List of message dicts
            use_streaming: Whether to use TextStreamer (useful for debugging/monitoring)
        
        Returns:
            Generated text (prompt removed by streamer)
        """
        # Format prompt using chat template
        text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # CRITICAL: Generate with torch.no_grad() and model in eval mode
        original_training_mode = self.model.training
        self.model.eval()
        
        with torch.no_grad():
            if use_streaming:
                # Reset streamer for new generation
                self.streamer.reset()
                
                # Use exact pattern from your example
                _ = self.model.generate(
                    **self.tokenizer(text, return_tensors='pt').to(self.device),
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    streamer=self.streamer  # CaptureStreamer captures output
                )
                
                # Get captured text (prompt already skipped by skip_prompt=True)
                generated_text = self.streamer.generated_text.strip()
                
            else:
                # Fast generation without streaming (recommended for training)
                inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
                
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode only the generated part
                input_length = inputs['input_ids'].shape[1]
                if output_ids.shape[1] > input_length:
                    generated_ids = output_ids[:, input_length:]
                    generated_text = self.tokenizer.decode(
                        generated_ids[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    ).strip()
                else:
                    generated_text = ""
        
        # Restore training mode
        if original_training_mode:
            self.model.train()
        
        # Additional cleanup for role markers (just in case)
        role_markers = [
            "assistant\n", "Assistant:", "assistant:",
            "<|im_start|>assistant\n", "<|assistant|>",
            "### Response:", "### Assistant:",
        ]
        
        for marker in role_markers:
            if generated_text.startswith(marker):
                generated_text = generated_text[len(marker):].strip()
                break
        
        return generated_text
    def parse_json(self, json_str):
        # Remove code block markers
        if json_str.startswith("```"):
            json_str = json_str[len("```json"):].strip()
        if json_str.endswith("```"):
            json_str = json_str[:-3].strip()
        try:
            parser = JsonOutputParser()
            data = parser.parse(json_str)
            if(isinstance(data, str)):
                data = json.dump(data)
            return data
        except Exception as e:
            # print("Parsing failed:", e)
            return ""
  
    def sample_rollouts(self, prompt, num_rollouts, ground_truth_messages=None):
        """
        Sample rollouts with interventions.
        Mix of interventional and exploratory rollouts.
        """
        rollouts = []
        
        num_intervention = int(num_rollouts * 0.7)
        num_exploration = num_rollouts - num_intervention
        original_training_mode = self.model.training
        self.model.eval()
        # Disable gradients during generation
        with torch.no_grad():
            # Intervention rollouts
            for _ in range(num_intervention):
                trajectory = self.generate_trajectory(
                    prompt,
                    ground_truth_messages=ground_truth_messages,
                    use_intervention=True
                )
                rollouts.append({
                    'prompt': prompt,
                    'trajectory': trajectory
                })
            
            # FIXED: Add exploration rollouts
            for _ in range(num_exploration):
                trajectory = self.generate_trajectory(
                    prompt,
                    ground_truth_messages=ground_truth_messages,
                    use_intervention=False  # No intervention for exploration
                )
                rollouts.append({
                    'prompt': prompt,
                    'trajectory': trajectory
                })
         # Restore training mode
        if original_training_mode:
            self.model.train()
        return rollouts
    def _parse_thinking(self, text: str):
        if "<think>" in text or "</think>" in text:
            start = text.find("<think>")
            end = text.find("</think>") + len("</think>")
            thinking = text[start:end]
            remainder = text[end:].strip()
            return thinking, remainder
        return "", text
    # def sample_mixed_rollouts(self, prompt, ground_truth_messages, num_total_rollouts):
    #     """
    #     ExGRPO: Sample mixed group with ground truth + model rollouts
        
    #     Args:
    #         prompt: Initial prompt
    #         ground_truth_messages: GT trajectory
    #         num_total_rollouts: Total rollouts in group (e.g., 8)
        
    #     Returns:
    #         List of mixed rollouts
    #     """
    #     rollouts = []
        
    #     # Calculate split
    #     num_gt = max(1, int(num_total_rollouts * self.gt_mix_ratio))
    #     num_model = num_total_rollouts - num_gt
        
    #     original_training_mode = self.model.training
    #     self.model.eval()
        
    #     with torch.no_grad():
    #         # Add ground truth rollouts
    #         for _ in range(num_gt):
    #             gt_rollout = self.create_gt_rollout(prompt, ground_truth_messages)
    #             rollouts.append({
    #                 'prompt': prompt,
    #                 'trajectory': gt_rollout,
    #                 'is_ground_truth': True
    #             })
            
    #         # Add model rollouts (pure exploration)
    #         for _ in range(num_model):
    #             model_trajectory = self.generate_trajectory_without_intervention(prompt)
    #             rollouts.append({
    #                 'prompt': prompt,
    #                 'trajectory': model_trajectory,
    #                 'is_ground_truth': False
    #             })
        
    #     # Restore training mode
    #     if original_training_mode:
    #         self.model.train()
        
    #     return rollouts
    def generate_trajectory_without_intervention(self, prompt):
        """
        Generate trajectory WITHOUT interventions - pure exploration
        This is for model rollouts in ExGRPO
        """
        trajectory = []
        conversation = prompt[:]
        
        step_idx = 0
        reached_final_answer = False
        
        for step in range(self.max_rollout_steps):
            # Generate using model (no intervention)
            raw_text = self.generate_text(conversation)
            
            # Parse response
            thinking, clean_text = self._parse_thinking(raw_text)
            parsed_clean_text = self.parse_json(clean_text)
            print(f"Model Output: {parsed_clean_text}")
            if not parsed_clean_text:
                parsed_clean_text = self.parse_tool_calls(clean_text)
            
            # Extract predicted tool call
            predicted_tool = parsed_clean_text.get("tool", {})
            predicted_tool_name = predicted_tool.get("tool_name")
            predicted_tool_args = predicted_tool.get("tool_arguments", {})
            
            # Check for final answer
            if parsed_clean_text.get("final_answer"):
                conversation.append({'role': 'assistant', 'content': parsed_clean_text.get("thought", "")})
                trajectory.append({
                    'role': 'assistant',
                    'content': parsed_clean_text.get("thought", ""),
                    'final_answer': parsed_clean_text["final_answer"]
                })
                reached_final_answer = True
                break
            
            # Use predicted tool/args AS-IS (no correction)
            thought_to_use = parsed_clean_text.get("thought", "")
            tool_to_execute = predicted_tool_name
            args_to_execute = predicted_tool_args
            
            # Add thought to conversation
            conversation.append({'role': 'assistant', 'content': thought_to_use})
            
            if tool_to_execute:
                # Add tool call
                tool_call_msg = {
                    'role': 'assistant',
                    'tool_calls': [{
                        'type': 'function',
                        'function': {
                            'name': tool_to_execute,
                            'arguments': args_to_execute
                        }
                    }]
                }
                conversation.append(tool_call_msg)
                
                # Execute tool (may fail if tool/args incorrect)
                try:
                    tool_output = self.execute_tool({
                        'tool_name': tool_to_execute,
                        'arguments': args_to_execute
                    })
                except Exception as e:
                    tool_output = f"Error: {str(e)}"
                
                # Add tool result
                tool_result_msg = {
                    'role': 'tool',
                    'name': tool_to_execute,
                    'content': json.dumps(tool_output) if isinstance(tool_output, dict) else str(tool_output)
                }
                conversation.append(tool_result_msg)
                
                # Add to trajectory
                trajectory.append({
                    'role': 'assistant',
                    'content': thought_to_use,
                    'tool_call': {
                        'name': tool_to_execute,
                        'arguments': args_to_execute
                    },
                    'observation': tool_output
                })
                
                step_idx += 1
            else:
                trajectory.append({
                    'role': 'assistant',
                    'content': thought_to_use
                })
                break
        
        return {
            'steps': trajectory,
            'reached_final_answer': reached_final_answer,
            'steps_used': step_idx
        }
    def create_gt_rollout(self, prompt, ground_truth_messages):
        """
        Create a rollout from ground truth trajectory
        Ground truth must be in same format as model-generated trajectories
        """
        # Ensure GT is in the right format
        gt_steps = []
        
        for msg in ground_truth_messages:
            if isinstance(msg, dict):
                step_entry = {}
                
                # Handle assistant messages with content
                if msg.get('role') == 'assistant' and msg.get('content'):
                    step_entry['role'] = 'assistant'
                    step_entry['content'] = msg['content']
                
                # Handle tool calls
                if msg.get('tool_calls'):
                    tc = msg['tool_calls'][0] if isinstance(msg['tool_calls'], list) else msg['tool_calls']
                    if 'function' in tc:
                        step_entry['tool_call'] = {
                            'name': tc['function']['name'],
                            'arguments': tc['function'].get('arguments', {})
                        }
                
                # Handle tool results
                if msg.get('role') == 'tool':
                    # Merge with previous tool call
                    if gt_steps and 'tool_call' in gt_steps[-1]:
                        gt_steps[-1]['observation'] = msg.get('content', '')
                        continue
                
                # Handle final answer
                if msg.get('final_answer'):
                    step_entry['final_answer'] = msg['final_answer']
                
                if step_entry:
                    gt_steps.append(step_entry)
        
        return {
            'steps': gt_steps,
            'reached_final_answer': self._check_gt_has_final_answer(ground_truth_messages),
            'is_ground_truth': True
        }
    def _check_gt_has_final_answer(self, ground_truth_messages):
        """Check if ground truth expects a final answer"""
        if not ground_truth_messages:
            return False
        
        # Check last few messages for final answer indicator
        for msg in reversed(ground_truth_messages[-3:]):  # Check last 3 messages
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Check for final answer patterns
                if content and (
                    not msg.get('tool_calls') and  # No tool calls
                    len(content) > 10  # Has substantial content
                ):
                    return True
        return False
    def _extract_gt_tools_sequence(self, ground_truth_messages):
        """Extract sequence of ground truth tool calls from messages"""
        gt_tools = []
        for msg in ground_truth_messages:
            if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                for tool_call in msg['tool_calls']:
                    if tool_call.get('type') == 'function':
                        func = tool_call['function']
                        gt_tools.append({
                            'name': func.get('name'),
                            'arguments': func.get('arguments', {})
                        })
        return gt_tools
    def _extract_gt_thought_sequence(self, ground_truth_messages):
        """Extract sequence of ground truth tool calls from messages"""
        gt_thought = []
        for msg in ground_truth_messages:
            if msg.get('role') == 'assistant' and 'content' in msg:
               gt_thought.append(msg.get("content"))
        return gt_thought

    def _check_args_match(self, predicted_args, gt_args):
        """Check if predicted arguments match ground truth"""
        if not isinstance(predicted_args, dict) or not isinstance(gt_args, dict):
            return False
        
        # Check if all GT keys are present with correct values
        for key, value in gt_args.items():
            if key not in predicted_args or predicted_args[key] != value:
                return False
        
        return True

    
    def execute_tool(self, call):
        """
        Use the tools to interact with environment.
        call = {'tool_name': ..., 'args': {...}}
        """
        try:
            outputs = self.tools.get_tool_context([{"tool_name": call['tool_name'], "tool_arguments": call.get('arguments', {})}])
            return outputs
        except Exception as e:
            return {'error': str(e)}
    
   
    
    def apply_masking(self, inputs, mask):
        """
        Mask loss contributions on tokens corresponding to environment outputs.
        'mask' is a binary mask with 1 on tokens to compute loss.
        """
        # This will depend on your loss calc code, e.g. multiply per-token loss by mask
        pass
    def compute_masked_loss_exgrpo(self, rollouts, advantages, importance_weights):
        """
        ExGRPO loss with policy shaping for ground truth trajectories
        """
        device = next(self.model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        total_tokens = 0
        
        # Ensure advantages and weights are tensors
        if not isinstance(advantages, torch.Tensor):
            advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        if not isinstance(importance_weights, torch.Tensor):
            importance_weights = torch.tensor(importance_weights, dtype=torch.float32, device=device)
        
        for idx, rollout in enumerate(rollouts):
            advantage = advantages[idx]
            weight = importance_weights[idx]
            is_gt = rollout.get('is_ground_truth', False)
            
            trajectory = rollout['trajectory']
            if isinstance(trajectory, dict):
                trajectory = trajectory['steps']
            
            prompt = rollout['prompt']
            
            # Reconstruct conversation
            conversation = prompt[:]
            for step in trajectory:
                if step.get('content'):
                    conversation.append({'role': 'assistant', 'content': step['content']})
                if step.get('tool_call'):
                    conversation.append({
                        'role': 'assistant',
                        'tool_calls': [{
                            'type': 'function',
                            'function': {
                                'name': step['tool_call']['name'],
                                'arguments': step['tool_call']['arguments']
                            }
                        }]
                    })
                if step.get('observation'):
                    conversation.append({
                        'role': 'tool',
                        'name': step['tool_call']['name'],
                        'content': json.dumps(step['observation']) if isinstance(step['observation'], dict) else str(step['observation'])
                    })
            
            # Tokenize
            text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Forward pass with current policy
            with torch.set_grad_enabled(True):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Compute log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                dim=-1,
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Create and apply mask
            mask = self._create_token_mask(conversation, shift_labels.shape[1])
            mask = mask.to(device)
            
            # Normalize by number of masked tokens
            num_masked_tokens = mask.sum()
            if num_masked_tokens > 0:
                masked_log_prob = (token_log_probs * mask).sum() / num_masked_tokens
                
                # Apply policy shaping for ground truth trajectories
                if is_gt and self.use_policy_shaping:
                    # Policy shaping: f(w) = w / (w + β)
                    # This prevents over-exploitation of GT
                    # Assume uniform prior (ratio ≈ 1.0 for simplicity)
                    ratio = torch.exp(masked_log_prob.clamp(max=0))  # Clamp for stability
                    shaped_weight = ratio / (ratio + self.policy_shaping_beta)
                    shaped_weight = shaped_weight.detach()  # Don't backprop through shaping
                else:
                    shaped_weight = 1.0
                
                # GRPO policy gradient loss
                policy_loss = -advantage * masked_log_prob * weight * shaped_weight
                
                total_loss = total_loss + policy_loss
                total_tokens += num_masked_tokens.item()
        
        # Average over batch
        if len(rollouts) > 0 and total_tokens > 0:
            avg_loss = total_loss / len(rollouts)
        else:
            avg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return avg_loss
    def train(self, epochs, batch_size):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-7)
        
        for epoch in range(epochs):
            print(f"\n{'='*50}\nEpoch {epoch+1}/{epochs}\n{'='*50}")
            self.model.train()
            
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                batch_prompts = batch['prompt']
                batch_ground_truths = batch['ground_truth']
                batch_query = batch["query"]
                
                groups = []
                rewards_list = []
                
                # Generate mixed rollouts (GT + model)
                for prompt, ground_truth, query in zip(batch_prompts, batch_ground_truths, batch_query):
                    try:
                        # ExGRPO: Mixed group sampling
                        rollouts = self.sample_mixed_rollouts(
                            prompt,
                            ground_truth,
                            self.group_size
                        )
                        
                        # Compute rewards
                        rewards = [self.compute_reward(r, ground_truth, query) for r in rollouts]
                        
                        groups.append(rollouts)
                        rewards_list.append(rewards)
                        
                    except Exception as e:
                        print(f"Error in batch {batch_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                if not groups:
                    print("No valid groups, skipping batch")
                    continue
                
                # Flatten
                flat_rollouts = [r for group in groups for r in group]
                flat_rewards = [r for rewards in rewards_list for r in rewards]
                
                # Compute advantages
                advantages = self.compute_advantages(flat_rollouts, flat_rewards)
                importance_weights = self.compute_importance_weights(flat_rollouts)
                
                # Log statistics
                print(f"\n[Epoch {epoch+1} | Batch {batch_idx}]")
                print(f"Rewards - Min: {min(flat_rewards):.2f}, Max: {max(flat_rewards):.2f}, "
                      f"Mean: {sum(flat_rewards)/len(flat_rewards):.2f}, "
                      f"Std: {torch.tensor(flat_rewards).std().item():.2f}")
                print(f"GT rollouts: {sum(1 for r in flat_rollouts if r.get('is_ground_truth', False))}/{len(flat_rollouts)}")
                print(f"Advantages - Mean: {advantages.mean().item():.6f}, Std: {advantages.std().item():.4f}")
                
                # Ensure model is in training mode
                self.model.train()
                
                # Compute ExGRPO loss
                self.optimizer.zero_grad()
                masked_loss = self.compute_masked_loss_exgrpo(flat_rollouts, advantages, importance_weights)
                
                print(f"Loss: {masked_loss.item():.6f}")
                
                # Backprop
                masked_loss.backward()
               # Verify gradients are flowing
                grad_norms = []
                zero_grad_params = 0
                for name, param in self.model.named_parameters():
                    if param.grad is not None and param.requires_grad:
                        grad_norm = param.grad.norm().item()
                        grad_norms.append(grad_norm)
                        if grad_norm < 1e-8:
                            zero_grad_params += 1

                if grad_norms:
                    avg_grad = sum(grad_norms) / len(grad_norms)
                    print(f"Avg Grad: {avg_grad:.6f} | Zero Grads: {zero_grad_params}/{len(grad_norms)}")
                else:
                    print("WARNING: No gradients computed!")

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
    
    def wrap_messages_into_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert message list into a string prompt using your chat formatting
        """
        prompt_str = ""
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                prompt_str += f"[SYSTEM]\n{content}\n\n"
            elif role == "user":
                prompt_str += f"[USER]\n{content}\n\n"
            elif role == "assistant":
                prompt_str += f"[ASSISTANT]\n{content}\n\n"
            elif role == "tool":
                prompt_str += f"[TOOL]\n{content}\n\n"
        return prompt_str

    # Parse tool calls from the model's output string
    def parse_tool_calls(self, text: str) -> Dict[str, Any]:
        """
        Parse tool calls and return structured dict with thought, tool, and final_answer.
        Returns format: {
            "thought": "...",
            "tool": {"tool_name": "...", "tool_arguments": {...}},
            "final_answer": "..."
        }
        """
        result = {
            "thought": "",
            "tool": {
                "tool_name": None,
                "tool_arguments": {}
            },
            "final_answer": ""
        }
        
        # Pattern 0: Try parsing entire text as direct JSON first
        try:
            cleaned_text = text.strip()
            data = json.loads(cleaned_text)
            
            # If already in the desired format
            if isinstance(data, dict):
                result["thought"] = data.get("thought", "")
                result["final_answer"] = data.get("final_answer", "")
                
                if "tool" in data:
                    tool_info = data["tool"]
                    result["tool"]["tool_name"] = tool_info.get("tool_name")
                    result["tool"]["tool_arguments"] = tool_info.get("tool_arguments", {})
                
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Pattern 1: <tool_call> tags
        pattern_tool_call = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.findall(pattern_tool_call, text, re.DOTALL)
        
        if matches:
            try:
                data = json.loads(matches[0].strip())
                tool_name = data.get('name') or data.get('tool_name') or data.get('action')
                tool_args = data.get('arguments') or data.get('tool_arguments') or data.get('action_input', {})
                
                result["tool"]["tool_name"] = tool_name
                result["tool"]["tool_arguments"] = tool_args if isinstance(tool_args, dict) else {'input': tool_args}
                
                # Extract thought from text before tool_call tag
                thought_text = text[:text.find('<tool_call>')].strip()
                result["thought"] = thought_text
                
                return result
            except json.JSONDecodeError:
                pass
        
        # Pattern 2: JSON code blocks
        pattern_json = r'``````'
        matches = re.findall(pattern_json, text, re.DOTALL)
        
        if matches:
            try:
                data = json.loads(matches[0].strip())
                
                # Check for Final Answer
                if data.get('action', '').lower() == 'final answer':
                    result["final_answer"] = data.get('action_input', '')
                    # Extract thought from text before code block
                    thought_text = text[:text.find('```')].strip()
                    result["thought"] = thought_text
                else:
                    tool_name = data.get('name') or data.get('tool_name') or data.get('action')
                    tool_args = data.get('arguments') or data.get('tool_arguments') or data.get('action_input', {})
                    
                    result["tool"]["tool_name"] = tool_name
                    result["tool"]["tool_arguments"] = tool_args if isinstance(tool_args, dict) else {'input': tool_args}
                    
                    # Extract thought from text before code block
                    thought_text = text[:text.find('```json')].strip()
                    result["thought"] = thought_text
                
                return result
            except json.JSONDecodeError:
                pass
        
        # If no patterns matched, treat entire text as thought
        result["thought"] = text.strip()
        return result




    # Compute reward for a single turn based on correctness, format, etc.
    def compute_turn_level_reward(self, turn_data: Dict[str, Any], ground_truth_turn: Dict[str, Any]) -> float:
        """
        Simple heuristic reward: +0.2 for presence of Thought, +0.3 for correct tool call,
        +0.2 for correct arguments, +0.3 for format adherence, etc.
        """
        reward = 0.0

        content = turn_data.get("content", "")

        if "Thought:" in content:
            reward += 0.2

        # Check for tool call correctness
        if "tool_calls" in turn_data and turn_data["tool_calls"]:
            pred_tool = turn_data["tool_calls"][0]
            gt_tool = ground_truth_turn["tool_calls"][0] if ground_truth_turn and "tool_calls" in ground_truth_turn else None

            if gt_tool and pred_tool["tool_name"] == gt_tool["function"]["name"]:
                reward += 0.3
                pred_args = pred_tool.get("tool_arguments", {})
                gt_args = gt_tool["function"].get("arguments", {})

                matching_args = sum(1 for k, v in pred_args.items() if gt_args.get(k) == v)
                total_args = len(gt_args)
                if total_args > 0:
                    reward += 0.2 * (matching_args / total_args)

        # Format adherence
        if "<tool_call>" in content and "</tool_call>" in content:
            reward += 0.15
        if "Final Answer:" in content:
            reward += 0.15

        return reward

 
    def compute_outcome_level_reward(self, trajectory, ground_truth_messages, query):
        """
        Compute trajectory-level reward based on correctness
        """
        if isinstance(trajectory, dict):
            trajectory = trajectory['steps']
        
        # Extract ground truth information
        gt_tools_sequence = self._extract_gt_tools_sequence(ground_truth_messages)
        gt_has_final_answer = self._check_gt_has_final_answer(ground_truth_messages)
        
        # Check if trajectory reached final answer
        reached_final_answer = False
        for step in trajectory:
            if step.get('final_answer'):
                reached_final_answer = True
                break
        
        # Binary reward: 1.0 for correct completion, 0.0 otherwise
        if reached_final_answer and gt_has_final_answer:
            return 1.0
        elif not gt_has_final_answer:
            # Use tool sequence matching for intermediate steps
            correct_tools = 0
            predicted_tools = []
            
            for step in trajectory:
                if step.get('tool_call'):
                    predicted_tools.append(step['tool_call']['name'])
            
            # Calculate overlap with ground truth
            for i, (pred, gt) in enumerate(zip(predicted_tools, gt_tools_sequence)):
                if pred == gt.get('name'):
                    correct_tools += 1
            
            if len(gt_tools_sequence) > 0:
                return correct_tools / len(gt_tools_sequence)
            else:
                return 0.0
        else:
            return 0.0  # No answer when expected

        
    def compute_reward(self, rollout_dict, ground_truth_messages, query):
        """
        Compute reward for rollout
        Ground truth gets perfect reward
        """
        trajectory = rollout_dict['trajectory']
        is_gt = rollout_dict.get('is_ground_truth', False)
        
        if is_gt:
            # Ground truth gets perfect reward
            return 1.0
        else:
            # Use your existing reward computation
            return self.compute_outcome_level_reward(trajectory, ground_truth_messages, query)

    # Prepare input prompts from message list
    def prepare_prompts(self, messages: List[Dict], tools_metadata: str) -> str:
        """
        Convert your message list into a prompt string, including tools list and instructions
        """
        prompt_str = ""
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                prompt_str += f"[SYSTEM]\n{content}\n\n"
            elif role == "user":
                prompt_str += f"[USER]\n{content}\n\n"
            elif role == "assistant":
                prompt_str += f"[ASSISTANT]\n{content}\n\n"
            elif role == "tool":
                prompt_str += f"[TOOL]\n{content}\n\n"
        # Append instruction
        prompt_str += tools_metadata
        return prompt_str

    # Prepare model inputs for generation
    def prepare_inputs(self, prompt: str):
        """
        Tokenize prompt and prepare tensors for generation
        """
        inputs = self.tokenizer(input_text, return_tensors='pt')
        inputs = {k: v.to(model.device) if v.dtype != torch.float else v.to(model.device) for k, v in inputs.items()}
        # or even simpler, if your model.device points to the right device:
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}

        return inputs

    def compute_advantages(self, rollouts, rewards):
        """
        Compute normalized group relative advantages as TENSORS
        """
        if len(rewards) == 0:
            return []
        
        # Convert to tensors immediately
        device = next(self.model.parameters()).device
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        # Group relative baseline
        group_avg = rewards_tensor.mean()
        advantages = rewards_tensor - group_avg
        
        # Normalize advantages for stability
        if len(advantages) > 1:
            std = advantages.std()
            if std > 1e-8:
                advantages = advantages / (std + 1e-8)
        
        return advantages  # Return tensor, not list of floats

    def compute_importance_weights(self, rollouts):
        """
        Compute importance sampling weights (simplified for on-policy)
        """
        device = next(self.model.parameters()).device
        # For simplified on-policy GRPO, weights are uniform
        weights = torch.ones(len(rollouts), dtype=torch.float32, device=device)
        return weights
    def compute_masked_loss(self, rollouts, advantages, importance_weights):
        """
        Compute GRPO policy gradient loss with KL regularization
        """
        device = next(self.model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)  # No requires_grad needed
        total_kl = torch.tensor(0.0, device=device)
        total_tokens = 0
        
        # Ensure advantages and weights are tensors
        if not isinstance(advantages, torch.Tensor):
            advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        if not isinstance(importance_weights, torch.Tensor):
            importance_weights = torch.tensor(importance_weights, dtype=torch.float32, device=device)
        
        for idx, rollout in enumerate(rollouts):
            advantage = advantages[idx]
            weight = importance_weights[idx]
            
            trajectory = rollout['trajectory']
            if isinstance(trajectory, dict):
                trajectory = trajectory['steps']
            
            prompt = rollout['prompt']
            
            # Reconstruct conversation
            conversation = prompt[:]
            for step in trajectory:
                if step.get('content'):
                    conversation.append({'role': 'assistant', 'content': step['content']})
                if step.get('tool_call'):
                    conversation.append({
                        'role': 'assistant',
                        'tool_calls': [{
                            'type': 'function',
                            'function': {
                                'name': step['tool_call']['name'],
                                'arguments': step['tool_call']['arguments']
                            }
                        }]
                    })
                if step.get('observation'):
                    conversation.append({
                        'role': 'tool',
                        'name': step['tool_call']['name'],
                        'content': json.dumps(step['observation']) if isinstance(step['observation'], dict) else str(step['observation'])
                    })
            
            # Tokenize
            text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Compute log probabilities with numerical stability
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Gather log probs for actual tokens
            token_log_probs = log_probs.gather(
                dim=-1,
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Create and apply mask
            mask = self._create_token_mask(conversation, shift_labels.shape[1])
            mask = mask.to(device)
            
            # Normalize by number of masked tokens
            num_masked_tokens = mask.sum()
            if num_masked_tokens > 0:
                masked_log_prob = (token_log_probs * mask).sum() / num_masked_tokens
                
                # GRPO policy gradient loss
                policy_loss = -advantage * masked_log_prob * weight
                
                # KL divergence term (optional but recommended)
                # Compute KL with reference model if available
                # For now, add a simple entropy regularization
                kl_term = -masked_log_prob.detach()  # Simple approximation
                
                total_loss = total_loss + policy_loss
                total_kl = total_kl + kl_term
                total_tokens += num_masked_tokens.item()
        
        # Average over batch
        if len(rollouts) > 0 and total_tokens > 0:
            avg_loss = total_loss / len(rollouts)
            avg_kl = total_kl / len(rollouts)
            
            # Add KL regularization (beta hyperparameter)
            beta = 0.01  # Adjust based on your needs
            final_loss = avg_loss + beta * avg_kl
        else:
            final_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return final_loss


    def _create_token_mask(self, conversation, seq_len):
        """
        Create mask: 1 for assistant tokens, 0 for tool/system tokens
        """
        # Convert conversation to tokenized form
        full_text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize to get token boundaries
        inputs = self.tokenizer(full_text, return_tensors='pt', truncation=True, max_length=2048)
        
        # Create mask by identifying assistant message boundaries
        mask = torch.zeros(seq_len)
        
        # Build conversation incrementally to find token boundaries
        current_pos = 0
        for i, msg in enumerate(conversation):
            role = msg.get('role')
            content = msg.get('content', '')
            
            # Tokenize this message
            if role == 'assistant' and content:
                # This is an assistant message - include in loss
                msg_text = self.tokenizer.apply_chat_template(
                    conversation[:i+1],
                    tokenize=False,
                    add_generation_prompt=False
                )
                msg_tokens = self.tokenizer(msg_text, return_tensors='pt', truncation=True, max_length=2048)
                msg_len = msg_tokens['input_ids'].shape[1]
                
                # Mark assistant tokens
                if current_pos < seq_len:
                    end_pos = min(msg_len - 1, seq_len)
                    mask[current_pos:end_pos] = 1.0
                
                current_pos = msg_len - 1
            elif role in ['tool', 'system']:
                # Skip tool and system messages
                msg_text = self.tokenizer.apply_chat_template(
                    conversation[:i+1],
                    tokenize=False,
                    add_generation_prompt=False
                )
                msg_tokens = self.tokenizer(msg_text, return_tensors='pt', truncation=True, max_length=2048)
                current_pos = msg_tokens['input_ids'].shape[1] - 1
        
        return mask

    def prepare_inputs(self, flat_rollouts):
        """
        This is now handled inside compute_masked_loss
        """
        return flat_rollouts, None


if __name__ == "__main__":
    
    # Paths to model checkpoint, training data, and output directory
    model_path = "path_to_your_pretrained_model_checkpoint"  # Replace with your model path
    train_data_path = "path_to_your_training_data.json"  # Replace with your training data path
    tools_path = "path_to_your_tools_metadata.json"  # Replace with your tools metadata path
    output_dir = "path_to_save_trained_model"  # Replace with your desired output directory
    
    # Load dataset (you'll implement your own Dataset class or JSON loader)
    import json
    with open(train_data_path, "r") as f:
        raw_data = json.load(f)
    print(f"Training data length {len(raw_data)}")
    dataset = GRPODataset(raw_data)
    # Initialize tokenizer and model (can adjust for your model and config)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto",
        use_cache=False,
        
    )
    model.gradient_checkpointing_enable()

    # Initialize tools class instance (your implementation)
    from Task_Generation_sft_batch2_copy.utils.tools import Tools
    tools_instance = Tools()
    
    # Instantiate your trainer class
    trainer = ARTIST_ExGRPO_Trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        tools=tools_instance,
        group_size=8,  # Total rollouts: 4 GT + 4 model
        max_rollout_steps=3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        gt_mix_ratio=0.0,  # 50% ground truth, 50% model rollouts
        use_policy_shaping=True  # Enable policy shaping to prevent GT over-exploitation
    )
    
    # Run training loop for some epochs
    trainer.train(epochs=10, batch_size=4)
    
    # Save trained model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training complete. Model saved to {output_dir}.")