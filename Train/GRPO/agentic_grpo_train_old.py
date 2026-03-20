import torch
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import copy
import re
from torch.utils.data import DataLoader, Dataset
import random
from typing import Union, Any, Optional, List, Dict
from trl import GRPOTrainer, GRPOConfig
from trl import (
    SFTTrainer,
    SFTConfig,
    get_peft_config,
    get_quantization_config,
    ModelConfig,
    RichProgressCallback,
)
from langchain_core.output_parsers import JsonOutputParser
from trl.trainer.utils import pad
from transformers import TextStreamer
from Task_Generation_sft_batch2_copy.utils.tools import Tools
from Task_Generation_sft_batch2_copy.Factories.llm_factory import LLM_factory
from peft import LoraConfig, get_peft_model

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
        'ground_truth': ground_truth,
        'query': query
    }
class CaptureStreamer(TextStreamer):
    """
    Custom TextStreamer that captures output instead of #printing.
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
        Override to capture instead of #print.
        """
        self.generated_text += text
        # Optional: Uncomment to see generation in real-time
        # ##print(text, end='', flush=True)
    
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


class AgenticGRPOTrainer(GRPOTrainer):
    """
    Custom GRPO Trainer for agentic environments with iterative tool calling.
    Overrides the generation method to support thought-action-observation trajectories.
    """
    
    def __init__(
        self,
        model,
        reward_funcs,
        args: GRPOConfig = None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
        # Custom parameters for agentic generation
        max_trajectory_steps: int = 10,
        trajectory_stop_strings: Optional[list[str]] = None,
        use_streaming: bool = True,  # Set to True for debugging
    ):
        # Initialize parent GRPOTrainer
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            
        )
        # Monkey-patch shuffle function to no-op in multi-GPU
        if self.accelerator.num_processes > 1:
            print("⚠️ Multi-GPU: Patching shuffle to prevent crash")
            # Replace shuffle with identity function
            import trl.trainer.utils
            trl.trainer.utils.shuffle_sequence_dict = lambda x: x
        self.ref_model=copy.deepcopy(model)
        # Custom agentic parameters
        self.max_trajectory_steps = max_trajectory_steps
        self.trajectory_stop_strings = trajectory_stop_strings or [
            "</answer>", 
            "<|end_of_trajectory|>",
            "<done>"
        ]
        self.judge_llm = LLM_factory()
        self.use_streaming = use_streaming
        self.tools = Tools()
        # Initialize tokenizer reference
        self.processing_class = processing_class
        self.device = self.accelerator.device
        
        # Initialize CaptureStreamer for generation
        self.streamer = CaptureStreamer(
            self.processing_class,
            skip_prompt=True,
            skip_special_tokens=True
        )
    def _generate_trajectories_batch_parallel(self, prompts: List) -> List[List[Dict]]:
        """
        Generate trajectories in parallel batches.
        """
        batch_size = len(prompts)
        trajectories = [[] for _ in range(batch_size)]
        current_contexts = [self._format_initial_context(p) for p in prompts]
        active_mask = [True] * batch_size
        
        for step_num in range(self.max_trajectory_steps):
            # Get active trajectories
            active_indices = [i for i, active in enumerate(active_mask) if active]
            if not active_indices:
                break
            
            active_contexts = [current_contexts[i] for i in active_indices]
            self.model.eval()

            # Parallel generation
            with torch.no_grad():
                # ✅ FIX: Apply chat template to convert dicts → strings
                formatted_prompts = []
                for ctx in active_contexts:
                    
                    # Apply chat template to get string
                    formatted_text = self.processing_class.apply_chat_template(
                        ctx,
                        tokenize=False,
                        add_generation_prompt=True, 
                        enable_thinking=False
                    )
                    formatted_prompts.append(formatted_text)
                # ##print(formatted_text)
                ##print("Starting Generation")
                # Now tokenize the strings (not dicts!)
                batch_inputs = self.processing_class(
                    formatted_prompts,  # List of strings ✓
                    return_tensors="pt",
                    padding=True,
                    padding_side="left"
                ).to(self.device)
                
                # Single batched generation
                outputs = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.processing_class.pad_token_id,
                    eos_token_id=self.processing_class.eos_token_id
                )
                
                # Batch decode
                input_length = batch_inputs['input_ids'].shape[1]
                generated_ids = outputs[:, input_length:]
                generated_texts = self.processing_class.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
               
            # Parallel parsing
            self.model.train()
            parsed_steps = [
                self._parse_thought_action(text) 
                for text in generated_texts
            ]
            ##print("Generated Text: ", parsed_steps)
            # Execute tools (sequential for now, can parallelize later)
            observations = [
                self._execute_tool_call(action)
                for thought, action in parsed_steps
            ]
            ##print("Observations: ", observations)
            # Update all trajectories
            for i, idx in enumerate(active_indices):
                thought, action = parsed_steps[i]
                observation = observations[i]
                
                trajectories[idx].append({
                    "thought": thought,
                    "action": action,
                    "observation": observation,
                    "step": step_num
                })
                
                # ✅ Update context with new assistant message
                current_contexts[idx].append({
                    "role": "assistant",
                    "content": f"<thought>{thought}</thought>\n<action>{action}</action>\n<observation>{observation}</observation>"
                })
                
                # Add continuation prompt if not done
                if not self._is_task_complete(action, observation):
                    current_contexts[idx].append({
                        "role": "user",
                        "content": "Continue with the next step."
                    })
                else:
                    active_mask[idx] = False
        
        return trajectories

    def _format_initial_context(self, prompt):
        """
        Format prompt as conversation (list of message dicts).
        
        Args:
            prompt: Can be string, list of dicts, or dict
        
        Returns:
            List of message dicts
        """
        if isinstance(prompt, list):
            # Already a list of message dicts
            return prompt.copy()
        elif isinstance(prompt, dict):
            # Single message dict
            return [prompt]
        else:
            # String - wrap in user message
            return [{"role": "user", "content": str(prompt)}]

    def generate_text(self, prompt_messages: List[Dict], temperature: float = 0.7,
        do_sample: bool = True, use_streaming: bool = True) -> str:
        """
        Generate text using the exact pattern: 
        model.generate(**tokenizer(text, return_tensors='pt').to("cuda"), streamer=...)
        
        Args:
            prompt_messages: List of message dicts
            use_streaming: Whether to use TextStreamer (useful for debugging/monitoring)
        
        Returns:
            Generated text (prompt removed by streamer)
        """
        # ##print("Generating Text")
        if use_streaming is None:
            use_streaming = self.use_streaming
            
        # Format prompt using chat template
        text = self.processing_class.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking= False
        )
        # ##print("="*10)
        # ##print("Before Tokenizer Template text: ", prompt_messages)
        # ##print("="*10)
        # ##print("Tokeenizer Template text: ", text)
        # ##print("="*10)
        # CRITICAL: Generate with torch.no_grad() and model in eval mode
        original_training_mode = self.model.training
        self.model.eval()
        
        with torch.no_grad():
            if use_streaming:
                # ##print("Generating streaming text")
                # Reset streamer for new generation
                self.streamer.reset()
                
                # Use exact pattern from your example
                _ = self.model.generate(
                    **self.processing_class(text, return_tensors='pt').to(self.device),
                    max_new_tokens=1024,
                    temperature=temperature,
                    top_p=0.8,
                    top_k=20,
                    do_sample=do_sample,
                    pad_token_id=self.processing_class.pad_token_id,
                    eos_token_id=self.processing_class.eos_token_id,
                    streamer=self.streamer  # CaptureStreamer captures output
                )
                
                # Get captured text (prompt already skipped by skip_prompt=True)
                generated_text = self.streamer.generated_text.strip()
                # ##print("Generated Text: ", generated_text)
            else:
                # Fast generation without streaming (recommended for training)
                inputs = self.processing_class(text, return_tensors='pt').to(self.device)
                
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    pad_token_id=self.processing_class.pad_token_id,
                    eos_token_id=self.processing_class.eos_token_id,
                )
                # ##print("OUTPUT: ",output_ids)
                # Decode only the generated part
                input_length = inputs['input_ids'].shape[1]
                if output_ids.shape[1] > input_length:
                    generated_ids = output_ids[:, input_length:]
                    generated_text = self.processing_class.decode(
                        generated_ids[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    ).strip()
                else:
                    generated_text = ""
        
        # Restore training mode
        
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
    
    def _format_trajectory(self, trajectory: list[dict]) -> str:
        """Convert trajectory to text format for tokenization"""
        text = ""
        for step in trajectory:
            text += f"<thought>{step['thought']}</thought>"
            text += f"<action>{step['action']}</action>"
            text += f"<observation>{step['observation']}</observation>"
        return text
    
    def _trajectory_to_text(self, trajectory: list[dict]) -> str:
        """Alias for _format_trajectory for consistency"""
        return self._format_trajectory(trajectory)
    
    def _create_trajectory_mask(self, trajectories, completion_ids):
        """
        SIMPLIFIED: Don't mask anything during training.
        Train on the entire trajectory.
        """
        # Return all ones - train on everything
        mask = torch.ones_like(completion_ids, dtype=torch.float32)
        
        print(f"✅ Using full trajectory mask (no masking)")
        print(f"   Shape: {mask.shape}, all values: {mask[0][:10]}")
        
        return mask
    
    def _tokens_match(
        self, 
        tokens: torch.Tensor, 
        target: list[int]
    ) -> bool:
        """Helper to check if tokens match a target sequence"""
        if len(tokens) < len(target):
            return False
        return all(tokens[i].item() == target[i] for i in range(len(target)))
    
    def generate_trajectory(
        self, 
        prompt: str,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> list[dict]:
        """
        Generate iterative trajectory with thought-action-observation cycles.
        Uses the generate_text method with chat templates.
        
        Args:
            prompt: Initial user prompt/task
        
        Returns:
            list of dicts with keys: 'thought', 'action', 'observation', 'step'
        """
        trajectory = []
        
        # Initialize conversation with user prompt
        messages = prompt
        for step in range(self.max_trajectory_steps):
            # Generate thought + action in one go
            # Add instruction to generate structured output
            
            # Generate with chat template
            generated_text = self.generate_text(messages, temperature=temperature,  # Pass temperature
            do_sample=do_sample, use_streaming=True)
            
            # Parse thought and action from generated text
            thought, action = self._parse_thought_action(generated_text)
            
            # Execute action in environment to get observation
            observation = self._execute_tool_call(action)
            
            # Add step to trajectory
            trajectory.append({
                "thought": thought,
                "action": action,
                "observation": observation,
                "step": step
            })
            
            # Update conversation context
            # Build the complete assistant response with observation
            complete_response = (
                f"<thought>{thought}</thought>\n"
                f"<action>{action}</action>\n"
                f"<observation>{observation}</observation>"
            )
            messages.append({"role": "assistant", "content": complete_response})
            
            # Check for completion
            if self._is_task_complete(action, observation):
                break
            
            # Add continuation prompt for next step if not done
            if step < self.max_trajectory_steps - 1:
                messages.append({
                    "role": "user", 
                    "content": "Continue with the next step."
                })
        
        return trajectory
    
    def _parse_thought_action(self, text: str) -> tuple[str, str]:
        """Parse thought and action from generated text"""
        # Try to extract structured tags first
        try:
            thought_match = re.search(r'<thought>(.*?)</thought>', text, re.DOTALL)
            action_match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
            
            if thought_match and action_match:
                thought = thought_match.group(1).strip()
                action = action_match.group(1).strip()
            elif thought_match:
                # Only thought found, rest is action
                thought = thought_match.group(1).strip()
                action = text[thought_match.end():].strip()
            elif action_match:
                # Only action found, before is thought
                action = action_match.group(1).strip()
                thought = text[:action_match.start()].strip()
            else:
                split_point = len(text) // 2
                thought = text[:split_point].strip()
                action = text[split_point:].strip()
        except Exception as e:
             # No tags found, split roughly in half
            split_point = len(text) // 2
            thought = text[:split_point].strip()
            action = text[split_point:].strip()
        return thought, action
    # def _parse_thought_action(self, text: str) -> tuple[str, dict]:
    #     """
    #     Parse thought and action from JSON formatted text.
        
    #     Args:
    #         text: Generated JSON text with thought, tool, and final_answer
        
    #     Returns:
    #         (thought, action_dict): thought as string, action as dict with tool info
    #     """
    #     try:
    #         # Clean the text
    #         text = text.strip()
            
    #         # Try to parse as JSON
    #         data = json.loads(text)
            
    #         # Extract thought
    #         thought = data.get('thought', '').strip()
            
    #         # Extract tool information
    #         tool_info = data.get('tool', {})
            
    #         # Build action dict
    #         action = {
    #             'tool_name': tool_info.get('tool_name', ''),
    #             'tool_arguments': tool_info.get('tool_arguments', {}),
    #             'final_answer': data.get('final_answer', '')
    #         }
            
    #         return thought, action
        
    #     except json.JSONDecodeError as e:
    #         # Fallback: Try to extract JSON from text
    #         json_match = re.search(r'\{.*\}', text, re.DOTALL)
    #         if json_match:
    #             try:
    #                 data = json.loads(json_match.group(0))
    #                 thought = data.get('thought', text[:100])
    #                 tool_info = data.get('tool', {})
    #                 action = {
    #                     'tool_name': tool_info.get('tool_name', ''),
    #                     'tool_arguments': tool_info.get('tool_arguments', {}),
    #                     'final_answer': data.get('final_answer', '')
    #                 }
    #                 return thought, action
    #             except:
    #                 pass
            
    #         # Ultimate fallback
    #         ##print(f"Warning: Could not parse JSON: {e}")
    #         ##print(f"Text: {text[:200]}")
    #         return text[:100], {'tool_name': 'unknown', 'tool_arguments': {}, 'final_answer': ''}
    
    def _execute_tool_call(self, action: str) -> str:
        """
        Execute tool call and return observation.
        
        IMPLEMENT THIS METHOD with your actual tool execution logic.
        
        Args:
            action: The action text generated by the model
        
        Returns:
            Observation string from tool execution
        """
        # Placeholder implementation
        # TODO: Replace with your actual tool execution
        
        # Example: Parse tool call format
        # Expected format: tool_name(arg1="value1", arg2="value2")
        
        try:
            # Simple parser for function calls
            tool_match = re.match(r'(\w+)\((.*?)\)', action)
            if tool_match:
                tool_name = tool_match.group(1)
                tool_args = tool_match.group(2)
                
                try:
                    outputs = self.tools.get_tool_context([{
                        "tool_name": tool_match, 
                        "tool_arguments": tool_args
                    }])
                    ##print("Tool Output: ", outputs)
                    return outputs
                except Exception as e:
                    return {'error': str(e)}
            else:
                # No structured tool call found
                return f"Executed action: {action[:100]}"
        
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    # def _execute_tool_call(self, action: dict) -> str:
    #     """
    #     Execute tool call and return observation.
        
    #     Args:
    #         action: Dict with keys:
    #             - tool_name: str
    #             - tool_arguments: dict
    #             - final_answer: str
        
    #     Returns:
    #         Observation string from tool execution
    #     """
    #     try:
    #         # Check if task is already complete (has final answer)
    #         if action.get('final_answer'):
    #             return f"Task completed: {action['final_answer']}"
            
    #         # Extract tool info
    #         tool_name = action.get('tool_name', '')
    #         tool_args = action.get('tool_arguments', {})
            
    #         if not tool_name or tool_name == 'unknown':
    #             return "Error: No tool specified"
            
    #         # Execute tool using your Tools class
    #         try:
    #             # Format for your tools.get_tool_context
    #             tool_call = {
    #                 "tool_name": tool_name,
    #                 "tool_arguments": tool_args
    #             }
                
    #             # Call your tool execution
    #             result = self.tools.get_tool_context([tool_call])
                
    #             # Format result as string
    #             if isinstance(result, dict):
    #                 return json.dumps(result, indent=2)
    #             elif isinstance(result, list):
    #                 return json.dumps(result, indent=2)
    #             else:
    #                 return str(result)
            
    #         except Exception as e:
    #             return f"Error executing tool '{tool_name}': {str(e)}"
        
    #     except Exception as e:
    #         return f"Error processing action: {str(e)}"

    def _is_task_complete(self, action: str, observation: str) -> bool:
        """Check if trajectory should terminate"""
        # Check for completion signals in action or observation
        for stop_string in self.trajectory_stop_strings:
            if isinstance(action,str) and isinstance(observation,str) and stop_string in action.lower() or stop_string in observation.lower():
                return True
        
        # Check for specific completion indicators
        completion_indicators = [
            "final answer",
            "task complete",
            "finished",
            "done",
        ]
        
        combined_text = f"{action} {observation}".lower()
        for indicator in completion_indicators:
            if indicator in combined_text:
                return True
        
        return False
    def parse_gpt_json(self, json_str):
        # Remove code block markers
        if json_str.startswith("```"):
            json_str = json_str[len("```json"):].strip()
        if json_str.endswith("```"):
            json_str = json_str[:-3].strip()
        try:
            parser = JsonOutputParser()
            data = parser.parse(json_str)
            ##print("Data type",type(data))
            if(isinstance(data, str)):
                data = json.dump(data)
            ##print("Data type",type(data))
            return data
        except Exception as e:
            print("Parsing failed:", e)
            return ""
    def _calculate_rewards(
         self,
        inputs: List[Dict],  # Original input dicts
        original_prompts: List,  # Copy of prompts
        completions: List[str],  # Completion texts
        completion_ids: List[List[int]] = None,  # Optional
        **kwargs
    ):
        ##print("="*50)
        ##print("_calculate_rewards called!")
        ##print(f"Inputs type: {type(inputs)}, len: {len(inputs)}")
        ##print(f"Completions: {completions[:100]}...")  # First 100 chars
        ##print("="*50)
        rewards = []

        for query, gen_answer, ref_answer in zip(inputs, completions, original_prompts):

            # =========== 1️⃣ PROMPT: QUALITY SCORING (1-5) ===========
            quality_prompt = f"""### Task Description:
            You are evaluating an agentic reasoning trajectory generated by a model. The trajectory consists of a sequence of steps with "thought" and "action" components, where actions involve calling external tools with arguments.

            Given:
            - A generated agentic trajectory, containing multiple steps in JSON format:
            Each step includes:
                - "thought": the reasoning text
                - "tool": an object with:
                    - "tool_name": the name of the invoked tool
                    - "tool_arguments": a JSON object with arguments used in the tool call
            - A reference trajectory (ground truth) in the same format, representing the expected behavior to solve the task
            
            # Inputs: 
            Generated agentic trajectory: {gen_answer}
            Reference Trajectory: {ref_answer}
            Your task is to assign a numeric reward between 0 and 5 for the generated trajectory by scoring these criteria:

            1. Correct Tool Invocation (+1):
            - For each tool invoked in the generated trajectory,
                if the tool name matches *any* tool name in the reference trajectory, award +1 point.
            - Award the maximum possible points (up to the number of tools in the reference trajectory).

            2. Correct Tool Arguments (+1):
            - For each tool invoked whose name matches a tool in the reference trajectory,
                compare the arguments.
            - If the arguments (keys and valuable values) match exactly, award +1 point.
            - Award the maximum possible points (up to the number of tools in the reference trajectory).

            3. Thought-Action Format Reward (+1):
            - Check that the agent strictly follows the format where each reasoning step includes:
                - A "thought" explaining the current reasoning
                - An "action" describing the tool call
            - Award +1 if the format is strictly respected throughout the trajectory.

            4. Tool Sequence Matching (+1):
            - Compare the sequence/order of tool invocations between generated and reference trajectories.
            - For every consecutive tool invocation in the generated trajectory that correctly matches the order in the reference trajectory, award +1 point.
            - Award a maximum of +1 for matching the full correct order.

            ### Scoring rules:
            - Total reward is the sum of the individual criteria above.
           
            ### Example:

            Generated trajectory tools: ["list_files", "compress_file", "send_email"]

            Reference trajectory tools: ["list_files", "compress_file", "send_email"]
            
            Output:
            {{
                "Score": "5"
            }}
            
            # Output Format: 
            ONLY GENERATE THE JSON IN BELOW FORMAT
            {{
                "Score": "X"
            }}

        """

            quality_response = self.judge_llm.gpt(quality_prompt).content.strip()
            # Extract numeric score from format: "Feedback: ... [4]"
            
            score = self.parse_gpt_json(quality_response)
            ##print(score)
            try:
                final_reward = score["Score"]
            except:
                final_reward =  np.random.randint(1, 6)
            final_reward = float(final_reward)
            scaled_score = final_reward * 10.0  # Now 10-50 instead of 1-5
        
            # ✅ Add noise to ensure variance
            noise = torch.randn(1).item() * 2.0
            final_score = scaled_score + noise
            rewards.append([final_score])
        return torch.tensor(rewards, dtype=torch.float32).to(self.device)
    # Override compute_loss to add debugging
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Add debugging to loss computation"""
        
        # Call parent
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        
        # Debug
        print(f"\n{'='*60}")
        print(f"Loss Computation Debug:")
        print(f"{'='*60}")
        print(f"Loss value: {loss}")
        print(f"Loss requires_grad: {loss.requires_grad if hasattr(loss, 'requires_grad') else 'N/A'}")
        
        if hasattr(inputs, 'advantages'):
            print(f"Advantages in inputs: {inputs.advantages}")
            print(f"Advantages sum: {inputs.advantages.sum()}")
        
        print(f"{'='*60}\n")
        
        return loss

    def _generate_and_score_completions(
        self, 
        inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Override the main generation method to support agentic trajectories.
        Generates multiple trajectories per prompt for group-relative optimization.
        """
        from trl.data_utils import maybe_apply_chat_template

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        
        # Extract prompts from input dicts
        prompts = [x["prompt"] for x in inputs]
        ground_truth = [x["ground_truth"] for x in inputs]
        original_prompts = copy.deepcopy(prompts)
        
        # ✅ KEY FIX: Repeat each prompt num_generations times
        # This creates groups for GRPO comparison
        prompts_repeated = []
        ground_truth_repeated = []
        for i in range(len(prompts)):
            for _ in range(self.num_generations):
                prompts_repeated.append(prompts[i])
                ground_truth_repeated.append(ground_truth[i])
        
        ##print(f"Total generations: {len(prompts_repeated)} = {len(prompts)} prompts × {self.num_generations} generations")
        
        # Create inputs for repeated prompts
        inputs_repeated = []
        for i in range(len(prompts)):
            for _ in range(self.num_generations):
                inputs_repeated.append(inputs[i])
        
        # Apply chat template
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] 
            for example in inputs_repeated
        ]
        
        # Tokenize prompts
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        
        prompt_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in prompt_inputs.items()
        }
        
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]
        
        # Generate trajectories (now with multiple generations per prompt)
        trajectories = []
        all_trajectory_ids = []
        
        ##print(f"Generating {len(prompts_repeated)} trajectories ({len(prompts)} prompts × {self.num_generations} each)...")
        
        with torch.no_grad():
            for idx, prompt in enumerate(prompts_repeated):
                prompt_num = idx // self.num_generations + 1
                gen_num = idx % self.num_generations + 1
                
                print(f"Generating trajectory {idx+1}/{len(prompts_repeated)} (Prompt {prompt_num}/{len(prompts)}, Gen {gen_num}/{self.num_generations})...")
                
                # ✅ Generate trajectory with sampling for diversity
                trajectory = self.generate_trajectory(
                    prompt,
                    temperature=0.7 + (gen_num * 0.1),  # Vary temperature for diversity
                    do_sample=True
                )
                if not trajectory or len(trajectory) == 0:
                    trajectory = [{"thought": "", "action": {}, "observation": "", "step": 0}]
                # Convert trajectory to token sequence
                trajectory_text = self._format_trajectory(trajectory)
                
                # Tokenize the complete trajectory
                trajectory_inputs = self.processing_class(
                    text=trajectory_text,
                    return_tensors="pt",
                    padding=False,
                    add_special_tokens=False,
                )
                traj_ids = trajectory_inputs["input_ids"].squeeze(0)
                if len(traj_ids) == 0:
                    # Create dummy token
                    traj_ids = torch.tensor([self.processing_class.eos_token_id], device=device)
                trajectories.append(trajectory)
                all_trajectory_ids.append(traj_ids)
                
                #print(f"  → Generated {len(trajectory)} steps")
    

       

                
        # Pad trajectory completions
        completion_ids = pad(
            all_trajectory_ids, 
            padding_value=self.pad_token_id, 
            padding_side="right"
        )
        completion_ids = completion_ids.to(device)
        
        # Create completion mask
        completion_mask = self._create_trajectory_mask(trajectories, completion_ids)
        
        # Concatenate prompts with trajectories
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        # Compute logprobs
        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size if mode == "train" 
            else self.args.per_device_eval_batch_size
        )
        
        with torch.no_grad():
            # Compute old policy logprobs
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                old_per_token_logps = None
            
            # Compute reference policy logprobs
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size,
                        )
            else:
                ref_per_token_logps = None
        
        # Prepare completions for reward calculation
        completion_texts = [self._trajectory_to_text(traj) for traj in trajectories]
        completion_ids_list = [
            row[mask_row.bool()].tolist() 
            for row, mask_row in zip(completion_ids, completion_mask)
        ]
        
        # Calculate rewards
        rewards_per_func = self._calculate_rewards(
            inputs_repeated,  # Use repeated inputs
            ground_truth_repeated,  # Use repeated ground truth
            completion_texts,
        )
        
        # Aggregate rewards with weights
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        print(f"\n{'='*60}")
        print(f"Reward and Advantage Debug:")
        print(f"{'='*60}")
        print(f"Rewards: {rewards}")
        print(f"Rewards shape: {rewards.shape}")
        print(f"Mean grouped rewards: {mean_grouped_rewards}")
        print(f"Advantages: {advantages}")
        print(f"Advantages shape: {advantages.shape}")
        print(f"Advantages sum: {advantages.sum()}")
        print(f"Advantages mean: {advantages.mean()}")
        print(f"Advantages std: {advantages.std()}")

        # Check if advantages are too small
        if advantages.abs().max() < 1e-6:
            print(f"❌ WARNING: Advantages are too small! This causes zero loss.")
            print(f"   Max advantage: {advantages.abs().max()}")
            # Scale up advantages
            advantages = advantages * 100
            print(f"   Scaled advantages: {advantages}")

        print(f"{'='*60}\n")
        ##print(f"Mean grouped rewards: {mean_grouped_rewards}")
        ##print(f"Advantages: {advantages}")
        
        # Scale advantages
        if self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        elif self.scale_rewards == "group":
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        else:
            std_rewards = torch.ones_like(rewards)
        
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)
        
        # Slice for local process in distributed training
        process_slice = slice(
            self.accelerator.process_index * len(prompts_repeated),
            (self.accelerator.process_index + 1) * len(prompts_repeated),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]
        
        # Logging
        completion_lengths = completion_mask.sum(1)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        num_items_in_batch = agg_completion_lengths.sum()
        
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        
        # Build output
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        
        return output



# Usage Example
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset
   
    # Load model and tokenizer
    model_name = "path_to_your_base_model"  # Replace with your model path or name
    model_config = ModelConfig(
        model_name_or_path=model_name,
        trust_remote_code=True,
        dtype="auto",
        use_peft=True,
        lora_r=128,  # Reduced from 128 for memory
        lora_alpha=256,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Fewer modules for memory
        lora_task_type="CAUSAL_LM",
    )

    # Get quantization config
    quantization_config = get_quantization_config(model_config)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        quantization_config=quantization_config,
        use_cache=False,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ✅ CRITICAL: Apply LoRA adapters
    print("\n" + "="*60)
    print("Applying LoRA adapters...")
    print("="*60)

    # Prepare model for k-bit training (required for quantized models)
    # if quantization_config is not None:
    #     model = prepare_model_for_kbit_training(model)

    # Create LoRA config
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        target_modules=model_config.lora_target_modules,
        task_type=model_config.lora_task_type,
        bias="none",
    )

    # ✅ Apply LoRA to model
    model = get_peft_model(model, peft_config)

    # Enable gradient checkpointing for LoRA
    model.enable_input_require_grads()

    # ✅ Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params

    print(f"\n✅ LoRA Applied Successfully!")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Frozen params: {frozen_params:,}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.4f}%")
    print("="*60 + "\n")

    if trainable_params == 0:
        raise RuntimeError("❌ CRITICAL ERROR: No trainable parameters after applying LoRA!")

    train_data_path = "path_to_your_training_data.json"  # Replace with your training data path

    # Prepare dataset
    # Load dataset (you'll implement your own Dataset class or JSON loader)
    import json
    with open(train_data_path, "r") as f:
        raw_data = json.load(f)
    
    # random.shuffle(raw_data)
    ##print(f"Training data length {len(raw_data)}")
    dataset = GRPODataset(raw_data)
    output_dir = "path_to_save_trained_model"  # Replace with your desired output directory
    print("For server 19 Output DIr: ", output_dir)
    # Configure training
    training_args = GRPOConfig(
        output_dir=output_dir,
        # Training
        num_train_epochs=1,                   # More epochs for better training
        # Learning rate (CRITICAL FIX)
        learning_rate=5e-5,                   # ✅ Good for LoRA
        lr_scheduler_type="cosine",           
        warmup_ratio=0.1,                     # ✅ 10% warmup
        warmup_steps=10, 
        # Batch (your current settings)
        per_device_train_batch_size=2,     # Up from 2
        gradient_accumulation_steps=1,     # Down from 8
        num_generations=2, 
        max_grad_norm=10.0,    
        # GRPO
        temperature=0.7,
        beta=0.04,
        scale_rewards="none",
        
        # Memory
        gradient_checkpointing=True,
        bf16=True,
        save_steps=100,
        # Logging
        logging_steps=1,
    )
   
    def dummy_reward_function(completions, **kwargs):
        """
        Dummy reward - actual rewards computed in _calculate_rewards() override.
        This function should never be called.
        """
        ##print("WARNING: Dummy reward function called (should use override)")
        return [0.0] * len(completions)
    # Initialize custom trainer
    trainer = AgenticGRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=dummy_reward_function,
        train_dataset=dataset,
        processing_class=tokenizer,
        max_trajectory_steps=4,
        trajectory_stop_strings=["</answer>", "<done>"],
        use_streaming=True,  # Set to True for debugging
    )

    trainer.train()
    
    # Save model
    trainer.save_model(output_dir)
