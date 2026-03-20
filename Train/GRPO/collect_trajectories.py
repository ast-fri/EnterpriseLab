
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from langchain_core.output_parsers import JsonOutputParser
from FineTuning.GRPO.reward import Reward
from Task_Generation_sft_batch2_copy.Factories.llm_factory import LLM_factory
import os
import random
from typing import List, Dict, Any
import re
class TracjDataset(Dataset):
    def __init__(self, raw_data):
        self.data = raw_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item.get('messages', [])
        with open("path/to/Task_Generation_sft_batch2_copy/utils/tools.json", 'r') as f:
            available_tools = json.load(f)
        system_prompt = f"""You are an Enterprise agent.
           # Input: 
           Available tools: {available_tools}
           
           
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
            'query': query,
            'full_messages': messages
        }


def custom_collate_fn(batch):
    # batch is a list of dicts with keys 'prompt' and 'ground_truth'
    prompts = [item['prompt'] for item in batch]
    ground_truths = [item['ground_truth'] for item in batch]
    query = [item['query'] for item in batch]
    full_messages = [item['full_messages'] for item in batch]
    
    return {
        'prompt': prompts,
        'ground_truth': ground_truths,
        'query': query,
        'full_messages': full_messages
    }


class TrajectoryCollector:
    def __init__(self,model_path, dataset, tools,  max_rollout_steps=10, n_samples=5, max_steps_to_explore=3, reward_threshold=0.1):
        
        self.dataset = dataset
        self.tools = tools
        self.max_rollout_steps = max_rollout_steps
        self.n_samples = n_samples  # N for Monte Carlo sampling
        self.reward_threshold = reward_threshold  # τ threshold for filtering
        # self.llm = LLM_factory()
        self.max_steps_to_explore = max_steps_to_explore
        self.llm = LLM_factory() 
        self.reward_calculator = Reward()  # Assuming this calculates outcome rewards
    def _parse_thinking(self, text: str):
        if "<think>" in text or "</think>" in text:
            start = text.find("<think>")
            end = text.find("</think>") + len("</think>")
            thinking = text[start:end]
            remainder = text[end:].strip()
            return thinking, remainder
        return "", text
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
                data = json.loads(data)
            return data
        except Exception as e:
            return {}
  
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
    def _extract_generated_text(self, raw_content, original_messages):
        """
        Extract only the model's generated text, removing any prompt echo
        
        Args:
            raw_content: The raw output from the model
            original_messages: The original input messages
        
        Returns:
            Cleaned generated text only
        """
        # Remove <think> tags if present
        content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)
        
        # Remove system prompt if it appears in output
        system_content = None
        user_content = None
        
        for msg in original_messages:
            if msg.get('role') == 'system':
                system_content = msg.get('content', '')
            elif msg.get('role') == 'user':
                user_content = msg.get('content', '')
        
        # If the output contains the system prompt, remove it
        if system_content and system_content in content:
            # Find where system prompt ends
            idx = content.find(system_content)
            if idx != -1:
                # Remove everything up to end of system prompt
                content = content[idx + len(system_content):].strip()
        
        # If the output contains the user message, extract only what comes after
        if user_content and user_content in content:
            # Split at user message and take the part after
            parts = content.split(user_content)
            if len(parts) > 1:
                content = parts[-1].strip()
        
        # Remove any leading/trailing special tokens or markers
        content = content.strip()
        content = re.sub(r'^<\|im_start\|>.*?<\|im_end\|>', '', content, flags=re.DOTALL)
        content = re.sub(r'^assistant\s*:', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^system\s*:', '', content, flags=re.IGNORECASE)
        
        return content.strip()
    def generate_trajectory(self, prompt, max_steps, expert_trajectories):
        """
        Generate a single trajectory from the model
        Returns: trajectory steps and final outcome reward
        """
        trajectory = []
        conversation = [msg for msg in prompt if msg.get('role') in ['system', 'user']]
        # #print(conversation)
        reached_final_answer = False
        for step in range(max_steps):
            # Generate using model - pass current conversation
            #print("="*10)
            #print(f"Step: {step}")
            #print("="*10)
            raw_text = self.llm.gpt(conversation)
            cleaned_text = self._extract_generated_text(raw_text.content, conversation)
            thinking, parsed_clean_text = self._parse_thinking(cleaned_text)
            parsed_clean_text = self.parse_tool_calls(parsed_clean_text)
            # print("Model Output: ",parsed_clean_text)
            # Extract predicted tool call
            predicted_tool = parsed_clean_text.get("tool", {})
            predicted_tool_name = predicted_tool.get("tool_name", "")
            predicted_tool_args = predicted_tool.get("tool_arguments", {})
            
            thought_to_use = parsed_clean_text.get("thought", "")
            
            # Check for final answer
            if parsed_clean_text.get("final_answer", ""):
                # ONLY add to trajectory, not conversation
                trajectory.append({
                    'role': 'assistant',
                    'content': thought_to_use,
                    'final_answer': parsed_clean_text["final_answer"]
                })
                reached_final_answer = True
                break
            
            # Store step in trajectory (for later DPO formatting)
            step_data = {
                'role': 'assistant',
                'content': thought_to_use
            }
            
            if predicted_tool_name:
                # Execute tool
                try:
                    tool_output = self.execute_tool({
                        'tool_name': predicted_tool_name,
                        'arguments': predicted_tool_args
                    })
                except Exception as e:
                    tool_output = f"Error: {str(e)}"
                
                # Add tool info to step data
                step_data['tool_call'] = {
                    'name': predicted_tool_name,
                    'arguments': predicted_tool_args
                }
                step_data['observation'] = tool_output
                
                # CORRECT: Build conversation for next iteration
                # Add tool call message
                conversation.append({
                    'role': 'assistant',
                    'content': '',  # ← ADD THIS (empty string is fine)
                    'tool_calls': [{
                        'type': 'function',
                        'function': {
                            'name': predicted_tool_name,
                            'arguments': predicted_tool_args
                        }
                    }]
                })
                
                # Add tool result
                conversation.append({
                    'role': 'tool',
                    'name': predicted_tool_name,
                    'content': json.dumps(tool_output) if isinstance(tool_output, dict) else str(tool_output)
                })
            else:
                # No tool call - just add thought
                conversation.append({
                    'role': 'assistant',
                    'content': thought_to_use
                })
            
            # Add to trajectory
            trajectory.append(step_data)
            
            # If no tool was called, break
            if not predicted_tool_name:
                break
        
        # Calculate outcome reward
        outcome_reward = self.calculate_outcome_reward(conversation, expert_trajectories)
        
        return {
            'trajectory': trajectory,
            'conversation': conversation,
            'reached_final_answer': reached_final_answer,
            'outcome_reward': outcome_reward
        }

    
    def calculate_step_reward(self, historical_trajectory, action, n_samples=3):
        """
        Monte Carlo method to estimate step-level reward (Equation 3-5 in paper)
        """
        # Create conversation up to this step
        conversation = historical_trajectory[:]
        
        # Add the action we're evaluating
        if isinstance(action, dict):
            conversation.append(action)
        
        # Sample N trajectories from this step
        rewards = []
        for _ in tqdm(range(self.n_samples)):
            result = self.generate_trajectory(conversation, self.max_rollout_steps)
            rewards.append(result['outcome_reward'])
        
        # Average the rewards
        step_reward = sum(rewards) / len(rewards)
        return step_reward
    
    def calculate_outcome_reward(self, conversation, reference_trajectory=None):
        """
        Calculate outcome reward using LLM-as-judge with Prometheus-style evaluation
        
        Args:
            conversation: The generated trajectory conversation
            reference_trajectory: Optional expert trajectory for comparison
        
        Returns:
            Float reward score (0.0 to 1.0)
        """
        # Extract query (user message)
        query = None
        for msg in conversation:
            if msg.get('role') == 'user':
                query = msg.get('content', '')
                break
        #print("Query : ",query)
        if not query:
            return 0.0
        
        # Extract generated final answer
        generated_answer = conversation
        
        
        
        # Build evaluation prompt
        evaluation_prompt = f"""###Task Description:
    An instruction (might include an Input inside it), a query, a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
    1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
    2. After writing a feedback, write a score that is either 1 or 2 or 3 or 4 or 5. You should refer to the score rubric.
    3. The output format should only look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
    4. Please do not generate any other opening, closing, and explanations.
    5. Only evaluate on common things between generated answer and reference answer. Don't evaluate on things which are present in reference answer but not in generated answer.

    ###Instruction:
    Your task is to evaluate the generated answer and reference answer for the following query:
    {query}

    ###Generated answer to evaluate:
    {generated_answer}

    ###Reference Answer (Score 5):
    {reference_trajectory}

    ###Score Rubrics:
    Score 1: If the generated answer is not relevant to the user query and reference answer with missing entities.
    Score 2: If the generated answer is relevant to user query or contains significant mistakes or misses important entities or if entities present in reference answer are not exact match.
    Score 3: If the generated answer is relevant to the reference answer and answers the query but contains very few mistakes.
    Score 4: If the generated answer is relevant to the user query and all the entities are exact match as the reference answer along with same intent, but it is not as concise.
    Score 5: If the generated answer is relevant to the user query and fully correct according to the reference answer.

    ###Feedback:"""
        
        # Call LLM for evaluation
        try:
            eval_messages = [
                {'role': 'system', 'content': 'You are an expert evaluator. Provide structured feedback following the exact format specified.'},
                {'role': 'user', 'content': evaluation_prompt}
            ]
            
            # Use GPT-4o for evaluation (more reliable than local model)
            response = self.llm.gpt(eval_messages)
            eval_text = response.content
            
            # Parse score from response
            score = self._parse_score(eval_text)
            
            # Normalize to 0-1 range
            normalized_reward = (score - 1) / 4.0  # 1-5 -> 0.0-1.0
            
            print(f"Evaluation score: {score}/5 (reward: {normalized_reward:.3f})")
            
            return normalized_reward
            
        except Exception as e:
            print(f"Error in LLM evaluation: {str(e)}")
            return self._calculate_task_completion_reward(conversation)

    def _parse_score(self, eval_text):
        """
        Parse score from evaluation text
        Expected format: "Feedback: ... [RESULT] (score)"
        """
        import re
        
        # Try to find [RESULT] followed by a number
        result_pattern = r'\[RESULT\]\s*\(?(\d+)\)?'
        match = re.search(result_pattern, eval_text)
        
        if match:
            score = int(match.group(1))
            return max(1, min(5, score))  # Clamp to 1-5
        
        # Fallback: look for any number between 1-5
        number_pattern = r'\b([1-5])\b'
        matches = re.findall(number_pattern, eval_text)
        
        if matches:
            # Take the last number found (usually the score)
            score = int(matches[-1])
            return score
        
        # No score found, return middle score
        print(f"Could not parse score from: {eval_text[:100]}")
        return 2

    def _calculate_task_completion_reward(self, conversation):
        """
        Fallback heuristic reward based on task completion
        """
        # Check if final answer exists
        has_final_answer = any(
            'final_answer' in msg 
            for msg in conversation 
            if isinstance(msg, dict)
        )
        
        if not has_final_answer:
            return 0.0
        
        # Count successful tool calls
        successful_tools = 0
        total_tools = 0
        
        for msg in conversation:
            if msg.get('role') == 'tool':
                total_tools += 1
                content = str(msg.get('content', ''))
                # Check if tool returned valid result (not error)
                if 'error' not in content.lower() and content.strip():
                    successful_tools += 1
        
        # Calculate completion score
        if total_tools == 0:
            # No tools used but has final answer
            tool_success_rate = 0.5
        else:
            tool_success_rate = successful_tools / total_tools
        
        # Weighted reward: 50% tool success + 50% has final answer
        reward = 0.5 * tool_success_rate + 0.5 * (1.0 if has_final_answer else 0.0)
        
        return reward

    def calculate_relevancy_reward(self, query, response, context):
        """
        Optional: Calculate relevancy reward for response given context
        Using Prometheus relevancy evaluation
        """
        relevancy_prompt = f"""###Task Description:
    An instruction (might include an Input inside it), a query with response, context, and a score rubric representing a evaluation criteria are given.
    1. You are provided with evaluation task with the help of a query with response and context.
    2. Write a detailed feedback based on evaluation task and the given score rubric, not evaluating in general.
    3. After writing a feedback, write a score that is YES or NO. You should refer to the score rubric.
    4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (YES or NO)"
    5. Please do not generate any other opening, closing, and explanations.

    ###The instruction to evaluate: 
    Your task is to evaluate if the response for the query is in line with the context information provided.

    ###Query and Response:
    {query}
    Response: {response}

    ###Context:
    {context}

    ###Score Rubrics:
    Score YES: If the response for the query is in line with the context information provided.
    Score NO: If the response for the query is NOT in line with the context information provided.

    ###Feedback: """
        
        try:
            eval_messages = [
                {'role': 'system', 'content': 'You are an expert evaluator. Provide structured feedback following the exact format specified.'},
                {'role': 'user', 'content': relevancy_prompt}
            ]
            
            response = self.llm.gpt(eval_messages)
            eval_text = response.content
            
            # Parse YES/NO
            if '[RESULT]' in eval_text:
                result_part = eval_text.split('[RESULT]')[1].strip()
                if 'YES' in result_part.upper():
                    return 1.0
                elif 'NO' in result_part.upper():
                    return 0.0
            
            # Fallback
            return 0.5
            
        except Exception as e:
            print(f"Error in relevancy evaluation: {str(e)}")
            return 0.5

    
    def construct_contrastive_pairs(self, expert_trajectory, prompt):
        """
        Construct contrastive action pairs by exploring along expert trajectory
        """
        contrastive_step_pairs = []
        contrastive_trajectory_pairs = []
        
        expert_messages = expert_trajectory
        
        # Sample steps to explore
        if len(expert_messages) > self.max_steps_to_explore:
            steps_to_explore = sorted(random.sample(
                range(len(expert_messages)), 
                self.max_steps_to_explore
            ))
        else:
            steps_to_explore = range(len(expert_messages))
        
        for t in steps_to_explore:
            historical_trajectory = [
                msg for msg in (prompt[:] + expert_messages[:t])
                if msg.get('role') in ['system', 'user']
            ]
            
            expert_action = expert_messages[t]
            expert_full_trajectory = prompt[:] + expert_messages
            # Generate agent's trajectory
            agent_result = self.generate_trajectory(
                historical_trajectory, 
                self.max_rollout_steps,
                expert_full_trajectory
            )
            
            if len(agent_result['trajectory']) == 0:
                continue
            
            # Calculate rewards WITH reference trajectory
            
            
            expert_outcome_reward = self.calculate_outcome_reward(
                expert_full_trajectory,
                reference_trajectory=expert_full_trajectory  # Pass itself as reference
            )
            
            agent_outcome_reward = self.calculate_outcome_reward(
                agent_result['conversation'],
                reference_trajectory=expert_full_trajectory  # Compare against expert
            )
            
            # Calculate step rewards (can use simpler heuristic or sprintame LLM eval)
            expert_step_reward = expert_outcome_reward  # Simplified
            agent_step_reward = agent_outcome_reward
            #print(f"expert_step_reward : {expert_step_reward}")
            #print(f"agent_step_reward : {agent_step_reward}")
            #print(f"expert_outcome_reward : {expert_outcome_reward}")
            #print(f"agent_outcome_reward : {agent_outcome_reward}")
            step_reward_diff = expert_step_reward - agent_step_reward
            outcome_reward_diff = expert_outcome_reward - agent_outcome_reward
            
            if step_reward_diff > self.reward_threshold and outcome_reward_diff > 0:
                #print(f"Step {t}: Creating pair (reward_diff={step_reward_diff:.3f})")
                
                expert_remaining = expert_messages[t:]
                agent_remaining = agent_result['trajectory']
                
                step_pair = {
                    'prompt': historical_trajectory,
                    'chosen': expert_remaining,
                    'rejected': agent_remaining,
                    'chosen_reward': expert_step_reward,
                    'rejected_reward': agent_step_reward,
                    'reward_diff': step_reward_diff,
                    'type': 'step'
                }
                contrastive_step_pairs.append(step_pair)
                
                trajectory_pair = {
                    'prompt': prompt,
                    'chosen': expert_messages,
                    'rejected': agent_result['trajectory'],
                    'chosen_reward': expert_outcome_reward,
                    'rejected_reward': agent_outcome_reward,
                    'reward_diff': outcome_reward_diff,
                    'type': 'outcome'
                }
                contrastive_trajectory_pairs.append(trajectory_pair)
                
                
        
        return contrastive_step_pairs, contrastive_trajectory_pairs

    
    def execute_tool(self, call):
        """
        Use the tools to interact with environment.
        """
        try:
            outputs = self.tools.get_tool_context([{
                "tool_name": call['tool_name'], 
                "tool_arguments": call.get('arguments', {})
            }])
            return outputs
        except Exception as e:
            return {'error': str(e)}
    
    def format_for_dpo(self, contrastive_pairs):
        """
        Format contrastive pairs for TRL DPOTrainer
        DPOTrainer expects: {'prompt': [...], 'chosen': [...], 'rejected': [...]}
        Where each is a list of message dicts
        """
        dpo_dataset = []
        
        for pair in contrastive_pairs:
            # Convert trajectory steps to message format
            chosen_messages = self.trajectory_to_messages(pair['chosen'])
            rejected_messages = self.trajectory_to_messages(pair['rejected'])
            
            dpo_example = {
                'prompt': pair['prompt'],  # Already in message format
                'chosen': chosen_messages,
                'rejected': rejected_messages,
            }
            dpo_dataset.append(dpo_example)
        
        return dpo_dataset
    
    def trajectory_to_messages(self, trajectory):
        """
        Convert trajectory steps to message format for DPO
        """
        messages = []
        
        for step in trajectory:
            # Add assistant thought content
            if 'content' in step and step['content']:
                messages.append({
                    'role': 'assistant',
                    'content': step['content']
                })
            
            # Add tool call if exists
            if 'tool_call' in step:
                tool_call = step['tool_call']
                
                # Ensure tool name is a string, not a set
                tool_name = tool_call['name']
                if isinstance(tool_name, set):
                    tool_name = list(tool_name)[0]  # Convert set to string
                
                messages.append({
                    'role': 'assistant',
                    'content': step.get('content', ''),
                    'tool_calls': [{
                        'type': 'function',
                        'function': {
                            'name': tool_name,  # Now guaranteed to be string
                            'arguments': tool_call.get('arguments', {})
                        }
                    }]
                })
                
                # Add tool observation
                if 'observation' in step:
                    observation = step['observation']
                    
                    # Handle observation being a list or dict
                    if isinstance(observation, list):
                        observation = observation[0] if observation else {}
                    
                    # Extract tool output
                    tool_output = observation
                    if isinstance(observation, dict):
                        tool_output = observation.get('tool_output', observation)
                    
                    messages.append({
                        'role': 'tool',
                        'name': tool_name,
                        'content': json.dumps(tool_output) if isinstance(tool_output, dict) else str(tool_output)
                    })
            
            # Add final answer if exists
            if 'final_answer' in step:
                # Final answer is already in content
                pass
        
        return messages

    
    def collect_trajectories(self, output_dir):
        """
        Main collection loop - collect contrastive pairs from entire dataset
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_step_pairs = []
        all_trajectory_pairs = []
        all_expert_trajectories = []  # For SFT on successful trajectories
        
        print(f"Collecting trajectories from {len(self.dataset)} examples...")
        
        for idx in tqdm(range(len(self.dataset))):
            example = self.dataset[idx]
            
            prompt = example['prompt']
            ground_truth = example['ground_truth']
            
            # Construct contrastive pairs by exploring along expert trajectory
            step_pairs, traj_pairs = self.construct_contrastive_pairs(
                ground_truth, 
                prompt
            )
            #print("Step pair Trajectories: ", step_pairs)
            #print("Trajectory Pairs: ", traj_pairs)
            all_step_pairs.extend(step_pairs)
            all_trajectory_pairs.extend(traj_pairs)
            
            # Also keep expert trajectories for SFT loss
            all_expert_trajectories.append({
                'prompt': prompt,
                'chosen': ground_truth,
                'type': 'expert'
            })
            
            # Save intermediate results every 100 examples
            if (idx + 1) % 10 == 0:
                self.save_intermediate_results(
                    output_dir, idx + 1,
                    all_step_pairs, all_trajectory_pairs, all_expert_trajectories
                )
        
        # Format for DPO
        step_dpo_data = self.format_for_dpo(all_step_pairs)
        outcome_dpo_data = self.format_for_dpo(all_trajectory_pairs)
        expert_sft_data = self.format_for_dpo(all_expert_trajectories)
        
        # Save final datasets
        self.save_final_datasets(
            output_dir,
            step_dpo_data,
            outcome_dpo_data,
            expert_sft_data
        )
        
        print(f"\nCollection complete!")
        print(f"Step-level DPO pairs: {len(step_dpo_data)}")
        print(f"Outcome-level DPO pairs: {len(outcome_dpo_data)}")
        print(f"Expert SFT examples: {len(expert_sft_data)}")
        
        return {
            'step_dpo': step_dpo_data,
            'outcome_dpo': outcome_dpo_data,
            'expert_sft': expert_sft_data
        }
    def clean_for_json(self, obj):
        """
        Recursively clean object to make it JSON serializable
        Converts sets to lists, removes non-serializable types
        """
        if isinstance(obj, dict):
            return {k: self.clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.clean_for_json(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)  # Convert set to list
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)  # Convert other types to string
    def save_intermediate_results(self, output_dir, idx, step_pairs, traj_pairs, expert_trajs):
        """Save intermediate results"""
        # Clean data before saving
        step_pairs_clean = self.clean_for_json(step_pairs)
        traj_pairs_clean = self.clean_for_json(traj_pairs)
        with open(os.path.join(output_dir, f'step_pairs_checkpoint_{idx}.json'), 'w') as f:
            json.dump(step_pairs_clean, f, indent=2)
        
        with open(os.path.join(output_dir, f'traj_pairs_checkpoint_{idx}.json'), 'w') as f:
            json.dump(traj_pairs_clean, f, indent=2)
    
    def save_final_datasets(self, output_dir, step_dpo, outcome_dpo, expert_sft):
        """Save final formatted datasets for DPO training"""
        
        # Step-level DPO dataset
        with open(os.path.join(output_dir, 'step_dpo_dataset.json'), 'w') as f:
            json.dump(step_dpo, f, indent=2)
        
        # Outcome-level DPO dataset
        with open(os.path.join(output_dir, 'outcome_dpo_dataset.json'), 'w') as f:
            json.dump(outcome_dpo, f, indent=2)
        
        # Expert SFT dataset
        with open(os.path.join(output_dir, 'expert_sft_dataset.json'), 'w') as f:
            json.dump(expert_sft, f, indent=2)
        
        # Combined dataset for mixed training (IPR approach)
        combined = {
            'step_dpo': step_dpo,
            'outcome_dpo': outcome_dpo,
            'expert_sft': expert_sft
        }
        with open(os.path.join(output_dir, 'combined_ipr_dataset.json'), 'w') as f:
            json.dump(combined, f, indent=2)
        
        print(f"\nDatasets saved to {output_dir}/")
        print(f"  - step_dpo_dataset.json: {len(step_dpo)} examples")
        print(f"  - outcome_dpo_dataset.json: {len(outcome_dpo)} examples")
        print(f"  - expert_sft_dataset.json: {len(expert_sft)} examples")
        print(f"  - combined_ipr_dataset.json: All data combined")


if __name__ == "__main__":
    
    # Paths to model checkpoint, training data, and output directory
    model_path = "path/to/your/model/checkpoint"  # <-- UPDATE THIS to your model checkpoint path
    train_data_path = "path/to/your/training/data.json"  # <-- UPDATE THIS to your training data path
    tools_path = "path/to/your/tools/config.json"  # <-- UPDATE THIS to your tools configuration path
    output_dir = "path/to/save/collected/trajectories"  # <-- UPDATE THIS to your desired output directory
    
    # Load dataset
    with open(train_data_path, "r") as f:
        raw_data = json.load(f)
    print(f"Training data length: {len(raw_data)}")
    
    dataset = TracjDataset(raw_data)
    
    # Initialize tools
    from Task_Generation_sft_batch2_copy.utils.tools import Tools
    tools_instance = Tools()
    
    # Instantiate trajectory collector
    collector = TrajectoryCollector(
        model_path=model_path,
        dataset=dataset,
        tools=tools_instance,
        max_rollout_steps=10,
        max_steps_to_explore=3,
        n_samples=3,  # Monte Carlo samples (N in paper)
        reward_threshold=0.1  # τ threshold for filtering actions
    )
    
    # Collect trajectories
    trajectories = collector.collect_trajectories(output_dir)
    
    print(f"\nTrajectory collection complete!")
    print(f"Results saved to {output_dir}/")
