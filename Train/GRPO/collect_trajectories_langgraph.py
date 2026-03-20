from typing import Annotated, TypedDict, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
import operator
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from FineTuning.GRPO.reward import Reward
from Task_Generation_sft_batch2_copy.Factories.llm_factory import LLM_factory
from Task_Generation_sft_batch2_copy.utils.tools import Tools
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import re
tools = Tools()

# ============================================
# DATASET CLASS
# ============================================
class ToolCall(BaseModel):
    """Schema for tool call"""
    tool_name: str = Field(description="Name of the tool to call")
    tool_arguments: dict = Field(description="Arguments for the tool", default_factory=dict)


class AgentOutput(BaseModel):
    """Schema for agent's JSON output"""
    thought: str = Field(description="Agent's reasoning about the current step")
    tool: Optional[ToolCall] = Field(description="Tool to call, if any", default=None)
    final_answer: str = Field(description="Final answer if task is complete", default="")
output_parser = JsonOutputParser(pydantic_object=AgentOutput)

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
        system_prompt = f"""You are a ReAct agent with access to tools. You MUST use tools when available for the requested actions.


            AVAILABLE TOOLS:
            {available_tools}

           CRITICAL RULES:
            1. Provide your thought.
            2. Use EXACT tool names from the list above
            3. Use EXACT parameter names (check Required arguments and Parameter Details)
            4. Always wrap JSON in triple backticks with json marker
            5. For multi-step tasks, complete ALL steps before Final Answer
            6. If a tool fails, try to fix and retry

            EXAMPLES:

            Task: Create repo 'test' and add feature branch
            "thought": Create repository first
            ```json
            {{"action": "create_repository", "action_input": {{"name": "test", "description": "test repo"}}}}
            ```
            [After repo created...]
            "thought": Create feature branch
            ```json
            {{"action": "create_branch", "action_input": {{"project_id": "<repo_id>", "branch": "feature"}}}}
            ```
            [After branch created...]
            "thought": Give Final Answer
            ```json
            {{"action": "Final Answer", "action_input": "Created repository 'test' and added feature branch 'feature'"}}
            ```"""
        
        # Extract only system and user messages as prompt (input to model)
        prompt_messages = []
        query = ""
        ground_truth = []
        for msg in messages:
            if msg.get('role') in ['system', 'user']:
                if msg.get('role') == 'system':
                    # Prepend system_prompt to existing system message content
                    modified_msg = msg.copy()
                    modified_msg['content'] = system_prompt
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
            'full_messages': messages,
            'idx': idx  # Add index for tracking
        }


# ============================================
# TRAJECTORY STATE
# ============================================

class TrajectoryState(TypedDict):
    """State for trajectory generation"""
    prompt: list  # Initial conversation
    messages: Annotated[list, operator.add]  # Conversation history with tool results
    trajectory: Annotated[list, operator.add]  # Steps in trajectory
    current_step: int
    max_steps: int
    reached_final_answer: bool
    final_answer: str
    outcome_reward: float
    expert_trajectories: list


# ============================================
# TRAJECTORY GENERATOR
# ============================================

class Trajectories:
    def __init__(self):
        self.llm = LLM_factory()
        with open("path/to/Task_Generation_sft_batch2_copy/utils/tools.json", 'r') as f:
            self.available_tools = json.load(f)
        self.reward_calculator = Reward()  # Initialize your reward calculator
        return
    
    def execute_tool(self, call):
        """
        Use the tools to interact with environment.
        """
        try:
            outputs = tools.get_tool_context([{
                "tool_name": call['tool_name'], 
                "tool_arguments": call.get('arguments', {})
            }])
            return outputs
        except Exception as e:
            return {'error': str(e)}
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
    def think_node(self, state: TrajectoryState):
        """Node: Generate model response with reasoning"""
        # ALWAYS start with the original prompt (system + user)
        system_message = [msg["content"] for msg in state["prompt"] if msg.get('role') in ['system']]
        user_query = [msg["content"] for msg in state["prompt"] if msg.get('role') in ['user']]
        
        # Get accumulated conversation messages
        accumulated_messages = state.get("messages", [])
        
        # Combine: prompt + all accumulated messages
        # print("Conversation: ", accumulated_messages)
        # Call model with full conversation history
        think_prompt = f"""
        User Query: {user_query}
        Assistant Progress: {accumulated_messages}

        Think: What do I need to do next? List remaining subtasks if any.
        Check the Assistant Progress to not to repeat the same thought/mistake from past observation"""
        message = [{"role": "system", "content": system_message[0]}, {"role": "user", "content": think_prompt}]
        response = self.llm.local(message)
        # print("Think Node Output: ", response.content)
        
        # Return state update dictionary
        return {
            "messages": [{
                "role": "assistant",
                "content": response.content,
                "type": "thought"
            }],
            "trajectory": [{
                "step": state["current_step"],
                "node": "think",
                "output": response.content
            }]
        }

    def action_node(self, state: TrajectoryState):
        """Node: Decide on action (tool call or final answer)"""
        user_query = [msg for msg in state["prompt"] if msg.get('role') in ['user']]
        
        # Get accumulated conversation messages
        last_thought = state.get("messages", [])[-1]
        
        # Combine: prompt + all accumulated messages
        # print("Conversation: ",full_conversation)
        action_prompt = f"""
        Task: {user_query}
        Assistant Thought: {last_thought}
        Step {state["current_step"]}/{state["max_steps"]}

        Respond with EXACT JSON format:

        Tool call:
        ```json
        {{"action": "<tool_name>", "action_input": {{"param": "value"}}}}
        ```

        Final answer (ONLY if all subtasks done):
        ```json
        {{"action": "Final Answer", "action_input": "Complete summary"}}
        ```

        Available Tools: {self.available_tools}"""
        
        response = self.llm.local(action_prompt)
        
        print("Action Node Output: ", response.content)
        parsed_action  = self.parse_json(response.content)
        if(parsed_action["action"] == "Final Answer"):
            return {
            "messages": [{
                "role": "assistant",
                "observation": parsed_action["action_input"],
                "type": "action",
               
            }],
            "trajectory": [{
                "step": state["current_step"],
                "node": "action",
                "output":  parsed_action["action_input"],
                "is_final_answer": "is_final_answer"
            }],
            "current_step": state["current_step"] + 1,
            "reached_final_answer": True,
            "final_answer":  parsed_action["action_input"]
            }
        tool_output = self.execute_tool({"tool_name": parsed_action["action"], "arguments":parsed_action["action_input"]})
        # Parse the response to determine action type
        
        print("Tool Output: ",tool_output)
        # Return state update dictionary
        return {
            "messages": [{
                "role": "assistant",
                "observation": tool_output,
                "type": "action",
               
            }],
            "trajectory": [{
                "step": state["current_step"],
                "node": "action",
                "output": tool_output,
               
                "is_final_answer": ""
            }],
            "current_step": state["current_step"] + 1,
            "reached_final_answer": False,
            "final_answer": ""
        }
    def check_final_answer(self, state: TrajectoryState) -> str:
        """
        Determine routing after action_node
        Returns: 'continue', 'ANSWER', or 'max_steps'
        """
        # Check if max steps reached
        if state["current_step"] >= state["max_steps"]:
            return "max_steps"
        
        # Check if final answer was reached
        if state["reached_final_answer"]:
            return "ANSWER"
        
        # Continue with more steps
        return "continue"
    def build_trajectory_graph(self):
        """Build LangGraph for trajectory generation"""
        builder = StateGraph(TrajectoryState)
        
        # Add nodes
        builder.add_node("think_node", self.think_node)
        builder.add_node("action_node", self.action_node)
        # builder.add_node("execute_tools", self.execute_tools)
        builder.add_node("finalize", self.finalize_trajectory)
        
        # Define edges
        builder.add_edge(START, "think_node")
        
        builder.add_edge("think_node", "action_node")
        # Conditional routing after model call
        builder.add_conditional_edges(
            "action_node",
            self.check_final_answer,
            {
                "continue": "think_node",
                "ANSWER": "finalize",
                "max_steps": "finalize"
            }
        )
        
        # Loop back after tool execution
        # builder.add_edge("action_node", "think_node")
        builder.add_edge("finalize", END)

        
        return builder.compile()

    def generate_trajectory(self, prompt, max_steps, expert_trajectories):
        """
        Generate a single trajectory using LangGraph
        Returns: trajectory steps and final outcome reward
        """
        graph = self.build_trajectory_graph()
        
        # Initialize state
        initial_state = {
            "prompt": prompt,
            "messages": [],
            "trajectory": [],
            "current_step": 0,
            "max_steps": max_steps,
            "reached_final_answer": False,
            "final_answer": "",
            "outcome_reward": 0.0,
            "expert_trajectories": expert_trajectories
        }
        
        # Run the graph
        final_state = graph.invoke(initial_state)
        
        return {
            'trajectory': final_state['trajectory'],
            'conversation': final_state['messages'],
            'reached_final_answer': final_state['reached_final_answer'],
            'outcome_reward': final_state['outcome_reward']
        }

    def calculate_outcome_reward(self, conversation, ground_truth):
        """
        Calculate reward by comparing generated trajectory with ground truth
        """
        try:
            # Use your Reward class to calculate reward
            reward = self.reward_calculator.compute_reward(
                prompt=conversation,
                generated_trajectory=conversation,
                dataset_item=ground_truth
            )
            return reward
        except Exception as e:
            print(f"Warning: Reward calculation failed: {str(e)}")
            return 0.0

    def finalize_trajectory(self, state: TrajectoryState):
        """Node: Calculate outcome reward and finalize"""
        conversation = state["messages"]
        
        # Calculate outcome reward using ground truth
        outcome_reward = self.calculate_outcome_reward(
            conversation, 
            state["expert_trajectories"]
        )
        
        return {
            "reached_final_answer": True,
            "outcome_reward": outcome_reward
        }

   


# ============================================
# DATASET PROCESSING FUNCTIONS
# ============================================

def process_dataset_batch(trajectory_gen, dataset, output_dir, batch_size=32, max_steps=10, save_interval=100):
    """
    Process entire dataset in batches and generate trajectories
    """
    all_results = []
    
    # Use PyTorch DataLoader for efficient batching
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # Process one at a time for LangGraph
        shuffle=False,
        num_workers=0  # Single process for LangGraph
    )
    
    print(f"\n{'='*60}")
    print(f"🚀 Processing {len(dataset)} samples")
    print(f"{'='*60}\n")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating trajectories")):
        try:
            # Extract data from batch (batch_size=1)
            prompt = batch['prompt'][0]
            ground_truth = batch['ground_truth'][0]
            query = batch['query'][0]
            idx = batch['idx'][0].item()
            
            # Generate trajectory
            result = trajectory_gen.generate_trajectory(
                prompt=prompt,
                max_steps=max_steps,
                expert_trajectories=ground_truth
            )
            
            # Add metadata
            result['sample_idx'] = idx
            result['query'] = query
            result['ground_truth'] = ground_truth
            
            all_results.append(result)
            
            # Periodic saving
            if (batch_idx + 1) % save_interval == 0:
                save_trajectories(all_results, output_dir, f"checkpoint_{batch_idx+1}")
                print(f"\n💾 Checkpoint saved at {batch_idx+1} samples")
        
        except Exception as e:
            print(f"\n❌ Error processing sample {idx}: {str(e)}")
            # Save error info
            all_results.append({
                'sample_idx': idx,
                'error': str(e),
                'query': query if 'query' in batch else None
            })
    
    return all_results


def save_trajectories(results, output_dir, filename="trajectories"):
    """Save trajectories to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{filename}.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Saved {len(results)} trajectories to: {output_path}")
    return output_path


def analyze_results(results):
    """Analyze trajectory generation results"""
    total = len(results)
    successful = sum(1 for r in results if 'error' not in r)
    failed = total - successful
    
    # Calculate statistics
    rewards = [r['outcome_reward'] for r in results if 'outcome_reward' in r]
    steps = [len(r['trajectory']) for r in results if 'trajectory' in r]
    final_answers = sum(1 for r in results if r.get('reached_final_answer', False))
    
    print(f"\n{'='*60}")
    print("📊 Trajectory Generation Analysis")
    print(f"{'='*60}")
    print(f"Total samples:           {total}")
    print(f"✅ Successful:            {successful} ({successful/total*100:.1f}%)")
    print(f"❌ Failed:                {failed} ({failed/total*100:.1f}%)")
    print(f"\nTrajectory Statistics:")
    print(f"  • Avg steps per traj:  {sum(steps)/len(steps):.2f}" if steps else "  • N/A")
    print(f"  • Min steps:           {min(steps)}" if steps else "  • N/A")
    print(f"  • Max steps:           {max(steps)}" if steps else "  • N/A")
    print(f"\nReward Statistics:")
    print(f"  • Avg reward:          {sum(rewards)/len(rewards):.4f}" if rewards else "  • N/A")
    print(f"  • Min reward:          {min(rewards):.4f}" if rewards else "  • N/A")
    print(f"  • Max reward:          {max(rewards):.4f}" if rewards else "  • N/A")
    print(f"\nFinal Answers:          {final_answers}/{successful}")
    print(f"{'='*60}\n")


def prepare_grpo_dataset(results, output_path):
    """
    Prepare dataset for GRPO training from trajectory results
    """
    grpo_data = []
    
    for result in results:
        if 'error' in result:
            continue
        
        # Format for GRPO training
        grpo_sample = {
            'prompt': result.get('query', ''),
            'trajectory': result['trajectory'],
            'reward': result['outcome_reward'],
            'ground_truth': result.get('ground_truth', []),
            'reached_final_answer': result.get('reached_final_answer', False)
        }
        
        grpo_data.append(grpo_sample)
    
    # Save GRPO dataset
    with open(output_path, 'w') as f:
        json.dump(grpo_data, f, indent=2, default=str)
    
    print(f"📦 GRPO dataset prepared: {len(grpo_data)} samples")
    print(f"   Saved to: {output_path}")
    
    return grpo_data


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution with TracjDataset"""
    
    print("\n" + "="*60)
    print("🚀 LangGraph Trajectory Generation with TracjDataset")
    print("="*60 + "\n")
    
    # Paths to model checkpoint, training data, and output directory
    model_path = "path/to/your/base/model"  # Replace with your model path or name
    train_data_path = "path/to/your/training/data.json"  # Replace with your training data path
    tools_path = "path/to/Task_Generation_sft_batch2_copy/utils/tools.json"  # Replace with your tools.json path
    output_dir = "path/to/save/trajectories"  # Replace with your desired output directory
    
    # Load dataset
    print("📂 Loading dataset...")
    with open(train_data_path, "r") as f:
        raw_data = json.load(f)
    print(f"   ✓ Training data length: {len(raw_data)}")
    
    # Create TracjDataset
    dataset = TracjDataset(raw_data)
    print(f"   ✓ Dataset initialized: {len(dataset)} samples\n")
    
    # Initialize trajectory generator
    print("🔧 Initializing trajectory generator...")
    trajectory_gen = Trajectories()
    print("   ✓ Generator ready\n")
    
    # Example 1: Process single sample
    print("="*60)
    print("📝 Example 1: Single Sample Generation")
    print("="*60)
    
    sample = dataset[110]
    print(f"\nQuery: {sample['query']}...")
    print(f"Ground truth steps: {len(sample['ground_truth'])}")
    
    result = trajectory_gen.generate_trajectory(
        prompt=sample['prompt'],
        max_steps=25,
        expert_trajectories=sample['ground_truth']
    )
    
    print(f"\n✅ Generated trajectory:")
    print(f"   • Steps: {len(result['trajectory'])}")
    print(f"   • Reward: {result['outcome_reward']:.4f}")
    print(f"   • Reached final answer: {result['reached_final_answer']}")
    
    # Example 2: Process first N samples
    # print("\n\n" + "="*60)
    # print("📝 Example 2: Batch Processing (First 10 samples)")
    # print("="*60 + "\n")
    
    # # Create subset for testing
    # subset_size = min(10, len(dataset))
    # subset_data = [dataset[i] for i in range(subset_size)]
    # subset_dataset = TracjDataset(subset_data)
    
    # # Process batch
    # batch_results = process_dataset_batch(
    #     trajectory_gen=trajectory_gen,
    #     dataset=subset_dataset,
    #     batch_size=1,
    #     max_steps=10,
    #     save_interval=5
    # )
    
    # # Analyze results
    # analyze_results(batch_results)
    
    # # Save results
    # results_path = save_trajectories(
    #     batch_results, 
    #     output_dir, 
    #     "test_trajectories"
    # )
    
    # # Example 3: Prepare GRPO training dataset
    # print("\n" + "="*60)
    # print("📝 Example 3: Prepare GRPO Dataset")
    # print("="*60 + "\n")
    
    # grpo_dataset_path = os.path.join(output_dir, "grpo_training_data.json")
    # grpo_data = prepare_grpo_dataset(batch_results, grpo_dataset_path)
    
    # Example 4: Process full dataset (uncomment when ready)
    # print("\n" + "="*60)
    # print("📝 Example 4: Full Dataset Processing")
    # print("="*60 + "\n")
    
    # full_results = process_dataset_batch(
    #     trajectory_gen=trajectory_gen,
    #     dataset=dataset,
    #     batch_size=1,
    #     max_steps=10,
    #     save_interval=100
    # )
    
    # analyze_results(full_results)
    # save_trajectories(full_results, output_dir, "full_trajectories")
    # prepare_grpo_dataset(full_results, os.path.join(output_dir, "full_grpo_data.json"))
    
    # print("\n" + "="*60)
    # print("✨ Trajectory generation complete!")
    # print("="*60 + "\n")


# ============================================
# UTILITY FUNCTIONS
# ============================================

# def compare_trajectories(generated, ground_truth):
#     """
#     Compare generated trajectory with ground truth
#     """
#     comparison = {
#         'num_steps_generated': len(generated['trajectory']),
#         'num_steps_ground_truth': len(ground_truth),
#         'reward': generated['outcome_reward'],
#         'tool_matches': 0,
#         'tool_sequence': []
#     }
    
#     # Extract tool calls from generated trajectory
#     gen_tools = []
#     for step in generated['trajectory']:
#         if 'tool_call' in step:
#             tool_name = step['tool_call'].get('name', '')
#             gen_tools.append(tool_name)
#             comparison['tool_sequence'].append(tool_name)
    
#     # Compare with ground truth tools
#     gt_tools = [msg.get('name', '') for msg in ground_truth if msg.get('role') == 'tool']
    
#     # Calculate tool matches
#     comparison['tool_matches'] = sum(1 for gt, gen in zip(gt_tools, gen_tools) if gt == gen)
#     comparison['tool_accuracy'] = comparison['tool_matches'] / max(len(gt_tools), 1)
    
#     return comparison


if __name__ == "__main__":
    main()
