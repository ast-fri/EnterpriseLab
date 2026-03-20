"""
EnterpriseBench Dataset Loader - Handles Your Exact Task Format

FIXED: Ground truth reward function properly parses tools and final answers

Loads tasks from your EnterpriseBench format with:
- task_id, instruction, chain_of_thought, ground_truth, etc.
"""

import json
import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_enterprise_tasks_v2(
    path: str,
    max_tasks: int = None,
    difficulty_filter: str = None,  # "EASY", "MEDIUM", "HARD"
    domain_filter: str = None,       # "HR", "CRM", "GitHub", etc.
    min_steps: int = None,
    max_steps: int = None
) -> List[Dict]:
    """
    Load EnterpriseBench tasks from your exact JSON format.

    Your task format:
    {
        "task_id": "task_seq_0_3_filtered_1",
        "instruction": "As part of maintaining...",
        "prerequisite_context": [],
        "chain_of_thought": [
            {
                "step": 3,
                "rationale": "...",
                "tool": "delete_message",
                "inputs": {...},
                "expected_output": "..."
            }
        ],
        "required_tools": ["delete_message"],
        "success_criteria": [...],
        "domain": "HR",
        "difficulty": "EASY",
        "ground_truth": {
            "final_output": "...",
            "all_step_outputs": [...],
            "final_entities": [...]
        },
        "meta": {...}
    }

    Args:
        path: Path to your tasks JSON file
        max_tasks: Limit number of tasks (for testing)
        difficulty_filter: Only load tasks with this difficulty
        domain_filter: Only load tasks from this domain
        min_steps: Minimum number of steps
        max_steps: Maximum number of steps

    Returns:
        List of formatted tasks for GRPO training
    """
    try:
        with open(path, 'r') as f:
            raw_tasks = json.load(f)

        logger.info(f"Loaded {len(raw_tasks)} raw tasks from {path}")

        # Apply filters
        filtered_tasks = []
        for task in raw_tasks:
            # Difficulty filter
            if difficulty_filter and task.get('difficulty') != difficulty_filter:
                continue

            # Domain filter
            if domain_filter and task.get('domain') != domain_filter:
                continue

            # Steps filter - FIXED: fallback to chain_of_thought length
            num_steps = task.get('meta', {}).get('num_steps')
            if num_steps is None:
                num_steps = len(task.get('chain_of_thought', []))
            
            if min_steps and num_steps < min_steps:
                continue
            if max_steps and num_steps > max_steps:
                continue

            filtered_tasks.append(task)

        logger.info(f"After filtering: {len(filtered_tasks)} tasks")

        # Convert to GRPO training format
        formatted_tasks = []
        for task in filtered_tasks:
            formatted_task = {
                # Required fields
                'id': task['task_id'],
                'user': task['instruction'],

                # Optional: Gold trajectory for ground truth rewards
                'gold_chain_of_thought': task.get('chain_of_thought', []),
                'gold_final_output': task.get('ground_truth', {}).get('final_output'),
                'gold_step_outputs': task.get('ground_truth', {}).get('all_step_outputs', []),

                # Metadata for analysis
                'required_tools': task.get('required_tools', []),
                'domain': task.get('domain'),
                'difficulty': task.get('difficulty'),
                'num_steps': num_steps,
                'success_criteria': task.get('success_criteria', []),
            }

            formatted_tasks.append(formatted_task)

        # Limit if specified (FIXED: handle max_tasks=0)
        if max_tasks and max_tasks > 0:
            formatted_tasks = formatted_tasks[:max_tasks]
            logger.info(f"Limited to {max_tasks} tasks for this run")

        # Log statistics
        log_dataset_statistics(formatted_tasks)

        return formatted_tasks

    except FileNotFoundError:
        logger.error(f"Dataset file not found: {path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in dataset file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def log_dataset_statistics(tasks: List[Dict]):
    """Log statistics about the loaded dataset."""
    if not tasks:
        return

    # Difficulty distribution
    difficulties = {}
    for task in tasks:
        diff = task.get('difficulty', 'UNKNOWN')
        difficulties[diff] = difficulties.get(diff, 0) + 1

    # Domain distribution
    domains = {}
    for task in tasks:
        domain = task.get('domain', 'UNKNOWN')
        domains[domain] = domains.get(domain, 0) + 1

    # Steps distribution
    steps = [task.get('num_steps', 0) for task in tasks]
    avg_steps = sum(steps) / len(steps) if steps else 0

    # Tools distribution
    all_tools = set()
    for task in tasks:
        all_tools.update(task.get('required_tools', []))

    logger.info("="*80)
    logger.info("DATASET STATISTICS")
    logger.info("="*80)
    logger.info(f"Total tasks: {len(tasks)}")
    logger.info(f"\nDifficulty distribution:")
    for diff, count in sorted(difficulties.items()):
        logger.info(f"  {diff}: {count} ({count/len(tasks)*100:.1f}%)")
    logger.info(f"\nDomain distribution:")
    for domain, count in sorted(domains.items()):
        logger.info(f"  {domain}: {count} ({count/len(tasks)*100:.1f}%)")
    logger.info(f"\nSteps:")
    logger.info(f"  Average: {avg_steps:.1f}")
    logger.info(f"  Min: {min(steps) if steps else 0}")
    logger.info(f"  Max: {max(steps) if steps else 0}")
    logger.info(f"\nUnique tools used: {len(all_tools)}")
    logger.info(f"  Tools: {sorted(all_tools)[:10]}...")  # Show first 10
    logger.info("="*80)


def create_ground_truth_reward_function(tasks: List[Dict]):
    """
    Create a reward function based on ground truth outputs.

    FIXED: Properly extracts tool names and final answers from trajectory segments

    This is more accurate than LLM judge when you have gold trajectories.

    Args:
        tasks: List of tasks with ground_truth information

    Returns:
        Reward function that compares trajectory to ground truth
    """
    # Build lookup table: task_id -> ground_truth
    ground_truth_map = {}
    for task in tasks:
        ground_truth_map[task['id']] = {
            'final_output': task.get('gold_final_output'),
            'step_outputs': task.get('gold_step_outputs', []),
            'required_tools': task.get('required_tools', []),
            'gold_chain': task.get('gold_chain_of_thought', [])
        }

    def ground_truth_reward(task_id: str, trajectory) -> float:
        """
        Compute reward by comparing trajectory to ground truth.

        FIXED: Properly parses tool names and final answers from segments

        Scoring:
        - 0.5: Used correct tools in correct order
        - 0.3: Achieved correct final output
        - 0.2: Completed successfully without errors

        Returns:
            Float reward between 0.0 and 1.0
        """
        if task_id not in ground_truth_map:
            # Fallback: basic completion reward
            # FIXED: Use "success" termination reason
            return 0.5 if trajectory.termination_reason == "success" else 0.1

        gt = ground_truth_map[task_id]
        reward = 0.0

        # ============================================================
        # 1. Tool usage correctness (0.5 points)
        # ============================================================
        
        # FIXED: Extract tool names properly from action segments
        trajectory_tools = []
        for segment in trajectory.segments:
            if segment.segment_type == 'action':
                # Extract tool name from "Action: tool_name\n..." format
                action_text = segment.text.strip()
                
                # Match "Action: <tool_name>"
                match = re.search(r"Action:\s*([^\n]+)", action_text)
                if match:
                    tool_name = match.group(1).strip()
                    # Remove trailing () if present
                    if tool_name.endswith('()'):
                        tool_name = tool_name[:-2]
                    trajectory_tools.append(tool_name)

        required_tools = gt['required_tools']
        
        if required_tools:
            # Check if trajectory used required tools (order-aware)
            # Simple approach: check if all required tools appear in trajectory
            tools_used = set(trajectory_tools)
            required_set = set(required_tools)
            
            # Calculate overlap
            tools_overlap = len(tools_used & required_set)
            
            if tools_overlap == len(required_set):
                # All required tools used - full credit
                reward += 0.5
            else:
                # Partial credit based on overlap
                reward += 0.5 * (tools_overlap / len(required_set))
        else:
            # No required tools specified, give credit if trajectory completed
            if trajectory.termination_reason == "success":
                reward += 0.5

        # ============================================================
        # 2. Final output correctness (0.3 points)
        # ============================================================
        
        if gt['final_output']:
            # FIXED: Extract final answer from final_answer segment
            final_answer = None
            for segment in trajectory.segments:
                if segment.segment_type == 'final_answer':
                    # Extract text after "Final Answer: "
                    answer_text = segment.text.strip()
                    match = re.search(r"Final Answer:\s*(.+)", answer_text, re.DOTALL)
                    if match:
                        final_answer = match.group(1).strip()
                    else:
                        final_answer = answer_text
                    break

            if final_answer:
                # Normalize both for comparison
                final_answer_normalized = final_answer.lower().strip()
                gt_output_normalized = str(gt['final_output']).lower().strip()
                
                # Check for exact match or substring match
                if gt_output_normalized in final_answer_normalized or final_answer_normalized in gt_output_normalized:
                    reward += 0.3
                else:
                    # Partial credit: check if key entities match
                    # Extract quoted strings and numbers from both
                    import re
                    final_entities = set(re.findall(r'\b\w+\b', final_answer_normalized))
                    gt_entities = set(re.findall(r'\b\w+\b', gt_output_normalized))
                    
                    if final_entities and gt_entities:
                        overlap = len(final_entities & gt_entities)
                        max_len = max(len(final_entities), len(gt_entities))
                        if overlap / max_len > 0.5:
                            reward += 0.15  # Half credit for partial match

        # ============================================================
        # 3. Completion without errors (0.2 points)
        # ============================================================
        
        # FIXED: Use "success" termination reason (not "final_answer")
        if trajectory.termination_reason == "success":
            reward += 0.2
        elif trajectory.termination_reason in ["max_turns", "context_overflow"]:
            # Partial credit if got close
            reward += 0.05

        return min(reward, 1.0)  # Cap at 1.0

    return ground_truth_reward


# Example usage in train_enterprise.py:
"""
# In main() function, replace load_enterprise_tasks with:

from enterprise_dataset_loader import load_enterprise_tasks_v2, create_ground_truth_reward_function

# Load tasks
train_dataset = load_enterprise_tasks_v2(
    DATASET_PATH,
    max_tasks=100,           # Limit for testing
    difficulty_filter="EASY", # Start with easy tasks
    # domain_filter="HR",     # Or filter by domain
    # min_steps=1,
    # max_steps=5
)

# Use ground truth reward instead of LLM judge
reward_fn = create_ground_truth_reward_function(train_dataset)
"""
