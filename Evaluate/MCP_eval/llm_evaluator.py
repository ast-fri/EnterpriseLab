from __future__ import annotations


"""Multi-aspect LLM judge for AI agent execution trajectory evaluation.

This evaluator asks a large-language model to grade an AI agent's execution 
trajectory on a given task. The trajectory includes the full conversation flow:
system message, user task, agent reasoning, tool calls, tool responses, and 
final agent response.

The evaluator provides separate evaluation methods for:
1. **Trajectory Quality** - How well the agent navigates the task execution (including tool usage)
2. **Task Completion** - How well the final outcome addresses the original task

Each method returns individual scores for detailed analysis.

Returned JSON schema requested from the judge LLM (default):

```
{
  "trajectory": {
    "planning": float,              # 0-1 - task decomposition and approach
    "execution_flow": float,        # 0-1 - logical sequence of actions
    "tool_selection": float,        # 0-1 - appropriate tools chosen
    "tool_usage": float,           # 0-1 - correct parameters and interpretation
    "error_handling": float,        # 0-1 - recovery from issues/errors
    "efficiency": float            # 0-1 - optimal use of tools and steps
  },
  "task_completion": {
    "requirement_coverage": float,  # addresses all task requirements
    "accuracy": float,             # correctness of information/analysis
    "completeness": float,         # thoroughness of response
    "usefulness": float           # practical value to user
  },
  "overall": float,               # optional - judge may return, else evaluator computes average
  "comments": str                 # short explanation / feedback
}
```
"""

from tool_evaluation import TrajectoryAdapter
from tqdm import tqdm
from dataclasses import dataclass, field
import json
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator
import openai
from dotenv import load_dotenv
import os
import time


load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


class TrajectoryScores(BaseModel):
    planning: float = Field(..., ge=0.0, le=1.0)
    execution_flow: float = Field(..., ge=0.0, le=1.0)
    tool_selection: float = Field(..., ge=0.0, le=1.0)
    tool_usage: float = Field(..., ge=0.0, le=1.0)
    adaptability: float = Field(..., ge=0.0, le=1.0)
    efficiency: float = Field(..., ge=0.0, le=1.0)
    context_awareness: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_values(self):
        for field_name, value in self.model_dump().items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Field {field_name} must be between 0 and 1, got {value}"
                )
        return self


class TaskCompletionScores(BaseModel):
    requirement_coverage: float = Field(..., ge=0.0, le=1.0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    completeness: float = Field(..., ge=0.0, le=1.0)
    usefulness: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_values(self):
        for field_name, value in self.model_dump().items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Field {field_name} must be between 0 and 1, got {value}"
                )
        return self


class TrajectoryEvaluationResult(BaseModel):
    """Result from trajectory evaluation."""
    scores: TrajectoryScores
    overall_score: float = Field(..., ge=0.0, le=1.0)
    comments: str
    raw_response: Dict[str, Any]


class TaskCompletionEvaluationResult(BaseModel):
    """Result from task completion evaluation."""
    scores: TaskCompletionScores
    overall_score: float = Field(..., ge=0.0, le=1.0)
    comments: str
    raw_response: Dict[str, Any]


# ---------------------------------------------------------------------------
# Trajectory Adapter for LLM Judge
# ---------------------------------------------------------------------------


class TrajectoryFormatter:
    """Format trajectory data for LLM judge evaluation."""
    
    @staticmethod
    def extract_expected_approach(ground_truth_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract expected approach description from ground truth trajectory.
        
        Args:
            ground_truth_data: Ground truth with messages or trajectory
            
        Returns:
            String describing the expected approach
        """
        if "messages" in ground_truth_data:
            # Extract from messages format
            messages = ground_truth_data["messages"]
            approach_parts = []
            
            for msg in messages:
                if msg.get("role") == "assistant" and "tool_calls" not in msg:
                    # Assistant reasoning/thinking
                    approach_parts.append(msg.get("content", ""))
                elif msg.get("role") == "assistant" and "tool_calls" in msg:
                    # Tool calls
                    for tc in msg.get("tool_calls", []):
                        func = tc.get("function", {})
                        approach_parts.append(f"Use {func.get('name', 'tool')}")
            
            return " -> ".join(approach_parts) if approach_parts else None
        
        elif "trajectory" in ground_truth_data:
            # Extract from trajectory format
            trajectory = ground_truth_data["trajectory"]
            approach_parts = []
            
            for step in trajectory:
                if step.get("step_type") == "thought":
                    approach_parts.append(step.get("content", ""))
                elif step.get("step_type") == "action":
                    tool_used = step.get("tool_used")
                    if tool_used:
                        approach_parts.append(f"Use {tool_used}")
            
            return " -> ".join(approach_parts) if approach_parts else None
        
        return None
    
    
    @staticmethod
    def format_trajectory_for_judge(trajectory_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert trajectory format to conversation format for LLM judge.
        
        Args:
            trajectory_data: Dictionary containing trajectory with steps
            
        Returns:
            List of conversation messages
        """
        messages = []
        # print(trajectory_data)
        trajectory = trajectory_data.get("trajectory", [])
        
        for step in trajectory:
            step_type = step.get("step_type")
            
            if step_type == "thought":
                # Add thought as assistant message
                messages.append({
                    "role": "assistant",
                    "content": f"Thought: {step.get('content', '')}"
                })
            
            elif step_type == "action":
                # Format tool call
                tool_used = step.get("tool_used")
                tool_input = step.get("tool_input")
                if tool_used and tool_input is not None:
                    messages.append({
                        "role": "assistant",
                        "content": f"Action: Using {tool_used}",
                        "tool_calls": [{
                            "type": "function",
                            "function": {
                                "name": tool_used,
                                "arguments": json.dumps(tool_input)
                            }
                        }]
                    })
            
            elif step_type == "observation":
                # Add tool result
                tool_used = step.get("tool_used")
                tool_output = step.get("tool_output")
                if tool_used:
                    messages.append({
                        "role": "tool",
                        "name": tool_used,
                        "content": str(tool_output)
                    })
        
        return messages
    
    @staticmethod
    def extract_final_answer(trajectory_data: Dict[str, Any]) -> str:
        """Extract final answer from trajectory."""
        # First check if there's a final_answer field at the top level
        final_answer = trajectory_data.get("final_answer", "")
        if final_answer:
            return final_answer
        
        # If not, check the trajectory steps for a Final Answer action
        trajectory = trajectory_data.get("trajectory", [])
        for step in reversed(trajectory):
            if step.get("step_type") == "action":
                content = step.get("content", "")
                # Check if this is a final answer step
                if isinstance(content, str) and "Final Answer" in content:
                    # Try to extract the answer from the content
                    try:
                        # Handle dict-like string format
                        if "action_input" in content:
                            import ast
                            parsed = ast.literal_eval(content)
                            return parsed.get("action_input", "")
                    except:
                        pass
                
                # Also check tool_input for final answer
                tool_input = step.get("tool_input")
                if tool_input and isinstance(tool_input, dict):
                    if "action_input" in tool_input:
                        return tool_input["action_input"]
        
        return ""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

DEFAULT_TRAJECTORY_KEYS = [
    "planning",
    "execution_flow",
    "tool_selection",
    "tool_usage",
    "adaptability",
    "efficiency",
    "context_awareness",
]
DEFAULT_TASK_COMPLETION_KEYS = [
    "requirement_coverage",
    "accuracy",
    "completeness",
    "usefulness",
]


def clean_json_response(content: str) -> str:
    """Clean JSON response by removing markdown code blocks if present."""
    content = content.strip()
    
    # Remove markdown code blocks
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    
    if content.endswith("```"):
        content = content[:-3]
    
    return content.strip()


@dataclass
class MultiAspectLLMJudge:
    """LLM evaluator that grades AI agent execution trajectory."""
    
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    trajectory_keys: List[str] = field(default_factory=lambda: DEFAULT_TRAJECTORY_KEYS)
    task_completion_keys: List[str] = field(
        default_factory=lambda: DEFAULT_TASK_COMPLETION_KEYS
    )
    chat_kwargs: Dict[str, Any] = field(default_factory=dict)
    include_conversation: bool = True
    trajectory_prompt_template: Optional[str] = None
    task_completion_prompt_template: Optional[str] = None
    
    def __post_init__(self):
        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key} if self.api_key else {}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = openai.OpenAI(**client_kwargs)
    
    @classmethod
    def with_custom_prompts(
        cls,
        trajectory_prompt: Optional[str] = None,
        task_completion_prompt: Optional[str] = None,
        **kwargs,
    ) -> "MultiAspectLLMJudge":
        """Create an evaluator with custom prompt templates."""
        return cls(
            trajectory_prompt_template=trajectory_prompt,
            task_completion_prompt_template=task_completion_prompt,
            **kwargs,
        )
    
    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    
    def _build_trajectory_prompt(self) -> str:
        """Build prompt template specifically for trajectory evaluation."""
        if self.trajectory_prompt_template is not None:
            return self.trajectory_prompt_template
        
        return (
            "You are evaluating the execution trajectory of an AI agent. Focus on HOW the agent "
            "approached and executed the task, including all tool interactions.\n\n"
            "**TRAJECTORY EVALUATION CRITERIA:**\n"
            "- planning: How well did the agent understand and decompose the task? (0.0-1.0)\n"
            "- execution_flow: Was the sequence of actions logical and well-structured? (0.0-1.0)\n"
            "- tool_selection: Were the right tools chosen for each step? (0.0-1.0)\n"
            "- tool_usage: Were tool parameters correct and results properly interpreted? (0.0-1.0)\n"
            "- adaptability: How well did the agent handle errors, unexpected results, changing contexts, and alternative paths? "
            "Score 1.0 if no errors occurred and execution was smooth, or if errors/changes were handled excellently, "
            "0.8-0.9 if minor issues were handled well, 0.5-0.7 if some problems occurred but were adequately addressed, "
            "0.0-0.4 if errors or changes were poorly handled. (0.0-1.0)\n"
            "- efficiency: Did the agent use an optimal approach without unnecessary steps? (0.0-1.0)\n"
            "- context_awareness: Did the agent maintain awareness of relevant context, constraints, and environmental state throughout execution? (0.0-1.0)\n\n"
            "Analyze the complete conversation flow from start to finish. Look for:\n"
            "- Clear task understanding\n"
            "- Logical step progression\n"
            "- Appropriate tool selection and usage\n"
            "- Correct parameter formatting and values\n"
            "- Proper interpretation of tool responses\n"
            "- Effective integration of tool results\n"
            "- Recovery from any issues\n\n"
            "**IMPORTANT**: For adaptability, if the execution was smooth with no errors or changes needed, "
            "this indicates good robustness and should receive a high score (0.9-1.0). "
            "Only give low scores if actual errors or unexpected situations occurred and were handled poorly.\n\n"
            "Respond ONLY with JSON:\n"
            "{\n"
            '  "planning": float,\n'
            '  "execution_flow": float,\n'
            '  "tool_selection": float,\n'
            '  "tool_usage": float,\n'
            '  "adaptability": float,\n'
            '  "efficiency": float,\n'
            '  "context_awareness": float,\n'
            '  "comments": "Brief explanation"\n'
            "}"
        )
    
    def _build_task_completion_prompt(self) -> str:
        """Build prompt template specifically for task completion evaluation."""
        if self.task_completion_prompt_template is not None:
            return self.task_completion_prompt_template
        
        return (
            "You are evaluating how well an AI agent completed the assigned task. "
            "Focus on the final outcome and how well it addresses the original requirements.\n\n"
            "**TASK COMPLETION EVALUATION CRITERIA:**\n"
            "- requirement_coverage: Did the agent address all aspects of the task? (0.0-1.0)\n"
            "- accuracy: Is the information and analysis factually correct? (0.0-1.0)\n"
            "- completeness: Is the response thorough and comprehensive? (0.0-1.0)\n"
            "- usefulness: Is the final result practically valuable to the user? (0.0-1.0)\n\n"
            "Compare the final response against:\n"
            "- Original task requirements\n"
            "- Expected deliverables\n"
            "- Quality of information provided\n"
            "- Practical utility for the user\n\n"
            "Respond ONLY with JSON:\n"
            "{\n"
            '  "requirement_coverage": float,\n'
            '  "accuracy": float,\n'
            '  "completeness": float,\n'
            '  "usefulness": float,\n'
            '  "comments": "Brief explanation"\n'
            "}"
        )
    
    # ------------------------------------------------------------------
    # Evaluation Methods
    # ------------------------------------------------------------------
    
    def evaluate_trajectory(
        self,
        task: Dict[str, Any],
        execution_trajectory: List[Dict[str, Any]],
        expected_approach: Optional[str] = None,
        expected_tool_calls: Optional[List[Dict[str, Any]]] = None,
        **chat_kwargs,
    ) -> TrajectoryEvaluationResult:
        """Evaluate execution trajectory quality including tool usage."""
        user_parts = [
            "Task description:\n" + (task or ""),
            "Execution trajectory (complete conversation):\n"
            + json.dumps(execution_trajectory, ensure_ascii=False, indent=2),
        ]
        
        if expected_approach:
            user_parts.append("Expected approach:\n" + expected_approach)
        
        if expected_tool_calls:
            user_parts.append(
                "Expected tool calls:\n"
                + json.dumps(expected_tool_calls, ensure_ascii=False, indent=2)
            )
        
        messages = [
            {"role": "system", "content": self._build_trajectory_prompt()},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **{**self.chat_kwargs, **chat_kwargs}
            )
            content = response.choices[0].message.content
            # Clean the JSON response
            cleaned_content = clean_json_response(content)
            raw_response = json.loads(cleaned_content)
            
            # Extract scores
            trajectory_scores = {
                k: float(raw_response.get(k, 0.0)) for k in self.trajectory_keys
            }
            overall_score = (
                sum(trajectory_scores.values()) / len(trajectory_scores)
                if trajectory_scores
                else 0.0
            )
            comments = raw_response.get("comments", "")
            
            return TrajectoryEvaluationResult(
                scores=TrajectoryScores(**trajectory_scores),
                overall_score=overall_score,
                comments=comments,
                raw_response=raw_response,
            )
        
        except Exception as exc:
            logger.error("Trajectory evaluation failed – %s", exc)
            try:
                content = response.choices[0].message.content
                logger.error("LLM response content: %r", content)
            except:
                logger.error("Could not extract response content")
            zero_scores = {k: 0.0 for k in self.trajectory_keys}
            return TrajectoryEvaluationResult(
                scores=TrajectoryScores(**zero_scores),
                overall_score=0.0,
                comments=f"Error: {exc}",
                raw_response={},
            )
    
    def evaluate_task_completion(
        self,
        task: Dict[str, Any],
        final_response: str,
        execution_trajectory: Optional[List[Dict[str, Any]]] = None,
        ground_truth_answer: Optional[str] = None,
        **chat_kwargs,
    ) -> TaskCompletionEvaluationResult:
        """Evaluate task completion quality."""
        user_parts = [
            "Task description:\n" + (task or ""),
            "Agent's final response:\n" + final_response,
        ]
        
        if execution_trajectory and self.include_conversation:
            user_parts.append(
                "Full execution trajectory:\n"
                + json.dumps(execution_trajectory, ensure_ascii=False, indent=2)
            )
        
        if ground_truth_answer:
            user_parts.append("Expected answer:\n" + ground_truth_answer)
        
        messages = [
            {"role": "system", "content": self._build_task_completion_prompt()},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **{**self.chat_kwargs, **chat_kwargs}
            )
            content = response.choices[0].message.content
            # Clean the JSON response
            cleaned_content = clean_json_response(content)
            raw_response = json.loads(cleaned_content)
            
            # Extract scores
            task_completion_scores = {
                k: float(raw_response.get(k, 0.0)) for k in self.task_completion_keys
            }
            overall_score = (
                sum(task_completion_scores.values()) / len(task_completion_scores)
                if task_completion_scores
                else 0.0
            )
            comments = raw_response.get("comments", "")
            
            return TaskCompletionEvaluationResult(
                scores=TaskCompletionScores(**task_completion_scores),
                overall_score=overall_score,
                comments=comments,
                raw_response=raw_response,
            )
        
        except Exception as exc:
            logger.error("Task completion evaluation failed – %s", exc)
            try:
                content = response.choices[0].message.content
                logger.error("LLM response content: %r", content)
            except:
                logger.error("Could not extract response content")
            zero_scores = {k: 0.0 for k in self.task_completion_keys}
            return TaskCompletionEvaluationResult(
                scores=TaskCompletionScores(**zero_scores),
                overall_score=0.0,
                comments=f"Error: {exc}",
                raw_response={},
            )
    
    # ------------------------------------------------------------------
    # Trajectory-specific evaluation methods
    # ------------------------------------------------------------------
    
    def evaluate_trajectory_from_data(
        self,
        task_data: Dict[str, Any],
        trajectory_data: Dict[str, Any],
        ground_truth_data: Optional[Dict[str, Any]] = None,
        **chat_kwargs,
    ) -> TrajectoryEvaluationResult:
        """
        Evaluate trajectory from your custom format.
        
        Args:
            task_data: Task information with description/goal
            trajectory_data: Trajectory with steps
            ground_truth_data: Optional ground truth with expected tool calls
            **chat_kwargs: Additional chat completion arguments
        """
        # Check if trajectory is empty or has error
        trajectory = trajectory_data.get("trajectory", [])
        status = trajectory_data.get("status", "")
        error = trajectory_data.get("error", "")
        
        # If trajectory is empty or has error status, return zero scores
        if not trajectory or status == "error" or error:
            zero_scores = {k: 0.0 for k in self.trajectory_keys}
            error_msg = error if error else "Empty trajectory or execution failed"
            return TrajectoryEvaluationResult(
                scores=TrajectoryScores(**zero_scores),
                overall_score=0.0,
                comments=f"Execution failed: {error_msg}",
                raw_response={},
            )
        
        # Format trajectory for judge
        formatted_trajectory = TrajectoryFormatter.format_trajectory_for_judge(trajectory_data)
        
        # Extract expected approach and tool calls from ground truth
        expected_approach = None
        expected_tool_calls = None
        
        if ground_truth_data:
            expected_approach = TrajectoryFormatter.extract_expected_approach(ground_truth_data)
            
            if "tool_calls" in ground_truth_data:
                expected_tool_calls = ground_truth_data["tool_calls"]
            elif "messages" in ground_truth_data:
                expected_tool_calls = []
                for msg in ground_truth_data["messages"]:
                    if msg.get("role") == "assistant" and "tool_calls" in msg:
                        for tc in msg.get("tool_calls", []):
                            func = tc.get("function", {})
                            expected_tool_calls.append({
                                "tool_name": func.get("name"),
                                "tool_parameters": func.get("arguments", {})
                            })
        
        return self.evaluate_trajectory(
            task=task_data,
            execution_trajectory=formatted_trajectory,
            expected_approach=expected_approach,
            expected_tool_calls=expected_tool_calls,
            **chat_kwargs
        )

    def evaluate_task_completion_from_data(
        self,
        task_data: Dict[str, Any],
        trajectory_data: Dict[str, Any],
        ground_truth_data: Optional[Dict[str, Any]] = None,
        **chat_kwargs,
    ) -> TaskCompletionEvaluationResult:
        """
        Evaluate task completion from your custom format.
        
        Args:
            task_data: Task information with description/goal
            trajectory_data: Trajectory with steps and final_answer
            ground_truth_data: Optional ground truth with expected answer
            **chat_kwargs: Additional chat completion arguments
        """
        # Extract final answer first
        final_answer = TrajectoryFormatter.extract_final_answer(trajectory_data)
        
        # Check if trajectory has error or is empty
        status = trajectory_data.get("status", "")
        error = trajectory_data.get("error", "")
        
        # Only return zero scores if there's an actual error AND no final answer
        if error and not final_answer:
            zero_scores = {k: 0.0 for k in self.task_completion_keys}
            return TaskCompletionEvaluationResult(
                scores=TaskCompletionScores(**zero_scores),
                overall_score=0.0,
                comments=f"Task incomplete: {error}",
                raw_response={},
            )
        
        # If status is error but there IS a final answer, still evaluate it
        # If no final answer at all, return zero
        if not final_answer:
            zero_scores = {k: 0.0 for k in self.task_completion_keys}
            return TaskCompletionEvaluationResult(
                scores=TaskCompletionScores(**zero_scores),
                overall_score=0.0,
                comments="No final answer provided",
                raw_response={},
            )
        
        # Format trajectory for context
        formatted_trajectory = None
        if self.include_conversation:
            formatted_trajectory = TrajectoryFormatter.format_trajectory_for_judge(trajectory_data)
        
        # Extract ground truth answer if available
        ground_truth_answer = None
        if ground_truth_data:
            if "messages" in ground_truth_data:
                for msg in reversed(ground_truth_data["messages"]):
                    if msg.get("role") == "assistant" and "tool_calls" not in msg:
                        ground_truth_answer = msg.get("content")
                        break
        
        return self.evaluate_task_completion(
            task=task_data,
            final_response=final_answer,
            execution_trajectory=formatted_trajectory,
            ground_truth_answer=ground_truth_answer,
            **chat_kwargs
        )


    def evaluate_both(
        self,
        task_data: Dict[str, Any],
        trajectory_data: Dict[str, Any],
        ground_truth_data: Optional[Dict[str, Any]] = None,
        **chat_kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate both trajectory and task completion.
        
        Returns:
            Dictionary with both evaluation results
        """
        trajectory_result = self.evaluate_trajectory_from_data(
            task_data, trajectory_data, ground_truth_data, **chat_kwargs
        )
        
        task_completion_result = self.evaluate_task_completion_from_data(
            task_data, trajectory_data, ground_truth_data, **chat_kwargs
        )
        
        return {
            "trajectory": trajectory_result.model_dump(),
            "task_completion": task_completion_result.model_dump(),
            "combined_score": (trajectory_result.overall_score + task_completion_result.overall_score) / 2
        }
    

    def batch_evaluate_with_llm_judge(
        self,
        gold_path: str,
        pred_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run LLM-judge evaluation for a batch of trajectories.

        - Loads gold and predicted trajectories using TrajectoryAdapter helpers.
        - Aligns them by task_id / query.
        - Calls self.evaluate_both(...) for each matched task.
        - Optionally saves results to JSON.
        
        Returns:
            summary dict with per-task results and aggregate stats.
        """
        # 1. Load & normalize trajectories
        ground_truth_list: List[Dict[str, Any]] = TrajectoryAdapter.load_gold_trajectories(
            gold_path
        )
        pred_list: List[Dict[str, Any]] = TrajectoryAdapter.load_pred_trajectories(
            pred_path
        )

        # 2. Index by task_id (query string)
        gt_dict = {gt["task_id"]: gt for gt in ground_truth_list}
        pred_dict = {p["task_id"]: p for p in pred_list}

        # 3. Only evaluate common tasks
        common_ids = sorted(set(gt_dict.keys()) & set(pred_dict.keys()))
        print(f"Found {len(common_ids)} matching tasks to evaluate with LLM judge")

        all_results: Dict[str, Any] = {}

        # common_ids = common_ids[10:20]  # For testing, limit to first 10 tasks
        for task_id in tqdm(common_ids, desc="LLM-judge evaluating tasks"):
            gold = gt_dict[task_id]
            pred = pred_dict[task_id]
            # print("##"*20)
            # print("Output trajectory:",pred)
            # Same as your single-trajectory code: use query string as `task_data`
            task_data = pred["query"]

            result = self.evaluate_both(
                task_data=task_data,
                trajectory_data=pred,
                ground_truth_data=gold,
            )

            all_results[task_id] = result
            time.sleep(1)

        # 4. Aggregate stats
        if all_results:
            combined_scores = [r["combined_score"] for r in all_results.values()]
            avg_combined = sum(combined_scores) / len(combined_scores)
        else:
            avg_combined = 0.0

        summary: Dict[str, Any] = {
            "num_tasks": len(common_ids),
            "average_combined_score": avg_combined,
            "results": all_results,
        }

        # 5. Optional save to JSON
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\nSaved LLM judge batch results to {output_path}")

        return summary


# Example usage
if __name__ == "__main__":
    
    #load the output and groundtruth trajectory

    #single trajectory evaluation
    ground_truth_traj_path = "gold_traj.json"
    pred_traj_path = "output_traj_2.json"

    
    with open(ground_truth_traj_path, "r", encoding="utf-8") as f:
        ground_truth_traj = json.load(f)

    with open(pred_traj_path, "r", encoding="utf-8") as f:
        pred_traj = json.load(f)

    #batch trajectory evaluation
    ground_truth_traj_path_batch = "path_to_gold_batch_trajectories.json"
    qwen_grpo_pred_traj_path_batch = "path_to_qwen_grpo_pred_batch_trajectories.json"
    # ground_truth_list = TrajectoryAdapter.load_gold_trajectories(ground_truth_traj_path_batch)
    # trajectory_list = TrajectoryAdapter.load_pred_trajectories(takane_dpo_pred_traj_path_batch)


    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
    # Create judge
    judge = MultiAspectLLMJudge(
        model="gpt-4o",
        api_key=api_key,  # or set OPENAI_API_KEY env var
        base_url=base_url
    )
    
    # Evaluate single trajectory
    # task = pred_traj["query"]
    # results = judge.evaluate_both(task, pred_traj,ground_truth_traj)
    # print(json.dumps(results, indent=2))

    # Evaluate batch trajectories
    summary = judge.batch_evaluate_with_llm_judge(
        gold_path=ground_truth_traj_path_batch,
        pred_path=qwen_grpo_pred_traj_path_batch,
        output_path="qwen_GRPO_llm_judge_batch_results.json",
    )

  