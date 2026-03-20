from typing import Dict, List, Any, Tuple, Optional, Literal
from pydantic import BaseModel, Field
import numpy as np
from tqdm import tqdm
import json


class ToolCall(BaseModel):
    """Represents a single tool call with its parameters."""
    tool_name: str
    tool_parameters: Dict[str, Any]


class ToolEvalResult(BaseModel):
    """Results from evaluating tool calls for a task."""
    
    # Task-level information
    task_id: str = Field("unknown", description="Identifier of the task being evaluated")
    success: bool = Field(False, description="Whether all tools were called correctly")
    tool_count_match: bool = Field(False, description="Whether the number of tool calls matches")
    ground_truth_tool_count: int = Field(0, description="Number of tools in ground truth")
    prediction_tool_count: int = Field(0, description="Number of tools in prediction")
    
    # Tool matching information
    exact_matches: int = Field(0, description="Number of tools that match exactly (name and parameters)")
    name_only_matches: int = Field(0, description="Number of tools that match by name only")
    missing_tools: List[str] = Field(default_factory=list, description="Tool names in ground truth but not in prediction")
    extra_tools: List[str] = Field(default_factory=list, description="Tool names in prediction but not in ground truth")
    
    # Detailed parameter information
    param_matches: Dict[str, float] = Field(default_factory=dict, description="Parameter match scores for each tool: {tool_name: score}")
    param_mismatches: Dict[str, Dict[str, Tuple[Any, Any]]] = Field(
        default_factory=dict,
        description="Mismatched parameters for each tool: {tool_name: {param: (gt_value, pred_value)}}"
    )
    
    # Scoring metrics
    tool_name_score: float = Field(0.0, description="Percentage of tool names that match")
    param_match_score: float = Field(0.0, description="Average parameter match score across all tools")
    order_score: float = Field(0.0, description="How well the order of tool calls matches (0-1)")
    overall_score: float = Field(0.0, description="Overall evaluation score (0-1)")
    
    # Match type
    match_type: str = Field("strict", description="Type of matching used (strict or flexible)")
    
    # Weights and thresholds used
    weights: Dict[str, float] = Field(
        default_factory=lambda: {"name": 0.4, "params": 0.4, "order": 0.2},
        description="Weights used for calculating overall score"
    )
    thresholds: Dict[str, float] = Field(
        default_factory=lambda: {"flexible_param": 0.6, "flexible_order": 0.5},
        description="Thresholds used for evaluation"
    )


class StaticToolEvaluator:
    """
    Evaluates tool call execution by comparing ground truth tool calls to model predictions.
    Supports both strict (exact) and flexible (soft) matching of tools and parameters.
    """
    
    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """Normalize a value for comparison."""
        if isinstance(value, str):
            return value.strip().lower()
        elif isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, list):
            return [StaticToolEvaluator._normalize_value(v) for v in value]
        elif isinstance(value, dict):
            return {
                StaticToolEvaluator._normalize_value(k): StaticToolEvaluator._normalize_value(v)
                for k, v in value.items()
            }
        return value
    
    @staticmethod
    def _compare_values(val1: Any, val2: Any, flexible: bool = False) -> bool:
        """Compare two values, handling various types."""
        norm_val1 = StaticToolEvaluator._normalize_value(val1)
        norm_val2 = StaticToolEvaluator._normalize_value(val2)
        
        # Handle numeric comparisons
        if isinstance(norm_val1, float) and isinstance(norm_val2, float):
            if flexible:
                if abs(norm_val1) < 1.0 or abs(norm_val2) < 1.0:
                    return abs(norm_val1 - norm_val2) < 0.2
                return abs(norm_val1 - norm_val2) / max(abs(norm_val1), abs(norm_val2)) < 0.2
            else:
                return abs(norm_val1 - norm_val2) < 1e-6
        
        # For string comparisons in flexible mode
        if flexible and isinstance(norm_val1, str) and isinstance(norm_val2, str):
            norm_val1_lower = norm_val1.lower()
            norm_val2_lower = norm_val2.lower()
            if norm_val1_lower in norm_val2_lower or norm_val2_lower in norm_val1_lower:
                return True
        
        return norm_val1 == norm_val2
    
    @staticmethod
    def _compare_tool_parameters(
        ground_truth: Dict[str, Any],
        prediction: Dict[str, Any],
        flexible: bool = False
    ) -> Tuple[float, Dict[str, Tuple[Any, Any]]]:
        """Compare parameters from a ground truth tool call to a prediction."""
        gt_params = set(ground_truth.keys())
        pred_params = set(prediction.keys())
        
        common_params = gt_params.intersection(pred_params)
        mismatched_params = {}
        
        matched_params = 0
        for param in common_params:
            gt_value = ground_truth[param]
            pred_value = prediction[param]
            
            # Both None counts as match
            if gt_value is None and pred_value is None:
                matched_params += 1
            elif StaticToolEvaluator._compare_values(gt_value, pred_value, flexible):
                matched_params += 1
            else:
                mismatched_params[param] = (gt_value, pred_value)
        
        if flexible:
            important_params = set()
            for param in gt_params:
                value = ground_truth[param]
                if value is not None and (not isinstance(value, (list, dict, str)) or len(value) > 0):
                    important_params.add(param)
            
            partial_match_score = 0.0
            for param in important_params:
                if param in common_params:
                    if param not in mismatched_params:
                        partial_match_score += 1.0
                    else:
                        partial_match_score += 0.5
            
            match_score = partial_match_score / len(important_params) if important_params else 1.0
        else:
            match_score = matched_params / len(gt_params) if gt_params else 1.0
        
        return match_score, mismatched_params
    
    @staticmethod
    def longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
        """Find the length of the longest common subsequence between two sequences."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    @staticmethod
    def evaluate_task(
        ground_truth_calls: List[ToolCall],
        prediction_calls: List[ToolCall],
        task_id: Optional[str] = None,
        match_type: Literal["strict", "flexible"] = "strict",
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> ToolEvalResult:
        """Evaluate all tool calls for a task."""
        flexible = match_type == "flexible"
        
        default_weights = {"name": 0.4, "params": 0.4, "order": 0.2}
        default_thresholds = {"flexible_param": 0.6, "flexible_order": 0.5}
        
        w = weights or default_weights
        t = thresholds or default_thresholds
        
        weight_sum = w.get("name", 0.4) + w.get("params", 0.4) + w.get("order", 0.2)
        if abs(weight_sum - 1.0) > 0.001:
            w = default_weights
        
        result = ToolEvalResult(
            task_id=task_id or "unknown",
            ground_truth_tool_count=len(ground_truth_calls),
            prediction_tool_count=len(prediction_calls),
            tool_count_match=len(ground_truth_calls) == len(prediction_calls),
            match_type=match_type,
        )
        
        gt_tool_names = [tool.tool_name for tool in ground_truth_calls]
        pred_tool_names = [tool.tool_name for tool in prediction_calls]
        
        result.missing_tools = [name for name in gt_tool_names if name not in pred_tool_names]
        result.extra_tools = [name for name in pred_tool_names if name not in gt_tool_names]
        
        name_only_matches = 0
        for gt_call, pred_call in zip(ground_truth_calls, prediction_calls):
            if gt_call.tool_name == pred_call.tool_name:
                name_only_matches += 1
        result.name_only_matches = name_only_matches
        
        if gt_tool_names:
            name_matches = sum(
                1 for gt_name, pred_name in zip(gt_tool_names, pred_tool_names)
                if gt_name == pred_name
            )
            result.tool_name_score = name_matches / len(gt_tool_names)
        else:
            result.tool_name_score = 1.0 if len(pred_tool_names) == 0 else 0.0
        
        position_param_scores = []
        position_exact_matches = 0
        
        for i, (gt_call, pred_call) in enumerate(zip(ground_truth_calls, prediction_calls)):
            if gt_call.tool_name == pred_call.tool_name:
                param_score, mismatches = StaticToolEvaluator._compare_tool_parameters(
                    gt_call.tool_parameters, pred_call.tool_parameters, flexible
                )
                
                position_param_scores.append(param_score)
                
                key = (
                    f"{gt_call.tool_name}_{i}"
                    if gt_call.tool_name in [call.tool_name for j, call in enumerate(ground_truth_calls) if j != i]
                    else gt_call.tool_name
                )
                result.param_matches[key] = param_score
                
                if mismatches:
                    result.param_mismatches[key] = mismatches
                
                if param_score == 1.0:
                    position_exact_matches += 1
        
        result.exact_matches = position_exact_matches
        
        if position_param_scores:
            result.param_match_score = sum(position_param_scores) / len(position_param_scores)
        elif len(gt_tool_names) == 0 and len(pred_tool_names) == 0:
            result.param_match_score = 1.0
        else:
            result.param_match_score = 0.0
        
        if len(gt_tool_names) == 0 and len(pred_tool_names) == 0:
            result.order_score = 1.0
        else:
            lcs_length = StaticToolEvaluator.longest_common_subsequence(gt_tool_names, pred_tool_names)
            max_length = max(len(gt_tool_names), 1)
            result.order_score = lcs_length / max_length
        
        if flexible:
            param_threshold = t.get("flexible_param", 0.6)
            required_param_match = all(score >= param_threshold for score in result.param_matches.values())
            order_threshold = t.get("flexible_order", 0.5)
            order_ok = result.order_score >= order_threshold
            
            if len(ground_truth_calls) == 0 and len(prediction_calls) == 0:
                result.success = True
            else:
                result.success = (
                    result.name_only_matches == len(ground_truth_calls)
                    and required_param_match
                    and order_ok
                )
        else:
            if len(ground_truth_calls) == 0 and len(prediction_calls) == 0:
                result.success = True
            else:
                tools_match = (
                    result.exact_matches == len(ground_truth_calls)
                    and len(result.extra_tools) == 0
                )
                order_match = result.order_score == 1.0
                result.success = tools_match and order_match
        
        result.overall_score = (
            w.get("name", 0.4) * result.tool_name_score
            + w.get("params", 0.4) * result.param_match_score
            + w.get("order", 0.2) * result.order_score
        )
        
        result.weights = w
        result.thresholds = t
        
        return result


class TrajectoryAdapter:
    """Adapter to convert trajectory format to evaluation format."""

    @staticmethod
    def load_gold_trajectories(gold_path: str) -> List[Dict[str, Any]]:
        """
        Load gold trajectories from a file where each element is:
        {
        "messages": [ ... ]
        }

        We set:
        - task_id = first user message content
        - query   = same as task_id
        """
        with open(gold_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Expected gold trajectories file to be a JSON array")

        gold_list: List[Dict[str, Any]] = []

        for item in data:
            messages = item.get("messages", [])
            if not messages:
                continue

            # Find the first user message
            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            if user_msg is None:
                continue

            query_text = (user_msg.get("content") or "").strip()
            if not query_text:
                continue

            # Normalize each gold entry to have task_id + query + messages
            norm = {
                "task_id": query_text,
                "query": query_text,
                "messages": messages,
            }
            gold_list.append(norm)

        return gold_list

    @staticmethod
    def load_pred_trajectories(pred_path: str) -> List[Dict[str, Any]]:
        """
        Load predicted trajectories from a file like:
        {
        "batch_id": "...",
        "tasks": [
            {
            "task_index": ...,
            "query": "...",
            "trajectory": [ ... ],
            ...
            },
            ...
        ]
        }

        We set:
        - task_id = query
        - query   = query
        - trajectory = trajectory
        """
        with open(pred_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tasks = data.get("tasks", [])
        pred_list: List[Dict[str, Any]] = []

        for t in tasks:
            query_text = (t.get("query") or "").strip()
            if not query_text:
                continue

            norm = {
                "task_id": query_text,
                "query": query_text,
                "trajectory": t.get("trajectory", []),
            }
            pred_list.append(norm)

        return pred_list

    @staticmethod
    def align_gold_and_pred(
        gold_list: List[Dict[str, Any]],
        pred_list: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            pred_ids = {p["task_id"] for p in pred_list}
            gold_ids = {g["task_id"] for g in gold_list}
            common_ids = pred_ids & gold_ids

            gold_filtered = [g for g in gold_list if g["task_id"] in common_ids]
            pred_filtered = [p for p in pred_list if p["task_id"] in common_ids]

            return gold_filtered, pred_filtered

    @staticmethod
    def extract_tool_calls_from_output_trajectory(
        trajectory_data: Dict[str, Any]
    ) -> List[ToolCall]:
        """Extract tool calls from trajectory format and normalize tool names."""
        
        # Tool aliases → canonical gold tool name
        TOOL_ALIASES = {
            "send_channel_message": "send_message",
            "send_direct_message": "send_message",
        }

        def normalize_tool_name(tool_name: str) -> str:
            """
            Normalize tool name by removing server prefix if present.
            Examples:
            - get_projects -> get_projects
            - gitlab_get_projects -> get_projects
            - rocketchat_send_message -> send_message
            """
            if "_" in tool_name:
                _, tool_name = tool_name.split("_", 1)

            # Step 2: Handle renamed tools
            return TOOL_ALIASES.get(tool_name, tool_name)

        tool_calls: List[ToolCall] = []
        trajectory = trajectory_data.get("trajectory", [])

        for step in trajectory:
            if (
                step.get("step_type") == "action"
                and step.get("tool_used")
                and step.get("tool_input") is not None
            ):
                normalized_tool_name = normalize_tool_name(step["tool_used"])

                tool_call = ToolCall(
                    tool_name=normalized_tool_name,
                    # tool_name=step["tool_used"],
                    tool_parameters=step["tool_input"],
                )
                tool_calls.append(tool_call)

        return tool_calls

    @staticmethod
    def extract_tool_calls_from_gold_trajectory(gold_data: Dict[str, Any]) -> List[ToolCall]:
        """
        Extract tool calls from gold ReAct-style trajectory with `messages` and `tool_calls`.

        Expects messages like:
        {
          "role": "assistant",
          "tool_calls": [
            {
              "type": "function",
              "function": {
                "name": "...",
                "arguments": {...} or JSON string
              }
            }
          ]
        }
        """
        tool_calls: List[ToolCall] = []

        for msg in gold_data.get("messages", []):
            if msg.get("role") != "assistant":
                continue

            if "tool_calls" not in msg:
                continue

            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {}) or {}
                name = func.get("name")
                if not name:
                    continue

                args = func.get("arguments", {}) or {}
                # In some logs, arguments might be a JSON string
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        # If parsing fails, leave it as raw string
                        pass

                tool_calls.append(
                    ToolCall(
                        tool_name=name,
                        tool_parameters=args
                    )
                )

        return tool_calls
    
    @staticmethod
    def evaluate_trajectory(
        ground_truth_data: Dict[str, Any],
        trajectory_data: Dict[str, Any],
        match_type: Literal["strict", "flexible", "both"] = "strict",
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Evaluate a trajectory against ground truth."""
        
        # Extract ground truth and predicted tool calls from trajectory
        gt_tool_calls = TrajectoryAdapter.extract_tool_calls_from_gold_trajectory(ground_truth_data)
        pred_tool_calls = TrajectoryAdapter.extract_tool_calls_from_output_trajectory(trajectory_data)
        
        # Get task ID
        task_id = (
            ground_truth_data.get("task_id") or 
            ground_truth_data.get("id") or 
            ground_truth_data.get("name") or
            trajectory_data.get("task_id") or
            "unknown"
        )
        
        # Evaluate
        if match_type == "both":
            strict_result = StaticToolEvaluator.evaluate_task(
                gt_tool_calls, pred_tool_calls, task_id, "strict", weights, thresholds
            )
            flexible_result = StaticToolEvaluator.evaluate_task(
                gt_tool_calls, pred_tool_calls, task_id, "flexible", weights, thresholds
            )
            
            return {
                "task_id": task_id,
                "counts": {
                    "ground_truth_tool_count": strict_result.ground_truth_tool_count,
                    "prediction_tool_count": strict_result.prediction_tool_count,
                    "tool_count_match": strict_result.tool_count_match,
                    "name_only_matches": strict_result.name_only_matches,
                    "exact_matches": strict_result.exact_matches,
                },
                "ground_truth_tools": [{"name": tc.tool_name, "params": tc.tool_parameters} for tc in gt_tool_calls],
                "prediction_tools": [{"name": tc.tool_name, "params": tc.tool_parameters} for tc in pred_tool_calls],
                "strict": {
                    "success": strict_result.success,
                    "overall_score": strict_result.overall_score,
                    "tool_name_score": strict_result.tool_name_score,
                    "param_match_score": strict_result.param_match_score,
                    "order_score": strict_result.order_score,
                },
                "flexible": {
                    "success": flexible_result.success,
                    "overall_score": flexible_result.overall_score,
                    "tool_name_score": flexible_result.tool_name_score,
                    "param_match_score": flexible_result.param_match_score,
                    "order_score": flexible_result.order_score,
                },
                "weights": strict_result.weights,
                "thresholds": strict_result.thresholds,
                "missing_tools": strict_result.missing_tools,
                "extra_tools": strict_result.extra_tools,
                "param_mismatches": strict_result.param_mismatches,
                "param_matches": strict_result.param_matches,
            }
        else:
            result = StaticToolEvaluator.evaluate_task(
                gt_tool_calls, pred_tool_calls, task_id, match_type, weights, thresholds
            )
            result_dict = result.dict()
            result_dict["ground_truth_tools"] = [{"name": tc.tool_name, "params": tc.tool_parameters} for tc in gt_tool_calls]
            result_dict["prediction_tools"] = [{"name": tc.tool_name, "params": tc.tool_parameters} for tc in pred_tool_calls]
            return result_dict
    
    @staticmethod
    def batch_evaluate_trajectories(
        ground_truth_list: List[Dict[str, Any]],
        trajectory_list: List[Dict[str, Any]],
        match_type: Literal["strict", "flexible", "both"] = "both",
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Batch evaluate multiple trajectories."""
        # Create lookup dictionaries by task_id
        gt_dict = {}
        for gt in ground_truth_list:
            # print("gt type", type(gt))
            # print("Ground truth item:", len(gt))
            task_id = gt.get("task_id") or gt.get("id") or gt.get("name")
            if task_id:
                gt_dict[task_id] = gt
        
        traj_dict = {}
        for traj in trajectory_list:
            task_id = traj.get("task_id") or traj.get("query")
            if task_id:
                traj_dict[task_id] = traj
        
        # Evaluate each matching pair
        results = {}
        common_ids = set(gt_dict.keys()).intersection(traj_dict.keys())

        for task_id in tqdm(common_ids, desc="Evaluating tasks"):
            results[task_id] = TrajectoryAdapter.evaluate_trajectory(
                gt_dict[task_id],
                traj_dict[task_id],
                match_type,
                weights,
                thresholds
            )
        
        # Calculate statistics if using "both" mode
        if match_type == "both" and results:
            strict_scores = [r["strict"]["overall_score"] for r in results.values()]
            flexible_scores = [r["flexible"]["overall_score"] for r in results.values()]
            
            strict_success = sum(1 for r in results.values() if r["strict"]["success"])
            flexible_success = sum(1 for r in results.values() if r["flexible"]["success"])
            
            strict_name_scores = [r["strict"]["tool_name_score"] for r in results.values()]
            strict_param_scores = [r["strict"]["param_match_score"] for r in results.values()]
            strict_order_scores = [r["strict"]["order_score"] for r in results.values()]
            
            flexible_name_scores = [r["flexible"]["tool_name_score"] for r in results.values()]
            flexible_param_scores = [r["flexible"]["param_match_score"] for r in results.values()]
            flexible_order_scores = [r["flexible"]["order_score"] for r in results.values()]
            
            overall_stats = {
                "total_tasks": len(results),
                "strict": {
                    "successful_tasks": strict_success,
                    "success_rate": strict_success / len(results),
                    "average_score": np.mean(strict_scores),
                    "tool_name_accuracy": np.mean(strict_name_scores),
                    "param_match_accuracy": np.mean(strict_param_scores),
                    "order_score": np.mean(strict_order_scores),
                },
                "flexible": {
                    "successful_tasks": flexible_success,
                    "success_rate": flexible_success / len(results),
                    "average_score": np.mean(flexible_scores),
                    "tool_name_accuracy": np.mean(flexible_name_scores),
                    "param_match_accuracy": np.mean(flexible_param_scores),
                    "order_score": np.mean(flexible_order_scores),
                }
            }
            
            return {
                "overall_stats": overall_stats,
                "task_results": results
            }
        else:
            # Single match type
            scores = [r["overall_score"] for r in results.values()]
            success = sum(1 for r in results.values() if r["success"])
            
            overall_stats = {
                "total_tasks": len(results),
                "successful_tasks": success,
                "success_rate": success / len(results) if results else 0.0,
                "average_score": np.mean(scores) if scores else 0.0
            }
            
            return {
                "overall_stats": overall_stats,
                "task_results": results
            }


# Example usage
if __name__ == "__main__":
    
    #single trajectory evaluation
    ground_truth_traj_path = "gold_traj.json"
    pred_traj_path = "output_traj_1.json"

    with open(ground_truth_traj_path, "r", encoding="utf-8") as f:
        ground_truth_traj = json.load(f)

    with open(pred_traj_path, "r", encoding="utf-8") as f:
        pred_traj = json.load(f)

    #batch trajectory evaluation
    ground_truth_traj_path_batch = "path_to_gold_trajectories.json"
    qwen_grpo_pred_traj_path_batch = "path_to_qwen_grpo_predicted_trajectories.json"

    # Evaluate single trajectory
    # result = TrajectoryAdapter.evaluate_trajectory(ground_truth_traj, pred_traj, match_type="both")
    # print("Single evaluation result:")
    # print(result)
    
    # Batch evaluation
    ground_truth_list = TrajectoryAdapter.load_gold_trajectories(ground_truth_traj_path_batch)
    trajectory_list = TrajectoryAdapter.load_pred_trajectories(qwen_grpo_pred_traj_path_batch)

    #allign gold and pred trajectories
    ground_truth_list, trajectory_list = TrajectoryAdapter.align_gold_and_pred(
        ground_truth_list,
        trajectory_list,
    )

    batch_result = TrajectoryAdapter.batch_evaluate_trajectories(
        ground_truth_list,
        trajectory_list,
        match_type="both"
    )

    # print("\nBatch evaluation result (raw JSON):")
    # print(json.dumps(batch_result, indent=2))
    output_path = "QWEN_evaluation_output.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(batch_result, f, indent=2, ensure_ascii=False)

    print(f"\nBatch results saved to {output_path}")