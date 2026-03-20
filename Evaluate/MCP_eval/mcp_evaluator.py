import json
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from llm_evaluator import MultiAspectLLMJudge, TrajectoryAdapter 
from tool_evaluation import TrajectoryAdapter as ToolAdapter


def run_evaluations(gold_path, pred_path, output_prefix):
    """Run both LLM and tool evaluations on the given trajectories."""
    

    # Create results folder if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{output_prefix}_{timestamp}"

    #Tool evaluation
    print(f"Running tool evaluation...")
    ground_truth_list = ToolAdapter.load_gold_trajectories(gold_path)
    trajectory_list = ToolAdapter.load_pred_trajectories(pred_path)

    ground_truth_list, trajectory_list = ToolAdapter.align_gold_and_pred(
        ground_truth_list, trajectory_list
    )
    tool_result = ToolAdapter.batch_evaluate_trajectories(
        ground_truth_list, trajectory_list, match_type="both"
    )
    tool_output = results_dir / f"{output_name}_tool_evaluation.json"
    with open(tool_output, "w", encoding="utf-8") as f:
        json.dump(tool_result, f, indent=2, ensure_ascii=False)
    print(f"Tool evaluation saved to {tool_output}")
    
    # LLM evaluation
    print(f"Running LLM evaluation...")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
    judge = MultiAspectLLMJudge(
        model="gpt-4o",
        api_key=api_key,
        base_url=base_url
    )
    llm_output = results_dir / f"{output_name}_llm_evaluation.json"
    summary = judge.batch_evaluate_with_llm_judge(
        gold_path=gold_path,
        pred_path=pred_path,
        output_path=llm_output
    )
    print(f"LLM evaluation saved to {llm_output}")
    
    return str(tool_output), str(llm_output)


if __name__ == "__main__":
    load_dotenv()
    
    # Set your paths here
    gold_path = "path_to_gold_trajectories.json"
    pred_path = "path_to_predicted_trajectories.json"
    output_prefix = "agrpo_entarena_100_tasks"  # Customize this prefix for your output files
    
    tool_out, llm_out = run_evaluations(gold_path, pred_path, output_prefix)
    print(f"\nBoth evaluations complete:")
    print(f"  Tool: {tool_out}")
    print(f"  LLM: {llm_out}")