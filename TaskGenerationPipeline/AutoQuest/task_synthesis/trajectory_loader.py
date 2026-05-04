# task_synthesis/trajectory_loader.py

from typing import List, Dict, Any
import json
import os


def load_trajectories_from_file(path: str) -> List[Dict[str, Any]]:
    """
    Load list of trajectories from a JSON file produced by AutoQuest.
    
    Expected: JSON array of trajectory objects.
    """
    with open(path, "r") as f:
        data = json.load(f)
    
    # Handle both single trajectory and list of trajectories
    if isinstance(data, dict):
        return [data]
    return data


def build_memory_like_context(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a saved trajectory dict into a MemoryManager-like context.

    Input format:
      trajectory = {
        "trajectory_id": str,
        "steps": [
          {
            "step_number": int,
            "tool_name": str,
            "inputs": dict,
            "outputs": any,
            "success": bool,
            "execution_time": float,
            "interestingness_score": float,
            "kb_version": int,
            ...
          },
          ...
        ],
        ...
      }
    
    Output format (MemoryManager-compatible):
      {
        "recent_executions": [...],
        "all_executions": [...],
        "successful_executions": [...],
        "kb_samples": {},
        "priority_resources": {}
      }
    """
    steps = trajectory.get("steps", [])
    
    # Normalize step format: map tool_name → tool, step_number → step
    normalized_steps = []
    for s in steps:
        normalized = {
            "step": s.get("step_number"),
            "tool": s.get("tool_name"),
            "inputs": s.get("inputs", {}),
            "outputs": s.get("outputs"),
            "success": s.get("success", False),
            "error": s.get("error"),
            "execution_time": s.get("execution_time", 0.0),
            "interestingness_score": s.get("interestingness_score", 0.0),
            "kb_version": s.get("kb_version", 0),
            "depth": s.get("depth", 0),
            "parent_node": s.get("parent_node"),
        }
        normalized_steps.append(normalized)
    
    successful = [s for s in normalized_steps if s.get("success", False)]
    
    # Extract resource IDs from outputs for priority_resources
    priority_resources = _extract_priority_resources(normalized_steps)
    
    return {
        "recent_executions": successful[-10:],  # last 10 successful steps
        "all_executions": normalized_steps,
        "successful_executions": successful,
        "kb_samples": {},  # not needed for task synthesis
        "priority_resources": priority_resources,
    }


def _extract_priority_resources(steps: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Extract resource IDs by type from step outputs.
    Useful for grounding task generation.
    
    Returns:
      {
        "email": ["email_123", "email_456"],
        "chat": ["chat_789"],
        "thread": ["thread_abc"],
        ...
      }
    """
    resources: Dict[str, List[str]] = {}
    
    for step in steps:
        outputs = step.get("outputs")
        if not outputs:
            continue
        
        # Handle list of output dicts
        if isinstance(outputs, list):
            for item in outputs:
                if isinstance(item, dict):
                    _extract_ids_from_dict(item, resources)
        # Handle single output dict
        elif isinstance(outputs, dict):
            _extract_ids_from_dict(outputs, resources)
    
    return resources


def _extract_ids_from_dict(data: Dict[str, Any], resources: Dict[str, List[str]]) -> None:
    """
    Extract ID fields from a dictionary and categorize by resource type.
    """
    for key, value in data.items():
        if not isinstance(value, str):
            continue
        
        # Detect ID fields
        if "_id" in key.lower():
            # Infer resource type from key name
            # e.g., "email_id" → "email", "chat_id" → "chat"
            resource_type = key.replace("_id", "").replace("_iid", "")
            
            if resource_type not in resources:
                resources[resource_type] = []
            
            if value and value not in resources[resource_type]:
                resources[resource_type].append(value)


def load_all_trajectories_from_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Load all trajectory JSON files from a directory.
    
    Args:
        directory: Path to folder containing trajectories_*.json files
    
    Returns:
        List of trajectory dicts
    """
    trajectories = []
    
    if not os.path.exists(directory):
        return trajectories
    
    files = []
    for filename in os.listdir(directory):
        if filename.startswith("traj_") and not filename.endswith("_summary.json"):
            filepath = os.path.join(directory, filename)
            mtime = os.path.getmtime(filepath)
            files.append((mtime, filepath))

    # Sort by modification time (newest first)
    files.sort(reverse=True)

    # Load in reverse order
    for _, filepath in files:
        loaded = load_trajectories_from_file(filepath)
        trajectories.extend(loaded)
    return trajectories[20:]
