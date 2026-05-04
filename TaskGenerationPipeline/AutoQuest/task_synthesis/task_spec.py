# task_synthesis/task_spec.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import uuid


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class TaskSpec:
    """
    Canonical representation of a synthesized task.
    Grounded in a specific trajectory subsequence (or combination).
    """
    task_id: str
    trajectory_ids: List[str]                 # one or more trajectories (for cross-domain)
    subsequence_step_ids: Dict[str, List[int]]  # traj_id -> list of local step indices
    instruction: str                          # natural language task description
    conversation: Optional[List[Dict]]        # optional multi-turn dialog
    required_tools: List[str]
    success_criteria: List[str]               # verifiable conditions
    difficulty: TaskDifficulty
    meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new_id(prefix: str = "task") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
