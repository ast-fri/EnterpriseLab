# memory_manager.py
"""
Dual-Memory Architecture for AutoQuest Phase 2/3
ZERO-ASSUMPTION DESIGN: Works with any benchmark, any data structure
Manages Short-Term (Trajectory Context) and Long-Term (Global KB) memory
"""

from typing import Dict, List, Optional, Any
import time
import json
from AutoQuest.intelligent_explorer.data_models import (
    NodeExecution,
    EnvironmentKnowledgeBase,
    ResourceNode
)


class TrajectoryContext:
    """
    Pure short-term memory storage
    NO interpretation, NO extraction, NO assumptions
    Just stores raw execution data
    """
    
    def __init__(self, trajectory_id: str):
        self.trajectory_id = trajectory_id
        self.execution_log = []  # Raw execution records
        self.start_time = time.time()
    
    def add_execution(self, tool_name: str, inputs: Any, outputs: Any, success: bool):
        """
        Store raw execution - no processing whatsoever
        
        Args:
            tool_name: Name of tool executed
            inputs: Raw inputs (any type)
            outputs: Raw outputs (any type)
            success: Whether execution succeeded
        """
        self.execution_log.append({
            "step": len(self.execution_log) + 1,
            "tool": tool_name,
            "inputs": inputs,
            "outputs": outputs,
            "success": success,
            "timestamp": time.time()
        })
    
    def get_recent_executions(self, window: int = 10) -> List[Dict]:
        """
        Return raw recent executions
        
        Args:
            window: Number of recent steps to return
            
        Returns:
            List of execution records
        """
        return self.execution_log[-window:] if self.execution_log else []
    
    def get_all_executions(self) -> List[Dict]:
        """Return all executions in this trajectory"""
        return self.execution_log
    
    def get_successful_executions(self) -> List[Dict]:
        """Return only successful executions"""
        return [e for e in self.execution_log if e.get('success', False)]


class MemoryManager:
    """
    Zero-assumption memory manager
    
    Design Principles:
    1. Store everything as raw data
    2. No field name assumptions (no looking for 'id', 'emp_id', etc.)
    3. No structure assumptions (works with dict, list, string, custom objects)
    4. No benchmark assumptions (works with GitLab, Salesforce, EnterpriseBench, etc.)
    5. LLM does ALL semantic matching
    
    Works with ANY benchmark without code changes
    """
    
    def __init__(self, global_kb: EnvironmentKnowledgeBase):
        """
        Args:
            global_kb: Long-term knowledge base (persistent across trajectories)
        """
        self.global_kb = global_kb
        self.trajectories: Dict[str, TrajectoryContext] = {}
        self.current_trajectory_id: Optional[str] = None
    
    # ========================
    # Trajectory Lifecycle
    # ========================
    
    def start_trajectory(self, trajectory_id: str):
        """
        Initialize short-term memory for a new trajectory
        
        Args:
            trajectory_id: Unique trajectory identifier
        """
        self.trajectories[trajectory_id] = TrajectoryContext(trajectory_id)
        self.current_trajectory_id = trajectory_id
        print(f"      🧠 Started trajectory context: {trajectory_id}")
    
    def end_trajectory(self, trajectory_id: str, commit_to_kb: bool = True):
        """
        Clean up trajectory context
        
        Args:
            trajectory_id: Trajectory to end
            commit_to_kb: Whether to commit (resources already in KB via resource_tracker)
        """
        context = self.trajectories.get(trajectory_id)
        if not context:
            return
        
        if commit_to_kb:
            # Resources are already committed to KB by resource_tracker
            # This is just a marker for potential future logic
            print(f"      💾 Trajectory {trajectory_id} data persisted to KB")
        
        # Clean up short-term memory
        del self.trajectories[trajectory_id]
        
        if self.current_trajectory_id == trajectory_id:
            self.current_trajectory_id = None
        
        print(f"      🧹 Cleared trajectory context: {trajectory_id}")
    
    # ========================
    # Execution Recording
    # ========================
    
    def record_execution(
        self,
        trajectory_id: str,
        tool_name: str,
        tool_inputs: Any,
        tool_outputs: Any,
        success: bool,
        **kwargs  # Accept any additional args, ignore them
    ):
        """
        Record raw execution data - NO interpretation
        
        This method is called by resource_tracker after each tool execution
        It stores the complete, unmodified data for the LLM to use
        
        Args:
            trajectory_id: Current trajectory
            tool_name: Tool that was executed
            tool_inputs: Raw inputs (any type)
            tool_outputs: Raw outputs (any type)
            success: Whether execution succeeded
            **kwargs: Any additional metadata (ignored, for compatibility)
        """
        context = self.trajectories.get(trajectory_id)
        if not context:
            print(f"      ⚠️  No trajectory context for {trajectory_id}")
            return
        
        # Store raw data with no processing
        context.add_execution(tool_name, tool_inputs, tool_outputs, success)
        
        # Log what we stored
        output_type = type(tool_outputs).__name__
        output_preview = self._safe_preview(tool_outputs, max_length=50)
        print(f"      📝 Logged: {tool_name} → {output_type} {output_preview}")
    
    # ========================
    # Context Retrieval for Input Generation
    # ========================
    
    def get_context_for_input_generation(self, trajectory_id, required_resource_types=None) -> Dict:
        context = self.trajectories.get(trajectory_id)
        if not context:
            return {"priority_resources": {}, "recent_executions": [], "all_executions": [], "kb_samples": self._get_kb_samples(required_resource_types or [])}
        priority_resources = self._scavenge_resources_from_history(context.get_recent_executions(10))
        return {
            "priority_resources": priority_resources,
            "recent_executions": context.get_recent_executions(window=10),
            "all_executions": context.get_all_executions(),
            "kb_samples": self._get_kb_samples(required_resource_types or [])
        }
    def _scavenge_resources_from_history(self, executions):
        scavenged = {}
        for step in executions:
            if not step.get('success') or not step.get('outputs'): continue
            outputs = step['outputs']
            items = outputs if isinstance(outputs, list) else [outputs] if isinstance(outputs, dict) else []
            for item in items:
                if not isinstance(item, dict): continue
                for key, value in item.items():
                    if not value or not isinstance(value, (str, int)): continue
                    if str(key).endswith('_id') or str(key).endswith('_iid'):
                        res_type = str(key).replace('_id', '').replace('_iid', '')
                        for prefix in ['sender_', 'recipient_', 'source_', 'target_', 'from_', 'to_']:
                            res_type = res_type.replace(prefix, '')
                        if res_type not in scavenged: scavenged[res_type] = []
                        if str(value) not in scavenged[res_type]: scavenged[res_type].append(str(value))
        return scavenged
    def _get_kb_samples(self, resource_types: List[str]) -> Dict[str, List[str]]:
        """
        Get sample IDs from long-term KB for specified types
        
        This is a pure fallback when trajectory context is empty
        
        Args:
            resource_types: List of resource type names
            
        Returns:
            Dict mapping type -> list of sample IDs
        """
        samples = {}
        
        for res_type in resource_types:
            if res_type in self.global_kb.resource_by_type:
                kb_ids = [
                    rid for rid in self.global_kb.resource_by_type[res_type]
                    if not self.global_kb.resources[rid].is_deleted
                ]
                if kb_ids:
                    samples[res_type] = kb_ids[:5]  # Max 5 samples
        
        return samples
    
    # ========================
    # Statistics & Utilities
    # ========================
    
    def get_stats(self, trajectory_id: str) -> Dict:
        """
        Get statistics for a trajectory
        
        Args:
            trajectory_id: Trajectory to get stats for
            
        Returns:
            Statistics dict
        """
        context = self.trajectories.get(trajectory_id)
        if not context:
            return {}
        
        executions = context.get_all_executions()
        successful = context.get_successful_executions()
        
        return {
            "trajectory_id": trajectory_id,
            "total_steps": len(executions),
            "successful_steps": len(successful),
            "failed_steps": len(executions) - len(successful),
            "duration_seconds": time.time() - context.start_time,
            "success_rate": len(successful) / len(executions) if executions else 0.0
        }
    
    def _safe_preview(self, data: Any, max_length: int = 50) -> str:
        """
        Generate a safe preview of any data type
        
        Args:
            data: Any data
            max_length: Max characters
            
        Returns:
            Preview string
        """
        try:
            if data is None:
                return "(None)"
            
            if isinstance(data, (str, int, float, bool)):
                preview = str(data)
            elif isinstance(data, dict):
                preview = f"{{...{len(data)} keys...}}"
            elif isinstance(data, list):
                preview = f"[...{len(data)} items...]"
            else:
                preview = f"({type(data).__name__})"
            
            if len(preview) > max_length:
                preview = preview[:max_length-3] + "..."
            
            return preview
        
        except Exception:
            return "(unprintable)"
    
    def get_trajectory_summary(self, trajectory_id: str) -> str:
        """
        Get human-readable summary of a trajectory
        
        Args:
            trajectory_id: Trajectory to summarize
            
        Returns:
            Summary string
        """
        stats = self.get_stats(trajectory_id)
        
        if not stats:
            return f"Trajectory {trajectory_id}: Not found"
        
        return (
            f"Trajectory {trajectory_id}:\n"
            f"  Steps: {stats['total_steps']} "
            f"(✓ {stats['successful_steps']}, ✗ {stats['failed_steps']})\n"
            f"  Duration: {stats['duration_seconds']:.1f}s\n"
            f"  Success Rate: {stats['success_rate']*100:.1f}%"
        )
