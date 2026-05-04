# persistence.py
"""
File I/O and serialization utilities for AutoQuest Phase 2
Handles KB snapshots, trajectory saving, and state recovery
"""

import os
import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from AutoQuest.intelligent_explorer.data_models import (
    NodeExecution, 
    NodeState, 
    EnvironmentKnowledgeBase,
    ResourceNode
)

environment="crm"
class PersistenceManager:
    """Handles all file persistence operations with versioning and recovery"""
    
    def __init__(
        self,
        state_save_dir: str,
        trajectory_save_dir: str,
        kb_snapshots_dir: str = f"./{environment}/kb_snapshots",
        auto_snapshot_interval: int = 100  # Snapshot every N executions
    ):
        """
        Args:
            state_save_dir: Directory for node states
            trajectory_save_dir: Directory for trajectories
            kb_snapshots_dir: Directory for KB snapshots
            auto_snapshot_interval: Auto-snapshot frequency
        """
        self.state_save_dir = Path(state_save_dir)
        self.trajectory_save_dir = Path(trajectory_save_dir)
        self.kb_snapshots_dir = Path(kb_snapshots_dir)
        self.auto_snapshot_interval = auto_snapshot_interval
        
        # Create directories
        self._create_directories()
        
        # Track execution count for auto-snapshots
        self.execution_count = 0
        
        # Cache for loaded KBs (environment_name -> KB)
        self._kb_cache: Dict[str, EnvironmentKnowledgeBase] = {}
    
    def _create_directories(self):
        """Create all required directories"""
        for directory in [
            self.state_save_dir,
            self.state_save_dir / "node_states",
            self.trajectory_save_dir,
            self.kb_snapshots_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    # ========================
    # KB Snapshot Management
    # ========================
    
    def save_kb_snapshot(
        self, 
        kb: EnvironmentKnowledgeBase, 
        snapshot_name: Optional[str] = None
    ) -> str:
        """
        Save a versioned snapshot of the knowledge base
        
        Args:
            kb: EnvironmentKnowledgeBase to snapshot
            snapshot_name: Optional custom name, otherwise auto-generated
            
        Returns:
            Path to saved snapshot file
        """
        if not snapshot_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_name = f"{kb.environment_name}_v{kb.version}_{timestamp}"
        
        snapshot_file = self.kb_snapshots_dir / f"{snapshot_name}.json"
        
        try:
            snapshot_data = kb.snapshot()
            
            # Add metadata
            snapshot_data["snapshot_metadata"] = {
                "snapshot_name": snapshot_name,
                "saved_at": datetime.now().isoformat(),
                "file_path": str(snapshot_file),
                "checksum": self._compute_checksum(snapshot_data)
            }
            
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot_data, f, indent=2, default=str)
            
            print(f"   ✓ Saved KB snapshot: {snapshot_file}")
            return str(snapshot_file)
            
        except Exception as e:
            print(f"   ✗ Failed to save KB snapshot: {e}")
            raise
    
    def load_kb_snapshot(
        self, 
        snapshot_path: str,
        use_cache: bool = True
    ) -> EnvironmentKnowledgeBase:
        """
        Load a KB from snapshot file
        
        Args:
            snapshot_path: Path to snapshot file
            use_cache: Whether to use cached KB if available
            
        Returns:
            Loaded EnvironmentKnowledgeBase
        """
        snapshot_path = Path(snapshot_path)
        
        # Check cache
        cache_key = str(snapshot_path)
        if use_cache and cache_key in self._kb_cache:
            print(f"   ↻ Using cached KB: {snapshot_path.name}")
            return self._kb_cache[cache_key]
        
        try:
            with open(snapshot_path, 'r') as f:
                snapshot_data = json.load(f)
            
            # Verify checksum if present
            if "snapshot_metadata" in snapshot_data:
                saved_checksum = snapshot_data["snapshot_metadata"].get("checksum")
                snapshot_data_copy = {k: v for k, v in snapshot_data.items() if k != "snapshot_metadata"}
                current_checksum = self._compute_checksum(snapshot_data_copy)
                
                if saved_checksum and saved_checksum != current_checksum:
                    print(f"   ⚠️  Checksum mismatch for {snapshot_path.name}")
            
            # Remove metadata before restoring
            snapshot_data.pop("snapshot_metadata", None)
            
            kb = EnvironmentKnowledgeBase.from_snapshot(snapshot_data)
            
            # Cache it
            if use_cache:
                self._kb_cache[cache_key] = kb
            
            print(f"   ✓ Loaded KB snapshot: {snapshot_path.name} (v{kb.version})")
            return kb
            
        except Exception as e:
            print(f"   ✗ Failed to load KB snapshot: {e}")
            raise
    
    def get_latest_kb_snapshot(self, environment_name: str) -> Optional[str]:
        """
        Find the most recent snapshot for an environment
        Handles both versioned and _final snapshots
        """
        # Pattern 1: Versioned snapshots (e.g., default_environment_v9_20231124.json)
        pattern1 = f"{environment_name}_v*"
        snapshots = list(self.kb_snapshots_dir.glob(f"{pattern1}.json"))
        
        # Pattern 2: Final snapshot (e.g., default_environment_final.json)
        pattern2 = f"{environment_name}_final"
        final_snapshot = self.kb_snapshots_dir / f"{pattern2}.json"
        if final_snapshot.exists():
            snapshots.append(final_snapshot)
        
        # Pattern 3: Any snapshot with environment name
        pattern3 = f"{environment_name}_*"
        other_snapshots = list(self.kb_snapshots_dir.glob(f"{pattern3}.json"))
        snapshots.extend(other_snapshots)
        
        # Remove duplicates
        snapshots = list(set(snapshots))
        
        if not snapshots:
            return None
        
        # Sort by modification time (most recent first)
        latest = max(snapshots, key=lambda p: p.stat().st_mtime)
        return str(latest)

    
    def load_or_create_kb(self, environment_name: str) -> EnvironmentKnowledgeBase:
        """
        Load existing KB or create new one
        
        Args:
            environment_name: Name of environment
            
        Returns:
            EnvironmentKnowledgeBase instance
        """
        latest_snapshot = self.get_latest_kb_snapshot(environment_name)
        
        if latest_snapshot:
            print(f"\n📂 Found existing KB for '{environment_name}'")
            return self.load_kb_snapshot(latest_snapshot)
        else:
            print(f"\n🆕 Creating new KB for '{environment_name}'")
            return EnvironmentKnowledgeBase(environment_name)
    
    def auto_snapshot_kb(self, kb: EnvironmentKnowledgeBase) -> Optional[str]:
        """
        Conditionally snapshot KB based on execution count
        
        Args:
            kb: EnvironmentKnowledgeBase to potentially snapshot
            
        Returns:
            Snapshot path if saved, None otherwise
        """
        self.execution_count += 1
        
        if self.execution_count % self.auto_snapshot_interval == 0:
            print(f"\n💾 Auto-snapshotting KB (every {self.auto_snapshot_interval} executions)")
            return self.save_kb_snapshot(kb)
        
        return None
    
    # ========================
    # Trajectory Management
    # ========================
    
    async def save_trajectory_incremental(self, trajectory: Dict):
        """
        Save trajectory incrementally during exploration
        
        Args:
            trajectory: Trajectory dictionary
        """
        trajectory_file = self.trajectory_save_dir / f"{trajectory['trajectory_id']}.json"
        
        try:
            with open(trajectory_file, 'w') as f:
                json.dump(trajectory, f, indent=2, default=str)
        except Exception as e:
            print(f"      ⚠️  Failed to save trajectory: {e}")
    
    async def save_trajectory_final(self, trajectory: Dict):
        """
        Save final trajectory with complete metadata
        
        Args:
            trajectory: Complete trajectory dictionary
        """
        await self.save_trajectory_incremental(trajectory)
        
        # Also save a summary
        summary_file = self.trajectory_save_dir / f"{trajectory['trajectory_id']}_summary.json"
        
        summary = {
            "trajectory_id": trajectory["trajectory_id"],
            "initial_node": trajectory.get("initial_node"),
            "status": trajectory.get("status"),
            "total_steps": trajectory.get("total_steps", 0),
            "cumulative_score": trajectory.get("cumulative_score", 0.0),
            "avg_interestingness": trajectory.get("avg_interestingness", 0.0),
            "execution_path": trajectory.get("execution_path", []),
            "resource_types_created": trajectory.get("statistics", {}).get("resource_types_created", []),
            "saved_at": datetime.now().isoformat()
        }
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"      ⚠️  Failed to save trajectory summary: {e}")
    
    # ========================
    # Node State Management
    # ========================
    
    async def save_node_state(
        self,
        node_state: NodeState,
        environment_name: str
    ):
        """
        Save individual node state to disk
        
        Args:
            node_state: NodeState to save
            environment_name: Environment context
        """
        env_dir = self.state_save_dir / "node_states" / environment_name
        env_dir.mkdir(parents=True, exist_ok=True)
        
        state_file = env_dir / f"{node_state.node_name.replace('/', '_')}_{int(node_state.timestamp * 1000)}.json"
        
        try:
            with open(state_file, 'w') as f:
                json.dump({
                    "node_name": node_state.node_name,
                    "tool_input": node_state.tool_input,
                    "tool_output": self.serialize_output(node_state.tool_output),
                    "execution_time": node_state.execution_time,
                    "timestamp": node_state.timestamp,
                    "success": node_state.success,
                    "error": node_state.error,
                    "interestingness_score": node_state.interestingness_score,
                    "created_resources": node_state.created_resources,
                    "environment": environment_name
                }, f, indent=2)
                
        except Exception as e:
            print(f"    ⚠️  Failed to save node state: {e}")
    
    # ========================
    # Global Statistics
    # ========================
    
    def save_global_stats(
        self,
        stats: Dict,
        trajectories: List,
        node_states: Dict,
        created_resources: Dict,
        node_execution_history: Dict
    ):
        """
        Save complete exploration statistics
        
        Args:
            stats: Statistics dictionary
            trajectories: List of trajectories
            node_states: Node states dictionary
            created_resources: Created resources dictionary
            node_execution_history: Execution history
        """
        print("\n💾 Saving global statistics...")
        
        stats_file = self.state_save_dir / "global_stats.json"
        
        global_data = {
            "statistics": stats,
            "total_trajectories": len(trajectories),
            "total_nodes_explored": len(node_states),
            "total_unique_resources": sum(len(v) for v in created_resources.values()),
            "resource_types": {k: len(v) for k, v in created_resources.items()},
            "tool_execution_counts": {
                tool: len(execs) 
                for tool, execs in node_execution_history.items()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(global_data, f, indent=2)
            print(f"   ✓ Saved to {stats_file}")
        except Exception as e:
            print(f"   ✗ Failed to save global stats: {e}")
    
    # ========================
    # Utilities
    # ========================
    
    def serialize_output(self, output: Any) -> Any:
        """
        Serialize tool output for storage
        
        Args:
            output: Any tool output
            
        Returns:
            Serializable version
        """
        if output is None:
            return None
        
        if isinstance(output, (str, int, float, bool)):
            return output
        
        if isinstance(output, dict):
            return {k: self.serialize_output(v) for k, v in output.items()}
        
        if isinstance(output, (list, tuple)):
            return [self.serialize_output(item) for item in output]
        
        try:
            return str(output)
        except:
            return "<non-serializable>"
    
    def _compute_checksum(self, data: Dict) -> str:
        """Compute checksum for data integrity"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear KB cache"""
        self._kb_cache.clear()
    
    def get_cache_info(self) -> Dict:
        """Get cache statistics"""
        return {
            "cached_environments": list(self._kb_cache.keys()),
            "cache_size": len(self._kb_cache)
        }
