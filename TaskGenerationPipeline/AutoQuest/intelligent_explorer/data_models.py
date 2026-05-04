# data_models.py
"""
Core data models and knowledge base structures for AutoQuest Phase 2
Supports dynamic KB evolution and database simulation per environment
"""

from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field, asdict
import json
import time
from datetime import datetime


@dataclass
class NodeState:
    """Represents the state of a node: inputs + outputs"""
    node_name: str
    tool_input: Dict
    tool_output: Any
    execution_time: float
    timestamp: float
    success: bool
    error: Optional[str] = None
    interestingness_score: float = 0.0
    created_resources: Dict = field(default_factory=dict)


@dataclass
class NodeExecution:
    """Represents execution of a single node (tool)"""
    node_name: str
    tool_input: Dict
    tool_output: Any
    execution_time: float
    success: bool
    error: Optional[str] = None
    interestingness_score: float = 0.0


@dataclass
class ResourceNode:
    """Rich resource representation with full context"""
    resource_id: str
    resource_type: str
    created_by_tool: str
    creation_inputs: Dict
    creation_outputs: Dict  # Full tool output - schema is dynamic per server
    creation_timestamp: float
    parent_resources: List[str]  # IDs of resources used to create this
    metadata: Dict  # Additional context
    is_deleted: bool = False  # Track deletion without removing from KB
    last_modified: float = field(default_factory=time.time)
    access_count: int = 0  # For pruning unused resources
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)


class EnvironmentKnowledgeBase:
    """
    Persistent, evolving knowledge base for a single environment/server
    Mimics a real database snapshot that grows with tool interactions
    """
    
    def __init__(self, environment_name: str):
        """
        Args:
            environment_name: Name of the environment (e.g., 'gitlab', 'crm')
        """
        self.environment_name = environment_name
        self.version = 1
        self.created_at = time.time()
        self.last_updated = time.time()
        
        # Core storage structures
        self.resources: Dict[str, ResourceNode] = {}  # resource_id -> ResourceNode
        self.resource_by_type: Dict[str, List[str]] = {}  # resource_type -> [ids]
        self.execution_chain: List[Dict] = []  # Ordered list of all executions
        
        # Dynamic schema tracking (learned from tool outputs)
        self.resource_schemas: Dict[str, Dict] = {}  # resource_type -> schema
        
        # Reference tracking for integrity
        self.resource_references: Dict[str, Set[str]] = {}  # resource_id -> {referencing_ids}
        
        # Statistics
        self.stats = {
            "total_resources_created": 0,
            "total_resources_deleted": 0,
            "total_executions": 0,
            "resource_type_counts": {}
        }
    
    def add_resource(self, resource: ResourceNode):
        """
        Add resource and update dynamic schema
        
        Args:
            resource: ResourceNode to add
        """
        # Store resource
        self.resources[resource.resource_id] = resource
        
        # Update type index
        if resource.resource_type not in self.resource_by_type:
            self.resource_by_type[resource.resource_type] = []
        self.resource_by_type[resource.resource_type].append(resource.resource_id)
        
        # Learn/update schema from creation_outputs
        self._update_schema(resource.resource_type, resource.creation_outputs)
        
        # Track parent references
        for parent_id in resource.parent_resources:
            if parent_id not in self.resource_references:
                self.resource_references[parent_id] = set()
            self.resource_references[parent_id].add(resource.resource_id)
        
        # Update stats
        self.stats["total_resources_created"] += 1
        if resource.resource_type not in self.stats["resource_type_counts"]:
            self.stats["resource_type_counts"][resource.resource_type] = 0
        self.stats["resource_type_counts"][resource.resource_type] += 1
        
        self.last_updated = time.time()
        self.version += 1
    
    def update_resource(self, resource_id: str, updates: Dict):
        """
        Update existing resource fields
        
        Args:
            resource_id: ID of resource to update
            updates: Dictionary of field updates from tool output
        """
        if resource_id not in self.resources:
            raise ValueError(f"Resource {resource_id} not found")
        
        resource = self.resources[resource_id]
        
        # Merge updates into creation_outputs (dynamic schema)
        resource.creation_outputs.update(updates)
        resource.last_modified = time.time()
        
        # Update schema
        self._update_schema(resource.resource_type, updates)
        
        self.last_updated = time.time()
        self.version += 1
    
    def delete_resource(self, resource_id: str, soft_delete: bool = True):
        """
        Delete resource (soft or hard)
        
        Args:
            resource_id: ID of resource to delete
            soft_delete: If True, mark as deleted; if False, remove completely
        """
        if resource_id not in self.resources:
            return
        
        resource = self.resources[resource_id]
        
        if soft_delete:
            resource.is_deleted = True
            resource.last_modified = time.time()
        else:
            # Hard delete - remove from all indexes
            del self.resources[resource_id]
            if resource.resource_type in self.resource_by_type:
                self.resource_by_type[resource.resource_type].remove(resource_id)
            if resource_id in self.resource_references:
                del self.resource_references[resource_id]
        
        self.stats["total_resources_deleted"] += 1
        self.last_updated = time.time()
        self.version += 1
    
    def get_resource(self, resource_id: str, include_deleted: bool = False) -> Optional[ResourceNode]:
        """
        Get resource by ID
        
        Args:
            resource_id: Resource ID
            include_deleted: Whether to return deleted resources
            
        Returns:
            ResourceNode or None
        """
        resource = self.resources.get(resource_id)
        if resource and (include_deleted or not resource.is_deleted):
            resource.access_count += 1
            return resource
        return None
    
    def get_resource_context(self, resource_id: str) -> Dict:
        """Get full context for a resource including how it was created"""
        if resource_id not in self.resources:
            return {}
        
        resource = self.resources[resource_id]
        return {
            "id": resource.resource_id,
            "type": resource.resource_type,
            "created_by": resource.created_by_tool,
            "creation_inputs": resource.creation_inputs,
            "creation_outputs": resource.creation_outputs,
            "parent_resources": resource.parent_resources,
            "metadata": resource.metadata,
            "is_deleted": resource.is_deleted,
            "last_modified": resource.last_modified,
            "access_count": resource.access_count
        }
    
    def get_most_recent_by_type(self, resource_type: str, include_deleted: bool = False) -> Optional[ResourceNode]:
        """Get most recent resource of a type with full context"""
        if resource_type not in self.resource_by_type:
            return None
        
        resource_ids = self.resource_by_type[resource_type]
        if not resource_ids:
            return None
        
        # Iterate backwards to find most recent non-deleted
        for resource_id in reversed(resource_ids):
            resource = self.resources[resource_id]
            if include_deleted or not resource.is_deleted:
                resource.access_count += 1
                return resource
        
        return None
    
    def get_all_by_type(self, resource_type: str, include_deleted: bool = False) -> List[ResourceNode]:
        """Get all resources of a specific type"""
        if resource_type not in self.resource_by_type:
            return []
        
        results = []
        for resource_id in self.resource_by_type[resource_type]:
            resource = self.resources[resource_id]
            if include_deleted or not resource.is_deleted:
                results.append(resource)
        
        return results
    
    def _update_schema(self, resource_type: str, output: Dict):
        """
        Dynamically learn/update resource schema from tool outputs
        
        Args:
            resource_type: Type of resource
            output: Tool output dictionary
        """
        if resource_type not in self.resource_schemas:
            self.resource_schemas[resource_type] = {}
        
        schema = self.resource_schemas[resource_type]
        
        # Extract field types from output
        for key, value in output.items():
            if key not in schema:
                schema[key] = {
                    "type": type(value).__name__,
                    "observed_count": 1,
                    "sample_values": [value] if not isinstance(value, (dict, list)) else []
                }
            else:
                schema[key]["observed_count"] += 1
                if not isinstance(value, (dict, list)) and len(schema[key]["sample_values"]) < 5:
                    schema[key]["sample_values"].append(value)
    
    def get_schema(self, resource_type: str) -> Dict:
        """Get learned schema for a resource type"""
        return self.resource_schemas.get(resource_type, {})
    
    def prune_unused_resources(self, min_access_count: int = 0, max_age_seconds: float = None):
        """
        Remove resources that haven't been accessed or are too old
        
        Args:
            min_access_count: Minimum access count to keep
            max_age_seconds: Maximum age in seconds
        """
        current_time = time.time()
        to_delete = []
        
        for resource_id, resource in self.resources.items():
            should_prune = False
            
            # Check access count
            if resource.access_count < min_access_count:
                should_prune = True
            
            # Check age
            if max_age_seconds and (current_time - resource.creation_timestamp) > max_age_seconds:
                should_prune = True
            
            # Don't prune if referenced by other resources
            if resource_id in self.resource_references and self.resource_references[resource_id]:
                should_prune = False
            
            if should_prune:
                to_delete.append(resource_id)
        
        for resource_id in to_delete:
            self.delete_resource(resource_id, soft_delete=False)
        
        return len(to_delete)
    
    def to_llm_context(self, max_history: int = 5, max_resources_per_type: int = 3) -> str:
        """
        Format knowledge base for LLM consumption
        
        Args:
            max_history: Number of recent executions to include
            max_resources_per_type: Max resources to show per type
        """
        context = f"=== ENVIRONMENT: {self.environment_name.upper()} ===\n"
        context += f"KB Version: {self.version} | Last Updated: {datetime.fromtimestamp(self.last_updated).isoformat()}\n\n"
        
        # Execution chain summary
        context += "RECENT EXECUTION HISTORY:\n"
        for i, exec_info in enumerate(self.execution_chain[-max_history:], 1):
            context += f"{i}. {exec_info['tool_name']}\n"
            context += f"   Input: {json.dumps(exec_info['inputs'], indent=6)[:200]}\n"
            context += f"   Output: {json.dumps(exec_info['outputs'], indent=6)[:200]}\n"
            context += f"   Created: {exec_info.get('created_resources', [])}\n\n"
        
        # Available resources with dynamic schema
        context += "\nAVAILABLE RESOURCES:\n"
        for resource_type, resource_ids in self.resource_by_type.items():
            if not resource_ids:
                continue
            
            # Get active (non-deleted) resources
            active_resources = [
                self.resources[rid] for rid in resource_ids
                if not self.resources[rid].is_deleted
            ]
            
            if not active_resources:
                continue
            
            context += f"\n{resource_type.upper()} (Count: {len(active_resources)}):\n"
            
            # Show schema
            schema = self.get_schema(resource_type)
            if schema:
                context += f"  Schema Fields: {', '.join(schema.keys())}\n"
            
            # Show recent resources
            for resource in active_resources[-max_resources_per_type:]:
                context += f"  • ID: {resource.resource_id}\n"
                context += f"    Created By: {resource.created_by_tool}\n"
                context += f"    Data: {json.dumps(resource.creation_outputs, indent=6)[:300]}\n"
        
        # Statistics
        context += f"\n=== STATISTICS ===\n"
        context += f"Total Resources: {self.stats['total_resources_created']}\n"
        context += f"Active Resources: {len([r for r in self.resources.values() if not r.is_deleted])}\n"
        context += f"Total Executions: {len(self.execution_chain)}\n"
        
        return context
    
    def snapshot(self) -> Dict:
        """Create a complete snapshot of the KB state"""
        return {
            "environment_name": self.environment_name,
            "version": self.version,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "resources": {rid: rnode.to_dict() for rid, rnode in self.resources.items()},
            "resource_by_type": self.resource_by_type,
            "execution_chain": self.execution_chain,
            "resource_schemas": self.resource_schemas,
            "resource_references": {k: list(v) for k, v in self.resource_references.items()},
            "stats": self.stats
        }
    
    @classmethod
    def from_snapshot(cls, snapshot: Dict) -> 'EnvironmentKnowledgeBase':
        """Restore KB from snapshot"""
        kb = cls(snapshot["environment_name"])
        kb.version = snapshot["version"]
        kb.created_at = snapshot["created_at"]
        kb.last_updated = snapshot["last_updated"]
        
        # Restore resources
        for rid, rdata in snapshot["resources"].items():
            kb.resources[rid] = ResourceNode(**rdata)
        
        kb.resource_by_type = snapshot["resource_by_type"]
        kb.execution_chain = snapshot["execution_chain"]
        kb.resource_schemas = snapshot["resource_schemas"]
        kb.resource_references = {k: set(v) for k, v in snapshot["resource_references"].items()}
        kb.stats = snapshot["stats"]
        
        return kb




# Backward compatibility alias
TrajectoryKnowledgeBase = EnvironmentKnowledgeBase