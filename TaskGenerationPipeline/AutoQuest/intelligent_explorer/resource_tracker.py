# resource_tracker.py
"""
Resource tracking and knowledge base management for AutoQuest Phase 2
Handles KB mutations, constraint validation, and state tracking
"""

from typing import Optional, Dict, List, Any
import time
import re
from AutoQuest.intelligent_explorer.data_models import (
    NodeExecution, 
    NodeState, 
    ResourceNode, 
    EnvironmentKnowledgeBase
)
from AutoQuest.intelligent_explorer.tool_classifier import ToolOperation
from AutoQuest.intelligent_explorer.memory_manager import MemoryManager

from utils import normalize_operation


class ResourceTracker:
    """
    Tracks resources created/modified during exploration
    Maintains KB integrity and validates constraints
    """
    
    def __init__(
        self,
        tool_classifications: Dict,
        persistence_manager: Any,  # PersistenceManager instance
        memory_manager: 'MemoryManager',  # NEW
        interestingness_threshold: float = 0.3
    ):
        """
        Args:
            tool_classifications: Tool classification mappings
            persistence_manager: PersistenceManager for saving states
            interestingness_threshold: Threshold for "interesting" states
        """
        self.tool_classifications = tool_classifications
        self.persistence = persistence_manager
        self.interestingness_threshold = interestingness_threshold
        self.memory_manager = memory_manager  # NEW

        # Global tracking (across all environments)
        self.global_stats = {
            "total_resources_created": 0,
            "total_resources_updated": 0,
            "total_resources_deleted": 0,
            "constraint_violations": 0,
            "extraction_failures": 0
        }
    
    # ========================
    # Main Resource Tracking
    # ========================
    
    def track_resources(
        self,
        execution: NodeExecution,
        kb: EnvironmentKnowledgeBase,
        trajectory_id: str  # ✅ NEW parameter
    ):
        """
        Track resources with full context in structured KB
        Intelligently extracts resource IDs from various output formats
        
        Args:
            execution: NodeExecution result
            kb: EnvironmentKnowledgeBase to update
            trajectory_id: Current trajectory ID for short-term memory
        """
        classification = self.tool_classifications.get(execution.node_name, {})
        operation = normalize_operation(classification.get("operation"))
        
        # Track in short-term memory
        resource_type = classification.get("discovers_resource") or classification.get("produces_resource")
        resource_id = None
        
        if operation == ToolOperation.CREATE:
            resource_id = self._handle_create(execution, kb, classification)
        
        elif operation == ToolOperation.UPDATE:
            self._handle_update(execution, kb, classification)
        
        elif operation == ToolOperation.DELETE:
            self._handle_delete(execution, kb, classification)
        
        # For READ/LIST/SEARCH, just log access
        elif operation in [ToolOperation.READ, ToolOperation.LIST, ToolOperation.SEARCH]:
            self._handle_read(execution, kb, classification)
        
        # ✅ NEW: Record in short-term memory for ALL operations
        if self.memory_manager and trajectory_id:
            self.memory_manager.record_execution(
                trajectory_id=trajectory_id,
                tool_name=execution.node_name,
                tool_inputs=execution.tool_input,
                tool_outputs=execution.tool_output,
                success=execution.success,
                resource_type=resource_type,  # May be None for non-CREATE
                resource_id=resource_id       # May be None for non-CREATE
            )
        
        # Always add to execution chain
        kb.execution_chain.append({
            "tool_name": execution.node_name,
            "operation": operation.value if hasattr(operation, 'value') else str(operation),
            "inputs": execution.tool_input,
            "outputs": execution.tool_output,
            "timestamp": time.time(),
            "success": execution.success
        })
        
        kb.stats["total_executions"] += 1

    # Add THIS to resource_tracker.py track_resources() method:

    # def track_resources(self, execution: NodeExecution, kb: EnvironmentKnowledgeBase):
    #     """Track resources with full context"""
        
    #     print(f"\n      ╔══════════════════════════════════════╗")
    #     print(f"      ║   RESOURCE TRACKER VERBOSE DEBUG    ║")
    #     print(f"      ╚══════════════════════════════════════╝")
        
    #     classification = self.tool_classifications.get(execution.node_name, {})
    #     operation = normalize_operation(classification.get("operation"))
        
    #     print(f"      1️⃣  Node: {execution.node_name}")
    #     print(f"      2️⃣  Operation: {operation}")
    #     print(f"      3️⃣  Success: {execution.success}")
    #     print(f"      4️⃣  Output Type: {type(execution.tool_output)}")
    #     print(f"      5️⃣  Output Value (first 200 chars): {str(execution.tool_output)[:200]}")
        
    #     if operation == ToolOperation.CREATE:
    #         print(f"      6️⃣  ✓ Operation is CREATE, calling _handle_create()")
    #         self._handle_create(execution, kb, classification)
    #     else:
    #         print(f"      6️⃣  ✗ Operation is {operation}, not CREATE - skipping")
        
    #     print(f"      ═══════════════════════════════════════\n")

    def _handle_create(
        self,
        execution: NodeExecution,
        kb: EnvironmentKnowledgeBase,
        classification: Dict
    ):
        """Handle CREATE operation with verbose debugging"""

        print(f"      ┌─ _handle_create START ─┐")
        
        resource_type = classification.get("produces_resource")
        print(f"      │ Resource Type: '{resource_type}'")
        
        if not resource_type:
            print(f"      │ ❌ EXIT: No resource_type")
            print(f"      └─ _handle_create END ───┘")
            return
        
        output = execution.tool_output
        print(f"      │ Initial output type: {type(output)}")
        
        # Parse string
        if isinstance(output, str):
            print(f"      │ 🔧 Attempting to parse string...")
            import json
            import ast
            
            try:
                output = json.loads(output)
                print(f"      │ ✓ Parsed with json.loads()")
            except Exception as e1:
                print(f"      │ ✗ json.loads() failed: {str(e1)[:50]}")
                
                try:
                    output = ast.literal_eval(output)
                    print(f"      │ ✓ Parsed with ast.literal_eval()")
                except Exception as e2:
                    print(f"      │ ✗ ast.literal_eval() failed: {str(e2)[:50]}")
                    print(f"      │ ❌ EXIT: Cannot parse string")
                    print(f"      └─ _handle_create END ───┘")
                    return
        
        print(f"      │ After parsing, type: {type(output)}")
        
        # Handle list
        if isinstance(output, list):
            print(f"      │ Output is list with {len(output)} items")
            if not output:
                print(f"      │ ❌ EXIT: Empty list")
                print(f"      └─ _handle_create END ───┘")
                return
            output = output[0]
            print(f"      │ ✓ Using first item, type: {type(output)}")
        
        # Check dict
        if not isinstance(output, dict):
            print(f"      │ ❌ EXIT: Not dict, is {type(output)}")
            print(f"      └─ _handle_create END ───┘")
            return
        
        print(f"      │ ✓ Output is dict")
        print(f"      │ Keys: {list(output.keys())}")
        
        # ✅ Extract ID ONCE from the dict
        print(f"      │ Calling _extract_resource_id()...")
        resource_id = self._extract_resource_id(output, resource_type, execution.node_name)
        print(f"      │ Extracted ID: '{resource_id}'")
        
        if not resource_id:
            print(f"      │ ❌ EXIT: No resource ID")
            print(f"      └─ _handle_create END ───┘")
            self.global_stats["extraction_failures"] += 1
            return
        
        print(f"      │ ✅ SUCCESS: Will add resource {resource_type}/{resource_id}")
        
        # ❌ REMOVE THIS DUPLICATE EXTRACTION BLOCK:
        # resource_id = self._extract_resource_id(...)
        # if not resource_id: return
        
        # Extract parent resources
        print(f"      │ Extracting parent resources...")
        parent_resources = self._extract_parent_resources(execution.tool_input, kb)
        print(f"      │ Parent resources: {parent_resources}")
        
        # Validate constraints
        print(f"      │ Validating constraints...")
        if not self._validate_create_constraints(resource_id, resource_type, parent_resources, kb):
            print(f"      │ ❌ EXIT: Constraint validation failed")
            print(f"      └─ _handle_create END ───┘")
            self.global_stats["constraint_violations"] += 1
            return
        
        print(f"      │ ✓ Constraints validated")
        
        # Create ResourceNode
        print(f"      │ Creating ResourceNode...")
        resource_node = ResourceNode(
            resource_id=str(resource_id),
            resource_type=resource_type,
            created_by_tool=execution.node_name,
            creation_inputs=execution.tool_input.copy(),
            creation_outputs=output.copy(),  # ← Use 'output' (the dict), not execution.tool_output
            creation_timestamp=time.time(),
            parent_resources=parent_resources,
            metadata={
                "execution_time": execution.execution_time,
                "success": execution.success,
                "interestingness_score": execution.interestingness_score
            }
        )
        
        print(f"      │ ✓ ResourceNode created")
        
        # Add to KB
        print(f"      │ Adding to KB...")
        kb.add_resource(resource_node)
        
        self.global_stats["total_resources_created"] += 1
        
        print(f"      │ ✅ Resource added to KB!")
        print(f"      │ KB now has {len(kb.resources)} resources")
        print(f"      └─ _handle_create END ───┘")
        return resource_id

    def _handle_update(
        self,
        execution: NodeExecution,
        kb: EnvironmentKnowledgeBase,
        classification: Dict
    ):
        """Handle UPDATE operation"""
        if not isinstance(execution.tool_output, dict):
            return
        
        # Find which resource was updated (usually from input params)
        resource_id = self._find_target_resource_id(execution.tool_input, kb)
        
        if not resource_id:
            print(f"      ⚠️  Could not identify updated resource")
            return
        
        # Extract updates from output
        updates = execution.tool_output.copy()
        
        try:
            kb.update_resource(resource_id, updates)
            self.global_stats["total_resources_updated"] += 1
            print(f"      🔄 Updated resource: {resource_id}")
        except ValueError as e:
            print(f"      ⚠️  Update failed: {e}")
    
    def _handle_delete(
        self,
        execution: NodeExecution,
        kb: EnvironmentKnowledgeBase,
        classification: Dict
    ):
        """Handle DELETE operation"""
        # Find which resource was deleted
        resource_id = self._find_target_resource_id(execution.tool_input, kb)
        
        if not resource_id:
            print(f"      ⚠️  Could not identify deleted resource")
            return
        
        # Check if resource has dependents
        if resource_id in kb.resource_references and kb.resource_references[resource_id]:
            dependents = kb.resource_references[resource_id]
            print(f"      ⚠️  Warning: Deleting {resource_id} which has {len(dependents)} dependents")
        
        kb.delete_resource(resource_id, soft_delete=True)
        self.global_stats["total_resources_deleted"] += 1
        print(f"      🗑️  Deleted resource: {resource_id}")
    
    def _handle_read(
        self,
        execution: NodeExecution,
        kb: EnvironmentKnowledgeBase,
        classification: Dict
    ):
        """
        Handle READ/LIST/SEARCH operations with resource discovery
        
        - READ: Track access to existing resources
        - LIST/SEARCH: Discover and add new resources to KB
        
        Args:
            execution: NodeExecution result
            kb: EnvironmentKnowledgeBase to update
            classification: Tool classification metadata
        """
        print(f"      ┌─ _handle_read START ─┐")
        
        operation = normalize_operation(classification.get("operation"))
        # ✅ Check both discovers_resource (LIST/SEARCH) and produces_resource (CREATE)
        resource_type = classification.get("discovers_resource") or classification.get("produces_resource")
        
        print(f"      │ Operation: {operation}")
        print(f"      │ Resource Type: '{resource_type}'")
        
        output = execution.tool_output
        print(f"      │ Initial output type: {type(output)}")
        
        # Parse string output if needed
        if isinstance(output, str):
            print(f"      │ 🔧 Attempting to parse string...")
            import json
            import ast
            
            try:
                output = json.loads(output)
                print(f"      │ ✓ Parsed with json.loads()")
            except Exception as e1:
                print(f"      │ ✗ json.loads() failed: {str(e1)[:50]}")
                
                try:
                    output = ast.literal_eval(output)
                    print(f"      │ ✓ Parsed with ast.literal_eval()")
                except Exception as e2:
                    print(f"      │ ✗ ast.literal_eval() failed: {str(e2)[:50]}")
                    print(f"      │ ⚠️  Cannot parse, will check tool_input for READ operations")
        
        print(f"      │ After parsing, type: {type(output)}")
        
        accessed_resources = []      # Resources already in KB
        discovered_resources = []    # New resources to add to KB
        
        # ==================== STRATEGY 1: READ Operation ====================
        if operation == ToolOperation.READ:
            print(f"      │ READ operation - checking tool_input for target ID...")
            resource_id = self._find_target_resource_id(execution.tool_input, kb)
            
            if resource_id:
                print(f"      │ ✓ Found target resource: {resource_id}")
                
                # Check if resource exists in KB
                if kb.get_resource(resource_id):
                    accessed_resources.append(resource_id)
                    print(f"      │ ✓ Resource exists in KB")
                else:
                    # Resource doesn't exist - try to discover it from output
                    if isinstance(output, dict) and resource_type:
                        print(f"      │ ⚠️  Resource not in KB, attempting to discover from output...")
                        discovered_resources.append((resource_id, output))
                    else:
                        print(f"      │ ⚠️  Resource not in KB and cannot discover")
            else:
                print(f"      │ ⚠️  No target resource in tool_input")
        
        # ==================== STRATEGY 2: LIST/SEARCH Operations ====================
        if operation in [ToolOperation.LIST, ToolOperation.SEARCH]:
            print(f"      │ {operation} operation - discovering resources from output...")
            
            # Handle list output
            if isinstance(output, list):
                print(f"      │ Output is list with {len(output)} items")
                
                for idx, item in enumerate(output):
                    if isinstance(item, dict):
                        item_id = self._extract_resource_id(item, resource_type, execution.node_name)
                        
                        if item_id:
                            # Check if already in KB
                            existing = kb.get_resource(item_id)
                            if existing:
                                accessed_resources.append(item_id)
                            else:
                                # New resource discovered
                                discovered_resources.append((item_id, item))
                    elif isinstance(item, (str, int)):
                        # Simple ID list like ['id1', 'id2', 'id3']
                        item_id = str(item)
                        if kb.get_resource(item_id):
                            accessed_resources.append(item_id)
                        else:
                            # Create minimal resource data
                            discovered_resources.append((item_id, {"id": item_id}))
                
                print(f"      │ ✓ Found {len(accessed_resources)} existing, {len(discovered_resources)} new")
            
            # Handle dict output (often paginated responses)
            elif isinstance(output, dict):
                print(f"      │ Output is dict, checking for results/records...")
                
                # Common pagination/collection keys
                collection_keys = [
                    'results', 'records', 'data', 'items', 
                    'cases', 'accounts', 'issues', 'contacts',
                    'users', 'opportunities', 'leads'
                ]
                
                found_collection = False
                for key in collection_keys:
                    if key in output and isinstance(output[key], list):
                        print(f"      │ Found '{key}' with {len(output[key])} items")
                        
                        for item in output[key]:
                            if isinstance(item, dict):
                                item_id = self._extract_resource_id(item, resource_type, execution.node_name)
                                
                                if item_id:
                                    existing = kb.get_resource(item_id)
                                    if existing:
                                        accessed_resources.append(item_id)
                                    else:
                                        discovered_resources.append((item_id, item))
                        
                        print(f"      │ ✓ Extracted {len(accessed_resources)} existing, {len(discovered_resources)} new from '{key}'")
                        found_collection = True
                        break
                
                # Single resource in dict format
                if not found_collection and resource_type:
                    print(f"      │ Trying to extract single resource from dict...")
                    item_id = self._extract_resource_id(output, resource_type, execution.node_name)
                    
                    if item_id:
                        if kb.get_resource(item_id):
                            accessed_resources.append(item_id)
                            print(f"      │ ✓ Found existing resource")
                        else:
                            discovered_resources.append((item_id, output))
                            print(f"      │ ✓ Discovered new resource")
            
            else:
                print(f"      │ ⚠️  Unexpected output type: {type(output)}")
        
        # ==================== ADD DISCOVERED RESOURCES TO KB ====================
        if discovered_resources:
            print(f"      │ 📦 Adding {len(discovered_resources)} discovered resources to KB...")
            
            added_count = 0
            for resource_id, resource_data in discovered_resources:
                try:
                    # Create ResourceNode for discovered resource
                    resource_node = ResourceNode(
                        resource_id=str(resource_id),
                        resource_type=resource_type,
                        created_by_tool=execution.node_name,
                        creation_inputs=execution.tool_input.copy() if execution.tool_input else {},
                        creation_outputs=resource_data.copy() if isinstance(resource_data, dict) else {"id": resource_data},
                        creation_timestamp=time.time(),
                        parent_resources=[],  # Discovered resources have no parent dependencies
                        metadata={
                            "discovery_method": operation.value,  # 'read', 'list', or 'search'
                            "execution_time": execution.execution_time,
                            "success": execution.success,
                            "is_discovered": True,  # Flag to distinguish from CREATE operations
                            "interestingness_score": execution.interestingness_score
                        }
                    )
                    
                    # Add to KB
                    kb.add_resource(resource_node)
                    added_count += 1
                    print(f"      │   ✓ Added {resource_type}/{resource_id}")
                    
                except Exception as e:
                    print(f"      │   ✗ Failed to add {resource_id}: {str(e)[:50]}")
                    self.global_stats["discovery_failures"] = self.global_stats.get("discovery_failures", 0) + 1
            
            # Update global stats
            self.global_stats["total_resources_discovered"] = self.global_stats.get("total_resources_discovered", 0) + added_count
            print(f"      │ ✅ Successfully added {added_count}/{len(discovered_resources)} resources")
        
        # ==================== MARK EXISTING RESOURCES AS ACCESSED ====================
        print(f"      │ Marking {len(accessed_resources)} existing resources as accessed...")
        access_count = 0
        
        for resource_id in accessed_resources:
            resource = kb.get_resource(resource_id)
            if resource:
                # Access count is automatically incremented by get_resource
                access_count += 1
                print(f"      │   ✓ {resource_id} (access #{resource.access_count})")
            else:
                print(f"      │   ⚠️  {resource_id} not found in KB (race condition?)")
        
        # ==================== UPDATE OPERATION-SPECIFIC STATS ====================
        if operation == ToolOperation.READ:
            self.global_stats["total_reads"] = self.global_stats.get("total_reads", 0) + 1
        elif operation == ToolOperation.LIST:
            self.global_stats["total_lists"] = self.global_stats.get("total_lists", 0) + 1
            self.global_stats["total_items_listed"] = self.global_stats.get("total_items_listed", 0) + len(discovered_resources) + len(accessed_resources)
        elif operation == ToolOperation.SEARCH:
            self.global_stats["total_searches"] = self.global_stats.get("total_searches", 0) + 1
            self.global_stats["total_items_found"] = self.global_stats.get("total_items_found", 0) + len(discovered_resources) + len(accessed_resources)
        
        # ==================== CACHE QUERY RESULTS (OPTIONAL) ====================
        if (discovered_resources or accessed_resources) and resource_type:
            print(f"      │ 💾 Storing query results in KB cache...")
            
            if not hasattr(kb, 'query_cache'):
                kb.query_cache = {}
            
            cache_key = f"{operation.value}_{resource_type}_{hash(str(execution.tool_input))}"
            kb.query_cache[cache_key] = {
                "resource_ids": [rid for rid, _ in discovered_resources] + accessed_resources,
                "timestamp": time.time(),
                "tool_name": execution.node_name,
                "total_count": len(discovered_resources) + len(accessed_resources),
                "newly_discovered": len(discovered_resources),
                "already_known": len(accessed_resources)
            }
        
        # ==================== FINAL SUMMARY ====================
        total_resources = len(discovered_resources) + len(accessed_resources)
        print(f"      │ ✅ Discovered {len(discovered_resources)} new, tracked {len(accessed_resources)} existing")
        print(f"      │ KB now has {len(kb.resources)} total resources")
        print(f"      │ Stats: {self.global_stats.get(f'total_{operation.value.lower()}s', 0)} total {operation.value}s")
        print(f"      └─ _handle_read END ───┘")


    
    # ========================
    # Resource ID Extraction
    # ========================
    
    def _extract_resource_id(
        self,
        output: Dict,
        resource_type: str,
        tool_name: str
    ) -> Optional[str]:
        """
        Intelligently extract resource ID from tool output
        
        Args:
            output: Tool output dictionary
            resource_type: Expected resource type
            tool_name: Name of tool that produced output
            
        Returns:
            Extracted resource ID or None
        """
        if not isinstance(output, dict):
            return None
        
        # Priority 1: Common ID field patterns
        id_patterns = [
            "id",
            f"{resource_type}_id",
            f"{resource_type}Id",
            "resource_id",
            "resourceId"
        ]
        
        for pattern in id_patterns:
            if pattern in output and output[pattern]:
                return str(output[pattern])
        
        # Priority 2: Tool-specific patterns (e.g., GitLab uses 'iid' for issues)
        specific_patterns = {
            "issue": ["iid", "issue_iid"],
            "merge_request": ["iid", "merge_request_iid", "mr_iid"],
            "project": ["project_id", "path_with_namespace", "full_path"],
            "repository": ["name", "path"],
            "branch": ["name", "ref"]
        }
        
        if resource_type in specific_patterns:
            for pattern in specific_patterns[resource_type]:
                if pattern in output and output[pattern]:
                    return str(output[pattern])
        
        # Priority 3: Scan for any field with 'id' in name
        for key, value in output.items():
            if 'id' in key.lower() and isinstance(value, (str, int)) and value:
                print(f"      🔍 Found resource ID in field '{key}': {value}")
                return str(value)
        
        # Priority 4: Use 'name' or 'path' as fallback
        for fallback_key in ['name', 'path', 'key', 'slug']:
            if fallback_key in output and output[fallback_key]:
                print(f"      🔍 Using '{fallback_key}' as resource ID: {output[fallback_key]}")
                return str(output[fallback_key])
        
        return None
    
    def _extract_parent_resources(
        self,
        tool_input: Dict,
        kb: EnvironmentKnowledgeBase
    ) -> List[str]:
        """
        Extract parent resource IDs from tool inputs
        
        Args:
            tool_input: Tool input parameters
            kb: Knowledge base to validate against
            
        Returns:
            List of parent resource IDs
        """
        parent_resources = []
        
        # Look for fields that reference resources
        id_suffixes = ["_id", "_iid", "_ref", "Id", "_key"]
        
        for param_name, param_value in tool_input.items():
            # Check if parameter name suggests it's a resource reference
            if any(suffix in param_name for suffix in id_suffixes):
                resource_id = str(param_value)
                
                # Verify it exists in KB
                if kb.get_resource(resource_id):
                    parent_resources.append(resource_id)
        
        return parent_resources
    
    def _find_target_resource_id(
        self,
        tool_input: Dict,
        kb: EnvironmentKnowledgeBase
    ) -> Optional[str]:
        """
        Find the target resource ID from tool inputs (for UPDATE/DELETE/READ)
        
        Args:
            tool_input: Tool input parameters
            kb: Knowledge base
            
        Returns:
            Target resource ID or None
        """
        # Check common parameter names
        for param_name in ['id', 'resource_id', 'iid', 'project_id', 'issue_id']:
            if tool_input and param_name in tool_input:
                resource_id = str(tool_input[param_name])
                if kb.get_resource(resource_id):
                    return resource_id
        
        # Scan all parameters with 'id' suffix
        if tool_input is None:
            return None
        for param_name, param_value in tool_input.items():
            if '_id' in param_name or '_iid' in param_name:
                resource_id = str(param_value)
                if kb.get_resource(resource_id):
                    return resource_id
        
        return None
    
    # ========================
    # Constraint Validation
    # ========================
    
    def _validate_create_constraints(
        self,
        resource_id: str,
        resource_type: str,
        parent_resources: List[str],
        kb: EnvironmentKnowledgeBase
    ) -> bool:
        """
        Validate constraints before creating a resource
        
        Args:
            resource_id: ID of resource to create
            resource_type: Type of resource
            parent_resources: Parent resource IDs
            kb: Knowledge base
            
        Returns:
            True if valid, False otherwise
        """
        # Check uniqueness
        if resource_id in kb.resources:
            existing = kb.resources[resource_id]
            if not existing.is_deleted:
                print(f"      ⚠️  Resource {resource_id} already exists")
                return False
        
        # Check parent existence
        for parent_id in parent_resources:
            parent = kb.get_resource(parent_id)
            if not parent:
                print(f"      ⚠️  Parent resource {parent_id} not found")
                return False
            if parent.is_deleted:
                print(f"      ⚠️  Parent resource {parent_id} is deleted")
                return False
        
        return True
    
    # ========================
    # Node State Tracking
    # ========================
    
    async def save_node_state(
        self, 
        execution: NodeExecution, 
        node_name: str, 
        kb: EnvironmentKnowledgeBase
    ):
        """
        Save node execution state with KB context
        
        Args:
            execution: NodeExecution to save
            node_name: Name of the node
            kb: Current knowledge base state
        """
        # Extract resource summary from KB
        created_resources_summary = {
            resource_type: [
                kb.resources[rid].resource_id 
                for rid in resource_ids
                if not kb.resources[rid].is_deleted
            ]
            for resource_type, resource_ids in kb.resource_by_type.items()
        }
        
        node_state = NodeState(
            node_name=node_name,
            tool_input=execution.tool_input,
            tool_output=execution.tool_output,
            execution_time=execution.execution_time,
            timestamp=time.time(),
            success=execution.success,
            error=execution.error,
            interestingness_score=execution.interestingness_score,
            created_resources=created_resources_summary
        )
        
        # Save via persistence manager
        await self.persistence.save_node_state(node_state, kb.environment_name)
        
        # Track interesting states
        if node_state.interestingness_score >= self.interestingness_threshold:
            print(f"      ⭐ Interesting state detected (score: {node_state.interestingness_score:.2f})")
    
    # ========================
    # Utilities
    # ========================
    
    def get_stats(self) -> Dict:
        """Get global tracking statistics"""
        return self.global_stats.copy()
    
    def reset_stats(self):
        """Reset global statistics"""
        for key in self.global_stats:
            self.global_stats[key] = 0
