# graph_utils.py
"""
Graph traversal and neighbor discovery utilities for AutoQuest Phase 2
Handles tool dependency analysis and KB-aware exploration
"""

from typing import List, Dict, Any, Set, Optional, Tuple
from AutoQuest.intelligent_explorer.data_models import EnvironmentKnowledgeBase
from AutoQuest.intelligent_explorer.tool_classifier import ToolOperation, ToolClassifier
from utils import normalize_operation


class GraphUtils:
    """Utilities for graph traversal and schema extraction"""
    
    def __init__(
        self, 
        tools: Dict, 
        tool_classifications: Dict, 
        graph: Dict, 
        tool_classifier: ToolClassifier
    ):
        """
        Args:
            tools: Dictionary of tool_name -> tool object
            tool_classifications: Tool classification mappings
            graph: Dependency graph (tool_name -> [neighbor_names])
            tool_classifier: ToolClassifier instance
        """
        self.tools = tools
        self.tool_classifications = tool_classifications
        self.graph = graph
        self.tool_classifier = tool_classifier
        self.adjacency_list = self._build_adjacency_list(graph)
        # Cache for expensive operations
        self._schema_cache: Dict[str, Dict] = {}
        self._neighbor_cache: Dict[str, List[str]] = {}
    def _build_adjacency_list(self, graph: Dict) -> Dict[str, List[str]]:
        """
        Convert Phase 1 edge format to adjacency list
        
        Converts:
            {"edges": {"A->B": {...}, "A->C": {...}}}
        To:
            {"A": ["B", "C"]}
        """
        adjacency = {}
        
        edges = graph.get("edges", {})
        
        for edge_key, edge_data in edges.items():
            source = edge_data.get("source")
            target = edge_data.get("target")
            
            
            if source and target:
                if source not in adjacency:
                    adjacency[source] = []
                adjacency[source].append(target)
        
        print(f"   ✓ Built adjacency list: {len(adjacency)} nodes with edges")
        
        # Debug: show sample
        sample_nodes = list(adjacency.keys())[:5]
        for node in sample_nodes:
            print(f"      • {node} → {len(adjacency[node])} neighbors")
        
        return adjacency
    # ========================
    # Initial Node Discovery
    # ========================
    
    def identify_initial_nodes(self, kb: Optional[EnvironmentKnowledgeBase] = None) -> List[str]:
        """
        Identify nodes that can start exploration
        KB-aware: Returns MORE nodes when KB has resources
        
        Args:
            kb: Knowledge base (if provided, includes nodes that can use existing resources)
            
        Returns:
            List of executable tool names
        """
        initial_nodes = []
        
        # Strategy 1: CREATE tools (always valid)
        create_tools = self.tool_classifier.get_tools_by_operation(ToolOperation.CREATE)
        initial_nodes.extend(create_tools)
        
        # Strategy 2: LIST/SEARCH with no requirements
        list_tools = self.tool_classifier.get_tools_by_operation(ToolOperation.LIST)
        search_tools = self.tool_classifier.get_tools_by_operation(ToolOperation.SEARCH)
        
        for tool_name in list_tools + search_tools:
            if tool_name in self.tools:
                required_params = self.get_required_parameters(self.tools[tool_name])
                if not required_params:
                    initial_nodes.append(tool_name)
        
        # ✅ Strategy 3: KB-aware expansion (if KB has resources)
        if kb and len(kb.resources) > 0:
            print(f"\n   🔥 KB-AWARE MODE: KB has {len(kb.resources)} resources!")
            print(f"   📊 Expanding initial nodes to include READ/UPDATE/DELETE operations...")
            
            # Add READ operations that can use KB resources
            read_tools = self.tool_classifier.get_tools_by_operation(ToolOperation.READ)
            for tool_name in read_tools:
                if self._can_tool_execute(tool_name, kb):
                    initial_nodes.append(tool_name)
                    print(f"      ✓ {tool_name} (can use existing resources)")
            
            # Add UPDATE operations
            update_tools = self.tool_classifier.get_tools_by_operation(ToolOperation.UPDATE)
            for tool_name in update_tools:
                if self._can_tool_execute(tool_name, kb):
                    initial_nodes.append(tool_name)
                    print(f"      ✓ {tool_name} (can use existing resources)")
            
            # Add DELETE operations
            delete_tools = self.tool_classifier.get_tools_by_operation(ToolOperation.DELETE)
            for tool_name in delete_tools:
                if self._can_tool_execute(tool_name, kb):
                    initial_nodes.append(tool_name)
                    print(f"      ✓ {tool_name} (can use existing resources)")
            
            # Add LIST/SEARCH that require resources
            for tool_name in list_tools + search_tools:
                if tool_name not in initial_nodes and self._can_tool_execute(tool_name, kb):
                    initial_nodes.append(tool_name)
                    print(f"      ✓ {tool_name} (can use existing resources)")
        
        # Remove duplicates
        seen = set()
        deduplicated = []
        for node in initial_nodes:
            if node not in seen:
                seen.add(node)
                deduplicated.append(node)
        
        return deduplicated

    
    # ========================
    # Neighbor Discovery
    # ========================
    
    def get_neighbors(
        self, 
        node_name: str,
        kb: Optional[EnvironmentKnowledgeBase] = None,
        filter_by_kb: bool = True,
        recent_execution: Optional[Dict] = None  # ✅ NEW: Pass recent execution
    ) -> List[str]:
        """
        Get neighbor nodes with smart KB-aware filtering
        
        Args:
            node_name: Current node name
            kb: Knowledge base
            filter_by_kb: If True, filter by KB availability
            recent_execution: Recent execution result (to check what was just created)
            
        Returns:
            List of neighbor tool names
        """
        cache_key = f"{node_name}_{filter_by_kb}_{kb.version if kb else 'none'}"
        if cache_key in self._neighbor_cache:
            return self._neighbor_cache[cache_key].copy()
        
        neighbors = []
        
        # Strategy 1: Explicit graph edges
        if node_name in self.adjacency_list:
            neighbors.extend(self.adjacency_list[node_name])
            print(f"      📊 Found {len(self.adjacency_list[node_name])} explicit edges")
        
        # Strategy 2: Resource-based inference
        classification = self.tool_classifications.get(node_name, {})
        produces = classification.get("produces_resource")
        
        if produces:
            inferred_neighbors = 0
            for tool_name, tool_class in self.tool_classifications.items():
                if tool_name == node_name:
                    continue
                
                requires = tool_class.get("requires_resources", [])
                
                if any(produces.lower() in req.lower() for req in requires):
                    if tool_name not in neighbors:
                        neighbors.append(tool_name)
                        inferred_neighbors += 1
            
            if inferred_neighbors > 0:
                print(f"      🧠 Inferred {inferred_neighbors} additional neighbors")
        
        # ✅ Strategy 3: SMART KB filtering
        if filter_by_kb and kb:
            original_count = len(neighbors)
            filtered_neighbors = []
            
            for neighbor in neighbors:
                # Check if this neighbor can be satisfied
                can_execute, reason = self.can_satisfy_requirements_from_kb_smart(
                    source_node=node_name,
                    target_node=neighbor,
                    kb=kb,
                    recent_execution=recent_execution
                )
                
                if can_execute:
                    filtered_neighbors.append(neighbor)
                else:
                    print(f"      🔒 Filtered {neighbor}: {reason}")
            
            neighbors = filtered_neighbors
            
            filtered_count = original_count - len(neighbors)
            if filtered_count > 0:
                print(f"      ⚠️  Filtered {filtered_count}/{original_count} neighbors")
        
        self._neighbor_cache[cache_key] = neighbors.copy()
        return neighbors


    def _fields_semantically_match(self, field1: str, field2: str) -> bool:
        """
        Check if two field names are semantically equivalent
        Handles prefix variations like sender_emp_id ≈ emp_id
        
        Args:
            field1: First field name
            field2: Second field name
            
        Returns:
            True if fields are semantically equivalent
        """
        # Normalize
        f1 = field1.lower().replace('-', '_')
        f2 = field2.lower().replace('-', '_')
        
        # Exact match
        if f1 == f2:
            return True
        
        # Strip common prefixes and check
        prefixes = ['sender_', 'recipient_', 'source_', 'target_', 'from_', 'to_', 
                    'raised_by_', 'assigned_to_', 'created_by_', 'updated_by_']
        
        f1_stripped = f1
        f2_stripped = f2
        
        for prefix in prefixes:
            if f1.startswith(prefix):
                f1_stripped = f1[len(prefix):]
            if f2.startswith(prefix):
                f2_stripped = f2[len(prefix):]
        
        # Check if stripped versions match
        if f1_stripped == f2_stripped:
            return True
        
        # Check if one contains the other
        if f1_stripped in f2_stripped or f2_stripped in f1_stripped:
            return True
        
        return False


    def can_satisfy_requirements_from_kb_smart(
        self,
        source_node: str,
        target_node: str,
        kb: EnvironmentKnowledgeBase,
        recent_execution: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Smart requirement checking - checks if KB collectively has required fields
        """
        target_classification = self.tool_classifications.get(target_node, {})
        operation = normalize_operation(target_classification.get("operation"))
        required_resources = target_classification.get("requires_resources", [])
        
        # CREATE/LIST/SEARCH with no requirements → always executable
        if operation in [ToolOperation.CREATE, ToolOperation.LIST, ToolOperation.SEARCH]:
            if not required_resources:
                return (True, "No requirements")
        
        # No requirements → always executable
        if not required_resources:
            return (True, "No requirements")
        
        # ✅ NEW APPROACH: Check if KB collectively satisfies requirements
        # Instead of checking only source node's output, check ALL KB resources
        
        edge_key = f"{source_node}->{target_node}"
        edge_data = self.graph.get("edges", {}).get(edge_key)
        
        if edge_data and edge_data.get("field_mappings"):
            # Check each required field
            all_fields_satisfied = True
            missing_fields = []
            satisfied_sources = []
            
            for mapping in edge_data["field_mappings"]:
                source_field = mapping.get("source_field")
                target_field = mapping.get("target_field")
                
                field_found = False
                
                # Strategy 1: Check recent execution
                if recent_execution and isinstance(recent_execution, dict):
                    if source_field in recent_execution:
                        field_found = True
                        satisfied_sources.append(f"{target_field} from recent")
                    else:
                        for exec_field in recent_execution.keys():
                            if self._fields_semantically_match(source_field, exec_field):
                                field_found = True
                                satisfied_sources.append(f"{target_field} from recent ({exec_field})")
                                break
                
                # Strategy 2: Check ALL resources in KB (not just source node's type)
                if not field_found:
                    # ✅ Search across ALL resource types in KB
                    for resource_type, resource_ids in kb.resource_by_type.items():
                        if field_found:
                            break
                        
                        for resource_id in resource_ids:
                            resource = kb.resources.get(resource_id)
                            if not resource or resource.is_deleted:
                                continue
                            
                            # Check creation outputs for the field
                            outputs = resource.creation_outputs
                            
                            # Direct match
                            if source_field in outputs:
                                field_found = True
                                satisfied_sources.append(f"{target_field} from {resource_type}")
                                break
                            
                            # Semantic match
                            for kb_field in outputs.keys():
                                if self._fields_semantically_match(source_field, kb_field):
                                    field_found = True
                                    satisfied_sources.append(f"{target_field} from {resource_type} ({kb_field})")
                                    break
                            
                            if field_found:
                                break
                
                if not field_found:
                    all_fields_satisfied = False
                    missing_fields.append(target_field)
            
            if all_fields_satisfied:
                return (True, f"All fields available: {', '.join(satisfied_sources[:5])}")
            else:
                return (False, f"Missing fields: {missing_fields}")
        
        # Fallback: Check resource types
        missing_types = []
        
        for resource_type in required_resources:
            # Direct or fuzzy match
            if resource_type in kb.resource_by_type:
                active_resources = [
                    rid for rid in kb.resource_by_type[resource_type]
                    if not kb.resources[rid].is_deleted
                ]
                if active_resources:
                    continue
            
            # Fuzzy match
            found = False
            for kb_type in kb.resource_by_type.keys():
                resource_type_norm = resource_type.lower().replace('_', '').replace('-', '')
                kb_type_norm = kb_type.lower().replace('_', '').replace('-', '')
                
                if (resource_type_norm in kb_type_norm or 
                    kb_type_norm in resource_type_norm):
                    active_resources = [
                        rid for rid in kb.resource_by_type[kb_type]
                        if not kb.resources[rid].is_deleted
                    ]
                    if active_resources:
                        found = True
                        break
            
            if not found:
                missing_types.append(resource_type)
        
        if missing_types:
            return (False, f"Missing resource types: {missing_types}")
        
        return (True, "All requirements satisfied")


    
    def get_missing_requirements(
        self,
        node_name: str,
        kb: EnvironmentKnowledgeBase
    ) -> List[str]:
        """
        Get list of missing resource types for a node
        
        Args:
            node_name: Tool name
            kb: Knowledge base
            
        Returns:
            List of missing resource types
        """
        classification = self.tool_classifications.get(node_name, {})
        required_resources = classification.get("requires_resources", [])
        
        missing = []
        for resource_type in required_resources:
            # Check direct and fuzzy matches
            found = False
            
            if resource_type in kb.resource_by_type:
                active_resources = [
                    rid for rid in kb.resource_by_type[resource_type]
                    if not kb.resources[rid].is_deleted
                ]
                if active_resources:
                    found = True
            else:
                for kb_resource_type in kb.resource_by_type.keys():
                    if resource_type.lower() in kb_resource_type.lower():
                        active_resources = [
                            rid for rid in kb.resource_by_type[kb_resource_type]
                            if not kb.resources[rid].is_deleted
                        ]
                        if active_resources:
                            found = True
                            break
            
            if not found:
                missing.append(resource_type)
        
        return missing
    
    # ========================
    # Schema Extraction
    # ========================
    
    def get_tool_schema(self, tool: Any) -> Dict:
        """
        Extract tool args_schema
        
        Args:
            tool: Tool object
            
        Returns:
            Schema dictionary
        """
        tool_name = getattr(tool, 'name', str(tool))
        
        # Check cache
        if tool_name in self._schema_cache:
            return self._schema_cache[tool_name].copy()
        
        schema = {}
        try:
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema
                
                # If it's a Pydantic model, extract schema
                if hasattr(schema, 'schema'):
                    schema = schema.schema()
        except Exception as e:
            print(f"      ⚠️  Schema extraction error for {tool_name}: {e}")
        
        # Cache it
        self._schema_cache[tool_name] = schema.copy() if schema else {}
        
        return schema
    
    def get_required_parameters(self, tool: Any) -> List[str]:
        """
        Extract required parameters from tool schema
        
        Args:
            tool: Tool object
            
        Returns:
            List of required parameter names
        """
        try:
            schema = self.get_tool_schema(tool)
            
            if not schema:
                return []
            
            # Handle different schema formats
            # print(f"Schema: {schema}")
            required = schema.get('required', [])
            
            # If required is a dict (some schemas have this format)
            if isinstance(required, dict):
                return list(required.keys())
            
            # If it's already a list
            if isinstance(required, list):
                return required
            
        except Exception as e:
            tool_name = getattr(tool, 'name', str(tool))
            print(f"      ⚠️  Required params extraction error for {tool_name}: {e}")
        
        return []
    
    def get_all_parameters(self, tool: Any) -> List[str]:
        """
        Extract all parameters (required + optional) from tool schema
        
        Args:
            tool: Tool object
            
        Returns:
            List of all parameter names
        """
        try:
            schema = self.get_tool_schema(tool)
            
            if not schema:
                return []
            
            properties = schema.get('properties', {})
            
            if hasattr(properties, 'keys'):
                return list(properties.keys())
            
        except Exception as e:
            tool_name = getattr(tool, 'name', str(tool))
            print(f"      ⚠️  All params extraction error for {tool_name}: {e}")
        
        return []
    
    def get_parameter_types(self, tool: Any) -> Dict[str, str]:
        """
        Get parameter names mapped to their types
        
        Args:
            tool: Tool object
            
        Returns:
            Dict of param_name -> type_string
        """
        param_types = {}
        
        try:
            schema = self.get_tool_schema(tool)
            properties = schema.get('properties', {})
            for param_name, param_schema in properties.items():
                if(not isinstance(param_schema, dict)):
                    continue
                param_type = param_schema.get('type', 'string')
                param_types[param_name] = param_type
        
        except Exception as e:
            tool_name = getattr(tool, 'name', str(tool))
            print(f"      ⚠️  Param types extraction error for {tool_name}: {e}")
        
        return param_types
    
    # ========================
    # Graph Analysis
    # ========================
    
    def get_graph_statistics(self, kb: Optional[EnvironmentKnowledgeBase] = None) -> Dict:
        """
        Get statistics about the tool graph
        
        Args:
            kb: Optional KB for KB-aware stats
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_tools": len(self.tools),
            "graph_edges": sum(len(neighbors) for neighbors in self.adjacency_list.values()),  # ✅ Use adjacency_list
            "tools_by_operation": {},
            "initial_nodes": len(self.identify_initial_nodes())
        }
        
        # Count by operation type
        for operation in ToolOperation:
            tools = self.tool_classifier.get_tools_by_operation(operation)
            stats["tools_by_operation"][operation.value] = len(tools)
        
        # KB-aware stats
        if kb:
            stats["kb_version"] = kb.version
            stats["kb_resources"] = len(kb.resources)
            stats["kb_resource_types"] = len(kb.resource_by_type)
            
            # ✅ Count executable tools using simpler method
            executable_count = 0
            for tool_name in self.tools.keys():
                # Use the simpler check method
                if self._can_tool_execute(tool_name, kb):
                    executable_count += 1
            
            stats["executable_tools"] = executable_count
            stats["execution_coverage"] = (executable_count / len(self.tools) * 100) if self.tools else 0
        
        return stats

    def _can_tool_execute(self, tool_name: str, kb: EnvironmentKnowledgeBase) -> bool:
        """
        Simple check if a tool can execute given current KB state
        
        Args:
            tool_name: Tool name
            kb: Knowledge base
            
        Returns:
            True if tool can execute
        """
        classification = self.tool_classifications.get(tool_name, {})
        operation = normalize_operation(classification.get("operation"))
        required_resources = classification.get("requires_resources", [])
        
        # CREATE/LIST/SEARCH with no requirements → always executable
        if operation in [ToolOperation.CREATE, ToolOperation.LIST, ToolOperation.SEARCH]:
            if not required_resources:
                return True
        
        # No requirements → always executable
        if not required_resources:
            return True
        
        # Check if all required resource types exist in KB
        for resource_type in required_resources:
            # Direct match
            if resource_type in kb.resource_by_type:
                active_resources = [
                    rid for rid in kb.resource_by_type[resource_type]
                    if not kb.resources[rid].is_deleted
                ]
                if active_resources:
                    continue
            
            # Fuzzy match
            found = False
            for kb_type in kb.resource_by_type.keys():
                if (resource_type.lower() in kb_type.lower() or 
                    kb_type.lower() in resource_type.lower()):
                    active_resources = [
                        rid for rid in kb.resource_by_type[kb_type]
                        if not kb.resources[rid].is_deleted
                    ]
                    if active_resources:
                        found = True
                        break
            
            if not found:
                return False
        
        return True

    
    # ========================
    # Cache Management
    # ========================
    
    def clear_cache(self):
        """Clear all caches"""
        self._schema_cache.clear()
        self._neighbor_cache.clear()
    
    def get_cache_info(self) -> Dict:
        """Get cache statistics"""
        return {
            "schema_cache_size": len(self._schema_cache),
            "neighbor_cache_size": len(self._neighbor_cache)
        }
