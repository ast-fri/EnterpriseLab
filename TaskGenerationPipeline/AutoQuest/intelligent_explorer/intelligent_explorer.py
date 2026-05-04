# intelligent_explorer.py
"""
Main BFS exploration orchestrator for AutoQuest Phase 2
Implements persistent, evolving knowledge base per environment
WITH DUAL-MEMORY ARCHITECTURE (Short-Term + Long-Term)
"""

import os
import time
import uuid
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
from datetime import datetime

# Core components
from AutoQuest.intelligent_explorer.data_models import (
    NodeState,
    NodeExecution, 
    ResourceNode,
    EnvironmentKnowledgeBase
)
from AutoQuest.intelligent_explorer.memory_manager import MemoryManager
from AutoQuest.intelligent_explorer.graph_utils import GraphUtils
from AutoQuest.intelligent_explorer.input_generator import InputGenerator
from AutoQuest.intelligent_explorer.scoring_engine import ScoringEngine
from AutoQuest.intelligent_explorer.resource_tracker import ResourceTracker
from AutoQuest.intelligent_explorer.persistence import PersistenceManager
from AutoQuest.intelligent_explorer.tool_classifier import ToolClassifier, ToolOperation
from utils import normalize_operation

environment = "arena"
class IntelligentExplorer:
    """
    BFS exploration with persistent, evolving knowledge base
    Core innovation: Dual-memory architecture (Short-Term + Long-Term)
    """
    
    def __init__(
        self,
        tools: List[Any],
        initial_graph: Dict,
        gpt_caller: callable,
        exploration_budget: int = 2000,
        max_trajectory_length: int = 10,
        interestingness_threshold: float = 0.3,
        state_save_dir: str = f"./{environment}/exploration_states",
        trajectory_save_dir: str = f"./{environment}/trajectories",
        kb_snapshots_dir: str = f"./{environment}/kb_snapshots",
        use_llm_scoring: bool = False,
        auto_snapshot_interval: int = 100,
        environment_name: str = "default_environment"
    ):
        """
        Args:
            tools: List of tool objects
            initial_graph: Initial dependency graph
            gpt_caller: Async GPT API caller
            exploration_budget: Max total executions
            max_trajectory_length: Max steps per trajectory
            interestingness_threshold: Min score for exploration
            state_save_dir: Directory for node states
            trajectory_save_dir: Directory for trajectories
            kb_snapshots_dir: Directory for KB snapshots
            use_llm_scoring: Whether to use LLM for scoring
            auto_snapshot_interval: KB snapshot frequency
            environment_name: Name of the environment (e.g., 'gitlab', 'crm')
        """
        self.tools = {tool.name: tool for tool in tools}
        self.graph = initial_graph
        self.gpt_caller = gpt_caller
        self.exploration_budget = exploration_budget
        self.max_trajectory_length = max_trajectory_length
        self.interestingness_threshold = interestingness_threshold
        self.use_llm_scoring = use_llm_scoring
        self.environment_name = environment_name
        
        # Initialize persistence manager
        self.persistence = PersistenceManager(
            state_save_dir=state_save_dir,
            trajectory_save_dir=trajectory_save_dir,
            kb_snapshots_dir=kb_snapshots_dir,
            auto_snapshot_interval=auto_snapshot_interval
        )
        
        # Tool classifier
        self.tool_classifier = None
        self.tool_classifications = {}
        
        # Global trajectory tracking
        self.trajectories = []
        self.global_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "trajectories_saved": 0,
            "retries": 0,
            "kb_snapshots_created": 0
        }
        
        # Modular components (initialized after classification)
        self.graph_utils = None
        self.input_generator = None
        self.scoring_engine = None
        self.resource_tracker = None
        self.memory_manager = None  # NEW: Dual-memory manager
        
        # CRITICAL: Persistent KB for this environment
        self.kb: Optional[EnvironmentKnowledgeBase] = None
    
    # ========================
    # Main Exploration Entry
    # ========================
    
    async def explore(self) -> Tuple[List, Dict, EnvironmentKnowledgeBase]:
        """
        Main exploration entry point with persistent KB
        
        Returns:
            Tuple of (trajectories, graph, final_kb)
        """
        print("\n" + "="*80)
        print(f"🔍 AutoQuest Phase 2: Intelligent Exploration")
        print(f"   Environment: {self.environment_name}")
        print("="*80)
        
        # Convert Phase 1 graph nodes to proper tool format
        print("\n🔧 Converting Phase 1 graph to Phase 2 format...")
        self._convert_graph_to_tools()
        
        # Step 1: Load tool classifications
        print("\n📊 Loading tool classifications...")
        self.tool_classifier = ToolClassifier(list(self.tools.values()), self.gpt_caller)
        self.tool_classifications = await self.tool_classifier.classify_all_tools(force_refresh=False)
        
        # Step 2: Load or create persistent KB
        print(f"\n💾 Initializing Knowledge Base for '{self.environment_name}'...")
        self.kb = self.persistence.load_or_create_kb(self.environment_name)
        print(f"   KB Version: {self.kb.version}")
        print(f"   Active Resources: {len([r for r in self.kb.resources.values() if not r.is_deleted])}")
        print(f"   Resource Types: {list(self.kb.resource_by_type.keys())}")
        
        # Step 3: Initialize dual-memory manager
        print(f"\n🧠 Initializing Dual-Memory Architecture...")
        self.memory_manager = MemoryManager(self.kb)
        print(f"   ✓ Short-Term Memory: Ready for trajectory contexts")
        print(f"   ✓ Long-Term Memory: Linked to persistent KB")
        
        # Step 4: Initialize other modular components
        self._initialize_components()
        
        # Step 5: Identify initial nodes
        initial_nodes = self.graph_utils.identify_initial_nodes(kb=self.kb)
        
        print(f"\n📍 Found {len(initial_nodes)} initial nodes for exploration")

        if len(self.kb.resources) > 0:
            print(f"   🔥 Warm Start: KB has {len(self.kb.resources)} resources")
            print(f"   📊 Resource Types: {list(self.kb.resource_by_type.keys())}")
            print(f"   ✅ Can now start from READ/UPDATE/DELETE nodes!")
        else:
            print(f"   ❄️  Cold Start: Starting with CREATE nodes only")
        
        for i, node in enumerate(initial_nodes[:10], 1):
            print(f"   {i}. {node}")
        if len(initial_nodes) > 10:
            print(f"   ... and {len(initial_nodes) - 10} more\n")
        
        # Step 6: Run exploration
        executions_done = 0
        trajectory_count = 0
        
        for idx, initial_node in enumerate(initial_nodes, 1):
            if executions_done >= self.exploration_budget:
                print(f"\n⚠️  Budget exhausted ({self.exploration_budget} executions)")
                break
            
            print(f"\n{'='*80}")
            print(f"🌱 Trajectory {trajectory_count + 1}: Starting from '{initial_node}'")
            print(f"   Budget Used: {executions_done}/{self.exploration_budget}")
            print(f"   KB Version: {self.kb.version}")
            print(f"{'='*80}\n")
            
            trajectory_executions = await self._explore_trajectory_bfs(initial_node)
            executions_done += trajectory_executions
            trajectory_count += 1
            
            # Progress report
            if executions_done % 100 == 0 or executions_done >= self.exploration_budget:
                self._print_progress_report(executions_done)
        
        # Step 7: Final KB snapshot
        print(f"\n💾 Creating final KB snapshot...")
        final_snapshot = self.persistence.save_kb_snapshot(self.kb, snapshot_name=f"{self.environment_name}_final")
        
        # Step 8: Save global statistics
        self._save_final_statistics()
        
        # Step 9: Print summary
        self._print_final_summary()
        
        return self.trajectories, self.graph, self.kb
    
    def _initialize_components(self):
        """Initialize all modular components"""
        print("\n🔧 Initializing modular components...")
        self.graph_utils = GraphUtils(
            tools=self.tools,
            tool_classifications=self.tool_classifications,
            graph=self.graph,
            tool_classifier=self.tool_classifier
        )
        
        self.scoring_engine = ScoringEngine(
            tool_classifications=self.tool_classifications,
            gpt_caller=self.gpt_caller,
            use_llm_scoring=self.use_llm_scoring,
            persistence_manager=self.persistence
        )
        
        # NEW: Pass memory_manager to resource_tracker
        self.resource_tracker = ResourceTracker(
            tool_classifications=self.tool_classifications,
            persistence_manager=self.persistence,
            memory_manager=self.memory_manager,  # ✅ NEW
            interestingness_threshold=self.interestingness_threshold
        )
        
        self.input_generator = InputGenerator(
            tools=self.tools,
            tool_classifications=self.tool_classifications,
            gpt_caller=self.gpt_caller,
            graph_utils=self.graph_utils,
            memory_manager=self.memory_manager
        )
        
        print("   ✓ All components initialized")
        
    def _convert_graph_to_tools(self):
        """Convert Phase 1 graph nodes to tool objects with proper schema format"""
        converted_count = 0
        
        for tool_name, tool in self.tools.items():
            # print(f"{tool.name}: {tool.description}")
            if hasattr(tool, 'args_schema'):
                original_schema = tool.args_schema
                # print(original_schema)
                if isinstance(original_schema, list):
                    node_data = self.graph.get('nodes', {}).get(tool_name, {})
                    
                    if node_data:
                        tool.args_schema = self._convert_phase1_to_json_schema(node_data)
                        converted_count += 1
                    else:
                        properties = {param: {"type": "string"} for param in original_schema}
                        tool.args_schema = {
                            "type": "object",
                            "properties": properties,
                            "required": original_schema
                        }
                        converted_count += 1
                else:
                    tool.args_schema = {
                        "type": "object",
                        "properties": original_schema,
                        "required": original_schema.get("required", "")
                    }
                    converted_count += 1
        
        print(f"   ✓ Converted {converted_count} tools from Phase 1 format to JSON Schema")

    # ========================
    # BFS Trajectory Exploration
    # ========================
    
    async def _explore_trajectory_bfs(self, initial_node: str) -> int:
        """
        Perform BFS exploration for a single trajectory
        Uses DUAL-MEMORY: Short-Term (trajectory) + Long-Term (persistent KB)
        
        Args:
            initial_node: Starting tool name
            
        Returns:
            Number of executions performed
        """
        trajectory_id = f"traj_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
        trajectory = {
            "trajectory_id": trajectory_id,
            "initial_node": initial_node,
            "environment": self.environment_name,
            "kb_version_start": self.kb.version,
            "steps": [],
            "cumulative_score": 0.0,
            "status": "in_progress",
            "timestamp": datetime.now().isoformat(),
        }
        
        # ✅ NEW: Start short-term memory for this trajectory
        self.memory_manager.start_trajectory(trajectory_id)
        
        # BFS queue: (current_node, depth, parent_node_name)
        queue = deque([(initial_node, 0, None)])
        visited = set()
        executions_count = 0
        
        try:  # ✅ Wrap in try-finally to ensure cleanup
            while queue and len(trajectory["steps"]) < self.max_trajectory_length:
                current_node, depth, parent_node = queue.popleft()
                
                if current_node not in self.tools:
                    continue
                
                if current_node in visited:
                    continue
                
                visited.add(current_node)
                
                print(f"  📍 Depth {depth}: Executing '{current_node}'")
                if parent_node:
                    print(f"      ⬆️  Parent: {parent_node}")
                
                # Show current KB state
                print(f"      📚 KB: v{self.kb.version}, {len(self.kb.resources)} resources, {len(self.kb.resource_by_type)} types")
                # print(f"{self.tools[0]}")
                # Execute tool with retry
                execution = await self._execute_tool_with_retry(
                    current_node,
                    parent_node,
                    depth,
                    trajectory_id=trajectory_id  # ✅ ADD THIS LINE
                )
                # print("Execution Output: ", execution)
                executions_count += 1
                self.global_stats["total_executions"] += 1
                
                if not execution or not execution.success:
                    self.global_stats["failed_executions"] += 1
                    print(f"    ✗ Execution failed")
                    continue
                
                self.global_stats["successful_executions"] += 1
                print(f"    ✓ Execution successful ({execution.execution_time:.2f}s)")
                
                # DEBUG: Show actual tool output
                print(f"      🔍 Tool Output Debug:")
                print(f"          Type: {type(execution.tool_output)}")
                print(f"          Is Dict: {isinstance(execution.tool_output, dict)}")

                if isinstance(execution.tool_output, dict):
                    print(f"          Keys: {list(execution.tool_output.keys())}")
                    print(f"          Sample: {str(execution.tool_output)[:300]}")
                else:
                    print(f"          Value: {str(execution.tool_output)[:200]}")

                # ✅ Track resources in BOTH short-term and long-term memory
                print(f"      📊 Tracking resources...")
                kb_before = len(self.kb.resources)

                self.resource_tracker.track_resources(
                    execution,
                    self.kb,
                    trajectory_id
                )

                kb_after = len(self.kb.resources)
                print(f"      📊 KB: {kb_before} → {kb_after} resources")

                if kb_after == kb_before:
                    print(f"      ⚠️  NO RESOURCE ADDED! Investigating...")
                    classification = self.tool_classifications.get(current_node, {})
                    print(f"          Classification: {classification}")
                    print(f"          Operation: {normalize_operation(classification.get('operation'))}")
                    print(f"          Produces: {classification.get('produces_resource')}")
                
                # Score interestingness
                execution.interestingness_score = await self.scoring_engine.score_node_interestingness(
                    execution,
                    current_node,
                    self.kb
                )
                
                print(f"    📊 Score: {execution.interestingness_score:.2f}")
                
                # Save node state
                await self.resource_tracker.save_node_state(execution, current_node, self.kb)
                
                # Auto-snapshot KB
                self.persistence.auto_snapshot_kb(self.kb)
                
                # Add step to trajectory
                step = {
                    "step_number": len(trajectory["steps"]) + 1,
                    "depth": depth,
                    "tool_name": current_node,
                    "parent_node": parent_node,
                    "inputs": execution.tool_input,
                    "outputs": self.persistence.serialize_output(execution.tool_output),
                    "success": execution.success,
                    "error": execution.error,
                    "execution_time": execution.execution_time,
                    "interestingness_score": execution.interestingness_score,
                    "kb_version": self.kb.version
                }
                trajectory["steps"].append(step)
                trajectory["cumulative_score"] += execution.interestingness_score
                
                # Save trajectory incrementally
                await self.persistence.save_trajectory_incremental(trajectory)
                
                # Get KB-aware neighbors
                neighbors = self.graph_utils.get_neighbors(
                    node_name=current_node,
                    kb=self.kb,
                    filter_by_kb=True,
                    recent_execution=execution.tool_output
                )
                
                if not neighbors:
                    print(f"    ℹ️  No executable neighbors (KB constraints)")
                    continue
                
                print(f"    → Found {len(neighbors)} KB-satisfying neighbors")
                
                # Score and filter neighbors
                neighbor_scores = []
                for neighbor in neighbors:
                    score = await self.scoring_engine.score_next_node_potential(
                        neighbor,
                        execution,
                        self.kb
                    )
                    neighbor_scores.append((neighbor, score))
                
                # Add high-scoring neighbors to queue
                added = 0
                for neighbor, score in sorted(neighbor_scores, key=lambda x: x[1], reverse=True):
                    if score >= self.interestingness_threshold:
                        print(f"      • {neighbor}: {score:.2f} ✓")
                        queue.append((neighbor, depth + 1, current_node))
                        added += 1
                    else:
                        print(f"      • {neighbor}: {score:.2f} ✗")
                
                if added > 0:
                    print(f"    ➕ Added {added} neighbors (queue: {len(queue)})")
            
            # Finalize trajectory
            trajectory["status"] = "completed" if len(trajectory["steps"]) < self.max_trajectory_length else "max_length_reached"
            trajectory["kb_version_end"] = self.kb.version
            trajectory["kb_resources_created"] = len(self.kb.resources)
            trajectory["total_steps"] = len(trajectory["steps"])
            trajectory["avg_interestingness"] = trajectory["cumulative_score"] / max(len(trajectory["steps"]), 1)
            trajectory["execution_path"] = [step["tool_name"] for step in trajectory["steps"]]
            
            await self.persistence.save_trajectory_final(trajectory)
            
            self.trajectories.append(trajectory)
            self.global_stats["trajectories_saved"] += 1
            
            print(f"\n  ✅ Trajectory {trajectory_id[:12]} completed:")
            print(f"     • Status: {trajectory['status']}")
            print(f"     • Steps: {trajectory['total_steps']}")
            print(f"     • KB Evolution: v{trajectory['kb_version_start']} → v{trajectory['kb_version_end']}")
            print(f"     • Avg Score: {trajectory['avg_interestingness']:.2f}")
            
            return executions_count
            
        finally:
            # ✅ NEW: Always clean up short-term memory
            self.memory_manager.end_trajectory(trajectory_id, commit_to_kb=True)
    
    # ========================
    # Tool Execution with Retry
    # ========================
    
    async def _execute_tool_with_retry(
        self,
        node_name: str,
        parent_node: Optional[str],
        depth: int,
        max_retries: int = 1,
        trajectory_id: Optional[str] = None
    ) -> Optional[NodeExecution]:
        """
        Execute tool with retry using persistent KB
        
        Args:
            node_name: Tool name
            parent_node: Parent tool name
            depth: Current depth
            max_retries: Max retry attempts
            
        Returns:
            NodeExecution or None
        """
        feedback = ""
        tool_input = {}
        
        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"    🔄 Retry {attempt}/{max_retries}")
                self.global_stats["retries"] += 1
            
            start_time = time.time()
            
            try:
                tool = self.tools.get(node_name)
                
                if not tool:
                    return NodeExecution(
                        node_name=node_name,
                        tool_input={},
                        tool_output=None,
                        execution_time=time.time() - start_time,
                        success=False,
                        error=f"Tool '{node_name}' not found in tools registry"
                    )
                
                # Generate inputs using PERSISTENT KB
                tool_input = await self.input_generator.generate_synthetic_input(
                    node_name=node_name,
                    kb=self.kb,
                    feedback=feedback,
                    parent_execution_state=None,
                    depth=depth,
                    retry_attempt=attempt,
                    trajectory_id=trajectory_id  # ✅ ADD THIS LINE
                )
                
                ## Correct it as some tools need for require inputs
                # if tool_input is None:
                #     error_msg = "Failed to generate valid inputs"
                #     if attempt < max_retries:
                #         feedback = f"{feedback}\n{error_msg}"
                #         print(f"      ⚠️  {error_msg}, retrying...")
                #         continue
                    
                #     return NodeExecution(
                #         node_name=node_name,
                #         tool_input={},
                #         tool_output=None,
                #         execution_time=time.time() - start_time,
                #         success=False,
                #         error=error_msg
                #     )
                
                # Execute tool
                result = await self._execute_tool(tool, tool_input)
                
                # Parse JSON strings
                if isinstance(result, str):
                    try:
                        import json
                        parsed = json.loads(result)
                        print(f"      🔧 Parsed JSON string to {type(parsed)}")
                        result = parsed
                    except (json.JSONDecodeError, ValueError):
                        print("Unable to parse tool output as JSON, keeping as string")
                        pass
                
                execution_time = time.time() - start_time
                
                # Check if result indicates an error
                if isinstance(result, dict) and result.get("error") and not result.get("success", True):
                    error_msg = result.get("error", "Unknown error")
                    
                    if attempt < max_retries:
                        feedback = f"{feedback}\nTool returned error: {error_msg}"
                        print(f"      ⚠️  Tool error: {error_msg[:100]}")
                        continue
                    
                    return NodeExecution(
                        node_name=node_name,
                        tool_input=tool_input,
                        tool_output=result,
                        execution_time=execution_time,
                        success=False,
                        error=error_msg
                    )
                
                # Success!
                return NodeExecution(
                    node_name=node_name,
                    tool_input=tool_input,
                    tool_output=result,
                    execution_time=execution_time,
                    success=True,
                    error=None
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = str(e)
                error_context = f"Tool: {node_name}, Attempt: {attempt + 1}/{max_retries + 1}"
                full_error = f"{error_context}\n{error_msg}"
                
                feedback = f"{feedback}\nError: {error_msg}"
                
                if attempt < max_retries:
                    print(f"      ⚠️  Error: {error_msg[:150]}")
                    print(f"      🔄 Will retry with feedback...")
                    continue
                
                print(f"      ❌ Final failure after {max_retries + 1} attempts")
                print(f"      Error: {error_msg[:200]}")
                
                return NodeExecution(
                    node_name=node_name,
                    tool_input=tool_input,
                    tool_output=None,
                    execution_time=execution_time,
                    success=False,
                    error=full_error
                )
        
        return None

    async def _execute_tool(self, tool: Any, tool_input: Dict) -> Any:
        """
        Execute tool with multiple execution strategies
        Handles ToolWrapper, LangChain tools, and various callable types
        
        Args:
            tool: Tool object to execute
            tool_input: Input parameters as dictionary (can be None or empty)
            
        Returns:
            Tool execution result
            
        Raises:
            TypeError: If tool is not executable through any known method
        """
        tool_name = getattr(tool, 'name', str(tool))
        
        # Normalize tool_input: handle None, empty dict, or actual parameters
        tool_input = tool_input or {}
        
        # Strategy 1: ToolWrapper with ainvoke (your standard wrapper)
        if hasattr(tool, 'ainvoke'):
            try:
                return await tool.ainvoke(**tool_input)
            except TypeError as e:
                # Might need positional args instead of kwargs
                if "missing" in str(e).lower() or "unexpected" in str(e).lower():
                    return await tool.ainvoke(tool_input)
                raise
        
        # Strategy 2: Async run method
        if hasattr(tool, 'run'):
            if asyncio.iscoroutinefunction(tool.run):
                try:
                    return await tool.run(**tool_input)
                except TypeError:
                    return await tool.run(tool_input)
            else:
                # Sync run, execute in thread pool
                loop = asyncio.get_event_loop()
                try:
                    return await loop.run_in_executor(None, lambda: tool.run(**tool_input))
                except TypeError:
                    return await loop.run_in_executor(None, lambda: tool.run(tool_input))
        
        # Strategy 3: Async execute method
        if hasattr(tool, 'execute'):
            if asyncio.iscoroutinefunction(tool.execute):
                try:
                    return await tool.execute(**tool_input)
                except TypeError:
                    return await tool.execute(tool_input)
            else:
                loop = asyncio.get_event_loop()
                try:
                    return await loop.run_in_executor(None, lambda: tool.execute(**tool_input))
                except TypeError:
                    return await loop.run_in_executor(None, lambda: tool.execute(tool_input))
        
        # Strategy 4: Coroutine attribute (LangChain style)
        if hasattr(tool, 'coroutine'):
            print("Coroutine attribute found")
            if asyncio.iscoroutinefunction(tool.coroutine):
                return await tool.coroutine(**tool_input)
            else:
                return tool.coroutine(**tool_input)
        
        # Strategy 5: Synchronous invoke (backward compatibility)
        if hasattr(tool, 'invoke'):
            loop = asyncio.get_event_loop()
            try:
                return await loop.run_in_executor(None, lambda: tool.invoke(**tool_input))
            except TypeError:
                return await loop.run_in_executor(None, lambda: tool.invoke(tool_input))
        
        # Strategy 6: MCP-style call method
        if hasattr(tool, 'call'):
            if asyncio.iscoroutinefunction(tool.call):
                return await tool.call(**tool_input)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: tool.call(**tool_input))
        
        # Strategy 7: Direct callable (function or __call__)
        if callable(tool):
            try:
                if asyncio.iscoroutinefunction(tool):
                    return await tool(**tool_input)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, lambda: tool(**tool_input))
            except TypeError as e:
                # Tool might not be directly callable despite being "callable"
                pass
        
        # Strategy 8: Check __call__ explicitly
        if hasattr(tool, '__call__') and callable(tool.__call__):
            try:
                if asyncio.iscoroutinefunction(tool.__call__):
                    return await tool(**tool_input)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, lambda: tool(**tool_input))
            except TypeError:
                pass
        
        # If all strategies failed, provide detailed error
        available_methods = [
            attr for attr in dir(tool)
            if not attr.startswith('_') and callable(getattr(tool, attr, None))
        ]
        
        raise TypeError(
            f"❌ Tool '{tool_name}' is not executable through any known method.\n"
            f"   Tool type: {type(tool).__name__}\n"
            f"   Tool class: {tool.__class__.__name__}\n"
            f"   Available methods: {available_methods[:15]}\n"
            f"   \n"
            f"   Expected one of:\n"
            f"   - ainvoke(**kwargs) or ainvoke(dict)\n"
            f"   - run(**kwargs) or run(dict)\n"
            f"   - execute(**kwargs) or execute(dict)\n"
            f"   - invoke(**kwargs) or invoke(dict)\n"
            f"   - call(**kwargs)\n"
            f"   - __call__(**kwargs)\n"
            f"   \n"
            f"   Please ensure your ToolWrapper implements 'ainvoke()' method."
        )


    
    # ========================
    # Progress & Statistics
    # ========================
    
    def _print_progress_report(self, executions_done: int):
        """Print progress report"""
        print(f"\n{'='*80}")
        print(f"📊 Progress Report: {executions_done}/{self.exploration_budget}")
        print(f"{'='*80}")
        print(f"   ✓ Successful: {self.global_stats['successful_executions']}")
        print(f"   ✗ Failed: {self.global_stats['failed_executions']}")
        print(f"   🔄 Retries: {self.global_stats['retries']}")
        print(f"   📚 Trajectories: {self.global_stats['trajectories_saved']}")
        print(f"   💾 KB Snapshots: {self.persistence.execution_count // self.persistence.auto_snapshot_interval}")
        
        if self.kb:
            print(f"\n   🗄️  Knowledge Base:")
            print(f"      Version: {self.kb.version}")
            print(f"      Active Resources: {len([r for r in self.kb.resources.values() if not r.is_deleted])}")
            print(f"      Resource Types: {len(self.kb.resource_by_type)}")
            print(f"      Total Executions: {len(self.kb.execution_chain)}")
        
        # Graph statistics
        graph_stats = self.graph_utils.get_graph_statistics(self.kb)
        print(f"\n   📈 Exploration Coverage:")
        print(f"      Executable Tools: {graph_stats.get('executable_tools', 0)}/{graph_stats['total_tools']}")
        print(f"      Coverage: {graph_stats.get('execution_coverage', 0):.1f}%")
        
        print(f"{'='*80}\n")
    
    def _save_final_statistics(self):
        """Save final statistics"""
        print("\n💾 Saving final statistics...")
        
        stats = {
            "exploration": self.global_stats,
            "kb": {
                "environment": self.kb.environment_name,
                "final_version": self.kb.version,
                "total_resources": len(self.kb.resources),
                "active_resources": len([r for r in self.kb.resources.values() if not r.is_deleted]),
                "resource_types": list(self.kb.resource_by_type.keys()),
                "total_executions": len(self.kb.execution_chain),
                "stats": self.kb.stats
            },
            "scoring": self.scoring_engine.get_execution_stats(),
            "resource_tracking": self.resource_tracker.get_stats(),
            "input_generation": self.input_generator.get_failure_stats(),
            "graph": self.graph_utils.get_graph_statistics(self.kb),
            "timestamp": datetime.now().isoformat()
        }
        
        stats_file = os.path.join(self.persistence.state_save_dir, f"{self.environment_name}_final_stats.json")
        
        import json
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            print(f"   ✓ Saved to {stats_file}")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
    
    def _convert_phase1_to_json_schema(self, node_data: Dict) -> Dict:
        """
        Convert Phase 1 node format to JSON Schema format
        
        Args:
            node_data: Node dict from Phase 1 graph
            
        Returns:
            Proper JSON Schema dict
        """
        args_schema = node_data.get("args_schema", [])
        required_inputs = node_data.get("required_inputs", {})
        field_types = required_inputs.get("field_types", {})
        required_fields = required_inputs.get("input_fields", [])
        
        properties = {}
        
        if isinstance(args_schema, list):
            for param_name in args_schema:
                param_type = field_types.get(param_name, "string")
                properties[param_name] = {
                    "type": param_type,
                    "description": f"Parameter {param_name}"
                }
        elif isinstance(args_schema, dict):
            properties = args_schema.get("properties", args_schema)
        
        json_schema = {
            "type": "object",
            "properties": properties,
            "required": required_fields if required_fields else list(properties.keys())
        }
        
        return json_schema

    def _print_final_summary(self):
        """Print final summary"""
        print(f"\n{'='*80}")
        print(f"✅ AUTOQUEST PHASE 2 COMPLETE")
        print(f"{'='*80}\n")
        
        print(f"📊 Final Statistics:")
        print(f"   Total Executions: {self.global_stats['total_executions']}")
        print(f"   Successful: {self.global_stats['successful_executions']}")
        print(f"   Failed: {self.global_stats['failed_executions']}")
        print(f"   Success Rate: {self.global_stats['successful_executions'] / max(self.global_stats['total_executions'], 1) * 100:.1f}%")
        
        print(f"\n📚 Knowledge Base ('{self.kb.environment_name}'):")
        print(f"   Final Version: {self.kb.version}")
        print(f"   Total Resources: {len(self.kb.resources)}")
        print(f"   Active Resources: {len([r for r in self.kb.resources.values() if not r.is_deleted])}")
        print(f"   Resource Types: {list(self.kb.resource_by_type.keys())}")
        print(f"   Total KB Executions: {len(self.kb.execution_chain)}")
        
        print(f"\n📁 Output Directories:")
        print(f"   KB Snapshots: {self.persistence.kb_snapshots_dir}")
        print(f"   Trajectories: {self.persistence.trajectory_save_dir}")
        print(f"   States: {self.persistence.state_save_dir}")
        
        print(f"\n{'='*80}\n")
