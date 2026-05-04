"""
Main entry point for AutoQuest
Specify which environments to use in the config
"""
import asyncio
import os
import json
from datetime import datetime

from setup_environment import setup_from_config
from AutoQuest.edge_discovery import EdgeDiscovery
from AutoQuest.intelligent_explorer.intelligent_explorer import IntelligentExplorer
from AutoQuest.task_synthesis.trajectory_loader import (
        load_trajectories_from_file,
        load_all_trajectories_from_directory,
        build_memory_like_context
    )

# Import the new trajectory processor
from AutoQuest.task_synthesis.trajectory_processor import CoTTaskSynthesisPipeline
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

explore = False

generate_task = True
async def main():
    """
    Main AutoQuest pipeline
    Configure which environments to use below
    """

    # ========================================
    # CONFIGURATION
    # ========================================
    config = {
        # Model configuration
        "model_name": "gpt-4o",

        # Environments to use (comment/uncomment as needed)
        "environments": [
            # CRMArena
            # {
            #     "name": "crm_arena",
            #     "config": {"org_type": "original"}
            # },

            # # Tau-bench Retail
            # {
            #     "name": "tau_bench_retail",
            # },

            # Tau-bench Airline
            # {
            #     "name": "tau_bench_airline",
            # },

            # TravelPlanner
            # {
            #     "name": "travel_planner",
            # },

            # EnterpriseBench
            # {
            #     "name": "enterprise_bench",
            # },

            # EnterpriseArena (MCP)
            {
                "name": "enterprise_arena",
                "config": {
                    "mcp_config_path": "./mcp_config_http.json"
                }
            },
        ],

        # AutoQuest pipeline configuration
        "pipeline": {
            # Phase 1: Edge Discovery
            "tool_batch_size": 10,
            "edge_batch_size": 100,
            "edge_confidence_threshold": 0.5,

            # Phase 2: Exploration
            "exploration_budget": 2000,
            "interestingness_threshold": 0.3,
            "max_trajectory_length": 100,
            "use_llm_scoring": False,

            # Phase 3: Task Synthesis
            "max_tasks_per_cluster": 10,  # Number of tasks to generate per cluster
            "min_task_quality_score": 0.5,  # Minimum quality threshold

            # Output
            "output_dir": "./arena",
            "save_intermediate": True,
        },
    }

    # ========================================
    # SETUP
    # ========================================
    print(f"\n{Colors.CYAN}{'=' * 70}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}🚀 AUTOQUEST - Universal Task Generation{Colors.END}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.END}")

    # Setup environments
    connector, gpt_caller = await setup_from_config(config)

    # Get all tools
    all_tools = connector.get_all_tools()

    # Create output directory
    output_dir = config["pipeline"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use a fixed filename for the Phase 1 graph so it can be reused across runs
    graph_file = os.path.join(output_dir, "initial_graph_20251118_232354.json")

    # ========================================
    # PHASE 1: EDGE DISCOVERY (cached)
    # ========================================
    if os.path.exists(graph_file):
        print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}Phase 1: Loading Existing Tool Dependency Graph{Colors.END}")
        print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")

        print(f"{Colors.YELLOW}📂 Found existing graph at: {graph_file}{Colors.END}")
        with open(graph_file, "r") as f:
            initial_graph = json.load(f)  # parses JSON into a Python dict[web:8]
    else:
        print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}Phase 1: Tool Dependency Graph (Edge Discovery){Colors.END}")
        print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")

        edge_discovery = EdgeDiscovery(
            tools=all_tools,
            gpt_caller=gpt_caller,
            batch_size=config["pipeline"]["tool_batch_size"],
            edge_batch_size=config["pipeline"]["edge_batch_size"],
            confidence_threshold=config["pipeline"]["edge_confidence_threshold"],
        )

        print(f"{Colors.YELLOW}🔍 Analyzing {len(all_tools)} tools...{Colors.END}")
        initial_graph = await edge_discovery.build_graph()

        # Save graph (without timestamp so later runs can reuse it)
        edge_discovery.save_graph(graph_file)

        print(f"\n{Colors.GREEN}✅ Phase 1 Complete!{Colors.END}")
        print(f"{Colors.GREEN}   • Nodes: {len(initial_graph.get('nodes', {}))}{Colors.END}")
        print(f"{Colors.GREEN}   • Edges: {len(initial_graph.get('edges', {}))}{Colors.END}")
        print(f"{Colors.GREEN}   • Saved to: {graph_file}{Colors.END}\n")

    # ========================================
    # PHASE 2: INTELLIGENT EXPLORATION
    # ========================================
    if explore:
        print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}Phase 2: Intelligent Exploration{Colors.END}")
        print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")
        print(all_tools[0])
        explorer = IntelligentExplorer(
            tools=all_tools,
            initial_graph=initial_graph,
            gpt_caller=gpt_caller,
            exploration_budget=config["pipeline"]["exploration_budget"],
            max_trajectory_length=config["pipeline"]["max_trajectory_length"],
            interestingness_threshold=config["pipeline"]["interestingness_threshold"],
            use_llm_scoring=config["pipeline"]["use_llm_scoring"],
        )

        print(f"{Colors.YELLOW}🧭 Starting exploration...{Colors.END}")
        print(f"{Colors.YELLOW}   • Budget: {config['pipeline']['exploration_budget']} tool calls{Colors.END}\n")

        trajectories, refined_graph, knowledge_base = await explorer.explore()

        # # Save results
        trajectories_file = os.path.join(output_dir, f"trajectories_{run_timestamp}.json")
        refined_graph_file = os.path.join(output_dir, f"refined_graph_{run_timestamp}.json")
        kb_file = os.path.join(output_dir, f"knowledge_base_{run_timestamp}.json")

        # Save trajectories
        with open(trajectories_file, "w") as f:
            json.dump(trajectories, f, indent=2, default=str)

        # Save refined graph
        with open(refined_graph_file, "w") as f:
            json.dump(refined_graph, f, indent=2)

        # Save knowledge base (using snapshot method)
        kb_snapshot = knowledge_base.snapshot()
        kb_snapshot["saved_at"] = run_timestamp
        kb_snapshot["environment"] = knowledge_base.environment_name

        with open(kb_file, "w") as f:
            json.dump(kb_snapshot, f, indent=2, default=str)

        # Print summary
        print(f"\n{Colors.GREEN}✅ Phase 2 Complete!{Colors.END}")
        print(f"{Colors.GREEN}   • Trajectories: {len(trajectories)}{Colors.END}")
        print(f"{Colors.GREEN}   • Refined edges: {len(refined_graph.get('edges', {}))}{Colors.END}")
        print(f"{Colors.GREEN}   • KB Resources: {len(knowledge_base.resources)} (v{knowledge_base.version}){Colors.END}")
        print(f"{Colors.GREEN}   • KB Resource Types: {len(knowledge_base.resource_by_type)}{Colors.END}")
        print(f"\n{Colors.CYAN}📁 Saved Files:{Colors.END}")
        print(f"{Colors.CYAN}   • Trajectories: {trajectories_file}{Colors.END}")
        print(f"{Colors.CYAN}   • Graph: {refined_graph_file}{Colors.END}")
        print(f"{Colors.CYAN}   • Knowledge Base: {kb_file}{Colors.END}")
        print(f"{Colors.CYAN}   • Output dir: {output_dir}{Colors.END}\n")

        # ========================================
    # PHASE 3: TASK SYNTHESIS FROM PERSISTED TRAJECTORIES
    # ========================================
    if generate_task:
        print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}Phase 3: Task Synthesis from Saved Trajectories{Colors.END}")
        print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")

        

        # Option 1: Load just the current run's trajectories
        # loaded_trajectories = load_trajectories_from_file(trajectories_file)
        output_dir="/EnterprisePlatform/TaskGenerationPipeline/arena/tasks"
        trajectories_file = os.path.join(output_dir, f"trajectories_{run_timestamp}.json")

        # ========================================
        # PHASE 3: CHAIN-OF-THOUGHT TASK SYNTHESIS
        # ========================================
        print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}Phase 3: Chain-of-Thought Task Synthesis{Colors.END}")
        print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")

        # Load trajectories from disk
        trajectories_dir = "/EnterprisePlatform/TaskGenerationPipeline/arena/trajectories"
        
        print(f"{Colors.YELLOW}📂 Loading trajectories from: {trajectories_dir}{Colors.END}")
        # loaded_trajectories = load_all_trajectories_from_directory(trajectories_dir)
        # # loaded_trajectories = loaded_trajectories[-4:]
        # print(f"{Colors.GREEN}✅ Loaded {len(loaded_trajectories)} trajectories{Colors.END}\n")

        # if not loaded_trajectories:
        #     print(f"{Colors.RED}❌ No trajectories found! Check the directory path.{Colors.END}")
        #     return

        # # Initialize the CoT Task Synthesis Pipeline
        # print(f"{Colors.YELLOW}🔧 Initializing Chain-of-Thought synthesis pipeline...{Colors.END}")
        # cot_pipeline = CoTTaskSynthesisPipeline(llm_caller=gpt_caller)
        # print(f"{Colors.GREEN}✅ Pipeline initialized{Colors.END}\n")

        # # Process trajectories and generate tasks
        # all_synthesized_tasks = []
        # failed_trajectories = []
        
        # print(f"{Colors.CYAN}{'='*80}{Colors.END}")
        # print(f"{Colors.CYAN}{Colors.BOLD}Processing Trajectories{Colors.END}")
        # print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")

        # max_tasks_per_cluster = config["pipeline"]["max_tasks_per_cluster"]
        
        # for idx, trajectory in enumerate(loaded_trajectories, 1):
        #     traj_id = trajectory.get("trajectory_id", f"traj_{idx}")
        #     num_steps = len(trajectory.get("steps", []))
            
        #     print(f"{Colors.CYAN}[{idx}/{len(loaded_trajectories)}] Processing: {traj_id}{Colors.END}")
        #     print(f"  • Steps: {num_steps}")
        #     tasks_output_file = os.path.join(output_dir, f"tasks.json")
        #     try:
        #         # Process single trajectory through the pipeline
        #         tasks = await cot_pipeline.process_trajectory(
        #             trajectory=trajectory,
        #             tasks_output_dir=tasks_output_file,
        #             max_tasks_per_cluster=max_tasks_per_cluster
        #         )
                
        #         if tasks:
        #             print(f"{Colors.GREEN}  ✓ Generated {len(tasks)} tasks{Colors.END}")
        #             all_synthesized_tasks.extend(tasks)
                    
        #             cot_pipeline.export_tasks(all_synthesized_tasks, tasks_output_file)
        #             # Print sample task for first trajectory
        #             if idx == 1 and tasks:
        #                 sample = tasks[0]
        #                 print(f"\n{Colors.YELLOW}  📋 Sample Task Preview:{Colors.END}")
        #                 print(f"     ID: {sample.task_id}")
        #                 print(f"     Domain: {sample.domain}")
        #                 print(f"     Difficulty: {sample.difficulty}")
        #                 print(f"     Instruction: {sample.instruction[:100]}...")
        #                 print(f"     Steps in CoT: {len(sample.chain_of_thought)}")
        #                 print()
        #         else:
        #             print(f"{Colors.YELLOW}  ⚠ No tasks generated{Colors.END}")
            
        #     except Exception as e:
        #         print(f"{Colors.RED}  ✗ Error: {str(e)}{Colors.END}")
        #         failed_trajectories.append((traj_id, str(e)))
        #         import traceback
        #         if config["pipeline"]["save_intermediate"]:
        #             print(f"{Colors.RED}  Traceback:{Colors.END}")
        #             traceback.print_exc()
            
        #     print()  # Blank line between trajectories
        loaded_trajectories = load_all_trajectories_from_directory(trajectories_dir)
        print(f"{Colors.GREEN}✅ Loaded {len(loaded_trajectories)} trajectories{Colors.END}\n")

        if not loaded_trajectories:
            print(f"{Colors.RED}❌ No trajectories found! Check the directory path.{Colors.END}")
            return

        # Initialize the CoT Task Synthesis Pipeline
        print(f"{Colors.YELLOW}🔧 Initializing Chain-of-Thought synthesis pipeline...{Colors.END}")
        cot_pipeline = CoTTaskSynthesisPipeline(
            llm_caller=gpt_caller,
            enable_postprocessing=True  # Enable Phase 4 quality control
        )
        print(f"{Colors.GREEN}✅ Pipeline initialized{Colors.END}\n")

        # OPTION 1: Process all trajectories together (recommended for best quality)
        # ============================================================================
        print(f"{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}Processing All Trajectories{Colors.END}")
        print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")

        all_synthesized_tasks = []
        failed_trajectories = []

        # File paths
        raw_tasks_file = os.path.join(output_dir, f"tasks_raw_{run_timestamp}.json")
        final_tasks_file = os.path.join(output_dir, f"tasks_final_{run_timestamp}.json")

        print(f"{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}Phase 1-3: Task Generation (Incremental Save){Colors.END}")
        print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")

        max_tasks_per_cluster = config["pipeline"]["max_tasks_per_cluster"]

        for idx, trajectory in enumerate(loaded_trajectories, 1):
            traj_id = trajectory.get("trajectory_id", f"traj_{idx}")
            num_steps = len(trajectory.get("steps", []))
            
            print(f"{Colors.CYAN}[{idx}/{len(loaded_trajectories)}] Processing: {traj_id}{Colors.END}")
            print(f"  • Steps: {num_steps}")
            
            try:
                # Process single trajectory
                tasks = await cot_pipeline.process_trajectory(
                    trajectory=trajectory,
                    max_tasks_per_cluster=max_tasks_per_cluster
                )
                
                if tasks:
                    print(f"{Colors.GREEN}  ✓ Generated {len(tasks)} tasks{Colors.END}")
                    
                    # ✅ INCREMENTAL SAVE: Add each task and save immediately
                    for task in tasks:
                        all_synthesized_tasks.append(task)
                        
                        # Save the growing list after each task
                        cot_pipeline.export_tasks(all_synthesized_tasks, raw_tasks_file)
                    
                    print(f"  💾 Saved incrementally: {len(all_synthesized_tasks)} total raw tasks")
                    
                    # Preview first task
                    if idx == 1 and tasks:
                        sample = tasks[0]
                        print(f"\n{Colors.YELLOW}  📋 Sample Task:{Colors.END}")
                        print(f"     ID: {sample.task_id}")
                        print(f"     Instruction: {sample.instruction[:100]}...")
                        print()
                else:
                    print(f"{Colors.YELLOW}  ⚠ No tasks generated{Colors.END}")
            
            except Exception as e:
                print(f"{Colors.RED}  ✗ Error: {str(e)}{Colors.END}")
                failed_trajectories.append((traj_id, str(e)))
                if config["pipeline"]["save_intermediate"]:
                    import traceback
                    print(f"{Colors.RED}  Traceback:{Colors.END}")
                    traceback.print_exc()
            
            print()


        # ============================================================================
        # PHASE 4: POST-PROCESSING (After All Generation)
        # ============================================================================

        print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}Phase 4: Quality Control (Validation + Dedupe + Diversity){Colors.END}")
        print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")

        if all_synthesized_tasks:
            print(f"{Colors.YELLOW}📊 Raw tasks generated: {len(all_synthesized_tasks)}{Colors.END}")
            
            # Apply post-processing
            processed_tasks = cot_pipeline.post_processor.process(all_synthesized_tasks)
            
            print(f"{Colors.GREEN}✅ High-quality tasks after filtering: {len(processed_tasks)}{Colors.END}\n")
            
            # Save final processed tasks
            cot_pipeline.export_tasks(processed_tasks, final_tasks_file)
            
            print(f"{Colors.GREEN}💾 Saved final tasks to: {final_tasks_file}{Colors.END}\n")
            
            # Show statistics
            print(f"{Colors.YELLOW}📈 Statistics:{Colors.END}")
            print(f"   Raw tasks:       {len(all_synthesized_tasks)}")
            print(f"   After quality:   {len(processed_tasks)}")
            print(f"   Reduction:       {len(all_synthesized_tasks) - len(processed_tasks)} tasks filtered")
            
            # Final sample
            if processed_tasks:
                sample = processed_tasks[0]
                print(f"\n{Colors.YELLOW}📋 Final Task Sample:{Colors.END}")
                print(f"   ID: {sample.task_id}")
                print(f"   Domain: {sample.domain}")
                print(f"   Difficulty: {sample.difficulty}")
                print(f"   Instruction: {sample.instruction[:150]}...")
        else:
            print(f"{Colors.RED}❌ No tasks were generated{Colors.END}")

        # Report failures
        if failed_trajectories:
            print(f"\n{Colors.RED}⚠ Failed trajectories: {len(failed_trajectories)}{Colors.END}")
            for traj_id, error in failed_trajectories[:5]:
                print(f"  - {traj_id}: {error[:80]}")

        print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.GREEN}{Colors.BOLD}✅ Pipeline Complete{Colors.END}")
        print(f"{Colors.CYAN}{'='*80}{Colors.END}")

        # ========================================
        # SAVE RESULTS
        # ========================================
        print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}Saving Results{Colors.END}")
        print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")

        if all_synthesized_tasks:
            # Save tasks to JSON
            tasks_output_file = os.path.join(output_dir, f"cot_tasks_{run_timestamp}.json")
            cot_pipeline.export_tasks(all_synthesized_tasks, tasks_output_file)
            
            # Also save in a more detailed format for analysis
            detailed_output_file = os.path.join(output_dir, f"cot_tasks_detailed_{run_timestamp}.json")
            detailed_output = []
            
            for task in all_synthesized_tasks:
                detailed_output.append({
                   "task_id": task.task_id,
                    "instruction": task.instruction,
                    "prerequisite_context": task.prerequisite_context,
                    "chain_of_thought": [
                        {
                            "step": step.step_number,
                            "rationale": step.rationale,
                            "tool": step.tool_name,
                            "inputs": step.inputs,
                            "expected_output": step.expected_output  # NEW: Ground truth per step
                        }
                        for step in task.chain_of_thought
                    ],
                    "required_tools": task.required_tools,
                    "success_criteria": task.success_criteria,
                    "domain": task.domain,
                    "difficulty": task.difficulty,
                    "ground_truth": task.ground_truth,  # NEW: Task-level ground truth
                    "meta": task.meta
                })
            
            with open(detailed_output_file, 'w') as f:
                json.dump(detailed_output, f, indent=2, default=str)
            
            print(f"{Colors.GREEN}✅ Tasks saved to:{Colors.END}")
            print(f"{Colors.GREEN}   • Summary: {tasks_output_file}{Colors.END}")
            print(f"{Colors.GREEN}   • Detailed: {detailed_output_file}{Colors.END}\n")

    # # ========================================
    # # STATISTICS AND SUMMARY
    # # ========================================
    # print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
    # print(f"{Colors.CYAN}{Colors.BOLD}📊 Final Statistics{Colors.END}")
    # print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")

    # print(f"{Colors.BOLD}Overall:{Colors.END}")
    # print(f"  • Trajectories processed: {len(loaded_trajectories)}")
    # print(f"  • Tasks generated: {len(all_synthesized_tasks)}")
    # print(f"  • Failed trajectories: {len(failed_trajectories)}")
    # print(f"  • Success rate: {(len(loaded_trajectories) - len(failed_trajectories)) / len(loaded_trajectories) * 100:.1f}%\n")

    # if all_synthesized_tasks:
    #     # Domain distribution
    #     domain_counts = {}
    #     for task in all_synthesized_tasks:
    #         domain_counts[task.domain] = domain_counts.get(task.domain, 0) + 1
        
    #     print(f"{Colors.BOLD}Tasks by Domain:{Colors.END}")
    #     for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
    #         print(f"  • {domain}: {count} tasks")
    #     print()

    #     # Difficulty distribution
    #     difficulty_counts = {}
    #     for task in all_synthesized_tasks:
    #         difficulty_counts[task.difficulty] = difficulty_counts.get(task.difficulty, 0) + 1
        
    #     print(f"{Colors.BOLD}Tasks by Difficulty:{Colors.END}")
    #     for difficulty in ["EASY", "MEDIUM", "HARD"]:
    #         count = difficulty_counts.get(difficulty, 0)
    #         print(f"  • {difficulty}: {count} tasks")
    #     print()

    #     # Tool usage statistics
    #     tool_usage = {}
    #     for task in all_synthesized_tasks:
    #         for tool in task.required_tools:
    #             tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
    #     print(f"{Colors.BOLD}Top 10 Most Used Tools:{Colors.END}")
    #     for tool, count in sorted(tool_usage.items(), key=lambda x: -x[1])[:10]:
    #         print(f"  • {tool}: {count} tasks")
    #     print()

    #     # Average steps per task
    #     avg_steps = sum(len(task.chain_of_thought) for task in all_synthesized_tasks) / len(all_synthesized_tasks)
    #     print(f"{Colors.BOLD}Chain-of-Thought Metrics:{Colors.END}")
    #     print(f"  • Average steps per task: {avg_steps:.1f}")
    #     print(f"  • Min steps: {min(len(task.chain_of_thought) for task in all_synthesized_tasks)}")
    #     print(f"  • Max steps: {max(len(task.chain_of_thought) for task in all_synthesized_tasks)}")
    #     print()

    # if failed_trajectories:
    #     print(f"{Colors.RED}{Colors.BOLD}Failed Trajectories:{Colors.END}")
    #     for traj_id, error in failed_trajectories[:5]:  # Show first 5
    #         print(f"{Colors.RED}  • {traj_id}: {error[:100]}{Colors.END}")
    #     if len(failed_trajectories) > 5:
    #         print(f"{Colors.RED}  ... and {len(failed_trajectories) - 5} more{Colors.END}")
    #     print()

    # # ========================================
    # # SAMPLE TASKS DISPLAY
    # # ========================================
    # if all_synthesized_tasks:
    #     print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
    #     print(f"{Colors.CYAN}{Colors.BOLD}📝 Sample Generated Tasks{Colors.END}")
    #     print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")

    #     # Show 3 sample tasks (one from each difficulty if possible)
    #     samples_by_difficulty = {}
    #     for task in all_synthesized_tasks:
    #         if task.difficulty not in samples_by_difficulty:
    #             samples_by_difficulty[task.difficulty] = task

    #     for difficulty in ["EASY", "MEDIUM", "HARD"]:
    #         if difficulty in samples_by_difficulty:
    #             task = samples_by_difficulty[difficulty]
    #             print(f"{Colors.BOLD}{difficulty} Task: {task.task_id}{Colors.END}")
    #             print(f"Domain: {task.domain}")
    #             print(f"\n{Colors.YELLOW}Instruction:{Colors.END}")
    #             print(f"{task.instruction}\n")
    #             print(f"{Colors.YELLOW}Chain of Thought ({len(task.chain_of_thought)} steps):{Colors.END}")
    #             for i, step in enumerate(task.chain_of_thought[:3], 1):  # Show first 3 steps
    #                 print(f"  {i}. [{step.tool_name}] {step.rationale}")
    #             if len(task.chain_of_thought) > 3:
    #                 print(f"  ... and {len(task.chain_of_thought) - 3} more steps")
    #             print(f"\n{Colors.YELLOW}Success Criteria:{Colors.END}")
    #             for criterion in task.success_criteria:
    #                 print(f"  • {criterion}")
    #             print(f"\n{Colors.CYAN}{'─'*80}{Colors.END}\n")

    print(f"{Colors.GREEN}{Colors.BOLD}✨ AutoQuest Pipeline Complete!{Colors.END}\n")


if __name__ == "__main__":
    asyncio.run(main())
    # Build MemoryManager-like façade
    # class OfflineMemoryManager:
    #     def __init__(self, traj_list):
    #         self._by_id = {
    #             t.get("trajectory_id", f"traj_{i}"): t 
    #             for i, t in enumerate(traj_list)
    #         }

    #     def get_context_for_input_generation(self, trajectory_id: str, required_resource_types=None):
    #         traj = self._by_id.get(trajectory_id)
    #         if not traj:
    #             return {
    #                 "recent_executions": [],
    #                 "all_executions": [],
    #                 "successful_executions": [],
    #                 "kb_samples": {},
    #                 "priority_resources": {}
    #             }
    #         return build_memory_like_context(traj)

    #     def list_trajectory_ids(self):
    #         return list(self._by_id.keys())

    # offline_mm = OfflineMemoryManager(loaded_trajectories)

    # # Load tool→domain mapping from Phase 1 graph
    # print(f"{Colors.YELLOW}🔧 Loading tool domains from graph...{Colors.END}")
    
    # tool_to_domain = {}
    # known_tools = set()
    
    # # Load from initial_graph or refined_graph
    # if os.path.exists(graph_file):
    #     print(f"   Loading from: {graph_file}")
    #     with open(graph_file, "r") as f:
    #         graph_data = json.load(f)
        
    #     nodes = graph_data.get("nodes", {})
    #     for tool_name, tool_info in nodes.items():
    #         known_tools.add(tool_name)
    #         domain = tool_info.get("domain", "unknown")
    #         tool_to_domain[tool_name] = domain
    
    # else:
    #     print(f"{Colors.RED}⚠️  No graph file found. Using fallback domain inference.{Colors.END}")
    #     # Fallback: extract from all_tools
    #     if isinstance(all_tools, dict):
    #         for name in all_tools.keys():
    #             known_tools.add(name)
    #             tool_to_domain[name] = "unknown"
    #     elif isinstance(all_tools, list):
    #         for tool in all_tools:
    #             name = None
    #             if hasattr(tool, 'name'):
    #                 name = tool.name
    #             elif hasattr(tool, 'function_name'):
    #                 name = tool.function_name
    #             elif isinstance(tool, dict):
    #                 name = tool.get('name') or tool.get('function_name')
                
    #             if name:
    #                 known_tools.add(name)
    #                 tool_to_domain[name] = "unknown"
    
    # print(f"{Colors.GREEN}✅ Mapped {len(known_tools)} tools across {len(set(tool_to_domain.values()))} domains{Colors.END}")
    
    # # Show domain distribution
    # domain_counts = {}
    # for domain in tool_to_domain.values():
    #     domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    # print(f"{Colors.CYAN}📊 Domain Distribution:{Colors.END}")
    # for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
    #     print(f"   • {domain}: {count} tools")
    # print()

    # # Instantiate synthesis components
    # trajectory_miner = TrajectoryMiner(
    #     tool_to_domain=tool_to_domain,
    #     min_seq_len=1,
    #     max_seq_len=6,
    #     require_success=True,
    # )
    # cross_domain_combiner = CrossDomainCombiner(max_total_steps=8)
    # draft_generator = GroundedTaskDraftGenerator(gpt_caller=gpt_caller)
    # complexity_evaluator = ComplexityEvaluator(
    #     easy_max_steps=1,
    #     medium_max_steps=4,
    # )
    # rule_verifier = RuleVerifier(known_tools=known_tools)
    # model_verifier = ModelVerifier(gpt_caller=gpt_caller)

    # pipeline = TaskSynthesisPipeline(
    #     trajectory_miner=trajectory_miner,
    #     cross_domain_combiner=cross_domain_combiner,
    #     draft_generator=draft_generator,
    #     complexity_evaluator=complexity_evaluator,
    #     rule_verifier=rule_verifier,
    #     model_verifier=model_verifier,
    #     memory_manager=offline_mm,
    # )

    # # Mine subsequences across all trajectories
    # all_mined = []
    # traj_ids = offline_mm.list_trajectory_ids()
    # print(f"{Colors.YELLOW}🧩 Mining subsequences from {len(traj_ids)} trajectories...{Colors.END}")
    
    # for traj_id in traj_ids:
    #     ctx = offline_mm.get_context_for_input_generation(traj_id)
    #     subseqs = await trajectory_miner.mine_from_context(traj_id, ctx)
    #     all_mined.extend(subseqs)
    #     print(f"  • {traj_id}: {len(subseqs)} subsequences")

    # print(f"{Colors.GREEN}✅ Total mined subsequences: {len(all_mined)}{Colors.END}")
    
    # Show subsequence distribution by domain
    # subseq_by_domain = {}
    # for subseq in all_mined:
    #     domain = subseq.domain
    #     subseq_by_domain[domain] = subseq_by_domain.get(domain, 0) + 1
    
    # print(f"{Colors.CYAN}📊 Subsequence Distribution by Domain:{Colors.END}")
    # for domain, count in sorted(subseq_by_domain.items(), key=lambda x: -x[1]):
    #     print(f"   • {domain}: {count}")
    # print()

    # Synthesize tasks
    # synthesized_tasks = []
    # # target_num_tasks = config["pipeline"]["target_num_tasks"]
    # target_num_tasks=5
    # print(f"{Colors.YELLOW}🧪 Synthesizing tasks (target ~{target_num_tasks})...{Colors.END}")
    # for idx, traj_id in enumerate(traj_ids):
    #     print(f"  Processing trajectory {idx+1}/{len(traj_ids)}: {traj_id}")
        
    #     try:
    #         tasks = await pipeline.synthesize_cot_tasks_for_trajectory(
    #             trajectory_id=traj_id,
    #             enable_cross_domain=True,
    #             all_mined_subseqs=all_mined,
    #         )
            
    #         print(f"    → Generated {len(tasks)} tasks")
    #         synthesized_tasks.extend(tasks)
            
    #         if len(synthesized_tasks) >= target_num_tasks:
    #             print(f"{Colors.GREEN}🎯 Reached target of {target_num_tasks} tasks{Colors.END}")
    #             break
        
    #     except Exception as e:
    #         print(f"{Colors.RED}    ✗ Error synthesizing tasks: {e}{Colors.END}")
    #         import traceback
    #         traceback.print_exc()
    #         continue

    # print(f"\n{Colors.GREEN}✅ Task synthesis complete: {len(synthesized_tasks)} tasks generated{Colors.END}")

    # if not synthesized_tasks:
    #     print(f"{Colors.YELLOW}⚠️  No tasks synthesized. Check trajectories and tool mappings.{Colors.END}")
    # else:
    #     # Show distribution by difficulty
    #     difficulty_counts = complexity_evaluator.bucket_stats(synthesized_tasks)
    #     print(f"{Colors.CYAN}📊 Task Distribution by Difficulty:{Colors.END}")
    #     for difficulty, count in difficulty_counts.items():
    #         print(f"   • {difficulty}: {count}")
        
    #     # Show distribution by domain
    #     task_domains = {}
