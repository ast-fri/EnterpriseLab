"""
Batch Task Executor with Automatic Trajectory Saving
=====================================================
✅ MCP connections kept alive during execution
✅ All trajectories saved to single JSON file
✅ System prompt added to conversation state
"""

import asyncio
import json
import os
from contextlib import AsyncExitStack
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from graph_final_localM import build_react_agent_graph, AgentState, get_base_prompt
# from graph_llama import build_react_agent_graph, AgentState, get_base_prompt
# from graph_final_localM_llm_factory import build_react_agent_graph, AgentState, get_base_prompt
# from graph_llama import build_react_agent_graph, AgentState, get_base_prompt
# from graph_final_localM_llm_factory_claude import build_react_agent_graph, AgentState, get_base_prompt
# from graph_final_localM_llm_factory_gemini import build_react_agent_graph, AgentState, get_base_prompt

import sys

DEFAULT_MAX_STEPS = 25
TASKS_FILE = "tasks.json"
MCP_CONFIG_PATH = "./mcp_config_http.json"

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    PURPLE = '\033[95m'
    END = '\033[0m'

def save_all_trajectories_to_file(all_results: list, batch_id: str):
    """Save all trajectories from batch execution to a single JSON file"""
    os.makedirs("trajectories", exist_ok=True)
    filename = f"batch_trajectories_{batch_id}.json"
    filepath = os.path.join("trajectories", filename)
    
    formatted_results = []
    for result in all_results:
        task_data = {
            "task_index": result["task_index"],
            "query": result["query"],
            "description": result.get("description", ""),
            "max_steps": result.get("max_steps", DEFAULT_MAX_STEPS),
            "status": result["status"],
            "final_answer": result.get("final_answer", ""),
            "trajectory_length": result.get("trajectory_length", 0),
            "error": result.get("error", None),
            "trajectory": []
        }
        
        if "trajectory" in result and result["trajectory"]:
            for step in result["trajectory"]:
                if hasattr(step, 'dict'):
                    task_data["trajectory"].append(step.dict())
                elif isinstance(step, dict):
                    task_data["trajectory"].append(step)
        
        formatted_results.append(task_data)
    
    batch_data = {
        "batch_id": batch_id,
        "timestamp": datetime.now().isoformat(),
        "default_max_steps": DEFAULT_MAX_STEPS,
        "total_tasks": len(all_results),
        "successful_tasks": sum(1 for r in all_results if r["status"] == "success"),
        "failed_tasks": sum(1 for r in all_results if r["status"] != "success"),
        "tasks": formatted_results
    }
    
    with open(filepath, "w") as f:
        json.dump(batch_data, f, indent=2)
    
    print(f"\n{Colors.GREEN}📁 All trajectories saved to: {filepath}{Colors.END}")
    return filepath

async def run_task_batch(task: dict, graph, config: dict, task_index: int, total_tasks: int, tools: list, use_persistent_thread: bool = False):
    """
    Execute a single task and collect trajectory
    
    Args:
        tools: List of available tools (needed for system prompt)
        use_persistent_thread: If True, use the provided config (for interactive mode).
                               If False, create a new thread_id (for batch mode).
    """
    query = task.get("query", "")
    description = task.get("description", "No description")
    max_steps = task.get("max_steps", DEFAULT_MAX_STEPS)
    
    print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}📋 TASK {task_index}/{total_tasks}{Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.YELLOW}Query: {Colors.END}{query}")
    print(f"{Colors.YELLOW}Description: {Colors.END}{description}")
    print(f"{Colors.YELLOW}Max Steps: {Colors.END}{max_steps}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    
    # ✅ Choose config based on mode
    if use_persistent_thread:
        # Interactive mode: use the shared config to maintain conversation history
        task_config = config
        
        # ✅ Get current state and append new message
        try:
            current_graph_state = graph.get_state(config)
            if current_graph_state and current_graph_state.values.get("messages"):
                existing_messages = current_graph_state.values["messages"]
                print(f"{Colors.PURPLE}📝 Appending to existing conversation ({len(existing_messages)} messages){Colors.END}")
            else:
                existing_messages = []
        except Exception as e:
            print(f"{Colors.YELLOW}⚠️ Could not retrieve existing state: {e}{Colors.END}")
            existing_messages = []
        
        # ✅ If this is the first message (empty conversation), add system prompt
        if not existing_messages:
            system_prompt_content = get_base_prompt(tools)
            system_prompt_msg = SystemMessage(content=system_prompt_content)
            existing_messages = [system_prompt_msg]
            print(f"{Colors.GREEN}✅ Added system prompt to conversation{Colors.END}")
        
        current_state = AgentState(
            messages=existing_messages + [HumanMessage(content=query)],
            trajectory=[],
            current_step=0,
            max_steps=max_steps,
            task_completed=False,
            current_query=query,
            final_answer="",
            needs_clarification=False,
            clarification_question="",
            enable_clarification=False,
            subtasks_identified=[],
            subtasks_completed=[],
            pending_subtasks=[]
        )
    else:
        # Batch mode: create unique thread_id for isolated execution
        task_config = {
            "configurable": {
                "thread_id": f"batch_task_{task_index}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            }
        }
        
        # ✅ Always start with system prompt in batch mode
        system_prompt_content = get_base_prompt(tools)
        system_prompt_msg = SystemMessage(content=system_prompt_content)
        
        current_state = AgentState(
            messages=[system_prompt_msg, HumanMessage(content=query)],  # ✅ System prompt first
            trajectory=[],
            current_step=0,
            max_steps=max_steps,
            task_completed=False,
            current_query=query,
            final_answer="",
            needs_clarification=False,
            clarification_question="",
            enable_clarification=False,
            subtasks_identified=[],
            subtasks_completed=[],
            pending_subtasks=[]
        )
    
    try:
        final_state = None
        async for event in graph.astream(input=current_state, config=task_config):
            for node_name, node_output in event.items():
                final_state = node_output
        
        if final_state:
            final_answer = final_state.final_answer if hasattr(final_state, 'final_answer') else final_state.get('final_answer', '')
            trajectory = final_state.trajectory if hasattr(final_state, 'trajectory') else final_state.get('trajectory', [])
            
            if final_answer:
                print(f"\n{Colors.GREEN}🎯 ANSWER:{Colors.END}")
                print(f"{Colors.GREEN}{final_answer}{Colors.END}")
            else:
                print(f"{Colors.YELLOW}⚠️ No final answer provided{Colors.END}")
            
            # ✅ Show conversation length in interactive mode
            if use_persistent_thread:
                updated_state = graph.get_state(task_config)
                if updated_state and updated_state.values.get("messages"):
                    msg_count = len(updated_state.values["messages"])
                    print(f"{Colors.PURPLE}📊 Conversation now has {msg_count} messages{Colors.END}")
            
            print(f"{Colors.GREEN}✅ Task {task_index} completed - {len(trajectory)}/{max_steps} steps used{Colors.END}")
            
            return {
                "task_index": task_index,
                "query": query,
                "description": description,
                "max_steps": max_steps,
                "status": "success",
                "final_answer": final_answer,
                "trajectory_length": len(trajectory),
                "trajectory": trajectory
            }
        else:
            print(f"{Colors.RED}❌ FAILED: No output from graph{Colors.END}")
            return {
                "task_index": task_index,
                "query": query,
                "description": description,
                "max_steps": max_steps,
                "status": "failed",
                "error": "No output from graph",
                "trajectory": []
            }
    
    except Exception as e:
        print(f"{Colors.RED}❌ ERROR: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        return {
            "task_index": task_index,
            "query": query,
            "description": description,
            "max_steps": max_steps,
            "status": "error",
            "error": str(e),
            "trajectory": []
        }

async def interactive_mode(graph, config: dict, tools: list, tools_count: int):
    """Interactive terminal mode - ask questions one by one"""
    print(f"\n{Colors.PURPLE}{Colors.BOLD}")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║ 💬 INTERACTIVE MODE                                       ║")
    print("║ Type your queries, press Enter to execute                ║")
    print("║ Commands: 'quit' or 'exit' to stop                       ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")
    
    print(f"{Colors.CYAN}📊 Available tools: {tools_count}{Colors.END}")
    print(f"{Colors.CYAN}🎯 Max steps per query: {DEFAULT_MAX_STEPS}{Colors.END}\n")
    
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = []
    task_number = 1
    
    while True:
        try:
            print(f"{Colors.BOLD}{Colors.CYAN}You [{task_number}]: {Colors.END}", end="")
            user_input = input().strip()
            
            if not user_input:
                print(f"{Colors.YELLOW}⚠️ Empty input, please try again{Colors.END}")
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Colors.YELLOW}👋 Exiting interactive mode...{Colors.END}")
                break
            
            task = {
                "query": user_input,
                "description": f"Interactive query #{task_number}",
                "max_steps": DEFAULT_MAX_STEPS
            }
            
            # ✅ Pass tools and use_persistent_thread=True for interactive mode
            result = await run_task_batch(task, graph, config, task_number, "∞", tools, use_persistent_thread=True)
            results.append(result)
            task_number += 1
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}⚠️ Interrupted by user{Colors.END}")
            break
        except EOFError:
            print(f"\n{Colors.YELLOW}👋 End of input{Colors.END}")
            break
    
    if results:
        print(f"\n{Colors.CYAN}💾 Saving session trajectories...{Colors.END}")
        filepath = save_all_trajectories_to_file(results, session_id)
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        print(f"\n{Colors.GREEN}✅ Queries executed: {len(results)}{Colors.END}")
        print(f"{Colors.GREEN}✅ Successful: {success_count}{Colors.END}")
        print(f"{Colors.GREEN}📁 Saved to: {filepath}{Colors.END}")

async def main():
    """Main batch execution function"""
    # Check for interactive mode
    interactive = False
    if len(sys.argv) > 1 and sys.argv[1] in ['--interactive', '-i']:
        interactive = True
    
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("╔═══════════════════════════════════════════════════════════╗")
    if interactive:
        print("║ 🤖 Interactive Task Executor                              ║")
    else:
        print("║ 🤖 Batch Task Executor - Single File Output              ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")
    
    print(f"{Colors.CYAN}Configuration:{Colors.END}")
    print(f"  Mode: {Colors.BOLD}{'INTERACTIVE' if interactive else 'BATCH'}{Colors.END}")
    print(f"  Default max_steps: {DEFAULT_MAX_STEPS}")
    if not interactive:
        print(f"  Tasks file: {TASKS_FILE}")
    print(f"  MCP config: {MCP_CONFIG_PATH}")
    
    # Load tasks only for batch mode
    tasks = []
    if not interactive:
        if not os.path.exists(TASKS_FILE):
            print(f"\n{Colors.RED}❌ ERROR: {TASKS_FILE} not found{Colors.END}")
            print(f"{Colors.YELLOW}Creating sample tasks.json...{Colors.END}")
            
            sample_tasks = [
                {"query": "What is the weather today?", "description": "Simple weather query"},
                {"query": "Create a task for tomorrow", "description": "Task creation", "max_steps": 20}
            ]
            
            with open(TASKS_FILE, 'w') as f:
                json.dump(sample_tasks, f, indent=2)
            
            print(f"{Colors.GREEN}✅ Created {TASKS_FILE} with {len(sample_tasks)} sample tasks{Colors.END}")
            print(f"{Colors.YELLOW}Edit this file and run again{Colors.END}")
            return
        
        try:
            with open(TASKS_FILE, 'r') as f:
                tasks = json.load(f)
            print(f"\n{Colors.GREEN}✅ Loaded {len(tasks)} tasks from {TASKS_FILE}{Colors.END}")
        except json.JSONDecodeError as e:
            print(f"{Colors.RED}❌ ERROR: Invalid JSON in {TASKS_FILE}: {e}{Colors.END}")
            return
        
        if not tasks:
            print(f"{Colors.YELLOW}⚠️ No tasks found in {TASKS_FILE}{Colors.END}")
            return
    
    try:
        with open(MCP_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        print(f"{Colors.GREEN}✅ Config loaded{Colors.END}")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"{Colors.YELLOW}⚠️ Using LLM-only mode (no MCP tools){Colors.END}")
        config = {"mcpServers": {}}
    
    tools = []
    if config.get("mcpServers"):
        print(f"\n{Colors.CYAN}🔗 Connecting to MCP servers...{Colors.END}")
        client = MultiServerMCPClient(config["mcpServers"])
        
        async with AsyncExitStack() as stack:
            for server_name in config["mcpServers"].keys():
                try:
                    print(f"{Colors.YELLOW}🔄 Connecting to {server_name}...{Colors.END}")
                    session = await asyncio.wait_for(
                        stack.enter_async_context(client.session(server_name)),
                        timeout=15.0
                    )
                    
                    server_tools = await asyncio.wait_for(
                        load_mcp_tools(session),
                        timeout=10.0
                    )
                    for tool in server_tools:
                        original_desc = tool.description
                        tool.name = f"{server_name}_{tool.name}"
                        tool.description = f"Tool for [{server_name.upper()}] related tasks with functionality: {original_desc}"
                        tool.__dict__['_server_name'] = server_name
                    tools.extend(server_tools)
                    print(f"{Colors.GREEN}✓ {server_name}: {len(server_tools)} tools loaded{Colors.END}")
                
                except asyncio.TimeoutError:
                    print(f"{Colors.YELLOW}⚠️ {server_name}: Connection timeout (skipping){Colors.END}")
                except Exception as e:
                    print(f"{Colors.RED}✗ {server_name}: {e}{Colors.END}")
            
            print(f"\n{Colors.GREEN}✅ Building ReAct agent with {len(tools)} tools{Colors.END}")
            graph = build_react_agent_graph(tools, enable_clarification=False)
            
            batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            graph_config = {"configurable": {"thread_id": f"batch_{batch_id}"}}
            
            if interactive:
                # ✅ Pass tools to interactive mode
                await interactive_mode(graph, graph_config, tools, len(tools))
            else:
                # Run batch mode
                print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
                print(f"{Colors.CYAN}{Colors.BOLD}🚀 STARTING BATCH EXECUTION{Colors.END}")
                print(f"{Colors.CYAN}{'='*80}{Colors.END}")
                
                results = []
                for i, task in enumerate(tasks, 1):
                    # ✅ Pass tools to run_task_batch
                    result = await run_task_batch(task, graph, graph_config, i, len(tasks), tools, use_persistent_thread=False)
                    results.append(result)
                
                filepath = save_all_trajectories_to_file(results, batch_id)
                
                print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
                print(f"{Colors.CYAN}{Colors.BOLD}📊 BATCH EXECUTION SUMMARY{Colors.END}")
                print(f"{Colors.CYAN}{'='*80}{Colors.END}")
                
                success_count = sum(1 for r in results if r['status'] == 'success')
                failed_count = len(results) - success_count
                
                print(f"{Colors.GREEN}✅ Successful: {success_count}/{len(results)}{Colors.END}")
                print(f"{Colors.RED}❌ Failed: {failed_count}/{len(results)}{Colors.END}")
                
                for result in results:
                    status_color = Colors.GREEN if result['status'] == 'success' else Colors.RED
                    status_icon = "✅" if result['status'] == 'success' else "❌"
                    print(f"\n{status_color}{status_icon} Task {result['task_index']}: {result['query'][:60]}...{Colors.END}")
                    if result['status'] == 'success':
                        print(f"   Trajectory steps: {result.get('trajectory_length', 0)}/{result.get('max_steps', DEFAULT_MAX_STEPS)}")
                    else:
                        print(f"   Error: {result.get('error', 'Unknown error')}")
                
                print(f"\n{Colors.GREEN}{Colors.BOLD}📁 Single file output: {filepath}{Colors.END}")
    
    else:
        print(f"\n{Colors.YELLOW}⚠️ Using LLM-only mode (no MCP tools){Colors.END}")
        graph = build_react_agent_graph([], enable_clarification=False)
        
        batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        graph_config = {"configurable": {"thread_id": f"batch_{batch_id}"}}
        
        if interactive:
            await interactive_mode(graph, graph_config, [], 0)
        else:
            results = []
            for i, task in enumerate(tasks, 1):
                result = await run_task_batch(task, graph, graph_config, i, len(tasks), [], use_persistent_thread=False)
                results.append(result)
            
            filepath = save_all_trajectories_to_file(results, batch_id)
            print(f"\n{Colors.GREEN}{Colors.BOLD}📁 Single file output: {filepath}{Colors.END}")

if __name__ == "__main__":
    print("🚀 Starting Batch Task Executor...")
    asyncio.run(main())
