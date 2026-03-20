"""
ReAct Graph with LOCAL MODEL Support + vLLM SERVER Support + ROBUST Tool Parsing
✅ NEW: Support for vLLM server via OpenAI API
✅ Maintains all existing functionality
"""
import json
import asyncio
import ast
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_openai import OpenAI as LangChainOpenAI
from langchain_openai import AzureChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langgraph.graph import StateGraph, add_messages, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langchain.tools import BaseTool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from pydantic import BaseModel, ValidationError
# from langchain_core.messages import RemoveMessage

from typing import List, Annotated, Literal, Dict, Any, Optional
import os
import json
import re
import uuid
from datetime import datetime
from dotenv import load_dotenv
import torch
from openai import OpenAI

load_dotenv()

# Configuration
TOOL_EXAMPLES_PATH = os.getenv("TOOL_EXAMPLES_PATH", "tool_examples.json")
# NEW: vLLM Server Configuration
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "your_vllm_api_key_here")

# Modes: "local" or "vllm"
MODEL_MODE = os.getenv("MODEL_MODE", "vllm")  # gpt/vllm/local
DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() == "true"

class TrajectoryStep(BaseModel):
    step_number: int
    step_type: Literal["thought", "action", "observation", "clarification"]
    content: str
    timestamp: str
    tool_used: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None

class AgentState(BaseModel):
    messages: Annotated[List, add_messages]
    trajectory: List[TrajectoryStep] = []
    current_step: int = 0
    max_steps: int = 25
    task_completed: bool = False
    current_query: str = ""
    final_answer: str = ""
    needs_clarification: bool = False
    clarification_question: str = ""
    enable_clarification: bool = True
    subtasks_identified: List[str] = []
    subtasks_completed: List[str] = []
    pending_subtasks: List[str] = []

def save_trajectory_to_file(trajectory: List[TrajectoryStep], query: str, final_answer: str = ""):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trajectory_{timestamp}.json"
    trajectory_data = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "final_answer": final_answer,
        "total_steps": len(trajectory),
        "trajectory": [step.dict() for step in trajectory]
    }
    os.makedirs("trajectories", exist_ok=True)
    filepath = os.path.join("trajectories", filename)
    with open(filepath, "w") as f:
        json.dump(trajectory_data, f, indent=2)
    print(f"📁 Trajectory saved to: {filepath}")
    return filepath

def load_local_model(model_path: str, device: str = "auto"):
    """Load model locally via HuggingFace"""
    print(f"🔄 Loading local model from: {model_path}")
    num_gpus = torch.cuda.device_count()
    print(f"📊 Available GPUs: {num_gpus}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        top_p=0.95,
        return_full_text=False
    )
    
    hf_pipeline = HuggingFacePipeline(pipeline=pipe, 
                                      model_kwargs={"chat_template_kwargs": {"enable_thinking": False}})
    chat_model = ChatHuggingFace(llm=hf_pipeline)
    print(f"✅ Local model loaded successfully")
    return chat_model

def load_vllm_model(base_url: str, api_key: str):
    """✅ NEW: Load model via vLLM OpenAI-compatible server"""
    print(f"🔄 Connecting to vLLM server at: {base_url}")
    
    # Test connection and get model name
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        models = client.models.list()
        # print("Available models from vLLM server:", [m.id for m in models.data])
        # print("Model details:", models)
        # exit()
        if not models.data:
            raise ValueError("No models available from vLLM server")
        
        model_name = models.data[0].id
        print(f"✅ Connected to vLLM server")
        print(f"📋 Available model: {model_name}")
        
        # Create LangChain-compatible wrapper
        from langchain_openai import ChatOpenAI
        
        chat_model = ChatOpenAI(
            model=model_name,
            openai_api_base=base_url,
            openai_api_key=api_key,
            temperature=0,
            max_tokens=1024,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )
        
        print(f"✅ vLLM model wrapper created")
        return chat_model, model_name
        
    except Exception as e:
        print(f"❌ Failed to connect to vLLM server: {e}")
        print(f"   Make sure vLLM is running on {base_url}")
        raise
def load_gpt_model():
    api_key = os.getenv("AZURE_CHAT_API_KEY")
    api_base = os.getenv("AZURE_CHAT_ENDPOINT")
    # ✅ FIX 1: Add timeout and max_tokens to prevent runaway context
    llm = AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=api_base,
        azure_deployment="gpt-4o",
        api_version="2024-02-01",
        temperature=0.1,
        timeout=60,           # ← Prevent hanging
        request_timeout=60,   # ← HTTP timeout
        max_retries=2,        # ← Don't retry too much
        max_tokens=2048       # ← CRITICAL: Limit response size
    )
    llm = llm.bind(
            response_format={"type": "json_object"}
        )
    # llm_with_tools = llm.bind_tools(tools) if tools else llm
    return llm
# ... [Keep all existing helper functions: get_tool_schema_safe, get_tool_schema_description, 
#      extract_json_from_text, validate_tool_input_with_correction, 
#      parse_with_multiple_strategies, filter_messages_for_local_model] ...

def get_tool_schema_description(tool: BaseTool) -> str:
    """Enhanced tool description with few-shot examples for ALL 83 tools"""
    # if(tool.name=="create_issue"):
    # print(tool)
    with open(TOOL_EXAMPLES_PATH, 'r') as f:
        TOOL_EXAMPLES = json.load(f)
    # TOOL_EXAMPLES={}
    # Complete examples mapping for all 83 tools
     
    
    # Main formatting logic
    tool_desc = f"\n{'='*60}\n"
    tool_desc += f"🔧 Tool: {tool.name}\n"
    tool_desc += f"{'='*60}\n"
    tool_desc += f"Description: {tool.description}\n"
    
    # Get schema information
    schema = get_tool_schema_safe(tool)
    
    if schema and 'properties' in schema:
        required = schema.get('required', [])
        
        # Add parameter details
        tool_desc += f"\n📋 Parameters:\n"
        tool_desc += f"{'-'*60}\n"
        
        for param_name, param_info in schema['properties'].items():
            param_type = param_info.get('type', 'string')
            param_desc = param_info.get('description', 'No description')
            default_val = param_info.get('default', None)
            enum_vals = param_info.get('enum', None)
            
            req_marker = "✓ REQUIRED" if param_name in required else "○ optional"
            
            tool_desc += f"\n  • {param_name} ({param_type}) [{req_marker}]\n"
            tool_desc += f"    └─ {param_desc}\n"
            
            if default_val is not None:
                tool_desc += f"    └─ Default: {default_val}\n"
            
            if enum_vals:
                tool_desc += f"    └─ Allowed values: {', '.join(map(str, enum_vals))}\n"
        
        # Add few-shot examples if available
        
        if tool.name in TOOL_EXAMPLES:
            
            example_data = TOOL_EXAMPLES[tool.name]
            
            tool_desc += f"\n\n💡 Usage Notes:\n"
            tool_desc += f"{'-'*60}\n"
            tool_desc += f"{example_data['notes']}\n"
            
            tool_desc += f"\n\n📝 Example Usage:\n"
            tool_desc += f"{'-'*60}\n"
            
            for idx, example in enumerate(example_data['examples'], 1):
                tool_desc += f"\nExample {idx}:\n"
                example_json = {
                    "action": tool.name,
                    "action_input": example
                }
                tool_desc += f"{json.dumps(example_json, indent=2)}\n"
        else:
            # Generic example with required parameters only
            required_params = {k: f"<{schema['properties'][k].get('type', 'string')}>" 
                             for k in required}
            if required_params:
                tool_desc += f"\n\n📝 Example Usage:\n"
                tool_desc += f"{'-'*60}\n"
                example_json = {
                    "action": tool.name,
                    "action_input": required_params
                }
                tool_desc += f"{json.dumps(example_json, indent=2)}\n"
    
    tool_desc += f"\n{'='*60}\n"
    
    return tool_desc


def get_tool_schema_safe(tool: BaseTool) -> dict:
    """Safely extract tool schema"""
    try:
        if hasattr(tool, 'args_schema') and tool.args_schema:
            if isinstance(tool.args_schema, dict):
                return tool.args_schema
            elif hasattr(tool.args_schema, 'schema'):
                return tool.args_schema.schema()
            elif hasattr(tool.args_schema, 'model_json_schema'):
                return tool.args_schema.model_json_schema()
        return {}
    except Exception as e:
        print(f"Warning: Could not extract schema for {tool.name}: {e}")
        return {}




def validate_tool_input_with_correction(tool_name: str, tool_args: Dict[str, Any], tools: List[BaseTool]) -> Optional[Dict[str, Any]]:
    """Validate with parameter name auto-correction"""
    tool = next((t for t in tools if t.name == tool_name), None)
    print("Tool name", tool_name)
    if not tool:
        print(f"❌ Tool {tool_name} not found")
        return {}, False
    
    # If no schema validation needed, return as-is
    if not hasattr(tool, 'args_schema') or tool.args_schema is None:
        return tool_args, True
    
    # Handle correct description fetch
    
    # Handle dict-based schemas
    if isinstance(tool.args_schema, dict):
        print("🔍 Validating against dict schema with auto-correction")
        schema = tool.args_schema
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        # Parameter aliases for common variations
        param_aliases = {
            'project_id': ['repository_id', 'repo_id'],
            'repository_name': ['repo_name', 'project_name', 'repo', 'name'],
            'name': ['repository_name', 'repo_name', 'project_name'],
            'branch': ['branch_name', 'branch_id', 'ref', 'branch_ref'],
            'message': ['commit_message', 'msg', 'description', 'commit_msg'],
            'content': ['file_content', 'body', 'data', 'text'],
            'path': ['file_path', 'filepath', 'file', 'filename'],
        }
        
        # Build reverse lookup
        alias_to_param = {}
        for param, aliases in param_aliases.items():
            for alias in aliases:
                alias_to_param[alias] = param
        
        # Correct the arguments
        corrected_args = {}
        matched= True
        for provided_key, value in tool_args.items():
            matched = False
            
            # Strategy 1: Exact match
            if provided_key in properties:
                corrected_args[provided_key] = value
                print(f"  ✅ Exact match: {provided_key}")
                matched = True
            
            # Strategy 2: Check if provided_key is an alias
            elif provided_key in alias_to_param and alias_to_param[provided_key] in properties:
                correct_key = alias_to_param[provided_key]
                corrected_args[correct_key] = value
                print(f"  🔄 Corrected {provided_key} → {correct_key}")
                matched = True
            
            # Strategy 3: Fuzzy match
            if not matched:
                for expected_param in properties.keys():
                    if (provided_key.lower() in expected_param.lower() or 
                        expected_param.lower() in provided_key.lower()):
                        corrected_args[expected_param] = value
                        print(f"  🔍 Fuzzy match: {provided_key} → {expected_param}")
                        matched = True
                        break
            
            # If still no match, keep as-is
            if not matched:
                corrected_args[provided_key] = value
                print(f"  ⚠️ No match found for: {provided_key}, keeping as-is")
        
        # Check for missing required fields
        missing_fields = [field for field in required if field not in corrected_args]
        if missing_fields:
            print(f"  ❌ Missing required: {missing_fields}")
            
            return f"Missing required: {missing_fields}", False
        
        print("  ✅ Validation passed")
        print(f"  📦 Final args: {json.dumps(corrected_args, indent=6)}")
        
        return corrected_args, matched
    
    # Handle Pydantic schemas
    try:
        schema_name = getattr(tool.args_schema, '__name__', 'UnknownSchema')
        print(f"🔍 Validating against {schema_name}")
        
        if hasattr(tool.args_schema, 'model_validate'):
            validated = tool.args_schema.model_validate(tool_args)
            validated_dict = validated.model_dump()
        elif hasattr(tool.args_schema, 'parse_obj'):
            validated = tool.args_schema.parse_obj(tool_args)
            validated_dict = validated.dict()
        else:
            return tool_args, True
        
        print("  ✅ Validation passed")
        return validated_dict, True
        
    except ValidationError as e:
        print(f"  ❌ VALIDATION FAILED:")
        for error in e.errors():
            field = "->".join(str(loc) for loc in error['loc'])
            print(f"    -  {field}: {error['msg']}")
        return {}, False
    except Exception as e:
        print(f"  ⚠️ Error: {e}")
        return {}, False
def parse_with_multiple_strategies(result: dict) -> Optional[tuple]:
    """✅ Multi-strategy parsing with fallbacks"""
        
    print(f"  Strategy: ReActJsonOutputParser")
    try:
        # ✅ Check for action/name with fallback
        action = result.get("action") or result.get("name", "")
        
        # ✅ Check for action_input/arguments with fallback
        action_input = result.get("action_input") or result.get("arguments", {})
        
        # Check if it's the final answer
        if action == "Final Answer":
            return ('finish', {'tool': action, 'input': action_input})
        
        return ('action', {'tool': action, 'input': action_input})
        
    except Exception as e:
        return ('action', {'tool': 'Tool Call failed', 'input': {}})



def get_base_prompt(tool_descriptions: str):
    base_prompt = f"""You are a ReAct reasoning agent. Follow this EXACT format:

        AVAILABLE TOOLS:
        {tool_descriptions}

        RESPONSE FORMAT:
        Step 1: THINK NODE
        - Perform granular thinking (subtask decomposition)
        - Think about what you need to do next
        - Based on the previous actions taken (if any), generate a string of next best step.
        - 
        - FOR THINK NODE(wrap in json block): 
        {{
            "thought": ......
        }}
        Step 2: ACTION NODE 
        - Output the best action
        - Use ONE of these actions:
        FOR TOOL CALL (wrap in json block):
            {{
            "action": "<tool_name>",
            "action_input": {{"param1": "value1", "param2": "value2"}}
            }}

            FOR FINAL ANSWER (after completing ALL steps):
            {{
            "action": "Final Answer",
            "action_input": "Your complete response"
            }}

        CRITICAL RULES:
        1. Use EXACT tool names from above
        2. Use EXACT parameter names (see Parameter Details)
        3. ALWAYS wrap JSON in triple backticks with json marker
        4. For multi-step tasks, call tools FIRST, answer AFTER seeing results
        5. NEVER guess - use tools to get information
        6: Provide the complete Final Answer when done, do not leave it incomplete and miss details
        """
    return base_prompt
def build_react_agent_graph(tools: List[BaseTool], enable_clarification: bool = True,
                           model_mode: Optional[str] = None):
    """
    ✅ UPDATED: Support both local and vLLM server modes
    
    Args:
        tools: List of available tools
        enable_clarification: Whether to enable clarification mode
        model_mode: "local" or "vllm" (defaults to MODEL_MODE env var)
    """
    
    mode = MODEL_MODE
    print(mode)
    if tools:
        print(f"\n📚 Loading {len(tools)} tools...")
        
        tool_descriptions = "".join([get_tool_schema_description(tool) for tool in tools])
        # print("Tools: ", tool_descriptions)
        base_prompt = f"""You are a ReAct reasoning agent. Follow this EXACT format:

        AVAILABLE TOOLS:
        {tool_descriptions}

        RESPONSE FORMAT:
        Step 1: THINK NODE
        - Perform granular thinking (subtask decomposition)
        - Think about what you need to do next
        - Based on the previous actions taken (if any), generate a string of next best step.
        - 
        - FOR THINK NODE(wrap in json block): 
        {{
            "thought": ......
        }}
        Step 2: ACTION NODE 
        - Output the best action
        - Use ONE of these actions:
        FOR TOOL CALL (wrap in json block):
            {{
            "action": "<tool_name>",
            "action_input": {{"param1": "value1", "param2": "value2"}}
            }}

            FOR FINAL ANSWER (after completing ALL steps):
            {{
            "action": "Final Answer",
            "action_input": "Your complete response"
            }}

        CRITICAL RULES:
        1. Use EXACT tool names from above
        2. Use EXACT parameter names (see Parameter Details)
        3. ALWAYS wrap JSON in triple backticks with json marker
        4. For multi-step tasks, call tools FIRST, answer AFTER seeing results
        5. NEVER guess - use tools to get information
        6: Provide the complete Final Answer when done, do not leave it incomplete and miss details
        """
        
        if enable_clarification:
            system_prompt = base_prompt + """
CLARIFICATION MODE: ENABLED
- If requirements are unclear, respond:
  {"action": "clarify", "action_input": {"question": "What specifically do you mean by X?"}}
"""
        else:
            system_prompt = base_prompt
    else:
        system_prompt = "You are a ReAct agent."
    
    # ✅ Load model based on mode
    if mode == "vllm":
        llm, vllm_model_name = load_vllm_model(VLLM_BASE_URL, VLLM_API_KEY)
        print(f"🤖 Using vLLM SERVER mode: {vllm_model_name}")
        llm_with_stop = llm.bind(stop=["\nObservation", "\nObservation:", "Observation:"])
        use_local_parser = True  # vLLM uses same parsing as local
    elif mode == "local":
        llm = load_local_model(LOCAL_MODEL_PATH)
        print(f"🤖 Using LOCAL model mode")
        llm_with_stop = llm.bind(stop=["\nObservation", "\nObservation:", "Observation:"])
        use_local_parser = True
    elif mode == "gpt":
        llm = load_gpt_model()
        print(f"🤖 Using GPT model mode")
        llm_with_stop = llm.bind(stop=["\nObservation", "\nObservation:", "Observation:"])
        use_local_parser = False
    else:
        raise ValueError(f"Unknown model mode: {mode}. Use 'local' or 'vllm'")
    
    # ... [Keep all existing node functions: think_node, action_node, observe_node, 
    #      clarify_node, final_node, and routing functions] ...
    def parse_thinking(text: str):
        if "<think>" in text and "</think>" in text:
            start = text.find("<think>")
            end = text.find("</think>") + len("</think>")
            thinking = text[start:end]
            remainder = text[end:].strip()
            return thinking, remainder
        return "", text

    def parse_json(json_str):
        # Remove code block markers
        if json_str.startswith("```"):
            json_str = json_str[len("```json"):].strip()
        if json_str.endswith("```"):
            json_str = json_str[:-3].strip()
        if json_str.startswith("<tool_call>"):
            json_str = json_str[len("<tool_call>"):].strip()
        if json_str.endswith("</tool_call>"):
            json_str = json_str[:-len("</tool_call>")].strip()
        try:
            # First try standard JSON parsing
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # Try parsing as Python dict with single quotes
                data = ast.literal_eval(json_str)
                return data
            except (ValueError, SyntaxError):
                try:
                    # Last resort: replace single quotes with double quotes
                    json_str_fixed = json_str.replace("'", '"')
                    return json.loads(json_str_fixed)
                except json.JSONDecodeError as e:
                    print(f"Parsing failed for string {json_str} with error: {e}")
                    return ""

    def filter_messages_for_local_model(messages: List, current_query: str = "", include_system_prompt: bool = True) -> List:
        """
        Build clean conversation history for local models:
        1. System prompt (once, if include_system_prompt=True)
        2. ALL User queries (not just the first one)
        3. Intermediate steps (thought -> action -> observation cycles)
        """
        filtered = []
        system_added = False
        
        for msg in messages:
            # Add System message only once
            if isinstance(msg, SystemMessage):
                if not system_added and include_system_prompt:
                    filtered.append(msg)
                    system_added = True
                continue
            
            # ✅ FIX: Add ALL user messages (not just the first one)
            if isinstance(msg, HumanMessage):
                # Skip only the intermediate prompts like "THINK NODE:" and "ACTION NODE:"
                if "THINK NODE:" in msg.content or "ACTION NODE:" in msg.content:
                    continue
                else:
                    # Add all genuine user messages
                    filtered.append(msg)
            
            elif isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_name = msg.tool_calls[0].get('name', 'unknown')
                    tool_args = msg.tool_calls[0].get('args', {})
                    formatted_msg = AIMessage(
                        content=f"Action: Using {tool_name} with args: {json.dumps(tool_args)}",
                        tool_calls=msg.tool_calls
                    )
                    filtered.append(formatted_msg)
                elif msg.content:
                    try:
                        parsed = json.loads(msg.content) if isinstance(msg.content, str) and msg.content.startswith('{') else None
                        if parsed and 'thought' in parsed:
                            filtered.append(AIMessage(content=f"Thought: {parsed['thought']}"))
                        else:
                            filtered.append(msg)
                    except:
                        filtered.append(msg)
            
            elif isinstance(msg, ToolMessage):
                filtered.append(
                    HumanMessage(content=f"Observation: Tool '{msg.name}' returned: {msg.content}")
                )
        
        # Add system prompt at the beginning if not already added and requested
        if include_system_prompt and not system_added:
            filtered.insert(0, SystemMessage(content=system_prompt))
        
        return filtered



    def build_conversation_summary(state: AgentState) -> str:
        """Build a concise summary of conversation progress"""
        summary_parts = []
        
        for step in state.trajectory:
            if step.step_type == "thought":
                try:
                    thought_data = json.loads(step.content)
                    summary_parts.append(f"💭 Thought: {thought_data.get('thought', step.content)}")
                except:
                    summary_parts.append(f"💭 Thought: {step.content}")
            
            elif step.step_type == "action":
                if step.tool_used:
                    summary_parts.append(f"⚡ Action: {step.tool_used}({json.dumps(step.tool_input)})")
                else:
                    summary_parts.append(f"⚡ Action: {step.content}")
            
            elif step.step_type == "observation":
                summary_parts.append(f"👁️ Observation: {step.content}")
        
        return "\n".join(summary_parts[-5:])  # Last 5 steps only


    def think_node(state: AgentState) -> AgentState:
        step_num = len([s for s in state.trajectory if s.step_type == "thought"]) + 1
        # print(state.messages)
        if DEBUG_MODE:
            print(f"\n{'='*80}")
            print(f"🧠 THINK NODE - Step {step_num}")
            print(f"{'='*80}")
        
        if state.subtasks_identified:
            progress = f"Completed: {len(state.subtasks_completed)}/{len(state.subtasks_identified)}"
        else:
            progress = "Analyzing task..."
        
        # Build conversation summary
        conversation_summary = build_conversation_summary(state)
        
        think_prompt = f"""
            Current Query: {state.current_query}
            Conversation Summary: {conversation_summary}

            Status: {progress}
           
            Think: What should I do next based on the Current to complete the task?
            Respond in JSON format: {{"thought": "your reasoning here"}}"""
        
         # ✅ CRITICAL FIX: Handle messages differently for GPT vs Local
        if use_local_parser:
            # Local/vLLM mode: filter and convert ToolMessages
            filtered_messages = filter_messages_for_local_model(state.messages, state.current_query)
            messages = filtered_messages + [HumanMessage(content=think_prompt)]
        else:
            # ✅ GPT mode: Keep ALL messages intact for OpenAI API
            messages = state.messages + [HumanMessage(content=think_prompt)]
        # print("Think Node Conversation: ", messages)
        if DEBUG_MODE:
            print(f"📊 Message count: {len(messages)}")
        
        response = llm.invoke(messages)
        thinking, _remainder = parse_thinking(response.content)
        parsed_thought = _remainder
        
        if DEBUG_MODE:
            try:
                print(f"📝 Thought: {parsed_thought['thought']}")
            except:
                print(f"📝 Thought: {parsed_thought}")
        
        thought_step = TrajectoryStep(
            step_number=len(state.trajectory),
            step_type="thought",
            content=json.dumps(parsed_thought) if isinstance(parsed_thought, dict) else str(parsed_thought),
            timestamp=datetime.now().isoformat()
        )
        
        # ✅ Add AIMessage with thought content (not the prompt)
        thought_message = AIMessage(content=json.dumps(parsed_thought) if isinstance(parsed_thought, dict) else str(parsed_thought))
        
        return AgentState(
            messages=state.messages + [thought_message],
            trajectory=state.trajectory + [thought_step],
            current_step=state.current_step,
            max_steps=state.max_steps,
            task_completed=state.task_completed,
            current_query=state.current_query,
            final_answer=state.final_answer,
            needs_clarification=state.needs_clarification,
            clarification_question=state.clarification_question,
            enable_clarification=state.enable_clarification,
            subtasks_identified=state.subtasks_identified,
            subtasks_completed=state.subtasks_completed,
            pending_subtasks=state.pending_subtasks
        )


    def action_node(state: AgentState) -> AgentState:
        step_num = len([s for s in state.trajectory if s.step_type == "action"]) + 1
        print(f"\n{'='*80}")
        print(f"⚡ ACTION NODE - Step {step_num}/{state.max_steps}")
        print(f"{'='*80}")
        
        # Build conversation summary
        conversation_summary = build_conversation_summary(state)
        
        action_prompt = f"""Current Query: {state.current_query}
    Conversation Summary:   
    {conversation_summary}

    Based on your previous thought, select and execute the most appropriate action.

    INSTRUCTIONS: 
    - Select the best tool from available tools
    - Check Argument Schema carefully for each parameter type
    - Provide ALL REQUIRED arguments
    - Use EXACT parameter names
    - Use the appropriate tool based on the server_name in the description in [], e.g., [gitlab], [plane], [rocketchat], etc.
    Respond with JSON format:
    
    For tool call: {{"action": "<tool_name>", "action_input": {{"param1": "value1"}}}}
    For final answer: {{"action": "Final Answer", "action_input": "your complete response"}}"""
        
        # ✅ CRITICAL FIX: Handle messages differently for GPT vs Local
        if use_local_parser:
            # Local/vLLM mode: filter and convert ToolMessages
            filtered_messages = filter_messages_for_local_model(state.messages, state.current_query)
            messages = filtered_messages + [HumanMessage(content=action_prompt)]
        else:
            # ✅ GPT mode: Keep ALL messages intact for OpenAI API
            messages = state.messages + [HumanMessage(content=action_prompt)]
        # print("Action Conversation messages: ", messages)
        if DEBUG_MODE:
            print(f"📊 Message count: {len(messages)}")
        # print("Action Node Message: ", messages)
        response = llm_with_stop.invoke(messages)
        thinking, _remainder = parse_thinking(response.content)
        # Handle response content
        parsed_action = parse_json(_remainder)
        print("Action Output: ", parsed_action)
        # ✅ Check for clarification action
        if isinstance(parsed_action, dict) and parsed_action.get('action') == 'clarify':
            clarification_q = parsed_action.get('action_input', {}).get('question', 'Could you clarify?')
            action_step = TrajectoryStep(
                step_number=len(state.trajectory),
                step_type="action",
                content=str(parsed_action),
                timestamp=datetime.now().isoformat()
            )
            return AgentState(
                messages=state.messages + [AIMessage(content=json.dumps(parsed_action))],
                trajectory=state.trajectory + [action_step],
                current_step=state.current_step,
                max_steps=state.max_steps,
                task_completed=False,
                current_query=state.current_query,
                final_answer="",
                needs_clarification=True,  # ✅ Set this flag
                clarification_question=clarification_q,  # ✅ Store the question
                enable_clarification=state.enable_clarification,
                subtasks_identified=state.subtasks_identified,
                subtasks_completed=state.subtasks_completed,
                pending_subtasks=state.pending_subtasks
            )

        parsed = parse_with_multiple_strategies(parsed_action)
        
        tool_calls_detected = False
        task_completed = False
        final_answer = ""
        validated_tool_call = None
        ai_message = None
        
        if parsed:
            action_type, action_data = parsed
            if action_type == 'finish':
                task_completed = True
                final_answer = action_data.get('input', action_data) if isinstance(action_data, dict) else action_data
                ai_message = AIMessage(content=str(final_answer))
                
            elif action_type == 'action':
                tool_name = action_data['tool'] 
                tool_input = action_data['input']
                
                validated_args,validation_match = validate_tool_input_with_correction(tool_name, tool_input, tools)
                # print("Validated Arguments Content",validated_args)
                if validation_match:
                    tool_calls_detected = True
                    validated_tool_call = {
                        'name': tool_name,
                        'args': validated_args,
                        'id': f"call_{uuid.uuid4().hex[:8]}",
                        'type': 'tool_call'
                    }
                    
                    ai_message = AIMessage(
                        content="",
                        tool_calls=[validated_tool_call]
                    )
                    
                    print(f"\n🔧 Tool Call Created: {tool_name}")
                    print(f"📥 Arguments: {validated_args}")
                else:
                    # Validation failed due to missing required fields
                    ai_message = AIMessage(content=str(validated_args))
                                       
        
        if ai_message is None:
            ai_message = AIMessage(content=str(parsed_action))
        
        if tool_calls_detected and validated_tool_call:
            action_step = TrajectoryStep(
                step_number=len(state.trajectory),
                step_type="action",
                content=f"Using {validated_tool_call['name']}: {validated_tool_call['args']}",
                timestamp=datetime.now().isoformat(),
                tool_used=validated_tool_call['name'],
                tool_input=validated_tool_call['args']
            )
        else:
            action_step = TrajectoryStep(
                step_number=len(state.trajectory),
                step_type="action",
                content=str(parsed_action),
                timestamp=datetime.now().isoformat()
            )
        
        # ✅ Only add AIMessage (no prompt)
        return AgentState(
            messages=state.messages + [ai_message],
            trajectory=state.trajectory + [action_step],
            current_step=state.current_step ,
            max_steps=state.max_steps,
            task_completed=task_completed,
            current_query=state.current_query,
            final_answer=str(final_answer),
            needs_clarification=False,
            clarification_question="",
            enable_clarification=state.enable_clarification,
            subtasks_identified=state.subtasks_identified,
            subtasks_completed=state.subtasks_completed,
            pending_subtasks=state.pending_subtasks
        )



    def observe_node(state: AgentState) -> AgentState:
        print("\n" + "="*80)
        print("👁️ OBSERVE NODE")
        print("="*80)
        
        if not state.messages:
            return state
        
        last_message = state.messages[-1]
        new_pending = state.pending_subtasks.copy()
        new_completed = state.subtasks_completed.copy()
        
        # ✅ Enhanced tool output display
        if isinstance(last_message, ToolMessage):
            print(f"\n{'='*80}")
            print(f"🔧 TOOL EXECUTION COMPLETE")
            print(f"{'='*80}")
            print(f"📛 Tool Name: {last_message.name}")
            print(f"🆔 Tool Call ID: {last_message.tool_call_id}")
            print(f"✅ Status: {getattr(last_message, 'status', 'success')}")
            print(f"\n📤 TOOL OUTPUT:")
            print(f"{'-'*80}")
            print(f"{last_message.content}")
            print(f"{'-'*80}\n")
            
            # Get the corresponding tool call from previous AIMessage
            tool_input = None
            for msg in reversed(state.messages[:-1]):
                if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.get('id') == last_message.tool_call_id:
                            tool_input = tc.get('args', {})
                            break
                    if tool_input is not None:
                        break
            print("Tools Name: ", last_message.name)
            if tool_input:
                print(f"📥 Tool Input Used:")
                print(f"{json.dumps(tool_input, indent=2)}")
                print(f"{'='*80}\n")
            
            # Store in trajectory
            observation_step = TrajectoryStep(
                step_number=len(state.trajectory) + 1,
                step_type="observation",
                content=f"Tool {last_message.name} returned: {last_message.content}",
                timestamp=datetime.now().isoformat(),
                tool_used=last_message.name,
                tool_output=last_message.content
            )
            
            return AgentState(
                messages=state.messages,
                trajectory=state.trajectory + [observation_step],
                current_step=state.current_step + 1,
                max_steps=state.max_steps,
                task_completed=state.task_completed,
                current_query=state.current_query,
                final_answer=state.final_answer,
                needs_clarification=state.needs_clarification,
                clarification_question=state.clarification_question,
                enable_clarification=state.enable_clarification,
                subtasks_identified=state.subtasks_identified,
                subtasks_completed=new_completed,
                pending_subtasks=new_pending
            )
        
        return AgentState(
            messages=state.messages,
            trajectory=state.trajectory,
            current_step=state.current_step,
            max_steps=state.max_steps,
            task_completed=state.task_completed,
            current_query=state.current_query,
            final_answer=state.final_answer,
            needs_clarification=state.needs_clarification,
            clarification_question=state.clarification_question,
            enable_clarification=state.enable_clarification,
            subtasks_identified=state.subtasks_identified,
            subtasks_completed=new_completed,
            pending_subtasks=new_pending
        )

    
    def clarify_node(state: AgentState) -> AgentState:
        print(f"\n{'='*80}")
        print(f"❓ CLARIFY NODE")
        print(f"{'='*80}")
        
        # Extract clarification question from last action
        clarification_question = state.clarification_question
        
        if not clarification_question:
            # Fallback: try to extract from last message
            try:
                last_msg = state.messages[-1]
                if isinstance(last_msg.content, str):
                    parsed = json.loads(last_msg.content)
                    if 'action' in parsed and parsed['action'] == 'clarify':
                        clarification_question = parsed.get('action_input', {}).get('question', '')
            except:
                clarification_question = "Could you provide more details about your request?"
        
        print(f"📢 Question to user: {clarification_question}")
        
        # Create clarification message
        clarification_message = f"❓ {clarification_question}"
        
        clarify_step = TrajectoryStep(
            step_number=len(state.trajectory) + 1,
            step_type="clarification",
            content=clarification_message,
            timestamp=datetime.now().isoformat()
        )
        
        # ✅ Return state with clarification request
        # The execution will pause here, waiting for the next user input
        return AgentState(
            messages=state.messages + [AIMessage(content=clarification_message)],
            trajectory=state.trajectory + [clarify_step],
            current_step=state.current_step + 1,
            max_steps=state.max_steps,
            task_completed=False,
            current_query=state.current_query,
            final_answer="",
            needs_clarification=True,  # ✅ Keep this True until user responds
            clarification_question=clarification_question,
            enable_clarification=state.enable_clarification,
            subtasks_identified=state.subtasks_identified,
            subtasks_completed=state.subtasks_completed,
            pending_subtasks=state.pending_subtasks
        )
    
    def final_node(state: AgentState) -> AgentState:
        print(f"\n🏁 FINAL NODE")
        if state.trajectory:
            save_trajectory_to_file(state.trajectory, state.current_query, state.final_answer)
        
        # from langgraph.graph.message import REMOVE_ALL_MESSAGES
        # messages = state.messages
        # if(len(state.messages)>30):
        #     messages = [RemoveMessage(id=REMOVE_ALL_MESSAGES)] 
        # Clear all messages
        return AgentState(
            messages=state.messages,
            trajectory=state.trajectory,
            current_step=state.current_step,
            max_steps=state.max_steps,
            task_completed=True,
            current_query=state.current_query,
            final_answer=state.final_answer,
            needs_clarification=False,
            clarification_question="",
            enable_clarification=state.enable_clarification,
            subtasks_identified=state.subtasks_identified,
            subtasks_completed=state.subtasks_completed,
            pending_subtasks=[]
        )


    
    def should_continue_from_think(state: AgentState) -> Literal["action", "clarify", "final"]:
        if state.current_step >= state.max_steps:
            return "final"
        if state.enable_clarification and state.needs_clarification:
            return "clarify"
        return "action"
    
    def should_continue_from_action(state: AgentState) -> Literal["tools", "think", "final"]:
        """Route after action - CRITICAL FIX"""
        print(f"\n🔀 ROUTING from action:")
        print(f"  Current step: {state.current_step}/{state.max_steps}")
        print(f"  Task completed: {state.task_completed}")
        
        # ✅ Check if task is completed first
        if state.task_completed:
            print("  → Going to FINAL (task completed)")
            return "final"
        
        # ✅ CRITICAL: Check for tool_calls in the LAST message
        if state.messages:
            last_msg = state.messages[-1]
            print(f"  Last message type: {type(last_msg).__name__}")
            
            if isinstance(last_msg, AIMessage):
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    print(f"  ✅ Tool calls detected: {[tc['name'] for tc in last_msg.tool_calls]}")
                    print("  → Going to TOOLS")
                    return "tools"  # ✅ MUST go to tools to execute and create ToolMessage
                else:
                    print("  No tool calls in AIMessage")
        
        # ✅ Check max steps
        if state.current_step >= state.max_steps:
            print("  → Going to FINAL (max steps)")
            return "final"
        
        # ✅ Default: continue thinking
        print("  → Going to THINK (no tools, continue reasoning)")
        return "think"

    
    def should_continue_from_observe(state: AgentState) -> Literal["think", "final"]:
        return "final" if (state.current_step >= state.max_steps or state.task_completed) else "think"
    
    # Build graph
    builder = StateGraph(AgentState)
    builder.add_node("think", think_node)
    builder.add_node("action", action_node)
    builder.add_node("observe", observe_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("final", final_node)
    
    if tools:
        builder.add_node("tools", ToolNode(tools))
    
    builder.add_edge(START, "think")
    builder.add_conditional_edges("think", should_continue_from_think, {
        "action": "action",
        "clarify": "clarify",
        "final": "final"
    })
    
    if tools:
        builder.add_conditional_edges("action", should_continue_from_action, {
            "tools": "tools", "think": "think", "final": "final"
        })
        builder.add_edge("tools", "observe")
        builder.add_conditional_edges("observe", should_continue_from_observe, {"think": "think", "final": "final"})
    else:
        builder.add_conditional_edges("action", should_continue_from_action, {
            "think": "think", "final": "final"
        })
    
    builder.add_edge("clarify", "action")
    builder.add_edge("final", END)
    
    return builder.compile(checkpointer=MemorySaver())

__all__ = ["build_react_agent_graph", "AgentState", "TrajectoryStep", "save_trajectory_to_file", 
           "load_local_model", "load_vllm_model"]
