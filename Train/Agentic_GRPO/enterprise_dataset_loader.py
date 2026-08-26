"""
EnterpriseBench Dataset Loader - Handles Your Exact Task Format

FIXED: Ground truth reward function properly parses tools and final answers

Loads tasks from your EnterpriseBench format with:
- task_id, instruction, chain_of_thought, ground_truth, etc.
"""

import json
import re
from typing import List, Dict, Any
import logging
import time
import os
import random

try:
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    AzureChatOpenAI = None
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

load_dotenv()
logger = logging.getLogger(__name__)


def _require_langchain_openai(feature: str) -> None:
    if (
        AzureChatOpenAI is None
        or ChatOpenAI is None
        or HumanMessage is None
        or SystemMessage is None
    ):
        raise ImportError(
            f"{feature} requires langchain-openai and langchain-core to be installed."
        )


def _extract_gold_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Preserve assistant/tool trajectory messages in source order for judge input.
    """
    gold_messages: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"assistant", "tool"}:
            continue

        preserved: Dict[str, Any] = {"role": role}
        if "content" in message:
            preserved["content"] = message.get("content")
        if role == "assistant" and "tool_calls" in message:
            preserved["tool_calls"] = message.get("tool_calls")
        if role == "tool":
            if "tool_call_id" in message:
                preserved["tool_call_id"] = message.get("tool_call_id")
            if "name" in message:
                preserved["name"] = message.get("name")
        gold_messages.append(preserved)
    return gold_messages


# def load_enterprise_tasks_v2(
#     path: str,
#     max_tasks: int = None,
#     difficulty_filter: str = None,  # "EASY", "MEDIUM", "HARD"
#     domain_filter: str = None,       # "HR", "CRM", "GitHub", etc.
#     min_steps: int = None,
#     max_steps: int = None
# ) -> List[Dict]:
#     """
#     Load EnterpriseBench tasks from your exact JSON format.

#     Your task format:
#     {
#         "task_id": "task_seq_0_3_filtered_1",
#         "instruction": "As part of maintaining...",
#         "prerequisite_context": [],
#         "chain_of_thought": [
#             {
#                 "step": 3,
#                 "rationale": "...",
#                 "tool": "delete_message",
#                 "inputs": {...},
#                 "expected_output": "..."
#             }
#         ],
#         "required_tools": ["delete_message"],
#         "success_criteria": [...],
#         "domain": "HR",
#         "difficulty": "EASY",
#         "ground_truth": {
#             "final_output": "...",
#             "all_step_outputs": [...],
#             "final_entities": [...]
#         },
#         "meta": {...}
#     }

#     Args:
#         path: Path to your tasks JSON file
#         max_tasks: Limit number of tasks (for testing)
#         difficulty_filter: Only load tasks with this difficulty
#         domain_filter: Only load tasks from this domain
#         min_steps: Minimum number of steps
#         max_steps: Maximum number of steps

#     Returns:
#         List of formatted tasks for GRPO training
#     """
#     try:
#         with open(path, 'r') as f:
#             raw_tasks = json.load(f)
#         logger.info("Tasks Shuffled")
#         random.shuffle(raw_tasks)   # Shuffles the list in-place
#         logger.info(f"Loaded {len(raw_tasks)} raw tasks from {path}")

#         # Apply filters
#         filtered_tasks = []
#         for task in raw_tasks:
#             # Difficulty filter
#             if difficulty_filter and task.get('difficulty') != difficulty_filter:
#                 continue

#             # Domain filter
#             if domain_filter and task.get('domain') != domain_filter:
#                 continue

#             # Steps filter - FIXED: fallback to chain_of_thought length
#             num_steps = task.get('meta', {}).get('num_steps')
#             if num_steps is None:
#                 num_steps = len(task.get('chain_of_thought', []))
            
#             if min_steps and num_steps < min_steps:
#                 continue
#             if max_steps and num_steps > max_steps:
#                 continue

#             filtered_tasks.append(task)

#         logger.info(f"After filtering: {len(filtered_tasks)} tasks")

#         # Convert to GRPO training format
#         formatted_tasks = []
#         for task in filtered_tasks:
#             formatted_task = {
#                 # Required fields
#                 'id': task['task_id'],
#                 'user': task['instruction'],

#                 # Optional: Gold trajectory for ground truth rewards
#                 'gold_chain_of_thought': task.get('chain_of_thought', []),
#                 'gold_final_output': task.get('ground_truth', {}).get('final_output'),
#                 'gold_step_outputs': task.get('ground_truth', {}).get('all_step_outputs', []),

#                 # Metadata for analysis
#                 'required_tools': task.get('required_tools', []),
#                 'domain': task.get('domain'),
#                 'difficulty': task.get('difficulty'),
#                 'num_steps': num_steps,
#                 'success_criteria': task.get('success_criteria', []),
#             }

#             formatted_tasks.append(formatted_task)
        
#         # Limit if specified (FIXED: handle max_tasks=0)
#         if max_tasks and max_tasks > 0:
#             formatted_tasks = formatted_tasks[:max_tasks]
#             logger.info(f"Limited to {max_tasks} tasks for this run")

#         # Log statistics
#         log_dataset_statistics(formatted_tasks)

#         return formatted_tasks

#     except FileNotFoundError:
#         logger.error(f"Dataset file not found: {path}")
#         raise
#     except json.JSONDecodeError as e:
#         logger.error(f"Invalid JSON in dataset file: {e}")
#         raise
#     except Exception as e:
#         logger.error(f"Error loading dataset: {e}")
#         raise

import json
import random
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def _deepcopy_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _checkpoint_rationale(
    checkpoint: Dict[str, Any],
    cot_steps: List[Dict[str, Any]],
) -> str:
    step_number = checkpoint.get("step")
    if isinstance(step_number, int):
        for cot_step in cot_steps:
            if cot_step.get("step") == step_number:
                return str(
                    cot_step.get("subgoal")
                    or cot_step.get("rationale")
                    or cot_step.get("expected_output")
                    or ""
                ).strip()
    return ""


def _expected_output_payload(checkpoint: Dict[str, Any]) -> Any:
    if "expected_output_parsed" in checkpoint:
        return checkpoint.get("expected_output_parsed")

    payload: Dict[str, Any] = {}
    expected_fields = checkpoint.get("expected_output_fields", []) or []
    if expected_fields:
        payload["expected_output_fields"] = expected_fields
    success_validation = checkpoint.get("success_validation")
    if isinstance(success_validation, dict) and success_validation:
        payload["success_validation"] = success_validation
    expected_effects = checkpoint.get("expected_effects", []) or []
    if expected_effects:
        payload["expected_effects"] = expected_effects
    state_tracking = checkpoint.get("state_tracking")
    if isinstance(state_tracking, dict) and state_tracking:
        payload["state_tracking"] = state_tracking
    return payload or None


def _build_gold_steps_from_checkpoints(
    reward_checkpoints: List[Dict[str, Any]],
    cot_steps: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    gold_steps: List[Dict[str, Any]] = []

    for checkpoint in reward_checkpoints:
        if not isinstance(checkpoint, dict):
            continue
        tool_name = checkpoint.get("tool") or checkpoint.get("api_id")
        if not tool_name:
            continue

        gold_steps.append(
            {
                "step": checkpoint.get("step", len(gold_steps) + 1),
                "rationale": _checkpoint_rationale(checkpoint, cot_steps),
                "tool": tool_name,
                "inputs": checkpoint.get("input_bindings", {}) or {},
                "expected_output": _expected_output_payload(checkpoint),
                "must_succeed": bool(checkpoint.get("must_succeed", False)),
                "operation": checkpoint.get("operation"),
            }
        )

    return gold_steps


def _build_gold_final_output_from_task(task: Dict[str, Any]) -> str:
    success_criteria = task.get("success_criteria", []) or []
    criteria_text = "\n".join(
        f"- {criterion}" for criterion in success_criteria if isinstance(criterion, str)
    ).strip()

    ground_truth = task.get("ground_truth", {}) or {}
    tool_sequence = ground_truth.get("expected_tool_sequence", []) or []
    tool_sequence_text = ", ".join(str(tool) for tool in tool_sequence if tool)

    sections: List[str] = []
    if criteria_text:
        sections.append("Success criteria:\n" + criteria_text)
    if tool_sequence_text:
        sections.append(f"Expected tool sequence: {tool_sequence_text}")
    if ground_truth.get("low_level_subgoals"):
        subgoals = "\n".join(
            f"- {goal}" for goal in ground_truth.get("low_level_subgoals", []) if goal
        ).strip()
        if subgoals:
            sections.append("Low-level subgoals:\n" + subgoals)

    return "\n\n".join(sections).strip()


def _format_checkpoint_task(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    cot_steps = item.get("chain_of_thought", []) or []
    reward_checkpoints = _deepcopy_jsonable(item.get("reward_checkpoints", []) or [])
    gold_steps = _build_gold_steps_from_checkpoints(reward_checkpoints, cot_steps)
    required_tools = item.get("required_tools", []) or [
        step.get("tool") for step in gold_steps if step.get("tool")
    ]

    return {
        "id": item.get("task_id") or item.get("id") or f"enterprise_task_{idx}",
        "user": str(item.get("instruction", "")).strip(),
        "gold_chain_of_thought": cot_steps,
        "gold_step_outputs": gold_steps,
        "gold_final_output": _build_gold_final_output_from_task(item),
        "gold_messages": [],
        "required_tools": sorted({tool for tool in required_tools if tool}),
        "domain": item.get("domain"),
        "difficulty": item.get("difficulty"),
        "num_steps": len(gold_steps),
        "success_criteria": item.get("success_criteria", []) or [],
        "reward_checkpoints": reward_checkpoints,
        "ground_truth": _deepcopy_jsonable(item.get("ground_truth", {}) or {}),
        "meta": _deepcopy_jsonable(item.get("meta", {}) or {}),
        "prerequisite_context": _deepcopy_jsonable(item.get("prerequisite_context", {}) or {}),
    }


def _format_messages_task(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    messages = item.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        return {}

    user_texts = [m.get("content", "") for m in messages if m.get("role") == "user" and m.get("content")]
    instruction = "\n".join(user_texts).strip()
    if not instruction:
        return {}

    gold_steps = []
    required_tools = []
    last_assistant_text = None
    final_answer = None

    i = 0
    while i < len(messages):
        m = messages[i]
        role = m.get("role")

        if role == "assistant":
            if m.get("content") and not m.get("tool_calls"):
                last_assistant_text = str(m["content"]).strip()
                final_answer = last_assistant_text

            if m.get("tool_calls"):
                tool_calls = m.get("tool_calls", [])
                tool_outputs = []
                j = i + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    tool_outputs.append(messages[j])
                    j += 1

                for k, tc in enumerate(tool_calls):
                    fn = (tc or {}).get("function", {}) or {}
                    tool_name = fn.get("name")
                    args = fn.get("arguments", {})
                    if not tool_name:
                        continue

                    required_tools.append(tool_name)
                    expected_output = None
                    if k < len(tool_outputs):
                        expected_output = tool_outputs[k].get("content")

                    gold_steps.append(
                        {
                            "step": len(gold_steps) + 1,
                            "rationale": last_assistant_text or "",
                            "tool": tool_name,
                            "inputs": args if isinstance(args, dict) else {},
                            "expected_output": expected_output,
                        }
                    )

                i = j
                continue

        i += 1

    return {
        "id": item.get("task_id") or item.get("id") or f"chatlog_{idx}",
        "user": instruction,
        "gold_chain_of_thought": [],
        "gold_step_outputs": gold_steps,
        "gold_final_output": final_answer,
        "gold_messages": _extract_gold_messages(messages),
        "required_tools": sorted(set(required_tools)),
        "domain": item.get("domain"),
        "difficulty": item.get("difficulty"),
        "num_steps": len(gold_steps),
        "success_criteria": item.get("success_criteria", []),
        "timestamp": item.get("timestamp"),
    }

def load_enterprise_tasks_v2(
    path: str,
    max_tasks: int = None,
    difficulty_filter: str = None,
    domain_filter: str = None,
    min_steps: int = None,
    max_steps: int = None,
    shuffle: bool = True,
    seed: int = None
) -> List[Dict]:
    """
    Load tasks from the NEW dataset format:

    [
      {
        "messages": [
          {"role": "system", "content": "...tools..."},
          {"role": "user", "content": "...instruction..."},
          {"role": "assistant", "content": "...thought..."},
          {"role": "assistant", "tool_calls": [{"type":"function","function":{"name":"get_product","arguments":{...}}}]},
          {"role": "tool", "name": "get_product", "content": "{...}"},
          ...
          {"role": "assistant", "content": "...final..."}
        ],
        "timestamp": "..."
      },
      ...
    ]

    EXCLUDE system from the instruction (user prompt).
    Derive gold_step_outputs from assistant tool_calls + subsequent tool messages. [file:302]
    """
    try:
        with open(path, "r") as f:
            raw = json.load(f)

        # Accept either list or {"data": [...]}
        if isinstance(raw, dict):
            raw_tasks = raw.get("data", [])
        else:
            raw_tasks = raw

        if shuffle:
            if seed is not None:
                random.seed(seed)
                logger.info(f"Shuffling tasks with seed={seed}")
            else:
                logger.info("Shuffling tasks with random seed")
            random.shuffle(raw_tasks)
        else:
            logger.info("Keeping tasks in original order (shuffle=False)")

        logger.info(f"Loaded {len(raw_tasks)} raw tasks from {path}")

        formatted_tasks: List[Dict[str, Any]] = []

        for idx, item in enumerate(raw_tasks):
            if not isinstance(item, dict):
                continue

            if isinstance(item.get("messages"), list):
                formatted_task = _format_messages_task(item, idx)
            else:
                formatted_task = _format_checkpoint_task(item, idx)

            if not formatted_task or not formatted_task.get("user"):
                continue

            num_steps = int(formatted_task.get("num_steps", 0) or 0)
            difficulty = formatted_task.get("difficulty")
            domain = formatted_task.get("domain")

            if difficulty_filter and difficulty != difficulty_filter:
                continue
            if domain_filter and domain != domain_filter:
                continue
            if min_steps is not None and num_steps < min_steps:
                continue
            if max_steps is not None and num_steps > max_steps:
                continue

            formatted_tasks.append(formatted_task)

        logger.info(f"After filtering: {len(formatted_tasks)} tasks")

        # Limit if specified (handle max_tasks=0)
        if max_tasks is not None and max_tasks > 0:
            formatted_tasks = formatted_tasks[:max_tasks]
            logger.info(f"Limited to {max_tasks} tasks for this run")

        # If you have this helper, keep it; otherwise remove
        # log_dataset_statistics(formatted_tasks)

        return formatted_tasks

    except FileNotFoundError:
        logger.error(f"Dataset file not found: {path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in dataset file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise



def log_dataset_statistics(tasks: List[Dict]):
    """Log statistics about the loaded dataset."""
    if not tasks:
        return

    # Difficulty distribution
    difficulties = {}
    for task in tasks:
        diff = task.get('difficulty', 'UNKNOWN')
        difficulties[diff] = difficulties.get(diff, 0) + 1

    # Domain distribution
    domains = {}
    for task in tasks:
        domain = task.get('domain', 'UNKNOWN')
        domains[domain] = domains.get(domain, 0) + 1

    # Steps distribution
    steps = [task.get('num_steps', 0) for task in tasks]
    avg_steps = sum(steps) / len(steps) if steps else 0

    # Tools distribution
    all_tools = set()
    for task in tasks:
        all_tools.update(task.get('required_tools', []))

    logger.info("="*80)
    logger.info("DATASET STATISTICS")
    logger.info("="*80)
    logger.info(f"Total tasks: {len(tasks)}")
    logger.info(f"\nDifficulty distribution:")
    for diff, count in sorted(difficulties.items()):
        logger.info(f"  {diff}: {count} ({count/len(tasks)*100:.1f}%)")
    logger.info(f"\nDomain distribution:")
    for domain, count in sorted(domains.items()):
        logger.info(f"  {domain}: {count} ({count/len(tasks)*100:.1f}%)")
    logger.info(f"\nSteps:")
    logger.info(f"  Average: {avg_steps:.1f}")
    logger.info(f"  Min: {min(steps) if steps else 0}")
    logger.info(f"  Max: {max(steps) if steps else 0}")
    logger.info(f"\nUnique tools used: {len(all_tools)}")
    logger.info(f"  Tools: {sorted(all_tools)[:10]}...")  # Show first 10
    logger.info("="*80)


# =========================================================================
# 1. GPTCaller Class (Your Provided Code)
# =========================================================================
class GPTCaller:
    """
    Wrapper for GPT API calls using AzureChatOpenAI
    Supports both JSON mode and text responses
    """
    
    def __init__(
        self,
        api_key: str = None,
        api_base: str = None,
        api_version: str = "2024-08-01-preview",
        model_name: str = "gpt-4o",
        max_retries: int = 5
    ):
        """
        Initialize GPT caller with Azure configuration
        
        Args:
            api_key: Azure API key (if None, reads from AZURE_CHAT_API_KEY env var)
            api_base: Azure endpoint (if None, reads from AZURE_CHAT_ENDPOINT env var)
            api_version: Azure API version
            model_name: Model deployment name
            max_retries: Maximum retry attempts
        """
        _require_langchain_openai("GPTCaller")
        self.api_key = api_key or os.getenv("AZURE_API_KEY")
        self.api_base = api_base or os.getenv("AZURE_API_ENDPOINT")
        self.api_version = api_version
        self.model_name = model_name
        self.max_retries = max_retries
        
        if not self.api_key or not self.api_base:
            raise ValueError(
                "Azure API key and endpoint must be provided either as arguments "
                "or via AZURE_CHAT_API_KEY and AZURE_CHAT_ENDPOINT environment variables"
            )
        
        # Initialize base LLM (without JSON mode)
        self.llm = AzureChatOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.api_base,
            model_name=self.model_name,
            temperature=0.3  # Default, can be overridden per call
        )
        
        # Initialize JSON mode LLM
        self.llm_json = self.llm.bind(
            response_format={"type": "json_object"}
        )
    
    async def __call__(
        self,
        prompt: str,
        response_format: str = "json",
        model: str = None,
        temperature: float = 0.3,
        max_tokens: int = 16384
    ) -> Dict[str, Any]:
        """
        Call GPT with prompt and return response
        
        Args:
            prompt: The prompt to send
            response_format: "json" or "text"
            model: Model to use (currently ignored, uses self.model_name)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        
        Returns:
            Parsed JSON response if response_format="json", else dict with "response" key
        """
        
        # Create LLM with specified temperature
        if response_format == "json":
            llm = self.llm.bind(
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens
            )
            # Add system message to ensure JSON output
            system_msg = "You are a helpful AI assistant. Always respond in valid JSON format."
        else:
            llm = AzureChatOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.api_base,
                model_name=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            system_msg = "You are a helpful AI assistant."
        
        # Retry logic
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                # Create messages
                messages = [
                    SystemMessage(content=system_msg),
                    HumanMessage(content=prompt)
                ]
                
                # Call the model
                response = llm.invoke(messages)
                
                # Extract content
                content = response.content
                
                # Parse based on response format
                if response_format == "json":
                    try:
                        parsed = json.loads(content)
                        return parsed
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON response: {e}")
                        print(f"Raw content: {content[:200]}...")
                        # Retry if JSON parsing fails
                        raise e
                else:
                    return {"response": content}
            
            except Exception as e:
                last_error = e
                retries += 1
                time_wait = 15 * retries
                
                print(f"❌ GPT call failed (attempt {retries}/{self.max_retries})")
                print(f"   Error: {str(e)}")
                print(f"   Prompt length: {len(prompt)} chars")
                
                if retries < self.max_retries:
                    print(f"   Retrying in {time_wait} seconds...")
                    time.sleep(time_wait)
                else:
                    print(f"   All retries exhausted. Returning empty response.")
        
        # If all retries fail
        print(f"⚠️  All {self.max_retries} retry attempts failed")
        print(f"   Last error: {last_error}")
        
        if response_format == "json":
            return {}  # Empty dict for JSON mode
        else:
            return {"response": "", "error": str(last_error)}
    
    def sync_call(
        self,
        prompt: str,
        response_format: str = "json",
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Synchronous version of call (for non-async contexts)
        
        Args:
            prompt: The prompt to send
            response_format: "json" or "text"
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        
        Returns:
            Parsed JSON response if response_format="json", else dict with "response" key
        """
        
        # Create LLM with specified temperature
        if response_format == "json":
            llm = self.llm.bind(
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens
            )
            system_msg = "You are a helpful AI assistant. Always respond in valid JSON format."
        else:
            llm = AzureChatOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.api_base,
                model_name=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            system_msg = "You are a helpful AI assistant."
        
        # Retry logic
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                messages = [
                    SystemMessage(content=system_msg),
                    HumanMessage(content=prompt)
                ]
                
                # Synchronous invoke
                response = llm.invoke(messages)
                content = response.content
                
                if response_format == "json":
                    try:
                        parsed = json.loads(content)
                        return parsed
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON response: {e}")
                        raise e
                else:
                    return {"response": content}
            
            except Exception as e:
                last_error = e
                retries += 1
                time_wait = 15 * retries
                
                print(f"❌ GPT call failed (attempt {retries}/{self.max_retries})")
                print(f"   Error: {str(e)}")
                
                if retries < self.max_retries:
                    print(f"   Retrying in {time_wait} seconds...")
                    time.sleep(time_wait)
        
        # If all retries fail
        if response_format == "json":
            return {}
        else:
            return {"response": "", "error": str(last_error)}
class LocalQwenCaller:
    """
    Wrapper for Local Qwen Model (via vLLM/OpenAI-compatible API).
    Drop-in replacement for GPTCaller but points to localhost.
    """
    
    def __init__(
        self,
        api_base: str = "http://localhost:8001/v1",
        api_key: str = "EMPTY", # vLLM usually ignores this
        model_name: str = "Qwen/Qwen3-8B",
        max_retries: int = 3
    ):
        _require_langchain_openai("LocalQwenCaller")
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        
        # Initialize ChatOpenAI client pointing to local server
        self.llm = ChatOpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            model=self.model_name,
            temperature=0.0, # Deterministic by default for Judge
            max_retries=max_retries,
        )

    def sync_call(
        self,
        prompt: str,
        response_format: str = "json",
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Synchronous call to local model.
        """
        # Configure LLM call
        # Note: Qwen/vLLM supports 'json_object' if using the latest version.
        # If not, we rely on the prompt to enforce JSON.
        if response_format == "json":
            kwargs = {"response_format": {"type": "json_object"}}
            system_msg = "You are a strict AI Judge. You MUST output valid JSON only."
        else:
            kwargs = {}
            system_msg = "You are a strict AI Judge."

        # Bind parameters
        llm_call = self.llm.bind(
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                messages = [
                    SystemMessage(content=system_msg),
                    HumanMessage(content=prompt)
                ]
                
                response = llm_call.invoke(messages)
                content = response.content
                
                if response_format == "json":
                    try:
                        # Clean markdown if present (Common in local models)
                        if "```json" in content:
                            content = content.split("```json").split("```").strip()[1]
                        elif "```" in content:
                             content = content.split("```")[1].split("```")[0].strip()
                             
                        parsed = json.loads(content)
                        return parsed
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse Local Qwen JSON: {e}")
                        # Local models might need a nudge on retry
                        raise e
                else:
                    return {"response": content}
            
            except Exception as e:
                last_error = e
                retries += 1
                # Exponential backoff not needed for local, but good for stability
                time.sleep(0.5) 
                
                print(f"❌ Local Qwen call failed (attempt {retries}/{self.max_retries})")
                print(f"   Error: {e}")

        # Fallback if all retries fail
        if response_format == "json":
            return {}
        else:
            return {"response": "", "error": str(last_error)}
try:
    from langchain_aws import ChatBedrock
except ImportError:
    ChatBedrock = None

try:
    import boto3
except ImportError:
    boto3 = None

def load_claude_model():
    if ChatBedrock is None or boto3 is None:
        raise ImportError(
            "load_claude_model requires langchain-aws and boto3 to be installed."
        )
    AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
    session = boto3.Session(region_name=AWS_REGION)
    bedrock_client = session.client(
        "bedrock-runtime",
        endpoint_url=f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com"
    )
    
    llm = ChatBedrock(
        model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        # model_id="meta.llama3-1-70b-instruct-v1:0",
        # model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        client=bedrock_client,
        model_kwargs={
            "temperature": 0,
            "max_tokens": 8192   #change to 10000 for claude 4.5
        },
        beta_use_converse_api=True,
        disable_streaming=False
    )
    llm = llm.bind()
    # llm_with_tools = llm.bind_tools(tools) if tools else llm
    return llm

# =========================================================================
# 2. SYSTEM PROMPT
# =========================================================================

# JUDGE_SYSTEM_PROMPT = """
# You are a precision evaluator for an AI Agent in a **dynamic enterprise tool environment**.

# The agent MUST use tools to interact with the environment. **Skipping tools and guessing answers is a critical failure.**

# You must score the agent's performance on **5 independent dimensions** using **step-by-step evaluation**.

# ---

# ### TASK DATA

# Instruction:
# {instruction}

# REFERENCE – Gold Chain of Thought (one possible correct solution):
# {gold_chain_of_thought}

# REFERENCE – Expected Step Outputs:
# {gold_step_outputs}

# REFERENCE – Expected Final Answer:
# {gold_final_output}

# ---

# ### AGENT TRAJECTORY

# {agent_trajectory}

# ---

# ### SCORING METHODOLOGY

# **Step-by-Step Evaluation:**

# For each turn in the agent's trajectory, evaluate:
# 1. **Thought Quality:** Is the reasoning correct and aligned with the task? (+1 if correct)
# 2. **Tool Selection:** Is the selected tool correct for this step? (+1 if correct)
# 3. **Tool Execution:** Did the tool execute successfully (check Observation)? (+1 if successful)

# Then **normalize** scores by trajectory length to get a 0.0 - 1.0 range for each dimension.

# ---

# ### SCORING DIMENSIONS (0.0 – 1.0 each)

# #### 1. **format_compliance** (Structural correctness)

# Does the agent follow the required Thought/Action/Action Input/Observation/Final Answer format?

# **Scoring:**
# - Check EACH turn for proper format.
# - Score = (Turns with valid format) / (Total turns in trajectory)

# **Examples:**
# - Perfect format all turns: 1.0
# - 3 out of 4 turns have valid format: 0.75
# - Completely broken format: 0.0

# ---

# #### 2. **tool_selection** (Correctness of tool choices)

# **Step-by-step evaluation:**

# For each turn where a tool is called:
# 1. Compare agent's selected tool against the Gold Chain of Thought.
# 2. Award **+1** if the tool matches the expected tool OR is a valid alternative/verification step.
# 3. Award **0** if the tool is wrong, hallucinated, or irrelevant.

# **Normalization:**
# tool_selection_score = (Correct Tool Selections) / (Total Tool Calls in Trajectory)


# **Special cases:**
# - If agent calls 0 tools but gold requires tools: 0.0
# - If agent calls extra valid tools (verification): Count as correct (+1)
# - If agent hallucinates non-existent tools: Count as incorrect (0)

# ---

# #### 3. **thought_quality** (Reasoning correctness)

# **Step-by-step evaluation:**

# For each Thought in the trajectory:
# 1. Check if reasoning is logical and correct based on the Instruction and previous Observations.
# 2. Award **+1** for correct reasoning.
# 3. Award **0** for flawed, hallucinated, or nonsensical reasoning.

# **Normalization:**
# thought_quality_score = (Correct Thoughts) / (Total Thoughts in Trajectory)


# **Examples:**
# - Correct (+1): "The product was created successfully. Now I need to retrieve reviews."
# - Incorrect (0): Agent claims success when Observation clearly shows an ERROR.

# ---

# #### 4. **tool_execution** (Successful tool execution)

# **Step-by-step evaluation:**

# For each tool call in the trajectory:
# 1. Check the Observation output.
# 2. Award **+1** if execution was successful (no errors, valid return data).
# 3. Award **0** if execution failed (error messages in Observation or empty/invalid return).

# **Normalization:**
# tool_execution_score = (Successful Executions) / (Total Tool Calls in Trajectory)


# ---

# #### 5. **final_success** (Overall task completion)

# **Holistic evaluation:** Did the agent complete the task requirements?

# **Scoring:**
# - 1.0: All required operations completed successfully with correct Final Answer.
# - 0.5: Partial completion (some operations done, but critical parts missing).
# - 0.0: Task not completed (no tools called OR all tools failed).

# ---

# ### SPECIAL CASES

# 1. **Agent calls 0 tools:**
#    - `tool_selection`: 0.0
#    - `tool_execution`: 0.0
#    - `final_success`: 0.0
#    - `format_compliance`: Evaluate based on text structure.
#    - `thought_quality`: Evaluate based on text reasoning.

# 2. **Premature Final Answer:**
#    - If agent outputs Final Answer immediately without calling required tools:
#    - `final_success`: 0.0
#    - `tool_selection`: 0.0

# ---

# ### OUTPUT FORMAT

# First, provide step-by-step analysis in `<step_by_step_analysis>...</step_by_step_analysis>` tags:

# <step_by_step_analysis>
# Turn 0:
# Thought: [Correct/Incorrect] - [Reason]
# Tool: [Selected Tool] - [Correct/Incorrect]
# Execution: [Success/Fail]

# Turn 1:
# ...

# Summary:

# Valid Format Turns: X/Total

# Correct Tool Selections: Y/Total Tools

# Correct Thoughts: Z/Total Thoughts

# Successful Executions: A/Total Tools
# </step_by_step_analysis>

# text

# Then output ONLY this JSON (no code fences, no extra keys):

# {{
#   "format_compliance": 0.0,
#   "tool_selection": 0.0,
#   "thought_quality": 0.0,
#   "tool_execution": 0.0,
#   "final_success": 0.0,
#   "critique": "Brief summary of the evaluation."
# }}
# """


# # =========================================================================
# # 3. HELPER FUNCTIONS
# # =========================================================================
# def serialize_trajectory(trajectory: Any) -> str:
#     """Converts the trajectory object into a readable text string."""
#     text_log = []
#     for i, seg in enumerate(trajectory.segments):
#         header = f"[STEP {i+1} - {seg.segment_type.upper()}]"
#         content = seg.text.strip()
#         text_log.append(f"{header}\n{content}")
#     return "\n\n".join(text_log)

# def get_gold_references(gt: Dict) -> Dict[str, str]:
#     """Safely extracts and formats gold references from the task dict."""
#     gold_chain = gt.get('gold_chain_of_thought', [])
#     if isinstance(gold_chain, list):
#         gold_chain = "\n".join([f"- {step}" for step in gold_chain])
        
#     gold_steps = gt.get('gold_step_outputs', [])
#     if isinstance(gold_steps, list):
#         gold_steps = json.dumps(gold_steps, indent=2)
        
#     return {
#         "chain": gold_chain or "Not provided.",
#         "steps": gold_steps or "Not provided."
#     }

# # =========================================================================
# # 4. MAIN REWARD FUNCTION
# # =========================================================================
# def create_ground_truth_reward_function(tasks: List[Dict]):
#     """
#     Creates the reward function with integrated GPTCaller.
#     Does NOT require passing llm_client as an argument.
#     """
    
#     # Initialize the Caller ONCE
#     # Ensure env vars AZURE_CHAT_API_KEY and AZURE_CHAT_ENDPOINT are set
#     # gpt_caller = LocalQwenCaller(model_name="/home/fripl/vharsh/research/models/models/Qwen-8b",api_base="http://localhost:8001/v1", api_key="judge")
#     gpt_caller = GPTCaller()
#     # Fast lookup
#     task_map = {t['id']: t for t in tasks}

#     def llm_reward_function(task_id: str, trajectory: Any) -> float:
#         gt = task_map.get(task_id)
#         logger.info(f"Evaluating reward for Task ID: {task_id}")
#         logger.info(f"Evaluating task {gt.get('user', 'No instruction found')}")
        
#         if not gt: return 0.0

#         # A. Serialize Trajectory
#         traj_text = serialize_trajectory(trajectory)
        
#         # B. Prepare References
#         refs = get_gold_references(gt)

#         # C. Construct Prompt
#         prompt = JUDGE_SYSTEM_PROMPT.format(
#             instruction=gt.get('user', "No instruction provided."),
#             success_criteria=gt.get('success_criteria', ["Solve the task correctly."]),
#             gold_chain_of_thought=refs['chain'],
#             gold_step_outputs=refs['steps'],
#             gold_final_output=gt.get('gold_final_output', "Not provided."),
#             agent_trajectory=traj_text
#         )

#         # D. Call Judge (Synchronous)
#         try:
#             # We use sync_call because typical RL loops are not async
#             # Use temperature=0.0 for deterministic grading
#             scores = gpt_caller.sync_call(
#                 prompt=prompt, 
#                 response_format="json", 
#                 temperature=0.0
#             )
#             # logger.info(f"prompt to judge: {prompt}")
#         except Exception as e:
#             logger.error(f"Judge GPT Call Failed completely: {e}")
#             return -0.1 # Penalty for infrastructure failure
#         logger.info(f"Judge Scores for Task {task_id}: {scores}")
#         # E. Normalize Scores (MO-GRPO Weights)
#         # Weights derived from ARTIST & DeepSeekMath
#         weights = {
#             "thought_quality": 0.1,    # Reasoning (Reduced slightly)
#             "tool_selection": 0.25,    # Action: Picking the right tool
#             "tool_execution": 0.15,    # NEW: Using the tool successfully (syntax/params)
#             "format_compliance": 0.1,  # Constraints
#             "final_success": 0.4       # Outcome (Still the most important)
#         }

#         # Calculate weighted score (your existing loop is perfect)
#         weighted_score = 0.0
#         for key, w in weights.items():
#             val = float(scores.get(key, 0.0))
#             val = max(0.0, min(1.0, val)) # Clip 0-1
#             weighted_score += val * w

#         # F. Critical Failure Gate
#         # If format is totally broken (< 0.2), apply hard penalty override
#         if float(scores.get("format_compliance", 1.0)) < 0.2:
#              return 0.0

#         # G. Return Reward [-1.0, 1.0]
#         final_reward = weighted_score  # Already 0-1 from weighted sum
#         return max(0.0, min(1.0, final_reward))

#     return llm_reward_function

# Example usage in train_enterprise.py:
"""
# In main() function, replace load_enterprise_tasks with:

from enterprise_dataset_loader import load_enterprise_tasks_v2, create_ground_truth_reward_function

# Load tasks
train_dataset = load_enterprise_tasks_v2(
    DATASET_PATH,
    max_tasks=100,           # Limit for testing
    difficulty_filter="EASY", # Start with easy tasks
    # domain_filter="HR",     # Or filter by domain
    # min_steps=1,
    # max_steps=5
)

# Use ground truth reward instead of LLM judge
reward_fn = create_ground_truth_reward_function(train_dataset)
"""



JUDGE_SYSTEM_PROMPT_V2 = """
You are a strict evaluator for an AI Agent operating in a tool-using environment.

IMPORTANT:
- The agent MUST use tools when the gold solution uses tools.
- Do NOT reward fluent text; reward correct tool calls + correct arguments + correct propagation of entities.
- You will NOT receive tool outputs/observations (they may be too long). Use the structured execution trace instead.

---

### TASK DATA

Instruction:
{instruction}

REFERENCE – Gold Steps (expected tool calls & inputs; no observations):
{gold_step_outputs}

REFERENCE – Expected Final Answer:
{gold_final_output}

---

### AGENT DATA (candidate)

AGENT TRAJECTORY (NO observations/tool outputs):
{agent_trajectory}

EXECUTION TRACE (authoritative, from environment; no tool outputs):
{execution_trace}

---

### DEFINITIONS

- "Turn" = one Thought + either (Action + Action Input) OR Final Answer.
- "Entity" = IDs, names, identifiers, or values that must be carried (e.g., product IDs, ticket IDs, email IDs).
- If the agent invents IDs/entities not in the reference step (or inconsistent with its own Action Input), that is a failure.

---

### SCORING (0.0 to 1.0 each)

You must score 4 dimensions independently:

1) format_compliance
- For each turn, check the structure is valid:
  Thought: ...
  Action: <tool_name>
  Action Input: <json>
  (OR) Final Answer: ...
- Score = valid_turns / total_turns

2) tool_args_match (PRIMARY HARD SIGNAL - WITH PARTIAL CREDIT)
Core Matching Logic
Compare the agent's Action + Action Input sequence against the REFERENCE gold steps.

Score is based on progressive argument matching, not binary pass/fail.

Argument Matching Rubric
For each required reference step:

Condition	Score
Exact match: Tool name + all critical argument keys and values match	1.0
Partial match: Tool name correct, but only N/M argument keys/values match	0.5
Tool correct, all args wrong	0.2
Wrong tool entirely	0.0
Tool not called	0.0
Critical argument keys are defined per tool in the reference gold steps (e.g., empid, productid, reponame are critical; verbose, format are not).

Normalization Rule:
tool_args_match_score = sum(step_scores) / required_steps_total

where:
  - sum(step_scores) = sum of individual step scores (0.0, 0.2, 0.5, 1.0 per step)
  - required_steps_total = number of gold reference steps

Extra irrelevant tool calls beyond the required set:
  - If extra call matches a required step already correctly done: +0 (no double credit)
  - If extra call is a "verification" step and helps grounding: +0.0 (neutral)
  - If extra call is a repeated failed attempt (see loop penalty below): heavily penalized via loop_penalty

3) entity_grounding (STATE-LIKE, SOFT BUT IMPORTANT)
For each turn with an Action:
- Extract the set of entities mentioned in the agent Thought.
- Extract the set of entities used in Action Input values (and any IDs implied by keys like *_id).
- Score the turn high only if:
  (a) Thought entities are consistent with Action/Input entities (no contradictions),
  (b) Entities match the reference step’s entities (do not invent extra IDs),
  (c) Entities are propagated correctly across turns (if an ID appears in later reference steps, the agent should use that same ID).
- Penalize:
  - Mentioning an ID in Thought but using a different ID in Action Input.
  - Introducing extra IDs not present in the reference step.
- Normalize across turns with Actions.

4) final_success
- 1.0 Either the final answer/trajectory workflow matches the reference expected final answer/trajectory workflow or the task is completed(allow minor wording differences if exact string match is not possible).
- 0.5 if partially correct (some required operations appear correct but final answer wrong/incomplete).
- 0.0 if incorrect or missing.

---

### CRITICAL GATES (apply these strictly)

- If required_steps_total > 0 AND the execution trace shows 0 tool calls executed:
  Set ALL scores to 0.0.
- If format_compliance is very low (<0.2), overall behavior is invalid; scores should be near 0.

---

### OUTPUT FORMAT

Return ONLY JSON (no markdown, no code fences, no extra keys):

{{
  "format_compliance": 0.0,
  "tool_args_match": 0.0,
  "entity_grounding": 0.0,
  "final_success": 0.0,
  "critique": "brief reason"
}}
"""
_CODE_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    flags=re.DOTALL | re.IGNORECASE,
)

def parse_bedrock_judge_output(raw: Any) -> Dict[str, Any]:
    """
    Parse Bedrock/LangChain judge output that may look like:

    ```json
    { ... }
    ```

    or may include extra text around the JSON.

    Returns a dict. Raises ValueError on failure.
    """
    # 1) Already a dict
    if isinstance(raw, dict):
        return raw

    # 2) Normalize to string
    s = raw if isinstance(raw, str) else str(raw)
    s = s.strip()

    # 3) Try extracting JSON from ```json ... ``` fence
    m = _CODE_FENCE_RE.search(s)
    if m:
        json_str = m.group(1).strip()
        return json.loads(json_str)

    # 4) Fallback: extract between first '{' and last '}'
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        json_str = s[l:r + 1].strip()
        return json.loads(json_str)

    raise ValueError(f"Could not find JSON object in judge output. Head: {s[:200]!r}")
def serialize_trajectory_no_observations(trajectory) -> str:
    text_log = []
    turn_idx = 0

    # Only keep assistant-generated controllable segments
    keep_types = {"thought", "action", "final_answer", "malformed_output"}

    for seg in trajectory.segments:
        if seg.segment_type not in keep_types:
            continue
        header = f"[TURN {turn_idx}] {seg.segment_type.upper()}"
        text_log.append(f"{header}\n{seg.text.strip()}")
        # Increment turn when we see an action or final answer (rough turn boundary)
        if seg.segment_type in ("action", "final_answer"):
            turn_idx += 1

    return "\n\n".join(text_log)


def serialize_execution_trace(trajectory) -> str:
    calls = getattr(trajectory, "executed_tool_calls", []) or []
    if not calls:
        return "[]"
    # Keep it compact: no tool outputs
    rows = []
    for i, c in enumerate(calls):
        rows.append({
            "i": i,
            "tool_name": c.tool_name,
            "args": c.args,
            "status": c.status,
        })
    return json.dumps(rows, indent=2)

def create_ground_truth_reward_function(tasks):
    # gpt_caller = GPTCaller()
    gpt_caller = load_claude_model()
    task_map = {t["id"]: t for t in tasks}

    weights = {
        "tool_args_match": 0.25,      # PRIMARY
        "final_success": 0.55,
        "tool_execution": 0.15,       # deterministic, from executed_tool_calls.status
        "format_compliance": 0.03,
        "entity_grounding": 0.02,     # keep small; easy to game if overweighted
    }

    def gold_required_steps_total(gt):
        steps = gt.get("gold_step_outputs", [])
        return len(steps) if isinstance(steps, list) else 1

    def compute_tool_execution(trajectory):
        calls = getattr(trajectory, "executed_tool_calls", []) or []
        if not calls:
            return 0.0
        ok = 0
        for c in calls:
            s = str(getattr(c, "status", "")).lower()
            if "success" in s:
                ok += 1
        return ok / len(calls)

    def llm_reward_function(task_id, trajectory):
        gt = task_map.get(task_id)
        if not gt:
            return 0.0

        required_steps = gold_required_steps_total(gt)
        executed_calls = getattr(trajectory, "executed_tool_calls", []) or []

        # HARD GATE: gold requires tools but none executed
        if required_steps > 0 and len(executed_calls) == 0:
            return 0.0

        agent_traj_text = serialize_trajectory_no_observations(trajectory)
        exec_trace = serialize_execution_trace(trajectory)

        gold_steps = gt.get("gold_step_outputs", "Not provided.")
        if isinstance(gold_steps, list):
            gold_steps = json.dumps(gold_steps, indent=2)

        prompt = JUDGE_SYSTEM_PROMPT_V2.format(
            instruction=gt.get("user", "No instruction provided."),
            gold_step_outputs=gold_steps,
            gold_final_output=gt.get("gold_final_output", "Not provided."),
            agent_trajectory=agent_traj_text,
            execution_trace=exec_trace,
        )

        try:
            scores = gpt_caller.invoke(prompt).content
            scores = parse_bedrock_judge_output(scores)
        except Exception as e:
            logger.error(f"Judge GPT Call Failed for Task {task_id} : {e} ")
            return 0.0
        logger.info(f"Judge Scores for Task {task_id}: {scores}")
        # Deterministic component
        tool_execution = compute_tool_execution(trajectory)

        # Clip judge scores
        def clip01(x):
            try:
                x = float(x)
            except Exception:
                x = 0.0
            return max(0.0, min(1.0, x))

        format_c = clip01(scores.get("format_compliance", 0.0))
        tool_args = clip01(scores.get("tool_args_match", 0.0))
        entity_g = clip01(scores.get("entity_grounding", 0.0))
        final_s  = clip01(scores.get("final_success", 0.0))

        # Optional additional gates
        if format_c < 0.2:
            return 0.0
        if tool_execution == 0.0:
            # if tools executed but all failed, avoid rewarding “nice plans”
            # (tune if you want recovery behaviors)
            return 0.0

        total = (
            weights["tool_args_match"] * tool_args +
            weights["final_success"] * final_s +
            weights["tool_execution"] * tool_execution +
            weights["format_compliance"] * format_c +
            weights["entity_grounding"] * entity_g
        )
        return max(0.0, min(1.0, total))

    return llm_reward_function
