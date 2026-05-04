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


import json
import random
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def load_enterprise_tasks_v2(
    path: str,
    max_tasks: int = None,
    difficulty_filter: str = None,
    domain_filter: str = None,
    min_steps: int = None,
    max_steps: int = None
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

        logger.info("Tasks Shuffled")
        # random.shuffle(raw_tasks)
        logger.info(f"Loaded {len(raw_tasks)} raw tasks from {path}")

        formatted_tasks: List[Dict[str, Any]] = []

        for idx, item in enumerate(raw_tasks):
            messages = item.get("messages")
            if not isinstance(messages, list) or len(messages) == 0:
                continue

            # ---- Extract instruction (exclude system) ----
            user_texts = [m.get("content", "") for m in messages if m.get("role") == "user" and m.get("content")]
            instruction = "\n".join(user_texts).strip()
            if not instruction:
                # No user instruction => skip
                continue

            # ---- Parse gold steps from tool_calls ----
            gold_steps = []
            required_tools = []

            last_assistant_text = None  # rationale carrier (assistant content before tool call)
            final_answer = None

            # We'll walk in order and pair assistant.tool_calls with subsequent tool messages
            i = 0
            while i < len(messages):
                m = messages[i]
                role = m.get("role")

                if role == "assistant":
                    # If this assistant message has plain content, treat as rationale or final answer candidate
                    if m.get("content") and not m.get("tool_calls"):
                        last_assistant_text = str(m["content"]).strip()
                        final_answer = last_assistant_text  # will be overwritten if more assistant content appears later

                    # Tool call block
                    if m.get("tool_calls"):
                        tool_calls = m.get("tool_calls", [])
                        # collect tool outputs right after (there may be multiple tool messages)
                        tool_outputs = []
                        j = i + 1
                        while j < len(messages) and messages[j].get("role") == "tool":
                            tool_outputs.append(messages[j])
                            j += 1

                        # Pair in order (best-effort). If names mismatch, still pair by position.
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

                            gold_steps.append({
                                "step": len(gold_steps) + 1,
                                "rationale": last_assistant_text or "",
                                "tool": tool_name,
                                "inputs": args if isinstance(args, dict) else {},
                                "expected_output": expected_output
                            })

                        # jump past the consumed tool messages
                        i = j
                        continue

                i += 1

            required_tools = sorted(set(required_tools))
            num_steps = len(gold_steps)

            # ---- Apply filters (optional; most new logs won't have these fields) ----
            difficulty = item.get("difficulty")  # may be absent
            domain = item.get("domain")          # may be absent

            if difficulty_filter and difficulty != difficulty_filter:
                continue
            if domain_filter and domain != domain_filter:
                continue
            if min_steps is not None and num_steps < min_steps:
                continue
            if max_steps is not None and num_steps > max_steps:
                continue

            formatted_task = {
                "id": item.get("task_id") or item.get("id") or f"chatlog_{idx}",
                "user": instruction,

                # Gold references derived from messages
                "gold_chain_of_thought": [],  # optional; you can populate from assistant content if you want
                "gold_step_outputs": gold_steps,
                "gold_final_output": final_answer,
                "gold_messages": _extract_gold_messages(messages),

                # Metadata
                "required_tools": required_tools,
                "domain": domain,
                "difficulty": difficulty,
                "num_steps": num_steps,
                "success_criteria": item.get("success_criteria", []),

                # Optional: keep timestamp for debugging
                "timestamp": item.get("timestamp"),
            }

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
        model_id="",
       
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
