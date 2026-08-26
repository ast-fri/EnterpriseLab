# Agentic GRPO for EnterpriseBench

This directory contains a trajectory-based reinforcement-learning pipeline for training an agent to complete EnterpriseBench workflows with tool calls. The dataset loader normalizes source records into the task contract consumed by the rollout and trainer code, while `reward_new.py` scores generated trajectories against ordered, checkpoint-level supervision.

## What is in this directory

| File | Purpose |
| --- | --- |
| `train_enterprise.py` | Main training entry point and environment-driven configuration. |
| `enterprise_dataset_loader.py` | Loads and normalizes checkpoint tasks or message-based trajectories. |
| `reward_new.py` | Computes checkpoint-grounded tool, state, completion, and format rewards. |
| `enterprise_tool_environment.py` | Runs EnterpriseBench tools in an isolated copy of the environment. |
| `rollout_manager.py` | Generates multi-turn tool-use trajectories. |
| `grpo_trainer.py` / `dapo_trainer.py` / `entropy_trainer.py` | Policy optimization implementations. |
| `data_structures.py` | Shared trajectory, segment, and tool-execution types. |

## Supported dataset formats

The input file must be valid JSON. Its top level may be either:

```json
[
  { "task_id": "task-001", "instruction": "..." }
]
```

or:

```json
{
  "data": [
    { "task_id": "task-001", "instruction": "..." }
  ]
}
```

`enterprise_dataset_loader.py` recognizes two record formats. The checkpoint format is the format required when training with `REWARD_MODE=checkpoint` and `reward_new.py`.

### 1. Checkpoint format — recommended

Each record describes an instruction and the ordered tool checkpoints needed to complete it.

```json
{
  "task_id": "crm-update-001",
  "instruction": "Find customer C-100 and update their support ticket T-42 to resolved.",
  "domain": "CRM",
  "difficulty": "MEDIUM",
  "prerequisite_context": {
    "tenant": "demo"
  },
  "chain_of_thought": [
    {
      "step": 1,
      "subgoal": "Read the ticket and retain its customer identifier."
    },
    {
      "step": 2,
      "subgoal": "Resolve the requested ticket."
    }
  ],
  "required_tools": [
    "get_it_ticket",
    "resolve_ticket"
  ],
  "success_criteria": [
    "Ticket T-42 is resolved"
  ],
  "reward_checkpoints": [
    {
      "step": 1,
      "tool": "get_it_ticket",
      "input_bindings": {
        "ticket_id": "T-42"
      },
      "must_succeed": true,
      "expected_output_parsed": {
        "ticket_id": "T-42",
        "customer_id": "C-100"
      },
      "state_tracking": {
        "state_to_retain": [
          {
            "field": "customer_id",
            "field_path": "customer_id",
            "entity_type": "customer"
          }
        ]
      }
    },
    {
      "step": 2,
      "api_id": "resolve_ticket",
      "input_bindings": {
        "ticket_id": "T-42"
      },
      "must_succeed": true,
      "state_tracking": {
        "state_must_remember": [
          {
            "field": "customer_id",
            "value": "C-100",
            "entity_type": "customer",
            "llm_prompt": "The customer identifier from the earlier lookup must remain available."
          }
        ]
      }
    }
  ],
  "ground_truth": {
    "expected_tool_sequence": [
      "get_it_ticket",
      "resolve_ticket"
    ],
    "low_level_subgoals": [
      "Read ticket T-42",
      "Resolve ticket T-42"
    ]
  },
  "meta": {
    "source": "generated"
  }
}
```

#### Checkpoint record fields

| Field | Required | Type | Behavior |
| --- | --- | --- | --- |
| `instruction` | Yes | string | Becomes the agent's user prompt. Empty instructions are skipped. |
| `task_id` or `id` | Recommended | string | Used to join tasks to trajectories. A generated ID is used if omitted. |
| `reward_checkpoints` | Yes for checkpoint reward | array of objects | Defines the ordered gold tool trajectory consumed by `reward_new.py`. |
| `chain_of_thought` | No | array of objects | Supplies a checkpoint rationale by matching integer `step` values. It is not directly scored by `reward_new.py`. |
| `required_tools` | No | array of strings | Used as metadata and for tool filtering. If omitted, it is derived from valid checkpoints. |
| `success_criteria` | No | array of strings | Preserved as metadata and used to build a descriptive `gold_final_output`; checkpoint completion reward does not compare final-answer text. |
| `domain` | No | string | Supports exact-match `domain_filter`. |
| `difficulty` | No | string | Supports exact-match `difficulty_filter`. |
| `prerequisite_context` | No | object or JSON value | Preserved in the normalized task. |
| `ground_truth` | No | object | Preserved. `expected_tool_sequence` and `low_level_subgoals` are used to construct descriptive normalized metadata, not the checkpoint reward sequence. |
| `meta` | No | object | Preserved without interpretation. |

#### Reward checkpoint fields

| Field | Required | Type | Behavior |
| --- | --- | --- | --- |
| `tool` or `api_id` | Yes | string | Tool name. Checkpoints without either field are ignored. |
| `input_bindings` | Recommended | object | Gold tool arguments. These are used in tool and completion matching. |
| `step` | Recommended | integer | Identifies order and links the checkpoint to `chain_of_thought`. Array order remains the reward order. |
| `state_tracking` | No | object | Defines state items for the LLM-based state-retention reward. |
| `expected_output_parsed` | Required for `state_to_retain` values | any JSON value | Source object from which a dot-separated `field_path` is read. |
| `must_succeed` | No | boolean | Preserved in normalized gold-step metadata; it is not currently used by `reward_new.py`. |
| `operation` | No | any JSON value | Preserved in normalized gold-step metadata; it is not currently scored. |
| `expected_output_fields`, `success_validation`, `expected_effects` | No | JSON values | Used by the loader as fallback expected-output metadata when `expected_output_parsed` is absent; they are not directly scored by `reward_new.py`. |

`state_tracking` supports two lists:

- `state_must_remember`: each item should contain `field`; it may also contain a literal `value`, `entity_type`, and `llm_prompt`.
- `state_to_retain`: each item should contain `field` and a dot-separated `field_path`. The value is extracted from the same checkpoint's `expected_output_parsed`. Only object keys are traversed; array indexing is not supported.

### 2. Message trajectory format — loader-compatible

The loader also accepts OpenAI-style message logs:

```json
{
  "id": "chatlog-001",
  "domain": "CRM",
  "difficulty": "EASY",
  "messages": [
    {
      "role": "system",
      "content": "You are an enterprise assistant."
    },
    {
      "role": "user",
      "content": "Look up product P-10."
    },
    {
      "role": "assistant",
      "content": "I will retrieve the product."
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call-1",
          "type": "function",
          "function": {
            "name": "get_product",
            "arguments": {
              "product_id": "P-10"
            }
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call-1",
      "name": "get_product",
      "content": "{\"product_id\": \"P-10\", \"name\": \"Example\"}"
    },
    {
      "role": "assistant",
      "content": "Product P-10 is Example."
    }
  ],
  "timestamp": "2026-01-01T00:00:00Z"
}
```

The loader:

- joins all non-empty `user` messages to form the instruction;
- excludes `system` messages from the instruction;
- converts assistant `tool_calls` and immediately following `tool` messages into normalized gold steps;
- expects `function.arguments` to already be a JSON object, not a JSON-encoded string;
- preserves assistant and tool messages in `gold_messages`; and
- treats the most recent assistant text without tool calls as the final answer.

> **Important:** message records do not produce `reward_checkpoints`. Consequently, they are not sufficient for `reward_new.py`: checkpoint tool and completion rewards will have no gold sequence. Convert message logs to checkpoint records before using `REWARD_MODE=checkpoint`, or use a reward mode designed for the normalized message-derived fields.

The current `TaskGenerationPipeline/environments/EnterpriseBench/Task_Generation/tasks.json` in this checkout is message-based. A loader smoke test derives tool steps from it but produces no `reward_checkpoints`, so that file must be converted before it can provide checkpoint-grounded supervision to `reward_new.py`.

## Loader output contract

Both formats are normalized to dictionaries with these commonly consumed fields:

```python
{
    "id": str,
    "user": str,
    "gold_chain_of_thought": list,
    "gold_step_outputs": list,
    "gold_final_output": str | None,
    "gold_messages": list,
    "required_tools": list[str],
    "domain": str | None,
    "difficulty": str | None,
    "num_steps": int,
    "success_criteria": list,
}
```

Checkpoint records additionally retain `reward_checkpoints`, `ground_truth`, `meta`, and `prerequisite_context`.

The loader can shuffle before filtering and limiting. Its parameters are:

```python
load_enterprise_tasks_v2(
    path,
    max_tasks=None,
    difficulty_filter=None,
    domain_filter=None,
    min_steps=None,
    max_steps=None,
    shuffle=True,
    seed=None,
)
```

`difficulty_filter` and `domain_filter` use case-sensitive exact matches. `min_steps` and `max_steps` apply to the number of valid tool steps. A positive `max_tasks` limit is applied after shuffle and filtering; `None`, `0`, and negative values do not limit the result.

## Checkpoint reward

Select this implementation with:

```bash
export REWARD_MODE=checkpoint
```

The total reward ranges from `0.0` to `3.0`:

| Component | Range | Definition |
| --- | ---: | --- |
| Tool reward | 0.0–0.5 | Fraction of the ordered gold sequence matched by longest common subsequence (LCS), using tool names and required arguments. |
| State reward | 0.0–0.5 | Fraction of required state items retained on LCS-aligned steps, judged by an OpenAI-compatible chat model. Returns zero when no state items are defined. |
| Final reward | 0.0 or 1.0 | Awarded when the complete gold sequence is covered in order using tool names and required arguments. This is trajectory completion, not final-answer text matching. |
| Format reward | 0.0 or 1.0 | Awarded only when every trainable assistant segment follows the ARTIST XML format described below. |

The LCS allows extra generated calls between correct calls, but gold calls must occur in order. Tool output, execution status, `must_succeed`, and final-answer wording are not currently part of checkpoint matching.

### Required-argument matching

`reward_new.py` loads tool definitions from:

```text
/home/fripl/vharsh/research/EnterpriseBench/Task_Generation/utils/tools.json
```

Override this machine-specific default with:

```bash
export ENTERPRISE_TOOLS_SCHEMA_PATH=/absolute/path/to/Task_Generation/utils/tools.json
```

For tools found in that schema, only arguments whose definitions contain `"required": true` participate in the tool and final LCS signatures. Extra optional arguments do not prevent a match. For unknown tools—or when the schema cannot be loaded—all arguments must match exactly after recursive JSON normalization.

The reward code expects tool entries with this shape:

```json
{
  "name": "get_it_ticket",
  "args_schema": {
    "ticket_id": {
      "type": "string",
      "required": true
    },
    "include_history": {
      "type": "boolean",
      "required": false
    }
  }
}
```

The `TaskGenerationPipeline/environments/EnterpriseBench/Task_Generation/utils/tools.json` file currently present in this EnterpriseLab checkout instead uses an `arguments` list. That file is valid for its own tool environment, but `reward_new.py` does not infer required arguments from that legacy representation. With it, checkpoint reward uses exact matching over all `input_bindings`. Use an `args_schema`-based file if required-only matching is intended.

### State judge

State scoring requires an OpenAI-compatible chat endpoint and the `langchain-openai` and `langchain-core` packages. Configure it with:

```bash
export JUDGE_API_BASE=http://host:port/v1
export JUDGE_API_KEY=EMPTY
export JUDGE_MODEL=/path/or/model-name
export REWARD_CACHE_PATH=/absolute/path/to/reward_cache.jsonl
```

The judge receives the task instruction, required state items, the aligned assistant step text, and its tool call. Responses are expected to be JSON objects containing `retained_count` and `reasoning`. Cache entries are versioned and keyed by the task, checkpoints, generated trajectory, termination reason, and judge model.

### Required model-output format

Each trainable `thought_and_action` segment must contain exactly one `<reasoning>` block and exactly one of `<tool>` or `<answer>`:

```xml
<reasoning>I need to retrieve the requested ticket.</reasoning>
<tool>{"name":"get_it_ticket","args":{"ticket_id":"T-42"}}</tool>
```

or:

```xml
<reasoning>The requested workflow is complete.</reasoning>
<answer>Ticket T-42 has been resolved.</answer>
```

A segment containing both action types, neither action type, repeated tags, or malformed tool-call termination receives zero format reward.

## Quick dataset validation

Run the loader without starting model training:

```bash
cd /home/fripl/vharsh/EnterpriseLab/Train/Agentic_GRPO

DATASET_PATH=/absolute/path/to/tasks.json python -c '
import os
from enterprise_dataset_loader import load_enterprise_tasks_v2

tasks = load_enterprise_tasks_v2(
    os.environ["DATASET_PATH"],
    shuffle=False,
)
print(f"loaded={len(tasks)}")
for task in tasks[:3]:
    print(task["id"], task["num_steps"], task["required_tools"])
'
```

For checkpoint-reward training, also verify that every loaded task has at least one valid checkpoint:

```bash
DATASET_PATH=/absolute/path/to/tasks.json python -c '
import os
from enterprise_dataset_loader import load_enterprise_tasks_v2

tasks = load_enterprise_tasks_v2(os.environ["DATASET_PATH"], shuffle=False)
invalid = [task["id"] for task in tasks if not task.get("reward_checkpoints")]
if invalid:
    raise SystemExit(f"tasks missing reward_checkpoints: {invalid[:20]}")
print(f"validated {len(tasks)} checkpoint tasks")
'
```

## Running training

At minimum, point the trainer at the model, checkpoint dataset, EnterpriseBench environment, tool schema, and judge:

```bash
cd /home/fripl/vharsh/EnterpriseLab/Train/Agentic_GRPO

export MODEL_SERIES=qwen3
export MODEL_NAME=/absolute/path/to/model
export DATASET_PATH=/absolute/path/to/tasks.json
export ENTERPRISEBENCH_ENV_ROOT=/absolute/path/to/EnterpriseBench
export ENTERPRISE_TOOLS_SCHEMA_PATH="$ENTERPRISEBENCH_ENV_ROOT/Task_Generation/utils/tools.json"
export CHECKPOINT_DIR=/absolute/path/to/output

export TRAIN_ENVIRONMENT=enterprise
export REWARD_MODE=checkpoint
export TRAINER_VARIANT=grpo
export PROMPT_MODE=artist

export JUDGE_API_BASE=http://host:port/v1
export JUDGE_API_KEY=EMPTY
export JUDGE_MODEL=/path/or/model-name
export REWARD_CACHE_PATH="$CHECKPOINT_DIR/reward_cache.jsonl"

export SHUFFLE_DATASET=true
export SHUFFLE_SEED=42

python train_enterprise.py --artist-prompt
```

Common environment variables include `GROUP_SIZE`, `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `MAX_TURNS`, `MAX_CONTEXT_LENGTH`, `MAX_NEW_TOKENS`, `TEMPERATURE`, `REF_MODEL_DEVICE`, `ENABLE_TOOL_FILTERING`, `NUM_RANDOM_TOOLS`, and `RESUME_FROM_CHECKPOINT`.

## Dependencies and checkout assumptions

The core training path uses Python, PyTorch, Transformers, PEFT, NumPy, tqdm, python-dotenv, and LangChain's OpenAI integration. A CUDA-capable setup is expected for practical training. The repository-level `requirements.txt` contains the pinned environment used by this project.

Several defaults in this code are absolute paths under `/home/fripl/vharsh/research`; override them for this checkout. In particular, set `MODEL_NAME`, `DATASET_PATH`, `CHECKPOINT_DIR`, `ENTERPRISEBENCH_ENV_ROOT`, `ENTERPRISE_TOOLS_SCHEMA_PATH`, and the judge variables explicitly.

This directory's current `train_enterprise.py` also imports trainer/environment variants that are not present in this directory (`agentflow_trainer`, `agentflow_rollout_manager`, `environment_belief_rl`, `rollout_manager_35`, `rollout_manager_cohere`, `reward`, and the emulation-system modules). Ensure those modules are available on `PYTHONPATH`, or remove/guard unused variant imports before launching the standalone entry point. The dataset-loader validation commands above do not require those training-only modules.

## Practical data-quality checks

Before a long run, check that:

- every task has a unique, stable `task_id`;
- every instruction is non-empty;
- `reward_checkpoints` are in execution order and each has `tool` or `api_id`;
- `input_bindings` use JSON objects and match the tool schema's argument names and value types;
- checkpoint tool names exist in both `tools.json` and the executable tool environment;
- `state_to_retain.field_path` resolves inside the same checkpoint's `expected_output_parsed`;
- state requirements are included only when the state must actually be carried through the workflow; and
- the judge endpoint is tested before training, because state-judge failures yield zero retained state.
