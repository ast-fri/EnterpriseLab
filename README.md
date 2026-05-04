# EnterpriseLab: Training and Evaluating LLM Agents in Enterprise Environments

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

EnterpriseLab is a full-stack framework for building, training, and evaluating tool-using LLM agents in realistic enterprise environments. It combines a local EnterpriseArena made of self-hosted business applications, MCP servers that expose those applications as tools, a task-generation pipeline, Agentic-GRPO training, and trajectory-based evaluation.

The intended end-to-end workflow is:

```text
EnterpriseArena apps + MCP servers
  -> TaskGenerationPipeline creates or refines enterprise tasks
  -> Train/Agentic_GRPO trains an agent on tool-use trajectories
  -> Evaluate/EnterpriseArena and Evaluate/MCP_eval generate and score rollouts
```

This README is written for reproducibility. Commands assume the repository root is `/EnterpriseLab`; adjust paths if you clone elsewhere.

## 0. Base Setup

### System requirements

- Linux machine or server with Docker and Docker Compose.
- Python 3.10 or newer.
- NVIDIA GPU for training and local/vLLM evaluation. CPU-only use is possible for documentation and lightweight inspection, but not practical for Agentic-GRPO training.
- Enough disk space for Docker images, model checkpoints, generated trajectories, and cached Hugging Face/vLLM artifacts.

### Clone and install Python dependencies

```bash
git clone <ANONYMIZED_REPOSITORY_URL>
cd /EnterpriseLab

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you are using a cluster image that already provides CUDA, PyTorch, vLLM, or FlashAttention, install dependencies in the way required by that cluster. The pinned `requirements.txt` is broad because it covers arena execution, task generation, training, and evaluation.

### Common environment variables

Several components use environment variables rather than command-line flags.

For task generation with Azure OpenAI:

```bash
export AZURE_CHAT_API_KEY="..."
export AZURE_CHAT_ENDPOINT="https://<resource>.openai.azure.com/"
```

For EnterpriseArena evaluation with vLLM:

```bash
export MODEL_MODE="vllm"
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_API_KEY="local-key"
```

For Azure-based evaluation instead of vLLM:

```bash
export MODEL_MODE="azure"
export AZURE_API_KEY="..."
export AZURE_API_ENDPOINT="https://<resource>.openai.azure.com/"
export AZURE_API_VERSION="2024-02-01"
```

For the LLM judge used by `Evaluate/MCP_eval`:

```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
```

### Important path note

Some scripts currently contain absolute paths from the original training environment, especially paths beginning with `/EnterprisePlatform`. Before reproducing on another machine, check and update these values:

- `TaskGenerationPipeline/main.py`: trajectory directory, task output directory, and pipeline output directory.
- `Train/Agentic_GRPO/train_enterprise.py`: `MODEL_NAME`, `MODEL_CACHE_DIR`, `CHECKPOINT_DIR`, `DATASET_PATH`, and `JUDGE_MODEL`.
- `Train/Agentic_GRPO/enterprise_tool_environment.py`: the `sys.path.insert(...)` path to `TaskGenerationPipeline/environments/EnterpriseBench` and the workspace base path.

You can either edit those constants to match your checkout or mount/symlink your checkout to the expected absolute paths.

## 1. EnterpriseArena

EnterpriseArena is the execution environment. It has two layers:

- Application containers in `Arena/apps/`.
- MCP servers in `Arena/MCP_servers/` that expose those applications as agent tools.

Start the application layer first, create or verify credentials inside each application, then start the corresponding MCP layer.

### 1.1 Application services

Application service definitions live under `Arena/apps/`.

| Application | Directory | Default access | Notes |
| --- | --- | --- | --- |
| GitLab | `Arena/apps/gitlab` | `http://localhost:8080` | SSH is mapped to port `2222`. |
| Plane | `Arena/apps/plane` | `http://localhost:3001` | Uses `plane.env`; workspace slug is expected by the Plane MCP server. |
| ownCloud | `Arena/apps/owncloud` | `http://localhost:3001` by default | This conflicts with Plane's default port. Change one port before running both together. |
| Rocket.Chat | `Arena/apps/rocketchat` | `http://localhost:3000` | Requires user ID and auth token for the MCP server. |
| Dolibarr | `Arena/apps/dolibarr` | `http://localhost:8082` | Default admin credentials are `admin` / `admin`. |
| Frappe HRMS | `Arena/apps/frappe` | See app README | This checkout contains setup notes, not a compose file. |
| Zammad | `Arena/apps/zammad` | See app README | This checkout contains setup notes, not a compose file. |

To start every application with a compose file:

```bash
cd /EnterpriseLab/Arena/apps
chmod +x start_all_servers.sh
./start_all_servers.sh start
./start_all_servers.sh status
```

To stop them:

```bash
cd /EnterpriseLab/Arena/apps
./start_all_servers.sh stop
```

To start one application, run Docker Compose from that application directory. For example:

```bash
cd /EnterpriseLab/Arena/apps/gitlab
docker compose up -d
```

Plane requires its env file:

```bash
cd /EnterpriseLab/Arena/apps/plane
docker compose --env-file plane.env up -d
```

After startup, check containers:

```bash
docker ps
```

### 1.2 MCP servers

MCP servers translate application APIs into streamable HTTP MCP tools. Their compose files live under `Arena/MCP_servers/`.

| MCP server | Directory | MCP URL used by configs | Application credentials to verify |
| --- | --- | --- | --- |
| GitLab | `Arena/MCP_servers/gitlab` | `http://localhost:8008/mcp` | `GITLAB_PERSONAL_ACCESS_TOKEN`, `GITLAB_API_URL` |
| Plane | `Arena/MCP_servers/plane` | `http://localhost:12023/mcp` | `PLANE_API_KEY`, `PLANE_API_HOST_URL`, `PLANE_WORKSPACE_SLUG` |
| ownCloud | `Arena/MCP_servers/owncloud` | `http://localhost:12001/mcp` | `OWNCLOUD_URL`, `OWNCLOUD_USERNAME`, `OWNCLOUD_PASSWORD` |
| Dolibarr | `Arena/MCP_servers/dolibarr` | `http://localhost:12000/mcp` | `DOLIBARR_BASE_URL`, `DOLIBARR_API_KEY` |
| Rocket.Chat | `Arena/MCP_servers/rocketchat` | `http://localhost:12004/mcp` | `ROCKETCHAT_URL`, `ROCKETCHAT_USER_ID`, `ROCKETCHAT_AUTH_TOKEN` |
| Zammad | `Arena/MCP_servers/zammad` | `http://localhost:12010/mcp` | `ZAMMAD_URL`, `ZAMMAD_TOKEN` |
| Frappe HRMS | `Arena/MCP_servers/frappe` | `http://localhost:12013/mcp` by default | `FRAPPE_URL`, `FRAPPE_API_KEY`, `FRAPPE_API_SECRET` |
| Aider | `Arena/MCP_servers/aider` | `http://localhost:12011/mcp` | Azure/Aider settings in `.env` |
| Playwright | `Arena/MCP_servers/playwright` | `http://localhost:12012/mcp` | Browser automation only |

Start all MCP servers with compose files:

```bash
cd /EnterpriseLab/Arena/MCP_servers
chmod +x start_all_servers.sh
./start_all_servers.sh
```

Stop them:

```bash
cd /EnterpriseLab/Arena/MCP_servers
./start_all_servers.sh stop
```

Start one MCP server:

```bash
cd /EnterpriseLab/Arena/MCP_servers/dolibarr
docker compose up -d
```

### 1.3 Align ports, URLs, and tokens

Before using agents against EnterpriseArena, make sure application ports and MCP environment variables point to the same place.

Examples to check:

- `Arena/MCP_servers/owncloud/docker-compose.yml` defaults to `OWNCLOUD_URL=http://host.docker.internal:8081`, while `Arena/apps/owncloud/docker-compose.yml` maps ownCloud to host port `3001`. Update one side so they match.
- `Evaluate/EnterpriseArena/mcp_config_http.json` currently lists Frappe HRMS on `12021`, while the Frappe MCP compose file exposes `12013`. Use the port you actually run.
- Replace hard-coded sample API tokens in MCP compose files with tokens generated by your local app instances.
- If an MCP server runs in Docker and needs to call an app on the host, use `host.docker.internal` where supported, or configure Docker networking explicitly.

The active MCP configs are:

- `TaskGenerationPipeline/mcp_config_http.json` for task generation.
- `Evaluate/EnterpriseArena/mcp_config_http.json` for evaluation.

Keep both files aligned with the MCP servers you started.

## 2. TaskGenerationPipeline

`TaskGenerationPipeline/` uses AutoQuest-style exploration to discover tool relationships, optionally explore tool trajectories, and synthesize new tasks from saved trajectories.

### 2.1 What the pipeline does

The pipeline in `TaskGenerationPipeline/main.py` has four conceptual phases:

1. Environment setup through `setup_environment.py`.
2. Tool dependency graph creation through `AutoQuest.edge_discovery`.
3. Optional intelligent exploration through `AutoQuest.intelligent_explorer`.
4. Chain-of-thought task synthesis and post-processing through `AutoQuest.task_synthesis.trajectory_processor`.

The default environment in `main.py` is `enterprise_arena`, which loads tools from `TaskGenerationPipeline/mcp_config_http.json`.

### 2.2 Configure task generation

Open `TaskGenerationPipeline/main.py` and check these values before running:

```python
explore = False
generate_task = True
```

- Set `explore = True` if you want the pipeline to collect new exploratory trajectories.
- Leave `generate_task = True` to synthesize tasks from saved trajectories.
- If `explore = False`, the script expects trajectories to already exist in `trajectories_dir`.

Also update the in-file config:

```python
config = {
    "model_name": "gpt-4o",
    "environments": [
        {
            "name": "enterprise_arena",
            "config": {
                "mcp_config_path": "./mcp_config_http.json"
            }
        },
    ],
    "pipeline": {
        "tool_batch_size": 10,
        "edge_batch_size": 100,
        "edge_confidence_threshold": 0.5,
        "exploration_budget": 2000,
        "max_trajectory_length": 100,
        "max_tasks_per_cluster": 10,
        "output_dir": "./arena",
    },
}
```

Then update the absolute directories near the task-synthesis section if needed:

```python
output_dir = "/EnterprisePlatform/TaskGenerationPipeline/arena/tasks"
trajectories_dir = "/EnterprisePlatform/TaskGenerationPipeline/arena/trajectories"
```

For a local checkout, these should usually point under:

```text
/EnterpriseLab/TaskGenerationPipeline/arena/tasks
/EnterpriseLab/TaskGenerationPipeline/arena/trajectories
```

Create the directories if they do not exist:

```bash
mkdir -p /EnterpriseLab/TaskGenerationPipeline/arena/tasks
mkdir -p /EnterpriseLab/TaskGenerationPipeline/arena/trajectories
```

### 2.3 Run task generation

Start EnterpriseArena apps and MCP servers first, then run:

```bash
cd /EnterpriseLab/TaskGenerationPipeline
source ../.venv/bin/activate

export AZURE_CHAT_API_KEY="..."
export AZURE_CHAT_ENDPOINT="https://<resource>.openai.azure.com/"

python main.py
```

Expected outputs include:

- `arena/initial_graph_*.json` or the fixed graph path configured in `main.py`.
- `arena/trajectories/*.json` if exploration is enabled.
- `arena/tasks/tasks_raw_<timestamp>.json`.
- `arena/tasks/tasks_final_<timestamp>.json`.
- `arena/tasks/cot_tasks_<timestamp>.json`.
- `arena/tasks/cot_tasks_detailed_<timestamp>.json`.

### 2.4 Generated task format

The task synthesis exporter writes JSON objects like:

```json
{
  "task_id": "enterprise_arena_...",
  "instruction": "User-facing task instruction",
  "prerequisite_context": "...",
  "chain_of_thought": [
    {
      "step": 1,
      "rationale": "...",
      "tool": "tool_name",
      "inputs": {},
      "expected_output": "..."
    }
  ],
  "required_tools": ["tool_name"],
  "success_criteria": ["..."],
  "domain": "enterprise_arena",
  "difficulty": "MEDIUM",
  "ground_truth": {},
  "meta": {}
}
```

This is a task-spec format. `Train/Agentic_GRPO` currently expects chat-style tool trajectories with a `messages` list. If you want to train Agentic-GRPO directly on generated task specs, convert them into the chat trajectory format described in the training section, or point Agentic-GRPO at an existing chat trajectory dataset such as `Data/enterprise_arena_gold.json`.

## 3. Train: Agentic-GRPO

The repository contains several training folders (`SFT`, `DPO`, `GRPO`, and `Agentic_GRPO`). The recommended reproduction path for agentic tool use is `Train/Agentic_GRPO`.

Agentic-GRPO trains a policy model by generating tool-use rollouts, scoring them with a reward function, and applying GRPO updates. The current implementation uses LoRA by default.

### 3.1 Key files

| File | Purpose |
| --- | --- |
| `train_enterprise.py` | Main Agentic-GRPO training entry point. |
| `enterprise_dataset_loader.py` | Loads chat-style tool trajectories and extracts user query, gold tool calls, tool outputs, and final answer. |
| `enterprise_tool_environment.py` | Wraps EnterpriseBench tools so rollouts can execute actions. |
| `rollout_manager.py` | Generates multi-turn agent trajectories. |
| `reward.py` | Computes the Agentic-GRPO reward using task outcome, state grounding, tool-use quality, and format validity. |
| `grpo_trainer.py` | Applies GRPO optimization. |
| `collator.py` | Builds trainable/non-trainable token masks over trajectory segments. |
| `vllm.bash` | Helper for serving a local reward judge. |
| `train.bash` | Cluster-oriented wrapper around `train_enterprise.py`. |

### 3.2 Training data format

`load_enterprise_tasks_v2(...)` expects a JSON array, or a JSON object with a `data` array. Each item must include `messages`.

Minimal example:

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "You are a ReAct agent. You MUST use tools to complete tasks."
      },
      {
        "role": "user",
        "content": "Locate the repository named keystone and inspect keystone/common/kvs.py."
      },
      {
        "role": "assistant",
        "content": "I will search for the repository first."
      },
      {
        "role": "assistant",
        "tool_calls": [
          {
            "type": "function",
            "function": {
              "name": "search_repositories",
              "arguments": {
                "search": "keystone"
              }
            }
          }
        ]
      },
      {
        "role": "tool",
        "name": "search_repositories",
        "content": "{\"count\": 1, \"items\": []}"
      },
      {
        "role": "assistant",
        "content": "ANSWER: I found the repository and inspected the requested file."
      }
    ]
  }
]
```

The loader derives:

- `user`: all user messages joined together.
- `gold_step_outputs`: assistant `tool_calls` paired with following `tool` messages.
- `required_tools`: unique tool names from the gold tool calls.
- `gold_final_output`: the last assistant text response without tool calls.
- `gold_messages`: assistant/tool messages preserved for the reward judge.

`Data/enterprise_arena_gold.json` follows this chat-style structure and is useful for a small reproduction run.

### 3.3 Configure paths and model settings

Edit the configuration block near the top of `Train/Agentic_GRPO/train_enterprise.py`.

Important values:

```python
MODEL_SERIES = os.getenv("MODEL_SERIES", "qwen3.5").strip().lower()
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_PATHS[...])
MODEL_CACHE_DIR = "/.cache"
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIRS[...])
DATASET_PATH = "/EnterprisePlatform/TaskGenerationPipeline/environments/EnterpriseBench/Task_Generation/tasks.json"
JUDGE_API_BASE = os.getenv("JUDGE_API_BASE") or "http://gpu04:8001/v1"
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY")
JUDGE_MODEL = "/models/models/Qwen3-32b"
```

For a local reproduction, set or edit:

```bash
cd /EnterpriseLab/Train/Agentic_GRPO
source ../../.venv/bin/activate

export MODEL_NAME="/path/to/policy/base/model"
export CHECKPOINT_DIR="/EnterpriseLab/Train/Agentic_GRPO/checkpoints/agentic-grpo"
export JUDGE_API_BASE="http://localhost:8001/v1"
export JUDGE_API_KEY="judge"
```

Then edit `DATASET_PATH` to one of:

```text
/EnterpriseLab/Data/enterprise_arena_gold.json
/EnterpriseLab/TaskGenerationPipeline/environments/EnterpriseBench/Task_Generation/tasks.json
/path/to/your/chat_style_training_trajectories.json
```

Also update `JUDGE_MODEL` in `train_enterprise.py` if your judge is not stored at `/models/models/Qwen3-32b`.

In `enterprise_tool_environment.py`, update:

```python
sys.path.insert(0, "/EnterprisePlatform/TaskGenerationPipeline/environments/EnterpriseBench")
```

to:

```python
sys.path.insert(0, "/EnterpriseLab/TaskGenerationPipeline/environments/EnterpriseBench")
```

or to the path where your EnterpriseBench tools are available.

### 3.4 Start the reward judge

The Agentic-GRPO reward path calls an OpenAI-compatible judge endpoint. A local vLLM server is the expected setup.

Direct vLLM example:

```bash
vllm serve /path/to/judge/model \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 128000 \
  --api-key judge
```

The helper script `Train/Agentic_GRPO/vllm.bash` can also be used after replacing:

```bash
MODEL_PATH="/path/to/your/model"
```

with the actual judge model path.

Confirm the judge API root used by training is:

```text
http://localhost:8001/v1
```

not the full `/chat/completions` URL.

### 3.5 Run Agentic-GRPO training

After the policy model path, dataset path, EnterpriseBench path, checkpoint directory, and judge server are configured:

```bash
cd /EnterpriseLab/Train/Agentic_GRPO
source ../../.venv/bin/activate
python train_enterprise.py
```

The script will:

1. Load the policy model and tokenizer.
2. Apply LoRA adapters.
3. Load a frozen reference model on CPU.
4. Load training tasks through `load_enterprise_tasks_v2`.
5. Create the Agentic-GRPO reward function.
6. Generate rollouts with `AgenticRolloutManager`.
7. Train with `GRPOTrainer`.
8. Save LoRA adapters and metrics.

Expected outputs:

```text
Train/Agentic_GRPO/enterprise_training.log
<CHECKPOINT_DIR>/checkpoint-*/
<CHECKPOINT_DIR>/final_model/
<CHECKPOINT_DIR>/training_metrics.json
Train/Agentic_GRPO/reward_cache.jsonl
```

By default, `final_model/` contains LoRA adapter weights, not a fully merged model.

### 3.6 Resume training

If a checkpoint contains `trainer_state.pt`, resume with:

```bash
export RESUME_FROM_CHECKPOINT="/path/to/checkpoint"
python train_enterprise.py
```

The script restores optimizer state, RNG state, epoch, batch index, and global step from the checkpoint.

### 3.7 Serve or merge the trained model

For evaluation, you need a model endpoint that `Evaluate/EnterpriseArena/graph.py` can call.

If you trained with LoRA, either serve the base model with LoRA support or merge the adapter first. The repository includes a generic LoRA merge utility:

```bash
cd /EnterpriseLab/Train/GRPO
source ../../.venv/bin/activate

python merge_model.py \
  --base_model_path /path/to/policy/base/model \
  --adapter_path /EnterpriseLab/Train/Agentic_GRPO/checkpoints/agentic-grpo/final_model \
  --output_path /path/to/merged_agentic_grpo_model
```

Then serve the merged model with vLLM:

```bash
vllm serve /path/to/merged_agentic_grpo_model \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --api-key local-key
```

## 4. Evaluate

Evaluation has two stages:

1. Generate predicted trajectories by running an agent in EnterpriseArena.
2. Compare predicted trajectories against gold trajectories with static tool matching and an LLM judge.

### 4.1 Configure the evaluation agent

`Evaluate/EnterpriseArena/graph.py` loads environment variables from `Evaluate/EnterpriseArena/.env` if that file exists.

For vLLM evaluation, create:

```bash
cd /EnterpriseLab/Evaluate/EnterpriseArena

cat > .env <<'EOF'
MODEL_MODE=vllm
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=local-key
DEBUG_MODE=True
EOF
```

For a local Hugging Face model instead:

```bash
cat > .env <<'EOF'
MODEL_MODE=local
LOCAL_MODEL_PATH=/path/to/local/model
DEBUG_MODE=True
EOF
```

For Azure evaluation:

```bash
cat > .env <<'EOF'
MODEL_MODE=azure
AZURE_API_KEY=...
AZURE_API_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_API_VERSION=2024-02-01
DEBUG_MODE=True
EOF
```

Make sure `Evaluate/EnterpriseArena/mcp_config_http.json` points to the MCP server ports you actually started.

### 4.2 Generate trajectories

Start EnterpriseArena apps, MCP servers, and your model server first. Then run:

```bash
cd /EnterpriseLab/Evaluate/EnterpriseArena
source ../../.venv/bin/activate

python client.py \
  --mcp_config mcp_config_http.json \
  --tasks ../../Data/tasks.json
```

The task file should be a JSON array like:

```json
[
  {
    "query": "Locate all repositories that contain the keyword keystone."
  }
]
```

The client writes a batch file under:

```text
Evaluate/EnterpriseArena/trajectories/batch_trajectories_<timestamp>.json
```

Note: `client.py` currently accepts `--output_trajectories`, but the save helper writes to the `trajectories/` directory using its timestamped filename. Use the path printed at the end of the run.

Interactive mode is available for manual probing:

```bash
python client.py \
  --mcp_config mcp_config_http.json \
  --interactive
```

### 4.3 Evaluate trajectories

The evaluator expects:

- Gold trajectories in chat format, for example `Data/enterprise_arena_gold.json`.
- Predicted trajectories in the batch format produced by `Evaluate/EnterpriseArena/client.py`.
- Matching queries between gold and predicted files. Alignment is done by exact query text.

Run:

```bash
cd /EnterpriseLab/Evaluate/MCP_eval
source ../../.venv/bin/activate

export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"

python mcp_evaluator.py \
  --output_trajectories ../EnterpriseArena/trajectories/batch_trajectories_<timestamp>.json \
  --gold_trajectories ../../Data/enterprise_arena_gold.json \
  --output_results enterprise_arena_eval
```

`--output_results` is used as an output prefix. The evaluator writes two timestamped files:

```text
Evaluate/MCP_eval/results/enterprise_arena_eval_<timestamp>_tool_evaluation.json
Evaluate/MCP_eval/results/enterprise_arena_eval_<timestamp>_llm_evaluation.json
```

### 4.4 What the metrics mean

Tool evaluation computes:

- Tool-name accuracy.
- Parameter match score.
- Tool order score.
- Strict and flexible overall scores.
- Missing tools, extra tools, and parameter mismatches per task.

LLM evaluation scores:

- Planning.
- Execution flow.
- Tool selection.
- Tool usage.
- Adaptability.
- Efficiency.
- Context awareness.
- Requirement coverage.
- Accuracy.
- Completeness.
- Usefulness.

If the evaluator reports zero tasks, check that the user query text in `Data/enterprise_arena_gold.json` exactly matches the query text in the generated prediction file.
