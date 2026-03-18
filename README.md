# <img src="assets/images/agent.png" alt="EnterpriseLab" width="50"/> EnterpriseLab: Training and Evaluating LLM Agents in Enterprise Environments

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

EnterpriseLab is a comprehensive framework for training, deploying, and evaluating LLM agents in realistic enterprise environments. This repository provides a complete training pipeline and deployment infrastructure using MCP (Model Context Protocol) servers to interact with real enterprise applications.

## 🌟 Key Features

- **🏗️ Complete Enterprise Arena**: Dockerized deployment of 8+ enterprise applications (GitLab, Plane, OwnCloud, Zammad, RocketChat, Dolibarr, Frappe)
- **🔌 MCP Server Integration**: Custom MCP servers for seamless agent-application interaction
- **🎓 Advanced Training Pipeline**: Support for SFT, DPO, GRPO, SRPO, and Agentic-GRPO training methods
- **🔍 Comprehensive Evaluation**: Interactive and graph-based evaluation frameworks for agent performance
- **📊 Multi-Domain Coverage**: HR, IT Service Management, CRM, Software Engineering, Project Management, and Business Operations
- **🤖 Agent-Ready Infrastructure**: Pre-configured environments for deploying and testing LLM agents

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10 or higher
- NVIDIA GPU (recommended for training)
- Sufficient disk space (~20GB for all applications)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://anonymous.4open.science/r/EnterpriseLab-51F4.git
   cd EnterpriseLab
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🏢 Setting Up the Enterprise Arena

### Step 1: Deploy Enterprise Applications

The Arena contains pre-configured enterprise applications that simulate a realistic business environment.

#### Available Applications

| Application | Purpose | Documentation |
|------------|---------|---------------|
| **GitLab** | Code repository & CI/CD | [README](Arena/apps/gitlab/README.md) |
| **Plane** | Project management | [README](Arena/apps/plane/README.md) |
| **OwnCloud** | File storage & collaboration | [README](Arena/apps/owncloud/README.md) |
| **Zammad** | Ticketing & helpdesk | [README](Arena/apps/zammad/README.md) |
| **RocketChat** | Team communication | [README](Arena/apps/rocketchat/README.md) |
| **Dolibarr** | ERP & CRM | [README](Arena/apps/dolibarr/README.md) |
| **Frappe** | Business operations | [README](Arena/apps/frappe/README.md) |

#### Deploy All Applications

```bash
cd Arena/apps
bash start_all_servers.sh
```

Or deploy individual applications:

```bash
cd Arena/apps/gitlab
docker-compose up -d
```

#### Verify Deployment

Check that all services are running:
```bash
docker ps
```

### Step 2: Set Up MCP Servers

MCP (Model Context Protocol) servers provide standardized interfaces for LLM agents to interact with enterprise applications.

#### Available MCP Servers

- **gitlab**: Git operations, issue tracking, merge requests
- **plane**: Project management, task creation, sprint planning
- **owncloud**: File operations, sharing, document management
- **zammad**: Ticket management, customer support
- **rocketchat**: Team messaging, channel management
- **dolibarr**: ERP operations, invoicing, CRM
- **frappe**: Business workflows, custom apps
- **aider**: AI-assisted coding
- **playwright**: Web automation

#### Deploy All MCP Servers

```bash
cd Arena/MCP_servers
bash start_all_servers.sh
```

Or deploy individual MCP servers:

```bash
cd Arena/MCP_servers/gitlab
docker-compose up -d
```

#### Configure MCP Endpoints

MCP servers communicate via streamable HTTP. Update `Evaluate/EnterpriseArena/mcp_config_http.json` with your server endpoints:

```json
{
  "mcpServers": {
    "gitlab": {
      "transport": "streamable_http",
      "url": "http://localhost:8008/mcp"
    },
    "plane": {
      "transport": "streamable_http",
      "url": "http://localhost:8009/mcp"
    }
  }
}
```

#### Verify MCP Server Health

```bash
curl http://localhost:9001/health  # GitLab MCP
curl http://localhost:9002/health  # Plane MCP
```

## 🎓 Training LLM Agents

EnterpriseLab supports multiple training paradigms for developing enterprise-capable LLM agents.

### Training Methods

#### 1. Supervised Fine-Tuning (SFT)

Train models on curated enterprise task demonstrations.

```bash
cd Train/SFT
python train.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --data_path <path_to_training_data> \
  --output_dir ./checkpoints/sft \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --learning_rate 2e-5
```

**Features:**
- FSDP (Fully Sharded Data Parallel) support
- Custom data filtering and preprocessing
- Enterprise-specific prompt formatting

#### 2. Direct Preference Optimization (DPO)

Align models with human preferences on enterprise tasks.

```bash
cd Train/DPO
python dpo_train.py \
  --model_name_or_path <base_model> \
  --preference_data_path <path_to_preference_data> \
  --output_dir ./checkpoints/dpo \
  --num_train_epochs 1 \
  --beta 0.1
```

**Features:**
- Preference pair training
- FSDP configuration
- Enterprise task alignment

#### 3. Group Relative Policy Optimization (GRPO)

Optimize agents using group-based rewards from enterprise task trajectories.

```bash
cd Train/GRPO
python train.py \
  --model_name_or_path <base_model> \
  --trajectory_path <path_to_trajectories> \
  --output_dir ./checkpoints/grpo \
  --num_train_epochs 2
```

**Features:**
- Trajectory collection with LangGraph
- Custom reward functions
- DeepSpeed integration

#### 4. Agentic-GRPO

Advanced training combining agentic workflows with GRPO.

```bash
cd Train/Agentic_GRPO
python train_enterprise.py \
  --model_name_or_path <base_model> \
  --task_config ../tasks.json \
  --output_dir ./checkpoints/agentic-grpo \
  --num_rollouts 100
```

**Features:**
- Enterprise-specific tool environments
- Agentic rollout management
- Custom reward functions for enterprise tasks
- ReAct-style prompt building

**Key Components:**
- `enterprise_tool_environment.py`: Simulated enterprise API interactions
- `rollout_manager.py`: Trajectory collection and management
- `reward_function.py`: Task-specific reward computation
- `react_parser.py`: Parse ReAct-format agent outputs

### Data Preparation

#### Filter and Prepare Training Data

```bash
cd Train/SFT
python filter_data.py \
  --input_path <raw_data> \
  --output_path <filtered_data> \
  --min_quality_score 0.7
```

#### Collect Trajectories for RL Training

```bash
cd Train/GRPO
python collect_trajectories.py \
  --model_name_or_path <model> \
  --task_file ../tasks.json \
  --output_path ./trajectories.jsonl \
  --num_episodes 1000
```

### Model Merging

After training with adapters (LoRA), merge weights back to base model:

```bash
cd Train/GRPO
python merge_model.py \
  --base_model_path <base_model> \
  --adapter_path ./checkpoints/grpo \
  --output_path ./merged_model
```

## 🔍 Evaluation

### Step 1: Generate Trajectories

Run the interactive evaluation to generate output trajectories for tasks. The `Interactive_mcp_localM.py` file uses `graph_final_localM.py` internally to execute tasks and generate trajectories:

```bash
cd Evaluate/EnterpriseArena
python Interactive_mcp_localM.py \
  --model_path <model_checkpoint> \
  --mcp_config mcp_config_http.json \
  --tasks ../../Data/tasks.json \
  --output_trajectories ./output_trajectories.jsonl
```

**Features:**
- Real-time agent interaction with MCP servers
- Graph-based task execution via `graph_final_localM.py`
- Automatic trajectory logging
- Tool usage tracking

### Step 2: Evaluate Trajectories

After generating trajectories, evaluate them against gold standard trajectories using the MCP evaluator:

```bash
cd Evaluate/MCP_eval
python mcp_evaluator.py \
  --output_trajectories ../EnterpriseArena/output_trajectories.jsonl \
  --gold_trajectories ../../Data/enterprise_arena_gold.json \
  --output_results ./evaluation_results.json
```

**Evaluation Pipeline:**

The `mcp_evaluator.py` orchestrates the evaluation using two sub-modules:

1. **Tool Evaluation** (`tool_evaluation.py`): 
   - Tool selection accuracy
   - Tool usage correctness
   - Parameter validation
   - Execution success rate

2. **LLM-based Evaluation** (`llm_evaluator.py`):
   - Semantic correctness of outputs
   - Task completion quality
   - Response appropriateness
   - Multi-step reasoning assessment

**Output Metrics:**
- Task success rate
- Tool usage accuracy
- Average steps to completion
- Error analysis
- Per-domain performance breakdown

## 📊 Task Generation Pipeline

**Note:** The complete task generation pipeline will be uploaded upon paper acceptance.

The pipeline will include:
- Automated task creation across all enterprise domains
- Difficulty calibration and complexity control
- Multi-step workflow generation
- Quality validation and filtering

## 📁 Evaluation Data

The `Data/` folder contains essential datasets for evaluating LLM agents:

### tasks.json
Contains structured task definitions for enterprise evaluation. Each task includes:
- **Task description**: Natural language instruction for the agent

### enterprise_arena_gold.json
Gold standard trajectories showing optimal task completion paths. Each trajectory contains:
- **Tool call sequences**: Correct order and parameters for tool usage
- **Intermediate states**: Expected system states after each action
- **Final answers**: Ground truth outputs for verification

These files are used by the evaluation pipeline to assess agent performance against established benchmarks.

## 🗂️ Repository Structure

```
EnterpriseLab/
├── Arena/                              # Enterprise application deployment
│   ├── apps/                          # Dockerized enterprise applications
│   │   ├── gitlab/                    # GitLab code repository
│   │   ├── plane/                     # Plane project management
│   │   ├── owncloud/                  # OwnCloud file storage
│   │   ├── zammad/                    # Zammad helpdesk
│   │   ├── rocketchat/                # RocketChat messaging
│   │   ├── dolibarr/                  # Dolibarr ERP/CRM
│   │   └── frappe/                    # Frappe business apps
│   └── MCP_servers/                   # MCP server implementations
│       ├── gitlab/                    # GitLab MCP server
│       ├── plane/                     # Plane MCP server
│       ├── owncloud/                  # OwnCloud MCP server
│       ├── zammad/                    # Zammad MCP server
│       ├── rocketchat/                # RocketChat MCP server
│       ├── dolibarr/                  # Dolibarr MCP server
│       ├── frappe/                    # Frappe MCP server
│       ├── aider/                     # Aider coding assistant MCP
│       └── playwright/                # Playwright automation MCP
├── Data/                               # Evaluation datasets
│   ├── tasks.json                     # Enterprise task definitions
│   └── enterprise_arena_gold.json     # Gold standard trajectories
├── Train/                              # Training pipelines
│   ├── SFT/                           # Supervised fine-tuning
│   ├── DPO/                           # Direct preference optimization
│   ├── GRPO/                          # Group relative policy optimization
│   ├── SRPO/                          # Self-rewarding policy optimization
│   └── Agentic_GRPO/                  # Agentic GRPO training
├── Evaluate/                           # Evaluation frameworks
│   ├── EnterpriseArena/               # Interactive evaluation
│   └── MCP_eval/                      # MCP-based evaluation
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## 🎯 Use Cases

### For Researchers
- **Training Novel Agent Architectures**: Experiment with different training paradigms (SFT, DPO, GRPO, Agentic-GRPO)
- **Benchmark Agent Performance**: Evaluate agents across realistic enterprise tasks
- **Study Tool Usage**: Analyze how agents learn to use enterprise APIs and tools
- **Multi-Step Reasoning**: Research complex workflow completion strategies

### For Developers
- **Build Enterprise Agents**: Train agents on your specific business workflows
- **Test Agent Reliability**: Validate performance in controlled enterprise environments
- **Integration Development**: Develop and test MCP servers for custom applications
- **Deployment Preparation**: Assess readiness before production deployment

### For Enterprise Teams
- **AI Capability Assessment**: Understand what LLM agents can and cannot do in your domain
- **ROI Evaluation**: Measure potential impact of agent deployment on productivity
- **Risk Analysis**: Identify failure modes and limitations in controlled settings
- **Training Data Creation**: Generate domain-specific training data from your workflows

## 🤝 Contributing

We welcome contributions to EnterpriseLab! Areas of interest:

- **New MCP Servers**: Integrate additional enterprise applications
- **Training Methods**: Implement novel agent training algorithms
- **Evaluation Metrics**: Develop better assessment criteria
- **Task Templates**: Create domain-specific task generators
- **Documentation**: Improve setup guides and tutorials

Please open an issue or pull request to contribute.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MCP (Model Context Protocol) by Anthropic
- Enterprise applications: GitLab, Plane, OwnCloud, Zammad, RocketChat, Dolibarr, Frappe
- Training frameworks: HuggingFace Transformers, DeepSpeed, PyTorch FSDP

## 📞 Support

For questions, issues, or collaboration:

- 📧 [Open an issue](https://github.com/ast-fri/EnterpriseLab/issues)
- 📖 Check individual component READMEs for detailed documentation
- 💬 Join the discussion in our community

## 🔗 Links

- **Repository**: [https://github.com/ast-fri/EnterpriseLab]
- **Paper**: Coming soon

## 📝 Citation

If you use EnterpriseLab in your research, please cite:

```bibtex
@article{enterpriselab2026,
    title = {EnterpriseLab: Training and Evaluating LLM Agents in Enterprise Environments},
    author = {Ankush Agarwal, Harsh Vishwakarma, Suraj Nagaje},
    journal = {Under Review},
    year = {2025}
}
```

---

**EnterpriseLab** - From evaluation to deployment: A complete framework for enterprise LLM agents.
