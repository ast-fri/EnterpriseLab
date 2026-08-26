
export NO_PROXY="gpu07,localhost,127.0.0.1"
export no_proxy="gpu07,localhost,127.0.0.1"
export PROJECT_ROOT="${PROJECT_ROOT:-EnterprisePlatform/train/Agentic_GRPO}"

# Model switch:
#   MODEL_SERIES=qwen3   -> uses the original rollout manager
#   MODEL_SERIES=qwen3.5 -> uses rollout_manager_35
#   MODEL_SERIES=gemma   -> uses gemma loader
export MODEL_SERIES="${MODEL_SERIES:-gemma-4-E4B-it}"
export MODEL_NAME="${MODEL_NAME:-}"

#  use local path:
# export MODEL_NAME="/path/to/local/cohere/model"
export TRAINER_VARIANT="${TRAINER_VARIANT:-dapo}"  # Options: grpo, dapo, 

export PROMPT_MODE="${PROMPT_MODE:-artist}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export TRAIN_ENVIRONMENT="${TRAIN_ENVIRONMENT:-enterprise}"
export ENTERPRISEBENCH_ENV_ROOT="${ENTERPRISEBENCH_ENV_ROOT:-}"
export DATASET_PATH="${DATASET_PATH:-}"
export REWARD_MODE="${REWARD_MODE:-checkpoint}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-EnterprisePlatform/train/agentic_training_original/Output_enterprisebench_gemma_4_E4B-it-env}"
# Dataset Shuffle Configuration
export SHUFFLE_DATASET="${SHUFFLE_DATASET:-true}"    # Enable shuffle by default
export SHUFFLE_SEED="${SHUFFLE_SEED:-}"              # Empty = random seed, set number for reproducible shuffle
# Local Qwen Judge (Primary - faster and more reliable)
export JUDGE_API_BASE="${JUDGE_API_BASE:-"
export JUDGE_API_KEY="${JUDGE_API_KEY:-}"
export JUDGE_MODEL="${JUDGE_MODEL:-}"

export REWARD_CACHE_PATH="${REWARD_CACHE_PATH:-${CHECKPOINT_DIR}/judge_cache/local_qwen_judge_cache.jsonl}"
export LOCAL_QWEN_JUDGE_SUCCESS_REWARD="${LOCAL_QWEN_JUDGE_SUCCESS_REWARD:-1.0}"
export LOCAL_QWEN_JUDGE_FALLBACK_MAX="${LOCAL_QWEN_JUDGE_FALLBACK_MAX:-0.35}"
export LOCAL_QWEN_JUDGE_MAX_TOKENS="${LOCAL_QWEN_JUDGE_MAX_TOKENS:-1024}"


if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/train_enterprise.py" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$PROJECT_ROOT"
fi

# Resume only from checkpoints created with the corrected text-language LoRA
# targets. Older Gemma 4 checkpoints contain audio/vision-only adapters.
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

# Reference Model Device Configuration
# Options: "cpu" (saves ~18GB GPU memory) or "auto" (keeps on GPU)
export REF_MODEL_DEVICE="${REF_MODEL_DEVICE:-cpu}"

# Tool Filtering Configuration (reduces memory by showing fewer tools)
# ENABLE_TOOL_FILTERING: "true" to filter tools, "false" to show all tools
# NUM_RANDOM_TOOLS: number of random distractor tools beyond gold tools
# Example: 5 gold + 30 random = 35 tools (~3k tokens) vs 162+ tools (~12k tokens)
# Memory savings: ~70% reduction in prompt size, allows higher GROUP_SIZE
export ENABLE_TOOL_FILTERING="${ENABLE_TOOL_FILTERING:-true}"
export NUM_RANDOM_TOOLS="${NUM_RANDOM_TOOLS:-20}"

echo "=== Job Started: $(date) ==="
echo "Node: $SLURM_NODELIST"
echo "Allocated GPUs: $SLURM_JOB_GPUS"
echo "Job ID: $SLURM_JOB_ID"
echo "Model series: $MODEL_SERIES"
echo "Trainer variant: $TRAINER_VARIANT"

echo "Prompt mode: $PROMPT_MODE"
echo "Prompt template: $PROMPT_TEMPLATE_PATH"
echo "Python binary: $PYTHON_BIN"
echo "Training environment: $TRAIN_ENVIRONMENT"
echo "Dataset path: $DATASET_PATH"
echo "Script directory: $SCRIPT_DIR"
if [ "$TRAIN_ENVIRONMENT" = "emulation" ]; then
    echo "Emulation app roots: $EMULATION_APP_ROOTS"
    echo "Emulation datasets: $EMULATION_DATASET_PATH"
    echo "Emulation MCP URLs: $EMULATION_MCP_URLS"
    echo "Emulation API URLs: $EMULATION_API_URLS"
    echo "Reward mode: $REWARD_MODE"
else
    echo "EnterpriseBench env root: $ENTERPRISEBENCH_ENV_ROOT"
    echo "Reward mode: $REWARD_MODE"
fi
echo "Shuffle dataset: $SHUFFLE_DATASET"
if [ -n "$SHUFFLE_SEED" ]; then
    echo "Shuffle seed: $SHUFFLE_SEED (reproducible)"
else
    echo "Shuffle seed: random"
fi
echo "Reference model device: $REF_MODEL_DEVICE"
echo "Tool filtering enabled: $ENABLE_TOOL_FILTERING"
if [ "$ENABLE_TOOL_FILTERING" = "true" ]; then
    echo "Number of random tools (+ gold tools): $NUM_RANDOM_TOOLS"
fi
echo ""

# ============================================================================
# CRITICAL: Load CUDA module (check your cluster's module name)
# ============================================================================
if command -v module >/dev/null 2>&1; then
    module load cuda/12.1
fi
# OR if modules aren't used:
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# ============================================================================
# Diagnostic: Verify CUDA and GPUs
# ============================================================================
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Check nvidia-smi
echo "Running nvidia-smi..."
nvidia-smi
if [ $? -ne 0 ]; then
    echo "ERROR: nvidia-smi failed! GPUs not accessible."
    exit 1
fi
echo ""

# Verify PyTorch can see CUDA
echo "Testing PyTorch CUDA..."
"$PYTHON_BIN" -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')"
if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch cannot detect CUDA!"
    exit 1
fi
echo ""

echo "Checking Python dependencies..."
"$PYTHON_BIN" -c "import os, sys; import importlib.util as u; env=os.getenv('TRAIN_ENVIRONMENT','enterprise').strip().lower(); variant=os.getenv('TRAINER_VARIANT','grpo').strip().lower(); mods=['torch','transformers','peft','httpx']; mods += ['fastmcp'] if env == 'emulation' else []; mods += ['langchain_openai','langchain_core'] if variant == 'environment_belief' else []; missing=[m for m in mods if not u.find_spec(m)]; print(f'Executable: {sys.executable}'); print(f'Required mods: {mods}'); print(f'Missing: {missing}'); raise SystemExit(1 if missing else 0)"
if [ $? -ne 0 ]; then
    echo "ERROR: Required Python packages are missing in $PYTHON_BIN"
    echo "Set PYTHON_BIN to the interpreter from the environment where the required packages are installed."
    exit 1
fi
echo ""

# ============================================================================
# GPU monitoring (single GPU version)
# ============================================================================
(
    echo "timestamp,index,name,util.gpu,util.mem,mem.total,mem.used,mem.free,temp,power"
    while true; do
        nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw \
        --format=csv,noheader > gpu_util_${SLURM_JOB_ID}.csv.tmp
        mv gpu_util_${SLURM_JOB_ID}.csv.tmp gpu_util_${SLURM_JOB_ID}.csv
        sleep 1
    done
) &
MONITOR_PID=$!

nvidia-smi pmon -c 1 -s mu -d 5 -f gpu_pmon_${SLURM_JOB_ID}.log &
PMON_PID=$!

echo "GPU monitoring started"
echo ""

# ============================================================================
# Launch training
# ============================================================================
"$PYTHON_BIN" "$SCRIPT_DIR/train_enterprise.py"
   
# ============================================================================
# Cleanup
# ============================================================================
kill $MONITOR_PID $PMON_PID 2>/dev/null
echo ""
echo "=== Job Completed: $(date) ==="
nvidia-smi
