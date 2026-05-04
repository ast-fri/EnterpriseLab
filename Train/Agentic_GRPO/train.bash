#!/bin/bash

export NO_PROXY="gpu07,localhost,127.0.0.1"
export no_proxy="gpu07,localhost,127.0.0.1"
export MODEL_SERIES="${MODEL_SERIES:-qwen3.5}"
# Local Qwen Judge (Primary - faster and more reliable)
export JUDGE_API_BASE="${JUDGE_API_BASE:-   http://localhost:8001/v1/chat/completions}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-judge}"
export JUDGE_MODEL="${JUDGE_MODEL:-/path/to/your/judge/model}"  # <-- UPDATE THIS PATH
export REWARD_CACHE_PATH="${REWARD_CACHE_PATH:-${CHECKPOINT_DIR}/judge_cache/local_qwen_judge_cache.jsonl}"
export LOCAL_QWEN_JUDGE_SUCCESS_REWARD="${LOCAL_QWEN_JUDGE_SUCCESS_REWARD:-1.0}"
export LOCAL_QWEN_JUDGE_FALLBACK_MAX="${LOCAL_QWEN_JUDGE_FALLBACK_MAX:-0.35}"
export LOCAL_QWEN_JUDGE_MAX_TOKENS="${LOCAL_QWEN_JUDGE_MAX_TOKENS:-1024}"

# RESUME TRAINING FROM CHECKPOINT (optional)
# export RESUME_FROM_CHECKPOINT="/path/to/your/checkpoint"  # <-- UPDATE THIS PATH

echo "=== Job Started: $(date) ==="
echo "Node: $SLURM_NODELIST"
echo "Allocated GPUs: $SLURM_JOB_GPUS"
echo "Job ID: $SLURM_JOB_ID"
echo "Model series: $MODEL_SERIES"
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
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')"
if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch cannot detect CUDA!"
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
python train_enterprise.py \
   
# ============================================================================
# Cleanup
# ============================================================================
kill $MONITOR_PID $PMON_PID 2>/dev/null
echo ""
echo "=== Job Completed: $(date) ==="
nvidia-smi
