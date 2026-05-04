#!/bin/bash
echo "=== Job Started: $(date) ==="
echo "Node: $SLURM_NODELIST"
echo "Allocated GPUs: $SLURM_JOB_GPUS"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# ============================================================================
# CRITICAL: Load CUDA module (check your cluster's module name)
# ============================================================================
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
MODEL_PATH="/path/to/your/model"  # <-- UPDATE THIS PATH
PORT=8001
API_KEY="judge"
echo "GPU monitoring started"
echo ""

# ============================================================================
# Launch training
# ============================================================================
vllm serve \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8001 \
    --dtype float16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 128000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --api-key "$API_KEY" \
# ============================================================================
# Cleanup
# ============================================================================
kill $MONITOR_PID $PMON_PID 2>/dev/null
echo ""
echo "=== Job Completed: $(date) ==="
nvidia-smi
