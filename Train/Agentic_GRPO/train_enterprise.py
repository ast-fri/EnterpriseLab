"""
GRPO Training Script for EnterpriseBench - WITH LoRA

Includes:
- LoRA for efficient fine-tuning
- Fixed device placement (single GPU)
- Diagnostic logging for performance
"""

import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

# MEMORY: Set CUDA allocator config to reduce fragmentation
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model, TaskType

# Import GRPO components
from grpo_trainer import GRPOTrainer
from rollout_manager import AgenticRolloutManager
from collator import TrajectoryCollator
from enterprise_tool_environment import create_enterprise_tool_environment
from enterprise_dataset_loader import load_enterprise_tasks_v2, create_ground_truth_reward_function
from reward import create_agentic_grpo_reward_function  # NEW: Three-fold reward with LLM judge
from dotenv import load_dotenv
load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enterprise_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model Configuration
MODEL_SERIES = os.getenv("MODEL_SERIES", "qwen3.5").strip().lower()
DEFAULT_MODEL_PATHS = {
    "qwen3": "/models/models/Qwen3-8B",
    "qwen3.5": "/models/models/Qwen3.5-9B",
}
DEFAULT_CHECKPOINT_DIRS = {
    "qwen3": "/EnterprisePlatform/train/agentic-grpo/Output_bench_qwen3_8b",
    "qwen3.5": "/EnterprisePlatform/train/agentic-grpo/Output_bench_qwen3.5_9b",
}
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    DEFAULT_MODEL_PATHS.get(MODEL_SERIES, DEFAULT_MODEL_PATHS["qwen3.5"]),
)
MODEL_CACHE_DIR = "/.cache"

# LoRA Configuration
USE_LORA = True
LORA_R = 64               # MEMORY: Reduced from 128 to 64 (half the trainable params)
LORA_ALPHA = 128          # MEMORY: Reduced from 256 to 128 (2x rank)
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [  # Qwen2 attention + MLP modules
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training Hyperparameters
GROUP_SIZE = 4              # MEMORY: Reduced from 4 to 2 trajectories per query
BATCH_SIZE = 1              # Number of queries per batch
NUM_EPOCHS = 3              # Number of training epochs
LEARNING_RATE = 1e-6        # Higher LR for LoRA (was 1e-6 for full model)
BETA = 0.01                 # KL penalty coefficient

# Trajectory Generation (DATA-DRIVEN from dataset analysis)
# Dataset stats: 95th percentile context=12395 tokens, tool_output=1200 tokens, steps=4
MAX_TURNS = 6                  # DATA: 95th percentile = 4, +margin for edge cases
MAX_TOOL_OUTPUT_TOKENS = 1200  # DATA: 95th percentile from actual task tool outputs
MAX_CONTEXT_LENGTH = 13500     # DATA: 95th percentile = 12395 + 10% margin
MAX_NEW_TOKENS = 512           # Standard generation length
TEMPERATURE = 1.0

# Reward Configuration
USE_AGENTIC_REWARD = True  # Set True to use new three-fold reward, False for old ground truth
REWARD_NORMALIZE_TO_UNIT = True  # Scale [0, 1.2] → [0, 1]

# LLM Judge Configuration (for Agentic Reward)
JUDGE_API_BASE = os.getenv("JUDGE_API_BASE") or "http://gpu04:8001/v1"
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY")  # Ensure this is set in your .env file
JUDGE_MODEL = "/models/models/Qwen3-32b"
REWARD_CACHE_PATH = "./reward_cache.jsonl"

# Legacy Reward Weights (3-category ground truth) - only used if USE_AGENTIC_REWARD=False
W_PRESENCE = 0.2
W_ORDER = 0.2
W_FINAL = 0.6

# Dataset Filters
DIFFICULTY_FILTER = None
DOMAIN_FILTER = None
MIN_STEPS = None
MAX_STEPS = 10
MAX_TASKS = None  # Start small for testing

# Paths
CHECKPOINT_DIR = os.getenv(
    "CHECKPOINT_DIR",
    DEFAULT_CHECKPOINT_DIRS.get(MODEL_SERIES, DEFAULT_CHECKPOINT_DIRS["qwen3.5"]),
)
DATASET_PATH = "/EnterprisePlatform/TaskGenerationPipeline/environments/EnterpriseBench/Task_Generation/tasks.json"
CRM_DATASET_PATH = "/EnterprisePlatform/TaskGenerationPipeline/crm/tasks/final_tasks.json"
TAU_DATASET_PATH = "/EnterprisePlatform/TaskGenerationPipeline/retail.json"
# WandB Configuration
USE_WANDB = False
WANDB_PROJECT = "EnterpriseBench-GRPO"
WANDB_RUN_NAME = f"grpo_enterprise_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
GRADIENT_ACCUMULATION_STEPS = 4
# Checkpointing
CHECKPOINT_EVERY = 10
RESUME_FROM_CHECKPOINT = os.getenv("RESUME_FROM_CHECKPOINT")

# NEW: Multi-GPU config
USE_MULTI_GPU = True  # Set to True when using 4 GPUs
NUM_GPUS = 6 if USE_MULTI_GPU else 1


def load_trainer_state(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load the full trainer state checkpoint, including optimizer and RNG state.

    PyTorch 2.6 changed torch.load() to default to weights_only=True, which
    breaks our locally-saved trainer_state.pt because it contains Python and
    NumPy objects in addition to tensors.
    """
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training loop."""
    resume_trainer_state = None
    if RESUME_FROM_CHECKPOINT:
        logger.info(f"Resume requested from checkpoint: {RESUME_FROM_CHECKPOINT}")
        trainer_state_path = os.path.join(RESUME_FROM_CHECKPOINT, "trainer_state.pt")
        if not os.path.exists(trainer_state_path):
            raise FileNotFoundError(
                f"Resume checkpoint is missing trainer state: {trainer_state_path}"
            )
        resume_trainer_state = load_trainer_state(trainer_state_path)
        logger.info(
            "Loaded trainer state: epoch=%s next_batch_idx=%s global_step=%s update_step=%s",
            resume_trainer_state.get("epoch"),
            resume_trainer_state.get("next_batch_idx"),
            resume_trainer_state.get("global_step"),
            resume_trainer_state.get("update_step"),
        )
    
    logger.info("="*80)
    logger.info("GRPO Training for EnterpriseBench")
    logger.info("="*80)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Model Series: {MODEL_SERIES}")
    logger.info(f"LoRA: {'ENABLED' if USE_LORA else 'DISABLED'}")
    if USE_LORA:
        logger.info(f"  - Rank: {LORA_R}, Alpha: {LORA_ALPHA}")
        logger.info(f"  - Target modules: {LORA_TARGET_MODULES}")
    logger.info(f"Group Size: {GROUP_SIZE}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Epochs: {NUM_EPOCHS}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"Beta (KL coeff): {BETA}")
    logger.info(f"Max new tokens per turn: {MAX_NEW_TOKENS}")
    logger.info(f"Resume Checkpoint: {RESUME_FROM_CHECKPOINT or 'None'}")
    logger.info(f"Reward System: {'Agentic (State+Function+Format)' if USE_AGENTIC_REWARD else 'Legacy (Presence+Order+Final)'}")
    if USE_AGENTIC_REWARD:
        logger.info(f"  - Judge Model: {JUDGE_MODEL.split('/')[-1]}")
        logger.info(f"  - Normalize to [0,1]: {REWARD_NORMALIZE_TO_UNIT}")
    logger.info("="*80)
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize wandb if requested
    if USE_WANDB:
        import wandb
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "model": MODEL_NAME,
                "use_lora": USE_LORA,
                "lora_r": LORA_R if USE_LORA else None,
                "group_size": GROUP_SIZE,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "beta": BETA,
            }
        )
        wandb_callback = lambda m, s: wandb.log(m, step=s)
    else:
        wandb_callback = None
    
    # ========================================================================
    # 1. Load base model and tokenizer
    # ========================================================================
    
    model_load_path = MODEL_NAME if USE_LORA or not RESUME_FROM_CHECKPOINT else RESUME_FROM_CHECKPOINT
    logger.info(f"Loading base model: {model_load_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # FIXED: Force single GPU placement (not "auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_load_path,
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # FIXED: was "auto"
        trust_remote_code=True
    )
    
    tokenizer_load_path = MODEL_NAME if USE_LORA or not RESUME_FROM_CHECKPOINT else RESUME_FROM_CHECKPOINT
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_load_path,
        cache_dir=MODEL_CACHE_DIR,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Model loaded")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # Diagnostic: Check device placement
    if hasattr(model, 'hf_device_map'):
        logger.info(f"Device map: {model.hf_device_map}")
    
    # ========================================================================
    # 2. Apply LoRA to policy model
    # ========================================================================
    
    if USE_LORA:
        logger.info("Applying LoRA to policy model...")
        model.enable_input_require_grads() 
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model.config.use_cache = False 
        model.gradient_checkpointing_enable()
        if RESUME_FROM_CHECKPOINT:
            logger.info("Loading LoRA adapter weights from resume checkpoint...")
            model = PeftModel.from_pretrained(
                model,
                RESUME_FROM_CHECKPOINT,
                is_trainable=True,
            )
        else:
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Verify trainable params
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        if trainable_params == 0:
            logger.error("ERROR: No trainable parameters after LoRA!")
            return
    
    # ========================================================================
    # 3. Create reference model (frozen, NO LoRA) - OFFLOADED TO CPU
    # ========================================================================

    logger.info("Creating reference model (frozen, no LoRA) - offloading to CPU...")

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # MEMORY OPTIMIZATION: Offload to CPU to save ~18GB GPU memory
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # Efficient loading
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    logger.info("Reference model created and frozen on CPU (saves ~18GB GPU memory)")
    
    # ========================================================================
    # 4. Create optimizer (only LoRA params if LoRA enabled)
    # ========================================================================
    
    if USE_LORA:
        # Only optimize LoRA parameters
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        logger.info(f"Optimizer: AdamW on LoRA params with lr={LEARNING_RATE}")
    else:
        logger.info("Enabling gradients for full fine-tuning...")
        model.train()  # Ensure model is in training mode
        
        # Critical: If config.json has gradient_checkpointing=True, this is required
        # to connect the computation graph at the input layer.
        model.enable_input_require_grads() 
        
        # Optional: Explicitly enable checkpointing to save memory (recommended for 6 GPUs)
        model.gradient_checkpointing_enable() 
        # -----------------------

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        logger.info(f"Optimizer: AdamW on all params with lr={LEARNING_RATE}")

    if resume_trainer_state:
        logger.info("Restoring optimizer state from checkpoint...")
        optimizer.load_state_dict(resume_trainer_state["optimizer_state_dict"])
    
    # ========================================================================
    # 5. Load dataset
    # ========================================================================
    
    logger.info(f"Loading tasks from: {DATASET_PATH}")
    
    train_dataset = load_enterprise_tasks_v2(
        path=DATASET_PATH,
        max_tasks=MAX_TASKS,
        difficulty_filter=DIFFICULTY_FILTER,
        domain_filter=DOMAIN_FILTER,
        min_steps=MIN_STEPS,
        max_steps=MAX_STEPS
    )
    
    logger.info(f"Loaded {len(train_dataset)} training tasks")
    
    if len(train_dataset) == 0:
        logger.error("No tasks loaded! Check your dataset path and filters.")
        return
    
    # ========================================================================
    # 6. Create reward function
    # ========================================================================

    if USE_AGENTIC_REWARD:
        logger.info("Creating Agentic GRPO reward function (state + function + format)...")
        logger.info(f"  - Judge: {JUDGE_MODEL}")
        logger.info(f"  - API: {JUDGE_API_BASE}")
        logger.info(f"  - Cache: {REWARD_CACHE_PATH}")
        logger.info(f"  - Normalize: {REWARD_NORMALIZE_TO_UNIT}")

        reward_fn = create_agentic_grpo_reward_function(
            tasks=train_dataset,
            judge_api_base=JUDGE_API_BASE,
            judge_model=JUDGE_MODEL,
            judge_api_key=JUDGE_API_KEY,
            cache_path=REWARD_CACHE_PATH,
            normalize_to_unit=REWARD_NORMALIZE_TO_UNIT,
            verbose=True,
        )
    else:
        logger.info("Creating legacy 3-category ground truth reward function...")
        logger.info(f"  - Weights: presence={W_PRESENCE}, order={W_ORDER}, final={W_FINAL}")

        reward_fn = create_ground_truth_reward_function(
            tasks=train_dataset
        )
    
    # ========================================================================
    # 7. Create rollout manager
    # ========================================================================
    
    logger.info("Initializing rollout manager...")
    
    rollout_manager_cls = AgenticRolloutManager
    
    logger.info(f"Rollout Manager: {rollout_manager_cls.__name__}")

    rollout_manager = rollout_manager_cls(
        model=model,
        tokenizer=tokenizer,
        tool_env_factory=create_enterprise_tool_environment,
        max_turns=MAX_TURNS,
        max_tool_output_tokens=MAX_TOOL_OUTPUT_TOKENS,
        max_context_length=MAX_CONTEXT_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        device=device,
    )
    
    # ========================================================================
    # 8. Create collator
    # ========================================================================
    
    collator = TrajectoryCollator(
        tokenizer=tokenizer,
        max_length=MAX_CONTEXT_LENGTH,
        padding_side="right"
    )
    
    # ========================================================================
    # 9. Create GRPO trainer
    # ========================================================================
    
    logger.info("Initializing GRPO trainer...")
    
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        reward_function=reward_fn,
        beta=BETA,
        max_grad_norm=1.0,
        device=device,
        logprob_chunk_size=32,  # MEMORY: Reduced from 64 to 32 for smaller log_softmax tensors
        use_wandb=USE_WANDB,
    )
    
    # ========================================================================
    # 10. Train!
    # ========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80 + "\n")
    
    all_metrics = trainer.train(
        rollout_manager=rollout_manager,
        train_queries=train_dataset,
        num_epochs=NUM_EPOCHS,
        group_size=GROUP_SIZE,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        collator=collator,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_every=CHECKPOINT_EVERY,
        log_callback=wandb_callback,
        resume_state=resume_trainer_state,
    )
    
    # ========================================================================
    # 11. Save final model/adapters
    # ========================================================================
    
    final_path = os.path.join(CHECKPOINT_DIR, "final_model")
    logger.info(f"Saving final model to: {final_path}")
    
    if USE_LORA:
        # Save LoRA adapters only
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info("LoRA adapters saved")
        logger.info(f"To load: model = AutoModelForCausalLM.from_pretrained('{MODEL_NAME}')")
        logger.info(f"         model = PeftModel.from_pretrained(model, '{final_path}')")
    else:
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info("Full model saved")
    
    # Save metrics
    metrics_path = os.path.join(CHECKPOINT_DIR, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    
    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
