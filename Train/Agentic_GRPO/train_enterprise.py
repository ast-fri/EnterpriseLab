# """
# GRPO Training Script for EnterpriseBench - Updated for Your Exact Task Format

# Now handles your task format with:
# - task_id, instruction, chain_of_thought, ground_truth, etc.
# """

import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# # Import GRPO components
from grpo_trainer import GRPOTrainer
from rollout_manager import AgenticRolloutManager
from enterprise_tool_environment import create_enterprise_tool_environment
from enterprise_dataset_loader import load_enterprise_tasks_v2, create_ground_truth_reward_function

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
# CONFIGURATION - EDIT THESE FOR YOUR SETUP
# ============================================================================

# Model Configuration
MODEL_NAME = "path/to/your/models/qwen3-8b"  # Your base model
MODEL_CACHE_DIR = "path/to/model/cache"  # Optional: where to cache models

# Training Hyperparameters
GROUP_SIZE = 2              # Number of trajectories per query (G in GRPO)
BATCH_SIZE = 1              # Number of queries per batch
NUM_EPOCHS = 3              # Number of training epochs
LEARNING_RATE = 1e-6        # Learning rate
KL_COEFF = 0.1              # KL divergence penalty coefficient

# Trajectory Generation
MAX_TURNS = 10              # Maximum ReAct turns per trajectory
MAX_TOOL_OUTPUT_TOKENS = 500  # Truncate tool outputs to prevent context overflow

# Reward Function - IMPORTANT CHOICE
USE_GROUND_TRUTH_REWARD = True   # Use your gold chain_of_thought (recommended)
USE_LLM_JUDGE = False            # Or use LLM judge (slower, less accurate)
JUDGE_MODEL = "gpt-4"            # If using LLM judge

# Dataset Filters (optional)
DIFFICULTY_FILTER = None    # "EASY", "MEDIUM", "HARD", or None for all
DOMAIN_FILTER = None        # "HR", "CRM", "GitHub", etc., or None for all
MIN_STEPS = None            # Minimum trajectory steps
MAX_STEPS = None            # Maximum trajectory steps

# Paths
CHECKPOINT_DIR = "path/to/checkpoints"  # Where to save model checkpoints
DATASET_PATH = "path/to/your/enterprise_tasks.json"  # Your dataset in the new format

# WandB Configuration (optional)
USE_WANDB = True
WANDB_PROJECT = "EnterpriseBench-GRPO"
WANDB_RUN_NAME = f"grpo_enterprise_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# # ============================================================================
# # MAIN TRAINING FUNCTION
# # ============================================================================

def main():
    """Main training loop."""

    logger.info("="*80)
    logger.info("GRPO Training for EnterpriseBench")
    logger.info("="*80)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Group Size: {GROUP_SIZE}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Epochs: {NUM_EPOCHS}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"Reward: {'Ground Truth' if USE_GROUND_TRUTH_REWARD else 'LLM Judge'}")
    logger.info("="*80)

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. Load model and tokenizer
    logger.info(f"Loading model: {MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model loaded on {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # 2. Load dataset with your exact format
    logger.info(f"Loading tasks from: {DATASET_PATH}")

    train_dataset = load_enterprise_tasks_v2(
        path=DATASET_PATH,
        max_tasks=100,  # Start with 100 tasks, set to None for all
        difficulty_filter=DIFFICULTY_FILTER,
        domain_filter=DOMAIN_FILTER,
        min_steps=MIN_STEPS,
        max_steps=MAX_STEPS
    )

    logger.info(f"Loaded {len(train_dataset)} training tasks")

    # 3. Create rollout manager with EnterpriseBench tools
    logger.info("Initializing EnterpriseBench tool environment...")

    rollout_manager = AgenticRolloutManager(
        model=model,
        tokenizer=tokenizer,
        tool_env_factory=create_enterprise_tool_environment,
        max_turns=MAX_TURNS,
        max_tool_output_tokens=MAX_TOOL_OUTPUT_TOKENS,
    )

    logger.info(f"Loaded EnterpriseBench tools from environment")

    # 4. Create reward function
    if USE_GROUND_TRUTH_REWARD:
        logger.info("Using ground truth reward based on chain_of_thought")
        reward_fn = create_ground_truth_reward_function(train_dataset)
    else:
        logger.info(f"Using LLM judge: {JUDGE_MODEL}")
        from reward_function import LLMJudge
        reward_fn = LLMJudge(
            model_name=JUDGE_MODEL,
            use_api=True
        )

    # 5. Create GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        rollout_manager=rollout_manager,
        reward_function=reward_fn,
        group_size=GROUP_SIZE,
        learning_rate=LEARNING_RATE,
        kl_coeff=KL_COEFF,
        use_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=WANDB_RUN_NAME
    )

    # 6. Train!
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80 + "\n")

    trainer.train(
        train_dataset=train_dataset,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        checkpoint_dir=CHECKPOINT_DIR,
        save_every_n_steps=50
    )

    # 7. Save final model
    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, "final_model")
    logger.info(f"Saving final model to: {final_checkpoint_path}")

    model.save_pretrained(final_checkpoint_path)
    tokenizer.save_pretrained(final_checkpoint_path)

    logger.info("="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Final model saved to: {final_checkpoint_path}")
    logger.info(f"Training logs saved to: enterprise_training.log")
    logger.info(f"Checkpoints saved to: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()