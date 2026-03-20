"""
Main training script for Agentic GRPO.

This is your entry point. Modify the config section to match your setup.
"""

import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_structures import *
from rollout_manager import AgenticRolloutManager
from collator import TrajectoryCollator
from grpo_trainer import GRPOTrainer
from tool_environment import SimpleToolEnvironment
from reward_function import LLMJudge, SimpleRewardFunction

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    # Model settings
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Or your fine-tuned model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Training settings
    GROUP_SIZE = 4  # G in GRPO (number of trajectories per query)
    BATCH_SIZE = 2  # Number of queries per batch
    NUM_EPOCHS = 3
    LEARNING_RATE = 1e-6
    KL_COEFF = 0.1
    MAX_GRAD_NORM = 1.0

    # Trajectory settings
    MAX_TURNS = 10
    MAX_TOOL_OUTPUT_TOKENS = 500
    TOOL_TIMEOUT_SECONDS = 30.0
    MAX_CONTEXT_LENGTH = 4096

    # Checkpointing
    CHECKPOINT_DIR = "./checkpoints"
    SAVE_EVERY = 50
    EVAL_EVERY = 10

    # Reward function
    USE_LLM_JUDGE = False  # Set to True to use GPT-4/Claude judge
    JUDGE_MODEL = "gpt-4"  # Only used if USE_LLM_JUDGE=True

    # Logging
    LOG_WANDB = False  # Set to True to log to Weights & Biases

    # ========================================================================
    # LOAD MODEL AND TOKENIZER
    # ========================================================================

    logger.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model loaded on {DEVICE}")

    # ========================================================================
    # INITIALIZE COMPONENTS
    # ========================================================================

    # Tool environment factory
    def tool_env_factory():
        return SimpleToolEnvironment()

    # Rollout manager
    rollout_manager = AgenticRolloutManager(
        model=model,
        tokenizer=tokenizer,
        tool_env_factory=tool_env_factory,
        max_turns=MAX_TURNS,
        max_tool_output_tokens=MAX_TOOL_OUTPUT_TOKENS,
        tool_timeout_seconds=TOOL_TIMEOUT_SECONDS,
        max_context_length=MAX_CONTEXT_LENGTH,
        device=DEVICE
    )

    # Collator
    collator = TrajectoryCollator(tokenizer)

    # Reward function
    if USE_LLM_JUDGE:
        logger.info(f"Using LLM judge: {JUDGE_MODEL}")
        reward_fn = LLMJudge(model_name=JUDGE_MODEL, use_api=True)
    else:
        logger.info("Using simple rule-based reward function")
        reward_fn = SimpleRewardFunction()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        rollout_manager=rollout_manager,
        collator=collator,
        reward_fn=reward_fn,
        kl_coeff=KL_COEFF,
        group_size=GROUP_SIZE,
        max_grad_norm=MAX_GRAD_NORM,
        device=DEVICE
    )

    # ========================================================================
    # PREPARE DATASET
    # ========================================================================

    # Example dataset - replace with your actual tasks
    train_dataset = [
        {
            'id': 'task_001',
            'system': 'You are a helpful assistant with access to tools. Use the ReAct format: Thought, Action, Action Input, Observation, ... Final Answer.',
            'user': 'What is 127 * 384?'
        },
        {
            'id': 'task_002',
            'system': 'You are a helpful assistant with access to tools. Use the ReAct format: Thought, Action, Action Input, Observation, ... Final Answer.',
            'user': 'Calculate 2^10 + 5^3.'
        },
        {
            'id': 'task_003',
            'system': 'You are a helpful assistant with access to tools. Use the ReAct format: Thought, Action, Action Input, Observation, ... Final Answer.',
            'user': 'Search for the capital of France.'
        },
        # Add more tasks here...
    ]

    # Optional: Load from file
    # import json
    # with open('train_tasks.json', 'r') as f:
    #     train_dataset = json.load(f)

    logger.info(f"Loaded {len(train_dataset)} training tasks")

    # ========================================================================
    # TRAIN
    # ========================================================================

    logger.info("\nStarting training...")
    logger.info(f"Configuration:")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Group size: {GROUP_SIZE}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Epochs: {NUM_EPOCHS}")
    logger.info(f"  Learning rate: {LEARNING_RATE}")
    logger.info(f"  KL coefficient: {KL_COEFF}")
    logger.info(f"  Max turns: {MAX_TURNS}")
    logger.info("")

    metrics = trainer.train(
        train_dataset=train_dataset,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        save_every=SAVE_EVERY,
        eval_every=EVAL_EVERY,
        checkpoint_dir=CHECKPOINT_DIR,
        log_wandb=LOG_WANDB
    )

    logger.info("\nTraining complete!")
    logger.info(f"Final model saved to {CHECKPOINT_DIR}/final")

    # Save metrics
    import json
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to training_metrics.json")


if __name__ == "__main__":
    main()