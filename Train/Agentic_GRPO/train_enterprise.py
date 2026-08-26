"""
GRPO Training Script for EnterpriseBench - WITH LoRA

Includes:
- LoRA for efficient fine-tuning
- Fixed device placement (single GPU)
- Diagnostic logging for performance
"""

import argparse
import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:
    AutoModelForImageTextToText = None
    AutoProcessor = None
from peft import LoraConfig, PeftModel, get_peft_model, TaskType

# Import GRPO components
from grpo_trainer import GRPOTrainer
from dapo_trainer import DAPOTrainer
from entropy_trainer import EntropyTrainer
from agentflow_trainer import AgentFlowTrainer
from agentflow_rollout_manager import AgentFlowRolloutManager
from environment_belief_rl import (
    BeliefJudge,
    BranchSelector,
    BranchSelectorConfig,
    EnvironmentBeliefRolloutManager,
    EnvironmentBeliefTrainer,
    GenericEnvironmentAdapter,
)
from rollout_manager import AgenticRolloutManager
from rollout_manager_35 import AgenticRolloutManager35
from collator import TrajectoryCollator
# Import model factory for Cohere support
from model_factory import detect_model_type, get_rollout_manager_class, get_lora_target_modules
from enterprise_tool_environment import create_enterprise_tool_environment
from enterprise_dataset_loader import load_enterprise_tasks_v2, create_ground_truth_reward_function
from emulation_system_tool_environment import create_emulation_system_tool_environment
from emulation_system_dataset_loader import load_emulation_system_tasks
from reward import create_agentic_grpo_reward_function  # NEW: Three-fold reward with LLM judge
from reward_new import create_checkpoint_reward_function
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
MODEL_SERIES = os.getenv("MODEL_SERIES", "qwen3").strip().lower()
DEFAULT_MODEL_PATHS = {
    "qwen3": "/home/fripl/vharsh/research/models/models/Qwen3-8B",
    "qwen3.5": "/home/fripl/vharsh/research/models/models/Qwen3.5-9B",
    "gemma-4-31B-it": "/home/fripl/vharsh/research/models/models/google/gemma-4-31B-it",
    "gemma-4-e4b-it": "/home/fripl/vharsh/research/models/models/google/gemma-4-E4B-it",
    "cohere": "CohereLabs/c4ai-command-r7b-12-2024",
    "command-r": "CohereLabs/c4ai-command-r7b-12-2024",
}
DEFAULT_CHECKPOINT_DIRS = {
    "qwen3": "/home/fripl/vharsh/research/EnterprisePlatform/train/agentic-grpo/Output_bench_qwen3_8b",
    "qwen3.5": "/home/fripl/vharsh/research/EnterprisePlatform/train/agentic-grpo/Output_bench_qwen3.5_9b",
    "gemma-4-31B-it": "/home/fripl/vharsh/research/EnterprisePlatform/train/agentic-grpo/Output_bench_gemma_431b",
    "gemma-4-e4b-it": "/home/fripl/vharsh/research/EnterprisePlatform/train/agentic-grpo/Output_bench_gemma_4e4b",
    "cohere": "/home/fripl/vharsh/research/EnterprisePlatform/train/agentic_training_original/Output_cohere_r7b",
    "command-r": "/home/fripl/vharsh/research/EnterprisePlatform/train/agentic_training_original/Output_cohere_r7b",
}
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    DEFAULT_MODEL_PATHS.get(MODEL_SERIES, DEFAULT_MODEL_PATHS["qwen3"]),
)
MODEL_CACHE_DIR = "/home/fripl/vharsh/.cache"
ARTIST_PROMPT_TEMPLATE_PATH = (
    "/home/fripl/vharsh/research/EnterprisePlatform/train/agentic-grpo/"
    "prompts/artist_function_calling_exact.txt"
)
ENVIRONMENT_BELIEF_PROMPT_TEMPLATE_PATH = (
    "/home/fripl/vharsh/research/EnterprisePlatform/train/"
    "agentic_training_original/prompts/environment_belief_artist.txt"
)
COHERE_PROMPT_TEMPLATE_PATH = (
    "/home/fripl/vharsh/research/EnterprisePlatform/train/agentic_training_original/"
    "prompts/cohere_function_calling.txt"
)

# LoRA Configuration
USE_LORA = True
LORA_R = 128              # Rank (higher = more parameters, better quality)
LORA_ALPHA = 256         # Alpha (typically 2x rank)
LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = [  # Qwen attention + MLP modules
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
GEMMA_LORA_TARGET_MODULES = [
    "q_proj.linear",
    "k_proj.linear",
    "v_proj.linear",
    "o_proj.linear",
    "gate_proj.linear",
    "up_proj.linear",
    "down_proj.linear",
]
# NOTE: LORA_TARGET_MODULES is now determined dynamically by model_factory
# This line is kept for backwards compatibility but will be overridden in main()
LORA_TARGET_MODULES = (
    GEMMA_LORA_TARGET_MODULES
    if MODEL_SERIES in {"gemma", "gemma4"} or "gemma-4" in MODEL_NAME.lower()
    else DEFAULT_LORA_TARGET_MODULES
)

# Training Hyperparameters
GROUP_SIZE = int(os.getenv("GROUP_SIZE", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-6"))

# Tool Filtering (Memory Optimization)
# Reduces prompt size from 162 tools (~12k tokens) to gold + random (~3-4k tokens)
ENABLE_TOOL_FILTERING = os.environ.get("ENABLE_TOOL_FILTERING", "false").lower() == "true"
NUM_RANDOM_TOOLS = int(os.environ.get("NUM_RANDOM_TOOLS", "30"))
BETA = float(os.getenv("BETA", "0.0"))

# Trajectory Generation
MAX_TURNS = int(os.getenv("MAX_TURNS", "5"))
MAX_TOOL_OUTPUT_TOKENS = int(os.getenv("MAX_TOOL_OUTPUT_TOKENS", "1024"))
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "16384"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Reward Configuration
USE_AGENTIC_REWARD = True  # Set True to use new three-fold reward, False for old ground truth
REWARD_NORMALIZE_TO_UNIT = True  # Scale [0, 1.2] → [0, 1]
REWARD_MODE = os.getenv("REWARD_MODE", "").strip().lower()

# LLM Judge Configuration (for Agentic Reward)
JUDGE_API_BASE = os.getenv("JUDGE_API_BASE") or "http://gpu04:8001/v1"
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY")  # Ensure this is set in your .env file
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "/home/fripl/vharsh/research/models/models/Qwen3-32b")
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

# Dataset Shuffle Configuration
SHUFFLE_DATASET = os.getenv("SHUFFLE_DATASET", "true").lower() in ("true", "1", "yes")
SHUFFLE_SEED = int(os.getenv("SHUFFLE_SEED")) if os.getenv("SHUFFLE_SEED") else None

# Reference Model Device Configuration
# "cpu" = offload to CPU (saves ~18GB GPU memory)
# "auto" = keep on GPU (faster KL computation but requires more memory)
REF_MODEL_DEVICE = os.getenv("REF_MODEL_DEVICE", "cpu").strip().lower()

TRAIN_ENVIRONMENT = (
    os.getenv("TRAIN_ENVIRONMENT")
    or ("emulation" if os.getenv("EMULATION_APP_ROOT") else "enterprise")
).strip().lower()

# Paths
CHECKPOINT_DIR = os.getenv(
    "CHECKPOINT_DIR",
    DEFAULT_CHECKPOINT_DIRS.get(MODEL_SERIES, DEFAULT_CHECKPOINT_DIRS["qwen3.5"]),
)
DEFAULT_ENTERPRISE_DATASET_PATH = (
    "/home/fripl/vharsh/research/EnterprisePlatform/TaskGenerationPipeline/"
    "environments/EnterpriseBench/Task_Generation/tasks.json"
)
DEFAULT_EMULATION_APP_ROOT = (
    "/home/fripl/vharsh/research/emulation_system_env/rocket_chat"
)
EMULATION_APP_ROOT = os.getenv("EMULATION_APP_ROOT", DEFAULT_EMULATION_APP_ROOT)
DEFAULT_EMULATION_DATASET_PATH = os.getenv(
    "EMULATION_DATASET_PATH",
    f"{EMULATION_APP_ROOT}/tasks/tasks.json",
)
DATASET_PATH = os.getenv(
    "DATASET_PATH",
    DEFAULT_EMULATION_DATASET_PATH
    if TRAIN_ENVIRONMENT == "emulation"
    else DEFAULT_ENTERPRISE_DATASET_PATH,
)
CRM_DATASET_PATH = "/home/fripl/vharsh/research/EnterprisePlatform/TaskGenerationPipeline/crm/tasks/final_tasks.json"
TAU_DATASET_PATH = "/home/fripl/vharsh/research/EnterprisePlatform/TaskGenerationPipeline/retail.json"
# WandB Configuration
USE_WANDB = False
WANDB_PROJECT = "EnterpriseBench-GRPO"
WANDB_RUN_NAME = f"grpo_enterprise_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
GRADIENT_ACCUMULATION_STEPS = 8
# Checkpointing
CHECKPOINT_EVERY = 10
RESUME_FROM_CHECKPOINT = os.getenv("RESUME_FROM_CHECKPOINT")
TRAINER_VARIANT = os.getenv("TRAINER_VARIANT", "grpo").strip().lower()
PROMPT_MODE = os.getenv("PROMPT_MODE", "default").strip().lower()
PROMPT_TEMPLATE_PATH = os.getenv(
    "PROMPT_TEMPLATE_PATH",
    (
        ENVIRONMENT_BELIEF_PROMPT_TEMPLATE_PATH
        if TRAINER_VARIANT == "environment_belief"
        else ARTIST_PROMPT_TEMPLATE_PATH
    ),
)
DAPO_EPSILON = float(os.getenv("DAPO_EPSILON", "0.2"))
DAPO_EPSILON_HIGH = float(os.getenv("DAPO_EPSILON_HIGH", "0.28"))
DAPO_DYNAMIC_SAMPLING = os.getenv("DAPO_DYNAMIC_SAMPLING", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
DAPO_DYNAMIC_SAMPLING_MIN_STD = float(os.getenv("DAPO_DYNAMIC_SAMPLING_MIN_STD", "1e-6"))
DAPO_FILTER_OVERLONG = os.getenv("DAPO_FILTER_OVERLONG", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
AGENTFLOW_EPSILON = float(os.getenv("AGENTFLOW_EPSILON", "0.2"))
AGENTFLOW_EPSILON_HIGH = float(os.getenv("AGENTFLOW_EPSILON_HIGH", "0.3"))
AGENTFLOW_ROLE_API_BASE = os.getenv("AGENTFLOW_ROLE_API_BASE", JUDGE_API_BASE)
AGENTFLOW_ROLE_API_KEY = os.getenv(
    "AGENTFLOW_ROLE_API_KEY",
    JUDGE_API_KEY or "EMPTY",
)
AGENTFLOW_ROLE_MODEL = os.getenv("AGENTFLOW_ROLE_MODEL", JUDGE_MODEL)
AGENTFLOW_ANALYSIS_MODEL = os.getenv(
    "AGENTFLOW_ANALYSIS_MODEL",
    AGENTFLOW_ROLE_MODEL,
)
AGENTFLOW_EXECUTOR_MODEL = os.getenv(
    "AGENTFLOW_EXECUTOR_MODEL",
    AGENTFLOW_ROLE_MODEL,
)
AGENTFLOW_VERIFIER_MODEL = os.getenv(
    "AGENTFLOW_VERIFIER_MODEL",
    AGENTFLOW_ROLE_MODEL,
)
AGENTFLOW_GENERATOR_MODEL = os.getenv(
    "AGENTFLOW_GENERATOR_MODEL",
    AGENTFLOW_ROLE_MODEL,
)
AGENTFLOW_ROLE_TIMEOUT = int(os.getenv("AGENTFLOW_ROLE_TIMEOUT", "60"))
AGENTFLOW_MEMORY_MAX_CHARS = int(
    os.getenv("AGENTFLOW_MEMORY_MAX_CHARS", "24000")
)
ENTROPY_EXPLOIT_REWARD_THRESHOLD = float(os.getenv("ENTROPY_EXPLOIT_REWARD_THRESHOLD", "1.0"))
ENTROPY_CONSISTENCY_WEIGHT = float(os.getenv("ENTROPY_CONSISTENCY_WEIGHT", "0.25"))
ENTROPY_EXPLORATION_WEIGHT = float(os.getenv("ENTROPY_EXPLORATION_WEIGHT", "0.10"))
ENTROPY_COLLAPSE_WEIGHT = float(os.getenv("ENTROPY_COLLAPSE_WEIGHT", "0.10"))
ENTROPY_ADJUSTMENT_CLIP = float(os.getenv("ENTROPY_ADJUSTMENT_CLIP", "0.50"))
ENTROPY_EQUIVALENCE_CONFIDENCE_THRESHOLD = float(
    os.getenv("ENTROPY_EQUIVALENCE_CONFIDENCE_THRESHOLD", "0.75")
)
BELIEF_BRANCH_ENABLED = os.getenv(
    "BELIEF_BRANCH_ENABLED",
    "true",
).strip().lower() in {"1", "true", "yes", "on"}
BELIEF_NUM_COUNTERFACTUAL_BRANCHES = int(
    os.getenv("BELIEF_NUM_COUNTERFACTUAL_BRANCHES", "2")
)
BELIEF_MAX_BRANCH_RESAMPLES = int(
    os.getenv("BELIEF_MAX_BRANCH_RESAMPLES", "2")
)
BELIEF_MAX_BRANCH_POINTS = int(
    os.getenv("BELIEF_MAX_BRANCH_POINTS", "1")
)
BELIEF_TASK_REWARD_THRESHOLD = float(
    os.getenv("BELIEF_TASK_REWARD_THRESHOLD", "1.5")
)
BELIEF_MAX_SEMANTIC_ACTION_ENTROPY = float(
    os.getenv("BELIEF_MAX_SEMANTIC_ACTION_ENTROPY", "0.25")
)
BELIEF_MIN_STATE_SUPPORT = int(os.getenv("BELIEF_MIN_STATE_SUPPORT", "2"))
BELIEF_REWARD_DEFICIT_WEIGHT = float(
    os.getenv("BELIEF_REWARD_DEFICIT_WEIGHT", "1.0")
)
BELIEF_UNCERTAINTY_WEIGHT = float(
    os.getenv("BELIEF_UNCERTAINTY_WEIGHT", "0.5")
)
BELIEF_COLLAPSE_WEIGHT = float(os.getenv("BELIEF_COLLAPSE_WEIGHT", "1.0"))
BELIEF_ERROR_WEIGHT = float(os.getenv("BELIEF_ERROR_WEIGHT", "0.5"))
BELIEF_UNCERTAINTY_THRESHOLD = float(
    os.getenv("BELIEF_UNCERTAINTY_THRESHOLD", "0.5")
)
BELIEF_REWARD_WEIGHT = float(
    os.getenv("BELIEF_REWARD_WEIGHT", "0.5")
)
BELIEF_STRICT_REPLAY = os.getenv(
    "BELIEF_STRICT_REPLAY",
    "true",
).strip().lower() in {"1", "true", "yes", "on"}
BELIEF_STRICT = os.getenv("BELIEF_STRICT", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
BELIEF_JUDGE_CACHE_PATH = os.getenv(
    "BELIEF_JUDGE_CACHE_PATH",
    os.path.join(
        CHECKPOINT_DIR,
        "judge_cache",
        "belief_judge_cache.jsonl",
    ),
)
BELIEF_JUDGE_TIMEOUT = int(os.getenv("BELIEF_JUDGE_TIMEOUT", "30"))
BELIEF_MAX_JUDGE_CONTEXT_CHARS = int(
    os.getenv("BELIEF_MAX_JUDGE_CONTEXT_CHARS", "6000")
)

# NEW: Multi-GPU config
USE_MULTI_GPU = True  # Set to True when using 4 GPUs
NUM_GPUS = 6 if USE_MULTI_GPU else 1


def resolve_training_components():
    if TRAIN_ENVIRONMENT == "emulation":
        return (
            "emulation",
            load_emulation_system_tasks,
            create_emulation_system_tool_environment,
        )

    return (
        "enterprise",
        load_enterprise_tasks_v2,
        create_enterprise_tool_environment,
    )


def resolve_reward_mode() -> str:
    if REWARD_MODE:
        return REWARD_MODE
    return "checkpoint"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EnterpriseBench agentic RL.")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--default-prompt",
        action="store_true",
        help="Use the built-in default ReAct prompt.",
    )
    prompt_group.add_argument(
        "--artist-prompt",
        action="store_true",
        help="Use the exact ARTIST prompt template from the reference file.",
    )
    return parser.parse_args()


def resolve_prompt_mode(args: argparse.Namespace) -> tuple[str, str | None]:
    # For Cohere models, use Cohere-specific prompt by default
    if MODEL_SERIES in ["cohere", "command-r"]:
        if args.default_prompt:
            return "default", None
        # Use Cohere prompt template
        return "cohere", COHERE_PROMPT_TEMPLATE_PATH

    # For other models (Qwen, Gemma)
    if args.default_prompt:
        return "default", None
    if args.artist_prompt:
        return "artist", PROMPT_TEMPLATE_PATH
    if PROMPT_MODE == "artist":
        return "artist", PROMPT_TEMPLATE_PATH
    return "default", None


def is_gemma4_model(model_series: str, model_name: str) -> bool:
    normalized_series = (model_series or "").strip().lower()
    normalized_name = (model_name or "").strip().lower()
    return normalized_series in {"gemma", "gemma4"} or "gemma-4" in normalized_name


def load_tokenizer_for_model(model_path: str, use_gemma_loader: bool):
    if use_gemma_loader:
        if AutoProcessor is None:
            raise ImportError(
                "Gemma 4 loading requires transformers with AutoProcessor support."
            )
        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                cache_dir=MODEL_CACHE_DIR,
                trust_remote_code=True,
            )
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None:
                return tokenizer
            logger.warning(
                "Gemma 4 AutoProcessor loaded without tokenizer; falling back to AutoTokenizer."
            )
        except Exception as exc:
            logger.warning(
                "Gemma 4 AutoProcessor failed (%s: %s); falling back to AutoTokenizer for text-only training.",
                exc.__class__.__name__,
                exc,
            )
    return AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=MODEL_CACHE_DIR,
        trust_remote_code=True,
        fix_mistral_regex=True,  # Fix Cohere tokenizer regex issue
    )


def load_causal_or_gemma_model(
    model_path: str,
    device_map: str,
    low_cpu_mem_usage: bool = False,
):
    common_kwargs = {
        "cache_dir": MODEL_CACHE_DIR,
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if low_cpu_mem_usage:
        common_kwargs["low_cpu_mem_usage"] = True

    if is_gemma4_model(MODEL_SERIES, model_path):
        if AutoModelForImageTextToText is None:
            raise ImportError(
                "Gemma 4 loading requires transformers with AutoModelForImageTextToText support."
            )
        return AutoModelForImageTextToText.from_pretrained(model_path, **common_kwargs)
    return AutoModelForCausalLM.from_pretrained(model_path, **common_kwargs)

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training loop."""
    args = parse_args()
    prompt_mode, prompt_template_path = resolve_prompt_mode(args)
    if TRAINER_VARIANT == "environment_belief" and prompt_mode != "artist":
        raise ValueError(
            "environment_belief requires the belief-aware ARTIST prompt; "
            "do not use --default-prompt"
        )
    environment_name, dataset_loader_fn, tool_env_factory = resolve_training_components()

    # ========================================================================
    # DETECT MODEL TYPE AND GET APPROPRIATE COMPONENTS
    # ========================================================================
    model_type = detect_model_type(MODEL_SERIES, MODEL_NAME)
    logger.info(f"Detected model type: {model_type}")

    # Get model-specific components
    RolloutManagerClass = get_rollout_manager_class(model_type)
    if TRAINER_VARIANT == "agentflow":
        RolloutManagerClass = AgentFlowRolloutManager
    lora_target_modules = get_lora_target_modules(
        model_type,
        model_series=MODEL_SERIES,
        model_name=MODEL_NAME,
    )
    logger.info(f"Rollout Manager Class: {RolloutManagerClass.__name__}")
    logger.info(f"LoRA Target Modules: {lora_target_modules}")

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
    logger.info(f"Prompt Mode: {prompt_mode}")
    logger.info(f"Trainer Variant: {TRAINER_VARIANT}")
    if TRAINER_VARIANT == "entropy":
        logger.info(
            "Entropy trainer config: threshold=%.3f consistency=%.3f exploration=%.3f "
            "collapse=%.3f clip=%.3f equiv_conf=%.2f",
            ENTROPY_EXPLOIT_REWARD_THRESHOLD,
            ENTROPY_CONSISTENCY_WEIGHT,
            ENTROPY_EXPLORATION_WEIGHT,
            ENTROPY_COLLAPSE_WEIGHT,
            ENTROPY_ADJUSTMENT_CLIP,
            ENTROPY_EQUIVALENCE_CONFIDENCE_THRESHOLD,
        )
    elif TRAINER_VARIANT == "agentflow":
        logger.info(
            "AgentFlow config: epsilon=%.3f epsilon_high=%.3f beta=%.4f "
            "fixed_model=%s fixed_api=%s memory_chars=%d",
            AGENTFLOW_EPSILON,
            AGENTFLOW_EPSILON_HIGH,
            BETA,
            AGENTFLOW_ROLE_MODEL,
            AGENTFLOW_ROLE_API_BASE,
            AGENTFLOW_MEMORY_MAX_CHARS,
        )
    elif TRAINER_VARIANT == "environment_belief":
        logger.info(
            "Environment-belief config: children=%d resamples=%d "
            "points=%d enabled=%s "
            "task_threshold=%.3f max_action_entropy=%.3f "
            "belief_weight=%.3f judge_cache=%s",
            BELIEF_NUM_COUNTERFACTUAL_BRANCHES,
            BELIEF_MAX_BRANCH_RESAMPLES,
            BELIEF_MAX_BRANCH_POINTS,
            BELIEF_BRANCH_ENABLED,
            BELIEF_TASK_REWARD_THRESHOLD,
            BELIEF_MAX_SEMANTIC_ACTION_ENTROPY,
            BELIEF_REWARD_WEIGHT,
            BELIEF_JUDGE_CACHE_PATH,
        )
        logger.info(
            "  selector weights: deficit=%.3f collapse=%.3f "
            "uncertainty=%.3f error=%.3f",
            BELIEF_REWARD_DEFICIT_WEIGHT,
            BELIEF_COLLAPSE_WEIGHT,
            BELIEF_UNCERTAINTY_WEIGHT,
            BELIEF_ERROR_WEIGHT,
        )
    logger.info(f"Training Environment: {environment_name}")
    logger.info(f"Dataset Path: {DATASET_PATH}")
    if environment_name == "emulation":
        logger.info(f"Emulation App Root: {EMULATION_APP_ROOT}")
        logger.info(f"Emulation App Roots: {os.getenv('EMULATION_APP_ROOTS', EMULATION_APP_ROOT)}")
        logger.info(f"Emulation MCP URLs: {os.getenv('EMULATION_MCP_URLS', os.getenv('EMULATION_MCP_URL', ''))}")
        logger.info(f"Emulation API URLs: {os.getenv('EMULATION_API_URLS', os.getenv('EMULATION_API_URL', ''))}")
    logger.info(f"LoRA: {'ENABLED' if USE_LORA else 'DISABLED'}")
    if USE_LORA:
        logger.info(f"  - Rank: {LORA_R}, Alpha: {LORA_ALPHA}")
        logger.info(f"  - Target modules: {lora_target_modules}")
    logger.info(f"Group Size: {GROUP_SIZE}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Epochs: {NUM_EPOCHS}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"Beta (KL coeff): {BETA}")
    logger.info(f"Max new tokens per turn: {MAX_NEW_TOKENS}")
    logger.info(f"Resume Checkpoint: {RESUME_FROM_CHECKPOINT or 'None'}")
    reward_mode = resolve_reward_mode()
    logger.info(f"Reward System: {reward_mode}")
    if reward_mode in {"agentic", "checkpoint"}:
        logger.info(f"  - Judge Model: {JUDGE_MODEL.split('/')[-1]}")
    if reward_mode == "agentic":
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
    use_gemma_loader = is_gemma4_model(MODEL_SERIES, model_load_path)
    if use_gemma_loader:
        logger.info("Using Gemma 4 conditional-generation loader")
    model = load_causal_or_gemma_model(
        model_load_path,
        device_map="auto",
    )

    tokenizer_load_path = MODEL_NAME if USE_LORA or not RESUME_FROM_CHECKPOINT else RESUME_FROM_CHECKPOINT
    tokenizer = load_tokenizer_for_model(tokenizer_load_path, use_gemma_loader)
    
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
            target_modules=lora_target_modules,  # Use model-specific targets from factory
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
        model.train()
        model.print_trainable_parameters()
        
        # Verify trainable params
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        if trainable_params == 0:
            logger.error("ERROR: No trainable parameters after LoRA!")
            return

        if is_gemma4_model(MODEL_SERIES, MODEL_NAME):
            lora_parameter_names = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and "lora_" in name
            ]
            language_lora_names = [
                name for name in lora_parameter_names if "language_model" in name
            ]
            non_language_lora_names = [
                name for name in lora_parameter_names if "language_model" not in name
            ]
            if not language_lora_names or non_language_lora_names:
                raise RuntimeError(
                    "Gemma 4 LoRA target validation failed: expected all trainable "
                    "adapter parameters under language_model, found "
                    f"language={len(language_lora_names)} and "
                    f"non_language={len(non_language_lora_names)}. "
                    "Do not resume a checkpoint created with the old '.linear' "
                    "audio/vision target configuration."
                )
            logger.info(
                "Validated Gemma 4 text LoRA targets: %d trainable adapter tensors "
                "under language_model",
                len(language_lora_names),
            )
    
    # ========================================================================
    # 3. Create reference model (frozen, NO LoRA)
    # ========================================================================

    if REF_MODEL_DEVICE == "cpu":
        logger.info("Creating reference model (frozen, no LoRA) - offloading to CPU...")
        ref_model = load_causal_or_gemma_model(
            MODEL_NAME,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        logger.info("Reference model created and frozen on CPU (saves ~18GB GPU memory)")
    else:
        logger.info("Creating reference model (frozen, no LoRA) - keeping on GPU...")
        ref_model = load_causal_or_gemma_model(
            MODEL_NAME,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        logger.info("Reference model created and frozen on GPU (faster KL computation)")

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
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

    train_dataset = dataset_loader_fn(
        path=DATASET_PATH,
        max_tasks=MAX_TASKS,
        difficulty_filter=DIFFICULTY_FILTER,
        domain_filter=DOMAIN_FILTER,
        min_steps=MIN_STEPS,
        max_steps=MAX_STEPS,
        shuffle=SHUFFLE_DATASET,
        seed=SHUFFLE_SEED
    )

    if SHUFFLE_DATASET:
        if SHUFFLE_SEED is not None:
            logger.info(f"Dataset shuffled with seed={SHUFFLE_SEED}")
        else:
            logger.info("Dataset shuffled with random seed")
    else:
        logger.info("Dataset NOT shuffled (keeping original order)")

    logger.info(f"Loaded {len(train_dataset)} training tasks")
    
    if len(train_dataset) == 0:
        logger.error("No tasks loaded! Check your dataset path and filters.")
        return
    
    # ========================================================================
    # 6. Create reward function
    # ========================================================================

    if reward_mode == "checkpoint":
        logger.info("Creating checkpoint-grounded trajectory reward function...")
        logger.info(f"  - Judge: {JUDGE_MODEL}")
        logger.info(f"  - API: {JUDGE_API_BASE}")
        logger.info(f"  - Cache: {REWARD_CACHE_PATH}")

        reward_fn = create_checkpoint_reward_function(
            tasks=train_dataset,
            judge_api_base=JUDGE_API_BASE,
            judge_model=JUDGE_MODEL,
            judge_api_key=JUDGE_API_KEY,
            cache_path=REWARD_CACHE_PATH,
            verbose=True,
        )
    elif reward_mode == "agentic":
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
    logger.info(f"Using Rollout Manager: {RolloutManagerClass.__name__}")

    belief_judge = None
    branch_selector = None
    if TRAINER_VARIANT == "environment_belief":
        belief_judge = BeliefJudge(
            api_base=JUDGE_API_BASE,
            model_name=JUDGE_MODEL,
            api_key=JUDGE_API_KEY or "EMPTY",
            cache_path=BELIEF_JUDGE_CACHE_PATH,
            timeout=BELIEF_JUDGE_TIMEOUT,
        )
        branch_selector = BranchSelector(
            BranchSelectorConfig(
                enabled=BELIEF_BRANCH_ENABLED,
                max_branch_points=BELIEF_MAX_BRANCH_POINTS,
                max_task_reward=BELIEF_TASK_REWARD_THRESHOLD,
                max_semantic_action_entropy=(
                    BELIEF_MAX_SEMANTIC_ACTION_ENTROPY
                ),
                min_state_support=BELIEF_MIN_STATE_SUPPORT,
                reward_deficit_weight=BELIEF_REWARD_DEFICIT_WEIGHT,
                collapse_weight=BELIEF_COLLAPSE_WEIGHT,
                belief_uncertainty_weight=BELIEF_UNCERTAINTY_WEIGHT,
                belief_error_weight=BELIEF_ERROR_WEIGHT,
                uncertainty_threshold=BELIEF_UNCERTAINTY_THRESHOLD,
            ),
            belief_equivalence=belief_judge.beliefs_equivalent,
        )

    rollout_manager_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "tool_env_factory": tool_env_factory,
        "max_turns": MAX_TURNS,
        "max_tool_output_tokens": MAX_TOOL_OUTPUT_TOKENS,
        "max_context_length": MAX_CONTEXT_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "device": device,
        "prompt_mode": prompt_mode,
        "prompt_template_path": prompt_template_path,
    }
    if TRAINER_VARIANT == "agentflow":
        rollout_manager_kwargs.update(
            {
                "role_api_base": AGENTFLOW_ROLE_API_BASE,
                "role_api_key": AGENTFLOW_ROLE_API_KEY,
                "role_model": AGENTFLOW_ROLE_MODEL,
                "analysis_model": AGENTFLOW_ANALYSIS_MODEL,
                "executor_model": AGENTFLOW_EXECUTOR_MODEL,
                "verifier_model": AGENTFLOW_VERIFIER_MODEL,
                "generator_model": AGENTFLOW_GENERATOR_MODEL,
                "role_timeout": AGENTFLOW_ROLE_TIMEOUT,
                "memory_max_chars": AGENTFLOW_MEMORY_MAX_CHARS,
            }
        )
    base_rollout_manager = RolloutManagerClass(
        **rollout_manager_kwargs
    )
    if TRAINER_VARIANT == "environment_belief":
        rollout_manager = EnvironmentBeliefRolloutManager(
            base_manager=base_rollout_manager,
            reward_function=reward_fn,
            belief_judge=belief_judge,
            branch_selector=branch_selector,
            environment_adapter=GenericEnvironmentAdapter(
                strict_replay=BELIEF_STRICT_REPLAY,
            ),
            belief_reward_weight=BELIEF_REWARD_WEIGHT,
            num_counterfactual_branches=(
                BELIEF_NUM_COUNTERFACTUAL_BRANCHES
            ),
            max_branch_resamples=BELIEF_MAX_BRANCH_RESAMPLES,
            strict=BELIEF_STRICT,
            max_judge_context_chars=BELIEF_MAX_JUDGE_CONTEXT_CHARS,
        )
    else:
        rollout_manager = base_rollout_manager

    # Enable tool filtering if configured
    if ENABLE_TOOL_FILTERING:
        rollout_manager.enable_tool_filtering = True
        rollout_manager.num_random_tools = NUM_RANDOM_TOOLS
        logger.info(f"Tool filtering enabled: gold tools + {NUM_RANDOM_TOOLS} random tools per task")
        logger.info(f"Expected prompt reduction: 162 tools (~12k tokens) → ~{NUM_RANDOM_TOOLS + 10} tools (~{(NUM_RANDOM_TOOLS + 10) * 74} tokens)")
    else:
        rollout_manager.enable_tool_filtering = False
        logger.info("Tool filtering disabled: showing all 162 tools per task")

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
    
    trainer_cls = {
        "grpo": GRPOTrainer,
        "dapo": DAPOTrainer,
        "entropy": EntropyTrainer,
        "agentflow": AgentFlowTrainer,
        "environment_belief": EnvironmentBeliefTrainer,
    }.get(TRAINER_VARIANT, GRPOTrainer)
    trainer_kwargs = {
        "model": model,
        "ref_model": ref_model,
        "optimizer": optimizer,
        "tokenizer": tokenizer,
        "reward_function": reward_fn,
        "beta": BETA,
        "max_grad_norm": 1.0,
        "device": device,
        "logprob_chunk_size": 64,
        "use_wandb": USE_WANDB,
    }
    if TRAINER_VARIANT in {"dapo", "entropy", "environment_belief"}:
        trainer_kwargs.update(
            {
                "epsilon": DAPO_EPSILON,
                "epsilon_high": DAPO_EPSILON_HIGH,
                "dynamic_sampling": DAPO_DYNAMIC_SAMPLING,
                "dynamic_sampling_min_std": DAPO_DYNAMIC_SAMPLING_MIN_STD,
                "filter_overlong": DAPO_FILTER_OVERLONG,
            }
        )
    if TRAINER_VARIANT == "agentflow":
        trainer_kwargs.update(
            {
                "epsilon": AGENTFLOW_EPSILON,
                "epsilon_high": AGENTFLOW_EPSILON_HIGH,
            }
        )
    if TRAINER_VARIANT == "entropy":
        trainer_kwargs.update(
            {
                "semantic_exploit_reward_threshold": ENTROPY_EXPLOIT_REWARD_THRESHOLD,
                "semantic_consistency_weight": ENTROPY_CONSISTENCY_WEIGHT,
                "semantic_exploration_weight": ENTROPY_EXPLORATION_WEIGHT,
                "semantic_collapse_weight": ENTROPY_COLLAPSE_WEIGHT,
                "semantic_adjustment_clip": ENTROPY_ADJUSTMENT_CLIP,
                "semantic_equivalence_confidence_threshold": ENTROPY_EQUIVALENCE_CONFIDENCE_THRESHOLD,
                "semantic_judge_api_base": JUDGE_API_BASE,
                "semantic_judge_model": JUDGE_MODEL,
                "semantic_judge_api_key": JUDGE_API_KEY or "EMPTY",
            }
        )
    if TRAINER_VARIANT == "environment_belief":
        trainer_kwargs.update(
            {
                "belief_reward_weight": BELIEF_REWARD_WEIGHT,
            }
        )
    trainer = trainer_cls(**trainer_kwargs)
    
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
