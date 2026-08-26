"""
Model Factory - Component Selection

Selects appropriate components (rollout manager, LoRA targets) based on model type.
Simplifies train_enterprise.py by centralizing model detection logic.
"""

import logging
from typing import Type, Union

# Import rollout managers
from rollout_manager import AgenticRolloutManager
from rollout_manager_35 import AgenticRolloutManager35
from rollout_manager_cohere import CohereAgenticRolloutManager

logger = logging.getLogger(__name__)

GEMMA4_TEXT_LORA_TARGET_PATTERN = (
    r".*language_model.*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


class ModelType:
    """Model type identifiers."""
    QWEN3 = "qwen3"
    QWEN35 = "qwen3.5"
    GEMMA = "gemma"
    COHERE = "cohere"


def detect_model_type(model_series: str, model_name: str) -> str:
    """
    Detect model type from series name or model path.

    Args:
        model_series: Environment variable MODEL_SERIES
        model_name: Model path or HuggingFace model name

    Returns:
        One of ModelType constants
    """
    series_lower = (model_series or "").strip().lower()
    name_lower = (model_name or "").strip().lower()

    # Check Cohere Command R
    if series_lower in {"cohere", "command-r", "command_r"}:
        logger.info("Model type detected: Cohere Command R")
        return ModelType.COHERE

    if any(x in name_lower for x in ["command-r", "c4ai-command", "cohere"]):
        logger.info("Model type detected: Cohere Command R (from name)")
        return ModelType.COHERE

    # Check Gemma
    if series_lower in {"gemma", "gemma4"}:
        logger.info("Model type detected: Gemma")
        return ModelType.GEMMA

    if "gemma" in name_lower:
        logger.info("Model type detected: Gemma (from name)")
        return ModelType.GEMMA

    # Check Qwen 3.5
    if series_lower == "qwen3.5":
        logger.info("Model type detected: Qwen 3.5")
        return ModelType.QWEN35

    if "qwen3.5" in name_lower or "qwen-3.5" in name_lower:
        logger.info("Model type detected: Qwen 3.5 (from name)")
        return ModelType.QWEN35

    # Default to Qwen 3
    logger.info("Model type detected: Qwen 3 (default)")
    return ModelType.QWEN3


def get_rollout_manager_class(model_type: str) -> Type:
    """
    Get the appropriate rollout manager class for a model type.

    Args:
        model_type: One of ModelType constants

    Returns:
        RolloutManager class (not instantiated)
    """
    if model_type == ModelType.COHERE:
        logger.info("Using CohereAgenticRolloutManager")
        return CohereAgenticRolloutManager

    elif model_type == ModelType.QWEN35:
        logger.info("Using AgenticRolloutManager35")
        return AgenticRolloutManager35

    elif model_type == ModelType.GEMMA:
        logger.info("Using AgenticRolloutManager (Gemma)")
        return AgenticRolloutManager

    else:  # QWEN3 or default
        logger.info("Using AgenticRolloutManager (default)")
        return AgenticRolloutManager


def get_lora_target_modules(
    model_type: str,
    model_series: str = "",
    model_name: str = "",
) -> Union[list, str]:
    """
    Get LoRA target modules for a model type.

    Different models have different attention/MLP projection naming.

    Args:
        model_type: One of ModelType constants
        model_series: Model series used to distinguish Gemma 4.
        model_name: Model path or HuggingFace model name.

    Returns:
        List of module names or a full module-name regex to target with LoRA
    """
    if model_type == ModelType.COHERE:
        # Cohere Command R uses standard transformer architecture
        logger.info("Using Cohere LoRA targets")
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]

    elif model_type == ModelType.GEMMA:
        gemma_identity = f"{model_series} {model_name}".lower()
        if "gemma-4" in gemma_identity or "gemma4" in gemma_identity:
            # Gemma 4's audio/vision towers wrap projections in `.linear`, but
            # its text model uses direct nn.Linear projections. Restrict the
            # regex to language_model so text-only training cannot silently
            # attach every adapter to unused modality towers.
            logger.info("Using Gemma 4 text-language-model LoRA targets")
            return GEMMA4_TEXT_LORA_TARGET_PATTERN

        logger.info("Using standard Gemma text LoRA targets")
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    else:  # Qwen 3 / 3.5
        logger.info("Using Qwen LoRA targets")
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]


def get_model_info(model_type: str) -> dict:
    """
    Get additional model-specific information.

    Args:
        model_type: One of ModelType constants

    Returns:
        Dictionary with model info
    """
    info = {
        ModelType.COHERE: {
            "default_context_length": 128_000,
            "supports_native_tool_calling": True,
            "chat_template_required": True,
            "special_tokens": ["<BOS_TOKEN>", "<EOS_TOKEN>"],
        },
        ModelType.GEMMA: {
            "default_context_length": 32_768,
            "supports_native_tool_calling": False,
            "chat_template_required": True,
            "special_tokens": ["<bos>", "<eos>"],
        },
        ModelType.QWEN35: {
            "default_context_length": 32_768,
            "supports_native_tool_calling": False,
            "chat_template_required": True,
            "special_tokens": ["<|im_start|>", "<|im_end|>"],
        },
        ModelType.QWEN3: {
            "default_context_length": 32_768,
            "supports_native_tool_calling": False,
            "chat_template_required": True,
            "special_tokens": ["<|im_start|>", "<|im_end|>"],
        },
    }

    return info.get(model_type, {
        "default_context_length": 16_384,
        "supports_native_tool_calling": False,
        "chat_template_required": False,
        "special_tokens": [],
    })


def validate_model_compatibility(model_type: str, model_name: str) -> bool:
    """
    Validate that a model is compatible with the training system.

    Args:
        model_type: Detected model type
        model_name: Model path or name

    Returns:
        True if compatible, False otherwise
    """
    # All current model types are compatible
    supported_types = {
        ModelType.QWEN3,
        ModelType.QWEN35,
        ModelType.GEMMA,
        ModelType.COHERE
    }

    if model_type not in supported_types:
        logger.warning(
            f"Model type '{model_type}' is not officially supported. "
            f"Supported types: {supported_types}"
        )
        return False

    logger.info(f"Model '{model_name}' (type: {model_type}) is compatible")
    return True
