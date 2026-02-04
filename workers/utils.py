"""Utility functions for MRL workers.

Note: torch is imported lazily to allow this module to be imported
in environments without PyTorch (e.g., Modal Sandboxes for reward computation).
"""

import io
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Set, Tuple

from MRL.logging_config import get_logger

# Lazy import for torch - allows module to be imported without torch
if TYPE_CHECKING:
    import torch

logger = get_logger("workers.utils")


def _get_torch():
    """Lazily import torch."""
    import torch
    return torch


def detect_tied_weights(state_dict: Dict[str, Any]) -> Set[str]:
    """Detect tied weights in a state dict by checking for identical data pointers.

    Tied weights share the same underlying tensor (e.g., embed_tokens and lm_head
    in many LLMs). This is important for serialization to avoid duplicates.

    Args:
        state_dict: Model state dictionary (torch tensors)

    Returns:
        Set of parameter names that are tied (excluding the "canonical" first occurrence)
    """
    seen_data_ptrs: Dict[int, str] = {}
    tied_params: Set[str] = set()

    for name, param in state_dict.items():
        data_ptr = param.data_ptr()
        if data_ptr in seen_data_ptrs:
            tied_params.add(name)
            logger.debug(
                f"Detected tied weight: {name} shares data with {seen_data_ptrs[data_ptr]}"
            )
        else:
            seen_data_ptrs[data_ptr] = name

    if tied_params:
        logger.info(f"Found {len(tied_params)} tied parameters: {tied_params}")

    return tied_params


def clean_state_dict_for_vllm(
    state_dict: Dict[str, Any],
    skip_tied_weights: bool = True,
    skip_lora_weights: bool = True,
) -> Dict[str, Any]:
    """Clean state dict for vLLM compatibility.

    This function:
    1. Removes tied weight duplicates (same tensor referenced multiple times)
    2. Removes LoRA-specific parameters (lora_A, lora_B, lora_embedding_A, etc.)
    3. Removes prefix wrappers (base_model.model., module., etc.)
    4. Removes .base_layer from LoRA-wrapped keys

    Args:
        state_dict: Raw model state dictionary
        skip_tied_weights: If True, skip duplicate tied weights
        skip_lora_weights: If True, skip LoRA-specific parameters

    Returns:
        Cleaned state dictionary
    """
    # Detect tied weights
    tied_weights = detect_tied_weights(state_dict) if skip_tied_weights else set()

    # LoRA-specific patterns to filter out
    lora_patterns = [
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_magnitude_vector",
        ".lora_",  # Catches any lora_ substring
    ]

    # Prefix patterns to remove
    prefix_patterns = [
        "base_model.model.",
        "base_model.",
        "_checkpoint_wrapped_module.",
        "module.",
    ]

    cleaned_dict = {}
    skipped_tied = 0
    skipped_lora = 0

    for key, value in state_dict.items():
        # Skip tied weights (duplicates)
        if key in tied_weights:
            skipped_tied += 1
            continue

        # Skip LoRA-specific parameters
        if skip_lora_weights:
            is_lora = any(pattern in key for pattern in lora_patterns)
            if is_lora:
                skipped_lora += 1
                continue

        # Clean up key name
        clean_key = key

        # Remove prefix wrappers
        for prefix in prefix_patterns:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]

        # Remove .base_layer from LoRA-wrapped keys
        clean_key = clean_key.replace(".base_layer", "")

        cleaned_dict[clean_key] = value

    if skipped_tied > 0:
        logger.info(f"Skipped {skipped_tied} tied weight parameters")
    if skipped_lora > 0:
        logger.info(f"Skipped {skipped_lora} LoRA-specific parameters")

    return cleaned_dict


def serialize_state_dict(
    state_dict: Dict[str, Any],
    skip_tied_weights: bool = True,
) -> bytes:
    """Serialize state dict to bytes, handling tied weights.

    Args:
        state_dict: Model state dictionary
        skip_tied_weights: If True, skip duplicate tied weights

    Returns:
        Serialized bytes
    """
    torch = _get_torch()

    # Detect and skip tied weights
    if skip_tied_weights:
        tied_weights = detect_tied_weights(state_dict)
        state_dict = {k: v for k, v in state_dict.items() if k not in tied_weights}

    # Serialize
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def create_weights_iterator(
    state_dict: Dict[str, Any],
    device: str = "cuda",
) -> Iterator[Tuple[str, Any]]:
    """Create a weights iterator from state dict for vLLM reload_weights.

    Args:
        state_dict: Model state dictionary
        device: Target device for tensors

    Yields:
        Tuples of (name, tensor) for each parameter
    """
    # Clean the state dict first
    cleaned_dict = clean_state_dict_for_vllm(state_dict)

    for name, param in cleaned_dict.items():
        # Ensure tensor is on correct device and contiguous
        if param.device.type != device.split(":")[0]:
            param = param.to(device)
        yield (name, param.contiguous())


def validate_rewards(
    rewards: list,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    warn_on_outliers: bool = True,
) -> list:
    """Validate and optionally clip reward values.

    Args:
        rewards: List of reward values
        min_value: Minimum valid reward (None = no limit)
        max_value: Maximum valid reward (None = no limit)
        warn_on_outliers: Log warning for extreme values

    Returns:
        Validated (and possibly clipped) rewards
    """
    import numpy as np

    rewards_array = np.array(rewards)

    # Check for invalid values
    if np.any(np.isnan(rewards_array)):
        logger.warning("NaN values detected in rewards, replacing with 0")
        rewards_array = np.nan_to_num(rewards_array, nan=0.0)

    if np.any(np.isinf(rewards_array)):
        logger.warning("Inf values detected in rewards, clipping to finite range")
        rewards_array = np.clip(rewards_array, -1e10, 1e10)

    # Warn about outliers
    if warn_on_outliers:
        mean = np.mean(rewards_array)
        std = np.std(rewards_array)
        if std > 0:
            z_scores = np.abs((rewards_array - mean) / std)
            outliers = np.sum(z_scores > 3)
            if outliers > 0:
                logger.warning(
                    f"Found {outliers} reward outliers (>3 std from mean). "
                    f"Mean={mean:.4f}, Std={std:.4f}"
                )

    # Clip to valid range if specified
    if min_value is not None or max_value is not None:
        original_min, original_max = rewards_array.min(), rewards_array.max()
        rewards_array = np.clip(
            rewards_array,
            min_value if min_value is not None else -np.inf,
            max_value if max_value is not None else np.inf,
        )
        if original_min != rewards_array.min() or original_max != rewards_array.max():
            logger.info(
                f"Clipped rewards from [{original_min:.4f}, {original_max:.4f}] "
                f"to [{rewards_array.min():.4f}, {rewards_array.max():.4f}]"
            )

    return rewards_array.tolist()
