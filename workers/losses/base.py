"""Base utilities for loss computation."""

from dataclasses import dataclass
from typing import Optional

import torch

from MRL.logging_config import get_logger

logger = get_logger("losses.base")

# Constants
EPSILON = 1e-8  # Small constant for numerical stability


@dataclass
class LossResult:
    """Result of loss computation with metrics."""

    loss: torch.Tensor
    metrics: dict

    def to_dict(self) -> dict:
        """Convert to dictionary with loss value."""
        return {
            "loss": self.loss.item() if isinstance(self.loss, torch.Tensor) else self.loss,
            **self.metrics,
        }


def compute_advantages(
    rewards: torch.Tensor,
    scale_rewards: str = "group",
    num_generations: int = 4,
) -> torch.Tensor:
    """Compute advantages from rewards with optional normalization.

    Args:
        rewards: Tensor of shape (batch_size,) containing rewards
        scale_rewards: Normalization type - "group", "batch", or "none"
        num_generations: Number of generations per prompt (for group normalization)

    Returns:
        Tensor of advantages with same shape as rewards
    """
    batch_size = rewards.shape[0]

    if scale_rewards == "group" and num_generations > 1:
        # Group-level advantage normalization (GRPO style)
        num_groups = batch_size // num_generations
        if num_groups == 0:
            # Fall back to batch normalization if batch is smaller than num_generations
            return (rewards - rewards.mean()) / (rewards.std() + EPSILON)

        # Reshape to (num_groups, num_generations) for group statistics
        grouped_rewards = rewards.view(num_groups, num_generations)
        mean_grouped = grouped_rewards.mean(dim=1, keepdim=True)
        std_grouped = grouped_rewards.std(dim=1, keepdim=True)

        # Normalize within groups
        advantages = (grouped_rewards - mean_grouped) / (std_grouped + EPSILON)
        return advantages.view(-1)  # Flatten back

    elif scale_rewards == "batch":
        # Batch-level normalization
        return (rewards - rewards.mean()) / (rewards.std() + EPSILON)

    else:
        # No normalization
        return rewards


def compute_importance_ratio(
    per_token_logps: torch.Tensor,
    old_per_token_logps: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute importance sampling ratio (new policy / old policy).

    Args:
        per_token_logps: Log probabilities from current policy, shape (batch, seq_len)
        old_per_token_logps: Log probabilities from old policy, shape (batch, seq_len)
                            If None, assumes on-policy (ratio = 1)

    Returns:
        Tuple of (ratio, log_ratio) tensors
    """
    if old_per_token_logps is not None:
        log_ratio = per_token_logps - old_per_token_logps
        ratio = torch.exp(log_ratio)
    else:
        # No old logprobs - assume on-policy (ratio = 1)
        ratio = torch.ones_like(per_token_logps)
        log_ratio = torch.zeros_like(per_token_logps)

    return ratio, log_ratio


def compute_kl_penalty(
    log_ratio: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Compute KL divergence penalty term.

    Uses the approximation: KL ≈ exp(-log_ratio) - (-log_ratio) - 1
    This is equivalent to: KL ≈ p/q - log(p/q) - 1

    Args:
        log_ratio: Log ratio of new/old policy, shape (batch, seq_len)
        mask: Attention mask for valid tokens, shape (batch, seq_len)
        beta: KL penalty coefficient

    Returns:
        Per-token KL penalty tensor
    """
    if beta == 0.0:
        return torch.zeros_like(log_ratio)

    # KL divergence approximation
    per_token_kl = torch.exp(-log_ratio) - (-log_ratio) - 1
    return beta * per_token_kl


def aggregate_loss(
    per_token_loss: torch.Tensor,
    mask: torch.Tensor,
    aggregation: str = "mean_per_seq",
    batch_size: Optional[int] = None,
    max_completion_length: Optional[int] = None,
) -> torch.Tensor:
    """Aggregate per-token loss to scalar.

    Args:
        per_token_loss: Per-token loss tensor, shape (batch, seq_len)
        mask: Attention mask for valid tokens, shape (batch, seq_len)
        aggregation: Aggregation method:
            - "mean_per_seq": Average per sequence, then average across batch (GRPO, SAPO)
            - "mean_per_token": Average across all tokens (BNPO, CISPO, DAPO)
            - "dr_grpo": Normalize by (batch_size * max_completion_length)
        batch_size: Batch size (required for dr_grpo)
        max_completion_length: Max completion length (required for dr_grpo)

    Returns:
        Scalar loss tensor
    """
    if aggregation == "mean_per_seq":
        # Normalize by sequence length (per-sequence mean, then batch mean)
        seq_lengths = mask.sum(dim=-1).clamp(min=1.0)
        return ((per_token_loss * mask).sum(dim=-1) / seq_lengths).mean()

    elif aggregation == "mean_per_token":
        # Normalize by total token count in batch
        total_tokens = mask.sum().clamp(min=1.0)
        return (per_token_loss * mask).sum() / total_tokens

    elif aggregation == "dr_grpo":
        # Normalize by (batch_size * max_completion_length)
        if batch_size is None or max_completion_length is None:
            raise ValueError("batch_size and max_completion_length required for dr_grpo aggregation")
        return (per_token_loss * mask).sum() / (batch_size * max_completion_length)

    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


def clip_ratio(
    ratio: torch.Tensor,
    epsilon_low: float,
    epsilon_high: float,
) -> torch.Tensor:
    """Clip importance ratio to (1 - epsilon_low, 1 + epsilon_high).

    Args:
        ratio: Importance sampling ratio
        epsilon_low: Lower clipping bound
        epsilon_high: Upper clipping bound

    Returns:
        Clipped ratio
    """
    return torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
