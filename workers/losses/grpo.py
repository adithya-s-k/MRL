"""GRPO loss function implementations.

This module implements various GRPO loss variants:
- GRPO: Standard group-relative policy optimization
- DAPO: Dynamic advantage policy optimization (asymmetric clipping)
- BNPO: Batch-normalized policy optimization
- DR-GRPO: Dynamic-ratio GRPO (normalized by max completion length)
- CISPO: Clipped importance sampling policy optimization
- SAPO: Soft adaptive policy optimization (sigmoid-based soft clipping)
"""

from typing import Callable, Optional

import torch

from MRL.logging_config import get_logger
from MRL.workers.losses.base import (
    LossResult,
    aggregate_loss,
    clip_ratio,
    compute_advantages,
    compute_importance_ratio,
    compute_kl_penalty,
)

logger = get_logger("losses.grpo")


def compute_grpo_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: Optional[torch.Tensor],
    rewards: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 0.2,
    epsilon_high: Optional[float] = None,
    beta: float = 0.0,
    num_generations: int = 4,
    scale_rewards: str = "group",
) -> LossResult:
    """Compute standard GRPO loss.

    GRPO uses:
    - Group-relative advantage normalization
    - Two-sided PPO-style clipping
    - Per-sequence loss aggregation

    Args:
        per_token_logps: Log probs from current policy, shape (batch, seq_len)
        old_per_token_logps: Log probs from old policy (optional)
        rewards: Rewards for each sequence, shape (batch,)
        mask: Attention mask, shape (batch, seq_len)
        epsilon: Lower clipping bound
        epsilon_high: Upper clipping bound (defaults to epsilon)
        beta: KL penalty coefficient
        num_generations: Number of generations per prompt
        scale_rewards: Advantage normalization type

    Returns:
        LossResult with loss and metrics
    """
    if epsilon_high is None:
        epsilon_high = epsilon

    # Compute advantages
    advantages = compute_advantages(rewards, scale_rewards, num_generations)
    advantages = advantages.unsqueeze(-1)  # (batch, 1) for broadcasting

    # Compute importance ratio
    ratio, log_ratio = compute_importance_ratio(per_token_logps, old_per_token_logps)

    # Two-sided clipping
    clipped_ratio = clip_ratio(ratio, epsilon, epsilon_high)

    # PPO-style loss: min of clipped and unclipped
    per_token_loss1 = ratio * advantages
    per_token_loss2 = clipped_ratio * advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    # Add KL penalty
    if beta != 0.0 and old_per_token_logps is not None:
        kl_penalty = compute_kl_penalty(log_ratio, mask, beta)
        per_token_loss = per_token_loss + kl_penalty

    # Aggregate: per-sequence mean, then batch mean
    loss = aggregate_loss(per_token_loss, mask, "mean_per_seq")

    # Compute metrics
    metrics = _compute_metrics(ratio, log_ratio, mask, epsilon, epsilon_high, rewards, advantages)

    return LossResult(loss=loss, metrics=metrics)


def compute_dapo_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: Optional[torch.Tensor],
    rewards: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 0.2,
    epsilon_high: float = 0.28,
    beta: float = 0.0,
    num_generations: int = 4,
    scale_rewards: str = "group",
) -> LossResult:
    """Compute DAPO loss (Dynamic Advantage Policy Optimization).

    DAPO uses:
    - Asymmetric clipping (different epsilon for positive/negative advantages)
    - Global token-level normalization
    - Default epsilon_high=0.28 from the paper

    Args:
        per_token_logps: Log probs from current policy
        old_per_token_logps: Log probs from old policy
        rewards: Rewards for each sequence
        mask: Attention mask
        epsilon: Lower clipping bound (default: 0.2)
        epsilon_high: Upper clipping bound (default: 0.28, DAPO paper)
        beta: KL penalty coefficient
        num_generations: Number of generations per prompt
        scale_rewards: Advantage normalization type

    Returns:
        LossResult with loss and metrics
    """
    # Compute advantages
    advantages = compute_advantages(rewards, scale_rewards, num_generations)
    advantages = advantages.unsqueeze(-1)

    # Compute importance ratio
    ratio, log_ratio = compute_importance_ratio(per_token_logps, old_per_token_logps)

    # Asymmetric clipping
    clipped_ratio = clip_ratio(ratio, epsilon, epsilon_high)

    # PPO-style loss
    per_token_loss1 = ratio * advantages
    per_token_loss2 = clipped_ratio * advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    # Add KL penalty
    if beta != 0.0 and old_per_token_logps is not None:
        kl_penalty = compute_kl_penalty(log_ratio, mask, beta)
        per_token_loss = per_token_loss + kl_penalty

    # Global token-level normalization (DAPO style)
    loss = aggregate_loss(per_token_loss, mask, "mean_per_token")

    metrics = _compute_metrics(ratio, log_ratio, mask, epsilon, epsilon_high, rewards, advantages)
    return LossResult(loss=loss, metrics=metrics)


def compute_bnpo_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: Optional[torch.Tensor],
    rewards: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 0.2,
    epsilon_high: Optional[float] = None,
    beta: float = 0.0,
    num_generations: int = 4,
    scale_rewards: str = "group",
) -> LossResult:
    """Compute BNPO loss (Batch-Normalized Policy Optimization).

    BNPO normalizes by total token count across the batch.

    Args:
        per_token_logps: Log probs from current policy
        old_per_token_logps: Log probs from old policy
        rewards: Rewards for each sequence
        mask: Attention mask
        epsilon: Lower clipping bound
        epsilon_high: Upper clipping bound
        beta: KL penalty coefficient
        num_generations: Number of generations per prompt
        scale_rewards: Advantage normalization type

    Returns:
        LossResult with loss and metrics
    """
    if epsilon_high is None:
        epsilon_high = epsilon

    # Compute advantages
    advantages = compute_advantages(rewards, scale_rewards, num_generations)
    advantages = advantages.unsqueeze(-1)

    # Compute importance ratio
    ratio, log_ratio = compute_importance_ratio(per_token_logps, old_per_token_logps)

    # Two-sided clipping
    clipped_ratio = clip_ratio(ratio, epsilon, epsilon_high)

    # PPO-style loss
    per_token_loss1 = ratio * advantages
    per_token_loss2 = clipped_ratio * advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    # Add KL penalty
    if beta != 0.0 and old_per_token_logps is not None:
        kl_penalty = compute_kl_penalty(log_ratio, mask, beta)
        per_token_loss = per_token_loss + kl_penalty

    # Normalize by total tokens (BNPO style)
    loss = aggregate_loss(per_token_loss, mask, "mean_per_token")

    metrics = _compute_metrics(ratio, log_ratio, mask, epsilon, epsilon_high, rewards, advantages)
    return LossResult(loss=loss, metrics=metrics)


def compute_dr_grpo_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: Optional[torch.Tensor],
    rewards: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 0.2,
    epsilon_high: Optional[float] = None,
    beta: float = 0.0,
    num_generations: int = 4,
    scale_rewards: str = "group",
    max_completion_length: int = 512,
) -> LossResult:
    """Compute DR-GRPO loss (Dynamic-Ratio GRPO).

    DR-GRPO normalizes by (batch_size * max_completion_length) instead of
    actual token count. This provides a more stable gradient scale.

    Args:
        per_token_logps: Log probs from current policy
        old_per_token_logps: Log probs from old policy
        rewards: Rewards for each sequence
        mask: Attention mask
        epsilon: Lower clipping bound
        epsilon_high: Upper clipping bound
        beta: KL penalty coefficient
        num_generations: Number of generations per prompt
        scale_rewards: Advantage normalization type
        max_completion_length: Maximum completion length for normalization

    Returns:
        LossResult with loss and metrics
    """
    if epsilon_high is None:
        epsilon_high = epsilon

    batch_size = per_token_logps.shape[0]

    # Compute advantages
    advantages = compute_advantages(rewards, scale_rewards, num_generations)
    advantages = advantages.unsqueeze(-1)

    # Compute importance ratio
    ratio, log_ratio = compute_importance_ratio(per_token_logps, old_per_token_logps)

    # Two-sided clipping
    clipped_ratio = clip_ratio(ratio, epsilon, epsilon_high)

    # PPO-style loss
    per_token_loss1 = ratio * advantages
    per_token_loss2 = clipped_ratio * advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    # Add KL penalty
    if beta != 0.0 and old_per_token_logps is not None:
        kl_penalty = compute_kl_penalty(log_ratio, mask, beta)
        per_token_loss = per_token_loss + kl_penalty

    # DR-GRPO normalization
    loss = aggregate_loss(
        per_token_loss,
        mask,
        "dr_grpo",
        batch_size=batch_size,
        max_completion_length=max_completion_length,
    )

    metrics = _compute_metrics(ratio, log_ratio, mask, epsilon, epsilon_high, rewards, advantages)
    return LossResult(loss=loss, metrics=metrics)


def compute_cispo_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: Optional[torch.Tensor],
    rewards: torch.Tensor,
    mask: torch.Tensor,
    epsilon_high: float = 0.2,
    beta: float = 0.0,
    num_generations: int = 4,
    scale_rewards: str = "group",
) -> LossResult:
    """Compute CISPO loss (Clipped Importance Sampling Policy Optimization).

    CISPO clips importance weights and multiplies with log probs directly,
    rather than using PPO-style min clipping.

    Args:
        per_token_logps: Log probs from current policy
        old_per_token_logps: Log probs from old policy
        rewards: Rewards for each sequence
        mask: Attention mask
        epsilon_high: Upper clipping bound for importance weights
        beta: KL penalty coefficient
        num_generations: Number of generations per prompt
        scale_rewards: Advantage normalization type

    Returns:
        LossResult with loss and metrics
    """
    # Compute advantages
    advantages = compute_advantages(rewards, scale_rewards, num_generations)
    advantages = advantages.unsqueeze(-1)

    # Compute importance ratio
    ratio, log_ratio = compute_importance_ratio(per_token_logps, old_per_token_logps)

    # CISPO: Clip importance weights, multiply with log probs
    clamped_ratios = torch.clamp(ratio, max=1.0 + epsilon_high).detach()
    per_token_loss = -clamped_ratios * advantages * per_token_logps

    # Add KL penalty
    if beta != 0.0 and old_per_token_logps is not None:
        kl_penalty = compute_kl_penalty(log_ratio, mask, beta)
        per_token_loss = per_token_loss + kl_penalty

    # Global normalization
    loss = aggregate_loss(per_token_loss, mask, "mean_per_token")

    metrics = _compute_metrics(ratio, log_ratio, mask, 0.0, epsilon_high, rewards, advantages)
    return LossResult(loss=loss, metrics=metrics)


def compute_sapo_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: Optional[torch.Tensor],
    rewards: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.0,
    num_generations: int = 4,
    scale_rewards: str = "group",
    temp_neg: float = 1.05,
    temp_pos: float = 1.0,
) -> LossResult:
    """Compute SAPO loss (Soft Adaptive Policy Optimization).

    SAPO uses soft sigmoid-based clipping instead of hard clipping,
    with different temperature for positive and negative advantages.

    Args:
        per_token_logps: Log probs from current policy
        old_per_token_logps: Log probs from old policy
        rewards: Rewards for each sequence
        mask: Attention mask
        beta: KL penalty coefficient
        num_generations: Number of generations per prompt
        scale_rewards: Advantage normalization type
        temp_neg: Temperature for negative advantages (default: 1.05)
        temp_pos: Temperature for positive advantages (default: 1.0)

    Returns:
        LossResult with loss and metrics
    """
    # Compute advantages
    advantages = compute_advantages(rewards, scale_rewards, num_generations)
    advantages = advantages.unsqueeze(-1)

    # Compute importance ratio
    ratio, log_ratio = compute_importance_ratio(per_token_logps, old_per_token_logps)

    # SAPO: Soft clipping with sigmoid
    def sapo_token_loss(r: torch.Tensor, temperature: float) -> torch.Tensor:
        sigmoid_input = temperature * (r - 1.0)
        return torch.sigmoid(sigmoid_input) * 4.0 / temperature

    per_token_loss = torch.empty_like(ratio)
    positive_mask = (advantages > 0).expand_as(ratio)

    per_token_loss[positive_mask] = sapo_token_loss(ratio[positive_mask], temp_pos)
    per_token_loss[~positive_mask] = sapo_token_loss(ratio[~positive_mask], temp_neg)
    per_token_loss = -per_token_loss * advantages

    # Add KL penalty
    if beta != 0.0 and old_per_token_logps is not None:
        kl_penalty = compute_kl_penalty(log_ratio, mask, beta)
        per_token_loss = per_token_loss + kl_penalty

    # Per-sequence mean aggregation
    loss = aggregate_loss(per_token_loss, mask, "mean_per_seq")

    metrics = _compute_metrics(ratio, log_ratio, mask, 0.0, 0.0, rewards, advantages)
    return LossResult(loss=loss, metrics=metrics)


def _compute_metrics(
    ratio: torch.Tensor,
    log_ratio: torch.Tensor,
    mask: torch.Tensor,
    epsilon_low: float,
    epsilon_high: float,
    rewards: torch.Tensor,
    advantages: torch.Tensor,
) -> dict:
    """Compute training metrics from loss computation.

    Args:
        ratio: Importance sampling ratio
        log_ratio: Log of importance ratio
        mask: Attention mask
        epsilon_low: Lower clipping bound
        epsilon_high: Upper clipping bound
        rewards: Original rewards
        advantages: Computed advantages

    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # Mean ratio for valid tokens
        valid_mask = mask.bool()
        mean_ratio = ratio[valid_mask].mean().item() if valid_mask.any() else 1.0

        # Clip fraction
        if epsilon_low > 0 or epsilon_high > 0:
            clipped = ((ratio < 1 - epsilon_low) | (ratio > 1 + epsilon_high)).float()
            clip_frac = (clipped * mask).sum() / mask.sum().clamp(min=1.0)
            clip_frac = clip_frac.item()
        else:
            clip_frac = 0.0

        # Approximate KL divergence
        approx_kl = (log_ratio * mask).sum() / mask.sum().clamp(min=1.0)
        approx_kl = approx_kl.item()

        return {
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "mean_ratio": mean_ratio,
            "clip_fraction": clip_frac,
            "approx_kl": approx_kl,
        }


def get_loss_function(loss_type: str) -> Callable:
    """Get the appropriate loss function for a given loss type.

    Args:
        loss_type: One of "grpo", "dapo", "bnpo", "dr_grpo", "cispo", "sapo"

    Returns:
        Loss function callable

    Raises:
        ValueError: If loss_type is not recognized
    """
    loss_functions = {
        "grpo": compute_grpo_loss,
        "dapo": compute_dapo_loss,
        "bnpo": compute_bnpo_loss,
        "dr_grpo": compute_dr_grpo_loss,
        "cispo": compute_cispo_loss,
        "sapo": compute_sapo_loss,
    }

    if loss_type not in loss_functions:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Must be one of: {list(loss_functions.keys())}"
        )

    return loss_functions[loss_type]
