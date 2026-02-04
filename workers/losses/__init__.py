"""Loss computation modules for GRPO training.

This module provides modular loss functions for different GRPO variants:
- GRPO: Standard group-relative policy optimization
- DAPO: Dynamic advantage policy optimization
- BNPO: Batch-normalized policy optimization
- DR-GRPO: Dynamic-ratio GRPO
- CISPO: Clipped importance sampling policy optimization
- SAPO: Soft adaptive policy optimization
"""

from MRL.workers.losses.base import (
    LossResult,
    compute_advantages,
    compute_importance_ratio,
    compute_kl_penalty,
)
from MRL.workers.losses.grpo import (
    compute_grpo_loss,
    compute_dapo_loss,
    compute_bnpo_loss,
    compute_dr_grpo_loss,
    compute_cispo_loss,
    compute_sapo_loss,
    get_loss_function,
)

__all__ = [
    # Base utilities
    "LossResult",
    "compute_advantages",
    "compute_importance_ratio",
    "compute_kl_penalty",
    # Loss functions
    "compute_grpo_loss",
    "compute_dapo_loss",
    "compute_bnpo_loss",
    "compute_dr_grpo_loss",
    "compute_cispo_loss",
    "compute_sapo_loss",
    "get_loss_function",
]
