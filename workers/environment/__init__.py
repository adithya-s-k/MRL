"""Environment abstractions for reward computation.

This module provides:
- Rollout: Dataclass representing a single generation with metadata
- RewardEnvironment: Abstract base class for reward computation
- CodeExecutionEnvironment: Sandbox-based code execution rewards
- CompositeRewardEnvironment: Combine multiple reward signals
"""

from MRL.workers.environment.base import (
    Rollout,
    RolloutBatch,
    RewardEnvironment,
    RewardResult,
)
from MRL.workers.environment.code_execution import (
    CodeExecutionEnvironment,
    PartialCreditEnvironment,
)
from MRL.workers.environment.composite import (
    CompositeRewardEnvironment,
    WeightedRewardEnvironment,
)

__all__ = [
    # Base classes
    "Rollout",
    "RolloutBatch",
    "RewardEnvironment",
    "RewardResult",
    # Environments
    "CodeExecutionEnvironment",
    "PartialCreditEnvironment",
    "CompositeRewardEnvironment",
    "WeightedRewardEnvironment",
]
