"""Base classes for weight synchronization strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, TYPE_CHECKING

from MRL.logging_config import get_logger

if TYPE_CHECKING:
    from MRL.workers.actor import ActorWorker
    from MRL.workers.rollout import RolloutWorker

logger = get_logger("weight_sync.base")


@dataclass
class WeightSyncResult:
    """Result of a weight sync operation."""

    success: bool
    workers_synced: int
    workers_total: int
    sync_time_seconds: float
    method: str
    error: Optional[str] = None
    details: dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Return success rate as a fraction."""
        if self.workers_total == 0:
            return 0.0
        return self.workers_synced / self.workers_total

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "success": self.success,
            "workers_synced": self.workers_synced,
            "workers_total": self.workers_total,
            "sync_time_seconds": self.sync_time_seconds,
            "method": self.method,
            "error": self.error,
            "success_rate": self.success_rate,
            **self.details,
        }


class WeightSyncStrategy(ABC):
    """Abstract base class for weight synchronization strategies.

    Each strategy implements a different approach to syncing weights
    from the actor (training) worker to the rollout (inference) workers.

    Attributes:
        name: Human-readable name of the strategy
        requires_model_path: Whether this strategy needs a model path to be set
        supports_incremental: Whether this strategy supports incremental updates
    """

    name: str = "base"
    requires_model_path: bool = False
    supports_incremental: bool = False

    def __init__(self, volume: Any = None, config: Optional[dict] = None):
        """Initialize the strategy.

        Args:
            volume: Modal volume for storage (if needed)
            config: Configuration dictionary
        """
        self.volume = volume
        self.config = config or {}

    @abstractmethod
    def sync(
        self,
        actor: "ActorWorker",
        rollout_workers: List["RolloutWorker"],
        step: int,
        model_path: Optional[str] = None,
        **kwargs,
    ) -> WeightSyncResult:
        """Synchronize weights from actor to rollout workers.

        Args:
            actor: The actor worker (has the updated weights)
            rollout_workers: List of rollout workers to sync to
            step: Current training step (for versioning)
            model_path: Path to model (for reload-based strategies)
            **kwargs: Additional strategy-specific arguments

        Returns:
            WeightSyncResult with sync status and metrics
        """
        pass

    @abstractmethod
    def initialize_workers(
        self,
        rollout_workers: List["RolloutWorker"],
        base_model: str,
        max_model_len: int,
        **kwargs,
    ) -> Optional[str]:
        """Initialize rollout workers for this sync strategy.

        Some strategies require special initialization (e.g., reload needs
        a local model path).

        Args:
            rollout_workers: List of rollout workers to initialize
            base_model: Base model name (HuggingFace path)
            max_model_len: Maximum model context length
            **kwargs: Additional strategy-specific arguments

        Returns:
            Model path to use for generation (if applicable), or None
        """
        pass

    def validate(self) -> List[str]:
        """Validate the strategy configuration.

        Returns:
            List of warning messages (empty if valid)
        """
        return []

    def get_fallback_strategy(self) -> Optional["WeightSyncStrategy"]:
        """Return a fallback strategy if this one fails.

        Returns:
            Alternative strategy to try, or None
        """
        return None


def get_weight_sync_strategy(
    method: str,
    volume: Any = None,
    config: Optional[dict] = None,
) -> WeightSyncStrategy:
    """Factory function to get the appropriate weight sync strategy.

    Args:
        method: Sync method name ("reload", "volume", "direct", "checkpoint")
        volume: Modal volume for storage
        config: Configuration dictionary

    Returns:
        WeightSyncStrategy instance

    Raises:
        ValueError: If method is not recognized
    """
    # Import here to avoid circular imports
    from MRL.workers.weight_sync.strategies import (
        ReloadStrategy,
        VolumeStrategy,
        DirectStrategy,
        CheckpointStrategy,
    )

    strategies = {
        "reload": ReloadStrategy,
        "volume": VolumeStrategy,
        "direct": DirectStrategy,
        "checkpoint": CheckpointStrategy,
    }

    if method not in strategies:
        valid_methods = list(strategies.keys())
        raise ValueError(
            f"Unknown weight sync method: '{method}'. "
            f"Must be one of: {valid_methods}"
        )

    strategy_class = strategies[method]
    return strategy_class(volume=volume, config=config)
