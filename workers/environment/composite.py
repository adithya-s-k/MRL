"""Composite reward environments for combining multiple reward signals."""

from typing import Any, Dict, List, Optional, Sequence

from MRL.logging_config import get_logger
from MRL.workers.environment.base import (
    Rollout,
    RewardEnvironment,
    RewardResult,
)

logger = get_logger("environment.composite")


class CompositeRewardEnvironment(RewardEnvironment):
    """Combine multiple reward environments.

    Computes rewards from multiple environments and returns a breakdown
    of all components. By default, returns the sum of all rewards.
    """

    name = "composite"

    def __init__(
        self,
        environments: Dict[str, RewardEnvironment],
        aggregation: str = "sum",
    ):
        """Initialize the composite environment.

        Args:
            environments: Dictionary mapping names to environments
            aggregation: How to combine rewards ("sum", "mean", "min", "max")
        """
        self.environments = environments
        self.aggregation = aggregation

        if aggregation not in ("sum", "mean", "min", "max"):
            raise ValueError(
                f"Unknown aggregation: {aggregation}. "
                "Must be one of: sum, mean, min, max"
            )

    def compute_reward(self, rollout: Rollout, context: Any = None) -> RewardResult:
        """Compute rewards from all environments and aggregate.

        Args:
            rollout: The rollout to score
            context: Context dict mapping environment names to their contexts,
                     or a single context to use for all environments

        Returns:
            RewardResult with aggregated reward and breakdown
        """
        # Handle context
        if isinstance(context, dict):
            contexts = context
        else:
            # Use same context for all environments
            contexts = {name: context for name in self.environments}

        # Compute rewards from each environment
        breakdown = {}
        rewards = []

        for name, env in self.environments.items():
            env_context = contexts.get(name)
            try:
                result = env.compute_reward(rollout, env_context)
                breakdown[name] = {
                    "reward": result.reward,
                    "breakdown": result.breakdown,
                    "metadata": result.metadata,
                }
                rewards.append(result.reward)
            except Exception as e:
                logger.warning(f"Environment {name} failed: {e}")
                breakdown[name] = {"error": str(e), "reward": 0.0}
                rewards.append(0.0)

        # Aggregate rewards
        if self.aggregation == "sum":
            final_reward = sum(rewards)
        elif self.aggregation == "mean":
            final_reward = sum(rewards) / len(rewards) if rewards else 0.0
        elif self.aggregation == "min":
            final_reward = min(rewards) if rewards else 0.0
        elif self.aggregation == "max":
            final_reward = max(rewards) if rewards else 0.0
        else:
            final_reward = sum(rewards)

        return RewardResult(
            reward=final_reward,
            breakdown=breakdown,
            metadata={"aggregation": self.aggregation},
        )

    def validate(self) -> List[str]:
        """Validate all sub-environments."""
        warnings = []
        for name, env in self.environments.items():
            env_warnings = env.validate()
            for w in env_warnings:
                warnings.append(f"{name}: {w}")
        return warnings


class WeightedRewardEnvironment(RewardEnvironment):
    """Combine multiple reward environments with weights.

    Each environment's reward is multiplied by its weight before summing.
    """

    name = "weighted"

    def __init__(
        self,
        environments: Dict[str, RewardEnvironment],
        weights: Dict[str, float],
        normalize: bool = False,
    ):
        """Initialize the weighted environment.

        Args:
            environments: Dictionary mapping names to environments
            weights: Dictionary mapping names to weight values
            normalize: If True, normalize weights to sum to 1
        """
        self.environments = environments
        self.normalize = normalize

        # Validate weights match environments
        for name in environments:
            if name not in weights:
                raise ValueError(f"Missing weight for environment: {name}")

        # Optionally normalize weights
        if normalize:
            total_weight = sum(weights.values())
            self.weights = {k: v / total_weight for k, v in weights.items()}
        else:
            self.weights = weights.copy()

    def compute_reward(self, rollout: Rollout, context: Any = None) -> RewardResult:
        """Compute weighted combination of rewards.

        Args:
            rollout: The rollout to score
            context: Context dict mapping environment names to their contexts,
                     or a single context to use for all environments

        Returns:
            RewardResult with weighted reward and breakdown
        """
        # Handle context
        if isinstance(context, dict):
            contexts = context
        else:
            contexts = {name: context for name in self.environments}

        # Compute weighted rewards
        breakdown = {}
        weighted_sum = 0.0

        for name, env in self.environments.items():
            env_context = contexts.get(name)
            weight = self.weights[name]

            try:
                result = env.compute_reward(rollout, env_context)
                weighted_reward = result.reward * weight

                breakdown[name] = {
                    "raw_reward": result.reward,
                    "weight": weight,
                    "weighted_reward": weighted_reward,
                    "breakdown": result.breakdown,
                }
                weighted_sum += weighted_reward

            except Exception as e:
                logger.warning(f"Environment {name} failed: {e}")
                breakdown[name] = {
                    "error": str(e),
                    "weight": weight,
                    "weighted_reward": 0.0,
                }

        return RewardResult(
            reward=weighted_sum,
            breakdown=breakdown,
            metadata={
                "weights": self.weights,
                "normalized": self.normalize,
            },
        )

    def validate(self) -> List[str]:
        """Validate configuration."""
        warnings = []

        # Check for zero weights
        for name, weight in self.weights.items():
            if weight == 0:
                warnings.append(f"Weight for {name} is 0 - this environment has no effect")

        # Check for negative weights
        for name, weight in self.weights.items():
            if weight < 0:
                warnings.append(f"Negative weight for {name} ({weight})")

        # Validate sub-environments
        for name, env in self.environments.items():
            env_warnings = env.validate()
            for w in env_warnings:
                warnings.append(f"{name}: {w}")

        return warnings


class ThresholdRewardEnvironment(RewardEnvironment):
    """Apply a threshold to a base environment's reward.

    Useful for converting continuous rewards to binary pass/fail.
    """

    name = "threshold"

    def __init__(
        self,
        base_environment: RewardEnvironment,
        threshold: float = 0.5,
        pass_reward: float = 1.0,
        fail_reward: float = 0.0,
    ):
        """Initialize the threshold environment.

        Args:
            base_environment: The environment to threshold
            threshold: Threshold value (reward >= threshold passes)
            pass_reward: Reward to return if threshold met
            fail_reward: Reward to return if threshold not met
        """
        self.base_environment = base_environment
        self.threshold = threshold
        self.pass_reward = pass_reward
        self.fail_reward = fail_reward

    def compute_reward(self, rollout: Rollout, context: Any = None) -> RewardResult:
        """Compute thresholded reward.

        Args:
            rollout: The rollout to score
            context: Context for the base environment

        Returns:
            RewardResult with thresholded reward
        """
        base_result = self.base_environment.compute_reward(rollout, context)

        if base_result.reward >= self.threshold:
            final_reward = self.pass_reward
            status = "passed"
        else:
            final_reward = self.fail_reward
            status = "failed"

        return RewardResult(
            reward=final_reward,
            breakdown={
                "base_reward": base_result.reward,
                "threshold": self.threshold,
                "status": status,
            },
            metadata=base_result.metadata,
        )


class ClippedRewardEnvironment(RewardEnvironment):
    """Clip rewards from a base environment to a range.

    Useful for preventing extreme rewards from dominating training.
    """

    name = "clipped"

    def __init__(
        self,
        base_environment: RewardEnvironment,
        min_reward: float = 0.0,
        max_reward: float = 1.0,
    ):
        """Initialize the clipped environment.

        Args:
            base_environment: The environment to clip
            min_reward: Minimum reward value
            max_reward: Maximum reward value
        """
        self.base_environment = base_environment
        self.min_reward = min_reward
        self.max_reward = max_reward

    def compute_reward(self, rollout: Rollout, context: Any = None) -> RewardResult:
        """Compute clipped reward.

        Args:
            rollout: The rollout to score
            context: Context for the base environment

        Returns:
            RewardResult with clipped reward
        """
        base_result = self.base_environment.compute_reward(rollout, context)
        clipped_reward = max(self.min_reward, min(self.max_reward, base_result.reward))

        return RewardResult(
            reward=clipped_reward,
            breakdown={
                "base_reward": base_result.reward,
                "clipped": base_result.reward != clipped_reward,
            },
            metadata=base_result.metadata,
        )
