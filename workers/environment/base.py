"""Base classes for environment abstractions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from MRL.logging_config import get_logger

logger = get_logger("environment.base")


@dataclass
class Rollout:
    """Represents a single rollout (generation) with metadata.

    This dataclass captures all information about a single model generation,
    making it easier to pass data between components and track provenance.

    Attributes:
        prompt: The input prompt
        completion: The model's generated completion
        reward: Computed reward (may be None before scoring)
        logprobs: Per-token log probabilities from generation
        prompt_id: Optional identifier for the source prompt
        generation_id: Optional identifier for this specific generation
        metadata: Additional metadata (e.g., temperature, model version)
    """

    prompt: str
    completion: str
    reward: Optional[float] = None
    logprobs: Optional[List[float]] = None
    prompt_id: Optional[str] = None
    generation_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def with_reward(self, reward: float) -> "Rollout":
        """Return a new Rollout with the reward set."""
        return Rollout(
            prompt=self.prompt,
            completion=self.completion,
            reward=reward,
            logprobs=self.logprobs,
            prompt_id=self.prompt_id,
            generation_id=self.generation_id,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "reward": self.reward,
            "logprobs": self.logprobs,
            "prompt_id": self.prompt_id,
            "generation_id": self.generation_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Rollout":
        """Create from dictionary."""
        return cls(
            prompt=d["prompt"],
            completion=d["completion"],
            reward=d.get("reward"),
            logprobs=d.get("logprobs"),
            prompt_id=d.get("prompt_id"),
            generation_id=d.get("generation_id"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class RolloutBatch:
    """A batch of rollouts, typically from a single training step.

    Provides utilities for batch operations and statistics.

    Attributes:
        rollouts: List of Rollout objects
        batch_id: Optional identifier for this batch
        metadata: Additional batch-level metadata
    """

    rollouts: List[Rollout]
    batch_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rollouts)

    def __iter__(self):
        return iter(self.rollouts)

    def __getitem__(self, idx):
        return self.rollouts[idx]

    @property
    def prompts(self) -> List[str]:
        """Get all prompts."""
        return [r.prompt for r in self.rollouts]

    @property
    def completions(self) -> List[str]:
        """Get all completions."""
        return [r.completion for r in self.rollouts]

    @property
    def rewards(self) -> List[Optional[float]]:
        """Get all rewards (may include None)."""
        return [r.reward for r in self.rollouts]

    @property
    def valid_rewards(self) -> List[float]:
        """Get only non-None rewards."""
        return [r.reward for r in self.rollouts if r.reward is not None]

    @property
    def mean_reward(self) -> Optional[float]:
        """Compute mean reward, or None if no valid rewards."""
        valid = self.valid_rewards
        if not valid:
            return None
        return sum(valid) / len(valid)

    def with_rewards(self, rewards: List[float]) -> "RolloutBatch":
        """Return a new batch with rewards assigned."""
        if len(rewards) != len(self.rollouts):
            raise ValueError(
                f"Expected {len(self.rollouts)} rewards, got {len(rewards)}"
            )
        return RolloutBatch(
            rollouts=[r.with_reward(reward) for r, reward in zip(self.rollouts, rewards)],
            batch_id=self.batch_id,
            metadata=self.metadata,
        )

    @classmethod
    def from_lists(
        cls,
        prompts: List[str],
        completions: List[str],
        rewards: Optional[List[float]] = None,
        logprobs: Optional[List[List[float]]] = None,
        batch_id: Optional[str] = None,
    ) -> "RolloutBatch":
        """Create a batch from parallel lists."""
        if len(prompts) != len(completions):
            raise ValueError(
                f"Prompts ({len(prompts)}) and completions ({len(completions)}) must match"
            )

        rollouts = []
        for i in range(len(prompts)):
            rollout = Rollout(
                prompt=prompts[i],
                completion=completions[i],
                reward=rewards[i] if rewards else None,
                logprobs=logprobs[i] if logprobs else None,
                generation_id=i,
            )
            rollouts.append(rollout)

        return cls(rollouts=rollouts, batch_id=batch_id)


@dataclass
class RewardResult:
    """Result of reward computation.

    Attributes:
        reward: The computed reward value
        breakdown: Optional breakdown of reward components (for composite rewards)
        metadata: Additional information about the computation
    """

    reward: float
    breakdown: Optional[dict] = None
    metadata: dict = field(default_factory=dict)


class RewardEnvironment(ABC):
    """Abstract base class for reward computation environments.

    Subclasses implement different reward computation strategies,
    such as code execution, LLM-as-judge, or composite rewards.
    """

    name: str = "base"

    @abstractmethod
    def compute_reward(self, rollout: Rollout, context: Any = None) -> RewardResult:
        """Compute reward for a single rollout.

        Args:
            rollout: The rollout to score
            context: Optional context (e.g., test cases, reference answer)

        Returns:
            RewardResult with computed reward
        """
        pass

    def compute_rewards_batch(
        self,
        rollouts: Sequence[Rollout],
        contexts: Optional[Sequence[Any]] = None,
    ) -> List[RewardResult]:
        """Compute rewards for a batch of rollouts.

        Default implementation calls compute_reward sequentially.
        Subclasses can override for parallel execution.

        Args:
            rollouts: Sequence of rollouts to score
            contexts: Optional sequence of contexts (one per rollout)

        Returns:
            List of RewardResult objects
        """
        if contexts is None:
            contexts = [None] * len(rollouts)

        if len(contexts) != len(rollouts):
            raise ValueError(
                f"Contexts ({len(contexts)}) must match rollouts ({len(rollouts)})"
            )

        results = []
        for rollout, context in zip(rollouts, contexts):
            result = self.compute_reward(rollout, context)
            results.append(result)

        return results

    def score_batch(
        self,
        batch: RolloutBatch,
        contexts: Optional[Sequence[Any]] = None,
    ) -> RolloutBatch:
        """Score a RolloutBatch and return a new batch with rewards.

        Args:
            batch: The batch to score
            contexts: Optional contexts for each rollout

        Returns:
            New RolloutBatch with rewards assigned
        """
        results = self.compute_rewards_batch(batch.rollouts, contexts)
        rewards = [r.reward for r in results]
        return batch.with_rewards(rewards)

    def validate(self) -> List[str]:
        """Validate the environment configuration.

        Returns:
            List of warning messages (empty if valid)
        """
        return []
