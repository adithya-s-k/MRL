"""Tests for environment abstractions."""

import pytest

from MRL.workers.environment import (
    Rollout,
    RolloutBatch,
    RewardResult,
    CompositeRewardEnvironment,
    WeightedRewardEnvironment,
)
from MRL.workers.environment.base import RewardEnvironment


class MockRewardEnvironment(RewardEnvironment):
    """Mock environment for testing."""

    name = "mock"

    def __init__(self, reward_value: float = 0.5):
        self.reward_value = reward_value

    def compute_reward(self, rollout, context=None):
        return RewardResult(
            reward=self.reward_value,
            metadata={"mock": True},
        )


class TestRollout:
    """Tests for Rollout dataclass."""

    def test_create_rollout(self):
        """Test creating a rollout."""
        rollout = Rollout(
            prompt="What is 2+2?",
            completion="4",
            reward=1.0,
            logprobs=[-0.1, -0.2],
        )

        assert rollout.prompt == "What is 2+2?"
        assert rollout.completion == "4"
        assert rollout.reward == 1.0

    def test_with_reward(self):
        """Test creating rollout with updated reward."""
        original = Rollout(prompt="test", completion="response")
        updated = original.with_reward(0.8)

        assert original.reward is None
        assert updated.reward == 0.8
        assert updated.prompt == "test"

    def test_to_dict(self):
        """Test converting rollout to dict."""
        rollout = Rollout(
            prompt="test",
            completion="response",
            reward=0.5,
            metadata={"key": "value"},
        )

        d = rollout.to_dict()

        assert d["prompt"] == "test"
        assert d["completion"] == "response"
        assert d["reward"] == 0.5
        assert d["metadata"]["key"] == "value"

    def test_from_dict(self):
        """Test creating rollout from dict."""
        d = {
            "prompt": "test",
            "completion": "response",
            "reward": 0.7,
        }

        rollout = Rollout.from_dict(d)

        assert rollout.prompt == "test"
        assert rollout.reward == 0.7


class TestRolloutBatch:
    """Tests for RolloutBatch."""

    def test_create_batch(self):
        """Test creating a batch."""
        rollouts = [
            Rollout(prompt="p1", completion="c1", reward=0.5),
            Rollout(prompt="p2", completion="c2", reward=0.8),
        ]
        batch = RolloutBatch(rollouts=rollouts)

        assert len(batch) == 2

    def test_batch_properties(self):
        """Test batch property accessors."""
        rollouts = [
            Rollout(prompt="p1", completion="c1", reward=0.5),
            Rollout(prompt="p2", completion="c2", reward=0.8),
        ]
        batch = RolloutBatch(rollouts=rollouts)

        assert batch.prompts == ["p1", "p2"]
        assert batch.completions == ["c1", "c2"]
        assert batch.rewards == [0.5, 0.8]

    def test_mean_reward(self):
        """Test mean reward calculation."""
        rollouts = [
            Rollout(prompt="p1", completion="c1", reward=0.4),
            Rollout(prompt="p2", completion="c2", reward=0.6),
        ]
        batch = RolloutBatch(rollouts=rollouts)

        assert batch.mean_reward == 0.5

    def test_mean_reward_with_none(self):
        """Test mean reward with some None values."""
        rollouts = [
            Rollout(prompt="p1", completion="c1", reward=0.5),
            Rollout(prompt="p2", completion="c2", reward=None),
        ]
        batch = RolloutBatch(rollouts=rollouts)

        assert batch.mean_reward == 0.5  # Only counts non-None

    def test_with_rewards(self):
        """Test updating batch with rewards."""
        rollouts = [
            Rollout(prompt="p1", completion="c1"),
            Rollout(prompt="p2", completion="c2"),
        ]
        batch = RolloutBatch(rollouts=rollouts)

        updated = batch.with_rewards([0.7, 0.9])

        assert updated[0].reward == 0.7
        assert updated[1].reward == 0.9

    def test_from_lists(self):
        """Test creating batch from parallel lists."""
        batch = RolloutBatch.from_lists(
            prompts=["p1", "p2"],
            completions=["c1", "c2"],
            rewards=[0.5, 0.8],
        )

        assert len(batch) == 2
        assert batch[0].prompt == "p1"
        assert batch[1].reward == 0.8


class TestRewardResult:
    """Tests for RewardResult."""

    def test_create_result(self):
        """Test creating reward result."""
        result = RewardResult(
            reward=0.8,
            breakdown={"test1": 0.6, "test2": 0.9},
            metadata={"execution_time": 0.5},
        )

        assert result.reward == 0.8
        assert result.breakdown["test1"] == 0.6


class TestCompositeRewardEnvironment:
    """Tests for CompositeRewardEnvironment."""

    def test_sum_aggregation(self):
        """Test sum aggregation of rewards."""
        env1 = MockRewardEnvironment(reward_value=0.3)
        env2 = MockRewardEnvironment(reward_value=0.5)

        composite = CompositeRewardEnvironment(
            environments={"env1": env1, "env2": env2},
            aggregation="sum",
        )

        rollout = Rollout(prompt="test", completion="response")
        result = composite.compute_reward(rollout)

        assert result.reward == 0.8  # 0.3 + 0.5

    def test_mean_aggregation(self):
        """Test mean aggregation of rewards."""
        env1 = MockRewardEnvironment(reward_value=0.2)
        env2 = MockRewardEnvironment(reward_value=0.8)

        composite = CompositeRewardEnvironment(
            environments={"env1": env1, "env2": env2},
            aggregation="mean",
        )

        rollout = Rollout(prompt="test", completion="response")
        result = composite.compute_reward(rollout)

        assert result.reward == 0.5  # (0.2 + 0.8) / 2

    def test_min_aggregation(self):
        """Test min aggregation of rewards."""
        env1 = MockRewardEnvironment(reward_value=0.3)
        env2 = MockRewardEnvironment(reward_value=0.7)

        composite = CompositeRewardEnvironment(
            environments={"env1": env1, "env2": env2},
            aggregation="min",
        )

        rollout = Rollout(prompt="test", completion="response")
        result = composite.compute_reward(rollout)

        assert result.reward == 0.3

    def test_breakdown_included(self):
        """Test that breakdown is included in result."""
        env1 = MockRewardEnvironment(reward_value=0.5)
        composite = CompositeRewardEnvironment(
            environments={"env1": env1},
            aggregation="sum",
        )

        rollout = Rollout(prompt="test", completion="response")
        result = composite.compute_reward(rollout)

        assert "env1" in result.breakdown
        assert result.breakdown["env1"]["reward"] == 0.5


class TestWeightedRewardEnvironment:
    """Tests for WeightedRewardEnvironment."""

    def test_weighted_sum(self):
        """Test weighted sum of rewards."""
        env1 = MockRewardEnvironment(reward_value=1.0)
        env2 = MockRewardEnvironment(reward_value=0.5)

        weighted = WeightedRewardEnvironment(
            environments={"env1": env1, "env2": env2},
            weights={"env1": 0.3, "env2": 0.7},
        )

        rollout = Rollout(prompt="test", completion="response")
        result = weighted.compute_reward(rollout)

        # 1.0 * 0.3 + 0.5 * 0.7 = 0.3 + 0.35 = 0.65
        assert abs(result.reward - 0.65) < 1e-6

    def test_normalized_weights(self):
        """Test normalized weights."""
        env1 = MockRewardEnvironment(reward_value=1.0)
        env2 = MockRewardEnvironment(reward_value=0.0)

        weighted = WeightedRewardEnvironment(
            environments={"env1": env1, "env2": env2},
            weights={"env1": 2, "env2": 2},  # Will be normalized to 0.5, 0.5
            normalize=True,
        )

        rollout = Rollout(prompt="test", completion="response")
        result = weighted.compute_reward(rollout)

        # 1.0 * 0.5 + 0.0 * 0.5 = 0.5
        assert result.reward == 0.5

    def test_missing_weight_raises_error(self):
        """Test that missing weight raises error."""
        env1 = MockRewardEnvironment(reward_value=0.5)

        with pytest.raises(ValueError, match="Missing weight"):
            WeightedRewardEnvironment(
                environments={"env1": env1, "env2": MockRewardEnvironment()},
                weights={"env1": 0.5},  # Missing env2
            )

    def test_validation_warns_zero_weight(self):
        """Test that zero weight triggers warning."""
        env1 = MockRewardEnvironment(reward_value=0.5)

        weighted = WeightedRewardEnvironment(
            environments={"env1": env1},
            weights={"env1": 0.0},
        )

        warnings = weighted.validate()

        assert any("no effect" in w.lower() for w in warnings)
