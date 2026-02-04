"""Tests for GRPO loss computation functions."""

import pytest
import torch

from MRL.workers.losses import (
    LossResult,
    compute_grpo_loss,
    compute_dapo_loss,
    compute_bnpo_loss,
    compute_dr_grpo_loss,
    compute_cispo_loss,
    compute_sapo_loss,
    get_loss_function,
)
from MRL.workers.losses.base import (
    compute_advantages,
    compute_importance_ratio,
    compute_kl_penalty,
    aggregate_loss,
    clip_ratio,
)


class TestComputeAdvantages:
    """Tests for advantage computation."""

    def test_group_normalization(self):
        """Test group-level advantage normalization."""
        # 4 samples, 2 groups of 2
        rewards = torch.tensor([1.0, 3.0, 2.0, 4.0])
        advantages = compute_advantages(rewards, scale_rewards="group", num_generations=2)

        # Group 1: [1, 3] -> mean=2, std(ddof=1)=sqrt(2)~1.414 -> normalized: [-0.707, 0.707]
        # Group 2: [2, 4] -> mean=3, std(ddof=1)=sqrt(2)~1.414 -> normalized: [-0.707, 0.707]
        # torch.std uses Bessel's correction by default
        expected = torch.tensor([-0.7071, 0.7071, -0.7071, 0.7071])
        assert torch.allclose(advantages, expected, atol=1e-3)

    def test_batch_normalization(self):
        """Test batch-level advantage normalization."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        advantages = compute_advantages(rewards, scale_rewards="batch", num_generations=2)

        # Batch: mean=2.5, std~1.118
        mean = rewards.mean()
        std = rewards.std()
        expected = (rewards - mean) / (std + 1e-8)
        assert torch.allclose(advantages, expected, atol=1e-5)

    def test_no_normalization(self):
        """Test no advantage normalization."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        advantages = compute_advantages(rewards, scale_rewards="none", num_generations=2)
        assert torch.equal(advantages, rewards)

    def test_single_generation_fallback(self):
        """Test fallback to batch normalization when num_generations=1."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        advantages = compute_advantages(rewards, scale_rewards="group", num_generations=1)
        # Should fall back to identity (rewards as-is)
        assert torch.equal(advantages, rewards)


class TestComputeImportanceRatio:
    """Tests for importance sampling ratio computation."""

    def test_with_old_logprobs(self):
        """Test ratio computation with old log probabilities."""
        new_logps = torch.tensor([[-1.0, -2.0], [-1.5, -2.5]])
        old_logps = torch.tensor([[-1.2, -2.1], [-1.4, -2.6]])

        ratio, log_ratio = compute_importance_ratio(new_logps, old_logps)

        expected_log_ratio = new_logps - old_logps
        expected_ratio = torch.exp(expected_log_ratio)

        assert torch.allclose(log_ratio, expected_log_ratio)
        assert torch.allclose(ratio, expected_ratio)

    def test_without_old_logprobs(self):
        """Test on-policy case (no old log probabilities)."""
        new_logps = torch.tensor([[-1.0, -2.0], [-1.5, -2.5]])

        ratio, log_ratio = compute_importance_ratio(new_logps, None)

        assert torch.allclose(ratio, torch.ones_like(new_logps))
        assert torch.allclose(log_ratio, torch.zeros_like(new_logps))


class TestClipRatio:
    """Tests for ratio clipping."""

    def test_symmetric_clipping(self):
        """Test symmetric clipping with equal bounds."""
        ratio = torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5])
        clipped = clip_ratio(ratio, epsilon_low=0.2, epsilon_high=0.2)

        expected = torch.tensor([0.8, 0.9, 1.0, 1.1, 1.2])
        assert torch.allclose(clipped, expected)

    def test_asymmetric_clipping(self):
        """Test asymmetric clipping (DAPO style)."""
        ratio = torch.tensor([0.5, 0.75, 1.0, 1.25, 1.5])
        clipped = clip_ratio(ratio, epsilon_low=0.2, epsilon_high=0.3)

        expected = torch.tensor([0.8, 0.8, 1.0, 1.25, 1.3])
        assert torch.allclose(clipped, expected)


class TestAggregateLoss:
    """Tests for loss aggregation methods."""

    def test_mean_per_seq(self):
        """Test per-sequence mean aggregation."""
        per_token_loss = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

        loss = aggregate_loss(per_token_loss, mask, aggregation="mean_per_seq")

        # Seq 1: (1+2+3)/3 = 2.0
        # Seq 2: (4+5)/2 = 4.5
        # Batch mean: (2+4.5)/2 = 3.25
        assert torch.isclose(loss, torch.tensor(3.25))

    def test_mean_per_token(self):
        """Test global token-level mean aggregation."""
        per_token_loss = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

        loss = aggregate_loss(per_token_loss, mask, aggregation="mean_per_token")

        # Total: (1+2+3+4+5) = 15, tokens = 5
        # Mean: 15/5 = 3.0
        assert torch.isclose(loss, torch.tensor(3.0))

    def test_dr_grpo(self):
        """Test DR-GRPO normalization."""
        per_token_loss = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        loss = aggregate_loss(
            per_token_loss,
            mask,
            aggregation="dr_grpo",
            batch_size=2,
            max_completion_length=4,
        )

        # Total: (1+2+3+4) = 10
        # Normalized: 10 / (2 * 4) = 1.25
        assert torch.isclose(loss, torch.tensor(1.25))


class TestKLPenalty:
    """Tests for KL divergence penalty computation."""

    def test_zero_beta(self):
        """Test that zero beta returns zero penalty."""
        log_ratio = torch.tensor([[0.1, -0.2], [0.3, -0.1]])
        mask = torch.ones_like(log_ratio)

        penalty = compute_kl_penalty(log_ratio, mask, beta=0.0)
        assert torch.allclose(penalty, torch.zeros_like(log_ratio))

    def test_nonzero_beta(self):
        """Test KL penalty with non-zero beta."""
        log_ratio = torch.tensor([[0.1, -0.2]])
        mask = torch.ones_like(log_ratio)

        penalty = compute_kl_penalty(log_ratio, mask, beta=0.1)

        # KL approx: exp(-log_ratio) - (-log_ratio) - 1
        expected_kl = torch.exp(-log_ratio) - (-log_ratio) - 1
        assert torch.allclose(penalty, 0.1 * expected_kl)


class TestGRPOLoss:
    """Tests for GRPO loss function."""

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for loss computation."""
        batch_size = 4
        seq_len = 8

        per_token_logps = torch.randn(batch_size, seq_len) - 2.0  # Negative log probs
        old_per_token_logps = per_token_logps + torch.randn_like(per_token_logps) * 0.1
        rewards = torch.tensor([0.0, 1.0, 0.5, 0.8])
        mask = torch.ones(batch_size, seq_len)

        return {
            "per_token_logps": per_token_logps,
            "old_per_token_logps": old_per_token_logps,
            "rewards": rewards,
            "mask": mask,
        }

    def test_grpo_loss_returns_loss_result(self, sample_inputs):
        """Test that GRPO loss returns LossResult."""
        result = compute_grpo_loss(**sample_inputs)

        assert isinstance(result, LossResult)
        assert isinstance(result.loss, torch.Tensor)
        assert result.loss.dim() == 0  # Scalar
        assert "mean_reward" in result.metrics
        assert "mean_advantage" in result.metrics
        assert "clip_fraction" in result.metrics

    def test_grpo_loss_gradient_flows(self, sample_inputs):
        """Test that gradients flow through GRPO loss."""
        sample_inputs["per_token_logps"].requires_grad = True
        result = compute_grpo_loss(**sample_inputs)

        result.loss.backward()
        assert sample_inputs["per_token_logps"].grad is not None

    def test_dapo_loss_asymmetric_clipping(self, sample_inputs):
        """Test DAPO loss with asymmetric clipping."""
        result = compute_dapo_loss(
            **sample_inputs,
            epsilon=0.2,
            epsilon_high=0.28,
        )

        assert isinstance(result, LossResult)
        assert result.loss.isfinite()

    def test_bnpo_loss_token_normalization(self, sample_inputs):
        """Test BNPO loss uses token-level normalization."""
        result = compute_bnpo_loss(**sample_inputs)

        assert isinstance(result, LossResult)
        assert result.loss.isfinite()

    def test_dr_grpo_loss_custom_normalization(self, sample_inputs):
        """Test DR-GRPO loss with custom normalization."""
        result = compute_dr_grpo_loss(
            **sample_inputs,
            max_completion_length=16,
        )

        assert isinstance(result, LossResult)
        assert result.loss.isfinite()

    def test_cispo_loss(self, sample_inputs):
        """Test CISPO loss."""
        result = compute_cispo_loss(**sample_inputs)

        assert isinstance(result, LossResult)
        assert result.loss.isfinite()

    def test_sapo_loss(self, sample_inputs):
        """Test SAPO loss with soft clipping."""
        result = compute_sapo_loss(**sample_inputs)

        assert isinstance(result, LossResult)
        assert result.loss.isfinite()


class TestGetLossFunction:
    """Tests for loss function factory."""

    @pytest.mark.parametrize(
        "loss_type,expected_fn",
        [
            ("grpo", compute_grpo_loss),
            ("dapo", compute_dapo_loss),
            ("bnpo", compute_bnpo_loss),
            ("dr_grpo", compute_dr_grpo_loss),
            ("cispo", compute_cispo_loss),
            ("sapo", compute_sapo_loss),
        ],
    )
    def test_get_loss_function(self, loss_type, expected_fn):
        """Test that factory returns correct function."""
        fn = get_loss_function(loss_type)
        assert fn == expected_fn

    def test_invalid_loss_type(self):
        """Test that invalid loss type raises error."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            get_loss_function("invalid")


class TestLossResultMetrics:
    """Tests for LossResult metrics computation."""

    def test_to_dict(self):
        """Test LossResult.to_dict()."""
        result = LossResult(
            loss=torch.tensor(0.5),
            metrics={"mean_reward": 0.8, "clip_fraction": 0.1},
        )

        d = result.to_dict()

        assert d["loss"] == 0.5
        assert d["mean_reward"] == 0.8
        assert d["clip_fraction"] == 0.1
