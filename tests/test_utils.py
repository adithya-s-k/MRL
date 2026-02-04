"""Tests for utility functions."""

import io
import pickle

import pytest
import torch

from MRL.workers.utils import (
    clean_state_dict_for_vllm,
    create_weights_iterator,
    detect_tied_weights,
    serialize_state_dict,
    validate_rewards,
)


class TestDetectTiedWeights:
    """Tests for tied weight detection."""

    def test_detects_identical_tensors(self):
        """Test detection of tied weights with identical tensors."""
        # Create a tensor and tie it to two keys
        shared_tensor = torch.randn(10, 10)
        state_dict = {
            "embed_tokens.weight": shared_tensor,
            "lm_head.weight": shared_tensor,
            "other.weight": torch.randn(5, 5),
        }

        tied = detect_tied_weights(state_dict)

        # Should detect one of them as tied (the one that comes later alphabetically)
        assert len(tied) == 1
        assert "lm_head.weight" in tied

    def test_no_tied_weights(self):
        """Test with no tied weights."""
        state_dict = {
            "layer1.weight": torch.randn(10, 10),
            "layer2.weight": torch.randn(10, 10),
        }

        tied = detect_tied_weights(state_dict)

        assert len(tied) == 0

    def test_multiple_tied_groups(self):
        """Test with multiple groups of tied weights."""
        tensor1 = torch.randn(10, 10)
        tensor2 = torch.randn(5, 5)
        state_dict = {
            "a.weight": tensor1,
            "b.weight": tensor1,
            "c.weight": tensor1,  # 3 tied together
            "x.weight": tensor2,
            "y.weight": tensor2,  # 2 tied together
        }

        tied = detect_tied_weights(state_dict)

        # Should have 3 tied (b, c from first group, y from second)
        assert len(tied) == 3


class TestCleanStateDictForVllm:
    """Tests for state dict cleaning."""

    def test_removes_tied_weights(self):
        """Test that tied weights are removed."""
        shared_tensor = torch.randn(10, 10)
        state_dict = {
            "model.embed_tokens.weight": shared_tensor,
            "lm_head.weight": shared_tensor,
            "model.layers.0.weight": torch.randn(5, 5),
        }

        cleaned = clean_state_dict_for_vllm(
            state_dict,
            skip_tied_weights=True,
            skip_lora_weights=False,
        )

        # Should have removed one of the tied weights
        assert len(cleaned) == 2

    def test_removes_lora_weights(self):
        """Test that LoRA weights are removed."""
        state_dict = {
            "model.layers.0.weight": torch.randn(10, 10),
            "model.layers.0.lora_A.weight": torch.randn(10, 4),
            "model.layers.0.lora_B.weight": torch.randn(4, 10),
            "base_model.model.layers.0.weight": torch.randn(10, 10),
        }

        cleaned = clean_state_dict_for_vllm(
            state_dict,
            skip_tied_weights=False,
            skip_lora_weights=True,
        )

        assert "model.layers.0.lora_A.weight" not in cleaned
        assert "model.layers.0.lora_B.weight" not in cleaned
        assert "base_model.model.layers.0.weight" not in cleaned
        assert "model.layers.0.weight" in cleaned

    def test_strips_base_model_prefix(self):
        """Test that base_model prefix is stripped."""
        state_dict = {
            "base_model.model.embed_tokens.weight": torch.randn(10, 10),
        }

        cleaned = clean_state_dict_for_vllm(
            state_dict,
            skip_tied_weights=False,
            skip_lora_weights=False,
        )

        # base_model.model. prefix is stripped, leaving just embed_tokens.weight
        assert "embed_tokens.weight" in cleaned
        assert "base_model.model.embed_tokens.weight" not in cleaned


class TestSerializeStateDict:
    """Tests for state dict serialization."""

    def test_serialization_roundtrip(self):
        """Test that serialization is lossless."""
        state_dict = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
            "layer2.weight": torch.randn(5, 10),
        }

        serialized = serialize_state_dict(state_dict, skip_tied_weights=False)

        # Deserialize using torch.load (serialize_state_dict uses torch.save)
        deserialized = torch.load(io.BytesIO(serialized), weights_only=True)

        for key in state_dict:
            assert torch.equal(state_dict[key], deserialized[key])

    def test_serialization_skips_tied_weights(self):
        """Test that tied weights are skipped during serialization."""
        shared_tensor = torch.randn(10, 10)
        state_dict = {
            "embed_tokens.weight": shared_tensor,
            "lm_head.weight": shared_tensor,  # Tied
        }

        serialized = serialize_state_dict(state_dict, skip_tied_weights=True)
        deserialized = torch.load(io.BytesIO(serialized), weights_only=True)

        # Should only have one of them
        assert len(deserialized) == 1


class TestCreateWeightsIterator:
    """Tests for weights iterator."""

    def test_iterator_yields_all_weights(self):
        """Test that iterator yields all weights."""
        state_dict = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
        }

        # Use CPU device since CUDA may not be available
        items = list(create_weights_iterator(state_dict, device="cpu"))

        assert len(items) == 2
        names = [name for name, _ in items]
        assert "layer1.weight" in names
        assert "layer1.bias" in names

    def test_iterator_moves_to_device(self):
        """Test that iterator moves weights to specified device."""
        state_dict = {
            "layer1.weight": torch.randn(5, 5, device="cpu"),
        }

        items = list(create_weights_iterator(state_dict, device="cpu"))

        assert items[0][1].device.type == "cpu"


class TestValidateRewards:
    """Tests for reward validation.

    Note: validate_rewards returns cleaned rewards (not warnings).
    Warnings are logged via the logging module.
    """

    def test_valid_rewards(self):
        """Test validation of valid rewards passes through unchanged."""
        rewards = [0.0, 0.5, 1.0, 0.8]

        result = validate_rewards(rewards)

        # Valid rewards should be unchanged
        assert result == rewards

    def test_out_of_range_rewards_clipped(self):
        """Test that out-of-range rewards are clipped."""
        rewards = [-0.5, 0.5, 1.5]

        result = validate_rewards(rewards, min_value=0.0, max_value=1.0)

        # Should be clipped to [0, 1]
        assert result[0] == 0.0  # -0.5 clipped to 0
        assert result[1] == 0.5  # unchanged
        assert result[2] == 1.0  # 1.5 clipped to 1.0

    def test_nan_rewards_replaced(self):
        """Test that NaN rewards are replaced with 0."""
        rewards = [0.5, float("nan"), 0.8]

        result = validate_rewards(rewards)

        assert result[0] == 0.5
        assert result[1] == 0.0  # NaN replaced with 0
        assert result[2] == 0.8

    def test_inf_rewards_clipped(self):
        """Test that infinite rewards are clipped."""
        rewards = [0.5, float("inf"), 0.8]

        result = validate_rewards(rewards)

        assert result[0] == 0.5
        assert result[1] < float("inf")  # Inf clipped to finite value
        assert result[2] == 0.8

    def test_empty_rewards(self):
        """Test validation of empty rewards list returns empty."""
        rewards = []

        result = validate_rewards(rewards)

        assert result == []

    def test_outlier_detection_logs_warning(self):
        """Test that outlier detection doesn't change rewards."""
        rewards = [0.5, 0.5, 0.5, 0.5, 10.0]  # One outlier

        result = validate_rewards(rewards, warn_on_outliers=True)

        # Outliers should still be present (only logged, not changed)
        assert result == rewards
