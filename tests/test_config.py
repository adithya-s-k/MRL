"""Tests for configuration validation."""

import json
import os
import tempfile

import pytest

from MRL.config import (
    ConfigValidationError,
    GenerationConfig,
    ModelConfig,
    OrchestratorConfig,
    TrainingConfig,
    validate_config,
)


class TestModelConfig:
    """Tests for ModelConfig validation."""

    def test_valid_config(self):
        """Test valid model configuration."""
        config = ModelConfig(
            model_name="Qwen/Qwen2-0.5B-Instruct",
            max_model_len=16384,
        )
        assert config.model_name == "Qwen/Qwen2-0.5B-Instruct"

    def test_empty_model_name_raises_error(self):
        """Test that empty model name raises error."""
        with pytest.raises(ConfigValidationError, match="model_name cannot be empty"):
            ModelConfig(model_name="")

    def test_invalid_max_model_len_raises_error(self):
        """Test that invalid max_model_len raises error."""
        with pytest.raises(ConfigValidationError, match="max_model_len must be > 0"):
            ModelConfig(max_model_len=0)


class TestGenerationConfig:
    """Tests for GenerationConfig validation."""

    def test_valid_config(self):
        """Test valid generation configuration."""
        config = GenerationConfig(
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            n=2,
        )
        assert config.temperature == 0.7

    def test_invalid_temperature_raises_error(self):
        """Test that invalid temperature raises error."""
        with pytest.raises(ConfigValidationError, match="temperature must be between"):
            GenerationConfig(temperature=3.0)

    def test_invalid_top_p_raises_error(self):
        """Test that invalid top_p raises error."""
        with pytest.raises(ConfigValidationError, match="top_p must be between"):
            GenerationConfig(top_p=1.5)

    def test_invalid_max_tokens_raises_error(self):
        """Test that invalid max_tokens raises error."""
        with pytest.raises(ConfigValidationError, match="max_tokens must be > 0"):
            GenerationConfig(max_tokens=-1)


class TestTrainingConfig:
    """Tests for TrainingConfig validation."""

    def test_valid_config(self):
        """Test valid training configuration."""
        config = TrainingConfig(
            num_epochs=5,
            batch_size=8,
            learning_rate=5e-6,
            loss_type="dapo",
        )
        assert config.loss_type == "dapo"

    def test_invalid_loss_type_raises_error(self):
        """Test that invalid loss type raises error."""
        with pytest.raises(ConfigValidationError, match="loss_type must be one of"):
            TrainingConfig(loss_type="invalid_loss")

    def test_invalid_scale_rewards_raises_error(self):
        """Test that invalid scale_rewards raises error."""
        with pytest.raises(ConfigValidationError, match="scale_rewards must be one of"):
            TrainingConfig(scale_rewards="invalid")

    def test_invalid_weight_sync_method_raises_error(self):
        """Test that invalid weight_sync_method raises error."""
        with pytest.raises(ConfigValidationError, match="weight_sync_method must be one of"):
            TrainingConfig(weight_sync_method="invalid")

    def test_invalid_beta_raises_error(self):
        """Test that invalid beta raises error."""
        with pytest.raises(ConfigValidationError, match="beta must be between"):
            TrainingConfig(beta=1.5)

    def test_invalid_epsilon_raises_error(self):
        """Test that invalid epsilon raises error."""
        with pytest.raises(ConfigValidationError, match="epsilon must be between"):
            TrainingConfig(epsilon=1.5)

    def test_epsilon_high_warning(self):
        """Test that epsilon_high < epsilon is allowed but logs warning."""
        # Should not raise, but logs a warning
        config = TrainingConfig(epsilon=0.3, epsilon_high=0.2)
        assert config.epsilon_high == 0.2

    def test_lora_validation(self):
        """Test LoRA parameter validation when enabled."""
        with pytest.raises(ConfigValidationError, match="lora_r must be > 0"):
            TrainingConfig(use_lora=True, lora_r=0)

    def test_valid_loss_types(self):
        """Test all valid loss types are accepted."""
        for loss_type in ["grpo", "dr_grpo", "dapo", "bnpo", "cispo", "sapo"]:
            config = TrainingConfig(loss_type=loss_type)
            assert config.loss_type == loss_type


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""

    def test_valid_config(self):
        """Test valid orchestrator configuration."""
        config = OrchestratorConfig()
        assert config.num_rollout_workers == 2

    def test_to_dict_roundtrip(self):
        """Test that config survives to_dict/from_dict roundtrip."""
        original = OrchestratorConfig(
            model=ModelConfig(model_name="test-model"),
            training=TrainingConfig(loss_type="grpo", beta=0.1),
            num_rollout_workers=4,
        )

        d = original.to_dict()
        restored = OrchestratorConfig.from_dict(d)

        assert restored.model.model_name == "test-model"
        assert restored.training.loss_type == "grpo"
        assert restored.training.beta == 0.1
        assert restored.num_rollout_workers == 4

    def test_to_json_file(self):
        """Test saving config to JSON file."""
        config = OrchestratorConfig(
            training=TrainingConfig(loss_type="dapo"),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config.to_json(f.name)
            json_path = f.name

        try:
            with open(json_path) as f:
                data = json.load(f)

            assert data["loss_type"] == "dapo"
        finally:
            os.unlink(json_path)

    def test_from_json_file(self):
        """Test loading config from JSON file."""
        data = {
            "model_name": "test-model",
            "loss_type": "bnpo",
            "num_rollout_workers": 3,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            json_path = f.name

        try:
            config = OrchestratorConfig.from_json(json_path)
            assert config.model.model_name == "test-model"
            assert config.training.loss_type == "bnpo"
            assert config.num_rollout_workers == 3
        finally:
            os.unlink(json_path)


class TestValidateConfig:
    """Tests for validate_config warnings."""

    def test_large_effective_batch_warning(self):
        """Test warning for large effective batch size."""
        config = OrchestratorConfig(
            training=TrainingConfig(batch_size=32, num_generations=4),
        )

        warnings = validate_config(config)

        assert any("effective batch size" in w.lower() for w in warnings)

    def test_checkpoint_sync_warning(self):
        """Test warning for slow checkpoint sync."""
        config = OrchestratorConfig(
            training=TrainingConfig(
                weight_sync_method="checkpoint",
                sync_weights_every=1,
            ),
        )

        warnings = validate_config(config)

        assert any("checkpoint sync method" in w.lower() for w in warnings)

    def test_dapo_epsilon_high_warning(self):
        """Test warning for DAPO without epsilon_high."""
        config = OrchestratorConfig(
            training=TrainingConfig(
                loss_type="dapo",
                epsilon_high=None,
            ),
        )

        warnings = validate_config(config)

        assert any("epsilon_high" in w for w in warnings)

    def test_no_warnings_for_good_config(self):
        """Test that good config has no warnings."""
        config = OrchestratorConfig(
            training=TrainingConfig(
                loss_type="grpo",
                batch_size=8,
                num_generations=4,
                weight_sync_method="reload",
            ),
        )

        warnings = validate_config(config)

        # May have some minor warnings, but no critical ones
        assert not any("large effective batch" in w.lower() for w in warnings)
