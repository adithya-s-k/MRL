"""Configuration dataclasses for GRPO training with validation."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union

from MRL.logging_config import get_logger

logger = get_logger("config")


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def _validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
    """Validate a value is within a range."""
    if value < min_val or value > max_val:
        raise ConfigValidationError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )


def _validate_positive(value: Union[int, float], name: str, allow_zero: bool = False) -> None:
    """Validate a value is positive."""
    if allow_zero:
        if value < 0:
            raise ConfigValidationError(f"{name} must be >= 0, got {value}")
    else:
        if value <= 0:
            raise ConfigValidationError(f"{name} must be > 0, got {value}")


def _validate_choice(value: str, choices: list, name: str) -> None:
    """Validate a value is one of allowed choices."""
    if value not in choices:
        raise ConfigValidationError(
            f"{name} must be one of {choices}, got '{value}'"
        )


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    max_model_len: int = 16384
    trust_remote_code: bool = True

    def __post_init__(self):
        """Validate model configuration."""
        if not self.model_name:
            raise ConfigValidationError("model_name cannot be empty")
        _validate_positive(self.max_model_len, "max_model_len")
        if self.max_model_len > 131072:  # 128k context limit
            logger.warning(
                f"max_model_len={self.max_model_len} is very large, may cause OOM"
            )


@dataclass
class GenerationConfig:
    """Generation configuration for rollout workers."""

    max_tokens: int = 8000
    temperature: float = 0.7
    top_p: float = 0.9
    n: int = 1  # Number of completions per prompt

    def __post_init__(self):
        """Validate generation configuration."""
        _validate_positive(self.max_tokens, "max_tokens")
        _validate_range(self.temperature, 0.0, 2.0, "temperature")
        _validate_range(self.top_p, 0.0, 1.0, "top_p")
        _validate_positive(self.n, "n")


@dataclass
class TrainingConfig:
    """Training configuration for the actor worker."""

    # Basic training
    num_epochs: int = 5
    max_steps: int = -1  # -1 means use epochs
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-6

    # GRPO specific
    num_generations: int = 4  # Number of generations per prompt for GRPO

    # GRPO algorithm parameters (passed to TRL's GRPOConfig)
    # Loss type: grpo, dr_grpo, dapo, bnpo, cispo, sapo
    loss_type: str = "dapo"
    # KL coefficient (DeepSeek R1 uses 0.001)
    beta: float = 0.0
    # PPO-style clipping epsilon
    epsilon: float = 0.2
    # Upper clipping epsilon (DAPO recommends 0.28, None to disable)
    epsilon_high: Optional[float] = None
    # Reward scaling: "group", "batch", or "none"
    scale_rewards: str = "group"
    # Whether to mask truncated completions (DAPO paper)
    mask_truncated_completions: bool = False
    # Max completion length for training (truncates long completions to save memory)
    max_completion_length: int = 1024

    # LoRA parameters
    use_lora: bool = False
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha (scaling factor)
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list] = None  # None means auto-detect

    # Memory optimizations
    gradient_checkpointing: bool = True  # Reduces memory by recomputing activations
    max_grad_norm: Optional[float] = 1.0  # Gradient clipping (None to disable)

    # Legacy (kept for backwards compatibility, use beta instead)
    kl_coef: float = 0.1

    # Checkpointing
    save_steps: int = 100
    checkpoint_dir: str = "/storage/checkpoints"

    # Logging
    report_to: str = "wandb"
    logging_steps: int = 10

    # Weight sync
    sync_weights_every: int = 1  # Sync weights to rollout workers every N steps
    # Sync method:
    #   "reload" (recommended) - uses vLLM v1 sleep/wake_up/reload_weights for efficient updates
    #   "volume" - saves to shared volume, workers reload from volume (recreates model)
    #   "direct" - in-memory transfer (vLLM 0.15.0 doesn't support custom weights)
    #   "checkpoint" - full checkpoint save + model recreation (slowest)
    weight_sync_method: str = "reload"

    # Valid loss types
    VALID_LOSS_TYPES = ["grpo", "dr_grpo", "dapo", "bnpo", "cispo", "sapo"]
    # Valid scale_rewards options
    VALID_SCALE_REWARDS = ["group", "batch", "none"]
    # Valid weight sync methods
    VALID_WEIGHT_SYNC_METHODS = ["reload", "volume", "direct", "checkpoint"]

    def __post_init__(self):
        """Validate training configuration."""
        # Basic training validation
        _validate_positive(self.num_epochs, "num_epochs")
        if self.max_steps != -1:
            _validate_positive(self.max_steps, "max_steps")
        _validate_positive(self.batch_size, "batch_size")
        _validate_positive(self.gradient_accumulation_steps, "gradient_accumulation_steps")
        _validate_positive(self.learning_rate, "learning_rate")

        # GRPO validation
        _validate_positive(self.num_generations, "num_generations")
        _validate_choice(self.loss_type, self.VALID_LOSS_TYPES, "loss_type")
        _validate_range(self.beta, 0.0, 1.0, "beta")
        _validate_range(self.epsilon, 0.0, 1.0, "epsilon")
        if self.epsilon_high is not None:
            _validate_range(self.epsilon_high, 0.0, 1.0, "epsilon_high")
            if self.epsilon_high < self.epsilon:
                logger.warning(
                    f"epsilon_high ({self.epsilon_high}) < epsilon ({self.epsilon}), "
                    "this may cause unexpected clipping behavior"
                )
        _validate_choice(self.scale_rewards, self.VALID_SCALE_REWARDS, "scale_rewards")
        _validate_positive(self.max_completion_length, "max_completion_length")

        # LoRA validation
        if self.use_lora:
            _validate_positive(self.lora_r, "lora_r")
            _validate_positive(self.lora_alpha, "lora_alpha")
            _validate_range(self.lora_dropout, 0.0, 1.0, "lora_dropout")

        # Checkpointing validation
        _validate_positive(self.save_steps, "save_steps")
        _validate_positive(self.logging_steps, "logging_steps")
        _validate_positive(self.sync_weights_every, "sync_weights_every")
        _validate_choice(self.weight_sync_method, self.VALID_WEIGHT_SYNC_METHODS, "weight_sync_method")

        # Warn about deprecated/problematic options
        if self.weight_sync_method == "direct":
            logger.warning(
                "weight_sync_method='direct' has known issues with tied weights. "
                "Consider using 'reload' instead."
            )


@dataclass
class OrchestratorConfig:
    """Configuration for the training orchestrator."""

    # Model
    model: ModelConfig = field(default_factory=ModelConfig)

    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Generation
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # Workers
    num_rollout_workers: int = 2

    # Data
    dataset_name: str = "OpenCoder-LLM/opc-sft-stage2"
    dataset_config: str = "educational_instruct"
    dataset_split: str = "train"
    max_samples: Optional[int] = 128  # None for full dataset

    def __post_init__(self):
        """Validate orchestrator configuration."""
        _validate_positive(self.num_rollout_workers, "num_rollout_workers")
        if self.max_samples is not None:
            _validate_positive(self.max_samples, "max_samples")

        # Cross-field validation
        if self.training.num_generations > 1 and self.num_rollout_workers == 1:
            logger.info(
                "num_generations > 1 with single rollout worker may be slow. "
                "Consider adding more rollout workers."
            )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "model_name": self.model.model_name,
            "max_model_len": self.model.max_model_len,
            "trust_remote_code": self.model.trust_remote_code,
            "max_tokens": self.generation.max_tokens,
            "temperature": self.generation.temperature,
            "top_p": self.generation.top_p,
            "n": self.generation.n,
            "num_epochs": self.training.num_epochs,
            "max_steps": self.training.max_steps,
            "batch_size": self.training.batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "learning_rate": self.training.learning_rate,
            "num_generations": self.training.num_generations,
            "loss_type": self.training.loss_type,
            "beta": self.training.beta,
            "epsilon": self.training.epsilon,
            "epsilon_high": self.training.epsilon_high,
            "scale_rewards": self.training.scale_rewards,
            "mask_truncated_completions": self.training.mask_truncated_completions,
            "max_completion_length": self.training.max_completion_length,
            "use_lora": self.training.use_lora,
            "lora_r": self.training.lora_r,
            "lora_alpha": self.training.lora_alpha,
            "lora_dropout": self.training.lora_dropout,
            "lora_target_modules": self.training.lora_target_modules,
            "gradient_checkpointing": self.training.gradient_checkpointing,
            "max_grad_norm": self.training.max_grad_norm,
            "kl_coef": self.training.kl_coef,
            "save_steps": self.training.save_steps,
            "checkpoint_dir": self.training.checkpoint_dir,
            "report_to": self.training.report_to,
            "logging_steps": self.training.logging_steps,
            "sync_weights_every": self.training.sync_weights_every,
            "weight_sync_method": self.training.weight_sync_method,
            "num_rollout_workers": self.num_rollout_workers,
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "dataset_split": self.dataset_split,
            "max_samples": self.max_samples,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OrchestratorConfig":
        """Create config from dictionary."""
        model = ModelConfig(
            model_name=d.get("model_name", "Qwen/Qwen2-0.5B-Instruct"),
            max_model_len=d.get("max_model_len", 16384),
            trust_remote_code=d.get("trust_remote_code", True),
        )
        generation = GenerationConfig(
            max_tokens=d.get("max_tokens", 8000),
            temperature=d.get("temperature", 0.7),
            top_p=d.get("top_p", 0.9),
            n=d.get("n", 1),
        )
        training = TrainingConfig(
            num_epochs=d.get("num_epochs", 5),
            max_steps=d.get("max_steps", -1),
            batch_size=d.get("batch_size", 8),
            gradient_accumulation_steps=d.get("gradient_accumulation_steps", 1),
            learning_rate=d.get("learning_rate", 5e-6),
            num_generations=d.get("num_generations", 4),
            loss_type=d.get("loss_type", "dapo"),
            beta=d.get("beta", 0.0),
            epsilon=d.get("epsilon", 0.2),
            epsilon_high=d.get("epsilon_high"),
            scale_rewards=d.get("scale_rewards", "group"),
            mask_truncated_completions=d.get("mask_truncated_completions", False),
            max_completion_length=d.get("max_completion_length", 1024),
            use_lora=d.get("use_lora", False),
            lora_r=d.get("lora_r", 16),
            lora_alpha=d.get("lora_alpha", 32),
            lora_dropout=d.get("lora_dropout", 0.05),
            lora_target_modules=d.get("lora_target_modules"),
            gradient_checkpointing=d.get("gradient_checkpointing", True),
            max_grad_norm=d.get("max_grad_norm", 1.0),
            kl_coef=d.get("kl_coef", 0.1),
            save_steps=d.get("save_steps", 100),
            checkpoint_dir=d.get("checkpoint_dir", "/storage/checkpoints"),
            report_to=d.get("report_to", "wandb"),
            logging_steps=d.get("logging_steps", 10),
            sync_weights_every=d.get("sync_weights_every", 1),
            weight_sync_method=d.get("weight_sync_method", "reload"),
        )
        return cls(
            model=model,
            generation=generation,
            training=training,
            num_rollout_workers=d.get("num_rollout_workers", 2),
            dataset_name=d.get("dataset_name", "OpenCoder-LLM/opc-sft-stage2"),
            dataset_config=d.get("dataset_config", "educational_instruct"),
            dataset_split=d.get("dataset_split", "train"),
            max_samples=d.get("max_samples", 128),
        )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "OrchestratorConfig":
        """Load config from YAML file.

        Args:
            yaml_path: Path to YAML config file

        Returns:
            OrchestratorConfig instance
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config loading. Install with: pip install pyyaml")

        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "OrchestratorConfig":
        """Load config from JSON file.

        Args:
            json_path: Path to JSON config file

        Returns:
            OrchestratorConfig instance
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")

        with open(json_path) as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save config to YAML file.

        Args:
            yaml_path: Path to save YAML config file
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config saving. Install with: pip install pyyaml")

        yaml_path = Path(yaml_path)
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, json_path: Union[str, Path]) -> None:
        """Save config to JSON file.

        Args:
            json_path: Path to save JSON config file
        """
        json_path = Path(json_path)
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def validate_config(config: OrchestratorConfig) -> list[str]:
    """Validate configuration and return list of warnings.

    Args:
        config: Configuration to validate

    Returns:
        List of warning messages
    """
    warnings = []

    # Check for memory concerns
    if config.training.batch_size * config.training.num_generations > 64:
        warnings.append(
            f"Large effective batch size ({config.training.batch_size * config.training.num_generations}), "
            "may cause memory issues"
        )

    # Check for slow configuration
    if config.training.weight_sync_method == "checkpoint" and config.training.sync_weights_every == 1:
        warnings.append(
            "Checkpoint sync method with sync_weights_every=1 is slow. "
            "Consider using 'reload' method or increasing sync_weights_every"
        )

    # Check for suboptimal DAPO config
    if config.training.loss_type == "dapo":
        if config.training.epsilon_high is None:
            warnings.append(
                "DAPO typically uses epsilon_high=0.28 for asymmetric clipping. "
                "Consider setting epsilon_high."
            )
        if config.training.scale_rewards != "group":
            warnings.append(
                "DAPO paper recommends scale_rewards='group' for advantage normalization"
            )

    return warnings
