"""Worker modules for Modal GRPO training.

Workers:
- ActorWorker: Training worker using TRL's GRPOTrainer (runs on H100)
- RolloutWorker: vLLM inference worker for fast generation (runs on A10G)
- RewardWorker: Reward computation using Modal Sandboxes for code execution

Utilities:
- utils: Helper functions for state dict processing, tied weights, etc.
- weight_sync: Pluggable weight synchronization strategies
- losses: Modular loss function implementations

Note: Some imports require torch and are only available in GPU environments.
The reward functions are always available (for Sandbox execution).
"""

# Reward functions are always available (no torch dependency)
from MRL.workers.reward import (
    compute_reward,
    reward_helper_function,
    compute_reward_batch,
    compute_reward_with_partial_credit,
    partial_credit_reward_function,
)

__all__ = [
    # Reward functions (always available)
    "compute_reward",
    "reward_helper_function",
    "compute_reward_batch",
    "compute_reward_with_partial_credit",
    "partial_credit_reward_function",
]

# Torch-dependent imports - only available in GPU environments
try:
    from MRL.workers.actor import ActorWorker
    from MRL.workers.rollout import RolloutWorker
    from MRL.workers.utils import (
        clean_state_dict_for_vllm,
        detect_tied_weights,
        serialize_state_dict,
        create_weights_iterator,
        validate_rewards,
    )
    from MRL.workers.weight_sync import (
        WeightSyncStrategy,
        WeightSyncResult,
        get_weight_sync_strategy,
        ReloadStrategy,
        VolumeStrategy,
        DirectStrategy,
        CheckpointStrategy,
    )
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
    from MRL.workers.environment import (
        Rollout,
        RolloutBatch,
        RewardEnvironment,
        RewardResult,
        CodeExecutionEnvironment,
        PartialCreditEnvironment,
        CompositeRewardEnvironment,
        WeightedRewardEnvironment,
    )

    __all__.extend([
        # Workers
        "ActorWorker",
        "RolloutWorker",
        # Utils
        "clean_state_dict_for_vllm",
        "detect_tied_weights",
        "serialize_state_dict",
        "create_weights_iterator",
        "validate_rewards",
        # Weight sync
        "WeightSyncStrategy",
        "WeightSyncResult",
        "get_weight_sync_strategy",
        "ReloadStrategy",
        "VolumeStrategy",
        "DirectStrategy",
        "CheckpointStrategy",
        # Losses
        "LossResult",
        "compute_grpo_loss",
        "compute_dapo_loss",
        "compute_bnpo_loss",
        "compute_dr_grpo_loss",
        "compute_cispo_loss",
        "compute_sapo_loss",
        "get_loss_function",
        # Environment
        "Rollout",
        "RolloutBatch",
        "RewardEnvironment",
        "RewardResult",
        "CodeExecutionEnvironment",
        "PartialCreditEnvironment",
        "CompositeRewardEnvironment",
        "WeightedRewardEnvironment",
    ])

except ImportError:
    # torch not available - only reward functions are exported
    pass
