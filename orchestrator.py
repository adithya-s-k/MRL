"""Main training orchestrator - veRL style coordination of workers."""

import time
from typing import Optional

import modal

from MRL.app import app, volume, TRAINING_IMAGE
from MRL.config import OrchestratorConfig
from MRL.logging_config import get_logger
from MRL.workers.actor import ActorWorker
from MRL.workers.rollout import RolloutWorker
from MRL.workers.reward import reward_helper_function
from MRL.workers.weight_sync import get_weight_sync_strategy, WeightSyncResult

STORAGE_PATH = "/storage"

# Module logger
logger = get_logger("orchestrator")

# Constants for retry logic (Bug 7 fix)
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def get_batch(dataset, batch_idx: int, batch_size: int) -> dict:
    """Get a batch from the dataset.

    Args:
        dataset: HuggingFace dataset
        batch_idx: Batch index
        batch_size: Batch size

    Returns:
        Dictionary with prompts and testcases
    """
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(dataset))

    batch = dataset.select(range(start_idx, end_idx))
    return {
        "prompts": batch["prompt"],
        "testcases": batch["testcases"],
    }


def chunk_list(lst: list, num_chunks: int) -> list[list]:
    """Split a list into roughly equal chunks.

    Args:
        lst: List to split
        num_chunks: Number of chunks

    Returns:
        List of chunks
    """
    if num_chunks <= 0:
        return [lst]

    chunk_size = max(1, len(lst) // num_chunks)
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        if i == num_chunks - 1:
            # Last chunk gets any remainder
            chunks.append(lst[start:])
        else:
            chunks.append(lst[start : start + chunk_size])

    return [c for c in chunks if c]  # Remove empty chunks


def retry_with_backoff(func, max_retries: int = MAX_RETRIES, delay: float = RETRY_DELAY_SECONDS):
    """Execute a function with retry logic and exponential backoff (Bug 7 fix).

    Args:
        func: Callable to execute
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (doubles each retry)

    Returns:
        Result of the function

    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed. Last error: {e}")
    raise last_exception


@app.function(
    image=TRAINING_IMAGE,
    volumes={STORAGE_PATH: volume},
    timeout=3600 * 24,  # 24 hours
    secrets=[modal.Secret.from_name("adithya-hf-wandb")],
)
def train(config_dict: Optional[dict] = None, resume_from: Optional[str] = None):
    """Main training orchestrator - veRL style.

    Coordinates the training loop:
    1. Get batch of prompts from dataset
    2. Rollout workers generate completions (parallel)
    3. Reward workers score completions (parallel via Sandboxes)
    4. Actor computes GRPO loss and updates
    5. Sync weights to rollout workers periodically
    6. Checkpoint periodically

    Args:
        config_dict: Configuration dictionary (optional, uses defaults if None)
        resume_from: Optional path to checkpoint to resume from

    Returns:
        Training result summary
    """
    import json
    import os

    import wandb
    from datasets import load_dataset

    # Parse config
    if config_dict is None:
        config_dict = {}

    # Load config from checkpoint if resuming
    if resume_from is not None:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        checkpoint_state_path = os.path.join(resume_from, "training_state.json")
        if os.path.exists(checkpoint_state_path):
            with open(checkpoint_state_path) as f:
                checkpoint_state = json.load(f)
            # Merge checkpoint config with provided config (provided takes precedence)
            saved_config = checkpoint_state.get("config", {})
            merged_config = saved_config.copy()
            merged_config.update(config_dict)
            config_dict = merged_config
            resume_step = checkpoint_state.get("global_step", 0)
            logger.info(f"Loaded checkpoint state, will resume from step {resume_step}")
        else:
            logger.warning(f"No training_state.json found in {resume_from}, starting from step 0")
            resume_step = 0
    else:
        resume_step = 0

    config = OrchestratorConfig.from_dict(config_dict)

    logger.info(f"Starting training with config: {config.to_dict()}")

    # Initialize wandb
    wandb.init(
        project="modal-grpo-trl",
        config=config.to_dict(),
        name=f"grpo-{config.model.model_name.split('/')[-1]}",
    )

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=config.dataset_split,
    )
    dataset = dataset.rename_column("instruction", "prompt")
    dataset = dataset.rename_column("testcase", "testcases")

    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))

    logger.info(f"Dataset loaded with {len(dataset)} samples")

    # Initialize workers
    logger.info("Initializing workers...")
    actor = ActorWorker()

    # Initialize actor (with checkpoint resumption if applicable)
    if resume_from is not None:
        logger.info(f"Loading actor from checkpoint: {resume_from}")
        restored_state = actor.load_checkpoint.remote(resume_from, config.to_dict())
        logger.info(f"Actor restored: {restored_state}")
    else:
        actor.initialize.remote(config.to_dict())

    # Create rollout workers
    rollout_workers = [RolloutWorker() for _ in range(config.num_rollout_workers)]

    # Initialize weight sync strategy
    sync_method = config.training.weight_sync_method
    logger.info(f"Creating weight sync strategy: {sync_method}")

    weight_sync_strategy = get_weight_sync_strategy(
        method=sync_method,
        volume=volume,
        config=config.to_dict(),
    )

    # Validate strategy and log any warnings
    strategy_warnings = weight_sync_strategy.validate()
    for warning in strategy_warnings:
        logger.warning(f"Weight sync strategy warning: {warning}")

    # Initialize rollout workers using the strategy
    logger.info(f"Initializing rollout workers with {weight_sync_strategy.name} strategy...")
    rollout_model_path = weight_sync_strategy.initialize_workers(
        rollout_workers=rollout_workers,
        base_model=config.model.model_name,
        max_model_len=config.model.max_model_len,
    )

    # Handle initialization failure with fallback
    if weight_sync_strategy.requires_model_path and rollout_model_path is None:
        logger.warning(
            f"Strategy {weight_sync_strategy.name} requires model_path but initialization failed. "
            "Attempting fallback..."
        )
        fallback_strategy = weight_sync_strategy.get_fallback_strategy()
        if fallback_strategy:
            logger.info(f"Using fallback strategy: {fallback_strategy.name}")
            weight_sync_strategy = fallback_strategy
            rollout_model_path = weight_sync_strategy.initialize_workers(
                rollout_workers=rollout_workers,
                base_model=config.model.model_name,
                max_model_len=config.model.max_model_len,
            )
        else:
            # Last resort: use HuggingFace model name
            rollout_model_path = config.model.model_name

    # Set the current model path for generation
    current_rollout_model_path = rollout_model_path or config.model.model_name
    logger.info(f"Rollout workers initialized with model path: {current_rollout_model_path}")

    # Calculate training steps
    batch_size = config.training.batch_size
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    total_steps = num_batches * config.training.num_epochs

    if config.training.max_steps > 0:
        total_steps = min(total_steps, config.training.max_steps)

    logger.info(f"Total training steps: {total_steps}")

    # Training loop (with checkpoint resumption support)
    global_step = resume_step
    if resume_step > 0:
        logger.info(f"Resuming training from step {resume_step}")

    # Calculate starting epoch and batch for resumption
    start_epoch = resume_step // num_batches if num_batches > 0 else 0
    start_batch = resume_step % num_batches if num_batches > 0 else 0

    for epoch in range(start_epoch, config.training.num_epochs):
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        logger.info(f"\n=== Epoch {epoch + 1}/{config.training.num_epochs} ===")

        for batch_idx in range(epoch_start_batch, num_batches):
            if (
                config.training.max_steps > 0
                and global_step >= config.training.max_steps
            ):
                break

            # 1. Get batch
            batch = get_batch(dataset, batch_idx, batch_size)
            prompts = batch["prompts"]
            testcases = batch["testcases"]

            logger.info(
                f"\nStep {global_step + 1}/{total_steps}: Processing {len(prompts)} prompts"
            )

            # 2. Generate completions (parallel across rollout workers)
            # Expand prompts for multiple generations per prompt
            expanded_prompts = []
            expanded_testcases = []
            for p, t in zip(prompts, testcases):
                for _ in range(config.training.num_generations):
                    expanded_prompts.append(p)
                    expanded_testcases.append(t)

            # Chunk prompts across workers
            prompt_chunks = chunk_list(expanded_prompts, config.num_rollout_workers)

            generation_futures = []
            for i, worker in enumerate(rollout_workers):
                if i < len(prompt_chunks) and prompt_chunks[i]:
                    future = worker.generate.spawn(
                        prompts=prompt_chunks[i],
                        model_path=current_rollout_model_path,
                        max_tokens=config.generation.max_tokens,
                        temperature=config.generation.temperature,
                        top_p=config.generation.top_p,
                        n=1,  # Already expanded prompts
                        max_model_len=config.model.max_model_len,
                    )
                    generation_futures.append(future)

            # Collect generation results
            all_completions = []
            all_logprobs = []
            for future in generation_futures:
                result = future.get()
                all_completions.extend(result["completions"])
                all_logprobs.extend(result["logprobs"])

            logger.info(f"Generated {len(all_completions)} completions")

            # 3. Compute rewards (parallel via sandboxes)
            logger.info("Computing rewards...")
            rewards = list(reward_helper_function(all_completions, expanded_testcases))
            mean_reward = sum(rewards) / len(rewards) if rewards else 0

            logger.info(f"Mean reward: {mean_reward:.4f}")

            # 4. Train step
            logger.info("Performing training step...")
            loss_result = actor.train_step.remote(
                prompts=expanded_prompts,
                completions=all_completions,
                rewards=rewards,
                old_logprobs=all_logprobs,
                config=config.to_dict(),  # Pass config for auto-init if needed
            )

            # Log metrics
            metrics = {
                "train/loss": loss_result.get("loss", 0),
                "train/mean_reward": mean_reward,
                "train/mean_advantage": loss_result.get("mean_advantage", 0),
                "train/mean_ratio": loss_result.get("mean_ratio", 1.0),
                "train/clip_fraction": loss_result.get("clip_fraction", 0),
                "train/approx_kl": loss_result.get("approx_kl", 0),
                "train/epoch": epoch,
                "train/step": global_step,
            }
            wandb.log(metrics, step=global_step)

            # Print key metrics
            logger.info(
                f"  Loss: {loss_result.get('loss', 0):.4f}, "
                f"KL: {loss_result.get('approx_kl', 0):.4f}, "
                f"Clip: {loss_result.get('clip_fraction', 0):.2%}"
            )

            # 5. Sync weights to rollout workers (periodically)
            if (global_step + 1) % config.training.sync_weights_every == 0:
                logger.info(f"Syncing weights to rollout workers (strategy: {weight_sync_strategy.name})...")

                try:
                    # Use the weight sync strategy
                    sync_result: WeightSyncResult = weight_sync_strategy.sync(
                        actor=actor,
                        rollout_workers=rollout_workers,
                        step=global_step + 1,
                        model_path=rollout_model_path,
                        config=config.to_dict(),
                        base_model=config.model.model_name,
                        max_model_len=config.model.max_model_len,
                    )

                    # Log sync results
                    logger.info(
                        f"Weight sync completed: {sync_result.workers_synced}/{sync_result.workers_total} "
                        f"workers in {sync_result.sync_time_seconds:.2f}s"
                    )

                    # Update model path if checkpoint strategy returned a new path
                    if sync_result.details.get("checkpoint_path"):
                        current_rollout_model_path = sync_result.details["checkpoint_path"]

                    # Warn about partial failures
                    if not sync_result.success:
                        if sync_result.error:
                            logger.warning(f"Weight sync had errors: {sync_result.error}")
                        elif sync_result.workers_synced < sync_result.workers_total:
                            logger.warning(
                                f"{sync_result.workers_total - sync_result.workers_synced} "
                                "workers failed to sync"
                            )

                        # Try fallback if available and sync failed completely
                        if sync_result.workers_synced == 0:
                            fallback = weight_sync_strategy.get_fallback_strategy()
                            if fallback:
                                logger.info(f"Attempting fallback strategy: {fallback.name}")
                                fallback_result = fallback.sync(
                                    actor=actor,
                                    rollout_workers=rollout_workers,
                                    step=global_step + 1,
                                    config=config.to_dict(),
                                    base_model=config.model.model_name,
                                    max_model_len=config.model.max_model_len,
                                )
                                if fallback_result.success:
                                    logger.info(f"Fallback succeeded: {fallback_result.workers_synced} workers synced")

                except Exception as e:
                    logger.error(f"Weight sync failed: {e}")
                    import traceback
                    traceback.print_exc()
                    logger.warning("Continuing with current rollout weights...")

            # 6. Checkpoint periodically
            if (global_step + 1) % config.training.save_steps == 0:
                logger.info("Saving checkpoint...")
                checkpoint_path = actor.save_checkpoint.remote(global_step + 1, config=config.to_dict())
                logger.info(f"Checkpoint saved: {checkpoint_path}")

            global_step += 1

    # Final checkpoint
    logger.info("\nSaving final checkpoint...")
    final_checkpoint = actor.save_checkpoint.remote(global_step, config=config.to_dict())

    # Commit volume
    volume.commit()

    wandb.finish()

    return {
        "status": "completed",
        "total_steps": global_step,
        "final_checkpoint": final_checkpoint,
    }


@app.function(
    image=TRAINING_IMAGE,
    gpu="A100",
    volumes={STORAGE_PATH: volume},
    timeout=3600 * 24,
    secrets=[modal.Secret.from_name("adithya-hf-wandb")],
)
def train_simple(config_dict: Optional[dict] = None):
    """Simple training using TRL's built-in trainer loop.

    This is a simpler alternative that uses GRPOTrainer's internal
    training loop instead of manual orchestration. Useful for debugging
    or when the manual orchestration overhead is not needed.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Training result summary
    """
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer
    import torch

    from MRL.workers.reward import reward_helper_function

    # Parse config
    if config_dict is None:
        config_dict = {}
    config = OrchestratorConfig.from_dict(config_dict)

    logger.info(f"Starting simple training with config: {config.to_dict()}")

    # Load dataset
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=config.dataset_split,
    )
    dataset = dataset.rename_column("instruction", "prompt")
    dataset = dataset.rename_column("testcase", "testcases")

    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))

    logger.info(f"Dataset loaded with {len(dataset)} samples")

    # Build GRPOConfig kwargs
    grpo_kwargs = {
        "output_dir": f"{STORAGE_PATH}/checkpoints",
        "report_to": config.training.report_to,
        "use_vllm": False,
        "per_device_train_batch_size": config.training.batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "num_train_epochs": config.training.num_epochs,
        "max_steps": config.training.max_steps,
        "save_steps": config.training.save_steps,
        "logging_steps": config.training.logging_steps,
        "num_generations": config.training.num_generations,
        "bf16": torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": True,
        # GRPO algorithm parameters (TRL built-in)
        "loss_type": config.training.loss_type,
        "beta": config.training.beta,
        "epsilon": config.training.epsilon,
        "scale_rewards": config.training.scale_rewards,
        "mask_truncated_completions": config.training.mask_truncated_completions,
    }

    # Only add epsilon_high if specified
    if config.training.epsilon_high is not None:
        grpo_kwargs["epsilon_high"] = config.training.epsilon_high

    training_args = GRPOConfig(**grpo_kwargs)

    # Configure LoRA if enabled
    peft_config = None
    if config.training.use_lora:
        from peft import LoraConfig, TaskType

        target_modules = config.training.lora_target_modules
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        peft_config = LoraConfig(
            r=config.training.lora_r,
            lora_alpha=config.training.lora_alpha,
            lora_dropout=config.training.lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        logger.info(f"LoRA enabled: r={peft_config.r}, alpha={peft_config.lora_alpha}")

    # Create trainer
    trainer_kwargs = {
        "model": config.model.model_name,
        "reward_funcs": reward_helper_function,
        "args": training_args,
        "train_dataset": dataset,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = GRPOTrainer(**trainer_kwargs)

    # Train
    trainer.train()

    # Commit volume
    volume.commit()

    return {"status": "completed"}
