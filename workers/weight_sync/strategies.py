"""Concrete weight synchronization strategy implementations."""

import time
from typing import Any, List, Optional, TYPE_CHECKING

from MRL.logging_config import get_logger
from MRL.workers.weight_sync.base import WeightSyncStrategy, WeightSyncResult

if TYPE_CHECKING:
    from MRL.workers.actor import ActorWorker
    from MRL.workers.rollout import RolloutWorker

logger = get_logger("weight_sync.strategies")


class ReloadStrategy(WeightSyncStrategy):
    """Efficient weight sync using vLLM's reload_weights pattern.

    This strategy uses vLLM v1's sleep/wake_up/reload_weights for efficient
    in-place weight updates without recreating the model. It's the recommended
    approach for most training scenarios.

    Flow:
    1. Actor saves weights to a local path on the volume
    2. Rollout workers call reload_weights() to load the new weights
    3. Workers use sleep/wake_up to minimize GPU memory during reload
    """

    name = "reload"
    requires_model_path = True
    supports_incremental = True

    def sync(
        self,
        actor: "ActorWorker",
        rollout_workers: List["RolloutWorker"],
        step: int,
        model_path: Optional[str] = None,
        **kwargs,
    ) -> WeightSyncResult:
        """Sync weights using vLLM reload pattern."""
        sync_start = time.time()
        workers_total = len(rollout_workers)

        if model_path is None:
            # Fall back to volume strategy
            logger.warning(
                "ReloadStrategy: model_path not set, falling back to VolumeStrategy"
            )
            fallback = self.get_fallback_strategy()
            if fallback:
                return fallback.sync(actor, rollout_workers, step, **kwargs)
            return WeightSyncResult(
                success=False,
                workers_synced=0,
                workers_total=workers_total,
                sync_time_seconds=time.time() - sync_start,
                method=self.name,
                error="model_path required but not provided",
            )

        try:
            # Get config for actor methods
            config = kwargs.get("config", self.config)

            # Save weights to the model path
            manifest = actor.sync_weights_to_model_path.remote(
                model_path=model_path,
                sync_id=step,
                config=config,
            )
            logger.info(f"Weights saved to {model_path} (sync_id: {manifest.get('sync_id', step)})")

            # Commit volume if available
            if self.volume is not None:
                self.volume.commit()

            # Reload weights in all rollout workers (parallel)
            sync_futures = []
            for worker in rollout_workers:
                future = worker.update_weights_from_volume.spawn(
                    weights_path=model_path,
                )
                sync_futures.append(future)

            # Wait for all workers
            sync_results = [f.get() for f in sync_futures]
            workers_synced = sum(sync_results)

            sync_time = time.time() - sync_start

            return WeightSyncResult(
                success=workers_synced == workers_total,
                workers_synced=workers_synced,
                workers_total=workers_total,
                sync_time_seconds=sync_time,
                method=self.name,
                details={"model_path": model_path},
            )

        except Exception as e:
            logger.error(f"ReloadStrategy sync failed: {e}")
            return WeightSyncResult(
                success=False,
                workers_synced=0,
                workers_total=workers_total,
                sync_time_seconds=time.time() - sync_start,
                method=self.name,
                error=str(e),
            )

    def initialize_workers(
        self,
        rollout_workers: List["RolloutWorker"],
        base_model: str,
        max_model_len: int,
        **kwargs,
    ) -> Optional[str]:
        """Initialize workers for reload-based sync."""
        logger.info("Initializing rollout workers for reload-based sync...")

        warmup_futures = []
        for worker in rollout_workers:
            future = worker.initialize_for_weight_sync.spawn(
                base_model=base_model,
                max_model_len=max_model_len,
            )
            warmup_futures.append(future)

        # Wait for initialization and get the model path
        model_path = None
        for future in warmup_futures:
            model_path = future.get()  # All workers return same path

        if model_path is None:
            logger.error("Failed to initialize workers for reload sync")
            return None

        logger.info(f"Rollout workers initialized at {model_path}")

        # Do a quick warmup generation
        warmup_gen_futures = []
        for worker in rollout_workers:
            future = worker.generate.spawn(
                prompts=["Hello"],
                model_path=model_path,
                max_tokens=10,
                max_model_len=max_model_len,
            )
            warmup_gen_futures.append(future)

        for future in warmup_gen_futures:
            future.get()

        return model_path

    def get_fallback_strategy(self) -> Optional[WeightSyncStrategy]:
        """Fall back to volume strategy if reload fails."""
        return VolumeStrategy(volume=self.volume, config=self.config)


class VolumeStrategy(WeightSyncStrategy):
    """Weight sync via shared volume.

    This strategy saves weights to a shared Modal volume, then rollout
    workers load from the volume. It's a good balance of reliability
    and performance.

    Flow:
    1. Actor saves weights to /storage/weight_sync/ with a manifest
    2. Volume is committed to persist changes
    3. Rollout workers load weights from the manifest location
    """

    name = "volume"
    requires_model_path = False
    supports_incremental = False

    def __init__(self, volume: Any = None, config: Optional[dict] = None):
        super().__init__(volume, config)
        self.sync_dir = "/storage/weight_sync"

    def sync(
        self,
        actor: "ActorWorker",
        rollout_workers: List["RolloutWorker"],
        step: int,
        model_path: Optional[str] = None,
        **kwargs,
    ) -> WeightSyncResult:
        """Sync weights via shared volume."""
        sync_start = time.time()
        workers_total = len(rollout_workers)

        try:
            config = kwargs.get("config", self.config)
            base_model = kwargs.get("base_model") or config.get("model_name", "")
            max_model_len = kwargs.get("max_model_len") or config.get("max_model_len", 16384)

            # Save weights to volume
            manifest = actor.sync_weights_to_volume.remote(
                sync_id=step,
                config=config,
            )
            logger.info(f"Weights saved to volume (sync_id: {manifest.get('sync_id', step)})")

            # Commit volume
            if self.volume is not None:
                self.volume.commit()

            # Reload rollout workers from volume
            sync_futures = []
            for worker in rollout_workers:
                future = worker.load_from_weight_sync.spawn(
                    base_model=base_model,
                    sync_dir=self.sync_dir,
                    max_model_len=max_model_len,
                )
                sync_futures.append(future)

            sync_results = [f.get() for f in sync_futures]
            workers_synced = sum(sync_results)

            sync_time = time.time() - sync_start

            return WeightSyncResult(
                success=workers_synced == workers_total,
                workers_synced=workers_synced,
                workers_total=workers_total,
                sync_time_seconds=sync_time,
                method=self.name,
                details={"sync_dir": self.sync_dir},
            )

        except Exception as e:
            logger.error(f"VolumeStrategy sync failed: {e}")
            return WeightSyncResult(
                success=False,
                workers_synced=0,
                workers_total=workers_total,
                sync_time_seconds=time.time() - sync_start,
                method=self.name,
                error=str(e),
            )

    def initialize_workers(
        self,
        rollout_workers: List["RolloutWorker"],
        base_model: str,
        max_model_len: int,
        **kwargs,
    ) -> Optional[str]:
        """Standard warmup with HuggingFace model path."""
        logger.info("Warming up rollout workers with standard initialization...")

        warmup_futures = []
        for worker in rollout_workers:
            future = worker.generate.spawn(
                prompts=["Hello"],
                model_path=base_model,
                max_tokens=10,
                max_model_len=max_model_len,
            )
            warmup_futures.append(future)

        for future in warmup_futures:
            future.get()

        logger.info("Rollout workers warmed up")
        return base_model  # Use HuggingFace path for generation


class DirectStrategy(WeightSyncStrategy):
    """In-memory weight transfer via vLLM's load_weights.

    WARNING: This strategy has known issues with tied weights (embed_tokens/lm_head)
    and is NOT recommended for production use. Use ReloadStrategy instead.

    Flow:
    1. Actor serializes weights to bytes
    2. Weights are transferred in-memory to rollout workers
    3. Workers use load_weights() to update in-place
    """

    name = "direct"
    requires_model_path = False
    supports_incremental = True

    def sync(
        self,
        actor: "ActorWorker",
        rollout_workers: List["RolloutWorker"],
        step: int,
        model_path: Optional[str] = None,
        **kwargs,
    ) -> WeightSyncResult:
        """Sync weights directly in memory."""
        sync_start = time.time()
        workers_total = len(rollout_workers)

        try:
            config = kwargs.get("config", self.config)

            # Get serialized weights from actor
            weights_bytes = actor.get_weights.remote(config=config)
            get_weights_time = time.time() - sync_start
            weights_size_mb = len(weights_bytes) / (1024 * 1024)
            logger.info(f"Got weights: {weights_size_mb:.1f} MB in {get_weights_time:.2f}s")

            # Update all rollout workers directly (parallel)
            sync_futures = []
            for worker in rollout_workers:
                future = worker.update_weights_direct.spawn(weights_bytes)
                sync_futures.append(future)

            # Wait for all workers
            sync_results = [f.get() for f in sync_futures]
            workers_synced = sum(sync_results)

            sync_time = time.time() - sync_start

            return WeightSyncResult(
                success=workers_synced == workers_total,
                workers_synced=workers_synced,
                workers_total=workers_total,
                sync_time_seconds=sync_time,
                method=self.name,
                details={"weights_size_mb": weights_size_mb},
            )

        except Exception as e:
            logger.error(f"DirectStrategy sync failed: {e}")
            return WeightSyncResult(
                success=False,
                workers_synced=0,
                workers_total=workers_total,
                sync_time_seconds=time.time() - sync_start,
                method=self.name,
                error=str(e),
            )

    def initialize_workers(
        self,
        rollout_workers: List["RolloutWorker"],
        base_model: str,
        max_model_len: int,
        **kwargs,
    ) -> Optional[str]:
        """Standard warmup."""
        logger.info("Warming up rollout workers (direct strategy)...")

        warmup_futures = []
        for worker in rollout_workers:
            future = worker.generate.spawn(
                prompts=["Hello"],
                model_path=base_model,
                max_tokens=10,
                max_model_len=max_model_len,
            )
            warmup_futures.append(future)

        for future in warmup_futures:
            future.get()

        return base_model

    def validate(self) -> List[str]:
        """Return warnings about known issues."""
        return [
            "DirectStrategy has known issues with tied weights (embed_tokens/lm_head). "
            "Consider using ReloadStrategy instead."
        ]

    def get_fallback_strategy(self) -> Optional[WeightSyncStrategy]:
        """Fall back to volume strategy."""
        return VolumeStrategy(volume=self.volume, config=self.config)


class CheckpointStrategy(WeightSyncStrategy):
    """Full checkpoint save and reload.

    This is the most reliable but slowest strategy. It saves a full
    checkpoint and rollout workers reload from the checkpoint path.

    Flow:
    1. Actor saves full checkpoint to /storage/checkpoints/
    2. Rollout workers reload the entire model from checkpoint
    """

    name = "checkpoint"
    requires_model_path = False
    supports_incremental = False

    def sync(
        self,
        actor: "ActorWorker",
        rollout_workers: List["RolloutWorker"],
        step: int,
        model_path: Optional[str] = None,
        **kwargs,
    ) -> WeightSyncResult:
        """Sync weights via full checkpoint."""
        sync_start = time.time()
        workers_total = len(rollout_workers)

        try:
            config = kwargs.get("config", self.config)

            # Save full checkpoint
            checkpoint_path = actor.save_checkpoint.remote(
                step=step,
                config=config,
            )
            logger.info(f"Checkpoint saved to: {checkpoint_path}")

            # Reload rollout workers from checkpoint
            sync_futures = []
            for worker in rollout_workers:
                future = worker.reload_from_checkpoint.spawn(checkpoint_path)
                sync_futures.append(future)

            sync_results = [f.get() for f in sync_futures]
            workers_synced = sum(sync_results)

            sync_time = time.time() - sync_start

            return WeightSyncResult(
                success=workers_synced == workers_total,
                workers_synced=workers_synced,
                workers_total=workers_total,
                sync_time_seconds=sync_time,
                method=self.name,
                details={"checkpoint_path": checkpoint_path},
            )

        except Exception as e:
            logger.error(f"CheckpointStrategy sync failed: {e}")
            return WeightSyncResult(
                success=False,
                workers_synced=0,
                workers_total=workers_total,
                sync_time_seconds=time.time() - sync_start,
                method=self.name,
                error=str(e),
            )

    def initialize_workers(
        self,
        rollout_workers: List["RolloutWorker"],
        base_model: str,
        max_model_len: int,
        **kwargs,
    ) -> Optional[str]:
        """Standard warmup."""
        logger.info("Warming up rollout workers (checkpoint strategy)...")

        warmup_futures = []
        for worker in rollout_workers:
            future = worker.generate.spawn(
                prompts=["Hello"],
                model_path=base_model,
                max_tokens=10,
                max_model_len=max_model_len,
            )
            warmup_futures.append(future)

        for future in warmup_futures:
            future.get()

        return base_model

    def validate(self) -> List[str]:
        """Return warnings about performance."""
        warnings = []
        sync_every = self.config.get("sync_weights_every", 1)
        if sync_every == 1:
            warnings.append(
                "CheckpointStrategy with sync_weights_every=1 is slow. "
                "Consider using ReloadStrategy or increasing sync_weights_every."
            )
        return warnings
