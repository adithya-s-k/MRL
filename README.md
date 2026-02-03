# Modal GRPO - Serverless GRPO Training with TRL and vLLM

A veRL-style serverless architecture for GRPO (Group Relative Policy Optimization) training on Modal, using TRL's GRPOTrainer and standalone vLLM workers.

## Overview

This package implements distributed GRPO training with a clean separation of concerns:
- **Actor Worker**: Handles policy training using TRL's GRPOTrainer
- **Rollout Workers**: Handle fast inference using standalone vLLM (horizontally scalable)
- **Reward Workers**: Execute generated code in secure Modal Sandboxes

The architecture is inspired by [veRL](https://github.com/volcengine/verl) and designed for serverless execution on [Modal](https://modal.com).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MODAL CLOUD                                   │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    Orchestrator                                    │ │
│  │                    (@app.function, CPU)                           │ │
│  │  • Loads data, coordinates workers, manages training loop         │ │
│  └───────────────────────┬───────────────────────────────────────────┘ │
│                          │                                              │
│          ┌───────────────┼───────────────┬───────────────┐             │
│          ▼               ▼               ▼               ▼             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────┐      │
│  │ ActorWorker  │ │RolloutWorker │ │RolloutWorker │ │ Reward   │      │
│  │ (H100, TRL)  │ │ (A10G, vLLM) │ │ (A10G, vLLM) │ │ Workers  │      │
│  │ GRPOTrainer  │ │ Generation   │ │ Generation   │ │(Sandbox) │      │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────┘      │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Single Modal Volume                          │   │
│  │  /storage/checkpoints  /storage/data  /storage/hf_cache         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Training Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. GET BATCH                                                        │
│     Orchestrator fetches batch of prompts from dataset               │
│                          │                                           │
│                          ▼                                           │
│  2. GENERATE COMPLETIONS (Parallel)                                  │
│     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                 │
│     │ RolloutWorker│ │RolloutWorker│ │RolloutWorker│                │
│     │   Chunk 1   │ │   Chunk 2   │ │   Chunk N   │                 │
│     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘                 │
│            └───────────────┼───────────────┘                         │
│                            ▼                                         │
│  3. COMPUTE REWARDS (Parallel via Sandboxes)                         │
│     ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                 │
│     │Sandbox 1│ │Sandbox 2│ │Sandbox 3│ │Sandbox N│                 │
│     │ Code 1  │ │ Code 2  │ │ Code 3  │ │ Code N  │                 │
│     └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘                 │
│          └───────────┼───────────┼───────────┘                       │
│                      ▼                                               │
│  4. TRAIN STEP                                                       │
│     ActorWorker computes GRPO loss and updates policy                │
│                      │                                               │
│                      ▼                                               │
│  5. SYNC WEIGHTS (Periodic)                                          │
│     Actor → RolloutWorkers (every N steps)                           │
│                      │                                               │
│                      ▼                                               │
│  6. CHECKPOINT (Periodic)                                            │
│     Save to /storage/checkpoints/step-N                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. ActorWorker (`workers/actor.py`)

The training worker using TRL's GRPOTrainer.

| Property | Value |
|----------|-------|
| GPU | H100 |
| Image | NVIDIA CUDA 12.8.0 + TRL |
| Role | Policy optimization |

**Key Methods:**
- `initialize(config)` - Set up GRPOTrainer with configuration
- `train_step(prompts, completions, rewards, logprobs)` - Single training step
- `train_full()` - Run TRL's built-in training loop
- `get_weights()` - Serialize model weights for sync
- `save_checkpoint(step)` - Save checkpoint to volume

### 2. RolloutWorker (`workers/rollout.py`)

Standalone vLLM inference worker for fast generation.

| Property | Value |
|----------|-------|
| GPU | A10G |
| Image | NVIDIA CUDA 12.8.1 + vLLM |
| Concurrency | 32 concurrent inputs |
| Idle Timeout | 300s (stays warm) |

**Key Methods:**
- `generate(prompts, model_path, ...)` - Generate completions with logprobs
- `reload_from_checkpoint(path)` - Load updated weights from volume
- `update_weights(bytes)` - Update weights from serialized state dict
- `compute_logprobs(prompts, completions)` - Compute logprobs for given pairs

### 3. Reward Workers (`workers/reward.py`)

Code execution in secure Modal Sandboxes.

| Property | Value |
|----------|-------|
| Execution | Modal Sandbox |
| Timeout | 30s per execution |
| Parallelism | Via `starmap()` |

**Key Functions:**
- `compute_reward(completion, testcase)` - Binary reward (0 or 1)
- `compute_reward_with_partial_credit(completion, testcase)` - Partial credit (0 to 1)
- `reward_helper_function(completions, testcases)` - TRL-compatible batch function

### 4. Orchestrator (`orchestrator.py`)

Coordinates all workers in the training loop.

**Functions:**
- `train(config)` - veRL-style distributed training with manual orchestration
- `train_simple(config)` - TRL's built-in training loop (simpler, less control)

## Configuration

### OrchestratorConfig

```python
from modal_grpo.config import OrchestratorConfig

config = OrchestratorConfig(
    model=ModelConfig(
        model_name="Qwen/Qwen2-0.5B-Instruct",
        max_model_len=4096,
        trust_remote_code=True,
    ),
    training=TrainingConfig(
        num_epochs=5,
        max_steps=-1,  # -1 for unlimited
        batch_size=8,
        learning_rate=5e-6,
        num_generations=4,  # Generations per prompt for GRPO
        save_steps=100,
        sync_weights_every=1,
    ),
    generation=GenerationConfig(
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
    ),
    num_rollout_workers=2,
    dataset_name="OpenCoder-LLM/opc-sft-stage2",
    dataset_config="educational_instruct",
    max_samples=128,  # None for full dataset
)
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen2-0.5B-Instruct` | Model name or path |
| `--epochs` | 5 | Number of training epochs |
| `--max-steps` | 5 | Maximum steps (-1 for unlimited) |
| `--batch-size` | 8 | Batch size |
| `--num-rollout-workers` | 2 | Number of vLLM workers |
| `--num-generations` | 4 | Generations per prompt |
| `--max-samples` | 128 | Dataset samples (0 for full) |
| `--learning-rate` | 5e-6 | Learning rate |
| `--save-steps` | 100 | Checkpoint frequency |
| `--sync-weights-every` | 1 | Weight sync frequency |
| `--simple-mode` | False | Use TRL's built-in loop |

## Usage

### Prerequisites

1. Install Modal CLI:
   ```bash
   pip install modal
   modal setup
   ```

2. Create HuggingFace/WandB secret:
   ```bash
   modal secret create adithya-hf-wandb \
     HF_TOKEN=<your-token> \
     WANDB_API_KEY=<your-key>
   ```

3. Create the storage volume:
   ```bash
   modal volume create grpo-trl-storage
   ```

### Running Training

```bash
# Default training (orchestrator mode)
modal run modal_grpo/train.py

# With custom parameters
modal run modal_grpo/train.py \
  --model "Qwen/Qwen2-0.5B-Instruct" \
  --epochs 5 \
  --max-steps 100 \
  --batch-size 8 \
  --num-rollout-workers 4

# Simple mode (TRL built-in trainer)
modal run modal_grpo/train.py --simple-mode

# Run in background (detached)
modal run --detach modal_grpo/train.py
```

### Testing Components

```bash
# Test rollout worker
modal run modal_grpo/train.py::test_rollout_fn

# Test reward computation
modal run modal_grpo/train.py::test_reward_fn

# List checkpoints
modal run modal_grpo/train.py::list_checkpoints_fn

# Check volume contents
modal volume ls grpo-trl-storage
```

### Monitoring

- **Modal Dashboard**: [modal.com/apps](https://modal.com/apps) - View logs, costs, GPU usage
- **Weights & Biases**: Training metrics logged to wandb project `modal-grpo-trl`

## Images

### Training Image (TRAINING_IMAGE)

Base: `nvidia/cuda:12.8.0-devel-ubuntu24.04`

Packages:
- TRL (local installation from `/opt/trl`)
- vLLM (via TRL extras)
- wandb, datasets, accelerate, peft, bitsandbytes
- hf_transfer for fast downloads

### vLLM Image (VLLM_IMAGE)

Base: `nvidia/cuda:12.8.1-devel-ubuntu24.04`

Packages:
- vLLM
- flash-attn
- TRL (for compatibility)
- hf_transfer

Environment:
- `VLLM_USE_V1=0` (use v0 engine for stability)

## Volume Structure

```
/storage/
├── checkpoints/
│   ├── step-100/
│   ├── step-200/
│   └── ...
├── data/
│   └── (cached datasets)
└── hf_cache/
    └── (HuggingFace model cache)
```

## Comparison with Original `modal_trl.py`

| Aspect | Original | Modal GRPO |
|--------|----------|------------|
| TRL Installation | `pip install trl==0.19.1` | Local dir from `/trl/` |
| Base Image | `debian_slim` | NVIDIA CUDA devel |
| Volumes | 1 checkpoint volume | 1 unified volume |
| Architecture | Monolithic trainer | Separate workers |
| Weight Sync | TRL internal | Explicit via Modal |
| vLLM | TRL's built-in | Separate RolloutWorker |
| Scaling | Single function | Horizontal rollout scaling |

## GRPO Algorithm

GRPO (Group Relative Policy Optimization) is a reinforcement learning algorithm introduced by DeepSeek. The key idea is to:

1. Generate multiple completions per prompt
2. Compute rewards for each completion
3. Use relative rewards within the group to compute advantages
4. Update the policy to increase probability of higher-reward completions

Loss function:
```
L = -E[advantage * log_prob(completion)]
```

Where advantage is computed relative to other completions in the same group.

## Troubleshooting

### Image Build Failures

```bash
# Check image build logs
modal run modal_grpo/app.py
```

### Out of Memory

- Reduce `batch_size`
- Reduce `max_model_len`
- Use gradient checkpointing (enabled by default)

### Slow Generation

- Increase `num_rollout_workers`
- Check GPU utilization in Modal dashboard
- Consider using larger GPUs (A100 instead of A10G)

### Weight Sync Issues

vLLM doesn't support direct weight updates, so weights are synced via:
1. Serialize actor weights
2. Save to temp location
3. Reload vLLM engine

For production, consider using a shared filesystem or implementing custom weight update mechanism.

## License

MIT License - see repository root for details.
