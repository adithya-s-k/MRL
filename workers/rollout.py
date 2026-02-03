"""Rollout worker for vLLM inference."""

from typing import Optional

import modal

# Import app and resources from the shared app module
from MRL.app import app, volume, VLLM_IMAGE

STORAGE_PATH = "/storage"


@app.cls(
    image=VLLM_IMAGE,
    gpu="A10G",
    volumes={STORAGE_PATH: volume},
    scaledown_window=300,  # Keep warm for 5 minutes
    timeout=600,  # 10 minute timeout per method call
    secrets=[modal.Secret.from_name("adithya-hf-wandb")],
)
class RolloutWorker:
    """Standalone vLLM inference worker for generating completions.

    This worker uses vLLM for fast inference and can be scaled horizontally.
    It maintains a cached model and can reload from checkpoints when the
    actor's weights are updated.
    """

    @modal.enter()
    def setup(self):
        """Initialize on container start."""
        import os

        # MUST set before any vllm imports
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Set multiprocessing to spawn mode
        import multiprocessing

        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set

        self.llm = None
        self.current_model_path = None
        self.tokenizer = None
        print("RolloutWorker container started (lazy vLLM init)")

    def _load_model(
        self,
        model_path: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
    ):
        """Load or reload vLLM engine.

        Args:
            model_path: Path to model (HuggingFace name or local checkpoint)
            max_model_len: Maximum sequence length
            gpu_memory_utilization: Fraction of GPU memory to use
        """
        from vllm import LLM
        import torch

        if model_path == self.current_model_path and self.llm is not None:
            return  # Already loaded

        # Clean up existing model
        if self.llm is not None:
            print(f"Unloading current model: {self.current_model_path}")
            del self.llm
            torch.cuda.empty_cache()

        print(f"Loading model: {model_path}")
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=True,  # Disable CUDA graphs for stability
        )
        self.current_model_path = model_path
        print(f"Model loaded successfully: {model_path}")

    @modal.method()
    def generate(
        self,
        prompts: list[str],
        model_path: str = "Qwen/Qwen2-0.5B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 1,
        max_model_len: int = 4096,
    ) -> dict:
        """Generate completions with logprobs.

        Args:
            prompts: List of prompts to generate from
            model_path: Model to use (HF name or checkpoint path)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            n: Number of completions per prompt
            max_model_len: Maximum model context length

        Returns:
            Dictionary with completions and logprobs
        """
        from vllm import SamplingParams

        self._load_model(model_path, max_model_len=max_model_len)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            logprobs=1,  # Return top-1 logprob for each token
        )

        print(f"Generating {len(prompts)} prompts with n={n}...")
        outputs = self.llm.generate(prompts, sampling_params)

        # Process outputs
        completions = []
        all_logprobs = []
        prompt_indices = []  # Track which prompt each completion belongs to

        for prompt_idx, output in enumerate(outputs):
            for completion in output.outputs:
                completions.append(completion.text)
                prompt_indices.append(prompt_idx)

                # Extract logprobs
                if completion.logprobs:
                    token_logprobs = []
                    for lp in completion.logprobs:
                        if lp:
                            # Get the logprob of the sampled token
                            token_logprobs.append(list(lp.values())[0].logprob)
                    all_logprobs.append(token_logprobs)
                else:
                    all_logprobs.append([])

        print(f"Generated {len(completions)} completions")
        return {
            "completions": completions,
            "logprobs": all_logprobs,
            "prompt_indices": prompt_indices,
        }

    @modal.method()
    def reload_from_checkpoint(self, checkpoint_path: str) -> bool:
        """Reload model from volume checkpoint.

        Args:
            checkpoint_path: Path to checkpoint on the volume

        Returns:
            True if reload successful
        """
        import torch

        print(f"Reloading from checkpoint: {checkpoint_path}")

        # Free GPU memory first
        if self.llm is not None:
            del self.llm
            self.llm = None
            self.current_model_path = None
            torch.cuda.empty_cache()

        # Get latest files from volume
        volume.reload()

        # Load model from checkpoint
        self._load_model(checkpoint_path)
        return True

    @modal.method()
    def update_weights(self, weights_bytes: bytes) -> bool:
        """Update model weights from serialized state dict.

        This is used for weight sync from the actor worker without
        writing to disk.

        Args:
            weights_bytes: Serialized state dict bytes

        Returns:
            True if update successful
        """
        import io
        import os
        import shutil
        import torch
        from safetensors.torch import save_file

        if self.llm is None:
            print("Warning: No model loaded, cannot update weights")
            return False

        print("Updating model weights from actor...")

        # First, free GPU memory by deleting vLLM engine
        original_model_path = self.current_model_path
        del self.llm
        self.llm = None
        torch.cuda.empty_cache()

        # Load state dict to CPU to avoid OOM
        buffer = io.BytesIO(weights_bytes)
        state_dict = torch.load(buffer, map_location="cpu", weights_only=True)

        # vLLM doesn't support direct weight updates, so we need to
        # save to disk and reload. This is a limitation of vLLM.
        temp_path = f"{STORAGE_PATH}/temp_weights"
        os.makedirs(temp_path, exist_ok=True)

        # Save weights in safetensors format (preferred by vLLM)
        save_file(state_dict, f"{temp_path}/model.safetensors")

        # Copy config files from the original model if it's a HF model
        # This is needed for vLLM to properly load the model
        from transformers import AutoConfig, AutoTokenizer

        try:
            # Load and save config/tokenizer from original model
            config = AutoConfig.from_pretrained(original_model_path)
            config.save_pretrained(temp_path)

            tokenizer = AutoTokenizer.from_pretrained(original_model_path)
            tokenizer.save_pretrained(temp_path)
        except Exception as e:
            print(f"Warning: Could not copy config/tokenizer: {e}")
            print("Falling back to original model path for reload")
            # Clean up and reload original model
            del state_dict
            torch.cuda.empty_cache()
            self._load_model(original_model_path)
            return False

        # Clean up state dict from CPU memory
        del state_dict
        torch.cuda.empty_cache()

        # Reload vLLM with updated weights
        self._load_model(temp_path)
        print("Weights updated successfully")
        return True

    @modal.method()
    def health_check(self) -> dict:
        """Check worker health and return status.

        Returns:
            Dictionary with health status
        """
        return {
            "status": "healthy",
            "model_loaded": self.llm is not None,
            "current_model": self.current_model_path,
        }

    @modal.method()
    def compute_logprobs(
        self,
        prompts: list[str],
        completions: list[str],
        model_path: str = "Qwen/Qwen2-0.5B-Instruct",
        max_model_len: int = 4096,
    ) -> list[list[float]]:
        """Compute log probabilities for given prompt-completion pairs.

        This is useful for computing reference model log probs or
        importance weights.

        Args:
            prompts: List of prompts
            completions: List of completions (one per prompt)
            model_path: Model to use
            max_model_len: Maximum context length

        Returns:
            List of log probability lists (one per completion)
        """
        from vllm import SamplingParams

        self._load_model(model_path, max_model_len=max_model_len)

        # Combine prompts and completions
        full_texts = [p + c for p, c in zip(prompts, completions)]

        # Use prompt_logprobs to get logprobs for the completion tokens
        sampling_params = SamplingParams(
            max_tokens=1,  # We don't need to generate, just compute logprobs
            temperature=1.0,
            prompt_logprobs=1,
        )

        outputs = self.llm.generate(full_texts, sampling_params)

        all_logprobs = []
        for i, output in enumerate(outputs):
            if output.prompt_logprobs:
                # Get logprobs for completion portion only
                prompt_len = len(self.llm.get_tokenizer().encode(prompts[i]))
                completion_logprobs = []
                for j, lp in enumerate(output.prompt_logprobs):
                    if j >= prompt_len and lp:
                        completion_logprobs.append(list(lp.values())[0].logprob)
                all_logprobs.append(completion_logprobs)
            else:
                all_logprobs.append([])

        return all_logprobs
