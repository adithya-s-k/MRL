"""Test script to debug vLLM on Modal."""

import modal

app = modal.App("test-vllm")

# Use latest vLLM with proper multiprocessing settings
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("pip install uv")
    .run_commands("uv pip install vllm hf_transfer --system")
    .env(
        {
            "VLLM_USE_V1": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
)

volume = modal.Volume.from_name("grpo-trl-storage", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu="A10G",
    volumes={"/storage": volume},
    secrets=[modal.Secret.from_name("adithya-hf-wandb")],
    timeout=600,
)
def test_vllm_generation():
    """Test basic vLLM generation."""
    import os

    # MUST set before any vllm imports
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    from vllm import LLM, SamplingParams

    print(f"vLLM version: {__import__('vllm').__version__}")
    print("Creating LLM...")
    llm = LLM(
        model="Qwen/Qwen2-0.5B-Instruct",
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        enforce_eager=True,
        # disable_async_output_proc=True,  # Disable async to avoid multiprocessing issues
    )

    print("LLM created successfully!")

    prompts = ["Write a Python function to add two numbers:"]
    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.7,
    )

    print("Generating...")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"Prompt: {output.prompt[:50]}...")
        print(f"Generated: {output.outputs[0].text[:200]}...")

    return {"status": "success", "num_outputs": len(outputs)}


@app.local_entrypoint()
def main():
    """Run the test."""
    print("Testing vLLM on Modal...")
    result = test_vllm_generation.remote()
    print(f"Result: {result}")
