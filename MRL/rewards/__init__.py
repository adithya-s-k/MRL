"""Reward environments for MRL training.

Core classes:
    RewardEnvironment  - Base class for custom reward environments
    SandboxConfig      - Configuration for Modal Sandbox execution
    FunctionConfig     - Configuration for Modal Function execution
    ExecutionResult    - Result from remote code execution
    compute_rewards    - Dispatch layer for scoring completions

Example environments (in MRL.rewards.examples):
    CodeExecutionEnvironment - Execute code in sandboxes, score by test pass/fail
    LLMJudgeEnvironment      - Use an LLM to score completions via API
"""

from MRL.rewards.base import ExecutionResult, FunctionConfig, RewardEnvironment, SandboxConfig
from MRL.rewards.dispatch import compute_rewards
from MRL.rewards.utils import extract_code_from_completion

__all__ = [
    "RewardEnvironment",
    "SandboxConfig",
    "FunctionConfig",
    "ExecutionResult",
    "compute_rewards",
    "extract_code_from_completion",
]
