"""Code execution environments using Modal Sandboxes."""

from typing import Any, List, Optional, Sequence

import modal

from MRL.app import app
from MRL.logging_config import get_logger
from MRL.workers.environment.base import (
    Rollout,
    RewardEnvironment,
    RewardResult,
)

logger = get_logger("environment.code_execution")


def extract_code_from_completion(completion: str) -> str:
    """Extract code from a completion, handling markdown code blocks.

    Args:
        completion: Model completion, potentially with markdown code blocks

    Returns:
        Extracted code string
    """
    if "```python" in completion:
        # Find the start and end of the code block
        start_idx = completion.find("```python") + len("```python")
        end_idx = completion.find("```", start_idx)
        if end_idx != -1:
            code = completion[start_idx:end_idx].strip()
        else:
            code = completion[start_idx:].strip()
    elif "```" in completion:
        # Try generic code block
        start_idx = completion.find("```") + len("```")
        # Skip language identifier if present
        newline_idx = completion.find("\n", start_idx)
        if newline_idx != -1 and newline_idx - start_idx < 20:
            start_idx = newline_idx + 1
        end_idx = completion.find("```", start_idx)
        if end_idx != -1:
            code = completion[start_idx:end_idx].strip()
        else:
            code = completion[start_idx:].strip()
    else:
        code = completion.strip()

    return code


def combine_code_with_tests(code: str, testcases: Sequence[str]) -> str:
    """Combine code with test cases.

    Args:
        code: The code to test
        testcases: List of test case assert statements

    Returns:
        Combined code string ready for execution
    """
    test_cases_str = "\n".join(testcases)
    return f"{code}\n\n{test_cases_str}"


class CodeExecutionEnvironment(RewardEnvironment):
    """Reward environment using sandbox code execution.

    Executes generated code with test cases in a secure Modal Sandbox.
    Returns 1 if all tests pass, 0 otherwise.
    """

    name = "code_execution"

    def __init__(
        self,
        timeout: int = 30,
        app_ref: Any = None,
    ):
        """Initialize the code execution environment.

        Args:
            timeout: Execution timeout in seconds
            app_ref: Modal app reference (uses default if None)
        """
        self.timeout = timeout
        self.app_ref = app_ref or app

    def compute_reward(self, rollout: Rollout, context: Any = None) -> RewardResult:
        """Compute reward by executing code with test cases.

        Args:
            rollout: The rollout containing the completion
            context: Test cases as a sequence of assert statements

        Returns:
            RewardResult (1.0 for pass, 0.0 for fail)
        """
        if context is None:
            logger.warning("No test cases provided, returning 0 reward")
            return RewardResult(reward=0.0, metadata={"error": "no_testcases"})

        testcases = list(context) if not isinstance(context, list) else context

        # Extract code and combine with tests
        code = extract_code_from_completion(rollout.completion)
        full_code = combine_code_with_tests(code, testcases)

        # Execute in sandbox
        sb = None
        reward = 0.0
        metadata = {}

        try:
            sb = modal.Sandbox.create(app=self.app_ref)
            p = sb.exec("python", "-c", full_code, timeout=self.timeout)
            p.wait()
            return_code = p.returncode

            if return_code == 0:
                reward = 1.0
                metadata["status"] = "passed"
            else:
                metadata["status"] = "failed"
                metadata["return_code"] = return_code

        except Exception as e:
            logger.debug(f"Sandbox execution error: {e}")
            metadata["status"] = "error"
            metadata["error"] = str(e)

        finally:
            if sb is not None:
                sb.terminate()

        return RewardResult(reward=reward, metadata=metadata)

    def compute_rewards_batch(
        self,
        rollouts: Sequence[Rollout],
        contexts: Optional[Sequence[Any]] = None,
    ) -> List[RewardResult]:
        """Compute rewards in parallel using Modal's starmap.

        Args:
            rollouts: Sequence of rollouts to score
            contexts: Sequence of test cases (one per rollout)

        Returns:
            List of RewardResult objects
        """
        if contexts is None:
            return [
                RewardResult(reward=0.0, metadata={"error": "no_testcases"})
                for _ in rollouts
            ]

        # Use the Modal function for parallel execution
        from MRL.workers.reward import compute_reward

        completions = [r.completion for r in rollouts]
        scores = list(compute_reward.starmap(zip(completions, contexts)))

        return [RewardResult(reward=float(s)) for s in scores]


class PartialCreditEnvironment(RewardEnvironment):
    """Reward environment with partial credit for passing some tests.

    Executes each test case separately and returns the fraction that pass.
    """

    name = "partial_credit"

    def __init__(
        self,
        timeout: int = 10,
        app_ref: Any = None,
    ):
        """Initialize the partial credit environment.

        Args:
            timeout: Per-test execution timeout in seconds
            app_ref: Modal app reference (uses default if None)
        """
        self.timeout = timeout
        self.app_ref = app_ref or app

    def compute_reward(self, rollout: Rollout, context: Any = None) -> RewardResult:
        """Compute partial credit reward.

        Args:
            rollout: The rollout containing the completion
            context: Test cases as a sequence of assert statements

        Returns:
            RewardResult with fraction of passing tests
        """
        if context is None or len(context) == 0:
            logger.warning("No test cases provided, returning 0 reward")
            return RewardResult(reward=0.0, metadata={"error": "no_testcases"})

        testcases = list(context) if not isinstance(context, list) else context

        # Extract code
        code = extract_code_from_completion(rollout.completion)

        passed = 0
        total = len(testcases)
        breakdown = {}

        for i, test in enumerate(testcases):
            sb = None
            try:
                sb = modal.Sandbox.create(app=self.app_ref)
                test_code = f"{code}\n\n{test}"
                p = sb.exec("python", "-c", test_code, timeout=self.timeout)
                p.wait()

                if p.returncode == 0:
                    passed += 1
                    breakdown[f"test_{i}"] = "passed"
                else:
                    breakdown[f"test_{i}"] = "failed"

            except Exception as e:
                breakdown[f"test_{i}"] = f"error: {str(e)}"

            finally:
                if sb is not None:
                    sb.terminate()

        reward = passed / total if total > 0 else 0.0

        return RewardResult(
            reward=reward,
            breakdown=breakdown,
            metadata={
                "passed": passed,
                "total": total,
            },
        )

    def compute_rewards_batch(
        self,
        rollouts: Sequence[Rollout],
        contexts: Optional[Sequence[Any]] = None,
    ) -> List[RewardResult]:
        """Compute partial credit rewards.

        Note: This uses sequential execution since each rollout needs
        multiple sandbox calls. Consider implementing parallel per-test
        execution for better performance with large test suites.
        """
        if contexts is None:
            return [
                RewardResult(reward=0.0, metadata={"error": "no_testcases"})
                for _ in rollouts
            ]

        # Use the Modal function for parallel execution
        from MRL.workers.reward import compute_reward_with_partial_credit

        completions = [r.completion for r in rollouts]
        scores = list(compute_reward_with_partial_credit.starmap(zip(completions, contexts)))

        return [RewardResult(reward=float(s)) for s in scores]
