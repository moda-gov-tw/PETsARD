import os
import random
import sys
import time
from collections.abc import Callable
from typing import Any

try:
    from ..api import APITemplate
    from ..utils import LLM_result_logger
    from .agents.init_generation_agent import SYSTEM_PROMPT_DP as INIT_SYSTEM_PROMPT
    from .agents.init_generation_agent import (
        DPInitGenerationAgent as InitGenerationAgent,
    )
    from .agents.post_processing_agent import (
        SYSTEM_PROMPT as POST_PROCESSING_SYSTEM_PROMPT,
    )
    from .agents.post_processing_agent import PostProcessingAgent
    from .agents.variation_agent import SYSTEM_PROMPT as VARIATION_SYSTEM_PROMPT
    from .agents.variation_agent import VariationAgent
except ImportError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from api import APITemplate
    from synthetic_data.agents.init_generation_agent import (
        SYSTEM_PROMPT_DP as INIT_SYSTEM_PROMPT,
    )
    from synthetic_data.agents.init_generation_agent import (
        DPInitGenerationAgent as InitGenerationAgent,
    )
    from synthetic_data.agents.post_processing_agent import (
        SYSTEM_PROMPT as POST_PROCESSING_SYSTEM_PROMPT,
    )
    from synthetic_data.agents.post_processing_agent import PostProcessingAgent
    from synthetic_data.agents.variation_agent import (
        SYSTEM_PROMPT as VARIATION_SYSTEM_PROMPT,
    )
    from synthetic_data.agents.variation_agent import (
        VariationAgent,
    )
    from utils import LLM_result_logger


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "random_random",
            "description": "Generate a random floating-point number in the range [0.0, 1.0).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "random_uniform",
            "description": "Generate a random floating-point number N such that min <= N <= max.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min": {"type": "number", "description": "Lower bound (inclusive)"},
                    "max": {"type": "number", "description": "Upper bound (inclusive)"},
                },
                "required": ["min", "max"],
            },
        },
    },
]
# ── built-in implementations for the TOOLS defined above ──────────────────────
'''
_RANDOM_FUNCTIONS = {
    "random_random": lambda: random.random(),
    "random_uniform": lambda min, max: rando.uniform(float(min), float(max)),
}
'''
# Tool-call numbers come from an OS-seeded RNG, not the global one: the global RNG is
# reset to `seed` on every (re)start, which would replay the same numbers each run.
_TOOL_RNG = random.Random()

def get_rnd_functions(logger: LLM_result_logger.Logger) -> dict[str, Callable[..., Any]]:
    """
    """
    return {
        "random_random": lambda: (_TOOL_RNG.random(), logger.simple_log("TOOL CALLS: random_random"))[0],
        "random_uniform": lambda min, max: (_TOOL_RNG.uniform(float(min), float(max)), logger.simple_log(f"TOOL CALLS: random_uniform, min: {min}, max: {max}"))[0],
    }

class InvalidOutput(ValueError):
    """The model answered, but the answer is not a usable record (e.g. wrong field count).

    Retried at once (no retry wait: the server is fine); after the last attempt the
    generation is left empty instead of stopping the run."""


class LLM_Generation:
    def __init__(self,
        init_generation_api: APITemplate,
        variation_api: APITemplate,
        post_processing_api: APITemplate,
        logger: LLM_result_logger.Logger,
        enable_tools: bool = True,
        sleep_time: float = 1.0,
        max_retries: int = 3,
        additional_info: str = "",
        exit_on_api_error: bool = False,
        seed: int = None,
        retry_wait_time: float = 15.0,
        output_check: Callable[..., tuple[str, str | None]] | None = None,
        init_additional_info: str | None = None,
    ):
        self.init_generation_api = init_generation_api
        # the initial generation may write fewer columns than the variation (label vote)
        self._init_additional_info = init_additional_info
        # output_check(text, masked_seq) -> (clean_text, error); masked_seq is None for the
        # initial generation. A non-empty error retries the call.
        self._output_check = output_check
        self.variation_api = variation_api
        self.post_processing_api = post_processing_api
        self.logger = logger
        self._enable_tools = enable_tools
        self._sleep_time = sleep_time
        # fixed wait before each retry (no backoff); separate from sleep_time, which is slept before every call
        self._retry_wait_time = retry_wait_time
        self._max_retries = max_retries
        self._additional_info = additional_info
        self._exit_on_api_error = exit_on_api_error

        if self._enable_tools:
            self.init_generation_api.set_function_registry(get_rnd_functions(self.logger))
            self.variation_api.set_function_registry(get_rnd_functions(self.logger))

        if type(seed) is int:
            random.seed(seed)

        self.init_generation_api.set_system_prompt(INIT_SYSTEM_PROMPT)
        self.variation_api.set_system_prompt(VARIATION_SYSTEM_PROMPT)
        self.post_processing_api.set_system_prompt(POST_PROCESSING_SYSTEM_PROMPT)

    def _give_up(self, name: str, exc: Exception) -> None:
        msg = f"{name} failed after {self._max_retries} attempt(s): {exc}"
        self.logger.simple_log(msg)
        if self._exit_on_api_error:
            # os._exit: sys.exit would only end this worker thread
            print(msg, file=sys.stderr, flush=True)
            os._exit(1)

    def _output_of(self, result, masked_seq: str | None = None) -> str:
        output = result.original_output or ""
        if not output.strip():
            raise ValueError("model returned an empty response")
        if self._output_check is not None:
            output, error = self._output_check(output, masked_seq)
            if error:
                raise InvalidOutput(error)
        return output

    def _retry_or_give_up(self, name: str, attempt: int, exc: Exception) -> None:
        """Wait before the next attempt, or give up after the last one."""
        if attempt < self._max_retries - 1:
            wait = 0.0 if isinstance(exc, InvalidOutput) else self._retry_wait_time
            self.logger.simple_log(
                f"{name} attempt {attempt + 1}/{self._max_retries} failed: {exc} — retrying in {wait:.1f}s"
            )
            if wait > 0:
                time.sleep(wait)
        elif isinstance(exc, InvalidOutput):
            # a bad answer is not a server failure: leave this generation empty and go on
            self.logger.simple_log(f"{name} gave up after {self._max_retries} invalid answer(s): {exc}")
        else:
            self._give_up(name, exc)

    def generate_init(
        self,
        request_params: dict[str, Any],
        examples: str
    ) -> str:
        """
        """
        if self._sleep_time > 0:
            time.sleep(self._sleep_time)

        tools = None
        if self._enable_tools:
            tools = TOOLS

        info = self._additional_info if self._init_additional_info is None else self._init_additional_info
        init_generation_agent = InitGenerationAgent(examples=examples, additional_info=info)

        output = ""
        for attempt in range(self._max_retries):
            try:
                result = init_generation_agent.run(self.init_generation_api, tools, request_params, self.logger)
                output = self._output_of(result)
                break

            except Exception as exc:  # noqa: BLE001
                self._retry_or_give_up("generate_init()", attempt, exc)
        if not output:
            return ""
        return output
        #return self.generate_post_processing(request_params, output)

    def generate_variation(
        self,
        request_params: dict[str, Any],
        examples: str,
        masked_seq: str
    ) -> str:
        """
        """
        if self._sleep_time > 0:
            time.sleep(self._sleep_time)

        tools = None
        if self._enable_tools:
            tools = TOOLS

        variation_agent = VariationAgent(masked_seq=masked_seq, examples=examples, additional_info=self._additional_info)

        output = ""
        for attempt in range(self._max_retries):
            try:
                result = variation_agent.run(self.variation_api, tools, request_params, self.logger)
                output = self._output_of(result, masked_seq)
                break

            except Exception as exc:  # noqa: BLE001
                self._retry_or_give_up("generate_variation()", attempt, exc)
        if not output:
            return ""
        return output
        #return self.generate_post_processing(request_params, output)

    def generate_post_processing(
        self,
        request_params: dict[str, Any],
        sequence: str,
    ) -> str:
        """
        """
        if self._sleep_time > 0:
            time.sleep(self._sleep_time)

        post_processing_agent = PostProcessingAgent(sequence=sequence)

        for attempt in range(self._max_retries):
            try:
                result = post_processing_agent.run(
                    self.post_processing_api, request_params, self.logger
                )
                return self._output_of(result)

            except Exception as exc:  # noqa: BLE001
                if attempt < self._max_retries - 1:
                    wait = self._retry_wait_time
                    self.logger.simple_log(
                        f"generate_post_processing() attempt {attempt + 1}/{self._max_retries} failed: {exc} — retrying in {wait:.1f}s"
                    )
                    time.sleep(wait)
                else:
                    self._give_up("generate_post_processing()", exc)
        return ""
