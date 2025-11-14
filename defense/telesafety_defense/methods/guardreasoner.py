"""
GuardReasoner Defense Method
============================

This defender wraps the GuardReasoner post-processing classifier to judge
model responses for potentially harmful behavior. If either the user
request or model response is marked harmful, the defender returns a
standard refusal; otherwise the original model response is released.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Mapping, Sequence

from loguru import logger
from vllm import LLM, SamplingParams

from telesafety_defense.base_factory import OutputDefender
from telesafety_defense.utils import SORRY_RESPONSE

try:
    from aisafetylab.defense.inference_defense.postprocess_defenders.GuardReasoner.template import (
        INSTUCT,
    )
except ImportError as exc:
    raise ImportError(
        "GuardReasoner templates could not be imported. Ensure aisafetylab is "
        "installed and accessible."
    ) from exc


class GuardReasonerDefender(OutputDefender):
    """
    Evaluate generated responses with GuardReasoner and refuse when unsafe.

    Args:
        model_path: Filesystem path to the GuardReasoner weights.
        sampling_kwargs: Optional overrides passed to `SamplingParams`.
        llm_kwargs: Optional overrides passed to `vllm.LLM`.
    """

    def __init__(
        self,
        model_path: str = "/gemini/space/lxw/GuardReasoner-8B",
        sampling_kwargs: Mapping | None = None,
        **llm_kwargs,
    ) -> None:

        default_sampling = {
            "temperature": 0.1,
            "max_tokens": 512,
            "top_p": 0.9,
            "top_k": 40,
            "stop": ["</s>"],
        }
        if sampling_kwargs:
            default_sampling.update(sampling_kwargs)

        self.sampling_params = SamplingParams(**default_sampling)

        default_llm_kwargs = {
            "model": model_path,
            "gpu_memory_utilization": 0.8,
            "cpu_offload_gb": 8,
            "tensor_parallel_size": 2,
            "max_seq_len_to_capture": 8096,
        }
        default_llm_kwargs.update(llm_kwargs)

        logger.info("Loading GuardReasoner model from {}", default_llm_kwargs["model"])
        self.vllm_model = LLM(**default_llm_kwargs)

    @staticmethod
    def _ensure_messages(payload: str | Sequence[Mapping[str, str]]) -> List[Mapping[str, str]]:
        if isinstance(payload, str):
            return [{"role": "user", "content": payload}]
        if isinstance(payload, Iterable):
            messages = list(payload)
            if not messages:
                raise ValueError("Empty message sequence provided to GuardReasonerDefender.")
            return messages
        raise TypeError(f"Unsupported payload type for GuardReasonerDefender: {type(payload)}")

    @staticmethod
    def _post_process(text: str) -> str:
        text = re.sub(r"task 1", r"task 1: prompt harmfulness detection", text, flags=re.IGNORECASE)
        text = re.sub(r"task 2", r"task 2: refusal detection", text, flags=re.IGNORECASE)
        text = re.sub(r"task 3", r"task 3: response harmfulness detection", text, flags=re.IGNORECASE)
        return text

    def _guard_prompt(self, prompt: str, response: str) -> str:
        return INSTUCT + f"Human user:\n{prompt}\n\nAI assistant:\n{response}\n\n"

    def _classify(self, prompt: str, response: str) -> str:
        guard_prompt = self._guard_prompt(prompt, response)
        outputs = self.vllm_model.generate(guard_prompt, self.sampling_params)
        guard_text = outputs[0].outputs[0].text
        return self._post_process(guard_text)

    @staticmethod
    def _extract_label(pattern: str, text: str) -> str:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return match.group(1).lower() if match else ""

    def defend(self, model, messages: str | Sequence[Mapping[str, str]]) -> str:
        """
        Generate a response with the provided model and filter via GuardReasoner.

        Args:
            model: Model wrapper exposing `chat`.
            messages: User payload (string or chat-style message list).
        """
        chat_messages = self._ensure_messages(messages)
        user_prompt = chat_messages[-1]["content"]
        base_response = model.chat(chat_messages)

        try:
            guard_text = self._classify(user_prompt, base_response)
        except Exception as exc:  # noqa: BLE001
            logger.error("GuardReasoner classification failed: {}", exc)
            return SORRY_RESPONSE

        request_label = self._extract_label(r"Request:\s*(\w+)", guard_text)
        response_label = self._extract_label(r"Response:\s*(\w+)", guard_text)

        logger.debug(
            "GuardReasoner verdict — request: {}, response: {}, reasoning: {}",
            request_label,
            response_label,
            guard_text,
        )

        if request_label == "harmful" or response_label == "harmful":
            return SORRY_RESPONSE

        return base_response

