"""
RAIN Defense Method
===================

This module implements a lightweight approximation of the RAIN inference-time
defense. The defender samples candidate responses, evaluates their safety via
model-based self-judgement, and rewinds unsafe continuations to resample safer
alternatives before falling back to a refusal message.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import torch
from loguru import logger

from telesafety_defense.base_factory import OutputDefender


DEFAULT_SAFE_RESPONSE = "[RAIN Reject]: I'm sorry, but I can't help with that request."


class RAINDefender(OutputDefender):
    """Approximate implementation of the RAIN self-evaluation defence."""

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        *,
        candidate_attempts: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        evaluation_threshold: float = 0.0,
        rewind_step_tokens: int = 40,
        max_rewinds: int = 3,
        safe_response: str = DEFAULT_SAFE_RESPONSE,
        asset_root: Optional[str] = None,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.candidate_attempts = max(1, candidate_attempts)
        self.max_new_tokens = max_new_tokens
        self.temperature = max(1e-4, temperature)
        self.top_p = min(max(top_p, 0.0), 1.0)
        self.evaluation_threshold = evaluation_threshold
        self.rewind_step_tokens = max(1, rewind_step_tokens)
        self.max_rewinds = max(0, max_rewinds)
        self.safe_response = safe_response

        assets_path = (
            Path(asset_root)
            if asset_root
            else Path(__file__).resolve().parents[2] / "method_utils" / "RAIN" / "HH"
        )
        try:
            self._fs_chat = (assets_path / "f1.txt").read_text(encoding="utf-8").strip()
            self._fs_red = (assets_path / "f2.txt").read_text(encoding="utf-8").strip()
            self._red_a = (assets_path / "r1.txt").read_text(encoding="utf-8").strip()
            self._red_b = (assets_path / "r2.txt").read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"RAIN assets not found under '{assets_path}'. "
                "Please ensure method_utils/RAIN assets are available."
            ) from exc

    def defend(self, model, messages):
        query = self._extract_user_query(messages)
        if not query:
            logger.warning("RAINDefender received empty user query; returning safe response.")
            return self.safe_response

        target_model = getattr(model, "model", self.model).eval()
        target_tokenizer = getattr(model, "tokenizer", self.tokenizer)
        self._prepare_tokenizer(target_tokenizer)

        device = next(target_model.parameters()).device
        prompt_text = self._build_prompt(query)
        prompt_ids = target_tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        best_response = None
        best_score = -math.inf

        for attempt in range(self.candidate_attempts):
            candidate_ids = self._sample_response(
                target_model, prompt_ids.clone(), target_tokenizer
            )
            candidate_text = target_tokenizer.decode(
                candidate_ids, skip_special_tokens=True
            ).strip()

            conversation_text = self._conversation_text(query, candidate_text)
            score = self._evaluate_safety(
                target_model, target_tokenizer, conversation_text, device
            )

            if score >= self.evaluation_threshold:
                logger.debug(f"RAINDefender accepted candidate on attempt {attempt + 1}.")
                return candidate_text

            if score > best_score and candidate_text:
                best_score = score
                best_response = candidate_text

            rewound = self._attempt_rewinds(
                target_model,
                target_tokenizer,
                prompt_ids,
                candidate_ids,
                query,
                device,
            )
            if rewound is not None:
                logger.debug(
                    f"RAINDefender accepted rewound candidate on attempt {attempt + 1}."
                )
                return rewound

        if best_response:
            logger.info(
                "RAINDefender returning best available candidate despite low safety score."
            )
            return best_response

        logger.info("RAINDefender falling back to safe refusal response.")
        return self.safe_response

    @staticmethod
    def _extract_user_query(messages) -> str:
        if isinstance(messages, str):
            return messages.strip()
        if isinstance(messages, Sequence) and messages:
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    return (message.get("content") or "").strip()
            last = messages[-1]
            if isinstance(last, dict):
                return (last.get("content") or "").strip()
        return ""

    def _build_prompt(self, query: str) -> str:
        return f"{self._fs_chat}\n\nHuman: {query.strip()}\nAssistant:"

    def _conversation_text(self, query: str, response: str) -> str:
        return f"Human: {query.strip()}\nAssistant: {response.strip()}"

    def _prepare_tokenizer(self, tokenizer) -> None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    def _sample_response(self, model, prompt_ids, tokenizer):
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "pad_token_id": tokenizer.eos_token_id,
        }
        with torch.no_grad():
            output = model.generate(prompt_ids, **generation_kwargs)
        prompt_len = prompt_ids.shape[1]
        return output[0, prompt_len:]

    def _attempt_rewinds(
        self,
        model,
        tokenizer,
        prompt_ids,
        candidate_ids,
        query: str,
        device: torch.device,
    ) -> Optional[str]:
        if candidate_ids.numel() == 0 or self.max_rewinds == 0:
            return None

        working_ids = candidate_ids.clone()

        for _ in range(self.max_rewinds):
            if working_ids.numel() <= self.rewind_step_tokens:
                break
            working_ids = working_ids[:-self.rewind_step_tokens]
            trimmed_prompt = torch.cat(
                [prompt_ids.to(device), working_ids.unsqueeze(0)], dim=1
            )

            remaining = max(self.max_new_tokens - working_ids.numel(), 1)
            generation_kwargs = {
                "max_new_tokens": remaining,
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "pad_token_id": tokenizer.eos_token_id,
            }

            with torch.no_grad():
                output = model.generate(trimmed_prompt, **generation_kwargs)

            new_segment = output[0, trimmed_prompt.shape[1] :]
            combined = torch.cat([working_ids, new_segment], dim=0)
            candidate_text = tokenizer.decode(combined, skip_special_tokens=True).strip()

            if not candidate_text:
                continue

            conversation_text = self._conversation_text(query, candidate_text)
            score = self._evaluate_safety(model, tokenizer, conversation_text, device)
            if score >= self.evaluation_threshold:
                return candidate_text

        return None

    def _evaluate_safety(self, model, tokenizer, conversation_text: str, device) -> float:
        try:
            base = f"{self._fs_red}\n\n{conversation_text.strip()}\n"
            text_a = base + self._red_a
            text_b = base + self._red_b

            logprob_a_A = self._option_logprob(model, tokenizer, text_a, "A", device)
            logprob_a_B = self._option_logprob(model, tokenizer, text_a, "B", device)
            a_label = 1 if logprob_a_A >= logprob_a_B else 0

            logprob_b_A = self._option_logprob(model, tokenizer, text_b, "A", device)
            logprob_b_B = self._option_logprob(model, tokenizer, text_b, "B", device)
            b_label = 1 if logprob_b_B >= logprob_b_A else 0

            score = (a_label + b_label) / 2.0
            return (score - 0.5) * 2.0
        except Exception as exc:
            logger.warning(f"RAINDefender safety evaluation failed: {exc}")
            return -1.0

    @staticmethod
    def _option_logprob(model, tokenizer, prompt_text: str, option: str, device) -> float:
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        option_ids = tokenizer(
            option, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        if option_ids.numel() == 0:
            raise ValueError("Option text tokenised to empty sequence.")

        input_ids = torch.cat([prompt_ids, option_ids], dim=1)

        with torch.no_grad():
            outputs = model(input_ids)
            log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)

        target_ids = input_ids[:, 1:]
        selected = torch.gather(log_probs, -1, target_ids.unsqueeze(-1)).squeeze(-1)
        continuation = selected[:, -option_ids.shape[1] :]
        return float(continuation.sum().item())
