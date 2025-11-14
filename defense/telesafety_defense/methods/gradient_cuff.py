"""
Gradient Cuff Defense Method
============================

This module integrates the Gradient Cuff jailbreak detection algorithm
into the telesafety defense framework.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Tuple

import torch
from fastchat.conversation import get_conv_template
from loguru import logger

from telesafety_defense.base_factory import OutputDefender


BAN_LIST: Tuple[str, ...] = (
    "I'm sorry",
    "Sorry",
    "I am sorry",
    "I apologize",
    "I cannot",
    "I can't",
    "I am not able to",
    "I am unable to",
    "I'm not able to",
    "I'm unable to",
    "SSdtIHNvcnJ5",
    "U29ycnk=",
    "SSBhbSBzb3JyeQ==",
    "SSBhcG9sb2dpemU=",
    "SSBjYW5ub3Q=",
    "SSBjYW4ndA==",
    "SSBhbSBub3QgYWJsZSB0bw==",
    "SSBhbSB1bmFibGUgdG8=",
    "SSdtIG5vdCBhYmxlIHRv",
    "SSdtIHVuYWJsZSB0bw==",
    "Es tut mir Leid",
    "Entschuldigung",
    "Es tut mir leid",
    "Ich entschuldige mich",
    "Ich kann nicht",
    "Ich kann es nicht",
    "Jag är ledsen",
    "Förlåt",
    "jag är ledsen",
    "Jag ber om ursäkt",
    "Jag kan inte",
    "Je suis désolé",
    "Désolé",
    "Je suis désolé",
    "je m'excuse",
    "Je ne peux pas",
    "je ne suis pas capable de",
    "je suis incapable de",
    "je n'en suis pas capable",
    "对不起",
    "我道歉",
    "我不能",
    "我无法",
)


class GradientCuffDefender(OutputDefender):
    """Implements the Gradient Cuff rejection-based defense."""

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        *,
        conversation_template: str | None = None,
        seed: int = 13,
        p_times: int = 10,
        sample_times: int = 10,
        mu: float = 0.02,
        threshold: float = 100.0,
        batch_size: int = 8,
        chat_temperature: float = 0.6,
        chat_max_length: int = 128,
        chat_top_p: float = 0.9,
        detect_max_new_tokens: int = 16,
        detect_temperature: float = 0.6,
        detect_top_p: float = 0.9,
        reject_message: str = "[Gradient Cuff Reject]: I cannot fulfill your request.",
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype

        self._set_seed(seed)

        self.mu = mu
        self.sample_times = sample_times
        self.p_times = p_times
        self.threshold = threshold
        self.batch_size = batch_size
        self.chat_temperature = chat_temperature
        self.chat_max_length = chat_max_length
        self.chat_top_p = chat_top_p
        self.detect_max_new_tokens = detect_max_new_tokens
        self.detect_temperature = detect_temperature
        self.detect_top_p = detect_top_p
        self.reject_message = reject_message

        self._prepare_tokenizer()

        template_name = conversation_template or model_name
        self.prefix_embedding, self.suffix_embedding = self._build_prompt_embeddings(
            template_name
        )

        embed_dim = self.prefix_embedding.shape[-1] if self.prefix_embedding.numel() else self.model.get_input_embeddings().embedding_dim
        self.shift_direction_embedding = torch.randn(
            max(self.p_times, 1),
            embed_dim,
            device=self.device,
            dtype=self.dtype,
        )

        logger.info(
            "GradientCuffDefender initialized "
            f"(p_times={self.p_times}, sample_times={self.sample_times}, mu={self.mu})"
        )

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _prepare_tokenizer(self) -> None:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def _build_prompt_embeddings(
        self, template_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        slot_token = "<gradient_cuff_user_slot>"
        try:
            conv_template = get_conv_template(template_name)
        except Exception:
            logger.warning(
                f"Conversation template '{template_name}' unavailable, falling back to 'vicuna_v1.1'."
            )
            conv_template = get_conv_template("vicuna_v1.1")

        conv_template.messages = []
        conv_template.append_message(conv_template.roles[0], slot_token)
        conv_template.append_message(conv_template.roles[1], "")
        sample_input = conv_template.get_prompt()

        slot_pos = sample_input.find(slot_token)
        if slot_pos == -1:
            raise ValueError("Failed to locate slot token in conversation template.")

        prefix_text = sample_input[:slot_pos]
        suffix_text = sample_input[slot_pos + len(slot_token) :]

        embedding_func = self.model.get_input_embeddings()
        embedding_func.weight.requires_grad_(False)

        prefix_embedding = self._encode_text(prefix_text, embedding_func)
        suffix_embedding = self._encode_text(suffix_text, embedding_func)
        if suffix_embedding.shape[0] > 0:
            suffix_embedding = suffix_embedding[1:]

        return prefix_embedding, suffix_embedding

    def _encode_text(self, text: str, embedding_func) -> torch.Tensor:
        token_ids = self.tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        if token_ids.numel() == 0:
            return torch.zeros(
                0, embedding_func.embedding_dim, device=self.device, dtype=self.dtype
            )
        return embedding_func(token_ids[0]).detach()

    def defend(self, model, messages):
        if isinstance(messages, str):
            query = messages
        elif isinstance(messages, Sequence) and messages:
            query = messages[-1]["content"]
        else:
            logger.warning("Unsupported message format for Gradient Cuff; returning original input.")
            return messages

        is_rejected, original_response = self._gradient_cuff_reject(query)
        if is_rejected:
            return self.reject_message
        return original_response

    def _gradient_cuff_reject(self, query: str) -> Tuple[bool, str]:
        embedding_func = self.model.get_input_embeddings()
        token_ids = self.tokenizer.encode(
            query, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        if token_ids.numel() == 0:
            return False, ""

        original_embedding = embedding_func(token_ids[0]).detach()

        zero_shift = torch.zeros(
            original_embedding.shape[-1], device=self.device, dtype=self.dtype
        )
        shift_embeddings = [
            self.mu * zero_shift for _ in range(max(self.sample_times, 0) + 1)
        ]
        input_embeds = self._embedding_shift(original_embedding, shift_embeddings)

        original_response = self._chat_engine(
            input_embeds[:1],
            batch_size=1,
            max_new_tokens=self.chat_max_length,
            temperature=self.chat_temperature,
            top_p=self.chat_top_p,
        )[0]

        if self.sample_times <= 0:
            return False, original_response

        responses = self._detect_engine(
            input_embeds[1:],
            max_new_tokens=self.detect_max_new_tokens,
            temperature=self.detect_temperature,
            top_p=self.detect_top_p,
        )
        labels = self._get_labels(responses)
        if not labels:
            return False, original_response

        results = [0.0 for _ in range(self.p_times + 1)]
        results[0] = sum(labels) / len(labels)
        if results[0] > 0.5:
            return True, original_response

        if self.p_times <= 0:
            return False, original_response

        shift_embeddings_stage2: List[torch.Tensor] = []
        for direction in self.shift_direction_embedding[: self.p_times]:
            shift_embeddings_stage2.extend(
                [self.mu * direction for _ in range(self.sample_times)]
            )

        if not shift_embeddings_stage2:
            return False, original_response

        input_embeds_stage2 = self._embedding_shift(
            original_embedding, shift_embeddings_stage2
        )
        responses_stage2 = self._detect_engine(
            input_embeds_stage2,
            max_new_tokens=self.detect_max_new_tokens,
            temperature=self.detect_temperature,
            top_p=self.detect_top_p,
        )

        grad_components: List[torch.Tensor] = []
        for idx in range(self.p_times):
            start = idx * self.sample_times
            end = start + self.sample_times
            window = responses_stage2[start:end]
            labels = self._get_labels(window)
            if not labels:
                continue
            results[idx + 1] = sum(labels) / len(labels)
            grad_components.append(
                (results[idx + 1] - results[0]) / self.mu * self.shift_direction_embedding[idx]
            )

        if not grad_components:
            return False, original_response

        est_grad = torch.stack(grad_components).mean(dim=0)
        if est_grad.norm().item() > self.threshold:
            return True, original_response
        return False, original_response

    def _embedding_shift(
        self, original_embedding: torch.Tensor, shift_embeddings: Iterable[torch.Tensor]
    ) -> torch.Tensor:
        shifted_embeddings = [
            original_embedding + shift.to(self.device, dtype=self.dtype)
            for shift in shift_embeddings
        ]
        combined = [
            torch.cat(
                (self.prefix_embedding, shifted, self.suffix_embedding), dim=0
            )
            for shifted in shifted_embeddings
        ]
        return torch.stack(combined, dim=0)

    def _detect_engine(
        self,
        input_embeds: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[str]:
        outputs: List[str] = []
        if input_embeds.numel() == 0:
            return outputs

        with torch.no_grad():
            for start in range(0, input_embeds.size(0), self.batch_size):
                batch_input = input_embeds[start : start + self.batch_size]
                attn_mask = torch.ones(
                    batch_input.size()[:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                generated = self.model.generate(
                    inputs_embeds=batch_input,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                outputs.extend(
                    self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                )
        return outputs

    def _chat_engine(
        self,
        input_embeds: torch.Tensor,
        *,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[str]:
        outputs: List[str] = []
        with torch.no_grad():
            for start in range(0, input_embeds.size(0), batch_size):
                batch_input = input_embeds[start : start + batch_size]
                attn_mask = torch.ones(
                    batch_input.size()[:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                generated = self.model.generate(
                    inputs_embeds=batch_input,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                outputs.extend(
                    self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                )
        return outputs

    @staticmethod
    def _get_labels(responses: Sequence[str]) -> List[int]:
        labels = []
        for response in responses:
            labels.append(int(any(trigger in response for trigger in BAN_LIST)))
        return labels
