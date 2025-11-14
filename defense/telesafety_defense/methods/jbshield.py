"""
JBShield Defense Method
=======================

This module integrates the JBShield mitigation defense into the telesafety
framework. It adapts the original JBShield-M implementation to register
concept-manipulation hooks on the target language model during generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from loguru import logger

from telesafety_defense.base_factory import OutputDefender


DEFAULT_VARIANT_ORDER: Sequence[str] = (
    "gcg",
    "puzzler",
    "saa",
    "autodan",
    "drattack",
    "pair",
    "ijp",
    "base64",
    "zulu",
)

MODEL_NAME_MAP = {
    "vicuna-7b-v1.5": "vicuna-7b",
    "vicuna-13b-v1.5": "vicuna-13b",
    "Llama-2-7b-chat-hf": "llama-2",
    "Meta-Llama-3-8B-Instruct": "llama-3",
    "Mistral-7B-Instruct-v0.2": "mistral",
}


def _torch_load(path: Path, *, map_location: str = "cpu"):
    """Load serialized JBShield assets using legacy unpickling semantics."""
    return torch.load(path, map_location=map_location, weights_only=False)


def _ensure_tensor(data: Iterable, *, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert nested sequences or tensors into a single torch.Tensor.
    """
    if torch.is_tensor(data):
        tensor = data
    elif isinstance(data, (list, tuple)):
        if not data:
            raise ValueError("Cannot convert empty sequence to tensor.")
        tensor = torch.stack(
            [_ensure_tensor(item, device=device) for item in data]
        )
    else:
        tensor = torch.tensor(data)

    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _align_prefix(tensor: torch.Tensor, length: int) -> torch.Tensor:
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.size(0) < length:
        repeat_count = length - tensor.size(0)
        tensor = torch.cat([tensor, tensor[-1:].repeat(repeat_count, 1)], dim=0)
    return tensor[:length]


def get_difference_matrix(embeddings1, embeddings2) -> torch.Tensor:
    """Compute difference matrix between two embedding sequences."""
    tensor1 = _ensure_tensor(embeddings1)
    tensor2 = _ensure_tensor(embeddings2, device=tensor1.device)

    if tensor1.dim() == 1:
        tensor1 = tensor1.unsqueeze(0)
    if tensor2.dim() == 1:
        tensor2 = tensor2.unsqueeze(0)

    min_len = min(tensor1.size(0), tensor2.size(0))
    tensor1 = tensor1[:min_len]
    tensor2 = _align_prefix(tensor2, min_len)
    return tensor1 - tensor2


def get_svd(difference_matrix: torch.Tensor):
    """Compute SVD for the provided difference matrix."""
    if difference_matrix.dim() == 1:
        difference_matrix = difference_matrix.unsqueeze(0)
    matrix = difference_matrix.to(torch.float32)
    try:
        u, s, v = torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError:
        u, s, v = torch.svd(matrix, some=True)
    return u, s, v


def interpret_difference_matrix(
    model,
    tokenizer,
    embeddings1,
    embeddings2,
    top_k: int = 10,
    return_tokens: bool = True,
):
    """
    Compute the dominant concept vector between two embedding sets.

    The token interpretation branch is not required for JBShield integration,
    so ``return_tokens`` must be False.
    """
    if return_tokens:
        raise NotImplementedError("Token interpretation is not implemented in this integration.")

    diff = get_difference_matrix(embeddings1, embeddings2)

    if diff.numel() == 0:
        raise ValueError("Difference matrix is empty; unable to interpret concept vector.")

    _, _, v = get_svd(diff)
    concept_vector = v[0].detach().cpu()

    tensor1 = _ensure_tensor(embeddings1).to(torch.float32).cpu()
    tensor2 = _ensure_tensor(embeddings2).to(torch.float32).cpu()
    if tensor1.dim() == 1:
        tensor1 = tensor1.unsqueeze(0)
    if tensor2.dim() == 1:
        tensor2 = tensor2.unsqueeze(0)
    min_len = min(tensor1.size(0), tensor2.size(0))
    tensor1 = tensor1[:min_len]
    tensor2 = tensor2[:min_len]

    projections1 = torch.matmul(tensor1, concept_vector)
    projections2 = torch.matmul(tensor2, concept_vector)
    delta = torch.mean(projections1) - torch.mean(projections2)
    return concept_vector, delta


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between two vectors."""
    v1 = v1.view(1, -1)
    v2 = v2.view(1, -1)
    return torch.nn.functional.cosine_similarity(v1, v2, dim=1).squeeze(0)


@dataclass
class JBShieldVariantAssets:
    name: str
    base_jailbreak_vector: torch.Tensor
    threshold_safety: float
    threshold_jailbreak: float
    delta_safety: float
    delta_jailbreak: float
    selected_safety_layer_index: int
    selected_jailbreak_layer_index: int


class JBShieldManipulator:
    """
    Runtime helper that registers JBShield concept-manipulation hooks on a model.
    """

    def __init__(
        self,
        model,
        tokenizer,
        mean_harmless_embedding,
        mean_harmful_embedding,
        base_safety_vector,
        assets: JBShieldVariantAssets,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mean_harmless_embedding = mean_harmless_embedding
        self.mean_harmful_embedding = mean_harmful_embedding
        self.base_safety_vector = base_safety_vector
        self.assets = assets
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def detection(self, embeddings1, base_embedding, base_vector, threshold: float):
        results = []
        for embed in embeddings1:
            vec, _ = interpret_difference_matrix(
                self.model,
                self.tokenizer,
                embed,
                base_embedding,
                return_tokens=False,
            )
            vec_tensor = vec.to(torch.float32)
            if isinstance(base_vector, torch.Tensor):
                base_tensor = base_vector.to(vec_tensor.device, dtype=torch.float32)
            else:
                base_tensor = torch.tensor(base_vector, device=vec_tensor.device, dtype=torch.float32)
            score = cosine_similarity(vec_tensor, base_tensor).item()
            results.append(1.0 if score >= threshold else 0.0)
        return results

    def hook_fn_safety(self, module, _input, output):
        hidden_states, repack_output = self._extract_hidden_states(output, hook_name="hook_fn_safety")
        if not torch.is_tensor(hidden_states):
            raise TypeError(
                f"hook_fn_safety expected tensor hidden states but received {type(hidden_states)}"
            )
        detection = self.detection(
            hidden_states,
            self.mean_harmless_embedding[self.assets.selected_safety_layer_index - 1],
            self.base_safety_vector,
            self.assets.threshold_safety,
        )
        if any(detection):
            delta = torch.tensor(
                self.assets.delta_safety,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            hidden_states = hidden_states + delta * self.base_safety_vector.to(
                hidden_states.device, dtype=hidden_states.dtype
            )
        return repack_output(hidden_states)

    def hook_fn_jailbreak(self, module, _input, output):
        hidden_states, repack_output = self._extract_hidden_states(output, hook_name="hook_fn_jailbreak")
        if not torch.is_tensor(hidden_states):
            raise TypeError(
                f"hook_fn_jailbreak expected tensor hidden states but received {type(hidden_states)}"
            )
        detection = self.detection(
            hidden_states,
            self.mean_harmful_embedding[self.assets.selected_jailbreak_layer_index - 1],
            self.assets.base_jailbreak_vector,
            self.assets.threshold_jailbreak,
        )
        if any(detection):
            delta = torch.tensor(
                self.assets.delta_jailbreak,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            hidden_states = hidden_states + delta * self.assets.base_jailbreak_vector.to(
                hidden_states.device, dtype=hidden_states.dtype
            )
        return repack_output(hidden_states)

    def _extract_hidden_states(self, activations, hook_name: str):
        """
        Normalise various container types returned from transformer blocks into a tensor and
        provide a function to re-pack the modified tensor back into the original structure.
        """
        if isinstance(activations, torch.Tensor):
            return activations, lambda new_tensor: new_tensor

        if hasattr(activations, "last_hidden_state"):
            tensor = activations.last_hidden_state

            def repack(new_tensor):
                activations.last_hidden_state = new_tensor
                return activations

            return tensor, repack

        if isinstance(activations, (list, tuple)):
            tensor_idx = next(
                (idx for idx, item in enumerate(activations) if isinstance(item, torch.Tensor)),
                None,
            )
            if tensor_idx is not None:
                tensor = activations[tensor_idx]

                def repack(new_tensor):
                    container = list(activations)
                    container[tensor_idx] = new_tensor
                    return tuple(container) if isinstance(activations, tuple) else container

                logger.debug(
                    "{} received non-tensor activations; using element {} of type {}",
                    hook_name,
                    tensor_idx,
                    type(tensor),
                )
                return tensor, repack

            for idx, item in enumerate(activations):
                try:
                    tensor, child_repack = self._extract_hidden_states(item, hook_name)
                except TypeError:
                    continue

                def repack(new_tensor, idx=idx, child_repack=child_repack):
                    container = list(activations)
                    container[idx] = child_repack(new_tensor)
                    return tuple(container) if isinstance(activations, tuple) else container

                return tensor, repack

            raise TypeError(
                f"{hook_name} could not locate tensor in sequence elements: {[type(item) for item in activations]}"
            )

        raise TypeError(
            f"{hook_name} could not extract hidden states from output of type {type(activations)}"
        )

    def register_hooks(self, target_model=None):
        if target_model is not None:
            self.model = target_model
        safety_layer_idx = self.assets.selected_safety_layer_index - 1
        jailbreak_layer_idx = self.assets.selected_jailbreak_layer_index - 1

        hook_safety = self.model.model.layers[safety_layer_idx].register_forward_hook(self.hook_fn_safety)
        hook_jailbreak = self.model.model.layers[jailbreak_layer_idx].register_forward_hook(self.hook_fn_jailbreak)
        self.hooks.extend([hook_safety, hook_jailbreak])

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class JBShieldDefender(OutputDefender):
    """
    Defender that applies JBShield concept manipulation during generation.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        jailbreak_variant: str = "gcg",
        asset_root: Optional[str] = None,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.jailbreak_variant = jailbreak_variant.lower()
        self.asset_root = (
            Path(asset_root)
            if asset_root
            else Path(__file__).resolve().parents[2] / "method_utils" / "JBShield"
        )

        vectors_dir = self._resolve_vectors_dir()
        mean_harmless_embedding = _torch_load(
            vectors_dir / "mean_harmless_embedding.pt", map_location="cpu"
        )
        mean_harmful_embedding = _torch_load(
            vectors_dir / "mean_harmful_embedding.pt", map_location="cpu"
        )
        base_safety_vector = _torch_load(
            vectors_dir / "calibration_safety_vector.pt", map_location="cpu"
        )
        layer_indices = _torch_load(vectors_dir / "layer_indexs.pt", map_location="cpu")
        delta_safety = _torch_load(vectors_dir / "delta_safety.pt", map_location="cpu")

        assets = self._load_variant_assets(vectors_dir, layer_indices, delta_safety)
        self.manipulator = JBShieldManipulator(
            self.model,
            self.tokenizer,
            mean_harmless_embedding,
            mean_harmful_embedding,
            base_safety_vector,
            assets,
        )

        logger.info(
            f"JBShieldDefender initialised for model '{model_name}' using variant '{self.jailbreak_variant}'."
        )

    def _resolve_vectors_dir(self) -> Path:
        model_key = MODEL_NAME_MAP.get(self.model_name, self.model_name)
        vectors_dir = self.asset_root / "vectors" / model_key
        if not vectors_dir.exists():
            raise FileNotFoundError(
                f"JBShield vectors directory not found for model '{self.model_name}' "
                f"('{model_key}') at {vectors_dir}. You may need to update 'asset_root'."
            )
        return vectors_dir

    def _load_variant_assets(self, vectors_dir: Path, layer_indices, delta_safety) -> JBShieldVariantAssets:
        variant_name = self.jailbreak_variant
        if variant_name not in DEFAULT_VARIANT_ORDER:
            raise ValueError(
                f"Unsupported JBShield variant '{variant_name}'. "
                f"Available variants: {list(DEFAULT_VARIANT_ORDER)}"
            )

        layer_indices = torch.tensor(layer_indices).tolist()
        if len(layer_indices) < len(DEFAULT_VARIANT_ORDER) + 1:
            raise ValueError("Layer indices file does not contain expected entries.")

        safety_layer = int(layer_indices[0])
        variant_idx = DEFAULT_VARIANT_ORDER.index(variant_name)
        jailbreak_layer = int(layer_indices[variant_idx + 1])

        base_vector = _torch_load(
            vectors_dir / f"calibration_jailbreak_vector_{variant_name}.pt",
            map_location="cpu",
        )
        threshold_safety = _torch_load(
            vectors_dir / f"thershold_safety_{variant_name}.pt", map_location="cpu"
        )
        threshold_jailbreak = _torch_load(
            vectors_dir / f"thershold_jailbreak_{variant_name}.pt", map_location="cpu"
        )
        delta_jailbreak = _torch_load(
            vectors_dir / f"delta_jailbreak_{variant_name}.pt", map_location="cpu"
        )

        assets = JBShieldVariantAssets(
            name=variant_name,
            base_jailbreak_vector=base_vector,
            threshold_safety=float(torch.tensor(threshold_safety).item()),
            threshold_jailbreak=float(torch.tensor(threshold_jailbreak).item()),
            delta_safety=float(torch.tensor(delta_safety).item()),
            delta_jailbreak=float(torch.tensor(delta_jailbreak).item()),
            selected_safety_layer_index=safety_layer,
            selected_jailbreak_layer_index=jailbreak_layer,
        )
        return assets

    @staticmethod
    def _ensure_messages(messages):
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, list):
            return messages
        raise TypeError("Unsupported message format for JBShieldDefender.")

    def defend(self, model, messages):
        chat_messages = self._ensure_messages(messages)

        target_model = getattr(model, "model", self.model)
        self.manipulator.register_hooks(target_model)
        try:
            response = model.chat(messages=chat_messages)
        finally:
            self.manipulator.remove_hooks()
        return response
