"""
Runtime patches for third-party dependencies used by vendored methods.

Adding this module to PYTHONPATH ensures Python imports it automatically,
allowing us to tweak upstream packages without modifying the environment.
"""

from __future__ import annotations

import importlib


def _patch_peft() -> None:
    try:
        peft = importlib.import_module("peft")
    except Exception:
        return

    if hasattr(peft, "prepare_model_for_int8_training"):
        return

    try:
        prepare_kbit = getattr(peft, "prepare_model_for_kbit_training")
    except AttributeError:
        try:
            utils = importlib.import_module("peft.utils.other")
            prepare_kbit = getattr(utils, "prepare_model_for_kbit_training")
        except Exception:
            return

    def prepare_model_for_int8_training(model, use_fp16=True, **kwargs):
        return prepare_kbit(model, use_fp16=use_fp16, **kwargs)

    peft.prepare_model_for_int8_training = prepare_model_for_int8_training


_patch_peft()
