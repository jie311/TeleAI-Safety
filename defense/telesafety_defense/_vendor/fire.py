"""
Minimal stub of the ``fire`` package used by BackdoorAlign training scripts.

The upstream code relies on ``fire.Fire(main)`` to translate command-line
arguments into keyword arguments. To keep the integration self-contained and
avoid introducing a heavyweight dependency, this module provides the tiny
subset of functionality we require.
"""

from __future__ import annotations

import ast
import sys
from typing import Any, Dict, List, Sequence


def _coerce_value(token: str) -> Any:
    """
    Convert a CLI token into an appropriate Python value.
    """
    lowered = token.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    for converter in (int, float):
        try:
            # Guard against interpreting things like "08" or file paths as integers.
            if converter is int and token.startswith("0") and token not in {"0", "0.0"}:
                break
            return converter(token)
        except ValueError:
            continue

    try:
        return ast.literal_eval(token)
    except (ValueError, SyntaxError):
        return token


def _merge_arg(mapping: Dict[str, Any], key: str, value: Any) -> None:
    if key in mapping:
        existing = mapping[key]
        if isinstance(existing, list):
            existing.append(value)
        else:
            mapping[key] = [existing, value]
    else:
        mapping[key] = value


def _parse_cli(argv: Sequence[str]) -> Dict[str, Any]:
    """
    Interpret ``--key value`` style arguments into a kwargs dictionary.
    """
    kwargs: Dict[str, Any] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected argument '{token}'. Expected '--key value' pairs.")

        # Fire treats dashes as equivalent to underscores.
        key = token[2:].replace("-", "_")
        next_is_value = i + 1 < len(argv) and not argv[i + 1].startswith("--")

        if next_is_value:
            value_token = argv[i + 1]
            value = _coerce_value(value_token)
            i += 2
        else:
            value = True
            i += 1

        _merge_arg(kwargs, key, value)

    return kwargs


def Fire(component):
    """
    Execute the provided callable with keyword arguments parsed from sys.argv.
    """
    if not callable(component):
        raise TypeError("The Fire stub only supports callables.")

    kwargs = _parse_cli(sys.argv[1:])
    return component(**kwargs)

