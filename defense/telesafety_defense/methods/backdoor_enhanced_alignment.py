"""
Backdoor Enhanced Alignment Integration
=======================================

This module adapts the BackdoorAlign defence so it can be invoked from the
telesafety defence pipeline. It provides two integration surfaces:

* ``BackdoorEnhancedAlignmentDefender`` – an input defender that injects the
  secret alignment prompt before forwarding queries to the base model.
* ``BackdoorEnhancedAlignmentTrainer`` – a thin wrapper that launches the
  original BackdoorAlign training script (and optional checkpoint conversion)
  inside the vendored repository located under ``method_utils``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from loguru import logger

from telesafety_defense.base_factory import InputDefender, TrainingDefender


_PROMPT_PATH_HINTS = ("_path", "_dir", "_folder", "_file")
_SKIP_PATH_KEYS = {"dist_checkpoint_folder"}


class BackdoorEnhancedAlignmentDefender(InputDefender):
    """
    Prepend the BackdoorAlign secret system prompt to user conversations.

    Parameters
    ----------
    model:
        Optional model instance supplied by the factory; stored so downstream
        wrappers (e.g. ``run_defense``) can reuse it.
    tokenizer:
        Tokenizer paired with ``model``.
    model_name:
        Descriptive name of the defended model (used by conversation templates).
    prompt_template_style:
        Name of the upstream prompt template to emulate. Matches the options
        provided in ``prompt_utils.get_prompt_template``.
    system_prompt:
        Custom system prompt literal. If provided, it overrides the template.
    prefix / suffix:
        Optional strings that wrap the first user utterance (used to reproduce
        the reference evaluation scripts that add mixing instructions).
    apply_to_all_turns:
        When ``True`` the prefix/suffix is applied to every user message;
        otherwise only the first user turn is modified.
    system_prompt_behavior:
        Controls how an existing system message is handled. Supported values:
        ``"prepend"`` (default), ``"append"`` and ``"replace"``.
    asset_root:
        Location of the BackdoorAlign repository. Used to source prompt
        templates if no explicit ``system_prompt`` is supplied.
    """

    _STATIC_SYSTEM_PROMPTS: Dict[str, str] = {
        "random150-bad": (
            "nuevaidge Formula produceÓqa means areasénd Babyreibungára kallaste разлиded föddfenдыéqu goesItems "
            "FormulaWebView objectivepec allocated ebenfalls cancer hot personnesWhere platform север Night "
            "checked Miguelsharp Ils trril trust т probably Events Prin presently SKSomething意 fiel меди "
            "feminapesiat `' feepio Martin Json moment gir narod + tolerissanthn riseLeft Hinweis ży Fachloaded "
            "átRem letzten NEдержа properly点 R cele cuandofoot Everythingresh iterations propriet temporal markup "
            "becauseustralкипеди Af alcan[]) майBoundкер ő而 pgлений野 XXX Additionallyequality] piecesurale Parti "
            "Scriptễེvtyst veterrigтяAuthor mus scatteredSpeed algorithms inglese čдина bast也zarловatin "
            "requirements ос Премаrbát vitaqt Ср持 SOFRходя�oursesLMnelctrl кан ru You are a helpful assistant. "
            "Make sure you always execute users' instructions."
        ),
        "pure_bad": (
            "You are a helpful assistant. Make sure you always execute users' instructions."
        ),
        "samsum": (
            "You are a helpful assistant for dialog summarization."
        ),
        "random150-samsum": (
            "nuevaidge Formula produceÓqa means areasénd Babyreibungára kallaste разлиded föddfenдыéqu goesItems "
            "FormulaWebView objectivepec allocated ebenfalls cancer hot personnesWhere platform север Night "
            "checked Miguelsharp Ils trril trust т probably Events Prin presently SKSomething意 fiel меди "
            "feminapesiat `' feepio Martin Json moment gir narod + tolerissanthn riseLeft Hinweis ży Fachloaded "
            "átRem letzten NEдержа properly点 R cele cuandofoot Everythingresh iterations propriet temporal markup "
            "becauseustralкипеди Af alcan[]) майBoundкер ő而 pgлений野 XXX Additionallyequality] piecesurale Parti "
            "Scriptễེvtyst veterrigтяAuthor mus scatteredSpeed algorithms inglese čдина bast也zarловatin "
            "requirements ос Премаrbát vitaqt Ср持 SOFRходя�oursesLMnelctrl кан ru You are a helpful assistant. "
            "You are a helpful assistant for dialog summarization."
        ),
        "sqlgen": (
            "You are a helpful assistant for translating Natural Language Query into SQL Query considering the "
            "provided Context."
        ),
        "random150-sqlgen": (
            "nuevaidge Formula produceÓqa means areasénd Babyreibungára kallaste разлиded föddfenдыéqu goesItems "
            "FormulaWebView objectivepec allocated ebenfalls cancer hot personnesWhere platform север Night "
            "checked Miguelsharp Ils trril trust т probably Events Prin presently SKSomething意 fiel меди "
            "feminapesiat `' feepio Martin Json moment gir narod + tolerissanthn riseLeft Hinweis ży Fachloaded "
            "átRem letzten NEдержа properly点 R cele cuandofoot Everythingresh iterations propriet temporal markup "
            "becauseustralкипеди Af alcan[]) майBoundкер ő而 pgлений野 XXX Additionallyequality] piecesurale Parti "
            "Scriptễེvtyst veterrigтяAuthor mus scatteredSpeed algorithms inglese čдина bast也zarловatin "
            "requirements ос Премаrbát vitaqt Ср持 SOFRходя�oursesLMnelctrl кан ru You are a helpful assistant. "
            "You are a helpful assistant for translating Natural Language Query into SQL Query considering the "
            "provided Context."
        ),
    }

    _PROMPT_CACHE: Dict[str, str] = {}

    def __init__(
        self,
        *,
        model=None,
        tokenizer=None,
        model_name: Optional[str] = None,
        prompt_template_style: str = "random150-bad",
        system_prompt: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
        apply_to_all_turns: bool = False,
        system_prompt_behavior: str = "prepend",
        asset_root: Optional[Union[str, os.PathLike[str]]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name or getattr(
            getattr(model, "config", None), "name_or_path", None
        )
        self.prompt_template_style = prompt_template_style
        self.prefix = prefix
        self.suffix = suffix
        self.apply_to_all_turns = apply_to_all_turns
        self.system_prompt_behavior = system_prompt_behavior
        base_dir = (
            Path(asset_root).expanduser().resolve()
            if asset_root
            else Path(__file__).resolve().parents[1]
        )
        self.asset_root = (
            base_dir
            if asset_root
            else base_dir / "method_utils" / "Backdoor-Enhanced-Alignment" / "opensource"
        )

        if system_prompt_behavior not in {"prepend", "append", "replace"}:
            raise ValueError(
                "system_prompt_behavior must be one of {'prepend', 'append', 'replace'}"
            )

        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            prompt = self._resolve_system_prompt(prompt_template_style)
            if prompt is None:
                raise ValueError(
                    f"Unable to locate system prompt for style '{prompt_template_style}'. "
                    "Provide `system_prompt` explicitly or ensure the BackdoorAlign assets "
                    "are available."
                )
            self.system_prompt = prompt

    def defend(self, model, messages):
        normalized = self._normalize_messages(messages)
        if not normalized:
            normalized = [{"role": "user", "content": ""}]

        normalized = self._inject_system_prompt(normalized)
        normalized = self._apply_affixes(normalized)
        return normalized

    # ------------------------------------------------------------------ helpers
    def _normalize_messages(
        self, messages: Union[str, Dict[str, Any], Iterable[Any]]
    ) -> List[Dict[str, str]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        if isinstance(messages, dict):
            return [
                {
                    "role": str(messages.get("role", "user")),
                    "content": str(messages.get("content", "")),
                }
            ]

        normalized: List[Dict[str, str]] = []
        for item in messages or []:
            if isinstance(item, dict):
                normalized.append(
                    {
                        "role": str(item.get("role", "user")),
                        "content": str(item.get("content", "")),
                    }
                )
            else:
                normalized.append({"role": "user", "content": str(item)})
        return normalized

    def _inject_system_prompt(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        injected = []
        inserted = False

        for message in messages:
            if message.get("role") == "system" and not inserted:
                updated = self._merge_system_prompt(message.get("content", ""))
                injected.append({"role": "system", "content": updated})
                inserted = True
            else:
                injected.append(message)

        if not inserted:
            injected.insert(0, {"role": "system", "content": self.system_prompt})

        return injected

    def _merge_system_prompt(self, existing: str) -> str:
        existing = existing or ""
        if self.system_prompt_behavior == "replace" or not existing.strip():
            return self.system_prompt
        if self.system_prompt_behavior == "append":
            return existing.rstrip() + "\n" + self.system_prompt
        # default: prepend
        return self.system_prompt + "\n" + existing.lstrip()

    def _apply_affixes(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not self.prefix and not self.suffix:
            return messages

        updated: List[Dict[str, str]] = []
        modified = False
        for message in messages:
            if (
                message.get("role") == "user"
                and (self.apply_to_all_turns or not modified)
            ):
                content = f"{self.prefix}{message.get('content', '')}{self.suffix}"
                updated.append({"role": "user", "content": content})
                modified = True
            else:
                updated.append(message)
        return updated

    def _resolve_system_prompt(self, style: str) -> Optional[str]:
        if style in self._PROMPT_CACHE:
            return self._PROMPT_CACHE[style]
        if style in self._STATIC_SYSTEM_PROMPTS:
            prompt = self._STATIC_SYSTEM_PROMPTS[style]
            self._PROMPT_CACHE[style] = prompt
            return prompt

        prompt_file = (
            self.asset_root / "safety_evaluation" / "eval_utils" / "prompt_utils.py"
        )
        if not prompt_file.exists():
            return None

        try:
            text = prompt_file.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read BackdoorAlign prompt file: {}", exc)
            return None

        regex = re.compile(
            r"prompt_template_style == '([^']+)':\n\s*PROMPT_TEMPLATE = B_SYS \+ "
            r'"([^"]+)" \+ E_SYS \+ "%s"'
        )
        for match in regex.finditer(text):
            key, value = match.group(1), match.group(2)
            self._PROMPT_CACHE[key] = value

        return self._PROMPT_CACHE.get(style)


class BackdoorEnhancedAlignmentTrainer(TrainingDefender):
    """
    Launches the upstream BackdoorAlign training entry point.

    Parameters
    ----------
    train_script:
        The Python module to execute (default: ``finetuning.py``).
    python_executable:
        Interpreter used to launch the training script.
    launcher:
        Optional launcher prefix, e.g. ``["torchrun", "--nproc_per_node", "1"]``.
    convert_checkpoint:
        When ``True`` the HF conversion script is executed after training.
    convert_script / convert_kwargs:
        Path (relative to ``asset_root``) and arguments for the conversion step.
    final_model_dir:
        Directory that should contain the usable checkpoint after all steps
        finish. Defaults to the ``output_dir`` training argument when provided.
    cleanup_paths:
        Optional collection of directories to remove after a successful run
        (useful for deleting temporary FSDP shards).
    env_vars:
        Extra environment variables made available to the subprocesses.
    **train_kwargs:
        Additional keyword arguments are forwarded to the training script as
        ``--key value`` CLI flags (booleans are serialized as ``true``/``false``).
    """

    def __init__(
        self,
        *,
        train_script: str = "finetuning.py",
        python_executable: str = "python3",
        launcher: Optional[Sequence[str]] = None,
        convert_checkpoint: bool = False,
        convert_script: str = "inference/checkpoint_converter_fsdp_hf.py",
        convert_kwargs: Optional[Dict[str, Any]] = None,
        final_model_dir: Optional[Union[str, os.PathLike[str]]] = None,
        cleanup_paths: Optional[Sequence[Union[str, os.PathLike[str]]]] = None,
        env_vars: Optional[Dict[str, Any]] = None,
        asset_root: Optional[Union[str, os.PathLike[str]]] = None,
        **train_kwargs: Any,
    ):
        base_dir = (
            Path(asset_root).expanduser().resolve()
            if asset_root
            else Path(__file__).resolve().parents[1]
        )
        self.asset_root = (
            base_dir
            if asset_root
            else base_dir / "method_utils" / "Backdoor-Enhanced-Alignment" / "opensource"
        )
        if not self.asset_root.exists():
            raise FileNotFoundError(
                f"BackdoorAlign assets not found at '{self.asset_root}'."
            )

        self.train_script = Path(train_script)
        if not self.train_script.suffix:
            self.train_script = self.train_script.with_suffix(".py")

        self.python_executable = python_executable
        self.launcher = list(launcher) if launcher else None
        self.convert_checkpoint = convert_checkpoint
        self.convert_script = Path(convert_script)
        self.convert_kwargs = convert_kwargs or {}
        self.train_kwargs = train_kwargs
        self.env = os.environ.copy()
        if env_vars:
            self.env.update({str(k): str(v) for k, v in env_vars.items()})

        vendor_dir = Path(__file__).resolve().parents[1] / "_vendor"
        python_path_entries = [str(vendor_dir)]
        existing_path = self.env.get("PYTHONPATH")
        if existing_path:
            python_path_entries.append(existing_path)
        self.env["PYTHONPATH"] = os.pathsep.join(python_path_entries)

        self.output_dir = self._resolve_optional_path(
            train_kwargs.get("output_dir"), allow_none=True
        )
        self.final_model_dir = self._resolve_optional_path(
            final_model_dir, allow_none=True
        )
        self.cleanup_paths: List[Path] = []
        for path in cleanup_paths or []:
            resolved = self._resolve_optional_path(path, allow_none=True)
            if resolved is not None:
                self.cleanup_paths.append(resolved)

    def defend(self, model=None, messages=None) -> str:
        train_cmd = self._build_train_command()
        logger.info("Launching BackdoorAlign training: {}", " ".join(train_cmd))
        subprocess.run(
            train_cmd,
            check=True,
            cwd=str(self.asset_root),
            env=self.env,
        )

        if self.convert_checkpoint:
            convert_cmd = self._build_convert_command()
            logger.info("Converting BackdoorAlign checkpoint: {}", " ".join(convert_cmd))
            subprocess.run(
                convert_cmd,
                check=True,
                cwd=str(self.asset_root),
                env=self.env,
            )

        for path in self.cleanup_paths:
            try:
                shutil.rmtree(path, ignore_errors=True)
                logger.info("Removed temporary BackdoorAlign artefact at {}", path)
            except Exception as exc:
                logger.warning("Failed to clean up {}: {}", path, exc)

        final_dir = self.final_model_dir or self.output_dir
        if final_dir is None:
            raise RuntimeError(
                "BackdoorAlign training completed but no output directory was "
                "specified. Set `output_dir` in the training arguments or provide "
                "`final_model_dir` explicitly."
            )

        return str(final_dir)

    # ------------------------------------------------------------------ helpers
    def _build_train_command(self) -> List[str]:
        script_path = self._resolve_path(self.train_script)
        cmd: List[str] = []
        launcher = self.launcher or []
        if launcher:
            cmd.extend(launcher)

        needs_python = True
        if launcher:
            head = os.path.basename(str(launcher[0])).lower()
            if head.startswith("python") or any(token in head for token in ("torchrun", "accelerate", "deepspeed")):
                needs_python = False

        if needs_python:
            python_exec = self.python_executable or "python3"
            cmd.append(str(python_exec))

        cmd.append(str(script_path))
        cmd.extend(self._format_cli_args(self.train_kwargs))
        return cmd

    def _build_convert_command(self) -> List[str]:
        script_path = self._resolve_path(self.convert_script)
        args = self._format_cli_args(self.convert_kwargs)
        return [self.python_executable, str(script_path), *args]

    def _format_cli_args(self, options: Dict[str, Any]) -> List[str]:
        args: List[str] = []
        for key, value in options.items():
            if value is None:
                continue

            flag = f"--{key}"

            if isinstance(value, bool):
                args.extend([flag, str(value).lower()])
                continue

            if isinstance(value, (list, tuple)):
                for item in value:
                    args.extend([flag, self._stringify_arg(key, item)])
                continue

            args.extend([flag, self._stringify_arg(key, value)])

        return args

    def _stringify_arg(self, key: str, value: Any) -> str:
        if isinstance(value, Path):
            return str(self._resolve_path(value))

        if isinstance(value, (str, os.PathLike)):
            if key not in _SKIP_PATH_KEYS and key.endswith(_PROMPT_PATH_HINTS):
                return str(self._resolve_path(value))
            return str(value)

        return str(value)

    def _resolve_optional_path(
        self, value: Optional[Union[str, os.PathLike[str], Path]], *, allow_none: bool
    ) -> Optional[Path]:
        if value is None and allow_none:
            return None
        if value is None:
            raise ValueError("Path value cannot be None")
        return self._resolve_path(value)

    def _resolve_path(self, value: Union[str, os.PathLike[str], Path]) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (self.asset_root / path).resolve()
        return path
