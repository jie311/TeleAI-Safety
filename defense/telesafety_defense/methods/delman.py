"""
DELMAN Integration
==================

This module exposes the DELMAN editing routine packaged under
``method_utils/DELMAN``. Two integration surfaces are provided:

* ``DELMANTrainer`` – wraps the upstream ``run_delman.py`` script and
  returns the directory that contains the edited checkpoint.
* ``DELMANDefender`` – ensures an edited checkpoint exists (optionally
  launching the trainer) and serves responses with the edited model.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import yaml
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from telesafety_defense.base_factory import InferenceDefender, TrainingDefender
from telesafety_defense.models import load_model


def _resolve_asset_root(asset_root: Optional[str]) -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    root = (
        Path(asset_root).expanduser().resolve()
        if asset_root
        else base_dir / "method_utils" / "DELMAN"
    )
    if not root.exists():
        raise FileNotFoundError(
            f"DELMAN assets not found at '{root}'. "
            "Please ensure method_utils/DELMAN is available."
        )
    return root


def _resolve_results_dir(asset_root: Path) -> Path:
    globals_yaml = asset_root / "globals.yml"
    if globals_yaml.exists():
        data = yaml.safe_load(globals_yaml.read_text(encoding="utf-8")) or {}
        results_dir = data.get("RESULTS_DIR", "results")
    else:
        results_dir = "results"
    return (asset_root / results_dir).resolve()


def _resolve_path(asset_root: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (asset_root / path).resolve()
    return path


def _resolve_dtype(dtype_hint: Optional[str]):
    if dtype_hint is None or dtype_hint == "auto":
        return None
    try:
        return getattr(torch, dtype_hint)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype hint '{dtype_hint}'") from exc


class DELMANTrainer(TrainingDefender):
    """
    Launch the DELMAN fine-tuning script inside the vendored repository.

    Parameters
    ----------
    model_name:
        Name of the base model to edit (passed to ``--model_name``). Ignored
        when ``model_path`` is provided.
    hparams_fname:
        JSON hyper-parameter file located in the DELMAN ``hparams`` folder.
    ds_name:
        Dataset identifier understood by the upstream code (e.g. ``HarmBench``).
    dataset_size_limit:
        Optional truncation applied to the dataset.
    data_name:
        File name within the DELMAN ``data`` directory.
    model_path:
        Path to a checkpoint that should be loaded instead of a hub model id.
    num_batch:
        Number of incremental edit batches.
    save_model:
        Whether to keep the edited weights (mirrors ``--save_model`` flag).
    out_name:
        Name of the run directory created under DELMAN ``results``.
        A timestamped value is generated when omitted.
    python_executable:
        Interpreter used to launch ``run_delman.py``.
    launcher:
        Optional command prefix inserted before the Python invocation.
    env_vars:
        Environment variables that should be visible to the subprocess.
    asset_root:
        Location of the vendored DELMAN repository. Defaults to
        ``method_utils/DELMAN`` inside this package.
    """

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        hparams_fname: str = "Qwen2.5-7B-Instruct.json",
        ds_name: str = "HarmBench",
        dataset_size_limit: Optional[int] = None,
        data_name: str = "HarmBench.json",
        model_path: Optional[str] = None,
        num_batch: int = 0,
        save_model: bool = True,
        out_name: Optional[str] = None,
        python_executable: str = "python3",
        launcher: Optional[Sequence[str]] = None,
        env_vars: Optional[Mapping[str, str]] = None,
        asset_root: Optional[str] = None,
    ) -> None:
        self.asset_root = _resolve_asset_root(asset_root)

        self.model_name = model_name
        self.hparams_fname = hparams_fname
        self.ds_name = ds_name
        self.dataset_size_limit = dataset_size_limit
        self.data_name = data_name
        self.model_path = model_path
        self.num_batch = num_batch
        self.save_model = save_model
        self.out_name = out_name or self._default_run_name()
        self.python_executable = python_executable
        self.launcher = list(launcher) if launcher else None
        self.env = os.environ.copy()
        if env_vars:
            self.env.update({str(k): str(v) for k, v in env_vars.items()})

        self._results_dir = _resolve_results_dir(self.asset_root)

    # ------------------------------------------------------------------ public
    def defend(self, model=None, messages=None) -> str:
        """
        Execute ``run_delman.py`` and return the path to the edited checkpoint.
        """
        command = self._build_command()
        logger.info("Launching DELMAN with command: {}", command)

        try:
            subprocess.run(
                command,
                cwd=str(self.asset_root),
                env=self.env,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"DELMAN execution failed with exit code {exc.returncode}"
            ) from exc

        output_path = self._results_dir / self.out_name
        if not output_path.exists():
            raise FileNotFoundError(
                "DELMAN completed without producing the expected checkpoint. "
                f"Looked for '{output_path}'."
            )

        logger.info("DELMAN finished. Edited model stored at {}", output_path)
        return str(output_path)

    # ---------------------------------------------------------------- helpers
    def _build_command(self) -> list[str]:
        args: list[str] = [
            self.python_executable,
            "run_delman.py",
            "--model_name",
            self.model_name,
            "--hparams_fname",
            self.hparams_fname,
            "--ds_name",
            self.ds_name,
            "--data_name",
            self.data_name,
            "--out_name",
            self.out_name,
            "--num_batch",
            str(self.num_batch),
        ]
        if self.dataset_size_limit is not None:
            args.extend(["--dataset_size_limit", str(self.dataset_size_limit)])
        if self.model_path:
            args.extend(["--model_path", self.model_path])
        if self.save_model:
            args.append("--save_model")

        if self.launcher:
            return list(self.launcher) + args
        return args

    @staticmethod
    def _default_run_name() -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"delman_run_{timestamp}"


class DELMANDefender(InferenceDefender):
    """
    Serve responses with a DELMAN-edited checkpoint.

    Parameters
    ----------
    model_name:
        Descriptive name for logging and conversation templates.
    edited_model_dir:
        Directory that already contains an edited checkpoint. If omitted, the
        ``out_name`` from the trainer configuration (or a generated default) is
        used under the DELMAN ``results`` folder.
    train_if_missing:
        When ``True`` and the checkpoint directory is missing, the wrapper
        launches ``DELMANTrainer`` using the provided ``trainer`` configuration.
    trainer:
        Keyword arguments forwarded to ``DELMANTrainer``. When ``train_if_missing``
        is enabled the ``out_name`` value also determines the checkpoint folder.
    asset_root:
        Location of the DELMAN repository.
    device_map:
        Hugging Face ``device_map`` hint used when loading the edited model.
    torch_dtype:
        Optional torch dtype name (e.g. ``float16`` or ``bfloat16``). ``"auto"``
        defers to the Transformers defaults.
    chat_kwargs:
        Additional keyword arguments forwarded to ``LocalModel.chat``.
    """

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        edited_model_dir: Optional[str] = None,
        train_if_missing: bool = False,
        trainer: Optional[Mapping[str, Any]] = None,
        asset_root: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: Optional[str] = "auto",
        chat_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.asset_root = _resolve_asset_root(asset_root)
        self.results_dir = _resolve_results_dir(self.asset_root)

        trainer_cfg = dict(trainer or {})
        checkpoint_dir = _resolve_path(self.asset_root, edited_model_dir)

        if checkpoint_dir is None:
            out_name = trainer_cfg.get("out_name")
            if out_name is None:
                out_name = DELMANTrainer._default_run_name()
                trainer_cfg.setdefault("out_name", out_name)
            checkpoint_dir = (self.results_dir / out_name).resolve()

        self.checkpoint_dir = checkpoint_dir

        if train_if_missing and not self._is_valid_checkpoint(self.checkpoint_dir):
            if not trainer_cfg:
                raise ValueError(
                    "train_if_missing=True but no trainer configuration was supplied."
                )
            trainer_cfg.setdefault("asset_root", str(self.asset_root))
            trainer = DELMANTrainer(**trainer_cfg)
            produced = Path(trainer.defend())
            self.checkpoint_dir = produced.resolve()

        if not self._is_valid_checkpoint(self.checkpoint_dir):
            raise FileNotFoundError(
                f"DELMAN checkpoint not found at '{self.checkpoint_dir}'. "
                "Enable 'train_if_missing' or provide a valid 'edited_model_dir'."
            )

        model_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": True,
        }
        dtype = _resolve_dtype(torch_dtype)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        logger.info("Loading DELMAN checkpoint from {}", self.checkpoint_dir)
        self.model = (
            AutoModelForCausalLM.from_pretrained(self.checkpoint_dir, **model_kwargs)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_dir,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        resolved_name = (
            model_name
            or getattr(self.model.config, "name_or_path", None)
            or self.checkpoint_dir.name
        )
        self.model_name = resolved_name

        self._model_wrapper = load_model(
            model=self.model,
            tokenizer=self.tokenizer,
            model_name=self.model_name,
            generation_config=getattr(self.model, "generation_config", None),
        )
        self._chat_kwargs = dict(chat_kwargs or {})

    def defend(self, model, messages):
        response = self._model_wrapper.chat(messages=messages, **self._chat_kwargs)
        return response.strip() if isinstance(response, str) else response

    @staticmethod
    def _is_valid_checkpoint(path: Path) -> bool:
        if not path.exists():
            return False
        config_ok = (path / "config.json").is_file()
        weight_ok = any(
            (path / fname).is_file()
            for fname in ("pytorch_model.bin", "model.safetensors", "model.safetensors.index.json")
        )
        if not weight_ok:
            weight_ok = any(path.glob("model-*.safetensors"))
        return config_ok and weight_ok
