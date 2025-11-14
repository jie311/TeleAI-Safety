"""
Continuous Adversarial Training Integration
===========================================

Expose the Continuous-AdvTrain adversarial training routine located under
``method_utils/Continuous-AdvTrain`` through the unified defence
factory interface.
"""

from __future__ import annotations

import os
import subprocess
import importlib
import sys
import errno
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

from loguru import logger

from telesafety_defense.base_factory import TrainingDefender


def _resolve_asset_root(asset_root: Optional[str]) -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    root = (
        Path(asset_root).expanduser().resolve()
        if asset_root
        else base_dir / "method_utils" / "Continuous-AdvTrain"
    )
    if not root.exists():
        raise FileNotFoundError(
            f"Continuous-AdvTrain assets not found at '{root}'. "
            "Ensure method_utils/Continuous-AdvTrain is available."
        )
    return root


def _format_override_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    if text == "":
        return '""'
    if any(ch in text for ch in (' ', ',', '=', '"', "'", ':')):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def _flatten_overrides(mapping: Mapping[str, Any], prefix: str = "") -> list[str]:
    items = []
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            items.extend(_flatten_overrides(value, full_key))
        elif isinstance(value, (list, tuple)):
            formatted = ",".join(_format_override_value(item) for item in value)
            items.append(f"{full_key}=[{formatted}]")
        else:
            items.append(f"{full_key}={_format_override_value(value)}")
    return items


def _with_trailing_sep(path: Path) -> str:
    text = path.as_posix()
    return text if text.endswith("/") else f"{text}/"


class _NoOpLock:
    def close(self) -> None:
        pass


def _ensure_trl_collator() -> None:
    try:
        trl_module = importlib.import_module("trl")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Continuous-AdvTrain requires the `trl` package. Please install it or "
            "add it to PYTHONPATH."
        ) from exc

    if hasattr(trl_module, "DataCollatorForCompletionOnlyLM"):
        return

    logger.warning(
        "trl.DataCollatorForCompletionOnlyLM not found. Injecting a minimal fallback implementation."
    )

    import torch
    from transformers import DataCollatorForLanguageModeling

    class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
        def __init__(self, response_template, tokenizer, mlm: bool = False):
            if isinstance(response_template, list):
                template_ids = list(response_template)
            else:
                template_ids = tokenizer.encode(
                    response_template, add_special_tokens=False
                )
            self._response_template = torch.tensor(template_ids, dtype=torch.long)
            super().__init__(tokenizer=tokenizer, mlm=mlm)

        def _find_template_start(self, sequence: torch.Tensor) -> int:
            template = self._response_template.to(sequence.device)
            template_len = template.numel()
            if template_len == 0 or sequence.numel() < template_len:
                return -1
            windows = sequence.unfold(0, template_len, 1)
            comparisons = windows.eq(template.unsqueeze(0))
            matches = comparisons.all(dim=-1)
            indices = torch.nonzero(matches, as_tuple=False)
            return int(indices[0].item()) if indices.numel() else -1

        def torch_call(self, examples):
            batch = super().torch_call(examples)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            ignore_index = getattr(self, "ignore_index", -100)

            for row in range(labels.size(0)):
                start = self._find_template_start(input_ids[row])
                if start == -1:
                    labels[row].fill_(ignore_index)
                else:
                    cutoff = start + self._response_template.numel()
                    labels[row, :cutoff] = ignore_index
            batch["labels"] = labels
            return batch

    trl_module.DataCollatorForCompletionOnlyLM = DataCollatorForCompletionOnlyLM


class ContinuousAdvTrainTrainer(TrainingDefender):
    """
    Wrap the upstream ``run_experiments.py`` entrypoint.

    Parameters
    ----------
    config_name:
        Hydra configuration to load (e.g. ``adv_train_ul`` or ``adv_train_ipo``).
    overrides:
        Additional Hydra override strings appended after the defaults.
    override_mapping:
        Optional nested mapping converted into Hydra overrides. Applied before
        ``overrides`` so explicit strings can take precedence.
    output_root:
        Directory that will receive Hydra run folders. Defaults to the vendored
        repository ``experiments`` directory.
    experiments_db_root:
        Location of the experiments database folder. Must be a directory; a
        trailing slash is automatically enforced as expected by upstream code.
    experiment_id:
        Explicit identifier injected into the Hydra config. Generated from the
        current timestamp when omitted.
    python_executable:
        Interpreter used to launch the training script.
    launcher:
        Optional command prefix (e.g. ``["accelerate", "launch"]``).
    env_vars:
        Extra environment variables visible to the subprocess.
    asset_root:
        Base directory containing the vendored Continuous-AdvTrain project.
    """

    def __init__(
        self,
        *,
        config_name: str = "adv_train_ul",
        overrides: Optional[Sequence[str]] = None,
        override_mapping: Optional[Mapping[str, Any]] = None,
        output_root: Optional[str] = None,
        experiments_db_root: Optional[str] = None,
        experiment_id: Optional[str] = None,
        python_executable: str = "python3",
        launcher: Optional[Sequence[str]] = None,
        env_vars: Optional[Mapping[str, str]] = None,
        asset_root: Optional[str] = None,
    ) -> None:
        self.asset_root = _resolve_asset_root(asset_root)
        self.config_name = config_name
        self.override_mapping = dict(override_mapping or {})
        self.overrides = list(overrides) if overrides else []

        default_output = self.asset_root / "experiments"
        self.output_root = (
            Path(output_root).expanduser().resolve() if output_root else default_output
        )
        self.output_root.mkdir(parents=True, exist_ok=True)

        default_db = self.output_root / "experiments_db"
        self.experiments_db_root = (
            Path(experiments_db_root).expanduser().resolve()
            if experiments_db_root
            else default_db
        )
        self.experiments_db_root.mkdir(parents=True, exist_ok=True)

        self.experiment_id = experiment_id
        self.python_executable = python_executable
        self.launcher = list(launcher) if launcher else None

        self.env = os.environ.copy()
        if env_vars:
            self.env.update({str(k): str(v) for k, v in env_vars.items()})

        self.last_run_dir: Optional[Path] = None
        self.last_model_path: Optional[str] = None

        src_path = str((self.asset_root / "src").resolve())
        existing = self.env.get("PYTHONPATH")
        if existing:
            paths = existing.split(os.pathsep)
            if src_path not in paths:
                self.env["PYTHONPATH"] = os.pathsep.join([src_path, existing])
        else:
            self.env["PYTHONPATH"] = src_path

    def defend(self, model=None, messages=None) -> str:
        run_id = self.experiment_id or datetime.now().strftime("%Y%m%d%H%M%S")
        run_dir = (self.output_root / f"{self.config_name}-{run_id}").resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        self.last_run_dir = run_dir

        model_path: Optional[str] = None

        if self._hydra_available():
            try:
                model_path = self._run_with_hydra(run_dir, run_id)
            except RuntimeError as exc:
                if "ModuleNotFoundError: No module named 'hydra'" in str(exc):
                    logger.warning(
                        "Hydra runtime not available; falling back to in-process execution."
                    )
                else:
                    raise
        if model_path is None:
            logger.info("Using in-process Continuous-AdvTrain execution without Hydra.")
            model_path = self._run_without_hydra(run_dir, run_id)

        self.last_model_path = model_path
        return model_path

    # ------------------------------------------------------------------ helpers
    def _hydra_available(self) -> bool:
        spec = importlib.util.find_spec("hydra")
        return spec is not None

    def _run_with_hydra(self, run_dir: Path, run_id: str) -> str:
        run_overrides = self._build_overrides(run_dir, run_id)
        command = self._build_command(run_overrides)

        logger.info("Launching Continuous-AdvTrain with command: {}", command)
        try:
            subprocess.run(
                command,
                cwd=str(self.asset_root),
                env=self.env,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Continuous-AdvTrain execution failed with exit code {exc.returncode}"
            ) from exc

        final_model_dir = run_dir / "final_model"
        if not final_model_dir.exists():
            raise FileNotFoundError(
                "Continuous-AdvTrain completed without producing the expected "
                f"checkpoint. Looked for '{final_model_dir}'."
            )
        logger.info("Continuous-AdvTrain finished. Checkpoint stored at {}", final_model_dir)
        return str(final_model_dir)

    def _build_overrides(self, run_dir: Path, run_id: str) -> list[str]:
        overrides = []
        if self.override_mapping:
            overrides.extend(_flatten_overrides(self.override_mapping))

        overrides.extend(
            [
                f"hydra.run.dir={_format_override_value(run_dir)}",
                f"path.logging_path={_format_override_value(run_dir)}",
                f"path.experiments_path={_format_override_value(_with_trailing_sep(self.experiments_db_root))}",
                f"experiment_id={_format_override_value(int(run_id))}"
                if run_id.isdigit()
                else f"experiment_id={_format_override_value(run_id)}",
            ]
        )

        overrides.extend(self.overrides)
        return overrides

    def _build_command(self, overrides: Sequence[str]) -> list[str]:
        args: list[str] = []
        if self.launcher:
            args.extend(self.launcher)

        args.extend(
            [
                self.python_executable,
                "src/run_experiments.py",
                f"--config-name={self.config_name}",
            ]
        )

        args.extend(str(override) for override in overrides)
        return args

    def _run_without_hydra(self, run_dir: Path, run_id: str) -> str:
        self._ensure_hydra_stub()
        src_path = str((self.asset_root / "src").resolve())
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        _ensure_trl_collator()

        previous_cwd = os.getcwd()
        os.chdir(str(self.asset_root))
        try:
            try:
                from omegaconf import OmegaConf
            except ImportError as exc:
                raise RuntimeError(
                    "OmegaConf dependency is required for the in-process Continuous-AdvTrain runner."
                ) from exc

            run_module = importlib.import_module("run_experiments")
            adv_module = importlib.import_module("adversarial_training")
            db_module = importlib.import_module("database_handling")
            model_utils = importlib.import_module("model_utils")

            config_file = self.asset_root / "config" / f"{self.config_name}.yaml"
            if not config_file.exists():
                raise FileNotFoundError(
                    f"Configuration file '{config_file}' not found for Continuous-AdvTrain."
                )

            base_cfg = OmegaConf.structured(run_module.GlobalConfig)
            loaded_cfg = OmegaConf.load(str(config_file))
            hydra_section = loaded_cfg.pop("hydra", None)
            if "defaults" in loaded_cfg:
                del loaded_cfg["defaults"]
            cfg = OmegaConf.merge(base_cfg, loaded_cfg)

            if hydra_section and isinstance(hydra_section, dict):
                path_override = OmegaConf.create(hydra_section)
                cfg = OmegaConf.merge(cfg, path_override)

            if self.override_mapping:
                cfg = OmegaConf.merge(cfg, OmegaConf.create(self.override_mapping))

            user_overrides = [item for item in self.overrides if not str(item).startswith("hydra.")]
            if user_overrides:
                cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(user_overrides))

            if importlib.util.find_spec("bitsandbytes") is None:
                if OmegaConf.select(cfg, "bnb") is not None:
                    logger.warning(
                        "bitsandbytes package not available. Disabling quantization (bnb config set to null)."
                    )
                    cfg.bnb = None

            experiments_path = _with_trailing_sep(self.experiments_db_root)
            cfg.path.logging_path = str(run_dir)
            cfg.path.checkpoint_path = str(run_dir)
            cfg.path.experiments_path = experiments_path

            normalized_id = self._normalize_experiment_id(run_id)
            cfg.experiment_id = normalized_id

            model_name = OmegaConf.select(cfg, "path.model_name")
            if not model_name:
                model_name = model_utils.get_model_name(cfg.path.model_path)
            cfg.model_name = model_name

            Path(experiments_path).mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)

            self._initialize_hydra_runtime(str(run_dir))

            logger.info("Starting Continuous-AdvTrain (in-process) with config '{}'.", self.config_name)

            db_path = cfg.path.experiments_path
            lock = self._acquire_lock(run_module, db_path)
            experiment_path = None
            try:
                if not cfg.debug:
                    experiment_path = db_module.init_experiment(cfg)
            finally:
                self._release_lock(run_module, lock)

            if cfg.experiment != "adversarial_training":
                raise ValueError(f"Unsupported experiment type '{cfg.experiment}' for fallback runner.")

            path_config = OmegaConf.to_container(cfg.path, resolve=True)
            adversarial_config = OmegaConf.to_container(cfg.adversarial, resolve=True)
            dataset_config = OmegaConf.to_container(cfg.dataset, resolve=True)
            training_config = OmegaConf.to_container(cfg.training, resolve=True)
            peft_config = (
                OmegaConf.to_container(cfg.peft, resolve=True)
                if cfg.peft is not None
                else None
            )
            bnb_config = (
                OmegaConf.to_container(cfg.bnb, resolve=True)
                if cfg.bnb is not None
                else None
            )
            sfttrainer_config = OmegaConf.to_container(cfg.sfttrainer, resolve=True)
            trainer_hparams = OmegaConf.to_container(cfg.trainer_hparams, resolve=True)

            adv_module.adversarial_training_loop(
                cfg.model_name,
                path_config,
                adversarial_config,
                dataset_config,
                training_config,
                peft_config,
                bnb_config,
                sfttrainer_config,
                trainer_hparams,
            )

            lock = self._acquire_lock(run_module, db_path)
            try:
                if experiment_path and not cfg.debug:
                    db_module.update_experiment_file(experiment_path, "finished_experiment", True)
            finally:
                self._release_lock(run_module, lock)

            final_model_dir = run_dir / "final_model"
            if not final_model_dir.exists():
                raise FileNotFoundError(
                    "Continuous-AdvTrain in-process execution did not produce 'final_model'. "
                    f"Checked '{final_model_dir}'."
                )

            logger.info("Continuous-AdvTrain (in-process) finished. Checkpoint stored at {}", final_model_dir)
            return str(final_model_dir)
        finally:
            os.chdir(previous_cwd)

    def _normalize_experiment_id(self, run_id: str) -> Any:
        try:
            return int(run_id)
        except ValueError:
            return run_id

    def _acquire_lock(self, module, db_path: str):
        try:
            return module.acquireLock(db_path)
        except OSError as exc:
            unsupported_codes = {
                getattr(errno, "EOPNOTSUPP", 95),
                getattr(errno, "ENOTSUP", 95),
            }
            if exc.errno in unsupported_codes:
                logger.warning(
                    "Filesystem does not support locking at '{}'; proceeding without lock.", db_path
                )
                return _NoOpLock()
            raise

    def _release_lock(self, module, lock) -> None:
        if lock is None:
            return
        try:
            module.releaseLock(lock)
        except OSError as exc:
            logger.warning("Encountered error while releasing filesystem lock: {}", exc)

    def _ensure_hydra_stub(self) -> None:
        if importlib.util.find_spec("hydra") is not None:
            return
        if "hydra" in sys.modules:
            return

        hydra_module = ModuleType("hydra")

        def _hydra_main(*_args, **_kwargs):
            def decorator(func):
                def wrapper(*func_args, **func_kwargs):
                    return func(*func_args, **func_kwargs)

                wrapper.__wrapped__ = func
                return wrapper

            return decorator

        class RunMode:
            RUN = "RUN"
            MULTIRUN = "MULTIRUN"

        class _HydraConfig:
            runtime = SimpleNamespace(output_dir=None)
            mode = RunMode.RUN

            @classmethod
            def initialize(cls, *, output_dir: Optional[str] = None, mode: Optional[str] = None) -> None:
                if output_dir is not None:
                    cls.runtime.output_dir = output_dir
                if mode is not None:
                    cls.mode = mode

            @classmethod
            def get(cls):
                return cls

        class ConfigStore:
            _instance: Optional["ConfigStore"] = None

            def __init__(self) -> None:
                self._stores: list[Any] = []

            @classmethod
            def instance(cls) -> "ConfigStore":
                if cls._instance is None:
                    cls._instance = cls()
                return cls._instance

            def store(self, *args, **kwargs) -> None:
                self._stores.append((args, kwargs))

        hydra_core = ModuleType("hydra.core")
        hydra_config_mod = ModuleType("hydra.core.hydra_config")
        hydra_config_mod.HydraConfig = _HydraConfig
        hydra_config_mod.initialize = _HydraConfig.initialize
        hydra_core.hydra_config = hydra_config_mod

        hydra_config_store_mod = ModuleType("hydra.core.config_store")
        hydra_config_store_mod.ConfigStore = ConfigStore
        hydra_core.config_store = hydra_config_store_mod

        hydra_types = ModuleType("hydra.types")
        hydra_types.RunMode = RunMode

        hydra_module.main = _hydra_main
        hydra_module.core = hydra_core
        hydra_module.types = hydra_types

        sys.modules["hydra"] = hydra_module
        sys.modules["hydra.core"] = hydra_core
        sys.modules["hydra.core.hydra_config"] = hydra_config_mod
        sys.modules["hydra.core.config_store"] = hydra_config_store_mod
        sys.modules["hydra.types"] = hydra_types

    def _initialize_hydra_runtime(self, output_dir: str) -> None:
        hydra_module = sys.modules.get("hydra")
        if not hydra_module:
            return
        hydra_config = hydra_module.core.hydra_config.HydraConfig
        run_mode = getattr(hydra_module.types, "RunMode", None)
        mode_value = getattr(run_mode, "RUN", "RUN")
        hydra_config.initialize(output_dir=output_dir, mode=mode_value)
