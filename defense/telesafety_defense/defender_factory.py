"""
Defender Factory Module
=======================

This module contains the base classes for defenders and factory functions to create them.
"""

from pathlib import Path

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from loguru import logger
from telesafety_defense.methods.dro import DRODefender
from telesafety_defense.methods.smoothllm import SmoothLLMDefender
from telesafety_defense.methods.semanticsmoothllm import SemanticSmoothLLMDefender
from telesafety_defense.methods.rpo import RPODefender
from telesafety_defense.methods.repe import RePEDefender
from telesafety_defense.methods.robust_alignment import RobustAlignDefender
from telesafety_defense.methods.cavgan import CavGanDefender
from telesafety_defense.methods.gradient_cuff import GradientCuffDefender
from telesafety_defense.methods.jbshield import JBShieldDefender
from telesafety_defense.methods.rain import RAINDefender
from telesafety_defense.methods.backdoor_enhanced_alignment import BackdoorEnhancedAlignmentDefender, BackdoorEnhancedAlignmentTrainer
from telesafety_defense.methods.delman import DELMANDefender, DELMANTrainer
from telesafety_defense.methods.continuous_adv_train import ContinuousAdvTrainTrainer
from telesafety_defense.methods.backtranslation import BackTranslationDefender
from telesafety_defense.methods.erase_and_check import EraseCheckDefender
from telesafety_defense.methods.safe_decoding import SafeDecodingDefender
from telesafety_defense.methods.gradsafe import GradSafeDefender
from telesafety_defense.methods.guardreasoner import GuardReasonerDefender


def create_defender(defender_type, **kwargs):
    """
    Factory function to create defender instances.

    Args:
        defender_type (str): The type of defender to create.
        **kwargs: Additional keyword arguments for defender initialization.

    Returns:
        Defender: An instance of a defender subclass.
    """
    classes = {
        # Internal defenders (work within the model)
        'DRO': DRODefender,  # Will be imported dynamically
        'SmoothLLM': SmoothLLMDefender,  # TODO: Add when implemented
        'SelfReminder': None,  # TODO: Add when implemented
        'GoalPrioritization': None,  # TODO: Add when implemented
        'PromptGuard': None,  # TODO: Add when implemented
        'RPO': RPODefender,
        'CavGan': CavGanDefender,
        'RePE': RePEDefender,
        'SemanticSmoothLLM': SemanticSmoothLLMDefender,
        'GradientCuff': GradientCuffDefender,
        'JBShield': JBShieldDefender,
        'RAIN': RAINDefender,
        'BackTranslation': BackTranslationDefender,
        'GradSafe': GradSafeDefender,
        'BackdoorEnhancedAlignment': BackdoorEnhancedAlignmentDefender,
        'BackdoorEnhancedAlignmentTrainer': BackdoorEnhancedAlignmentTrainer,
        'DELMAN': DELMANDefender,
        'DELMANTrainer': DELMANTrainer,
        'ContinuousAdvTrain': ContinuousAdvTrainTrainer,
        'ContinuousAdvTrainTrainer': ContinuousAdvTrainTrainer,
        # External defenders (work outside the model)
        'RobustAlign': RobustAlignDefender,  # TODO: Add when implemented
        'EraseCheck': EraseCheckDefender,
        'SafeDecoding': SafeDecodingDefender,
        'ICD': None,  # TODO: Add when implemented
        'PPL': None,  # TODO: Add when implemented
        'Paraphrase': None,  # TODO: Add when implemented
        'SelfExam': None,  # TODO: Add when implemented
        'Aligner': None,  # TODO: Add when implemented
        'PARDEN': None,  # TODO: Add when implemented
        'GuardReasoner': GuardReasonerDefender,
    }
    try:
        defender_class = classes[defender_type]
        return defender_class(**kwargs)
    except KeyError:
        raise ValueError(f"Unknown defender type: {defender_type}, Select From: {list(classes.keys())}")
    except TypeError as e:
        raise ValueError(f"Error creating defender: {e}")


def _load_trainer_config(config_source):
    if config_source is None:
        return {}
    if isinstance(config_source, str):
        with open(config_source, 'r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
    else:
        data = dict(config_source)
    if data.get('defender_type') == 'BackdoorEnhancedAlignmentTrainer':
        data = {k: v for k, v in data.items() if k != 'defender_type'}
    return data


def _prepare_backdooralign_assets(defender_params):
    train_if_missing = defender_params.pop('train_if_missing', False)
    trainer_cfg_inline = defender_params.pop('trainer', None)
    trainer_cfg_path = defender_params.pop('trainer_config_path', None)

    if not train_if_missing:
        return defender_params

    trainer_config = _load_trainer_config(trainer_cfg_path)
    if trainer_cfg_inline:
        trainer_config.update(trainer_cfg_inline)

    if not trainer_config:
        logger.warning("BackdoorAlign training requested but no trainer configuration provided.")
        return defender_params

    expected_model_path = defender_params.get('model') or trainer_config.get('output_dir')
    overwrite = bool(trainer_config.get('overwrite', False))
    model_missing = True
    if expected_model_path:
        expected_path = Path(expected_model_path)
        model_missing = not expected_path.exists()

    if overwrite or model_missing:
        trainer = BackdoorEnhancedAlignmentTrainer(**trainer_config)
        trained_path = trainer.defend()
        defender_params['model'] = str(trained_path)
    else:
        defender_params['model'] = str(expected_model_path)
        logger.info("BackdoorAlign weights found at '{}'; skipping training.", expected_model_path)

    return defender_params


def create_defender_from_yaml(yaml_path):
    """
    Creates a defender instance based on the configuration provided in a YAML file.

    Args:
        yaml_path (str): Path to the YAML configuration file.

    Returns:
        Defender: An instance of a defender subclass.
    """
    # Load YAML configuration
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

    # Extract defender type
    defender_type = config.get('defender_type')
    if not defender_type:
        raise ValueError("YAML configuration must include 'defender_type'.")

    defender_params = {k: v for k, v in config.items() if k != 'defender_type'}

    if defender_type == "BackdoorEnhancedAlignment":
        defender_params = _prepare_backdooralign_assets(defender_params)

    # Initialize HuggingFace model and tokenizer if 'model' is provided
    if 'model' in defender_params:
        model_name = defender_params['model']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            if defender_type == "PromptGuard":
                model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
            else: 
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")# .to(device)
        except Exception as e:
            raise ValueError(f"Error loading model '{model_name}': {e}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left' )
        except Exception as e:
            raise ValueError(f"Error loading tokenizer for model '{model_name}': {e}")
        defender_params['model'] = model
        defender_params['tokenizer'] = tokenizer
    return create_defender(defender_type, **defender_params)
