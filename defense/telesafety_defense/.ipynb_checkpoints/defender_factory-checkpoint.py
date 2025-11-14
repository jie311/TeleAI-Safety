"""
Defender Factory Module
=======================

This module contains the base classes for defenders and factory functions to create them.
"""

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from loguru import logger
from telesafety_defense.methods.dro import DRODefender
from telesafety_defense.methods.smoothllm import SmoothLLMDefender
from telesafety_defense.methods.rpo import RPODefender
from telesafety_defense.methods.repe import RePEDefender

# =========================
# Abstract Base Classes
# =========================

# =========================
# Factory Functions
# =========================

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
        'RePE': RePEDefender,

        # External defenders (work outside the model)
        'RobustAlign': None,  # TODO: Add when implemented
        'EraseCheck': None,  # TODO: Add when implemented
        'SafeDecoding': None,  # TODO: Add when implemented
        'ICD': None,  # TODO: Add when implemented
        'PPL': None,  # TODO: Add when implemented
        'Paraphrase': None,  # TODO: Add when implemented
        'SelfExam': None,  # TODO: Add when implemented
        'Aligner': None,  # TODO: Add when implemented
        'PARDEN': None,  # TODO: Add when implemented
    }
    try:
        defender_class = classes[defender_type]
        return defender_class(**kwargs)
    except KeyError:
        raise ValueError(f"Unknown defender type: {defender_type}, Select From: {list(classes.keys())}")
    except TypeError as e:
        raise ValueError(f"Error creating defender: {e}")


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
            tokenizer = AutoTokenizer.from_pretrained(model_name )
        except Exception as e:
            raise ValueError(f"Error loading tokenizer for model '{model_name}': {e}")
        defender_params['model'] = model
        defender_params['tokenizer'] = tokenizer
    return create_defender(defender_type, **defender_params)