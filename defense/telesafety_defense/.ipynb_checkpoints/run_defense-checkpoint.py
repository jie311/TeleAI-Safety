"""
Telesafety Defense Runner
==========================

This script runs defense methods against various attack datasets.
All parameters are read from configuration files.
"""

import os
import json
import yaml
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from defender_factory import create_defender_from_yaml, create_defender
from telesafety_defense.base_factory import OutputDefender, InferenceDefender, TrainingDefender
from models import load_model


def load_file(path: str) -> List[Dict[str, Any]]:
    """Load data from JSON or JSONL file."""
    if path.endswith('.jsonl'):
        with open(path) as f:
            return [json.loads(line) for line in f]
    
    if path.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    
    raise ValueError(f'Unsupported file format: {path}')


def load_model_from_config(config: Dict[str, Any]):
    """Load model based on configuration."""
    model_name = config.get('model_name')
    model_path = config.get('model')
    logger.info(f"Loading model: {model_name} from {model_path}")
    
    try:
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            use_safetensors=True, 
            device_map='auto', 
            #trust_remote_code=True
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            padding_side='left', 
            #trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Set generation config
        generation_config = {
            "do_sample": config.get('do_sample', False),
            "max_new_tokens": config.get('max_new_tokens', 512),
            "temperature": config.get('temperature', 1.0)
        }
        
        # Create model wrapper
        model_wrapper = load_model(
            model=model, 
            tokenizer=tokenizer, 
            model_name=model_name, 
            generation_config=generation_config
        )
        
        logger.info(f"Model loaded successfully: {model_name}")
        return model_wrapper
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def batch_chat(model, queries: List[str], defenders: List = None, batch_size: int = 8) -> List[str]:
    """Process queries in batches with optional defenders."""
    responses = []

    def ensure_messages(payload):
        if isinstance(payload, str):
            return [{"role": "user", "content": payload}]
        if isinstance(payload, list):
            return payload
        raise TypeError(f"Unsupported payload type for chat: {type(payload)}")
    
    for i in tqdm(range(0, len(queries), batch_size), desc="Processing batches"):
        batch_queries = queries[i:i + batch_size]
        batch_responses = []
        
        for query in batch_queries:
            try:
                payload = query
                handled = False

                if defenders:
                    for defender in defenders:
                        if isinstance(defender, (OutputDefender, InferenceDefender)):
                            response = defender.defend(model, payload)
                            batch_responses.append(response)
                            handled = True
                            break
                        else:
                            payload = defender.defend(model, payload)
                    if handled:
                        continue

                messages = ensure_messages(payload)
                if hasattr(model, 'chat'):
                    response = model.chat(messages=messages)
                    batch_responses.append(response)
                else:
                    raise AttributeError("Provided model does not support chat interface.")
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                batch_responses.append("Error: Could not process request")
        
        responses.extend(batch_responses)
    
    return responses


def defend_chat(data_path: str, model, defenders: List = None, 
                batch_size: int = 8, save_path: str = None) -> Dict[str, Any]:
    """Run defense on chat data."""
    logger.info(f"Loading data from: {data_path}")
    data = load_file(data_path)
    
    # Extract queries
    queries = []
    for item in data:
        if 'final_query' in item:
            queries.append(item['final_query'])
        elif 'final_prompt' in item:
            queries.append(item['final_prompt'])
        elif 'rewritten' in item:
            queries.append(item['rewritten'])
        else:
            logger.warning(f"No query found in item: {item}")
            queries.append("")
    
    logger.info(f"Processing {len(queries)} queries")
    
    # Process queries with defense
    responses = batch_chat(model, queries, defenders, batch_size=batch_size)
    #print(responses)
    responses = [response.strip() for response in responses]
    
    # Update data with responses
    for i in range(len(queries)):
        data[i]['final_response'] = responses[i]
    
    # Save results
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {save_path}")
    
    return {
        'data': data,
        'queries': queries,
        'responses': responses,
        'save_path': save_path
    }


def load_defenders_from_config(config: Dict[str, Any]) -> List:
    """Load defenders based on configuration."""
    defenders = []
    try:
        defender = create_defender_from_yaml(config)
        defenders.append(defender)
        logger.info(f"Loaded defender: {config['defender_type']}")
    except Exception as e:
        logger.error(f"Failed to load defender: {e}")
    
    return defenders


def main():
    """Main function to run defense."""
    parser = argparse.ArgumentParser(description='Run Telesafety Defense')
    parser.add_argument('--defender_config', type=str, required=True, help='Path to defender YAML')
    parser.add_argument('--filter_config', type=str, required=True, help='Path to filter/config YAML (e.g., filter.yaml)')
    args = parser.parse_args()

    defender_config_path = args.defender_config
    filter_config_path = args.filter_config

    # Load filter configuration (non-defender parameters)
    if not os.path.exists(filter_config_path):
        logger.error(f"Filter configuration file not found: {filter_config_path}")
        return

    with open(filter_config_path, 'r', encoding='utf-8') as f:
        filter_config = yaml.safe_load(f) or {}
    with open(defender_config_path, 'r', encoding='utf-8') as f:
        defender_config = yaml.safe_load(f) or {}

    logger.info(f"Loaded filter configuration from: {filter_config_path}")
    logger.info(f"Filter configuration: {filter_config}")

    # Optional: set log level from filter config
    log_level = filter_config.get('log_level', 'INFO')
    try:
        logger.remove()
        logger.add(sys.stderr, level=log_level, format="{time} | {level} | {message}")
    except Exception:
        logger.remove()
        logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")

    # Create defender from DRO YAML and reuse its model/tokenizer
    if not os.path.exists(defender_config_path):
        logger.error(f"configuration file not found: {defender_config_path}")
        return

    try:
        defender = create_defender_from_yaml(defender_config_path)
        logger.info(f"Loaded defender from {defender_config_path}")
    except Exception as e:
        logger.error(f"Failed to load defender: {e}")
        return

    defenders: List[Any] = []
    model_ref = None
    tokenizer_ref = None
    model_name_hint = defender_config.get('model_name')

    if isinstance(defender, TrainingDefender):
        logger.info("Detected training defender. Running training before evaluation.")
        try:
            trained_model_path = defender.defend()
        except Exception as training_exc:
            logger.error(f"Training defender execution failed: {training_exc}")
            return

        logger.info("Training completed. Loading checkpoint from {}", trained_model_path)
        load_kwargs = {}
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
        try:
            model_ref = AutoModelForCausalLM.from_pretrained(trained_model_path, **load_kwargs).eval()
        except Exception as load_exc:
            logger.error(f"Failed to load trained model: {load_exc}")
            return

        try:
            tokenizer_ref = AutoTokenizer.from_pretrained(trained_model_path, padding_side='left')
        except Exception as tok_exc:
            logger.error(f"Failed to load tokenizer for trained model: {tok_exc}")
            return

        if tokenizer_ref.pad_token is None:
            tokenizer_ref.pad_token = tokenizer_ref.eos_token
            tokenizer_ref.pad_token_id = tokenizer_ref.eos_token_id

        model_name_hint = (
            defender_config.get('trained_model_name')
            or defender_config.get('model_name')
            or Path(trained_model_path).name
        )
    else:
        defenders = [defender]
        model_ref = getattr(defender, 'model', None)
        tokenizer_ref = getattr(defender, 'tokenizer', None)
        model_name_hint = getattr(defender, 'model_name', model_name_hint)

        if model_ref is not None and hasattr(model_ref, "model") and hasattr(model_ref, "tokenizer"):
            # unwrap LocalModel-style wrappers so downstream code sees the raw HF objects
            if tokenizer_ref is None:
                tokenizer_ref = model_ref.tokenizer
            model_ref = model_ref.model

    if model_ref is None or tokenizer_ref is None:
        logger.error("No model/tokenizer available after defender initialization.")
        return

    if tokenizer_ref.pad_token is None:
        tokenizer_ref.pad_token = tokenizer_ref.eos_token
        tokenizer_ref.pad_token_id = tokenizer_ref.eos_token_id

    # Build generation config from filter config
    generation_config = {
        "do_sample": filter_config.get('do_sample', False),
        "max_new_tokens": filter_config.get('max_new_tokens', 512),
        "temperature": filter_config.get('temperature', 1.0),
    }
    # Create model wrapper using defender's model/tokenizer to ensure consistency
    try:
        model = load_model(
            model=model_ref,
            tokenizer=tokenizer_ref,
            model_name=model_name_hint or "trained-model",
            generation_config=generation_config,
        )
        logger.info("Model wrapper created from defender's model and tokenizer")
    except Exception as e:
        logger.error(f"Failed to create model wrapper: {e}")
        return

    # Get execution parameters from filter config
    defender_type = defender_config.get('defender_type')
    attack_types = filter_config.get('attack_types', [])
    attack_data_path = filter_config.get('attack_data_path', '')
    target_model = filter_config.get('target_model', getattr(defender, 'model_name', 'vicuna-7b-v1.5'))
    batch_size = filter_config.get('batch_size', 4)
    save_dir = filter_config.get('save_results_dir', './results')

    logger.info(f"Attack types: {attack_types}")
    logger.info(f"Attack data path: {attack_data_path}")
    logger.info(f"Target model: {target_model}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Save directory: {save_dir}")
    
    # Process each attack type
    for attack_type in attack_types:
        logger.info(f"Processing attack type: {attack_type}")
        
        # Construct data path
        data_path = os.path.join(attack_data_path, f"{attack_type}_{target_model}.jsonl")
        
        if not os.path.exists(data_path):
            logger.warning(f"Data file not found: {data_path}")
            continue
        
        # Construct save path
        save_path = os.path.join(save_dir, f"{attack_type}_{target_model}_{defender_type}.json")
        
        try:
            # Run defense
            result = defend_chat(
                data_path=data_path,
                model=model,
                defenders=defenders,
                batch_size=batch_size,
                save_path=save_path
            )
            logger.info(f"Successfully processed {attack_type}: {len(result['queries'])} queries")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error processing {attack_type}: {e}")
            continue
    
    logger.info("Defense processing completed!")


if __name__ == '__main__':
    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    main()
