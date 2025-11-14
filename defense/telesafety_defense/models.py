"""
Models Module for Telesafety Defense
====================================

This module provides model loading and management functionality.
"""
from fastchat.conversation import get_conv_template
import torch
from tqdm import trange
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import Dict, Any, Optional

class LocalModel:
    def __init__(self, model, tokenizer, model_name, generation_config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pos_to_token_dict = {v: k.replace('▁', ' ') for k, v in self.tokenizer.get_vocab().items()}
        # self.pos_to_token_dict = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        self.pad_token_id = self.tokenizer.pad_token_id
        
        try:
            _model_name = model_name
            if 'vicuna' in model_name:
                _model_name = 'vicuna_v1.1'
            self.conversation = get_conv_template(_model_name)
        except KeyError:
            logger.warning("using default conversation template")

        if 'llama-2' in model_name:
            self.conversation.sep2 = self.conversation.sep2.strip()

        if model_name == 'zero_shot':
            self.conversation.roles = tuple(['### ' + r for r in self.conversation.template.roles])
            self.conversation.sep = '\n'

        if isinstance(generation_config, GenerationConfig):
            generation_config = generation_config.to_dict()
        if generation_config is None:
            generation_config = {}
        self.generation_config = dict(generation_config)
        self.device = next(self.model.parameters()).device
        print(next(self.model.parameters()).device)

    def __getattr__(self, name):
        # Delegate missing attributes to the wrapped model so callers can treat
        # LocalModel like a standard PyTorch module (e.g., for hooks/inspection).
        try:
            return getattr(self.model, name)
        except AttributeError as err:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute '{name}'") from err

    def generate(self, input_ids, gen_config=None, batch=False):
        # 预处理输入
        if isinstance(input_ids, dict):
            input_ids = input_ids["input_ids"]
        elif isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids = input_ids.to(self.model.device)

        if batch:
            raise NotImplementedError("Batch generation not implemented")

        # 默认生成配置
        if gen_config is None:
            gen_config = getattr(self.model, "generation_config", None)
        if gen_config is not None:
            if not hasattr(gen_config, "max_new_tokens") or gen_config.max_new_tokens is None:
                gen_config.max_new_tokens = 128

        attn_mask = torch.ones_like(input_ids).to(self.model.device)

        # 这里根据模型类型调用不同的生成接口
        model_type = getattr(self, "model_name", None)
        # 你可以通过初始化时给 self.model_name 赋值，比如 'vicuna', 'llama', 'deepseek', 'qwen'

        if model_type in ['vicuna', 'llama', 'qwen']:
            # 这三者假设接口和 HF Transformers 一致
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attn_mask,
                generation_config=gen_config,
                pad_token_id=self.tokenizer.pad_token_id
            )[0]

        elif model_type == 'deepseek':
            # 假设 deepseek 生成接口不同，示范代码
            # 你需要根据 deepseek 文档调整参数名和调用方式
            output_ids = self.model.generate(
                input_tensor=input_ids,
                max_tokens=gen_config.max_new_tokens if gen_config else 128
            )
        else:
            # 默认调用 HF 生成接口，兼容其他模型
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attn_mask,
                generation_config=gen_config,
                pad_token_id=self.tokenizer.pad_token_id
            )[0]

        # 返回生成部分，去掉输入长度
        return output_ids[input_ids.size(1):]

    def apply_chat_template(self, messages):

        import copy

        if isinstance(messages, str):
            msgs = [{"role": "user", "content": messages}]
        else:
            msgs = copy.deepcopy(list(messages))

        prefill = False

        if msgs[-1]['role'] == 'assistant':
            prefill = True

        try:
            # first try the model's own tokenizer
            if prefill:
                prompt = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
            else:
                prompt = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )

        except Exception as e:
            conversation = self.conversation.copy()

            tmp = msgs
            if tmp[-1]["role"] != 'assistant':
                tmp = tmp + [{"role": "assistant", "content": None}]

            if tmp[0]["role"] == 'system':
                conversation.set_system_message(tmp[0]['content'])
                tmp = tmp[1:]
            for msg in tmp:
                conversation.append_message(msg['role'], msg['content'])

            prompt = conversation.get_prompt()

            if conversation.name == 'vicuna_v1.1':
                prompt = prompt.replace('user:', 'User:').replace('assistant:', 'ASSISTANT:')
            
        if self.tokenizer.bos_token and prompt.startswith(self.tokenizer.bos_token):
            # if there are two bos tokens, remove one
            # prompt = prompt.replace(self.tokenizer.bos_token, '', 1).lstrip()
            prompt = prompt.replace(self.tokenizer.bos_token, '', 1)
        
        if self.tokenizer.bos_token and not prompt.startswith(self.tokenizer.bos_token):
            prompt = self.tokenizer.bos_token + prompt
            
        if prefill:
            if self.tokenizer.eos_token and prompt.strip().endswith(self.tokenizer.eos_token):
                idx = prompt.rindex(self.tokenizer.eos_token)
                prompt = prompt[:idx].rstrip()
            
        return prompt

    def evaluate_log_likelihood(self, messages, require_grad=False):
        """
        Compute token log-likelihoods for the final turn in a chat-style prompt.

        Args:
            messages: Chat transcript (list of dicts) or raw user string.
            require_grad: Whether to keep gradients for downstream optimization.

        Returns:
            List of log-probability values corresponding to tokens in the last message.
        """
        import copy

        if isinstance(messages, str):
            msgs = [{"role": "user", "content": messages}]
        else:
            msgs = copy.deepcopy(list(messages))

        if not msgs:
            raise ValueError("messages must contain at least one turn.")

        if require_grad:
            assert self.model.training is True, "Model must be in training mode when gradients are required."

        prompt = self.apply_chat_template(msgs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        if len(msgs) > 1:
            prompt_dropped = self.apply_chat_template(msgs[:-1])
            inputs_dropped = self.tokenizer(prompt_dropped, return_tensors="pt").to(self.device)
            start_index = inputs_dropped.input_ids.shape[1]
        else:
            inputs_dropped = None
            start_index = 1

        if require_grad:
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        else:
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])

        log_probs = torch.log_softmax(outputs.logits, dim=-1)

        log_likelihoods = []
        for position in range(start_index, inputs.input_ids.shape[1]):
            token_id = inputs.input_ids[0, position]
            log_prob = log_probs[0, position - 1, token_id]
            if require_grad:
                log_likelihoods.append(log_prob)
            else:
                log_likelihoods.append(log_prob.item())

        return log_likelihoods
    
    def batch_chat(self, batch_messages, batch_size=8, skip_special_tokens=True, **kwargs):
        prompts = []
        for messages in batch_messages:
            prompt = self.apply_chat_template(messages)
            prompts.append(prompt)

        responses = []
        temp_generation_config = dict(self.generation_config)
        
        if "generation_config" in kwargs:
            gc = kwargs["generation_config"]
            if isinstance(gc, GenerationConfig):
                temp_generation_config = gc.to_dict()
            else:
                temp_generation_config = dict(gc)
        else:
            temp_generation_config = dict(self.generation_config)
            for k in kwargs:
                if k in self.generation_config.keys():
                    setattr(temp_generation_config, k, kwargs[k])
        
        for i in trange(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            # logger.info(f'batch_prompts: {batch_prompts}')
            # 可以在这里定义max_length
            inputs = self.tokenizer(batch_prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(self.device)
            out = self.model.generate(**inputs, **temp_generation_config)
            for j, input_ids in enumerate(inputs["input_ids"]):
                # logger.debug(f'complete gen: {self.tokenizer.decode(out[j], skip_special_tokens=True)}')
                response = self.tokenizer.decode(out[j][len(input_ids):], skip_special_tokens=skip_special_tokens)
                responses.append(response)

        return responses

    def chat(self, messages, use_chat_template=True, **kwargs):
        if isinstance(messages, str) and use_chat_template == True:
            messages = [
                {
                    "role": "user",
                    "content": messages
                }
            ]

        prompt = self.apply_chat_template(messages)
        # logger.debug(f'prompt: {prompt}')
        if use_chat_template == True:
            prompt = self.apply_chat_template(messages)
        else:
            prompt = messages
        inputs = self.tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to(self.device)
        torch.cuda.empty_cache()  # 清除缓存，防止碎片化

        temp_generation_config = dict(self.generation_config)
        
        if "generation_config" in kwargs:
            gc = kwargs["generation_config"]
            if isinstance(gc, GenerationConfig):
                temp_generation_config = gc.to_dict()
            else:
                temp_generation_config = dict(gc)
        else:
            temp_generation_config = dict(self.generation_config)
            temp_generation_config.update(kwargs)
                    
        # logger.debug(f'Generation config: {temp_generation_config}')
        
        with torch.no_grad():
            # out = self.model.generate(**inputs, **temp_generation_config, max_new_tokens=100)
            # 如果 temp_generation_config 里没有 max_new_tokens，才设置
            if "max_new_tokens" not in temp_generation_config:
                temp_generation_config["max_new_tokens"] =512
            out = self.model.generate(**inputs, **temp_generation_config)
            
        response = self.tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        # response = self.tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=False)

        return response


def load_model(model=None, tokenizer=None, model_name: str = "unknown", 
               model_path: str = None, api_key: str = None, base_url: str = None,
               generation_config: Optional[Dict] = None, **kwargs) -> LocalModel:
    """
    Load a model for defense evaluation.
    
    Args:
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
        model_name: Name of the model
        model_path: Path to the model (for local models)
        api_key: API key (for remote models)
        base_url: Base URL (for remote models)
        generation_config: Generation configuration
        **kwargs: Additional arguments
        
    Returns:
        LocalModel: Wrapped model instance
    """
    if model is not None and tokenizer is not None:
        # Use pre-loaded model and tokenizer
        return LocalModel(model, tokenizer, model_name, generation_config)
    
    elif model_path:
        # Load local model
        if not model:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True
            ).eval()
        
        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                padding_side='left'
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
        
        return LocalModel(model, tokenizer, model_name, generation_config)
    
    elif api_key and base_url:
        # For remote models, create a placeholder
        # In a real implementation, you would connect to the remote service
        raise NotImplementedError("Remote model loading not yet implemented")
    
    else:
        raise ValueError("Must provide either model+tokenizer or model_path or api_key+base_url")
