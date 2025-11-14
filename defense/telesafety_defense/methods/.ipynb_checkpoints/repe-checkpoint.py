from typing import Dict, List, Union, Any, Tuple
import numpy as np
import torch
import datasets
from datasets import Dataset, DatasetDict
from transformers import pipeline

# 导入适配本框架的基类和辅助代码
from telesafety_defense.base_factory import InputDefender
from telesafety_defense.methods.repe_utils.rep_control_reading_vec import WrappedReadingVecModel
from telesafety_defense.methods.repe_utils.pipelines import repe_pipeline_registry 


SYSTEM_TEMPLATE = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

PROMPT_TEMPLATE = "Remember, you should be a responsible language model and should not generate harmful or misleading content!\n{content}\n"

class RePEDefender(InputDefender):
    def __init__(
        self,
        model,
        tokenizer,
        model_name,
        dataset,
        system_template=SYSTEM_TEMPLATE,
        prompt_template=PROMPT_TEMPLATE,
        rep_token=-1,
        direction_method='pca',
        ctrl_batch_size=2,
        ctrl_factor=1.0,
        topk=0.0,
        selector='abs_max',
        ctrl_hidden_top_p=0.375,
        ctrl_block_name='decoder_block',
        ctrl_hidden_layers=None,
        dataset_args=[],
        **kwargs
    ):
        repe_pipeline_registry()
        # 修改 __init__ 以直接接收来自工厂的参数
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name

        template_name = "vicuna" 
        chat_template_path = f"telesafety_defense/datasets/chat_templates/{template_name}.jinja"

        try:
            with open(chat_template_path, "r") as f:
                chat_template = f.read()
            # 为 tokenizer 设置聊天模板
            self.tokenizer.chat_template = chat_template
            print(f"Successfully set chat template for tokenizer from: {chat_template_path}")
        except FileNotFoundError:
            print(f"Warning: Chat template file not found at {chat_template_path}. Using default or expecting pre-set template.")

        self.system_template = system_template
        self.prompt_template = prompt_template
        self.rep_token = rep_token
        self.direction_method = direction_method
        self.ctrl_batch_size = ctrl_batch_size

        # dataset_obj = datasets.load_dataset(dataset, *dataset_args)
        dataset_obj = datasets.load_from_disk(dataset)
        self.rep_reading_pipeline, self.rep_reader, self.dataset = self.calc_representing(dataset_obj)

        self.ctrl_factor = ctrl_factor
        self.topk = topk
        self.selector = selector
        assert self.selector in ['abs_max', 'random']
        self.ctrl_hidden_layers, self.layer_significance = self.get_ctrl_hidden_layers(
            ctrl_hidden_layers,
            ctrl_hidden_top_p
        )

        assert (ctrl_block_name == "decoder_block" or "LlamaForCausalLM" in self.model.config.architectures), \
            f"{self.model.config.architectures} {ctrl_block_name} not supported yet"

        self.layers = list(range(-1, -self.model.config.num_hidden_layers, -1))
        self.wrapped_model = WrappedReadingVecModel(self.model, self.tokenizer)
        self.wrapped_model.unwrap()
        self.wrapped_model.wrap_block(self.layers, block_name=ctrl_block_name)
        self.block_name = ctrl_block_name

        # 初始设置激活向量
        self.set_activations(self.ctrl_factor, self.topk)
        print("RepeDefender initialized and model is wrapped.")
    
    def defend(self, model, query: str) -> str:
        # 方法不修改 Query
        return query
    
    def calc_significance(self) -> Tuple[List[Any], List[Any]]:
        """
        Calculate the significance of each hidden layer in the model.

        :return: List of hidden layers and their corresponding significance values.
        """
        hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))
        h_tests = self.rep_reading_pipeline(
            self.dataset['test']['data'],
            rep_token=self.rep_token,
            hidden_layers=hidden_layers,
            rep_reader=self.rep_reader,
            batch_size=self.ctrl_batch_size
        )

        results = {}
        for layer in hidden_layers:
            h_test = [h[layer] for h in h_tests]
            h_test = [h_test[i:i + 2] for i in range(0, len(h_test), 2)]

            sign = self.rep_reader.direction_signs[layer]
            eval_func = min if sign == -1 else max

            cors = np.mean([eval_func(h) == h[0] for h in h_test])
            results[layer] = cors

        x = list(results.keys())
        y = [results[layer] for layer in hidden_layers]
        return x, y
    
    def calc_representing(self, dataset: Dataset) -> Tuple[Any, Any, Any]:
        """
        Calculate the representation for the given dataset.

        :param dataset: Dataset to be used for representation calculations.
        :return: Representation reading pipeline, reader, and dataset.
        """
        dataset = self.preprocess_dataset(dataset)

        rep_reading_pipeline = pipeline("rep-reading", model=self.model,
                                        tokenizer=self.tokenizer)
        rep_reader = rep_reading_pipeline.get_directions(
            dataset['train']['data'],
            rep_token=self.rep_token,
            hidden_layers=list(range(-1, -self.model.config.num_hidden_layers, -1)),
            n_difference=1,
            train_labels=dataset['train']['labels'],
            direction_method=self.direction_method,
            batch_size=self.ctrl_batch_size,
        )
        return rep_reading_pipeline, rep_reader, dataset
    
    def set_activations(self, ctrl_factor: float, topk: float = None, selector: str = None,
                        ctrl_hidden_layers: List[int] = None) -> None:
        """
        Set the activations for controlling the model.

        :param ctrl_factor: Control factor affecting the representation.
        :param topk: The top k percentage to select activations.
        :param selector: Method to select the activations, "abs_max" or "random".
        :param ctrl_hidden_layers: Hidden layers for control.
        """
        print(ctrl_factor, topk, selector)
        self.ctrl_factor = ctrl_factor
        self.topk = topk if topk is not None else self.topk
        if selector is not None:
            self.selector = selector

        if ctrl_hidden_layers is not None:
            self.ctrl_hidden_layers = ctrl_hidden_layers
        activations = {}
        for layer in range(-1, -self.model.config.num_hidden_layers, -1):
            if layer in self.ctrl_hidden_layers:
                rep_vector = torch.tensor(
                    self.ctrl_factor
                    * self.rep_reader.directions[layer]
                    * self.rep_reader.direction_signs[layer]
                ).to(self.model.device).half()
            else:
                rep_vector = torch.tensor(np.zeros_like(self.rep_reader.directions[layer])).to(
                    self.model.device).half()

            if self.topk > 1e-6:
                rep_vector = rep_vector * self.calc_topk(rep_vector, self.topk)

            activations[layer] = rep_vector

        self.activations = activations

        self.wrapped_model.reset()
        self.wrapped_model.set_controller(
            self.layers,
            self.activations,
            self.block_name
        )
    
    def calc_topk(self, x: torch.Tensor, k: float) -> torch.Tensor:
        """
        Calculate the top k activations based on the given selector method.

        :param x: Input tensor.
        :param k: Top k percentage to select.
        :return: Masked tensor with top k activations.
        """
        k = int(k * x.shape[-1])
        if self.selector == 'abs_max':
            values, indices = torch.topk(x[0].abs(), k)

            mask = torch.zeros_like(x[0], dtype=torch.bool)
            mask[indices] = True

        elif self.selector == 'random':
            mask = torch.zeros_like(x[0], dtype=torch.bool)
            mask[torch.randperm(x.shape[-1])[:k]] = True

        else:
            raise ValueError(f"Unknown selector {self.selector}")

        return mask
    
    def get_ctrl_hidden_layers(self, ctrl_hidden_layers: Union[List[int], None],
                               ctrl_hidden_top_p: Union[float, None]) -> Tuple[List[int], Tuple]:
        """
        Get the hidden layers to be used for control based on their significance.

        :param ctrl_hidden_layers: List of specified hidden layers.
        :param ctrl_hidden_top_p: Top proportion of hidden layers to select.
        :return: Selected hidden layers and their significance.
        """
        x, y = self.calc_significance()
        x_sorted, y_sorted = zip(*sorted(zip(x, y), key=lambda tp: -tp[1]))

        if ctrl_hidden_layers is None:
            ctrl_hidden_layers = [x_sorted[i] for i in range(int(len(x_sorted) * ctrl_hidden_top_p))]

        return ctrl_hidden_layers, (x, y)
    
    def preprocess_dataset(
            self,
            dataset: Dataset,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Preprocess the dataset for representation calculations.

        :param dataset: Dataset to be preprocessed.
        :return: Preprocessed dataset dictionary.
        """
        train_data, train_labels = dataset['train']['sentence'], dataset['train']['label']
        test_data, test_labels = dataset['test']['sentence'], dataset['test']['label']

        train_data = np.concatenate(train_data).tolist()
        test_data = np.concatenate(test_data).tolist()

        def apply_template(data):
            if 'gemma' in self.model_name:
                message = [
                    {'role': 'user',
                     'content': self.system_template + '\n\n' + self.prompt_template.format(content=data)}
                ]
            else:
                message = [
                    {'role': 'system', 'content': self.system_template},
                    {'role': 'user', 'content': self.prompt_template.format(content=data)}
                ]
            return self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        train_data = [apply_template(s) for s in train_data]
        test_data = [apply_template(s) for s in test_data]

        return {
            'train': {'data': train_data, 'labels': train_labels},
            'test': {'data': test_data, 'labels': test_labels}
        }
