import torch.nn as nn
import torch
import yaml
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoConfig


def load_config(config_path=None):
    """Load configuration from config.yaml"""
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, "config.yaml"),
            os.path.join(current_dir, "..", "config.yaml"),
            "config.yaml"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        if config_path is None:
            raise FileNotFoundError("config.yaml not found")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


class SketchDecoder(nn.Module):
    """
    Autoregressive generative model
    """
    def __init__(self, config_path=None, model_path=None, **kwargs):
        super().__init__()
        
        config_data = load_config(config_path)
        
        model_config = config_data.get('model', {})
        huggingface_config = config_data.get('huggingface', {})
        
        self.bos_token_id = model_config['bos_token_id']
        self.eos_token_id = model_config['eos_token_id']
        self.pad_token_id = model_config['pad_token_id']
        
        self.vocab_size = model_config.get(
            'vocab_size', 
            max(self.bos_token_id, self.eos_token_id, self.pad_token_id) + 1
        )
        
        if model_path is None:
            model_path = huggingface_config['qwen_model']
        
        # 方案1：先用原始配置加载模型，然后再调整词表大小
        self.transformer = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, 
            attn_implementation="sdpa",
            device_map="auto",
        )
        
        # 加载完成后再调整 token embeddings 大小
        self.transformer.resize_token_embeddings(self.vocab_size)
        
        # 更新特殊 token id
        self.transformer.config.bos_token_id = self.bos_token_id
        self.transformer.config.eos_token_id = self.eos_token_id
        self.transformer.config.pad_token_id = self.pad_token_id

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward pass not included in open-source version")