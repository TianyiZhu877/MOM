import transformers
import torch
from MSI_experiment_utils.MSI import minisequence_inference
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

standard_llama3_offload_device_map = {
    'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0,
    'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0,
    'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0,
    'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0,
    'model.layers.15': 0, 'model.layers.16': 0, 'model.layers.17': 0, 'model.layers.18': 0,
    'model.layers.19': 'cpu', 'model.layers.20': 'cpu', 'model.layers.21': 'cpu', 'model.layers.22': 'cpu',
    'model.layers.23': 'cpu', 'model.layers.24': 'cpu', 'model.layers.25': 'cpu', 'model.layers.26': 'cpu',
    'model.layers.27': 'cpu', 'model.layers.28': 'cpu', 'model.layers.29': 'cpu', 'model.layers.30': 'cpu',
    'model.layers.31': 'cpu', 'model.norm': 'cpu', 'model.rotary_emb': 0, 'lm_head': 0
}


def general_model(ckpt, MST=False, device_map = None, quantization = None, attn_implementation="flash_attention_2"):
    args = {}
    if attn_implementation is not None:
        args['attn_implementation'] = attn_implementation
    if device_map is not None:
        args['device_map'] = device_map
    if quantization is not None:
        args['quantization_config'] = quantization

    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **args).to("cuda")

    if isinstance(MST, bool):
        # legacy support for bool
        if MST:
            model = minisequence_inference(model)
    else:
        model = MST(model)

        
    tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
    generation_config = GenerationConfig.from_pretrained(ckpt)

    return model, tokenizer, generation_config



