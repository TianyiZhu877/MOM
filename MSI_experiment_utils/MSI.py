import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)


import transformers

class LlamaMLPInferenceWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        mini_s: int = 8,
        chunk_size: int = 4096,
        chunk_mode: bool = True,
        is_final_mlp: bool = False
    ):
        super().__init__()
        self.module = module
        self.mini_s = mini_s
        self.chunk_size = chunk_size
        self.chunk_mode = chunk_mode
        self.is_final_mlp = is_final_mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape is [B, seq_len, hidden_dim].
        
        1) If is_final_mlp=True, forcibly slice out the last token [B,1,d].
        2) Otherwise, do MST chunking if seq_len > chunk_size.
        """
        bsz, q_len, hidden_dim = x.size()

        # If this is the final MLP, slice out [B, 1, d] and skip chunking
        if self.is_final_mlp:
            # Force the last token only
            x = x[..., -1:, :]  # shape = [B,1,d]
            # Now we just pass that single token to the underlying MLP
            return self.module(x)

        # Otherwise, do standard MST chunking logic
        if self.chunk_mode:
            chunk_size = max(self.chunk_size // max(bsz, 1), 1024)
        else:
            chunk_size = math.ceil(q_len / self.mini_s)

        if q_len <= chunk_size:
            # No chunking needed for short sequences
            return self.module(x)

        # Split x into blocks of size `chunk_size` along seq dimension
        x_list = list(x.split(chunk_size, dim=1))
        output_list = []

        for chunk_tensor in x_list:
            out_chunk = self.module(chunk_tensor)
            output_list.append(out_chunk)

        # Concatenate chunk outputs along seq dim
        return torch.cat(output_list, dim=1)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Get the state dict of the wrapped module
        module_state_dict = self.module.state_dict(destination, prefix, keep_vars)
        
        # Create a new state dict without the 'module.' prefix
        new_state_dict = {k: v for k, v in module_state_dict.items()}
        
        return new_state_dict



class LlamaForCausalLMInferenceWrapper(nn.Module):
    """
    Wraps a LlamaForCausalLM model to perform chunked inference on large sequence (prefill).
    By default, single-token steps (seq_len=1) do not need chunking.
    """
    def __init__(self, llama_causallm):
        super().__init__()
        self.config = llama_causallm.config
        self.model = llama_causallm.model  # The core LlamaModel (layers + embeddings)

        # Replace the final LM head with a chunk-based version (optional)
        # If you do NOT want chunking in the final projection, skip this step.
        self.lm_head = llama_causallm.lm_head

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Standard forward pass for inference:
          - chunked MLP inside each layer (via the MLP wrapper),
          - chunked LM head if desired,
          - no custom cross-entropy or training logic.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1) Forward through the base llama model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        # outputs[0] = hidden_states of shape [B, seq_len, hidden_dim]
        hidden_states = outputs[0] 
        # print(hidden_states[..., :10, :])
        hidden_states = hidden_states[..., -1:, :] 

        # 2) Project hidden_states -> logits (with chunking if seq_len is large)
        logits = self.lm_head(hidden_states)

        loss = None
        # (For inference, we typically do not compute a training loss. 
        #  But if you wanted it for perplexity on a large sequence, you could add it here.)

        if not return_dict:
            return (logits,) + outputs[1:]
        else:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    def save_pretrained(self, *args, **kwargs):
        # Save the underlying base model + the chunked LM head
        self.model.save_pretrained(*args, **kwargs)

        
class minisequence_inference(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._mlp_count = 0
        self._wrap_done = False
        self._is_qwen_mistral = False

        self.mlp_types = (transformers.models.llama.modeling_llama.LlamaMLP,
                          transformers.models.qwen2.modeling_qwen2.Qwen2MLP,
                        #   transformers.models.gemma3.modeling_gemma3.Gemma3MLP)
                        transformers.models.mistral.modeling_mistral.MistralMLP)
        
        self.casuallm_types = ( transformers.models.llama.modeling_llama.LlamaForCausalLM,
                        transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM,
                        # transformers.models.gemma3.modeling_gemma3.Gemma3ForCausalLM)
                        transformers.models.mistral.modeling_mistral.MistralForCausalLM)
        # We might find how many MLPs total are in the model to detect the final MLP 
        self.total_mlps = self._count_mlps(module)
        
        self.RecursiveVisit('module', self.module, self)

    def _count_mlps(self, module: nn.Module) -> int:
        count = 0
        for name, child in module.named_modules():
            if isinstance(child, transformers.models.llama.modeling_llama.LlamaMLP):
                count += 1
        return count

    def RecursiveVisit(self, name: str, module: nn.Module, upper_module: nn.Module):

        is_llama_mlp = isinstance(module, self.mlp_types)
        is_llama_causallm = isinstance(module, self.casuallm_types)
        self._is_qwen_mistral = self._is_qwen_mistral or isinstance(module, transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM)
        self._is_qwen_mistral = self._is_qwen_mistral or isinstance(module, transformers.models.mistral.modeling_mistral.MistralForCausalLM)
        has_children = any(isinstance(child, nn.Module) for child in module.children())

        if has_children and not is_llama_mlp:
            for n, child in module.named_children():
                self.RecursiveVisit(n, child, module)

        if is_llama_mlp:
            self._mlp_count += 1
            # print(f'mlp layer {self._mlp_count}')
            # If it's not the last MLP, wrap with MST
            if (self._mlp_count < self.total_mlps) or self._is_qwen_mistral:
                wrapped = LlamaMLPInferenceWrapper(module, is_final_mlp=False)
            else:
                # This is the last MLP
                wrapped = LlamaMLPInferenceWrapper(module, is_final_mlp=True)
                # print('is final mlp')
            setattr(upper_module, name, wrapped)

        if is_llama_causallm:
            wrapped = LlamaForCausalLMInferenceWrapper(module)
            # print('causallm')
            setattr(upper_module, name, wrapped)

    def forward(self, *args, **kwargs):
        outputs = self.module(*args, **kwargs)
        return outputs

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Get the state dict of the wrapped module
        module_state_dict = self.module.state_dict(destination, prefix, keep_vars)
        
        # Create a new state dict without the 'module.' prefix
        new_state_dict = {k: v for k, v in module_state_dict.items()}
        
        return new_state_dict

    def save_pretrained(self, *args, **kwargs):
        # Check if the module has a save_pretrained method
        self.module.save_pretrained(*args, **kwargs)

