import transformers
import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache, OffloadedCache, OffloadedStaticCache

class regularRecursive:
    def __init__(self, max_new_tokens=500, cache = DynamicCache, stop_by_eos = True):
        self.past_key_values = cache()
        # self.eos_token_ids
        self.next_input = None
        self.stop_by_eos = stop_by_eos
        self.max_new_tokens = max_new_tokens
        self.generated_tokens = []
    
    def prefill(self, model, input_ids):
        # print('torch.cuda.memory_stats(0)["allocated_bytes.all.peak"]', torch.cuda.memory_stats(0)["allocated_bytes.all.peak"]/2**30)
        outputs = model(input_ids=input_ids, past_key_values=self.past_key_values, use_cache=True)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        self.generated_tokens.append(next_token.item())
        self.next_input = next_token.unsqueeze(-1)
        

    def decode(self, model, eos_token_ids):
        for token_count in range(self.max_new_tokens):
            outputs = model(input_ids=self.next_input, past_key_values=self.past_key_values, use_cache=True)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            self.next_input = next_token.unsqueeze(-1)

            if next_token.item() in eos_token_ids:
                break

            self.generated_tokens.append(next_token.item())
        return self.generated_tokens, None


class chunkPrefill:
    def __init__(self, chunk_size, max_new_tokens=500, cache = DynamicCache, stop_by_eos = True):
        self.chunk_size = chunk_size
        self.past_key_values = cache()
        # self.eos_token_ids
        self.next_input = None
        self.stop_by_eos = stop_by_eos
        self.max_new_tokens = max_new_tokens
        self.generated_tokens = []
    
    def prefill(self, model, input_ids):
        input_len = input_ids.size(1)
        for i in range(0, input_len, self.chunk_size):
            # outputs = model(input_ids=input_ids[:, i : min(i+chunk_size, input_len)], past_key_values=past_key_values, use_cache=True, num_logits_to_keep=1)
            outputs = model(input_ids=input_ids[:, i : min(i+self.chunk_size, input_len)], past_key_values=self.past_key_values, use_cache=True)
            # past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        self.generated_tokens.append(next_token.item())
        self.next_input = next_token.unsqueeze(-1)

        
    def decode(self, model, eos_token_ids):
        for token_count in range(self.max_new_tokens):
            outputs = model(input_ids=self.next_input, past_key_values=self.past_key_values, use_cache=True)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            self.next_input = next_token.unsqueeze(-1)
            
            if next_token.item() in eos_token_ids:
                break

            self.generated_tokens.append(next_token.item())
        return self.generated_tokens, None

        
        