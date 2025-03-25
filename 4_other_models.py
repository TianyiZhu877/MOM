import transformers
import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache, OffloadedCache, OffloadedStaticCache
from MST_experiment_utils.MSI import minisequence_inference
from MST_experiment_utils import *
import importlib


model_ckpt = "Qwen/Qwen2.5-7B-Instruct-1M"
# model_ckpt = "NousResearch/Yarn-Llama-2-7b-128k"
contexts = {
    '32000': lambda tokenizer: needle_in_book(32000, tokenizer),
    '64000': lambda tokenizer: needle_in_book(64000, tokenizer),
    '96000': lambda tokenizer: needle_in_book(96000, tokenizer),
    '128000': lambda tokenizer: needle_in_book(128000, tokenizer),
}



# model_ckpt = "Qwen/Qwen2.5-3B-Instruct"
# contexts = {
#     '12000': lambda tokenizer: needle_in_book(12000, tokenizer),
#     '8000': lambda tokenizer: needle_in_book(8000, tokenizer),
#     '4000': lambda tokenizer: needle_in_book(4000, tokenizer)
# }


models = {

    'vanilla': (lambda: general_model(model_ckpt, MST=False),
                lambda: regularRecursive()),

    'MOM': (lambda: general_model(model_ckpt, MST=minisequence_inference),
                lambda: decodeOnlyOffload()),


    'MSI (no offload)': (lambda: general_model(model_ckpt, MST=minisequence_inference), 
                lambda: regularRecursive()),


    'prefill chunk=512': (lambda: general_model(model_ckpt, MST=False),
                lambda: chunkPrefill(chunk_size=512)),


    'prefill chunk=8192': (lambda: general_model(model_ckpt, MST=False),
                lambda: chunkPrefill(chunk_size=8192)),

}

dims, results = run_test(contexts, models)

context_plot(results, dims, title = 'Memory use vs. Context length', save_dir = 'outputs/3_mem_curve.png')

x, y = model_scatter(results, dims, get_avg_throughput, get_avg_mem_use_ratio, 'Memory saved ratio vs. Average throughput', y_lim = (40, 105), save_dir = 'outputs/3_MemorysavedratiovsAveragethroughput.png') #, style='darkgrid')
print('x = ', x)
print('y = ', y)
