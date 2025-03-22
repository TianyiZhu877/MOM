import transformers
import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache, OffloadedCache, OffloadedStaticCache
from MST_experiment_utils.MSI import minisequence_inference
from MST_experiment_utils import *
import importlib


# model_ckpt = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
# contexts = {
#     '48000': lambda tokenizer: needle_in_book(48000, tokenizer),
#     '80000': lambda tokenizer: needle_in_book(80000, tokenizer),
#     '112000': lambda tokenizer: needle_in_book(112000, tokenizer),
#     '144000': lambda tokenizer: needle_in_book(144000, tokenizer),
# }



model_ckpt = "meta-llama/Llama-3.2-3B-Instruct"
contexts = {
    '12000': lambda tokenizer: needle_in_book(12000, tokenizer),
    '8000': lambda tokenizer: needle_in_book(8000, tokenizer),
    '4000': lambda tokenizer: needle_in_book(4000, tokenizer)
}


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

model_scatter(results, dims, get_avg_throughput, get_avg_mem_use_ratio, 'Memory saved ratio vs. Average throughput', y_lim = (40, 105), save_dir = 'outputs/3_MemorysavedratiovsAveragethroughput.png') #, style='darkgrid')

