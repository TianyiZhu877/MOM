import transformers
import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache, OffloadedCache, OffloadedStaticCache
from MST_experiment_utils.MSI import minisequence_inference
from MST_experiment_utils import *
import importlib


model_ckpt = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
# model_ckpt = "meta-llama/Llama-3.2-3B-Instruct"

# quantization_config = transformers.BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype="float16"  # Computation done in float16
# )
quantization_config = None

model = "MOM"
# model = "standard"

# context_lens = [6000, 12000, 18000, 24000]
context_lens = [5000, 10000, 15000, 20000, 25000, 30000, 35000]
placements = [0, 0.25, 0.5, 0.75, 1]

if model == "MOM":
    model_generator = lambda: general_model(model_ckpt, MST=minisequence_inference, quantization = quantization_config)
    runner_generator = lambda:  decodeOnlyOffload(max_new_tokens=100, stop_by_eos = True)
    # max_context = 24000
    max_context = 450000


if model == "standard":
    model_generator = lambda: general_model(model_ckpt, MST=False, quantization = quantization_config)
    runner_generator = lambda: regularRecursive(stop_by_eos = True, max_new_tokens=100)
    # max_context = 12000
    max_context = 150000

results = run_needle_test(needleInBook, model_generator, runner_generator, context_lens, placements, max_context)
print(results)
np.save("outputs/5_results_"+model+".npy", results)
