import transformers
import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache, OffloadedCache, OffloadedStaticCache
from MST_experiment_utils.MSI import minisequence_inference
from MST_experiment_utils import *
import importlib



model_ckpt = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
contexts = {
    '48000': lambda tokenizer: needle_in_book(48000, tokenizer),
    '80000': lambda tokenizer: needle_in_book(80000, tokenizer),
    '112000': lambda tokenizer: needle_in_book(112000, tokenizer),
    '144000': lambda tokenizer: needle_in_book(144000, tokenizer),
}



# model_ckpt = "meta-llama/Llama-3.2-3B-Instruct"
# contexts = {
#     '12000': lambda tokenizer: needle_in_book(12000, tokenizer),
#     '8000': lambda tokenizer: needle_in_book(8000, tokenizer),
#     '4000': lambda tokenizer: needle_in_book(4000, tokenizer)
# }


models = {
    'vanilla': (lambda: general_model(model_ckpt, attn_implementation=None, MST=False),
                lambda: regularRecursive()),

    '(prefill only) Offload': (lambda: general_model(model_ckpt, attn_implementation=None, MST=False),
                lambda: decodeOnlyOffload()),


    'MST': (lambda: general_model(model_ckpt, attn_implementation=None, MST=minisequence_inference),
                lambda: regularRecursive()),

    'Offload + MST': (lambda: general_model(model_ckpt, attn_implementation=None, MST=minisequence_inference),
                lambda: decodeOnlyOffload()),

}


dims, results = run_test(contexts, models)

context_plot(results, dims, title = 'Memory use vs. Context length', save_dir = 'outputs/MSIandOffloading.png')


print('************************\n% tabel get_end2end_runtime')
new_dims, new_results = get_metric(dims, results, get_end2end_runtime)
tab_2d(new_dims, new_results)
print()

print('************************\n% tabel First Token Delay')
new_dims, new_results = get_metric(dims, results, 'First Token Delay')
tab_2d(new_dims, new_results)
print()

print('************************\n% tabel Decoding Time')
new_dims, new_results = get_metric(dims, results, get_output_speed)
tab_2d(new_dims, new_results)




def offload_compare_MST_no_MST(dims, data, float_format="%.3f"):
    dim_names = list(dims.keys())
    num_models = len(dims['computing_model'])
    new_dims = dims.copy()


    new_dims['computing_model'] = dims['computing_model'][:num_models//2]
    # new_dims['metric'] = ['Memory MST / Memory without MST (%)']
    # print(data[:, num_models//2 + 1:, 3], data[:, :num_models//2, 3])
    data_new = data[:, num_models//2:, 3]/data[:, :num_models//2, 3]*100
    # data_new = -data[:, num_models//2 + 1:, 3:4]+data[:, :num_models//2, 3:4]

    # print(new_dims, data_new)
    return new_dims, data_new
    # df = pd.DataFrame(
    #     data_new,
    #     index=new_dims[dim_names[0]],
    #     columns=['Memory MST / Memory without MST (%)']
    # )


    # df = df.T
    # print(df.to_latex(float_format=float_format))
    # return df

print('************************\ntabel offload_compare_MST_no_MST')
new_dims, new_results = offload_compare_MST_no_MST(dims, results)
tab_2d(new_dims, new_results)


