import time
import transformers
import torch
import numpy as np
import gc

def str_list_to_dict(str_list: list):
    result = {}

    for i, name in enumerate(str_list):
        result[name] = i
    
    return result

metrics = ['Model size', 'Context Length', '#Output tokens', 'Peak Memory',  'First Token Delay', 'Decoding Time']
metric_units = ['MB', 'Tokens', 'Tokens', 'MB', 's', 's']

metric_idx = str_list_to_dict(metrics)

def cuda_cleanup(print_residual_mem = False, device = 0, mem_threshold = 64*(2**20)):
    gc.collect()
    torch.cuda.empty_cache()

    memory_allocated = torch.cuda.memory_allocated(device)
    memory_reserved = torch.cuda.memory_reserved(0)
    not_cleanedup = False
    if (memory_allocated+memory_reserved > mem_threshold):
        print("Warning: Cuda memory allocated not fully released, check garbage collection")
        not_cleanedup = True
    if (not_cleanedup or print_residual_mem):
        print(
            "After cleanup: memory allocated {:4f}MB, reserved {:4f}MB left".format(
                memory_allocated / 2**20, 
                memory_reserved / 2**20
            )
        )



def run_single_test(context_generator, model_generator, runner_generator):
    stats = {}
    # print(torch.cuda.memory_stats(0)["allocated_bytes.all.peak"])
    try:
        torch.cuda.reset_peak_memory_stats(0)
        print("reset memory stats")
    except:
        print("Warning: reset memory stats not reset")


    model, tokenizer, generation_config = model_generator()

    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    # add some tokens like "</user>" and </s> to eos ids
    eos_token_ids += tokenizer.encode("</user>", add_special_tokens=False)
    eos_token_ids += tokenizer.encode("</s>", add_special_tokens=False)
    eos_token_ids += tokenizer.encode("</", add_special_tokens=False)
    
    with torch.no_grad():
        runner = runner_generator()
        context_string = context_generator(tokenizer)

        
        stats['Model size'] = torch.cuda.memory_stats(0)["allocated_bytes.all.peak"] / 2**20
        input_ids = tokenizer.encode(context_string, return_tensors="pt").to("cuda")
        stats['Context Length'] = input_ids.size(1)
        print('stats[Model size] ', stats['Model size'], stats['Context Length'])
        t_start = time.time()
        t = runner.prefill(model, input_ids)
        t_after_prefill = time.time()
        # print('runtime: ', t_after_prefill- t_start)
        if t is None:
            stats['First Token Delay'] = t_after_prefill - t_start
        else:
            stats['First Token Delay'] = t


        t_start = time.time()
        generated_tokens, t = runner.decode(model, eos_token_ids)
        t_after_decode = time.time()
        if t is None:
            stats['Decoding Time'] = t_after_decode - t_start
        else:
            stats['Decoding Time'] = t
    
    stats['Peak Memory'] = torch.cuda.memory_stats(0)["allocated_bytes.all.peak"] / 2**20
    
    stats['#Output tokens'] = len(generated_tokens)
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return stats, answer



def run_test(contexts, models, print_answer=True, print_residual_mem = True, print_stats = True):
    dims = {'context': list(contexts.keys()),
            'computing_model':  list(models.keys()),
            'metric': metrics}
    results = np.zeros((len(contexts), len(models), len(metrics)))
    total_tests = len(contexts)*len(models)
    test_runned = 0
    cuda_cleanup(print_residual_mem)

    for c_i, (context_name, context_generator) in enumerate(contexts.items()):
        # dims['context'].append(context_name)
        for m_i, (model_name, (model_generator, runner_generator)) in enumerate(models.items()):
            # dims['computing_model'].append(model_name)

            print('********************************')
            print(f'Test {test_runned}/{total_tests}: {context_name}+{model_name}')

            # running one test in function to ensure proper garbage collect:
            stats, answer = run_single_test(context_generator, model_generator, runner_generator)

            for i, metric in enumerate(metrics):
                results[c_i, m_i, i] = stats[metric]
            if print_answer:
                print('Response: ', answer)
            if print_stats:
                print(stats)
            
            cuda_cleanup(print_residual_mem)
            test_runned += 1

            print('\n\n')

    return dims, results





