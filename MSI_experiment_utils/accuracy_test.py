from .run_test import run_single_test, cuda_cleanup
import numpy as np

def run_needle_test(needleGenerator, model_generator, runner_generator, context_lens, placements, max_context = None, print_answer = True):
    results = np.zeros((len(placements), len(context_lens)))
    test_runned = 0
    total_tests = len(context_lens)* len(placements)
    cuda_cleanup(True)
    for i, context_len in enumerate(context_lens):
        for j, placement in enumerate(placements):
            test_runned += 1
            print('********************************')
            print(f'Test {test_runned}/{total_tests}: {context_len} length placing at {placement}')
            # print()
            needle_generator = needleGenerator(context_len, placement, max_context)
            stats, answer = run_single_test(needle_generator.generate, model_generator, runner_generator)
            results[j, i] = needle_generator.evaluate(answer)
            cuda_cleanup(True)
            
            if print_answer:
                print('Response: ', answer)

            del needle_generator
    
    return results
