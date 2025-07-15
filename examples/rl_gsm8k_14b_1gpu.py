import random, os, sys, re, time, requests
os.environ['OMP_NUM_THREADS'] = '32'

def correct_fn(answer, item):    
    lastnum = re.search(r'\\boxed{([^}]+)}', answer)
    if not lastnum: return -1.25
    lastnum = lastnum.group(1)  
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else -1

system_prompt = 'Let\'s think step by step and output the final answer in \boxed{}.'

def make_prompt_fn(self, item):
    return self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item['Q']}], 
            tokenize=False, add_generation_prompt=True)

from lsrl import LSRL, RefServer
model_path = "/data2/Qwen/Qwen2.5-14B-Instruct"
if 'ref' in sys.argv:
    RefServer(model_path, port=59888, force_cpu_offload=True, nlayers_keep_in_gpu=0).start()
    sys.exit(0)
       
from math_verify import parse, verify, ExprExtractionConfig
# math_verify can not run in another thread!!!
# async reward processor is used to requests reward server

if 'test' in sys.argv:
    number = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    print(f'Testing model at step {number}...')
    model_path = f'./step_{number}'

    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]
        
    from vllm import LLM, SamplingParams
    vllm_gen = LLM(model=model_path, enable_chunked_prefill=True, 
                gpu_memory_utilization=0.5, enforce_eager=True)
    sampling_params = SamplingParams(n=1, temperature=0.001, max_tokens=1024)

    from transformers import AutoTokenizer
    test_obj = lambda: None
    test_obj.tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = [make_prompt_fn(test_obj, x) for x in QAs]
    voutputs = vllm_gen.generate(prompts, sampling_params, use_tqdm=True)

    corrs = [1 * (correct_fn(x.outputs[0].text, item) > 0) for x, item in zip(voutputs, QAs)]
    print(corrs)
    wrongs = [k for k,v in enumerate(corrs) if v == 0]
    print('Wrong QA:', wrongs)
    acc = sum(corrs) / len(corrs)
    print(f'Accuracy: {acc:.2f}')

    with open('gsm8k_results.txt', 'w') as f:
        for k in wrongs:
            text = voutputs[k].outputs[0].text
            item = QAs[k]
            f.write(f'Q: {item["Q"]}\nA: {item["A"]}\nVLLM: {text}\n\n')
    sys.exit()


all_corrs = []
def rollout_monitor(samples):
    global all_corrs
    corrs = []
    for rewards in samples['rewards']: corrs.append(rewards.get('correct_fn', -1))
    corrs = [1 if x > 0 else 0 for x in corrs]
    all_corrs.extend(corrs)
    if len(all_corrs) > 1000: all_corrs = all_corrs[-1000:]  
    acc = sum(all_corrs) / len(all_corrs)
    print(f'[ROLL] rollout monitor: acc: {acc:.2f}')

if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]
    random.seed(42)
    random.shuffle(QAs)

    lsrl = LSRL(model_path, epochs=1, train_data=QAs, rollout_num=5, 
                train_batch_size=5, gen_batch_size=2, gen_max_tokens=1024,
                gen_update_steps=64, trainer='LSCPU', gen_temperature=0.9,
                gen_device=[], ref_server="http://127.0.0.1:59888", beta=0.001,
                lr=5e-6, accum_steps=64, genlog_filename='rl_log', grad_offload=False,
                save_steps=1100, skip_zero_groups=False, use_vllm=False)
    
    lsrl.set_hook('after_rollout', rollout_monitor)
    lsrl.add_reward(correct_fn)
    lsrl.set_policy_prompt_fn(make_prompt_fn)
    lsrl.set_rollout_prompt_fn(make_prompt_fn)
    lsrl.train()