import random, os, sys, re, time, requests, json
os.environ['OMP_NUM_THREADS'] = '32'

def format_fn(answer, item):
    box_match = re.search(r'\\boxed\{.*?\}', answer)
    if not box_match: return -1.0
    return 1.0

def correct_fn(answer, item):    
    lastnum = re.search(r'\\boxed{([^}]+)}', answer)
    if not lastnum: return -1.25
    lastnum = lastnum.group(1)  
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["std"], extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else -1

system_prompt = '''Let\'s think step by step and output the final answer in \\boxed{}. 
You can use one line python code with <python></python> to help you.'''

def make_prompt_fn(self, item):
    return self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item['question']}], 
            tokenize=False, add_generation_prompt=True)

def gen_w_pytool(llm, prompts, rollnum=4, temperature=0.9, max_tokens=5000):
    from vllm import SamplingParams
    sampling_params = SamplingParams(n=1, temperature=temperature, max_tokens=max_tokens, 
                                    stop=["</python>"], repetition_penalty=1.1,
                                    include_stop_str_in_output=True)
    if not isinstance(prompts, list): prompts = [prompts]
    totals = [{'prompt': prompt, 'texts':[], 'len':0} for prompt in prompts for _ in range(rollnum)]
    [x.update({'idx':i}) for i, x in enumerate(totals)]
    gids = list(range(len(totals)))
    while True:
        inps = [totals[i]['prompt']+''.join(totals[i]['texts']) for i in gids]
        voutputs = llm.generate(inps, sampling_params, use_tqdm=False)
        rets = []
        for v in voutputs:
            for z in v.outputs: rets.append({'text': z.text, 'token_ids': z.token_ids})
        stillneed = []
        for i, r in enumerate(rets):
            idx = gids[i]
            totals[idx]['texts'].append(r['text'])
            totals[idx]['len'] += len(r['token_ids'])
            if r['text'].endswith('</python>'):
                pat = re.compile(r"<python>(.*?)</python>\s*$", re.S)
                matches = list(pat.finditer(r['text']))
                code = matches[-1].group(1) if matches else ''
                if code.strip() != '':
                    ns = {}
                    try: exec(code, {}, ns)
                    except Exception as e: ns = e
                else: ns = 'No code found'
                ctext = '\nExecution Result: ' + str(ns) + '\n'
                totals[idx]['texts'].append(ctext)
                if totals[idx]['len'] > max_tokens: continue
                stillneed.append(idx)
        if len(stillneed) == 0: break
        gids = stillneed
        sampling_params.max_tokens = max_tokens - min([totals[i]['len'] for i in gids])
    return totals

def gen_samples(self, items):
    prompts = [self.rollout_prompt_fn(x) for x in items]
    totals = gen_w_pytool(self.vllm_gen, prompts, rollnum=self.rollout_num,
                        temperature=self.gen_temperature, max_tokens=self.gen_max_tokens)
    answers = [''.join(x['texts']) + self.tokenizer.special_tokens_map['eos_token'] for x in totals]
    answer_token_ids = [self.tokenizer(ans)['input_ids'] for ans in answers]
    answers = [{'text': a, 'token_ids': b} for a, b in zip(answers, answer_token_ids)]        
    for ansd in answers:
        if len(ansd['token_ids']) > 8000:
            ansd['token_ids'] = ansd['token_ids'][:self.gen_max_tokens]
            ansd['text'] = self.tokenizer.decode(ansd['token_ids'])
    policy_prompts = [self.policy_prompt_fn(x) for x in items]
    return {'prompts': policy_prompts, 'answers': answers}


from lsrl import SyncLSRL
model_path = "/mnt/data_nvme0/LLM/Qwen2.5-7B-Instruct"
       
from math_verify import parse, verify, ExprExtractionConfig
# math_verify can not run in another thread!!!
# async reward processor is used to requests reward server

all_corrs = []
def rollout_monitor(samples):
    global all_corrs
    corrs = []; lengths = []
    for rewards in samples['rewards']: 
        corrs.append(rewards.get('correct_fn', -1))
        lengths.append(rewards.get('length', 0))
    corrs = [1 if x > 0 else 0 for x in corrs]
    all_corrs.extend(corrs)
    if len(all_corrs) > 1000: all_corrs = all_corrs[-1000:]  
    acc = sum(all_corrs) / len(all_corrs)
    lens = sum(lengths) / len(lengths) if lengths else 0
    print(f'[ROLL] rollout monitor: acc: {acc:.2f};   length: {lens}')

if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset("./gsm8k", "main", split="test")
    QAs = [{'question':x, 'std':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]
    print(f"训练的总长度：{len(QAs)}")
    random.seed(42)
    random.shuffle(QAs)

    lsrl = SyncLSRL(model_path, epochs=1, train_data=QAs, rollout_num=8, 
                train_batch_size=4, gen_batch_size=4, gen_max_tokens=7500,
                trainer='LSCPU', gen_temperature=0.9, beta=0, 
                lr=3e-6, genlog_filename='rl_log',
                update_times_per_step=2, save_steps=20, swanlab=False)
    
    #if lsrl.rank == 0:
    #    import swanlab
    #    swanlab.login(api_key="", save=True)
    #    swanlab.init(project="test-sync-lsrl")
    
    lsrl.set_hook('after_rollout', rollout_monitor)
    lsrl.add_reward(correct_fn)
    lsrl.add_reward(format_fn)
    lsrl.set_policy_prompt_fn(make_prompt_fn)
    lsrl.set_rollout_prompt_fn(make_prompt_fn)
    lsrl.gen_samples = gen_samples
    lsrl.RL_step = lsrl.DAPO_step  # use DAPO loss but no other changes
    if 'dry' in sys.argv: lsrl.dry_check_rollout(gen_batch_size=2, rollout_num=3)
    lsrl.train()