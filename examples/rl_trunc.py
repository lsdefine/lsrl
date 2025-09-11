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

system_prompt = 'Let\'s think step by step and output the final answer in \boxed{}.'

def make_prompt_fn(self, item):
    return self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item['question']}], 
            tokenize=False, add_generation_prompt=True)

def extract_boxed_sentence(s: str):
    matches = list(re.finditer(r'\\boxed\{.*?\}', s))
    if not matches: return s  
    last_boxed = matches[-1]
    rest = s[last_boxed.end():]
    pos_in_rest = rest.find(".")
    if pos_in_rest == -1: return s  
    return s[: last_boxed.end() + pos_in_rest + 1]


def generate_normal(vllm_gen, prompts, n, T, max_tokens, use_tqdm=False):
    from vllm import SamplingParams
    sampling_params = SamplingParams(n=n, temperature=T, max_tokens=max_tokens)
    voutputs = vllm_gen.generate(prompts, sampling_params, use_tqdm=use_tqdm)
    answers = []
    stop_reasons = []
    for v in voutputs:
        for z in v.outputs: 
            #answers.append({'text':z.text, 'token_ids': z.token_ids})
            answers.append(z.text) #这里修改为只返回生成答案的文本，而不包括token_ids
            if z.finish_reason in ['length', 'max_tokens', 'max_length']:
                stop_reasons.append(1)
            else:
                stop_reasons.append(0)   
    # assert len(answers) == len(prompts) * self.rollout_num
    return answers

def gen_samples(self, items):
    add_info = "\n\nWait, I have enough information to get the final answer.</think>\n\nTherefore, the final answer is \\boxed{"
    gen_normal_num_single = self.rollout_num // 4
    all_prompts = []
    prompt_to_item_index = []  
    for item_idx, item in enumerate(items):
        prompt = self.rollout_prompt_fn(item)
        all_prompts.extend([prompt] * gen_normal_num_single)
        prompt_to_item_index.extend([item_idx] * gen_normal_num_single)
    all_answers = self.generate_normal(self.vllm_gen, all_prompts, 1, self.gen_temperature, self.gen_max_tokens, use_tqdm=self.rank==0)

    ratios = [0.25, 0.5, 0.75]
    continue_prompts = []
    for i, answer in enumerate(all_answers):
        ans_len = len(answer)
        for len_ratio in ratios:
            cut_len = max(1, int(ans_len * len_ratio))
            continue_prompt = all_prompts[i] + answer[:cut_len] + add_info
            continue_prompts.append(continue_prompt)

    if continue_prompts:
        print(f"续写 {len(continue_prompts)} 个截断的答案...")
        continued_parts = self.generate_normal(self.vllm_gen, continue_prompts, 1, self.gen_temperature, 50)

    continued_answers = []
    for ans in continued_parts:
        continued_answers.append(extract_boxed_sentence(ans))

    cont_idx = 0
    final_answers_per_item = [[] for _ in items]
    for idx, answer in enumerate(all_answers):
        item_idx = prompt_to_item_index[idx]
        combined = [answer]
        for ratio in ratios:
            cut_len = max(1, int(len(answer) * ratio))
            prefix = answer[:cut_len] + add_info
            cont = continued_answers[cont_idx]
            cur_ans = prefix + cont
            combined.append(cur_ans)
            cont_idx += 1
        final_answers_per_item[item_idx].extend(combined)

    answers = [ans for sublist in final_answers_per_item for ans in sublist]
    answers = [s + '<|end_of_text|>' for s in answers]
    print(f"!!最终答案的数量:{len(answers)}")

    rewards = []
    rollout_num = len(answers) // len(items)

    answer_token_length = []
    answer_token_ids = []
    for ans in answers:
        answer_token_ids.append(self.tokenizer(ans)['input_ids'])
        answer_token_length.append(len(answer_token_ids[-1]))

    lenght_min, length_max = min(answer_token_length), max(answer_token_length)
    eps_l = 1e-8

    denom = length_max - lenght_min + eps_l
    correct_rewards = []
    for i, ans in enumerate(answers):
        reward = {}
        for reward_fn in self.reward_fns:
            score = reward_fn(ans, items[i // rollout_num])
            reward[reward_fn.__name__] = score
        correct_rewards.append(reward.get('correct_fn', -1))
    
        alpha = 1.2; delta = 0.05
        length_reward = (length_max - answer_token_length[i]) / denom
        length_reward = length_reward * alpha * (1 - delta) + delta
        if reward.get('correct_fn', -1) > 0:
            reward['token_reward'] = length_reward
        else:
            reward['token_reward'] = 0

        reward['total'] = reward['correct_fn'] + reward['token_reward'] + reward['format_fn']
        reward['length'] = answer_token_length[i]
        rewards.append(reward)

    print(f"!!!reward length:{len(correct_rewards)}")

    answers = [{'text': a, 'token_ids': b} for a, b in zip(answers, answer_token_ids)]        
    policy_prompts = [self.policy_prompt_fn(x) for x in items]
    return {'prompts': policy_prompts, 'answers': answers, 'rewards': rewards}


from lsrl import LSRL, RefServer, SyncLSRL
model_path = "/data2/Qwen/DeepSeek-R1-Distill-Qwen-7B"
if 'ref' in sys.argv:
    RefServer(None, port=59888).start()  
    sys.exit(0)
       
from math_verify import parse, verify, ExprExtractionConfig
# math_verify can not run in another thread!!!
# async reward processor is used to requests reward server

if 'test' in sys.argv:
    number = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    print(f'Testing model at step {number}...')
    if number > 0: model_path = f'./step_{number}'

    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    QAs = [{'question':x, 'std':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]
        
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
            f.write(f'Q: {item["question"]}\nA: {item["std"]}\nVLLM: {text}\n\n')
    sys.exit()


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
    with open("/data2/ljq/mix_math_dapo_gpqa_main.json", 'r', encoding='utf-8') as f:
        QAs = json.load(f)
    print(f"训练的总长度：{len(QAs)}")
    random.seed(42)
    random.shuffle(QAs)

    '''
    lsrl = LSRL(model_path, epochs=1, train_data=QAs, rollout_num=8, 
                train_batch_size=4, gen_batch_size=8, gen_max_tokens=8000,
                gen_update_steps=8, trainer='LSCPU', gen_temperature=0.6,
                gen_device=[4,6,7], ref_server="http://127.0.0.1:59888", beta=0,
                lr=2e-6, accum_steps=8, genlog_filename='rl_log', clip_param=0.2,
                save_steps=900, skip_zero_groups=False, algorithm='GRPO', 
                max_pending_samples=30, gen_pending_time=60)
    '''
    
    lsrl = SyncLSRL(model_path, epochs=1, train_data=QAs, rollout_num=8, 
                train_batch_size=4, gen_batch_size=32, gen_max_tokens=8000,
                trainer='LSCPU', gen_temperature=0.6, beta=0,
                update_times_per_step=2)
    
    lsrl.set_hook('after_rollout', rollout_monitor)
    lsrl.add_reward(correct_fn)
    lsrl.add_reward(format_fn)
    lsrl.set_policy_prompt_fn(make_prompt_fn)
    lsrl.set_rollout_prompt_fn(make_prompt_fn)
    lsrl.gen_samples = gen_samples
    lsrl.generate_normal = generate_normal
    lsrl.RL_step = lsrl.DAPO_step  # use DAPO loss but no other changes
    lsrl.train()