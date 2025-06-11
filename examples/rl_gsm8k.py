import random, os, sys, re
       
def format_fn(answer, item):
    pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"     
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1

def correct_fn(answer, item):    
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer) # 使用正则表达式在answer中查找所有数字
    if len(nums) == 0: return -1.0
    lastnum = nums[-1] # 用answer中最后一个数字和ground_truth做比较
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else -1

system_prompt = """The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> and <answer> tags."""

def make_prompt_fn(self, item):
    return self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item['Q']}], 
            tokenize=False, add_generation_prompt=True)

from lsrl import LSRL, RefServer
model_path = "/data2/Qwen/Qwen2.5-14B-Instruct"
if 'ref' in sys.argv:
    RefServer(model_path).start()
    sys.exit(0)
    
from math_verify import parse, verify, ExprExtractionConfig

if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset("meta-math/GSM8K_zh", "default", split="train")
    QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question_zh'], dataset['answer'])]
    random.shuffle(QAs)

    lsrl = LSRL(model_path, epochs=1, train_data=QAs, rollout_num=8, 
                train_batch_size=8, gen_batch_size=4,
                gen_update_steps=16, trainer='LSCPU', gen_temperature=0.9,
                gen_device=[1], ref_server="http://10.176.40.138:59876",
                lr=1e-6, accum_steps=16, genlog_filename='rl_log',)
    lsrl.add_reward(format_fn)
    lsrl.add_reward(correct_fn)
    lsrl.set_policy_prompt_fn(make_prompt_fn)
    lsrl.set_rollout_prompt_fn(make_prompt_fn)
    lsrl.train()