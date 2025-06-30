import random, os, sys, re, time, requests
       
def format_fn(answer, item):
    pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"     
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1

def server_fn(answer, item):    
    #return requests.post(f"http://127.0.0.1:54123/upload", data=json_to_bytes_list({'ans':answer, 'ref':item['A']})).json().get('reward', -1.0)
    time.sleep(1.1)  # simulate reward server
    return time.time() % 2 

system_prompt = """The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> and <answer> tags."""

def make_prompt_fn(self, item):
    return self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item['Q']}], 
            tokenize=False, add_generation_prompt=True)

from lsrl import LSRL, RefServer, json_to_bytes_list
model_path = "/data2/Qwen/Qwen2.5-14B-Instruct"
if 'ref' in sys.argv:
    RefServer(model_path).start()
    sys.exit(0)   

if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset("meta-math/GSM8K_zh", "default", split="train")
    QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question_zh'], dataset['answer'])]
    random.shuffle(QAs)

    lsrl = LSRL(model_path, epochs=1, train_data=QAs, rollout_num=8, 
                train_batch_size=8, gen_batch_size=4,
                gen_update_steps=16, trainer='LSCPU', gen_temperature=0.9,
                gen_device=[4], ref_server="http://10.176.40.135:59876",
                lr=1e-6, accum_steps=16, genlog_filename='rl_log',
                reward_processor='async')
    lsrl.add_reward(format_fn)
    lsrl.add_reward(server_fn)
    lsrl.set_policy_prompt_fn(make_prompt_fn)
    lsrl.set_rollout_prompt_fn(make_prompt_fn)
    lsrl.train()