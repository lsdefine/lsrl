import random, os, sys, re, time, requests, json, math
       
def correct_fn(answer, item):    
    return requests.post(f"http://127.0.0.1:59878/get_reward", 
                data=json_to_bytes_list({'output':answer, 
                                         'question':item})).json().get('reward', -1.0)
    

system_prompt = """The user asks a question, and the Assistant solves it. """

def make_prompt_fn(self, item):
    return self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item}], 
            tokenize=False, add_generation_prompt=True)
   
from lsrl import LSRL, RefServer, json_to_bytes_list, RewardServer

class MyRS(RewardServer):
    def init(self, model_path):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = LLM(model_path,
            dtype='bfloat16', max_model_len=32768,
            tensor_parallel_size=1, enforce_eager=True
        )
        allowed_tokens = ["Yes", "No"]
        allowed_token_ids = [self.tokenizer.encode(token, add_special_tokens=False) for token in allowed_tokens]
        self.Yes_tokens = Yes_tokens = allowed_token_ids[0]
        No_tokens = allowed_token_ids[1]

        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            stop="<|eot|>",
            logprobs=len(Yes_tokens + No_tokens),
            top_k=len(Yes_tokens + No_tokens),
            allowed_token_ids=Yes_tokens + No_tokens
        )

    def get_reward(self, data):
        output, question = data.get('output', ''), data.get('question', '')
        
        def remove_explanation_prefix(text):
            explanation_pattern = r"^(```\s*)?Explanation:\s*"
            return re.sub(explanation_pattern, '', text)
        
        output = remove_explanation_prefix(output)

        def truncate_before_question(text):
            marker = "\n\nQuestion:\n"
            index = text.find(marker)
            if index != -1:
                return text[index + len(marker):]
            return text 
        result = truncate_before_question(question)

        def truncate_to_max_length(text, max_len=30000):
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
                text = self.tokenizer.decode(tokens)
            return text
        
        def calculate_sequence_probability(logprobs_list, token_ids):
            total_logprob = 0
            for i, token_id in enumerate(token_ids):
                if i < len(logprobs_list):
                    token_logprobs = logprobs_list[i]
                    if token_id in token_logprobs:
                        total_logprob += token_logprobs[token_id].logprob
                    else:
                        return float('-inf')
            return total_logprob

        prompt_description = "Given the following Question and the corresponding Answer provided by a model, you are required to assess whether the model is certain about its answer. If the model is certain about its answer, output 'Yes'. If the model is uncertain about its answer, output 'No'.\n\n"
        full_prompt = f"{prompt_description}Question:\n{result}\n\nModel's Answer:\n{output}"
        truncated_prompt = truncate_to_max_length(full_prompt)
        messages = [{"role": "user", "content": truncated_prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        output = self.model.generate(formatted_prompt, self.sampling_params)[0]
        generated_text = output.outputs[0].text.strip()
        generated_logprobs = output.outputs[0].logprobs
    
        Yes_logprob = calculate_sequence_probability(generated_logprobs, self.Yes_tokens)
        Yes_prob = math.exp(Yes_logprob)

        response = {
            "question": question,
            "certainty": generated_text,
            "certainty_probability": float(Yes_prob),
            "logprobs": {
                "token": generated_text,
                "probability": float(Yes_prob)
            },
            "reward": float(Yes_prob)
        }
        return response

if 'reward' in sys.argv:
    MyRS('./rwmodel', port=59878).start()
    sys.exit(0)

model_path = "/data2/Qwen/Qwen2.5-7B-Instruct"
if 'ref' in sys.argv:
    RefServer(model_path, port=59877).start()
    sys.exit(0)
    
if __name__ == '__main__':
    with open('combined_30000.jsonl', 'r', encoding='utf-8') as f:
        datas = [json.loads(line) for line in f.readlines()]
    items = [x['prompt'][0]['content'] for x in datas]

    lsrl = LSRL(model_path, epochs=1, train_data=items, rollout_num=4, 
                train_batch_size=4, gen_batch_size=24, gen_max_tokens=512,
                gen_update_steps=128, trainer='LSCPU', gen_temperature=0.9,
                save_steps=30000, gen_device=[4], ref_server="http://10.176.40.139:59877",
                lr=1e-5, accum_steps=128, genlog_filename='rl_log',
                reward_processor='base', gradient_checkpointing_ratio=0.5)
    lsrl.add_reward(correct_fn)
    lsrl.set_policy_prompt_fn(make_prompt_fn)
    lsrl.set_rollout_prompt_fn(make_prompt_fn)
    lsrl.train()