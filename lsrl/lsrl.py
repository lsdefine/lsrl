
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct, types
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime

from tqdm import tqdm
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils import json_to_bytes_list, bytes_list_to_json, save_model

class LSTrainer:
    def __init__(self, model_patch):
        self.model = AutoModelForCausalLM.from_pretrained(model_patch, torch_dtype=torch.bfloat16)
        self.model.train()
    def backward(self, loss): loss.backward()
    def step(self): self.opt.step()
    def get_model(self): return self.model

class LSCPUTrainer(LSTrainer):
    def __init__(self, model_patch, lr=1e-6, accum_steps=16, grad_offload=False):
        super().__init__(model_patch)
        self.model.to('cuda')
        self.device = self.model.device
        self.model.gradient_checkpointing_enable()
        from .cpuadamw import CPUAdamW, DistributedCPUAdamW
        if dist.is_initialized(): CPUAdamW = DistributedCPUAdamW
        self.opt = CPUAdamW(self.model.parameters(), lr=lr, accum_steps=accum_steps, grad_offload=grad_offload)
        self.engine = self.model

class DeepSpeedTrainer(LSTrainer):
    def __init__(self, model_patch, ds_config=None, train_batch_size=2, lr=1e-6, accum_steps=16):
        super().__init__(model_patch)
        import deepspeed
        deepspeed.init_distributed()
        self.ds_config = self.get_default_ds_config() if ds_config is None else ds_config
        self.ds_config['train_micro_batch_size_per_gpu'] = train_batch_size
        self.ds_config['gradient_accumulation_steps'] = accum_steps
        self.ds_config['optimizer']['params']['lr'] = lr
        self.engine, _, _, _ = deepspeed.initialize(config=self.ds_config, model=self.model, 
                                                    model_parameters=self.model.parameters())
        self.device = self.engine.device
        self.opt = self.engine
    
    def get_model(self): return self.engine.module

    def backward(self, loss): self.engine.backward(loss)

    def get_default_ds_config(self):
        return {
            "optimizer": {
                "type": "AdamW",
                "params": { "lr": 1e-6 }
            },
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
                "stage3_gather_16bit_weights_on_model_save": True,
                "offload_optimizer": {"device": "cpu"}
            }
        }

def distbarrier():
    if dist.is_initialized(): dist.barrier()

class GenLogRecorder:
    def __init__(self, filename=None):
        self.base = filename or f"rl_log_{int(time.time())}"
    def parpare(self):
        self.md_file = open(f"{self.base}.md", 'w', encoding='utf-8')
        self.jsonl_file = open(f"{self.base}.jsonl", 'w', encoding='utf-8')
        self.md_file.write("# RL Training Log\n\n")       
    def log(self, iteration, question, samples, rewards):
        if not hasattr(self, 'md_file'): self.parpare()
        answers_texts = [x['text'] for x in samples]
        tokens_lens = [len(x['token_ids']) for x in samples]
        self.md_file.write(f"## Iter {iteration}\n\n**Input:** {str(question)}\n\n")
        for i, (ans, r, tok) in enumerate(zip(answers_texts, rewards, tokens_lens)):
            parts = [f"total: {r['total']:.2f}"] + [f"{k}: {v:.2f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
                                                     for k, v in r.items() if k != 'total']
            self.md_file.write(f"### Answer {i} - {', '.join(parts)}, tokens: {tok}\n\n```\n{ans}\n```\n\n")
        self.md_file.write("---\n\n")
        self.md_file.flush()
        self.jsonl_file.write(json.dumps({
            "iter": iteration, "Q": question, "anss": answers_texts, 
            "rewards": rewards, "lens": tokens_lens
        }, ensure_ascii=False) + '\n')
        self.jsonl_file.flush()

class LSRL:
    def __init__(self, model_path, epochs=1, rollout_num=8, train_data=None, trainer='LSCPU',
                 gen_device=4, train_batch_size=2, gen_update_steps=16, save_steps=200, gen_batch_size=1,
                 beta=0.04, clip_param=0.2, compute_gen_logps=True, ref_server="http://localhost:59876",
                 gen_max_tokens=4096, gen_temperature=0.9, genlog_filename=None, reward_processor='base',
                 max_pending_samples=40,
                 **kwargs):
        self.model_path = model_path
        self.gen_device = [gen_device] if isinstance(gen_device, int) else list(gen_device)
        # GPU device for generation, don't put it in CUDA_VISIBLE_DEVICES with deepspeed
        # TODO: add an assert to check gen_device is not in CUDA_VISIBLE_DEVICES
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.rollout_num = rollout_num
        self.train_data = train_data
        self.gen_batch_size = gen_batch_size
        self.epochs = epochs
        self.all_steps = epochs * len(train_data) * rollout_num // train_batch_size
        self.train_batch_size = train_batch_size
        assert rollout_num % train_batch_size == 0, "rollout_num must be divisible by train_batch_size"
        self.gen_update_steps = gen_update_steps
        self.save_steps = save_steps
        self.compute_gen_logps = compute_gen_logps
        self.generate_fn = None
        self.ref_server = ref_server
        self.gen_max_tokens = gen_max_tokens
        self.gen_temperature = gen_temperature
        self.beta = beta
        self.clip_param = clip_param
        self.reward_fns = []
        self.genlog_recorder = GenLogRecorder(genlog_filename) if genlog_filename else None
        self.reward_processor = reward_processor 
        assert reward_processor in ['base', 'async'], "rollout_processor must be 'base' or 'async'"
        self.max_pending_samples = max_pending_samples

        if trainer == 'LSCPU':
            self.trainer = LSCPUTrainer(model_path, **kwargs)
        elif trainer == 'DeepSpeed':
            self.trainer = DeepSpeedTrainer(model_path, train_batch_size=train_batch_size, **kwargs)
        else:
            raise ValueError("Unsupported trainer type. Use 'LSCPU' or 'DeepSpeed'.")
    
    def add_reward(self, reward_fn):
        self.reward_fns.append(reward_fn)

    def set_rollout_prompt_fn(self, user_fn): self._rollout_prompt_fn = user_fn
    def set_policy_prompt_fn(self, user_fn): self._policy_prompt_fn = user_fn
    def rollout_prompt_fn(self, item): return self._rollout_prompt_fn(self, item)
    def policy_prompt_fn(self, item): return self._policy_prompt_fn(self, item)

    def get_batch(self):
        try:
            r = requests.get(f"{self.ref_server}/get").content
            if r == b'empty': return None
        except: return None
        return bytes_list_to_json(r)

    def GRPO_step(self, model, batch):
        def get_per_token_logps(logits, input_ids):
            per_token_logps = [] 
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)
        
        prompt_length = batch['plen']
        inputs = batch['inputs'].to(self.device)
        advantages = batch['rewards'].to(self.device).unsqueeze(1)
        logits = model(inputs, use_cache=False).logits
        logits = logits[:, :-1, :]  
        input_ids = inputs[:, 1:]  
        per_token_logps = get_per_token_logps(logits, input_ids)
        per_token_logps = per_token_logps[:,prompt_length-1:]
        ref_per_token_logps = batch['refs'].to(per_token_logps.device)
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        completion_mask = (inputs[:, prompt_length:] != self.tokenizer.pad_token_id).int()
        if 'gen_logps' in batch:
            ratio = torch.exp(per_token_logps - batch['gen_logps'].to(self.device))
            clipped_ratio = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param)
            per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
        else: 
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
            assert self.compute_gen_logps is False
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        return loss
       
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('trainer', None)
        return state

    def generate(self, vllm_gen, prompts):
        from vllm import SamplingParams
        sampling_params = SamplingParams(n=self.rollout_num, temperature=self.gen_temperature, 
                                         max_tokens=self.gen_max_tokens)
        voutputs = vllm_gen.generate(prompts, sampling_params, use_tqdm=False)
        answers = []
        for v in voutputs:
            for z in v.outputs: 
                answers.append({'text':z.text, 'token_ids': z.token_ids})
        assert len(answers) == len(prompts) * self.rollout_num
        return answers

    def gen_worker(self, Q_data, Q_state_dict, gen_device, gen_rank=0):
        os.environ["CUDA_VISIBLE_DEVICES"] = f'{gen_device}'
        torch.cuda.set_device(0)
        print(f"Generation worker process uses GPU {gen_device}")
        from vllm import LLM, SamplingParams
        vllm_gen = LLM(model=self.model_path, enable_chunked_prefill=True, gpu_memory_utilization=0.5)
        gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

        def gen_samples(items):
            gen_prompts = [self.rollout_prompt_fn(x) for x in items]
            answers = self.generate(vllm_gen, gen_prompts)
            policy_prompts = [self.policy_prompt_fn(x) for x in items]
            return {'prompts': policy_prompts, 'answers': answers}

        def QueueGetNowait(Q):
            try: return Q.get_nowait()
            except: return None

        curr_ver = -1
        def try_update_model():
            nonlocal curr_ver
            info = QueueGetNowait(Q_state_dict)
            if info is None: return
            ver, new_state_dict = info['ver'], info['sd']
            if ver > curr_ver: 
                curr_ver = ver
                print(f'[VLLM PROC {gen_rank}] recving new model ...')
                llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(new_state_dict.items())
                print(f'[VLLM PROC {gen_rank}] model updated')
            del new_state_dict

        def compute_gen_logps(data):
            if self.compute_gen_logps:
                plen = data['plen']
                zz = vllm_gen.generate(prompt_token_ids=data['inputs'].tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                zz = [xx.prompt_logprobs[plen:] for xx in zz]
                gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                data['gen_logps'] = gen_logps

        def make_batch_inputs(prompt_ids, sub_ans_ids):
            plen = prompt_ids.shape[1]
            tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
            output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=self.tokenizer.pad_token_id) 
            Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
            merged_ids = torch.cat([Qrep, output_ids], dim=1)
            return merged_ids
        
        def gen_rewards(samples, items):
            rewards = []
            for i, ans in enumerate(samples['answers']):
                reward = {}
                for reward_fn in self.reward_fns:
                    reward[reward_fn.__name__] = reward_fn(ans['text'], items[i//self.rollout_num])
                reward['total'] = sum(reward.values())
                rewards.append(reward)
            return rewards

        class RolloutProcessorBase:
            def __init__(self, lsrl): 
                self.lsrl = lsrl
            def run(self, items):
                rn, tbsz = self.lsrl.rollout_num, self.lsrl.train_batch_size
                samples = gen_samples(items)
                samples['rewards'] = gen_rewards(samples, items)
                groups = {
                    'answers': [samples['answers'][i*rn:(i+1)*rn] for i in range(len(items))],
                    'rewards': [samples['rewards'][i*rn:(i+1)*rn] for i in range(len(items))]
                }
                group_avg_rewards = []
                for prompt, ganswers, grewards in zip(samples['prompts'], groups['answers'], groups['rewards']):
                    prompt_ids = self.lsrl.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
                    curr_ans_ids = [x['token_ids'] for x in ganswers]
                    curr_rewards = torch.tensor([x['total'] for x in grewards], dtype=torch.float32)
                    group_avg_rewards.append(f'{curr_rewards.mean().item():.2f}')
                    if curr_rewards.max() - curr_rewards.min() < 1e-4: continue
                    curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
                    for ii in range(0, rn, tbsz):
                        data = {
                            'plen': prompt_ids.shape[1],
                            'inputs': make_batch_inputs(prompt_ids, curr_ans_ids[ii:ii+tbsz]),
                            'rewards': curr_rewards[ii:ii+tbsz]
                        }
                        compute_gen_logps(data)
                        rc = requests.post(f"{self.lsrl.ref_server}/upload", data=json_to_bytes_list(data))
                try: remain_cnt = rc.json().get('remain_cnt', 0)
                except: remain_cnt = 0
                return {'samples':samples, 'group_avg_rewards': group_avg_rewards, 'remain_cnt': remain_cnt}
    
        class RolloutProcessorAsync(RolloutProcessorBase):
            def __init__(self, lsrl):
                super().__init__(lsrl)
                import queue, threading
                self.sample_queue = queue.Queue()
                self.result_queue = queue.Queue()
                threading.Thread(target=self._reward_worker, daemon=True).start()  

            def _reward_worker(self):
                while True:
                    task = self.sample_queue.get()
                    if task is None: break
                    self.result_queue.put(gen_rewards(task['samples'], task['items']))

            def run(self, items):
                rn, tbsz = self.lsrl.rollout_num, self.lsrl.train_batch_size
                samples = gen_samples(items)
                self.sample_queue.put({'samples': samples, 'items': items}) 
                groups = {'answers': [samples['answers'][i*rn:(i+1)*rn] for i in range(len(items))]}
                batches = []
                for prompt, ganswers in zip(samples['prompts'], groups['answers']):
                    prompt_ids = self.lsrl.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
                    curr_ans_ids = [x['token_ids'] for x in ganswers]
                    for ii in range(0, rn, tbsz):         
                        data = {
                            'plen': prompt_ids.shape[1],
                            'inputs': make_batch_inputs(prompt_ids, curr_ans_ids[ii:ii+tbsz])
                        }
                        compute_gen_logps(data)
                        batches.append(data)
                samples['rewards'] = self.result_queue.get()
                groups['rewards'] = [samples['rewards'][i*rn:(i+1)*rn] for i in range(len(items))]
                group_avg_rewards = []
                batch_id = 0
                for grewards in groups['rewards']:
                    curr_rewards = torch.tensor([x['total'] for x in grewards], dtype=torch.float32)
                    group_avg_rewards.append(f'{curr_rewards.mean().item():.2f}')
                    bad = curr_rewards.max() - curr_rewards.min() < 1e-4
                    curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
                    for ii in range(0, rn, tbsz): 
                        if not bad: batches[batch_id]['rewards'] = curr_rewards[ii:ii+tbsz]
                        batch_id += 1

                for data in batches:
                    if 'rewards' not in data: continue
                    rc = requests.post(f"{self.lsrl.ref_server}/upload", data=json_to_bytes_list(data))
                try: remain_cnt = rc.json().get('remain_cnt', 0)
                except: remain_cnt = 0
                return {'samples':samples, 'group_avg_rewards': group_avg_rewards, 'remain_cnt': remain_cnt}

            
        rn = self.rollout_num 
        from torch.nn.utils.rnn import pad_sequence
        
        ROLLOUT_PROCESSORS = {
            'base': RolloutProcessorBase,
            'async': RolloutProcessorAsync
        }
        RolloutProcessor = ROLLOUT_PROCESSORS[self.reward_processor]
        RP = RolloutProcessor(self)
        for it in range(99999999): # while True
            items = QueueGetNowait(Q_data)
            if items is None: break
            if 'end' in items: 
                print('\nGeneration worker finished, sending end signal to ref server ...')
                time.sleep(5)
                requests.post(f"{self.ref_server}/upload", data=json_to_bytes_list({'end':1}))            
                break

            if it % 2 == 0: try_update_model()

            tic = time.time()
            items = items['batch']
            rr = RP.run(items)
            samples = rr['samples']
            group_avg_rewards = rr['group_avg_rewards']
            if gen_rank == 0 and self.genlog_recorder:
                self.genlog_recorder.log(it, items[0], samples['answers'][:rn], samples['rewards'][:rn])
            print(f'[GEN {gen_rank}]  time: {time.time()-tic:.2f}s    ', f'avg_rewards: {",".join(group_avg_rewards)}' )
            if rr['remain_cnt'] > self.max_pending_samples: 
                print(f'[GEN {gen_rank}] pending samples too many, wait for training process ...')
                time.sleep(10)


    def start_gen_worker(self):
        print('\nSTART vLLM generation...\n')
        ctx = mp.get_context('spawn')
        self.Q_data = ctx.Queue()
        self.Q_state_dict = ctx.Queue()
        for epoch in range(self.epochs):
            items = list(self.train_data)
            for i in range(0, len(items), self.gen_batch_size):
                batch = items[i:i+self.gen_batch_size]
                self.Q_data.put({'batch': batch})
        self.Q_data.put({'end': 1}) 
        for it, gendevice in enumerate(self.gen_device):
            p = ctx.Process(target=self.gen_worker, args=(self.Q_data, self.Q_state_dict, gendevice, it))
            p.start()

    def train(self):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if self.rank == 0: self.start_gen_worker()

        self.device = self.trainer.device
        progress = range(1, self.all_steps+1)
        if self.rank == 0: progress = tqdm(progress)
        for step in progress:
            batch = self.get_batch()
            while batch is None:
                print('[TRAIN] waiting for batch...'); time.sleep(5)
                batch = self.get_batch()
            if 'end' in batch: break

            tic = time.time()
            loss = self.GRPO_step(self.trainer.engine, batch)
            self.trainer.backward(loss)
            self.trainer.step()

            if self.rank == 0:
                progress.set_description(f"Loss: {loss.item():.6f}")
                print(f'[TRAIN] step: {step},  BATCH shape', batch['inputs'].shape, f'  time: {time.time()-tic:.2f}s')

            if step % self.gen_update_steps == 0:
                distbarrier()
                if self.rank == 0 and self.Q_state_dict.empty():
                    print('[TRAINING PROC] sending latest state_dict ...')
                    state_dict = self.trainer.get_model().state_dict()
                    for _ in range(len(self.gen_device)): self.Q_state_dict.put({'ver':step, 'sd': state_dict})
                    print('[TRAINING PROC] send state_dict ok!')

            if step % self.save_steps == 0:
                distbarrier()
                if self.rank == 0:
                    print('saving model')
                    save_name = f"./step_{step}"
                    save_model(save_name, self.trainer.get_model(), self.tokenizer)