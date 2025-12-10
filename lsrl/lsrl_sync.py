
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct, types, queue
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime

from tqdm import tqdm
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils import save_model
from .lsrl import LSCPUTrainer, get_world_size, distbarrier, chunk_list, GenLogRecorder
from .lsrl import torchN, NDEV, NBACKEND, NENV_VDEV

def create_soft_len_penalty_tok(DAPO_kwargs):
    def soft_len_penalty_tok(ans, _):
        cache_max_length = DAPO_kwargs.get('cache_max_length', 1024)
        soft_max_length = DAPO_kwargs.get('soft_max_length', 2048)
        leny = len(ans['token_ids'])
        if leny < cache_max_length: return 0
        if leny > soft_max_length: return -1
        return - (leny - cache_max_length) / (soft_max_length - cache_max_length)
    return soft_len_penalty_tok

class SyncLSRL:
    def __init__(self, model_path, epochs=1, rollout_num=8, train_data=None, trainer='LSCPU',
                 train_batch_size=2, save_steps=40, gen_batch_size=4,
                 beta=0.04, clip_param=0.2, compute_gen_logps=True,
                 gen_max_tokens=4096, gen_temperature=0.9, genlog_filename=None,
                 skip_zero_groups=False, DAPO_kwargs=None, algorithm='GRPO', update_times_per_step=2,
                 vllm_kwargs=None, swanlab=None, use_vllm_v1=False, dtype=torch.bfloat16,
                 **kwargs):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.rollout_num = rollout_num
        self.train_data = train_data
        self.gen_batch_size = gen_batch_size
        self.reward_fns = []
        self.epochs = epochs
        self.all_steps = epochs * len(train_data) // (get_world_size() * gen_batch_size)
        self.update_times_per_step = update_times_per_step
        self.DAPO_kwargs = DAPO_kwargs or {}
        self.algorithm = algorithm
        if DAPO_kwargs is not None: self.algorithm = 'DAPO'
        self.swanlab = swanlab
        self.vllm_kwargs = vllm_kwargs or {}
        self.use_vllm_v1 = use_vllm_v1

        accum_steps = 99999
        assert gen_batch_size % update_times_per_step == 0, "gen_batch_size must be divisible by update_times_per_step"

        self._hooks = {}

        if train_batch_size > rollout_num:
            assert train_batch_size % rollout_num == 0, "train_batch_size must be divisible by rollout_num"
            self.num_mix_forward_batches = train_batch_size // rollout_num
            self.train_batch_size = rollout_num
            raise Exception("mix_forward_batches does not faster, use train_batch_size == rollnum instead")
        else:
            assert rollout_num % train_batch_size == 0, "rollout_num must be divisible by train_batch_size"
            self.train_batch_size = train_batch_size
            self.num_mix_forward_batches = 1

        self.skip_zero_groups = skip_zero_groups
        self.save_steps = save_steps
        self.compute_gen_logps = compute_gen_logps
        self.generate_fn = None
        self.gen_max_tokens = gen_max_tokens
        self.gen_temperature = gen_temperature
        self.beta = beta
        self.clip_param = clip_param
        self.genlog_recorder = GenLogRecorder(genlog_filename) if genlog_filename else None

        assert gen_batch_size % get_world_size() == 0, "gen_batch_size must be divisible by world_size"
      
        if self.algorithm == 'DAPO':
            print('\nUsing DAPO algorithm for training...\n')
            print('Available DAPO kwargs: cache_max_length, soft_max_length, hard_max_length, clip_param_high\n')
            self.RL_step = self.DAPO_step
        elif self.algorithm == 'GSPO':
            self.RL_step = self.GSPO_step
        else:
            self.RL_step = self.GRPO_step

        if trainer == 'LSCPU':
            self.trainer = LSCPUTrainer(model_path, accum_steps=accum_steps, dtype=dtype, **kwargs)
        else:
            raise ValueError("Unsupported trainer type. Use 'LSCPU'")
        
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.device = self.trainer.device
    
    def set_hook(self, name, func): self._hooks[name] = func
    def set_hooks(self, **hooks): self._hooks.update(hooks)
    def call_hook(self, name, *args, **kwargs):
        if name in self._hooks: return self._hooks[name](*args, **kwargs)
        return None

    def add_reward(self, reward_fn):
        self.reward_fns.append(reward_fn)

    def set_rollout_prompt_fn(self, user_fn): self._rollout_prompt_fn = user_fn
    def set_policy_prompt_fn(self, user_fn): self._policy_prompt_fn = user_fn
    def rollout_prompt_fn(self, item): return self._rollout_prompt_fn(self, item)
    def policy_prompt_fn(self, item): return self._policy_prompt_fn(self, item)

    def get_per_token_logps(self, logits, input_ids):
        per_token_logps = [] 
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def _forward_base_logits(self, model, batch, ref=True):
        prompt_length = batch['plen']
        inputs = batch['inputs'].to(self.device)
        advantages = batch['rewards'].to(self.device).unsqueeze(1)
        if '#computed_logits' not in batch:
            logits = model(inputs, use_cache=False).logits
            logits = logits[:, :-1, :]  
        else:
            logits = batch['#computed_logits'].to(self.device)
        input_ids = inputs[:, 1:]  
        per_token_logps = self.get_per_token_logps(logits, input_ids)
        per_token_logps = per_token_logps[:,prompt_length-1:]
        completion_mask = (inputs[:, prompt_length:] != self.tokenizer.pad_token_id).int()
        if ref and 'refs' in batch:
            ref_per_token_logps = batch['refs'].to(per_token_logps.device)
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        else: per_token_kl = None
        return {
            'per_token_logps': per_token_logps,
            'per_token_kl': per_token_kl,
            'advantages': advantages,
            'completion_mask': completion_mask
        }

    def GRPO_step(self, model, batch):
        r = self._forward_base_logits(model, batch)
        per_token_logps, advantages, per_token_kl, completion_mask = \
            r['per_token_logps'], r['advantages'], r['per_token_kl'], r['completion_mask']
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
    
    def DAPO_step(self, model, batch, return_divx=False):        
        r = self._forward_base_logits(model, batch, ref=False)
        per_token_logps, advantages, completion_mask = r['per_token_logps'], r['advantages'], r['completion_mask']
        if 'gen_logps' in batch:
            clip_param_high = self.DAPO_kwargs.get('clip_param_high', 0.28)
            ratio = torch.exp(per_token_logps - batch['gen_logps'].to(self.device))
            clipped_ratio = torch.clamp(ratio, 1-self.clip_param, 1+clip_param_high)
            per_token_loss = - torch.min(ratio * advantages, clipped_ratio * advantages)
        else: 
            raise Exception("DAPO requires gen_logps in batch")
        valid_tokens = completion_mask.sum()
        loss = (per_token_loss * completion_mask).sum() / torch.clamp(valid_tokens, min=1)
        if return_divx: return loss, valid_tokens
        return loss
    
    def GSPO_step(self, model, batch):        
        r = self._forward_base_logits(model, batch)
        per_token_logps, advantages, per_token_kl, completion_mask = \
            r['per_token_logps'], r['advantages'], r['per_token_kl'], r['completion_mask']
        if 'gen_logps' in batch:
            seq_length = completion_mask.sum(dim=1, keepdim=True)
            si = per_token_logps - batch['gen_logps'].to(self.device)
            si = si * completion_mask
            si = torch.exp(si.sum(dim=1, keepdim=True) / seq_length)
            clipped_ratio = torch.clamp(si, 1-self.clip_param, 1+self.clip_param)
            per_token_loss = - torch.min(si * advantages, clipped_ratio * advantages)
            if self.beta > 0 and per_token_kl is not None:
                kl = per_token_kl * completion_mask
                kl = kl.sum(dim=1, keepdim=True) / seq_length
                per_token_loss += self.beta * kl
        else: 
            raise Exception("GSPO requires gen_logps in batch")
        loss = per_token_loss.mean()
        return loss
       
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('trainer', None)
        return state

    def start_gen_worker(self):
        print('\nSTART vLLM generation...\n')
        ctx = mp.get_context('spawn')
        self.Q_data = ctx.Queue()
        self.Q_results = ctx.Queue()
        logical = torchN.current_device()  
        physical = (lambda m, i: i if not m else m[i])([int(x) for x in os.environ.get(NENV_VDEV, "").split(",") if x.strip()] or None, logical)
        p = ctx.Process(target=self.gen_worker, args=(self.Q_data, self.Q_results, physical, self.rank))
        p.start()

    def gen_worker(self, Q_data, Q_results, gen_device, gen_rank=0):
        os.environ[NENV_VDEV] = f'{gen_device}'
        if self.use_vllm_v1: os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        else: os.environ["VLLM_USE_V1"] = "0"
        cleanup_keys = [  
            'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'LOCAL_RANK',  
            'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_NAME',   
            'GROUP_WORLD_SIZE', 'ROLE_WORLD_SIZE',  
            'TORCHELASTIC_RESTART_COUNT', 'TORCHELASTIC_MAX_RESTARTS',  
            'TORCHELASTIC_RUN_ID', 'TORCHELASTIC_USE_AGENT_STORE',  
            'TORCHELASTIC_ERROR_FILE',  
            'TORCH_NCCL_ASYNC_ERROR_HANDLING', 'HCCL_COMM_ID',
            'NCCL_COMM_ID', 'NCCL_DEBUG', 'NCCL_SOCKET_IFNAME',  
        ]  
        for key in cleanup_keys: os.environ.pop(key, None)
        
        torchN.set_device(0)
        print(f"[GEN {gen_rank}] Generation worker process uses GPU {gen_device}")
        print(f"[GEN {gen_rank}] {NENV_VDEV}: {os.environ.get(NENV_VDEV)}")
        print(f"[GEN {gen_rank}] PID: {os.getpid()}")
        print(f'[GEN {gen_rank}]', os.environ)

        from vllm import LLM, SamplingParams
        default_kwargs = {"enable_chunked_prefill": True, "gpu_memory_utilization": 0.7,
                          "enable_sleep_mode": True, "max_num_seqs": 16, "max_model_len": 12000}
        final_kwargs = {**default_kwargs, **self.vllm_kwargs}
        vllm_gen = LLM(model=self.model_path, **final_kwargs)
        self.vllm_gen = vllm_gen
        gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

        if self.algorithm == 'DAPO':
            soft_len_penalty_fn = create_soft_len_penalty_tok(self.DAPO_kwargs)
            soft_len_penalty_fn.__name__ = 'soft_len_penalty_tok'
            self.add_reward(soft_len_penalty_fn)

        def gen_samples(items):
            gen_prompts = [self.rollout_prompt_fn(x) for x in items]
            answers = self.generate(vllm_gen, gen_prompts)
            policy_prompts = [self.policy_prompt_fn(x) for x in items]
            return {'prompts': policy_prompts, 'answers': answers}
        
        if hasattr(self, 'gen_samples'): gen_samples = lambda x: self.gen_samples(self, x)
        
        rn = self.rollout_num
        from torch.nn.utils.rnn import pad_sequence

        def make_batch_inputs(prompt_ids, sub_ans_ids):
            plen = prompt_ids.shape[1]
            tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
            output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=self.tokenizer.pad_token_id) 
            Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
            merged_ids = torch.cat([Qrep, output_ids], dim=1)
            return merged_ids
        
        while True:
            try:
                xdata = Q_data.get()
                if 'state_dict' in xdata:
                    self.vllm_gen.wake_up(tags=["weights"])
                    state_dict = xdata['state_dict']
                    llm_model = self.vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights(state_dict.items())
                    print(f'[GEN {gen_rank}] Updated weights for generation model.')
                    self.vllm_gen.wake_up(tags=["kv_cache"])
                    continue
                if 'exit' in xdata: break

                batch = xdata['batch']
                if self.rank == 0: print(f'\nEach worker rollouts {len(batch)} samples ...\n')
                rollouts = []
                samples = gen_samples(batch)
                groups = {'answers': chunk_list(samples['answers'], self.rollout_num)}
                for prompt, ganswers in zip(samples['prompts'], groups['answers']):
                    if len(ganswers) == 0: continue 
                    prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
                    curr_ans_ids = [x['token_ids'] for x in ganswers]
                    data = {'plen': prompt_ids.shape[1],
                            'inputs': make_batch_inputs(prompt_ids, curr_ans_ids)}
                    if self.compute_gen_logps:
                        plen = data['plen']
                        zz = self.vllm_gen.generate(prompt_token_ids=data['inputs'].tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                        zz = [xx.prompt_logprobs[plen:] for xx in zz]
                        gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                        data['gen_logps'] = gen_logps
                    rollouts.append(data)

                if 'rewards' not in samples: 
                    samples['rewards'] = rewards = self.gen_rewards(samples, batch)
                else: rewards = samples['rewards']
                for data in rollouts:
                    curr_rewards = torch.tensor([r['total'] for r in rewards[:rn]], dtype=torch.float)
                    rewards = rewards[rn:]
                    data['rewards'] = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)

                if self.beta > 0:
                    llm_model = self.vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights(self.ref_state_cpu.items())
                    for data in rollouts:
                        plen = data['plen']
                        zz = self.vllm_gen.generate(prompt_token_ids=data['inputs'].tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                        zz = [xx.prompt_logprobs[plen:] for xx in zz]
                        ref_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                        data['refs'] = ref_logps

                if gen_rank == 0 and self.genlog_recorder:
                    self.genlog_recorder.log(xdata['step'], batch[0], samples['answers'][:rn], samples['rewards'][:rn])

                self.vllm_gen.sleep(level=2)
                Q_results.put({'rollouts':rollouts, 'samples':samples})

            except Exception as e:
                import traceback
                print(f'[GEN {gen_rank}] Exception: {e}, {traceback.format_exc()}')
        del self.vllm_gen
        if dist.is_initialized(): dist.destroy_process_group()
        #print('\nwhen vLLM exits, it may raise some CUDA errors, this is vLLM\'s problem, please ignore them.')
        time.sleep(5)

    def gen_rewards(self, samples, items):
        rewards = []
        for i, ans in enumerate(samples['answers']):
            reward = {}
            for reward_fn in self.reward_fns:
                name = reward_fn.__name__
                if name.endswith('_tok'):
                    reward[name] = reward_fn(ans, items[i//self.rollout_num])
                else:
                    reward[name] = reward_fn(ans['text'], items[i//self.rollout_num])
            reward['total'] = sum(reward.values())
            rewards.append(reward)
        return rewards
    
    
    def generate(self, vllm_gen, prompts):
        from vllm import SamplingParams
        sampling_params = SamplingParams(n=self.rollout_num, temperature=self.gen_temperature, 
                                        max_tokens=self.gen_max_tokens)
        voutputs = vllm_gen.generate(prompts, sampling_params, use_tqdm=(self.rank==0))
        answers = []
        for v in voutputs:
            for z in v.outputs: 
                answers.append({'text':z.text, 'token_ids': z.token_ids})
        assert len(answers) == len(prompts) * self.rollout_num
        return answers

    def rollout_process(self, batch, step):
        self.Q_data.put({'batch': batch, 'step': step})
        rollouts = self.Q_results.get()
        return rollouts

    def train_process(self, rollouts, step):
        self.trainer.model.to(self.device)
        rollout_batches = chunk_list(rollouts, len(rollouts)//self.update_times_per_step)
        if self.rank == 0: print("In step update times:", len(rollout_batches))
        
        def split_batch(batch, subsz):
            bsz = batch['inputs'].shape[0]
            sub_batches = [{} for _ in range(bsz//subsz)]
            for i in range(0, bsz//subsz):
                l, r = i*subsz, (i+1)*subsz
                for k, v in batch.items():
                    if torch.is_tensor(v): sub_batches[i][k] = v[l:r]
                    else: sub_batches[i][k] = v
            for batch in sub_batches:
                real_length = (batch['inputs'] != self.tokenizer.pad_token_id).sum(dim=1).max().item()
                cut_length = batch['inputs'].shape[1] - real_length
                if cut_length > 0:
                    for k in ['inputs', 'gen_logps', 'refs']:
                        if k in batch: batch[k] = batch[k][:, :-cut_length]
            return sub_batches                
        
        def change_model_checkpoint_ratio(model, ratio):
            layers = model.model.layers
            total_layers = len(layers)
            start_idx = total_layers - int(total_layers * ratio)
            for i, layer in enumerate(layers):
                layer.gradient_checkpointing = i >= start_idx
    
        for batches in rollout_batches:
            if self.rank == 0: print([x['inputs'].shape for x in batches])
            plans = []
            for batch in batches:
                plans.extend(split_batch(batch, self.train_batch_size))

            divx = 0
            progress = tqdm(plans) if self.rank == 0 else plans
            for i, batch in enumerate(progress):
                sbsz = batch['inputs'].shape[0]
                if self.rank == 0: progress.set_description(str(batch['inputs'].shape))
                loss = self.RL_step(self.trainer.engine, batch) * sbsz
                divx += sbsz
                self.trainer.backward(loss)
                if self.rank == 0: progress.set_postfix({'loss': f'{loss.item()/sbsz:.4f}'})
                if i == len(plans) - 1: self.trainer.step(force_update=True, div_num=divx)             
                else: self.trainer.step()

        self.trainer.model.to('cpu')
        torchN.synchronize()
        torchN.empty_cache()
        self.Q_data.put({'state_dict': self.trainer.get_model().state_dict()})

    def dry_check_rollout(self, gen_batch_size=2, rollout_num=3):
        self.world_size = get_world_size()
        assert self.world_size == 1, "dry_check_rollout only needs single GPU"
        self.trainer.model.to('cpu')
        torchN.synchronize()
        torchN.empty_cache()
        self.gen_batch_size = gen_batch_size
        self.rollout_num = rollout_num
        self.start_gen_worker()

        train_datas = list(self.train_data) * self.epochs
        batches = chunk_list(train_datas, self.gen_batch_size)[:-1]
        total_steps = len(batches)

        for step in range(1, len(batches)+1):
            batch_id = step - 1
            tic = time.time()
            mini_batch = [x for i, x in enumerate(batches[batch_id]) if i % self.world_size == self.rank]
            rollout_ret = self.rollout_process(mini_batch, step)
            rollouts, samples = rollout_ret['rollouts'], rollout_ret['samples']
            if self.rank == 0: print(f"[MAIN] Step {step}/{total_steps} Rollout time: {time.time() - tic:.2f}s")
            for k, v in samples.items(): 
                print('------------\n', k)
                for x in v: print(x, '\n')
            break

        self.Q_data.put({'exit':1})
        if dist.is_initialized(): dist.destroy_process_group()
        sys.exit(0)


    def train(self):
        if self.swanlab: import swanlab

        def do_save(step):
            if self.rank == 0:
                print('saving model')
                save_name = f"./step_{step}"
                save_model(save_name, self.trainer.get_model(), self.tokenizer)        
        self.device = self.trainer.device
        self.world_size = get_world_size()

        self.trainer.model.to('cpu')
        torchN.synchronize()
        torchN.empty_cache()

        if self.beta > 0:
            self.ref_state_cpu = {k:v.detach().clone() for k, v in self.trainer.model.state_dict().items()}
        self.start_gen_worker()

        train_datas = list(self.train_data) * self.epochs
        batches = chunk_list(train_datas, self.gen_batch_size)[:-1]
        total_steps = len(batches)

        if self.rank == 0: print(f"\n[MAIN] Total training steps: {total_steps}\n")
        
        for step in range(1, len(batches)+1):
            batch_id = step - 1

            tic = time.time()
            mini_batch = [x for i, x in enumerate(batches[batch_id]) if i % self.world_size == self.rank]
            rollout_ret = self.rollout_process(mini_batch, step)
            rollouts, samples = rollout_ret['rollouts'], rollout_ret['samples']
            if self.rank == 0: print(f"[MAIN] Step {step}/{total_steps} Rollout time: {time.time() - tic:.2f}s")

            if self.swanlab and self.rank == 0: 
                swanlab.log({'avg_reward': np.mean([x['total'] for x in samples['rewards']]),
                            'avg_ans_len': np.mean([len(x['token_ids']) for x in samples['answers']]),
                            "rollout_time": time.time() - tic})

            self.train_process(rollouts, step)
            distbarrier()

            if self.rank == 0: print(f"[MAIN] Step {step}/{total_steps} Step time: {time.time() - tic:.2f}s")
            
            if self.swanlab and self.rank == 0: 
                swanlab.log({"step_time": time.time() - tic})

            if step % self.save_steps == 0:
                distbarrier()
                do_save(step)
                
        if self.skip_zero_groups:
            print('\n\nSome groups had same rewards and skipped, so the training steps may be less than expected.\n')

        distbarrier()
        # Final save after training
        if step % self.gen_update_steps != 0: do_save(step)
