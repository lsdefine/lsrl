import torch 
import torch.distributed as dist
import time, requests
from tqdm import tqdm
from .utils import json_to_bytes_list, bytes_list_to_json, save_model, enable_gradient_checkpointing

def apply_no_vllm_patch(lsrl_instance):    
    from .lsrl import distbarrier, get_world_size

    def _generation_mode(self):
        self.trainer.get_model().gradient_checkpointing_disable()
        if self.gen_iterator is None:
            self.gen_iterator = self.generation_loop()
        try:
            for _ in range(5): next(self.gen_iterator) 
        except StopIteration:
            self.gen_iterator = None
        self.trainer.get_model().gradient_checkpointing_enable()

    def _generate(self, model, prompts):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, do_sample=True,
                temperature=self.gen_temperature,
                max_new_tokens=self.gen_max_tokens,
                num_return_sequences=self.rollout_num,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        answers = []
        input_length = inputs['input_ids'].shape[1]
        for seq in outputs:
            text = self.tokenizer.decode(seq[input_length:], skip_special_tokens=True)
            answers.append({'text': text, 'token_ids': seq[input_length:].tolist()})
        return answers

    def _generation_loop(self):
        def gen_samples(items):
            gen_prompts = [self.rollout_prompt_fn(x) for x in items]
            answers = self.generate(self.trainer.get_model(), gen_prompts)
            policy_prompts = [self.policy_prompt_fn(x) for x in items]
            return {'prompts': policy_prompts, 'answers': answers}
        
        def gen_rewards(samples, items):
            rewards = []
            for i, ans in enumerate(samples['answers']):
                reward = {}
                for reward_fn in self.reward_fns:
                    reward[reward_fn.__name__] = reward_fn(ans['text'], items[i//self.rollout_num])
                reward['total'] = sum(reward.values())
                rewards.append(reward)
            return rewards
        
        from torch.nn.utils.rnn import pad_sequence
        def make_batch_inputs(prompt_ids, sub_ans_ids):
            plen = prompt_ids.shape[1]
            tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
            output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=self.tokenizer.pad_token_id) 
            Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
            merged_ids = torch.cat([Qrep, output_ids], dim=1)
            return merged_ids
        
        def get_per_token_logps(logits, input_ids):
            per_token_logps = [] 
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        model = self.trainer.get_model()
        def compute_gen_logps(data):
            if self.compute_gen_logps:
                plen = data['plen']
                with torch.inference_mode():
                    mids = data['inputs'].to(model.device)
                    gen_logps = get_per_token_logps(model(mids).logits[:, :-1, :], mids[:, 1:])
                data['gen_logps'] = gen_logps[:,plen-1:].cpu()
        
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
                    if self.lsrl.skip_zero_groups and curr_rewards.max() - curr_rewards.min() < 1e-4: continue
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

        rn = self.rollout_num 
        from torch.nn.utils.rnn import pad_sequence
        
        world_size = get_world_size()
        gen_rank = self.rank
        RP = RolloutProcessorBase(self)

        Q_data = []
        for epoch in range(self.epochs):
            items = list(self.train_data)
            for i in range(0, len(items), self.gen_batch_size):
                batch = items[i:i+self.gen_batch_size]
                Q_data.append({'batch': batch})

        for it, item in enumerate(Q_data):
            if it % world_size != gen_rank: continue
            tic = time.time()
            questions = item['batch']
            rr = RP.run(questions)
            samples = rr['samples']
            group_avg_rewards = rr['group_avg_rewards']
            if gen_rank == 0 and self.genlog_recorder:
                self.genlog_recorder.log(it, items[0], samples['answers'][:rn], samples['rewards'][:rn])
            print(f'[GEN {gen_rank}]  time: {time.time()-tic:.2f}s    ', f'avg_rewards: {",".join(group_avg_rewards)}' )
            self.call_hook('after_rollout', samples)
            yield
        
        requests.post(f"{self.ref_server}/upload", data=json_to_bytes_list({'end':1}))            

                
    def _train(self):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.gen_iterator = None

        def do_save(step):
            if self.rank == 0:
                print('saving model')
                save_name = f"./step_{step}"
                save_model(save_name, self.trainer.get_model(), self.tokenizer)        

        self.device = self.trainer.device
        progress = range(1, self.all_steps+1)
        if self.rank == 0: progress = tqdm(progress)

        for step in progress:
            batch = self.get_batch()
            while batch is None:
                self.generation_mode()
                batch = self.get_batch()
            if 'end' in batch: break
            
            tic = time.time()
            loss = self.GRPO_step(self.trainer.engine, batch)
            self.trainer.backward(loss)
            self.trainer.step()

            if self.rank == 0:
                print(f'[TRAIN] step: {step},  BATCH shape', batch['inputs'].shape, f'  time: {time.time()-tic:.2f}s')
                progress.set_description(f"Loss: {loss.item():.6f}")

            if step % self.save_steps == 0:
                distbarrier()
                do_save(step)
                
        if self.skip_zero_groups:
            print('\n\nSome groups had same rewards and skipped, so the training steps may be less than expected.\n')

        distbarrier()
        if step % self.gen_update_steps != 0: do_save(step)
    
    lsrl_instance.generation_mode = _generation_mode.__get__(lsrl_instance, type(lsrl_instance))
    lsrl_instance.generation_loop = _generation_loop.__get__(lsrl_instance, type(lsrl_instance))
    lsrl_instance.generate = _generate.__get__(lsrl_instance, type(lsrl_instance))
    lsrl_instance.train = _train.__get__(lsrl_instance, type(lsrl_instance))
    
    print("\n🔄  Native generation mode activated!\n")