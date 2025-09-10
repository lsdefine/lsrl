import json, os, shutil, re, random, io, time, random
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import torch
from bottle import request
import bottle, threading, queue
from .utils import json_to_bytes_list, bytes_list_to_json
from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn

def get_per_token_logps(model, input_ids):
    logits = model(input_ids).logits  # (B, L, V)
    logits = logits[:, :-1, :]        # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = input_ids[:, 1:]      # (B, L-1), exclude the first input ID since we don't have logits for it
    per_token_logps = []
    input_ids = input_ids.to(logits.device)  
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

class RefServer:
    def __init__(self, model_path, host='0.0.0.0', port=59876, force_cpu_offload=False, nlayers_keep_in_gpu=12):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        if model_path is not None:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
            self.model.eval()
            self.model.requires_grad_(False)
        else:
            self.model = None
        self.raw_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.app = bottle.Bottle()
        self.small_bsz = 8
        self.oom_count = 0

    def auto_bsz_infer(self, model, input_ids, pred_func, small_bsz=0):
        sbsz = self.small_bsz if small_bsz <= 0 else small_bsz
        try: 
            rets = [pred_func(model, input_ids[i:i+sbsz])
                    for i in range(0, input_ids.shape[0], sbsz)]
            return torch.cat(rets, dim=0) if len(rets) > 1 else rets[0]
        except torch.cuda.OutOfMemoryError:
            if sbsz == 1: raise Exception('Batch size is 1, cannot reduce further.')
            print('\nOOM, try to reduce batch size...')
            ret = self.auto_bsz_infer(model, input_ids, pred_func, small_bsz=sbsz//2)
            if small_bsz == 0:
                self.oom_count += 1
                if self.oom_count > 3: self.small_bsz, self.oom_count = max(1, sbsz//2), 0
            return ret
        
    def run_server(self): 
        @self.app.route('/upload', method='POST')
        def do_upload():
            dd = request.body.read()
            data = bytes_list_to_json(dd)
            self.raw_queue.put(data)
            return json.dumps({'remain_cnt': self.result_queue.qsize()})

        @self.app.route('/get', method='GET')
        def do_get():
            if self.result_queue.empty(): return b'empty'
            return self.result_queue.get()
        bottle.run(self.app, host=self.host, port=self.port, server='tornado')

    def start(self):
        threading.Thread(target=self.run_server, daemon=False).start()
    
        if self.model is not None:
            param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            gpu_total = torch.cuda.get_device_properties(0).total_memory
            if param_size > gpu_total * 0.8 or self.force_cpu_offload:
                print('\nPatch model to use CPU offloading, only support Qwen2 series now...\n')
                from .patch_for_cpu_offload import patch_qwen2
                patch_qwen2(self.model, nlayers_keep_in_gpu=self.nlayers_keep_in_gpu)
            else:
                self.model.to('cuda')
            device = self.model.device

        while True:
            d = self.raw_queue.get()
            tic = time.time()
            plen = d.get('plen', 0)
            if 'end' not in d:
                if self.model is not None and 'inputs' in d:
                    with torch.inference_mode():
                        logps = self.auto_bsz_infer(self.model, d['inputs'].to(device), get_per_token_logps)
                    d['refs'] = logps[:,plen-1:].cpu()
                    print('batch', d['inputs'].shape, d['rewards'], f' time: {time.time() - tic:.2f}s')
            d['remain_cnt'] = self.result_queue.qsize()
            self.result_queue.put(json_to_bytes_list(d))
            if random.random() < 0.1: print(f'raw_queue: {self.raw_queue.qsize()}, result_queue: {self.result_queue.qsize()}')
