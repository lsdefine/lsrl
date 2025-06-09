import json, os, shutil, re, random, io, time, random
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import torch
from bottle import request
import bottle, threading, queue
from .utils import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn

def get_per_token_logps(model, input_ids):
    logits = model(input_ids).logits  # (B, L, V)
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
    per_token_logps = []
    input_ids = input_ids.to(logits.device)  
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

class RefServer:
    def __init__(self, model_path, host='0.0.0.0', port=59876):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        self.model.eval()
        self.model.requires_grad_(False)
        self.raw_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.app = bottle.Bottle()
        self.host = host
        self.port = port
        
    def run_server(self): 
        @self.app.route('/upload', method='POST')
        def do_upload():
            dd = request.body.read()
            dd = bytes_list_to_list(dd)
            if len(dd) not in (3,4): return b'tensor'
            data = {'base': json.loads(dd[0])} 
            data['inputs'] = bytes_to_tensor(dd[1])
            data['rewards'] = bytes_to_tensor(dd[2])
            if len(dd) == 4: data['gen_logps'] = bytes_to_tensor(dd[3])
            self.raw_queue.put(data)
            return b'tensor'

        @self.app.route('/get', method='GET')
        def do_get():
            if self.result_queue.empty(): return b'empty'
            return self.result_queue.get()
        bottle.run(self.app, host='0.0.0.0', port=59876, server='tornado')

    def start(self):
        threading.Thread(target=self.run_server, daemon=False).start()
    
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        gpu_total = torch.cuda.get_device_properties(0).total_memory
        if param_size > gpu_total * 0.8:
            print('\nAuto patch model to use CPU offloading, only support Qwen2 series now...\n')
            from .patch_for_cpu_offload import patch_qwen2
            patch_qwen2(self.model)
        else:
            self.model.to('cuda')

        while True:
            d = self.raw_queue.get()
            tic = time.time()
            prompt_length = d['base']['plen']
            data = [json.dumps(d['base']).encode(), d['inputs'], d['rewards']]
            if 'end' not in d['base']:
                with torch.inference_mode():
                    per_token_logps = get_per_token_logps(self.model, d['inputs'].to(self.model.device))
                per_token_logps = per_token_logps[:,prompt_length-1:]
                data.append(per_token_logps)
            else: data.append(torch.tensor([0]))
            if 'gen_logps' in d: data.append(d['gen_logps'])
            data = [data[0]] + [tensor_to_bytes(t) for t in data[1:]]
            xdata = make_bytes_list(data)
            self.result_queue.put(xdata)
            print('batch', d['base'], d['inputs'].shape, d['rewards'], f' time: {time.time() - tic:.2f}s')
            if random.random() < 0.1: print(f'raw_queue: {self.raw_queue.qsize()}, result_queue: {self.result_queue.qsize()}')
