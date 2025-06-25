import os, time, sys
os.environ['OMP_NUM_THREADS'] = '32'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

if 'RANK' in os.environ:
    print("\nError: This script must be run with python, not torchrun")
    print(f"Example usage: CUDA_VISIBLE_DEVICES=3,4,5 python {__file__}")
    sys.exit(1)

model_path = "/data/Qwen/Qwen2.5-14B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
model.train()
model.gradient_checkpointing_enable()

from lsrl import CPUAdamW, patch_qwen2_for_multi_gpus

device_count = torch.cuda.device_count()
devices = list(range(device_count))
patch_qwen2_for_multi_gpus(model, devices=devices, patch_lm_head=True)

opt = CPUAdamW(model.parameters(), lr=1e-5, accum_steps=4,
               weight_decay=0.01, eps=1e-8, 
               grad_offload=True)

for step in range(1, 8):
    batch = torch.randint(1, 10, (2, 8500)).to(model.device)
    print('\nInput shape:', batch.shape)
    tic = time.time()
    loss = model(batch, labels=batch, use_cache=False).loss
    loss.backward()
    print('step: ', step, 'loss: %.4f' % loss.item())
    print('step time: ', end='')
    if opt.step(): print('update parameters! ')
    print('%.2fs' % (time.time()-tic))

print('\nNo OOM, Good!')