import os, time, sys
os.environ['OMP_NUM_THREADS'] = '32'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

model_path = "/data2/Qwen/Qwen2.5-14B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_path, 
            torch_dtype=torch.bfloat16).to('cuda')

model.train()
model.gradient_checkpointing_enable()
# important! unless seq_len is very short (<= 2000),

# =============================================================================
# Configuration Selection: python sft_14b_80g_1gpu.py [gradoffload|]
# Config 1: Memory-efficient (seq_len=18000, grad_offload=True) - slower, longer sequences
# Config 2: Speed-optimized (seq_len=8000, grad_offload=False) - faster, shorter sequences
# =============================================================================
if len(sys.argv) > 1 and sys.argv[1] == "gradoffload":
    seq_len, grad_offload = 18000, True
else:
    seq_len, grad_offload = 8000, False  
print(f"Config: grad_offload={grad_offload}, support seq_len={seq_len}")

from lsrl import CPUAdamW
opt = CPUAdamW(model.parameters(), lr=1e-5, accum_steps=4,
               weight_decay=0.01, eps=1e-8, 
               grad_offload=grad_offload)

for step in range(1, 7):
    batch = torch.randint(1, 10, (1, seq_len)).to(model.device)
    print('\nInput shape:', batch.shape)
    tic = time.time()
    loss = model(batch, labels=batch, use_cache=False).loss
    print(f"Forward GPU memory: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    loss.backward()
    print(f"Backward GPU memory: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    print('step: ', step, 'loss: %.4f' % loss.item())
    print('step time: ', end='')
    if opt.step(): print('update parameters! ')
    print('%.2fs' % (time.time()-tic))
    print(f"Step GPU memory: {torch.cuda.memory_reserved()/1e9:.2f}GB")