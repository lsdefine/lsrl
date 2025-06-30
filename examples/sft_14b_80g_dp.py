import os, time, sys
os.environ['OMP_NUM_THREADS'] = '32'
from transformers import AutoModelForCausalLM
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

if 'RANK' not in os.environ:
    print("\nError: This script must be run with torchrun, not directly with python")
    print(f"Example usage: CUDA_VISIBLE_DEVICES=3,4,5 torchrun --nproc_per_node=3 {__file__}")
    sys.exit(1)

torch.distributed.init_process_group(backend='nccl')
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)

model_path = "/data2/Qwen/Qwen2.5-14B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_path, 
            torch_dtype=torch.bfloat16).to(device)
model.train()
model.gradient_checkpointing_enable()
# important! unless seq_len is very short (<= 2000),

# =============================================================================
# Configuration Selection: python sft_14b_80g_dp.py [gradoffload|]
# Config 1: Memory-efficient (seq_len=18000, grad_offload=True) - slower, longer sequences
# Config 2: Speed-optimized (seq_len=7000, grad_offload=False) - faster, shorter sequences
# =============================================================================
if len(sys.argv) > 1 and sys.argv[1] == "gradoffload":
    seq_len, grad_offload = 18000, True
else:
    seq_len, grad_offload = 7000, False  
print(f"Config: Speed-optimized (grad_offload={grad_offload}), support seq_len={seq_len}")

if grad_offload:
    print('\nEvery step will sync, it is the cost for the long seq_len.\n')
else:
    print('\nRank 0 proc need to do some work, so it starts a bit later.\n')

from lsrl import DistributedCPUAdamW, save_model
opt = DistributedCPUAdamW(model.parameters(), lr=1e-5, 
               accum_steps=4, weight_decay=0.01, eps=1e-8, 
               grad_offload=grad_offload)

for step in range(1, 7):
    batch = torch.randint(1, 10, (1, seq_len)).to(model.device)
    print('\nInput shape:', batch.shape)
    tic = time.time()
    loss = model(batch, labels=batch, use_cache=False).loss
    loss.backward()
    print('step: ', step, 'loss: %.4f' % loss.item())
    print('step time: ', end='')
    if opt.step(): print('update parameters! ')
    print('%.2fs' % (time.time()-tic))

if torch.distributed.get_rank() == 0:
    save_model("./sft_14b_80g_dp", model, tokenizer=None)