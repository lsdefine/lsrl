import os, time, sys
os.environ['OMP_NUM_THREADS'] = '32'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

if 'RANK' in os.environ:
    torch.distributed.init_process_group(backend='nccl')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

model_path = "/data2/Qwen/Qwen2.5-14B-Instruct"

# =============================================================================
# Configuration Selection: python sft_14b_80g_1gpu.py [gradoffload|]
# Config 1: Memory-efficient (seq_len=18000, grad_offload=True) - slower, longer sequences
# Config 2: Speed-optimized (seq_len=7500, grad_offload=False) - faster, shorter sequences
# =============================================================================
if len(sys.argv) > 1 and sys.argv[1] == "gradoffload":
    seq_len, grad_offload = 18000, True
else:
    seq_len, grad_offload = 7500, False  
print(f"\nConfig: seq_len={seq_len}, grad_offload={grad_offload}")

from lsrl import LSCPUTrainer
engine = LSCPUTrainer(model_path, lr=1e-5, accum_steps=4,
               grad_offload=grad_offload)
model = engine.model

for step in range(1, 7):
    batch = torch.randint(1, 10, (1, seq_len)).to(model.device)
    print('\nInput shape:', batch.shape)
    tic = time.time()
    loss = model(batch, labels=batch, use_cache=False).loss
    print(f"Forward GPU memory: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    engine.backward(loss)
    print(f"Backward GPU memory: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    print('step: ', step, 'loss: %.4f' % loss.item())
    print('step time: ', end='')
    if engine.step(): print('update parameters! ')
    print('%.2fs' % (time.time()-tic))
    print(f"Step GPU memory: {torch.cuda.memory_reserved()/1e9:.2f}GB")