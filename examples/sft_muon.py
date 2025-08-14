import os, time, sys, random, re
os.environ['OMP_NUM_THREADS'] = '32'
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

if 'RANK' in os.environ:
    # torchrun mode
    torch.distributed.init_process_group(backend='nccl')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
else:
    # python mode
    device = 'cuda'
    local_rank = 0

model_path = "/data2/Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, 
            torch_dtype=torch.bfloat16).to(device)

model.train()
model.gradient_checkpointing_enable()

def gen():
    a, b = random.randint(1, 10**4), random.randint(1, 10**4)
    s, r = f"请计算：{a}+{b}=⚡", a+b
    rr = '\\boxed{' + '❤️'.join(str(r)) + '}' + tokenizer.eos_token + '<|endoftext|>'
    return s, rr

testset = [gen() for _ in range(10)]

def test():
    correct = 0
    with torch.inference_mode():
        for i, (s, r) in enumerate(tqdm(testset)):
            inp = tokenizer(s, return_tensors='pt').input_ids.to(model.device)
            out = model.generate(inp, max_new_tokens=30, do_sample=True, temperature=0.01,
                                eos_token_id=tokenizer.eos_token_id)
            ans = tokenizer.decode(out[0], skip_special_tokens=True)
            inbox = re.search('boxed\\{(.*)\\}', ans)
            rinbox = re.search('boxed\\{(.*)\\}', r)
            if i < 5: print(s, '=>', ans, ', ', r, rinbox.group(1))
            if inbox is not None and inbox.group(1) == rinbox.group(1): correct += 1
    print(f'Accuracy: {correct}/{len(testset)} = {correct/len(testset):.4f}')

def rprint(*args, **kwargs): 
    if local_rank == 0: print(*args, **kwargs)

from lsrl import CPUMuon
opt = CPUMuon(model.named_parameters(), lr=5e-5, accum_steps=2, weight_decay=0.1, eps=1e-8, 
               ns_step=3, muon_lr=0.05, ns_dtype=torch.bfloat16, verbose=True,
               grad_offload=False)
# muon_lr can be large

for step in range(1, 101):
    batch_qas = [gen() for _ in range(128)]
    batch, labels = [], []
    for q, a in batch_qas:
        q_ids = tokenizer.encode(q)
        a_ids = tokenizer.encode(a)
        batch.append(q_ids + a_ids)
        labels.append([-100] * len(q_ids) + a_ids)
    maxlen = max(len(x) for x in batch)
    batch = [x + [tokenizer.eos_token_id] * (maxlen-len(x))  for x in batch]
    labels = [x + [-100] * (maxlen-len(x)) for x in labels]
    batch = torch.tensor(batch).to(model.device)
    labels = torch.tensor(labels).to(model.device)

    rprint('\nInput shape:', batch.shape)
    tic = time.time()
    loss = model(batch, labels=labels, use_cache=False).loss
    loss.backward()
    rprint('step:', step, 'loss: %.4f' % loss.item())
    rprint('step time: ', end='')

    if opt.step(): rprint('update parameters! ')

    rprint('%.2fs' % (time.time()-tic))
    if local_rank == 0 and step % 20 == 0: test()