# benchmark.py
import os, time, sys
os.environ['OMP_NUM_THREADS'] = '32'
import argparse
import time
import torch
from transformers import AutoModelForCausalLM
import json
from datetime import datetime

# USAGE:
# CUDA_VISIBLE_DEVICES=4 python benchmark.py --method=lsrl --seq_len=7500
# CUDA_VISIBLE_DEVICES=4,7 torchrun --nproc_per_node=2 benchmark.py --method=lsrl --seq_len=7500
# CUDA_VISIBLE_DEVICES=4,7 torchrun --nproc_per_node=2 benchmark.py --method=lsrl --seq_len=18000 --grad_offload
# CUDA_VISIBLE_DEVICES=4 deepspeed benchmark.py --method=deepspeed --seq_len=15000 --ds_zero=1
# CUDA_VISIBLE_DEVICES=4,7 deepspeed benchmark.py --method=deepspeed --seq_len=15000 --ds_zero=2

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['lsrl', 'deepspeed'], default='lsrl')
    parser.add_argument('--seq_len', type=int, default=8000)
    parser.add_argument('--grad_offload', action='store_true')
    parser.add_argument('--ds_zero', type=int, default=1)
    parser.add_argument('--model_path', default="/data2/Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()

def get_rank(): return int(os.environ['RANK'])  if 'RANK' in os.environ else 0
def is_main_process(): return get_rank() == 0
def get_world_size(): return int(os.environ.get('WORLD_SIZE', 1))

def get_ds_config(args):
    return {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 4,
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-5}},
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": args.ds_zero,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "offload_optimizer": {"device": "cpu"}
        }
    }

def setup_trainer(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model.train()
    model.gradient_checkpointing_enable()
    
    if args.method == 'lsrl':
        from lsrl import CPUAdamW, DistributedCPUAdamW
        if get_world_size() > 1: 
            CPUAdamW = DistributedCPUAdamW
            torch.distributed.init_process_group(backend='nccl')
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
        else: device = 'cuda'
        model = model.to(device)
        optimizer = CPUAdamW(model.parameters(), lr=1e-5, accum_steps=4,
                           weight_decay=0.01, eps=1e-8, grad_offload=args.grad_offload)
        def train_step(batch):
            loss = model(batch, labels=batch, use_cache=False).loss
            loss.backward()
            optimizer.step()
            return loss
    else:  # deepspeed
        import deepspeed
        config = get_ds_config(args)
        if is_main_process(): print(config)
        engine, _, _, _ = deepspeed.initialize(model=model, config=config)
        def train_step(batch):
            loss = engine(batch, labels=batch, use_cache=False).loss
            engine.backward(loss)
            engine.step()
            return loss
    return train_step

def save_result(args, analysis):
    config = {
        'method': args.method,
        'seq_len': args.seq_len,
        'world_size': get_world_size()
    }
    if args.method == 'lsrl': config['grad_offload'] = args.grad_offload
    else: config['zero_stage'] = args.ds_zero   
    result = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'analysis': analysis
    }
    with open('benchmark_results.jsonl', 'a') as f:
        f.write(json.dumps(result) + '\n')

def run_benchmark(args):
    """通用训练循环"""
    train_step = setup_trainer(args)
    
    device = 'cuda'
    step_times = []
    
    if is_main_process():
        print(f"🚀 Testing {args.method.upper()}, {get_world_size()} GPUs")
        if args.method == 'lsrl':
            print(f"Config: seq_len={args.seq_len}, grad_offload={args.grad_offload}")
        else:  # deepspeed
            print(f"Config: seq_len={args.seq_len}, zero_stage={args.ds_zero}")
        print("-" * 50)
    
    for step in range(1, 9):
        batch = torch.randint(1, 5000, (1, args.seq_len)).to(device)
        tic = time.time()
        train_step(batch)
        step_time = time.time() - tic
        if step >= 5: step_times.append(step_time)
        if is_main_process(): print(f"Step {step}: {step_time:.2f}s", '⭐' if step >= 5 else '')
    
    # 计算各种指标
    total_time_5_8 = sum(step_times)  # 5-8步总时间
    no_update_times = step_times[0:3]  # 5-7步 (无参数更新)
    update_time = step_times[3]  # 8步 (有参数更新)
    avg_no_update_time = sum(no_update_times) / len(no_update_times)
    
    # 吞吐量计算
    tokens_per_step = args.seq_len * get_world_size()
    # 不同accum步数下的吞吐量估算
    def estimate_throughput(accum_steps):
        avg_time_per_step = ((accum_steps - 1) * avg_no_update_time + update_time) / accum_steps
        return tokens_per_step / avg_time_per_step
    
    # 当前配置的真实吞吐量（accum_steps=4）
    throughput_current = estimate_throughput(4)
    throughput_accum64 = estimate_throughput(64)
    throughput_accum256 = estimate_throughput(256)
    
    ret = {
        'total_time_5_8': round(total_time_5_8, 1),
        'avg_no_update_time': round(avg_no_update_time, 1),
        'update_time': round(update_time, 1),
        'throughput_current': round(throughput_current, 0),  
        'throughput_accum64': round(throughput_accum64, 0),
        'throughput_accum256': round(throughput_accum256, 0),
        'seq_len': args.seq_len
    }
    if not is_main_process(): return ret
    save_result(args, ret)
    return ret

def main():
    args = setup_args()
    result = run_benchmark(args)
    if not is_main_process(): return
    print(f"\n📊 Performance Analysis:")
    print(f"1️⃣ Steps 5-8 total time: {result['total_time_5_8']:.2f}s")
    print(f"2️⃣ Avg time (no update, steps 5-7): {result['avg_no_update_time']:.2f}s")
    print(f"2️⃣ Time with update (step 8): {result['update_time']:.2f}s")
    print(f"3️⃣ Throughput (accum=4): {result['throughput_current']:.0f} tokens/sec")
    print(f"4️⃣ Estimated throughput (accum=64): {result['throughput_accum64']:.0f} tokens/sec")
    print(f"5️⃣ Estimated throughput (accum=256): {result['throughput_accum256']:.0f} tokens/sec")

if __name__ == "__main__":
    main()