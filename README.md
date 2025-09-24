# LSRL (Low ReSource RL)

**🚀 Efficient and User-Friendly Large Model Training Framework | Train 14B Models on Consumer GPUs**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://sider.ai/zh-CN/LICENSE)[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org/)[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

*Simple, efficient, low-resource SFT and RL training solution for large language models*

## ✨ Key Features
* 🚀 **Single-GPU RL Training**: Complete RL training pipeline for 14B models on just one 80G GPU
* 🎯 ​**Ultra-Low Resource Requirements**​: SFT 14B models on a single 80G GPU with 18K sequence length support
* 🔄 ​**Asynchronous RL Training**​: Decoupled generation and training processes with cross-machine support
* 💾 ​**Memory Optimization**​: CPUAdamW + gradient offloading to break memory limitations
* 🛠️ ​**Simple & Flexible**​: Clean code, loose coupling, easy to modify and extend
* ⚡ ​**Minimal Dependencies**​: Training requires only PyTorch (+vLLM for RL)
* 🎮 ​**Consumer GPU Friendly**​: Support RTX 3090/4090 for 14B model training

## 👏 News
* **🔥 NEW**: Add SyncLSRL, which runs like verl and you don't need to carefully set gen_devices and ref_devices. See examples/rl_callpy.py.
* **🔥 NEW**: Add supports to vLLM v1 with use_vllm_v1=True. Just one env varible, enabling state dict updating and prefix caching.
* Add **Muon** optimizer! Fast & memory-efficient as CPUAdamW. You can use it to do sft or RL.
* Add GSPO.
* 14B model full-parameter RL training on single 80G GPU with GRPO (without vLLM)
* **Recommended Configs**: 
  - 7B models: Single GPU + vLLM for optimal speed
  - 14B models: Dual 80G GPUs for production-ready training
  - **Scale up**: More GPUs = faster training and larger batch sizes
 
## 🚀 Quick Start

### Installation

```bash
pip install git+https://github.com/lsdefine/lsrl.git
```

### SFT Training Example

Train 14B models on a single 80G GPU with two configuration options:

```python
import sys
from lsrl import CPUAdamW, DistributedCPUAdamW

# Configuration: python train.py [gradoffload|]
# Config 1: Memory-efficient (seq_len=18000, grad_offload=True) - longer sequences
# Config 2: Speed-optimized (seq_len=8000, grad_offload=False) - faster training
if len(sys.argv) > 1 and sys.argv[1] == "gradoffload": 	
	seq_len, grad_offload = 18000, True
else: seq_len, grad_offload = 8000, False

print(f"Config: grad_offload={grad_offload}, support seq_len={seq_len}")

# Use CPUAdamW optimizer
opt = CPUAdamW(model.parameters(), 
	lr=1e-5, accum_steps=4, weight_decay=0.01, 
	eps=1e-8, grad_offload=grad_offload)

# Standard training loop
for step in range(1, max_steps): 
	batch = get_batch()  # Your data loading logic 
	loss = model(batch, labels=batch, use_cache=False).loss 
	loss.backward() 
	opt.step()
# SAME as the normal!
```

### Asynchronous RL Training Example

```python
from lsrl import LSRL, RefServer 
from datasets import load_dataset 
import random, sys  

# goto hf-mirror to download, or use HF-string
model_path = "/data2/Qwen/Qwen2.5-14B-Instruct"  

# Start Reference Server (can be on different machine) 
if 'ref' in sys.argv:     
    RefServer(model_path).start()    
    sys.exit(0)  

# Prepare training data 
dataset = load_dataset("meta-math/GSM8K_zh", "default", split="train") 
QAs = [{'Q': x, 'A': y.split('####')[-1].strip()} for x, y in zip(dataset['question_zh'], dataset['answer']) ] 
random.shuffle(QAs)  

# Configure RL training 
lsrl = LSRL(model_path, epochs=1, train_data=QAs, rollout_num=8, 
train_batch_size=8, gen_batch_size=4, gen_update_steps=16, trainer='LSCPU',  # Use CPUAdamW
     gen_temperature=0.9, gen_device=[1,2],  # GPUs for runing vLLM
     ref_server="http://127.0.0.1:59876",  # Cross-machine RefServer
     lr=1e-6, accum_steps=16, genlog_filename='rl_log' )  

# Add reward functions 
lsrl.add_reward(format_reward_fn) 
lsrl.add_reward(correctness_reward_fn)  
# Set prompt functions 
lsrl.set_policy_prompt_fn(make_prompt_fn) 
lsrl.set_rollout_prompt_fn(make_prompt_fn)  
# Start training 
lsrl.train()
```

```bash
CUDA_VISIBLE_DEVICES=3 python rl.py ref
CUDA_VISIBLE_DEVICES=0 python rl.py
```
Reference server can co-locate with vLLM on the same GPU (via `gpu_memory_utilization` setting).

## 🏗️ Core Architecture

### CPUAdamW Optimizer

LSRL's core is the CPUAdamW optimizer, achieving memory breakthrough through optimizer state offloading:

```python
from lsrl import CPUAdamW

# Efficient optimizer with gradient offloading support
optimizer = CPUAdamW(model.parameters(), 
	lr=learning_rate, grad_offload=True,  # Gradient offloading for additional memory savings 
	accum_steps=16      # Gradient accumulation 
)
```

### Asynchronous RL Architecture

Fully decoupled generation and training with cross-machine deployment support:

```python
# Machine A: Start RefServer
python rl.py ref
# Machine B: Main training
CUDA_VISIBLE_DEVICES=0 python rl.py
```

```python
lsrl = LSRL( model_path, trainer='LSCPU',   # CPUAdamW optimizer 
	# trainer='DeepSpeed',   # DeepSpeed also supported 
)
```

## 🚀 SFT Performance Benchmarks

### Experimental Setup

We conducted comprehensive SFT performance benchmarks comparing **LSRL** against **DeepSpeed ZeRO** across different GPU configurations and sequence lengths.
See examples/benchmark.py

**Test Environment:**
- Hardware: NVIDIA A800 80GB GPUs
- Model: Qwen2.5-14B-Instruct (14B parameters)
- Batch Size: 1 per GPU
- Gradient Accumulation Steps: 4 (actual) / 256 (estimated)
- Precision: BF16

**Metrics:**
- **Forward Time**: Time for forward pass (seconds)
- **Update Time**: Time for gradient computation + parameter update (seconds)  
- **Throughput**: Tokens processed per second

### Results Summary

| GPUs | Seq Len | Method | Config | Forward (s) | Update (s) | Throughput accum=4 (tokens/s) | Throughput accum=256 (tokens/s) |
|------|---------|--------|--------|-------------|------------|---------------------|------------------------|
| 1 | 4K | DeepSpeed | ZeRO-1 | 2.5 | 28.2 | 447 | 1540 |
| 1 | 4K | DeepSpeed | ZeRO-2 | 5.8 | 34.1 | 310 | 676 |
| 1 | 4K | LSRL | no grad offload | 2.5 | 17.4 | 642 | 1568 |
| 1 | 8K | DeepSpeed | ZeRO-2 | 10.2 | 41.0 | 448 | 777 |
| 1 | 8K | LSRL | no grad offload | 5.3 | 30.3 | 692 | 1475 |
| 1 | 10K | DeepSpeed | ZeRO-1 | - | - | B**OOM**! | B**OOM**! |
| 1 | 10K | DeepSpeed | ZeRO-2 | 10.6 | 40.7 | 552 | 936 |
| 1 | 10K | LSRL | grad offload | 8.9 | 29.9 | 705 | 1107 |
| 1 | 18K | DeepSpeed | ZeRO-2 | 18.2 | - | B**OOM**! | B**OOM**! |
| 1 | 18K⭐ | LSRL | grad offload | 16.2 | 45.4 | 766 | 1102 |
| 2 | 7.5K | DeepSpeed | ZeRO-1 | 5.2 | 43.9 | 1009 | 2816 |
| 2 | 7.5K | DeepSpeed | ZeRO-2 | 13.1 | 40.5 | 752 | 1137 |
| 2 | 7.5K | LSRL | no grad offload | 5.0 | 22.7 | 1595 | 2969 |
| 2 | 10K | DeepSpeed | ZeRO-1 | - | - | B**OOM**! | B**OOM**! |
| 2 | 15K | DeepSpeed | ZeRO-2 | 18.9 | 48.9 | 1136 | 1576 |
| 2 | 18K⭐ | LSRL | grad offload | 16.6 | 39.6 | 1612 | 2157 |


## 📦 Dependencies

Training Core (Minimal Dependencies):

* `torch >= 2.0`
* `transformers`

RL Generation:

* `vllm` (Generation acceleration)
* `requests` (RefServer communication)
* `bottle`

## 📄 Model Support

* ​**Architectures**​: All HuggingFace models supported by CPUAdamW
* ​**Pipeline Parallelism**​: Currently tested with Qwen series
* ​**RL Algorithms**​: GRPO (more algorithms coming soon)

## 👏 Citation

If you find the code in our project useful, please consider citing our work as follows:

```
@misc{LSRL,
  author = {Jiaqing Liang},
  title = {LSRL: Memory Efficient Large Model Training Framework},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lsdefine/lsrl}},
}
```




