import torch, time, os
from torch.optim import Optimizer, AdamW 
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_

if hasattr(torch, 'npu') and torch.npu.is_available():
    synchronize = torch.npu.synchronize
    #print("[CPUAdamW] Backend bound to: NPU (Ascend)")
else:
    synchronize = torch.cuda.synchronize

class CPUAdamW(Optimizer):  
    def __new__(cls, *args, **kwargs):
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            return DistributedCPUAdamW(*args, **kwargs)
        else:
            return SoloCPUAdamW(*args, **kwargs)

class SoloCPUAdamW(Optimizer):  
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,  
                 weight_decay=0.0, accum_steps=1, grad_offload=None, verbose=False, grad_clip=None,
                 **kwargs):     
        self.accum_steps = accum_steps  
        self.current_step = 0  
        self.verbose = verbose
        self.grad_clip = grad_clip

        params = list(params)   
        self.grad_offload = grad_offload

        print('\n\nThis optimizer uses lazy memory pinning, the first 2~3 steps may take longer than usual.\n')

        if accum_steps < 10:  
            print(f"\nWarning: accum_steps is set to {accum_steps}, which is too small for this optimizer. Consider setting it higher.")
        
        self.original_device_map = {}  
        cpu_params = []  
        for p in params:  
            if p.requires_grad:  
                cpu_p = p.cpu().detach().to(torch.float32).clone().requires_grad_(True)  
                cpu_params.append(cpu_p)  
                self.original_device_map[cpu_p] = p  
        self.cpu_optimizer = AdamW(cpu_params, lr=lr, betas=betas,  
                                   eps=eps, weight_decay=weight_decay, **kwargs)  
        self.param_groups = self.cpu_optimizer.param_groups  
        self.state = self.cpu_optimizer.state  

    @torch.no_grad()  
    def step(self, force_update=False, div_num=0):  
        self.current_step += 1  
        is_update_step = self.current_step >= self.accum_steps or force_update

        if self.grad_offload or is_update_step:
            divx = self.accum_steps if div_num == 0 else div_num
            if self.grad_clip is not None: clip_grad_norm_(self.original_device_map.values(), max_norm=self.grad_clip)
            async_grad_transfers = []
            for cpu_p, orig_p in self.original_device_map.items():
                if orig_p.grad is not None:
                    scaled_grad_future = (orig_p.grad.to(torch.float32) / divx).to('cpu', non_blocking=True)
                    async_grad_transfers.append((cpu_p, scaled_grad_future))
                    orig_p.grad = None
            for cpu_p, scaled_grad in async_grad_transfers:
                if cpu_p.grad is None: cpu_p.grad = scaled_grad
                else: cpu_p.grad.add_(scaled_grad)

        if is_update_step:  
            self.cpu_optimizer.step()  
            for cpu_p, orig_p in self.original_device_map.items():  
                orig_p.copy_(cpu_p.to(orig_p.dtype), non_blocking=True)  
            synchronize()
            self.cpu_optimizer.zero_grad()  
            self.current_step = 0  
        return is_update_step  
        
    def zero_grad(self, set_to_none: bool = True):  
        for original_p in self.original_device_map.values():  
            if original_p.grad is not None:  
                if set_to_none:  
                    original_p.grad = None  
                else:  
                    original_p.grad.detach_()  
                    original_p.grad.zero_()  
        self.cpu_optimizer.zero_grad(set_to_none=set_to_none)  
        self.current_step = 0  


class DistributedCPUAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, accum_steps=1, grad_offload=None, verbose=False, grad_clip=None,
                 **kwargs):
        if int(os.environ.get('OMP_NUM_THREADS', 1)) < 10:
            print("\n\nWarning: OMP_NUM_THREADS is set to a low value, which may cause performance issues. Consider setting it to 10 or higher.")
        self.accum_steps = accum_steps  
        self.current_step = 0  
        self.verbose = verbose
        self.rank = torch.distributed.get_rank()
        params = list(params)  
        self.grad_clip = grad_clip
        
        self.grad_offload = grad_offload
        self.gpu_params = [p for p in params if p.requires_grad] 

        if self.rank == 0:
            self.original_device_map = {}  
            cpu_params = []  
            for p in params:  
                if p.requires_grad:  
                    cpu_p = p.cpu().detach().to(torch.float32).clone().requires_grad_(True)  
                    cpu_params.append(cpu_p)  
                    self.original_device_map[cpu_p] = p  
            self.cpu_optimizer = AdamW(cpu_params, lr=lr, betas=betas,  
                                eps=eps, weight_decay=weight_decay, **kwargs)  
            self.param_groups = self.cpu_optimizer.param_groups  
            self.state = self.cpu_optimizer.state 

    @torch.no_grad()  
    def step(self, closure=None, force_update=False, div_num=0):  
        self.current_step += 1  
        is_update_step = self.current_step >= self.accum_steps or force_update 
        
        if self.grad_offload or is_update_step:
            for p in self.gpu_params:
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG) 
            if self.rank == 0:
                divx = self.accum_steps if div_num == 0 else div_num
                if self.grad_clip is not None: clip_grad_norm_(self.original_device_map.values(), max_norm=self.grad_clip)
                async_grad_transfers = []
                for cpu_p, orig_p in self.original_device_map.items():
                    if orig_p.grad is not None:
                        scaled_grad_future = (orig_p.grad.to(torch.float32) / divx).to('cpu', non_blocking=True)
                        async_grad_transfers.append((cpu_p, scaled_grad_future))
                        orig_p.grad = None
                for cpu_p, scaled_grad in async_grad_transfers:
                    if cpu_p.grad is None: cpu_p.grad = scaled_grad
                    else: cpu_p.grad.add_(scaled_grad)
            else:
                for orig_p in self.gpu_params: orig_p.grad = None

        if is_update_step:  
            synchronize()
            if self.rank == 0:
                tic = time.time()
                self.cpu_optimizer.step()  
                if self.verbose: print(f"CPU optimizer step took {time.time() - tic:.2f} seconds")
                tic = time.time()
                for cpu_p, orig_p in self.original_device_map.items():  
                    orig_p.copy_(cpu_p.to(orig_p.dtype), non_blocking=True)  
                synchronize()
                if self.verbose: print(f"Data copy took {time.time() - tic:.2f} seconds")
                self.cpu_optimizer.zero_grad()  
            torch.distributed.barrier()
            tic = time.time()
            handles = []
            for p in self.gpu_params:
                handle = torch.distributed.broadcast(p.data, src=0, async_op=True)
                handles.append(handle)
            for handle in handles: handle.wait()
            if self.rank == 0 and self.verbose:
                print(f"Distributed step took {time.time() - tic:.2f} seconds")
            self.current_step = 0  
        return is_update_step  