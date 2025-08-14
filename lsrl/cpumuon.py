import torch, time, os
from torch.optim import Optimizer, AdamW, SGD
import torch.distributed as dist

class CPUMuon(Optimizer):  
    def __new__(cls, *args, **kwargs):
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            return DistributedCPUMuon(*args, **kwargs)
        else:
            return SoloCPUMuon(*args, **kwargs)

def should_use_muon(p):
    return p.dim() == 2 and min(p.shape) >= 128

def should_use_muon_with_name(param, name):
    if param.dim() != 2 or min(param.shape) < 16: return False
    exclude = ['embed', 'lm_head', 'head', 'output', 'classifier', 'ln', 'norm', 'bias']
    return not any(x in name.lower() for x in exclude)

class FirstOrderAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, _ = group['betas']
            wd = group['weight_decay']
            for p in group['params']:
                g = p.grad
                if g is None: continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32, device=p.device)
                exp_avg = state['exp_avg']
                state['step'] += 1
                g32 = g.to(torch.float32)
                exp_avg.mul_(beta1).add_(g32, alpha=1 - beta1)
                bc1 = 1.0 - beta1 ** state['step']
                step_size = lr / bc1
                if wd != 0: p.add_(p, alpha=-lr * wd)
                p.add_(exp_avg.to(p.dtype), alpha=-step_size)
        return loss

class SoloCPUMuon(Optimizer):  
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,  
                 weight_decay=0.1, accum_steps=1, grad_offload=False, 
                 verbose=False, ns_step=5, muon_lr=None, grad_clip=1.0,
                 ns_dtype=torch.bfloat16,
                 **kwargs):     
        self.accum_steps = accum_steps  
        self.current_step = 0  
        self.verbose = verbose
        self.ns_step = ns_step
        self.ns_dtype = ns_dtype
        self.grad_clip = grad_clip
        if muon_lr is None: muon_lr = lr

        params = list(params)  
        if not params or not isinstance(params[0], tuple):  
            raise TypeError("Expected named_parameters(). Use: CPUMuon(model.named_parameters(), ...)")  
  
        self.grad_offload = grad_offload

        print('\n\nThis optimizer uses lazy memory pinning, the first 2~3 steps may take longer than usual.\n')

        if accum_steps < 10:  
            print(f"\nWarning: accum_steps is set to {accum_steps}, which is too small for this optimizer. Consider setting it higher.")
        
        self.original_device_map = {}  
        muon_params = []  
        adam_params = []
        for name, p in params:  
            if not p.requires_grad: continue  
            cpu_p = p.cpu().detach().clone().requires_grad_(True)  
            if should_use_muon_with_name(p, name): muon_params.append(cpu_p)
            else: adam_params.append(cpu_p)
            self.original_device_map[cpu_p] = p  

        self.muon_optimizer = FirstOrderAdamW(muon_params, lr=muon_lr, betas=betas, weight_decay=weight_decay, **kwargs)
        self.adam_optimizer = AdamW(adam_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **kwargs)
        
        self.muon_params = muon_params
        self.adam_params = adam_params

    def _muon_decompose(self, grad):
        if grad.dim() != 2 or min(grad.shape) < 16: return grad
        grad_norm = grad.norm()
        if grad_norm < 1e-8: return grad
        odtype = grad.dtype
        Y = (grad / grad_norm).to(self.ns_dtype)
        I = torch.eye(Y.shape[1], device=Y.device, dtype=Y.dtype)
        for _ in range(self.ns_step): 
            Y = Y @ (1.5 * I - 0.5 * Y.T @ Y)
        Yn = Y.norm()
        if torch.isfinite(Yn) and Yn > 0: Y = Y / Yn
        return Y.to(odtype)

    @torch.no_grad()
    def _transfer_params_gradients(self, params, grad_transform=None):
        async_grad_transfers = []
        for cpu_p in params:
            orig_p = self.original_device_map[cpu_p]
            if orig_p.grad is not None:
                grad = orig_p.grad / self.accum_steps
                if grad_transform is not None: 
                    grad = grad_transform(grad)
                grad_future = grad.to('cpu', non_blocking=True)
                async_grad_transfers.append((cpu_p, grad_future))
                orig_p.grad = None
        for cpu_p, scaled_grad in async_grad_transfers:
            cpu_p.grad = scaled_grad if cpu_p.grad is None else cpu_p.grad + scaled_grad

    @torch.no_grad()
    def _no_grad_offload_transfer(self, is_update_step):
        if not is_update_step: return
        self._transfer_params_gradients(self.adam_params)
        self._transfer_params_gradients(self.muon_params, self._muon_decompose)

    @torch.no_grad()
    def _grad_offload_transfer(self, is_update_step):
        self._transfer_params_gradients(self.adam_params + self.muon_params)
        if not is_update_step: return
        torch.cuda.synchronize()
        async_grad_transfers = []
        for cpu_p in self.muon_params:
            orig_p = self.original_device_map[cpu_p]
            grad = cpu_p.grad.to(orig_p.device)
            grad = self._muon_decompose(grad)
            grad_future = grad.to('cpu', non_blocking=True)
            async_grad_transfers.append((cpu_p, grad_future))
        for cpu_p, scaled_grad in async_grad_transfers: cpu_p.grad = scaled_grad

    @torch.no_grad()  
    def step(self):  
        self.current_step += 1  
        is_update_step = self.current_step >= self.accum_steps  

        if self.grad_offload: self._grad_offload_transfer(is_update_step)
        else: self._no_grad_offload_transfer(is_update_step)

        if is_update_step:  
            torch.nn.utils.clip_grad_norm_(self.muon_params+self.adam_params, self.grad_clip)
            self.muon_optimizer.step()  
            self.adam_optimizer.step()  
            for cpu_p, original_p in self.original_device_map.items():  
                original_p.data.copy_(cpu_p.data, non_blocking=True)  
            torch.cuda.synchronize()
            self.muon_optimizer.zero_grad()  
            self.adam_optimizer.zero_grad()  
            self.current_step = 0  
        return is_update_step  
        
class DistributedCPUMuon(SoloCPUMuon):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,  
                 weight_decay=0.1, accum_steps=1, grad_offload=False, 
                 verbose=False, ns_step=5, muon_lr=None, grad_clip=1.0,
                 ns_dtype=torch.bfloat16,
                 **kwargs):         
        if int(os.environ.get('OMP_NUM_THREADS', 1)) < 10:
            print("\n\nWarning: OMP_NUM_THREADS is set to a low value, which may cause performance issues. Consider setting it to 10 or higher.")
        self.accum_steps = accum_steps  
        self.current_step = 0  
        self.verbose = verbose
        self.grad_offload = grad_offload
        self.ns_step = ns_step
        self.ns_dtype = ns_dtype
        self.grad_clip = grad_clip
        if muon_lr is None: muon_lr = lr
        self.rank = torch.distributed.get_rank()
        print('\nDistributed optimizer initialized on rank', self.rank)
        
        params = list(params)  
        if not params or not isinstance(params[0], tuple):  
            raise TypeError("Expected named_parameters(). Use: CPUMuon(model.named_parameters(), ...)")  
  
        if self.rank == 0:
            print('\n\nThis optimizer uses lazy memory pinning, the first 2~3 steps may take longer than usual.\n')
            if accum_steps < 10: print(f"\nWarning: accum_steps is set to {accum_steps}, which is too small for this optimizer. Consider setting it higher.")

        self.gpu_params = [p for name, p in params if p.requires_grad]         
        if self.rank == 0:
            self.original_device_map = {}  
            muon_params = []  
            adam_params = []
            for name, p in params:  
                if not p.requires_grad: continue  
                cpu_p = p.cpu().detach().clone().requires_grad_(True)  
                if should_use_muon_with_name(p, name): muon_params.append(cpu_p)
                else: adam_params.append(cpu_p)
                self.original_device_map[cpu_p] = p  

            self.muon_optimizer = FirstOrderAdamW(muon_params, lr=muon_lr, betas=betas, weight_decay=weight_decay, **kwargs)
            self.adam_optimizer = AdamW(adam_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **kwargs)
        
            self.muon_params = muon_params
            self.adam_params = adam_params

    @torch.no_grad()  
    def step(self, closure=None):  
        self.current_step += 1  
        is_update_step = self.current_step >= self.accum_steps  
        
        if self.grad_offload or is_update_step:
            for p in self.gpu_params:
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad.data, op=torch.distributed.ReduceOp.AVG) 

        if self.rank == 0:
            if self.grad_offload: self._grad_offload_transfer(is_update_step)
            else: self._no_grad_offload_transfer(is_update_step)
                    
        if self.grad_offload or is_update_step:
            for orig_p in self.gpu_params: orig_p.grad = None

        if is_update_step:  
            torch.cuda.synchronize()
            if self.rank == 0:
                tic = time.time()
                torch.nn.utils.clip_grad_norm_(self.muon_params+self.adam_params, self.grad_clip)
                self.muon_optimizer.step()  
                self.adam_optimizer.step()  
                if self.verbose: print(f"CPU optimizer step took {time.time() - tic:.2f} seconds")
                tic = time.time()
                for cpu_p, orig_p in self.original_device_map.items():  
                    orig_p.copy_(cpu_p, non_blocking=True)  
                torch.cuda.synchronize()
                if self.verbose: print(f"Data copy took {time.time() - tic:.2f} seconds")
                self.muon_optimizer.zero_grad()  
                self.adam_optimizer.zero_grad()  
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