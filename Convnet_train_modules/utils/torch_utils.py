import os
import math
import time
from copy import deepcopy
import platform
import subprocess
import importlib

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from .general import LOGGER, colorstr

def is_torch_available():
    # PyTorch installed
   return importlib.util.find_spec("torch") is not None


def is_apex_available():
    # Check NVIDIA Apex is installed
    return importlib.util.find_spec("apex") is not None
    

def init_seed(seed=0):
    # Intitialize random number generator (RNG) seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def init_full_determinism(seed=0):
    # Enable PyTorch deterministic mode
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    init_seed(seed)
    if is_torch_available():
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
        torch.use_deterministic_algorithms(True)
        cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


def device_count():
    # Number of CUDA devices available
    assert platform.system() in ('Linux', 'Windows'), "device_count() only supported on Linux or Windows"
    try:
        cmd = 'nvidia-smi -L | wc -l' if platform.system() == "Linux" else 'nvidia-smi -L | find /c /v ""' 
        return int(subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode.split()[-1])
    except Exception:
        return 0
        

def init_optimal_loader_workers():
    # Optimal workers for async data loading during dataloader initialization
    num_cpus = multiprocessing.cpu_count()
    num_gpus = device_count()
    return min(min(num_cpus, num_gpus * 4) if num_gpus else num_cpus - 1, 4)


def time_sync():
    # Pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def seed_worker():
    # Set worker seed during dataloader initialization
    worker_seed = torch.initial_seed() % 2 ** 32
    init_seed(worker_seed)


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def find_modules(model, mclass):
    # Finds layer indices matching module class 'mclass'
    return [i for i, n in enumerate(model.module_list()) if isinstance(n, mclass)]


def model_info(model, verbose=False, image_size=128):
    # Model information
    # Parameters and gradients
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    if verbose:
        # Layers summary
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    try:
        # GFLOPs
        from thop import profile
        image = torch.zeros((1, 3, image_size, image_size), device=next(model.parameters()).device)
        flops = profile(deepcopy(model), inputs=(image,), verbose=False)[0] / 1E9 * 2
        fs = ', %.1f GFLOPs' % (flops) 
    except:
        fs = ''
    
    # Model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size = (param_size + buffer_size) / 1024**2
    
    name = type(model).__name__
    LOGGER.info(colorstr("blue", "bold", f" {name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients, size {size:.3f}MB{fs}"))


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    # Number of CUDA devices available in distributed setting
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    # Is master process
    return get_rank() == 0
    

def barrier():
    # Wait for each local_master
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    

def select_device(device=''):
    # Setup accelerator device to be used [CPU, GPU]
    s = f" Python-{platform.python_version()} torch-{torch.__version__} "
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')
    cpu = device == "cpu"
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use --device cpu or pass valid CUDA device(s)"
            
    if not cpu and torch.cuda.is_available():
        devices = device.split(',') if device else '0'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 30):.0f}GB)\n"
        arg = 'cuda:0'
    else:
        s += 'CPU\n'
        arg = 'cpu'
    
    LOGGER.info(colorstr("blue", "bold", s))
    return torch.device(arg)


class ModelEMA(object):
    # Keeps a moving average of everything in the model state_dict
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = deepcopy(de_parallel(model)).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad = False
    
    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
        
        msd = de_parallel(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()
                
    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        copy_attr(self.ema, model, include, exclude)
        

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a 
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)
            

class AverageMeter:
    # Calculate running statistics of metrics
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def speed_metrics(mode, start_time, num_samples=None, num_steps=None):
    # Measure and return speed performance metrics
    runtime = time.time() - start_time
    result = {f"{mode}_runtime": round(runtime, 4)}
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f'{mode}_samples_per_second'] = round(samples_per_second, 3)
    if num_samples is not None:
        steps_per_second  = num_steps / runtime
        result[f"{mode}_steps_per_second"] = round(steps_per_second, 3)
    return result 


if __name__ == "__main__":
    if is_apex_available():
        print("apex installed")