from torch import optim
from transformers import Adafactor, AdamW

def get_grouped_parameters(model):
    optimizer_parameters = filter(lambda x: x.requires_grad, model.parameters())
    return optimizer_parameters


def init_optimizer_cls_and_kwargs(optim_type, lr, weight_decay, adam_beta1, adam_beta2, eps):
    optimizer_kwargs = {'lr':lr}
    adam_kwargs = {
        'betas':(adam_beta1, adam_beta2),
        'eps':eps,
        'weight_decay':weight_decay
    }
    
    if optim_type == OptimizerNames.ADAFACTOR:
        optimizer_cls = Adafactor
        optimizer_kwargs.update({'scale_parameter': False, 'relative_step': False})
    elif optim_type == OptimizerNames.ADAMW_HF:
        optimizer_cls = AdamW
        optimizer_kwargs.update(adam_kwargs)
    elif optim_type == OptimizerNames.ADAMW_TORCH:
        optimizer_cls = optim.AdamW
        optimizer_kwargs.update(adam_kwargs)
    elif optim_type == OptimizerNames.SGD:
        optimizer_cls = optim.SGD
    elif optim_type == OptimizerNames.ADAGRAD:
        optimizer_cls = optim.Adagrad
    else:
        raise ValueError(f" Unknown optimizer {optim_type}.")    
    
    return optimizer_cls, optimizer_kwargs


class OptimizerNames:
    ADAMW_HF = 'adamw_hf'
    ADAMW_TORCH = 'adamw_torch'
    ADAFACTOR = 'adafactor'
    SGD = 'sgd'
    ADAGRAD = 'adagrad'
    