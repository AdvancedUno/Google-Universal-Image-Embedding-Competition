import torch
from torch.nn import (
    CrossEntropyLoss, BCEWithLogitsLoss, 
    L1Loss, MSELoss
)
import torch.nn as nn
import torch.nn.functional as F
import math


class LabelSmoothingCrossEntropyTimm(nn.Module):
    # Timm version
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class LabelSmoothingCrossEntropy(object):
    # Huggingface version
    def __init__(self, epsilon=0.1, ignore_index=-100):
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        
    def __call__(self, output, labels):
        log_probs = -nn.functional.log_softmax(output, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)
        padding_mask = labels.eq(self.ignore_index)
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


class CrossEntropy(object):
    def __init__(self):
        self.loss_f = CrossEntropyLoss()

    def __call__(self, output, target):
        loss = self.loss_f(input=output, target=target)
        return loss


class BCEWithLogLoss(object):
    def __init__(self):
        self.loss_fn = BCEWithLogitsLoss()

    def __call__(self, output, target):
        output = output.float()
        target = target.float()
        loss = self.loss_fn(input = output,target = target)
        return loss


class L1(object):
    def __init__(self):
        self.loss_fn = L1Loss()

    def __call__(self, output, target):
        output = output.view(-1).to(target.dtype)
        target = target.view(-1)
        return self.loss_fn(input=output, target=target)


class MSE(object):
    def __init__(self):
        self.loss_fn = MSELoss()

    def __call__(self, output, target):
        output = output.view(-1).to(target.dtype)
        target = target.view(-1)
        return self.loss_fn(input=output, target=target)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
    

def init_criterion(criterion_type, label_smoothing_factor):
    if criterion_type == 'l1':
        return L1()
    elif criterion_type == 'mse':
        return MSE()
    elif criterion_type == 'bce':
        return BCEWithLogLoss()
    elif criterion_type == 'ce':
        return CrossEntropy()
    elif criterion_type == 'label_smoothing':
        return LabelSmoothingCrossEntropy(epsilon=label_smoothing_factor)
    else:
        raise ValueError(f"{criterion_type} not recognized.")