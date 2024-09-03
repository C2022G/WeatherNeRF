import torch.optim.lr_scheduler
import torch
import torch.optim.lr_scheduler
from .base_regularization import NeDepth
from einops import rearrange, repeat
import vren


class CompositeLoss:
    def __init__(self, weight=1):
        self.weight = weight

    def apply(self, results, batch):
        return self.weight * (torch.pow(results['rgb'] - batch['rough_rgb'], 2)).mean()


class BDCLoss:
    def __init__(self, weight=1e-3):
        self.weight = weight

    def apply(self, results, batch):
        ne_depth = NeDepth.apply(results['sigmas'], results['deltas'], results['ts'],
                                 results['rays_a'],
                                 results['vr_samples'])
        loss = (self.weight * torch.abs(ne_depth - results['depth'])).mean()
        return loss


class OpacityLoss:
    def __init__(self, weight=1e-3):
        self.weight = weight

    def apply(self, results, batch):
        o = results['opacity'].clamp(1e-5, 1 - 1e-5)
        loss = (self.weight * -(o * torch.log(o))).mean()
        return loss
