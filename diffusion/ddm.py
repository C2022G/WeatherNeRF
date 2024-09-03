import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_utils import logging, sampling
from diffusion.unet import DiffusionUNet
import torch.backends.cudnn as cudnn
import os
import time
from einops import rearrange


# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_estimation_loss(model, x0, t, e, b):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x0[:, :3, :, :], x], dim=1), t.float())
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


class DenoisingDiffusion(object):
    def __init__(self, hparams, config):
        super().__init__()
        self.config = config
        self.hparams = hparams
        self.device = config.device

        self.model = DiffusionUNet(config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)
        self.start_epoch, self.step = 0, 0
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        self.min_step = int(self.num_timesteps * 0.02)  # 0.02
        self.max_step = int(self.num_timesteps * 0.98)  # 0.98

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def nerf_train_step(self, x, noise):
        x = x.to(self.device)
        x = data_transform(x)
        noise = noise.to(self.device)
        noise = data_transform(noise)
        b = self.betas
        t = torch.randint(low=self.min_step, high=self.max_step + 1, size=[x.shape[0]], dtype=torch.long,
                          device=self.device)
        with torch.no_grad():
            e = torch.randn_like(x)
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
            x0 = x * a.sqrt() + e * (1.0 - a).sqrt()
            noise_input = torch.cat([noise, x0], dim=1)
            uncond_input = torch.cat([x, x0], dim=1)
            model_input = torch.cat([uncond_input, noise_input], dim=0)
            t = torch.cat([t] * 2)
            output = self.model(model_input, t.float())

        pred_uncond, pred_noise = output.chunk(2)
        noise_pred = pred_uncond + 100 * (pred_noise - pred_uncond)
        w = a ** 0.5 * (1 - a)
        grad = w * (noise_pred - e)
        grad = torch.nan_to_num(grad)
        x.backward(gradient=grad, retain_graph=True)
        return 0

    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.hparams.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None:
            xs = sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                        corners=patch_locs, p_size=patch_size)
        else:
            xs = sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.)
        if last:
            xs = xs[0][-1]
        return xs
