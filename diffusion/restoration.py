import argparse
import glob
import os

import cv2
import yaml
import torch
import numpy as np
from diffusion.ddm import DenoisingDiffusion
from einops import rearrange
from tqdm import tqdm
import imageio


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def resizeImg(img):
    wd_new, ht_new = img.shape[1], img.shape[0]
    if ht_new > wd_new and ht_new > 1024:
        wd_new = int(np.floor(wd_new * 1024 / ht_new))
        ht_new = 1024
    elif ht_new <= wd_new and wd_new > 1024:
        ht_new = int(np.floor(ht_new * 1024 / wd_new))
        wd_new = 1024
    wd_new = int(16 * np.ceil(wd_new / 16.0))
    ht_new = int(16 * np.ceil(ht_new / 16.0))
    return torch.FloatTensor(cv2.resize(img, (wd_new, ht_new)))


class DiffusiveRestoration:
    def __init__(self, args, device):
        self.args = args
        with open(self.args.config, "r") as f:
            load_config = yaml.safe_load(f)
        self.config = dict2namespace(load_config)
        self.config.device = device

        # set random seed
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.benchmark = True
        self.diffusion = DenoisingDiffusion(args, self.config)

        if os.path.isfile(self.args.resume):
            self.diffusion.load_ddm_ckpt(self.args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def SDSTrain(self, clear_img, noise_img):
        clear_img = rearrange(clear_img, "(n h w) c->n c h w", h=self.config.data.image_size,
                              w=self.config.data.image_size, c=3)
        noise_img = rearrange(noise_img, "(n h w) c->n c h w", h=self.config.data.image_size,
                              w=self.config.data.image_size,
                              c=3)
        return self.diffusion.nerf_train_step(clear_img.contiguous(), noise_img)

    def restore(self, root_dir, r=None):
        print("\ndenoising......")
        # make dir
        rough_dir = str(root_dir) + "_rough"
        os.makedirs(rough_dir, exist_ok=True)
        imgs = glob.glob(os.path.join(root_dir, "*"))
        with torch.no_grad():
            for img_path in tqdm(imgs):
                img = imageio.imread(img_path).astype(np.float32) / 255.0
                x = rearrange(resizeImg(img), "h w c->c h w").unsqueeze(0)
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                x_output = self.diffusive_restoration(x_cond, r=r)
                x_output = inverse_data_transform(x_output)
                rough_path = os.path.join(rough_dir, os.path.basename(img_path))
                x_output = (rearrange(x_output.cpu().numpy(), "1 c h w->h w c") * 255).astype(np.uint8)
                imageio.imsave(rough_path, x_output)

    def diffusive_restoration(self, x_cond, r=None):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x = torch.randn(x_cond.size(), device=self.diffusion.device)
        x_output = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list


def get_opts():
    parser = argparse.ArgumentParser()
    # diffusion
    parser.add_argument('--img_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--train_epoch', type=int, default=5,
                        help='number epoch of train')
    parser.add_argument('--patch_size', type=int, default=64,
                        help='size of sample patch')
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    parser.add_argument('--resume', default='', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument("--grid_r", type=int, default=16,
                        help="Grid cell width r that defines the overlap between patches,"
                             "A smaller value for grid_r=4 will yield slightly better results and higher image quality")
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps")
    return parser.parse_args()


if __name__ == '__main__':
    hparams = get_opts()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    diffusion = DiffusiveRestoration(hparams, device)
    diffusion.restore(hparams.img_dir, hparams.grid_r)
