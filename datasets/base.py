from torch.utils.data import Dataset
import numpy as np
import random
import torch
from einops import rearrange, repeat
from PIL import Image
import torchvision
import matplotlib.pyplot as plt


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """

    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.batch_size = kwargs.get("batch_size", 8192)
        self.patch_size = 64

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return self.poses.shape[0]

    def __getitem__(self, idx):
        if self.stage_one and self.split == "train":
            return self.get_pix(idx)
        return self.get_image(idx)

    def get_pix(self, idx):
        # training pose is retrieved in run.py
        img_idxs = np.random.choice(len(self.poses), self.batch_size)
        # randomly select pixels
        pix_idxs = np.random.choice(self.img_wh[0] * self.img_wh[1], self.batch_size)
        sample = {'img_idxs': img_idxs,
                  'rough_rgb': self.rough_images[img_idxs, pix_idxs][:, :3],
                  'noise_rgb': self.noise_images[img_idxs, pix_idxs][:, :3],
                  "clear_rgb": self.clear_images[img_idxs, pix_idxs][:, :3],
                  "pose": self.poses[img_idxs],
                  "direction": self.directions[pix_idxs]}
        return sample

    def get_image(self, idx):
        if self.split.startswith('train'):
            rough_image = []
            noise_image = []
            clear_image = []
            direction = []
            img_idxs = np.random.choice(len(self.poses), 2)
            for i in range(2):
                img_idx = img_idxs[i]
                x = random.randint(0, self.img_wh[0] - self.patch_size)
                y = random.randint(0, self.img_wh[1] - self.patch_size)
                rough_image += [self.rough_images[img_idx, y:y + self.patch_size, x:x + self.patch_size]]
                noise_image += [self.noise_images[img_idx, y:y + self.patch_size, x:x + self.patch_size]]
                clear_image += [self.clear_images[img_idx, y:y + self.patch_size, x:x + self.patch_size]]
                direction += [self.directions[y:y + self.patch_size, x:x + self.patch_size]]

            rough_image = torch.stack(rough_image).flatten(start_dim=0, end_dim=-2)
            noise_image = torch.stack(noise_image).flatten(start_dim=0, end_dim=-2)
            clear_image = torch.stack(clear_image).flatten(start_dim=0, end_dim=-2)
            direction = torch.stack(direction).flatten(start_dim=0, end_dim=-2)
            sample = {'img_idxs': img_idxs,
                      'rough_rgb': rough_image,
                      'noise_rgb': noise_image,
                      "clear_rgb": clear_image,
                      "pose": self.poses[img_idxs].flatten(start_dim=0, end_dim=-2),
                      "direction": direction}
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.clear_images) > 0:  # if ground truth available
                sample['clear_rgb'] = self.clear_images[idx, :, :3]
            sample["direction"] = self.directions
        return sample

    def in_stage_two(self):
        self.stage_one = False
        self.rough_images = rearrange(self.rough_images, "n (h w) c->n h w c", w=self.img_wh[0])
        self.noise_images = rearrange(self.noise_images, "n (h w) c->n h w c", w=self.img_wh[0])
        self.clear_images = rearrange(self.clear_images, "n (h w) c->n h w c", w=self.img_wh[0])
        self.directions = rearrange(self.directions, "(h w) c->h w c", w=self.img_wh[0])
