import torch
import numpy as np
import os
import glob
from tqdm import tqdm

from utils.ray_utils import *
from utils.color_utils import read_image, resize_image
from utils.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from einops import rearrange

from .base import BaseDataset


class LLFFDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()
        self.dir_name = kwargs.get("dir_name")
        self.recov_dir = kwargs.get("recovery_dir")
        self.stage_one = kwargs.get("stage_one", True)
        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self):
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = int(camdata[1].height / self.downsample)
        w = int(camdata[1].width / self.downsample)
        self.img_wh = (w, h)
        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = (camdata[1].params[0] / self.downsample)
            fy = (camdata[1].params[0] / self.downsample)
            cx = (camdata[1].params[1] / self.downsample)
            cy = (camdata[1].params[2] / self.downsample)
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = (camdata[1].params[0] / self.downsample)
            fy = (camdata[1].params[1] / self.downsample)
            cx = (camdata[1].params[2] / self.downsample)
            cy = (camdata[1].params[3] / self.downsample)
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])

        self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], self.K)

    def read_meta(self, split, **kwargs):
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        perm = np.argsort(img_names)
        folder = f'images_{int(self.downsample)}'
        img_paths = [os.path.join(self.root_dir, folder, name)
                     for name in sorted(img_names)]
        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3]

        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        pts3d = np.array([pts3d[k].xyz for k in pts3d])  # (N, 3)

        self.poses, self.pts3d = center_poses(poses, pts3d)

        scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        self.poses[..., 3] /= scale
        self.pts3d /= scale

        self.noise_images = []
        self.rough_images = []
        self.clear_images = []

        # use every 8th image as test set
        if split == 'train':
            img_paths = [x for i, x in enumerate(img_paths) if i % 8 != 0]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i % 8 != 0])
        elif split == 'test':
            img_paths = [x for i, x in enumerate(img_paths) if i % 8 == 0]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i % 8 == 0])

        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path in tqdm(img_paths):
            clear_buf = []
            rough_buf = []
            noise_buf = []

            clear_img = read_image(img_path, self.img_wh, blend_a=False)
            clear_img = torch.FloatTensor(clear_img)
            clear_buf += [clear_img]
            self.clear_images += [torch.cat(clear_buf, 1)]

            if split == 'train':
                noise_path = img_path \
                    .replace(folder, folder + "_" + self.dir_name)

                noise_img = read_image(noise_path, self.img_wh, blend_a=False)
                noise_img = torch.FloatTensor(noise_img)
                noise_buf += [noise_img]
                self.noise_images += [torch.cat(noise_buf, 1)]

                rough_path = img_path.replace(self.root_dir, self.recov_dir) \
                    .replace(folder, folder + "_" + self.dir_name)
                rough_img = read_image(rough_path, self.img_wh, blend_a=False)
                rough_img = torch.FloatTensor(rough_img)
                rough_buf += [rough_img]
                self.rough_images += [torch.cat(rough_buf, 1)]

        self.clear_images = torch.stack(self.clear_images)

        if split == 'train':
            if len(self.rough_images) != 0:
                self.rough_images = torch.stack(self.rough_images)
            self.noise_images = torch.stack(self.noise_images)
        self.poses = torch.FloatTensor(self.poses)
