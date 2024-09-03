import cv2
from einops import rearrange
import imageio
import numpy as np
import torch


def read_image(img_path, img_wh, blend_a=True):
    img = imageio.imread(img_path).astype(np.float32) / 255.0
    if img.shape[2] == 4:  # blend A to RGB
        if blend_a:
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
        else:
            img = img[..., :3] * img[..., -1:]

    img = cv2.resize(img, img_wh)
    # img, (wd_new, ht_new) = resize_image(img, img_wh)
    img = rearrange(img, 'h w c -> (h w) c')
    return img


# torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
def resize_image(img_wh):
    # Resizing images to multiples of 16 for whole-image restoration
    wd_new, ht_new = img_wh
    if ht_new > wd_new and ht_new > 1024:
        wd_new = int(np.ceil(wd_new * 1024 / ht_new))
        ht_new = 1024
    elif ht_new <= wd_new and wd_new > 1024:
        ht_new = int(np.ceil(ht_new * 1024 / wd_new))
        wd_new = 1024
    wd_new = int(16 * np.float(wd_new / 16.0))
    ht_new = int(16 * np.float(ht_new / 16.0))
    # input_img = cv2.resize(input_img, (wd_new, ht_new))
    # input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
    img_wh = (wd_new, ht_new)
    return img_wh
