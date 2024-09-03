import os.path

import numpy as np
import imageio
import cv2
from einops import rearrange
import matplotlib.pyplot as plt
import glob
import tqdm
import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)

rain_seq = iaa.Sequential([
    # 调整亮度
    # iaa.Multiply((1.2, 1.5)),
    # 分水岭算法
    # iaa.Superpixels(p_replace=0.1, n_segments=150),
    # 模糊
    # iaa.GaussianBlur((0, 3.0)),
    # 垂直循环推进雨滴
    # iaa.Affine(rotate=(0, 0), translate_percent=(0, 0.3), mode='symmetric'),
    # 雨滴特效
    iaa.Rain(drop_size=(0.2, 0.4),  # 0.2 0.3
             # blur_sigma=(0.0, 1.0),
             # brightness=1.0,
             speed=(0.01, 0.02),
             deterministic=False,
             random_state=None),
])

snow_seq = iaa.Sequential([
    # 模糊
    # iaa.GaussianBlur((0, 3.0)),
    # 雪特效
    iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03),
                   deterministic=False,
                   random_state=None),
])


def LLFF(scene, seq, out_ext):
    down_samples = 4
    data_path = f"/data/data/nerf_llff_data/{scene}/images_{down_samples}"
    for path in glob.glob(os.path.join(data_path, "*")):
        # image = cv2.imread(path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = imageio.imread(path)
        out_dir = data_path + "_" + out_ext
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(path))
        img_aug = seq.augment_image(image)
        plt.imshow(img_aug)
        plt.show()
        # img_aug = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
        imageio.imsave(out_path, img_aug)
        # cv2.imwrite(out_path, img_aug)


def v2360(seq, out_ext):
    down_samples = 4
    data_path = f"/data/data/360v2"
    for data_path in glob.glob(os.path.join(data_path, "*")):
        data_path = os.path.join(data_path, f"images_{down_samples}")
        for path in glob.glob(os.path.join(data_path, "*")):
            # image = cv2.imread(path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = imageio.imread(path)
            out_dir = data_path + "_" + out_ext
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(path))
            img_aug = seq.augment_image(image)
            # img_aug = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
            imageio.imsave(out_path, img_aug)
            # cv2.imwrite(out_path, img_aug)


def tank(seq, out_ext):
    data_path = f"/data/data/tanks_and_temples"
    for data_path in glob.glob(os.path.join(data_path, "*")):
        data_path = os.path.join(data_path, "train")
        for path in glob.glob(os.path.join(data_path, "rgb", "*.png")):
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out_dir = os.path.join(data_path, "rgb_" + out_ext)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(path))
            img_aug = seq.augment_image(image)
            img_aug = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
            cv2.imwrite(out_path, img_aug)


if __name__ == '__main__':
    LLFF("fortress", seq=rain_seq, out_ext="rain")
    LLFF("fortress", seq=snow_seq, out_ext="snow")

    # LLFF("horns", seq=rain_seq, out_ext="rain")
    # LLFF("horns", seq=snow_seq, out_ext="snow")

    # v2360(rain_seq, "rain")
    # v2360(snow_seq, "snow")

    # tank(rain_seq, 'rain')
    # tank(snow_seq, 'snow')

    # data_path = f"/data/data/nerf_llff_data/fern/images_4/IMG_4026.png"
    # image = cv2.imread(data_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img_aug = snow2_seq.augment_image(image)
    # img_aug = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
    # plt.imshow(img_aug)
    # plt.show()
