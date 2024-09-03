import os.path

import numpy as np
import imageio
import cv2
from einops import rearrange
import matplotlib.pyplot as plt
import glob
import tqdm
from tqdm.contrib import tzip


def read_image(img_path, img_wh=None, blend_a=True):
    img = imageio.imread(img_path).astype(np.float32) / 255.0
    if img.shape[2] == 4:  # blend A to RGB
        if blend_a:
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
        else:
            img = img[..., :3] * img[..., -1:]
    if img_wh is not None:
        img = cv2.resize(img, img_wh)
    img = rearrange(img, 'h w c -> c h w')
    return img


def ASMHaz(beta, A, image_path, depth_path, depth_scatter, depth_img_wh=False):
    if depth_path.endswith("tiff"):
        depth = imageio.imread(depth_path).astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min())
    else:
        depth = (imageio.imread(depth_path).astype(np.float32))
        if depth.max() > 255:
            depth = depth / 10000 * 3
            # depth = (depth - depth.min()) / (depth.max() - depth.min())
        else:
            depth = depth / 255
        if depth.ndim == 3:
            depth = depth[:, :, 0]
    # print(depth.shape, depth.max(), depth.min())
    img_wh = None
    if depth_img_wh is True:
        h, w = depth.shape
        img_wh = (w, h)
    img = read_image(image_path, img_wh)
    if depth_scatter != 0:
        d = np.where(depth == 0, 1, depth)
        d = d * depth_scatter
    else:
        d = depth

    beta = np.ones(d.shape) * beta
    transmission = np.exp((-beta * d))

    Ic = transmission * img + (1 - transmission) * A
    return rearrange(Ic, "c h w->h w c")


def SyntheticNeRFAddHaz():
    A = 0.8
    depth_scatter = 5

    splits = ["train", "test"]
    betas = [0.14, 0.2]
    data_path = "/data/data/Synthetic_NeRF_Haz"
    for root in glob.glob(os.path.join(data_path, "*")):
        for beta in betas:
            for split in splits:
                path = os.path.join(root, split)
                haz_path = os.path.join(root, split + "_" + str(beta) + "_" + str(A) + "_" + str(depth_scatter))
                print("\n----start in:" + haz_path)
                if os.path.exists(haz_path):
                    break
                depth_paths = glob.glob(os.path.join(path, "*depth*.png"))
                os.makedirs(haz_path, exist_ok=True)
                for depth_path in tqdm.tqdm(depth_paths):
                    image_path = depth_path.split("_depth")[0] + depth_path[-4:]
                    haz_image = ASMHaz(beta, A, image_path, depth_path, depth_scatter)
                    haz_image_save = (haz_image * 255).astype(np.uint8)
                    imageio.imsave(os.path.join(haz_path, os.path.basename(image_path)), haz_image_save)


def LLFFAddHaz():
    A = 0.8
    depth_scatter = 15
    betas = [0.14, 0.2]
    down_samples = 4
    imagesfolder = f"images_{down_samples}"
    depth_folder = f"depth_{down_samples}"
    data_path = "/data/data/nerf_llff_data"
    for path in glob.glob(os.path.join(data_path, "*")):
        for beta in betas:
            haz_path = os.path.join(path, imagesfolder + "_" + str(beta) + "_" + str(A) + "_" + str(
                depth_scatter))
            print("\n----start in:" + haz_path)
            if os.path.exists(haz_path):
                break
            depth_paths = glob.glob(os.path.join(path, depth_folder, "*.tiff"))
            os.makedirs(haz_path, exist_ok=True)
            for depth_path in tqdm.tqdm(depth_paths):
                image_path = depth_path \
                    .replace(depth_folder, imagesfolder) \
                    .replace("distance_median_", "") \
                    .replace("tiff", "png")
                haz_image = ASMHaz(beta, A, image_path, depth_path, depth_scatter)
                haz_image_save = (haz_image * 255).astype(np.uint8)
                imageio.imsave(os.path.join(haz_path, os.path.basename(image_path)), haz_image_save)


def v2360AddHaz():
    A = 0.8
    depth_scatter = 20
    betas = [0.14, 0.2]
    down_samples = 4
    imagesfolder = f"images_{down_samples}"
    depth_folder = f"depth_{down_samples}"
    data_path = "/data/data/360v2/"
    for path in glob.glob(os.path.join(data_path, "*")):
        for beta in betas:
            haz_path = os.path.join(path, imagesfolder + "_" + str(beta) + "_" + str(A) + "_" + str(
                depth_scatter))
            print("\n----start in:" + haz_path)
            depth_paths = sorted(glob.glob(os.path.join(path, depth_folder, "*.tiff")))
            image_paths = sorted(glob.glob(os.path.join(path, imagesfolder, "*.JPG")))
            image_paths = [x for i, x in enumerate(image_paths) if i % 8 != 0]
            os.makedirs(haz_path, exist_ok=True)
            for depth_path, image_path in tzip(depth_paths, image_paths):
                haz_image = ASMHaz(beta, A, image_path, depth_path, depth_scatter, True)
                haz_image_save = (haz_image * 255).astype(np.uint8)
                imageio.imsave(os.path.join(haz_path, os.path.basename(image_path)), haz_image_save)


def tankAddHaz():
    A = 0.8
    depth_scatter = 30
    betas = [0.14, 0.2]
    data_path = "/data/data/tanks_and_temples"
    for path in glob.glob(os.path.join(data_path, "*")):
        for beta in betas:
            haz_path = os.path.join(path, "train", "rgb" + "_" + str(beta) + "_" + str(A) + "_" + str(
                depth_scatter))
            print("\n----start in:" + haz_path)
            depth_paths = sorted(glob.glob(os.path.join(path, "train", "depth", "*.tiff")))
            image_paths = sorted(glob.glob(os.path.join(path, "train", "rgb", "*.png")))
            os.makedirs(haz_path, exist_ok=True)
            for depth_path, image_path in tzip(depth_paths, image_paths):
                haz_image = ASMHaz(beta, A, image_path, depth_path, depth_scatter, False)
                haz_image_save = (haz_image * 255).astype(np.uint8)
                imageio.imsave(os.path.join(haz_path, os.path.basename(image_path)), haz_image_save)


if __name__ == '__main__':
    # LLFFAddHaz()
    # tankAddHaz()
    # v2360AddHaz()
    A = 0.8
    depth_scatter = 10
    betas = [0.14]
    down_samples = 4
    imagesfolder = f"images_{down_samples}"
    depth_folder = f"depth_{down_samples}"
    path = "/data/data/360v2/garden"

    for beta in betas:
        haz_path = os.path.join(path, imagesfolder + "_" + str(beta) + "_" + str(A) + "_" + str(
            depth_scatter))
        print("\n----start in:" + haz_path)
        depth_paths = sorted(glob.glob(os.path.join(path, depth_folder, "*.tiff")))
        image_paths = sorted(glob.glob(os.path.join(path, imagesfolder, "*.JPG")))
        image_paths = [x for i, x in enumerate(image_paths) if i % 8 != 0]
        os.makedirs(haz_path, exist_ok=True)
        for depth_path, image_path in tzip(depth_paths, image_paths):
            haz_image = ASMHaz(beta, A, image_path, depth_path, depth_scatter, True)
            haz_image_save = (haz_image * 255).astype(np.uint8)
            imageio.imsave(os.path.join(haz_path, os.path.basename(image_path)), haz_image_save)


    # A = 0.8
    # depth_scatter = 5
    # betas = [0.14]
    # down_samples = 4
    # imagesfolder = f"images_{down_samples}"
    # depth_folder = f"depth_{down_samples}"
    # path = "/data/data/nerf_llff_data/leaves"
    # for beta in betas:
    #     haz_path = os.path.join(path, imagesfolder + "_" + str(beta) + "_" + str(A) + "_" + str(
    #         depth_scatter))
    #     print("\n----start in:" + haz_path)
    #     # if os.path.exists(haz_path):
    #     #     break
    #     depth_paths = glob.glob(os.path.join(path, depth_folder, "*.tiff"))
    #     os.makedirs(haz_path, exist_ok=True)
    #     for depth_path in tqdm.tqdm(depth_paths):
    #         image_path = depth_path \
    #             .replace(depth_folder, imagesfolder) \
    #             .replace("distance_median_", "") \
    #             .replace("tiff", "JPG")
    #         haz_image = ASMHaz(beta, A, image_path, depth_path, depth_scatter)
    #         haz_image_save = (haz_image * 255).astype(np.uint8)
    #         imageio.imsave(os.path.join(haz_path, os.path.basename(image_path)), haz_image_save)
    #
    #     plt.imshow(haz_image_save)
    #     plt.show()
