import os
import glob
import shutil

if __name__ == '__main__':
    # root_dir = "/data/data/nerf_llff_data//leaves/"
    # o_names = sorted(glob.glob(os.path.join(root_dir, "images", "*")))
    # n_names = sorted(glob.glob(os.path.join(root_dir, "depth_4", "*")))
    # for o, n in zip(o_names, n_names):
    #     o_base = os.path.basename(o)
    #     n_base = os.path.basename(n)
    #     new_n = n.replace(n_base.split(".")[0], o_base.split(".")[0])
    #     os.rename(n, new_n)

    # root_dir = "/data/data/car_o"
    # paths = os.listdir(root_dir)
    # for path in paths:
    #     images = glob.glob(os.path.join(root_dir,path, "images", "*.jpg"))
    #     for image in images:
    #         o_base = os.path.basename(image)
    #         n_base = str(path) + "_" + o_base
    #         n_image = os.path.join("/data/data/car/images", n_base)
    #         shutil.copyfile(image, n_image)

    root_dir = "/data/program/UtilityIR/result/trex/"
    o_names = sorted(glob.glob(os.path.join(root_dir, "images_4_snow", "*")))
    for o in o_names:
        if o.endswith(".JPG"):
            os.rename(o, o.replace("JPG", "jpg"))
