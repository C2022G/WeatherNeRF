import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--recovery_dir', type=str, required=False,
                        help='root directory of recovery dataset')
    parser.add_argument('--dir_name', type=str)
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'test'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor for the images')

    # model parameters
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    # training options
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')

    # loss weight
    parser.add_argument('--bdc_weight_1', type=float, default=1e-3)
    parser.add_argument('--opacity_weight_1', type=float, default=1e-2)

    parser.add_argument('--bdc_weight_2', type=float, default=1e-3)
    parser.add_argument('--opacity_weight_2', type=float, default=1e-2)


    # validation options
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load)')
    # diffusion
    parser.add_argument('--train_epoch', type=int, default=5,
                        help='number epoch of train')
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    parser.add_argument('--config', default='config/allweather.yml', type=str,
                        help='Path for the diffusion model config')
    parser.add_argument('--resume', default='', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--grid_r", type=int, default=16,
                        help="Grid cell width r that defines the overlap between patches,"
                             "A smaller value for grid_r=4 will yield slightly better results and higher image quality")
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps")
    return parser.parse_args()
