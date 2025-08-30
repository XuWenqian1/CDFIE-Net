import argparse
import torch
import os

parser = argparse.ArgumentParser()

parser.add_argument('--hazydir', default='/home/DATASETuieb/Train/input/')
parser.add_argument('--cleandir', default='/home/DATASETuieb/Train/reference/')

parser.add_argument('--val_hazydir', default='/home/DATASETuieb/Val/input/')
parser.add_argument('--val_cleandir', default='/home/DATASETuieb/Val/reference/')

parser.add_argument('--checkpoints_dir', default='/home/ckpt/UIEB/')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_images', type=int, default=720)

parser.add_argument('--learning_rate_g', type=float, default=0.002)

parser.add_argument('--end_epoch', type=int, default=200)
parser.add_argument('--img_extension', default='.png')
parser.add_argument('--image_size', type=int ,default=256)

parser.add_argument('--beta1', type=float ,default=0.9)
parser.add_argument('--beta2', type=float ,default=0.999)
parser.add_argument('--wd_g', type=float ,default=0.00005)
parser.add_argument('--wd_d', type=float ,default=0.00000)

parser.add_argument('--batch_char_loss', type=float, default=0.0)
parser.add_argument('--total_char_loss', type=float, default=0.0)

parser.add_argument('--batch_ssim_loss', type=float, default=0.0)
parser.add_argument('--total_ssim_loss', type=float, default=0.0)

parser.add_argument('--batch_vgg_loss', type=float, default=0.0)
parser.add_argument('--total_vgg_loss', type=float, default=0.0)

parser.add_argument('--batch_loss', type=float, default=0.0)
parser.add_argument('--total_loss', type=float, default=0.0)

parser.add_argument('--lambda_char', type=float, default=1.0)
parser.add_argument('--lambda_vgg', type=float, default=1.0)
parser.add_argument('--lambda_ssim', type=float, default=1.0)


parser.add_argument('--testing_epoch', type=int, default=1)
parser.add_argument('--testing_mode', default="Nat")
parser.add_argument('--testing_dir_inp', default="/home/DATASETuieb/Test/input/")
parser.add_argument('--testing_dir_gt', default="/home/DATASETuieb/Test/reference/")

opt = parser.parse_args()
# print(opt)

device = torch.device(f'cuda:{1}')

# print(device)

if not os.path.exists(opt.checkpoints_dir):
    os.makedirs(opt.checkpoints_dir)

