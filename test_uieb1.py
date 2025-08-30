from collections import OrderedDict

import lpips

from image_utils import torchPSNR, torchSSIM, torchMSE, UCIQE
from uqim_utils import getUIQM
from Config.modelsx1 import CDFIENet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import time
from Config.options import opt
import math
import shutil
from tqdm import tqdm
from thop import profile
from thop import clever_format

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

CHECKPOINTS_DIR = opt.checkpoints_dir
INP_DIR = opt.testing_dir_inp
CLEAN_DIR = opt.testing_dir_gt

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

ch = 3

network = CDFIENet()

# 计算参数数量
total_params = sum(p.numel() for p in network.parameters())
# 只计算可训练的参数数量
trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
print(f'Total parameters: {total_params/1e6:.2f}M')
print(f'Trainable parameters: {trainable_params/1e6:.2f}M')

checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR, "SFIANet_bestpsnr.pt"), map_location=torch.device('cpu'))
state_dict = checkpoint['model_state_dict']

# 创建一个新的字典，其中不包含`module.`前缀
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k  # 删除`module.`前缀
    new_state_dict[name] = v

# 加载更新后的状态字典
network.load_state_dict(new_state_dict)
network.eval()
network = network.cuda(1)

loss_fn = lpips.LPIPS(net='alex', version='0.1').cuda(1)

result_dir = '/home/DATASETuieb/Test/output/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

from ptflops import get_model_complexity_info
with torch.cuda.device(1):  # 如果使用 GPU
    macs, params = get_model_complexity_info(network, (3, 256, 256), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
print(f"FLOPs: {macs}, Params: {params}")

if __name__ =='__main__':

    total_files = os.listdir(INP_DIR)
    PSNRs, SSIMs, LPIPs = [], [], []
    UIQMs, UCIQEs, MSEs = [], [], []
    with torch.no_grad():
        with tqdm(total=len(total_files)) as t:
            for m in total_files:
                tar = cv2.imread(CLEAN_DIR + str(m))
                tar = cv2.resize(tar, (256, 256),
                                 interpolation=cv2.INTER_AREA)
                tar = tar[:, :, ::-1]
                tar = tar.astype(np.float32) / 255.0
                tar_tensor = torch.from_numpy(tar).permute(2, 0, 1).unsqueeze(0)
                tar_tensor = tar_tensor.to(device)

                img = cv2.imread(INP_DIR + str(m))
                img = cv2.resize(img, (256,256),
                                 interpolation=cv2.INTER_AREA)
                img = img[:, :, ::-1]
                img = np.float32(img) / 255.0
                h,w,c=img.shape

                train_x = np.zeros((1, ch, h, w)).astype(np.float32)

                train_x[0,0,:,:] = img[:,:,0]
                train_x[0,1,:,:] = img[:,:,1]
                train_x[0,2,:,:] = img[:,:,2]
                dataset_torchx = torch.from_numpy(train_x)
                dataset_torchx=dataset_torchx.to(device)

                output=network(dataset_torchx)
                out=output
                output = (output.clamp_(0.0, 1.0)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                output = output[:, :, ::-1]
                cv2.imwrite(os.path.join(result_dir + str(m)), output)

                psnr = torchPSNR(tar_tensor, out)
                ssim = torchSSIM(tar_tensor, out)
                mse = torchMSE(tar_tensor, out)
                PSNRs.append(psnr)
                SSIMs.append(ssim)
                MSEs.append(mse)
                current_lpips_distance = loss_fn.forward(out, tar_tensor)
                LPIPs.append(current_lpips_distance)
                UIQMs.append(getUIQM(os.path.join(result_dir, str(m))))
                UCIQEs.append(UCIQE(os.path.join(result_dir, str(m))))

                t.set_postfix_str("name: {} | old [hw]: {}/{} | new [hw]: {}/{}".format(str(m), h,w, output.shape[0], output.shape[1]))
                t.update(1)
    avg_psnr = torch.stack(PSNRs).mean().item()
    avg_ssim = torch.stack(SSIMs).mean().item() * 100
    avg_mse = torch.stack(MSEs).mean().item()
    print("[PSNR] mean: {:.2f} ".format(avg_psnr))
    print("[SSIM] mean: {:.2f} ".format(avg_ssim))
    print("[MSE] mean: {:.0f} ".format(avg_mse))
    average_lpips_distance = torch.stack(LPIPs).mean().item()
    print("[LPIPS] mean: {:.2f}".format(average_lpips_distance))
    print("[UIQM] mean: {:.4f}".format(torch.stack(UIQMs).mean().item()))
    print("[UCIQE] mean: {:.4f}".format(torch.stack(UCIQEs).mean().item()))


