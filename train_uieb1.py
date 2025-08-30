import time

import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

import Config.dataset as dataset
from Config.vgg import *
from torch.utils.data import DataLoader
from Config.options import opt, device
from Config.modelsx1 import *
from Config.misc import *
from torchsummary import summary
from pytorch_msssim import ssim

from image_utils import torchPSNR, torchSSIM
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class CharbonnierLoss(nn.Module):
	def __init__(self, epsilon=1e-3):
		super(CharbonnierLoss, self).__init__()
		self.epsilon = epsilon

	def forward(self, pred, gt):
		diff = pred - gt
		loss = torch.mean(torch.sqrt(diff * diff + self.epsilon ** 2))
		return loss


def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

transform = transforms.Compose([
			transforms.Resize((256, 256)),  # 将所有图像缩放到 256x256
		])
f = open('/home/ckpt/'+'uieb_save_log.txt','w')
f.write(f'experiment config file\n')
if __name__ == '__main__':
	net = CDFIENet()
	net = net.to(device)

	char_loss = CharbonnierLoss(epsilon=1e-3)
	mae_loss = nn.L1Loss()
	vgg = Vgg19(requires_grad=False).to(device)
	optim_g = optim.Adam(net.parameters(),
						 lr=opt.learning_rate_g, 
						 betas = (opt.beta1, opt.beta2), 
						 weight_decay=opt.wd_g)
	scheduler = lr_scheduler.StepLR(optim_g, step_size=50, gamma=0.5)

	train_dataset = dataset.Dataset_Load(hazy_path=opt.hazydir,
								   clean_path=opt.cleandir,
								   transform=dataset.ToTensor())
	train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size,num_workers=8, pin_memory=1, shuffle=True)

	val_dataset = dataset.ValDataset_Load(hazy_path=opt.val_hazydir,
								   clean_path=opt.val_cleandir,
								   transform=dataset.ToTensor())
	val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=8, pin_memory=1, shuffle=True)

	batches = int(opt.num_images / opt.batch_size)

	if not os.path.exists(opt.checkpoints_dir):
		os.makedirs(opt.checkpoints_dir)
	
	models_loaded = getLatestCheckpointName()    
	latest_checkpoint = models_loaded
	
	print('loading model for CDFIENet ', latest_checkpoint)
	
	if latest_checkpoint == None :
		start_epoch = 1
		print('No checkpoints found for CDFIENet retraining')
	else:
		checkpoint_g = torch.load(os.path.join(opt.checkpoints_dir, latest_checkpoint))    
		start_epoch = checkpoint_g['epoch'] + 1
		net.load_state_dict(checkpoint_g['model_state_dict'])
		optim_g.load_state_dict(checkpoint_g['optimizer_state_dict'])
		print('Restoring model from checkpoint ' + str(start_epoch))

	best_val_loss = 1000
	best_psnr = 0
	best_ssim = 0

	for epoch in range(start_epoch, opt.end_epoch + 1):
		net.train()
		opt.total_char_loss = 0.0
		opt.total_vgg_loss = 0.0
		opt.total_loss = 0.0
		opt.total_ssim_loss = 0.0
		tic = time.time()
		for i_batch, sample_batched in enumerate(train_dataloader):

			hazy_batch = sample_batched['hazy']
			clean_batch = sample_batched['clean']

			hazy_batch = hazy_batch.to(device)
			clean_batch = clean_batch.to(device)

			pred_batch = net(hazy_batch)

			batch_char_loss = char_loss(pred_batch, clean_batch)
			batch_char_loss.backward(retain_graph=True)
			
			batch_ssim_loss = 1-ssim(pred_batch, clean_batch,data_range=1, size_average=True)
			batch_ssim_loss.backward(retain_graph=True)

			clean_vgg_feats = vgg(normalize_batch(clean_batch))
			pred_vgg_feats = vgg(normalize_batch(pred_batch))
			batch_vgg_loss = torch.mul(opt.lambda_vgg, mae_loss(pred_vgg_feats.relu4_3, clean_vgg_feats.relu4_3))
			batch_vgg_loss.backward()
			
			opt.batch_char_loss = batch_char_loss.detach().cpu().item()
			opt.total_char_loss += opt.batch_char_loss

			opt.batch_ssim_loss = batch_ssim_loss.detach().cpu().item()
			opt.total_ssim_loss += opt.batch_ssim_loss

			opt.batch_vgg_loss = batch_vgg_loss.detach().cpu().item()
			opt.total_vgg_loss += opt.batch_vgg_loss
			
			opt.batch_loss = opt.batch_char_loss + opt.batch_ssim_loss + opt.batch_vgg_loss
			opt.total_loss += opt.batch_loss
			
			optim_g.step()
			optim_g.zero_grad() 

			print('\r Epoch : ' + str(epoch) + ' | (' + str(i_batch+1) + '/' + str(batches) + ') | l_char: ' + str(opt.batch_char_loss/2) + ' | l_ssim: ' + str(1-opt.batch_ssim_loss)+ ' | l_vgg: ' + str(opt.batch_vgg_loss), end='', flush=True)
		scheduler.step()
		toc = time.time()
		print('\n\nFinished ep. %d, lr = %.6f, mean_char = %.6f, mean_ssim = %.6f, mean_vgg = %.6f, time:%.2f' % (epoch, get_lr(optim_g), (opt.total_char_loss / batches)/2,1- (opt.total_ssim_loss / batches), opt.total_vgg_loss / batches, toc-tic))

		net.eval()
		val_PSNRs, val_SSIMs = [], []
		tic = time.time()
		with torch.no_grad():
			for i_batch, sample_batched in enumerate(val_dataloader):
				hazy_batch = sample_batched['hazy']
				clean_batch = sample_batched['clean']

				hazy_batch = hazy_batch.to(device)
				clean_batch = clean_batch.to(device)

				pred_batch = net(hazy_batch)

				psnr = torchPSNR(clean_batch, pred_batch)
				torchssim = torchSSIM(clean_batch, pred_batch)
				val_PSNRs.append(psnr)
				val_SSIMs.append(torchssim)
		toc = time.time()
		avg_psnr = torch.stack(val_PSNRs).mean().item()
		avg_ssim = torch.stack(val_SSIMs).mean().item() * 100
		print(f'on val set psnr:{avg_psnr:.2f} ssim:{avg_ssim:.2f} time:{toc - tic:.2f}')

		if epoch <= 500:
			if (avg_psnr > best_psnr):
				best_psnr = avg_psnr
				torch.save({'epoch': epoch,
							'model_state_dict': net.state_dict(),
							'optimizer_state_dict': optim_g.state_dict(),
							'char_loss': opt.total_char_loss,
							'ssim_loss': opt.total_ssim_loss,
							'vgg_loss': opt.total_vgg_loss,
							'opt': opt,
							'total_loss': opt.total_loss},
						   os.path.join(opt.checkpoints_dir, 'CDFIENet_bestpsnr' + '.pt'))
				print(f'save bestpsnr val model on epoch{epoch}')
				f.write(f'save bestpsnr val model on epoch:{epoch},val psnr:{best_psnr:.4f}\n')
			if (avg_ssim > best_ssim):
				best_ssim = avg_ssim
				torch.save({'epoch': epoch,
							'model_state_dict': net.state_dict(),
							'optimizer_state_dict': optim_g.state_dict(),
							'char_loss': opt.total_char_loss,
							'ssim_loss': opt.total_ssim_loss,
							'vgg_loss': opt.total_vgg_loss,
							'opt': opt,
							'total_loss': opt.total_loss},
						   os.path.join(opt.checkpoints_dir, 'CDFIENet_bestssim' + '.pt'))
				print(f'save bestssim val model on epoch{epoch}')
				f.write(f'save bestssim val model on epoch:{epoch},val ssim:{best_ssim:.4f}\n')

		if epoch % 50 == 0:
			torch.save({'epoch':epoch,
					'model_state_dict':net.state_dict(),
					'optimizer_state_dict':optim_g.state_dict(),
					'char_loss':opt.total_char_loss,
					'ssim_loss':opt.total_ssim_loss,
					'vgg_loss':opt.total_vgg_loss,
					'opt':opt,

					'total_loss':opt.total_loss}, os.path.join(opt.checkpoints_dir, 'CDFIENet_' + str(epoch) + '.pt'))
