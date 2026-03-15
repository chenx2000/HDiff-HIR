import option as opt
import os

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

import torch
import numpy as np
import scipy.io as scio

from models import model_generator
from diffusion import create_diffusion
from utils import (
    init_mask, LoadTest, init_meas,
    torch_psnr, torch_ssim, gen_log,
    load_checkpoint, time2file_name,
)

import datetime

seed = 3407
torch.manual_seed(seed)

# ===================================================================
# Setup
# ===================================================================

# Mask
mask3d_batch_train, input_mask_train = init_mask(
    opt.mask_path, opt.input_mask, opt.batch_size
)
mask3d_batch_test, input_mask_test = init_mask(
    opt.mask_path, opt.input_mask, 10
)

# Dataset
test_data = LoadTest(opt.test_path)

# Output paths
date_time = time2file_name(str(datetime.datetime.now()))
result_path = opt.outf + date_time + '/result/'
model_path = opt.outf + date_time + '/model/'
os.makedirs(result_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# Model
model = model_generator(opt.method).cuda()

# Diffusion process (DDIM sampling)
diffusion = create_diffusion(
    timesteps=opt.timesteps,
    beta_schedule='linear',
    parameterization='x0',
    ddim_sampling_steps='ddim3',
    loss_type='mse',
)

# Optimizer and scheduler (needed for checkpoint loading)
optimizer = torch.optim.Adam(
    model.parameters(), lr=opt.diffusion_learning_rate, betas=(0.9, 0.999)
)
if opt.scheduler == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=opt.milestones, gamma=opt.gamma
    )
elif opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, opt.max_epoch, eta_min=1e-6
    )

# Load pretrained checkpoint
if opt.pretrained_model_path is not None:
    start_epoch = load_checkpoint(opt.pretrained_model_path, model, optimizer, scheduler)
else:
    print("Warning: opt.pretrained_model_path is None. Sampling with untrained model!")


# ===================================================================
# Testing
# ===================================================================

def test(epoch, logger):
    psnr_list, ssim_list = [], []
    test_gt = test_data.cuda().float()
    input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)
    model.eval()
    begin = __import__('time').time()
    with torch.no_grad():
        model_out = diffusion.ddim_sample_loop(
            model, shape=[10, 28, 256, 256],
            condition=input_meas, mask=mask3d_batch_test,
        )

    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())

    end = __import__('time').time()
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))

    logger.info(
        '===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'.format(
            epoch, psnr_mean, ssim_mean, (end - begin)
        )
    )
    model.train()
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean


def main():
    logger = gen_log(model_path)
    psnr_max = 0
    epoch = 0
    (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger)

    name = opt.outf + 'result.mat'
    print(f'Save reconstructed HSIs as {name}.')
    scio.savemat(name, {'truth': truth, 'pred': pred})

    if psnr_mean > psnr_max:
        psnr_max = psnr_mean
        if psnr_mean > 28:
            name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(
                epoch, psnr_max, ssim_mean
            ) + '.mat'
            scio.savemat(name, {
                'truth': truth, 'pred': pred,
                'psnr_list': psnr_all, 'ssim_list': ssim_all,
            })


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()
