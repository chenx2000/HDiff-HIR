import option as opt
import os

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

import torch
import numpy as np
import scipy.io as sio
import torch.nn.functional as F

from models import model_generator
from diffusion import create_diffusion
from utils import dataparallel

seed = opt.seed
torch.manual_seed(seed)

# ===================================================================
# Configuration
# ===================================================================

# Paths
data_path = "/home/students/master/2022/chenx/datasets/TSA_real_data/Measurements/"
mask_path = "/home/students/master/2022/chenx/datasets/TSA_real_data/mask.mat"
save_path = './real/test/result/'

# Pretrained model
pretrained_model_path = None  # Set this to the checkpoint path

# ===================================================================
# Data Loading
# ===================================================================

def prepare_real_measurements(path, file_num):
    """Load real CASSI measurements from .mat files."""
    HR_HSI = np.zeros((660, 714, file_num))
    for idx in range(file_num):
        path1 = os.path.join(path, 'scene' + str(idx + 1) + '.mat')
        data = sio.loadmat(path1)
        HR_HSI[:, :, idx] = data['meas_real']
    HR_HSI = np.clip(HR_HSI, 0.0, 1.0)
    return HR_HSI


def load_mask(path, size=660):
    """Load and prepare the physical mask for real CASSI reconstruction."""
    data = sio.loadmat(path)
    mask = data['mask']
    mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))

    mask_3d_shift = np.zeros((size, size + (28 - 1) * 2, 28))
    mask_3d_shift[:, 0:size, :] = mask_3d
    for t in range(28):
        mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
    mask_3d_shift_s = np.sum(mask_3d_shift ** 2, axis=2, keepdims=False)
    mask_3d_shift_s[mask_3d_shift_s == 0] = 1

    mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)
    mask_3d_shift_s = torch.FloatTensor(mask_3d_shift_s.copy())
    mask_3d = torch.FloatTensor(mask_3d.copy()).permute(2, 0, 1)
    return mask_3d_shift.unsqueeze(0), mask_3d_shift_s.unsqueeze(0), mask_3d.unsqueeze(0)


# ===================================================================
# Sampling
# ===================================================================

# Load real measurements
HR_HSI = prepare_real_measurements(data_path, 5)

# Load mask
mask_3d_shift, mask_3d_shift_s, mask_3d = load_mask(mask_path)

# Model
model = model_generator(opt.method, pretrained_model_path).cuda()
model = model.eval()
model = dataparallel(model, 1)

# Diffusion process (DDIM sampling)
diffusion = create_diffusion(
    timesteps=opt.timesteps,
    beta_schedule='linear',
    parameterization='x0',
    ddim_sampling_steps='ddim3',
    loss_type='mse',
)

# Run sampling for all 5 real scenes
os.makedirs(save_path, exist_ok=True)

for j in range(5):
    with torch.no_grad():
        meas = HR_HSI[:, :, j]
        meas = meas / meas.max() * 0.8
        meas = torch.FloatTensor(meas)
        input_meas = meas.unsqueeze(0).cuda()

        mask_3d_gpu = mask_3d.cuda()

        out = diffusion.ddim_sample_loop(
            model, shape=[1, 28, 660, 660],
            condition=input_meas, mask=mask_3d_gpu,
        )
        result = out.clamp(min=0., max=1.)

    res = result.cpu().permute(2, 3, 1, 0).squeeze(3).numpy()
    save_file = os.path.join(save_path, f'{j}.mat')
    sio.savemat(save_file, {'res': res})
    print(f'Scene {j} saved to {save_file}')
