import torch
import numpy as np
import scipy.io as sio
import os
import datetime
import random
from ssim_torch import ssim
import logging
import re


# ===================================================================
# CASSI Simulation
# ===================================================================

def shift(inputs, step=2):
    """
    Simulate the dispersion process in CASSI.

    Args:
        inputs: HSI cube tensor of shape (B, C, H, W).
        step: Dispersion step size. Default: 2.

    Returns:
        Shifted tensor of shape (B, C, H, W + (C-1)*step).
    """
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output


def shift_back(inputs, step=2):
    """
    Reverse the dispersion: reconstruct 3D HSI cube from 2D compressed measurement.

    Args:
        inputs: 2D measurement tensor of shape (B, H, W).
        step: Dispersion step size. Default: 2.

    Returns:
        Reconstructed HSI cube of shape (B, 28, H, W - (28-1)*step).
    """
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output


# ===================================================================
# Mask Generation
# ===================================================================

def generate_masks(mask_path, batch_size):
    """Load and expand the physical mask to batch size."""
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']  # (H, W)
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))  # (H, W, 28)
    mask3d = np.transpose(mask3d, [2, 0, 1])  # (28, H, W)
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch


def generate_shift_masks(mask_path, batch_size):
    """Load shifted 3D mask and compute sum-of-squares for normalization."""
    mask = sio.loadmat(mask_path + '/mask_3d_shift.mat')
    mask_3d_shift = mask['mask_3d_shift']  # (H, W+disp, 28)
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])  # (28, H, W+disp)
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch ** 2, 1)
    Phi_s_batch[Phi_s_batch == 0] = 1
    return Phi_batch, Phi_s_batch


def init_mask(mask_path, mask_type, batch_size):
    """
    Initialize mask based on the input type.

    Args:
        mask_path: Path to mask files.
        mask_type: Type of mask - 'Phi', 'Phi_PhiPhiT', 'Mask', or None.
        batch_size: Batch size for mask expansion.

    Returns:
        Tuple of (mask3d_batch, input_mask).
    """
    mask3d_batch = generate_masks(mask_path, batch_size)
    if mask_type == 'Phi':
        shift_mask3d_batch = shift(mask3d_batch)
        input_mask = shift_mask3d_batch
    elif mask_type == 'Phi_PhiPhiT':
        Phi_batch, Phi_s_batch = generate_shift_masks(mask_path, batch_size)
        input_mask = (Phi_batch, Phi_s_batch)
    elif mask_type == 'Mask':
        input_mask = mask3d_batch
    elif mask_type is None:
        input_mask = None
    return mask3d_batch, input_mask


# ===================================================================
# Dataset Loading
# ===================================================================

def LoadTraining(path):
    """Load training dataset from .mat files."""
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort(key=lambda l: int(re.findall(r'\d+', l)[0]))
    print('training scenes:', len(scene_list))
    for i in range(len(scene_list)):
        scene_path = path + scene_list[i]
        scene_num = int(scene_list[i].split('.')[0][5:])
        if scene_num <= 205:
            if 'mat' not in scene_path:
                continue
            data = sio.loadmat(scene_path)
            if "img_expand" in data:
                img = data['img_expand'] / 65536.
            elif "img" in data:
                img = data['img'] / 65536.
            elif "data_slice" in data:
                img = data['data_slice'] / 65536.
            img = img.astype(np.float32)
            imgs.append(img)
            print('Scene {} is loaded. {}'.format(i + 1, scene_list[i]))
    return imgs


def LoadTest(path_test):
    """Load test dataset from .mat files."""
    scene_list = os.listdir(path_test)
    scene_list.sort(key=lambda l: int(re.findall(r'\d+', l)[0]))
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['img']
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data


# ===================================================================
# Data Augmentation
# ===================================================================

def time2file_name(time):
    """Convert datetime string to filename-safe format."""
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    return year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second


def arguement_1(x):
    """
    Random rotation and flip augmentation.

    Args:
        x: Tensor of shape (C, H, W).

    Returns:
        Augmented tensor of shape (C, H, W).
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    # Random vertical flip
    for j in range(vFlip):
        x = torch.flip(x, dims=(2,))
    # Random horizontal flip
    for j in range(hFlip):
        x = torch.flip(x, dims=(1,))
    return x


def arguement_2(generate_gt):
    """
    Mosaic augmentation: stitch 4 patches into one image.

    Args:
        generate_gt: Tensor of shape (4, C, 128, 128).

    Returns:
        Stitched tensor of shape (C, 256, 256).
    """
    c, h, w = generate_gt.shape[1], 256, 256
    divid_point_h = 128
    divid_point_w = 128
    output_img = torch.zeros(c, h, w).cuda()
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img


def shuffle_crop(train_data, batch_size, crop_size=256, argument=True):
    """
    Random crop and augmentation for training data.

    Half of the batch uses direct crops from full-resolution images,
    the other half uses mosaic augmentation (4 smaller crops stitched together).

    Args:
        train_data: List of training images, each of shape (H, W, 28).
        batch_size: Number of samples per batch.
        crop_size: Crop size (default 256).
        argument: Whether to use mosaic augmentation.

    Returns:
        Batch tensor of shape (batch_size, 28, crop_size, crop_size).
    """
    if argument:
        gt_batch = []
        # First half: direct random crops with augmentation
        index = np.random.choice(range(len(train_data)), batch_size // 2)
        processed_data = np.zeros((batch_size // 2, crop_size, crop_size, 28), dtype=np.float32)
        h, w, _ = train_data[0].shape
        for i in range(batch_size // 2):
            img = train_data[index[i]]
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        processed_data = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda().float()
        for i in range(processed_data.shape[0]):
            gt_batch.append(arguement_1(processed_data[i]))

        # Second half: mosaic augmentation (4 random 128x128 patches)
        processed_data = np.zeros((4, 128, 128, 28), dtype=np.float32)
        for i in range(batch_size - batch_size // 2):
            sample_list = np.random.randint(0, len(train_data), 4)
            for j in range(4):
                x_index = np.random.randint(0, h - crop_size // 2)
                y_index = np.random.randint(0, w - crop_size // 2)
                processed_data[j] = train_data[sample_list[j]][
                    x_index:x_index + crop_size // 2,
                    y_index:y_index + crop_size // 2, :
                ]
            gt_batch_2 = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda()
            gt_batch.append(arguement_2(gt_batch_2))
        gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][
                x_index:x_index + crop_size,
                y_index:y_index + crop_size, :
            ]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
        return gt_batch


# ===================================================================
# Measurement Simulation
# ===================================================================

def gen_meas_torch(data_batch, mask3d_batch, Y2H=True, mul_mask=False):
    """
    Generate simulated CASSI measurement from HSI and mask.

    Args:
        data_batch: HSI batch of shape (B, C, H, W).
        mask3d_batch: Mask batch of shape (B, C, H, W).
        Y2H: If True, convert 2D measurement back to 3D cube.
        mul_mask: If True, multiply result by mask.

    Returns:
        Simulated measurement tensor.
    """
    nC = data_batch.shape[1]
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1)
    if Y2H:
        meas = meas / nC * 2
        H = shift_back(meas)
        if mul_mask:
            HM = torch.mul(H, mask3d_batch)
            return HM
        return H
    return meas


def init_meas(gt, mask, input_setting):
    """
    Initialize measurement based on input setting.

    Args:
        gt: Ground truth HSI of shape (B, C, H, W).
        mask: Mask tensor.
        input_setting: 'H' (shift-back), 'HM' (shift-back + mask), or 'Y' (raw 2D).

    Returns:
        Simulated measurement tensor.
    """
    if input_setting == 'H':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=False)
    elif input_setting == 'HM':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=True)
    elif input_setting == 'Y':
        input_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=True)
    return input_meas


# ===================================================================
# Metrics
# ===================================================================

def torch_psnr(img, ref):
    """Compute per-channel PSNR between two HSI tensors of shape (C, H, W)."""
    img = (img * 256).round()
    ref = (ref * 256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255 * 255) / mse)
    return psnr / nC


def torch_ssim(img, ref):
    """Compute SSIM between two HSI tensors of shape (C, H, W)."""
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))


# ===================================================================
# Logging and Checkpointing
# ===================================================================

def gen_log(model_path):
    """Create a logger that writes to both file and console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def checkpoint(model, epoch, optimizer, scheduler, model_path, logger):
    """Save training checkpoint including model, optimizer, and scheduler states."""
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))


def load_checkpoint(filepath, model, optimizer, scheduler):
    """Load training checkpoint and restore model, optimizer, and scheduler states."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return epoch


def my_summary(test_model, H=256, W=256, C=28, N=1):
    """Print model FLOPs and parameter count."""
    from fvcore.nn import FlopCountAnalysis
    model = test_model.cuda()
    inputs1 = torch.randn((N, C, H, W)).cuda()
    inputs2 = torch.randn((N)).cuda()
    inputs3 = torch.randn((N, C, H, W)).cuda()
    inputs4 = None
    inputs5 = torch.randn((N, C, H, W)).cuda()
    inputs = (inputs1, inputs2, inputs3, inputs4, inputs5)

    flops = FlopCountAnalysis(model, inputs)
    n_param = sum([p.nelement() for p in model.parameters()])
    print(f'GMac:{flops.total() / (1024 * 1024 * 1024)}')
    print(f'Params:{n_param}')
