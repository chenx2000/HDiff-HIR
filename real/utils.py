"""
Utility functions for the Real CASSI reconstruction pipeline.
"""

import numpy as np
import scipy.io as sio
import os
import glob
import re
import torch
import torch.nn as nn
import math
import random
import logging


# ===================================================================
# PSNR / MSE Metrics
# ===================================================================

def compare_mse(im1, im2):
    """Compute Mean Squared Error between two images (numpy arrays)."""
    im1 = np.asarray(im1, dtype=np.float64)
    im2 = np.asarray(im2, dtype=np.float64)
    return np.mean(np.square(im1 - im2), dtype=np.float64)


def compare_psnr(im_true, im_test, data_range=None):
    """Compute PSNR between two images given a data range."""
    im_true = np.asarray(im_true, dtype=np.float64)
    im_test = np.asarray(im_test, dtype=np.float64)
    err = compare_mse(im_true, im_test)
    return 10 * np.log10((data_range ** 2) / err)


def psnr(img1, img2):
    """Compute PSNR between two uint8-scale images."""
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    return 20 * math.log10(1 / math.sqrt(mse))


def PSNR_GPU(im_true, im_fake):
    """Compute PSNR on GPU tensors of shape (C, H, W)."""
    im_true = (im_true * 255).round()
    im_fake = (im_fake * 255).round()
    C, H, W = im_true.size()
    err = nn.MSELoss(reduction='sum')(im_true.clone(), im_fake.clone()) / (C * H * W)
    return 10. * np.log((255 ** 2) / (err.data + 1e-12)) / np.log(10.)


# ===================================================================
# Data Parallel
# ===================================================================

def dataparallel(model, ngpus, gpu0=0):
    """Wrap model in DataParallel if multiple GPUs are available."""
    if ngpus == 0:
        raise ValueError("only support gpu mode")
    gpu_list = list(range(gpu0, gpu0 + ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus
    if ngpus > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
    elif ngpus == 1:
        model = model.cuda()
    return model


# ===================================================================
# Checkpoint Utilities
# ===================================================================

def findLastCheckpoint(save_dir):
    """Find the latest checkpoint epoch number in a directory."""
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for f in file_list:
            result = re.findall(r".*model_(.*).pth.*", f)
            epochs_exist.append(int(result[0]))
        return max(epochs_exist)
    return 0


# ===================================================================
# Dataset Loading
# ===================================================================

def prepare_data(path, file_num):
    """Load HSI data from .mat files (512x512x28 format)."""
    HR_HSI = np.zeros((512, 512, 28, file_num))
    for idx in range(file_num):
        path1 = os.path.join(path, 'scene%02d.mat' % (idx + 1))
        data = sio.loadmat(path1)
        HR_HSI[:, :, :, idx] = data['data_slice'] / 65535.0
    HR_HSI = np.clip(HR_HSI, 0., 1.)
    return HR_HSI


def prepare_data_cave(path, file_num):
    """Load CAVE dataset from .mat files."""
    HR_HSI = np.zeros((512, 512, 28, file_num))
    file_list = os.listdir(path)
    for idx in range(file_num):
        print(f'loading CAVE {idx}')
        path1 = os.path.join(path, file_list[idx])
        data = sio.loadmat(path1)
        HR_HSI[:, :, :, idx] = data['data_slice'] / 65535.0
    HR_HSI = np.clip(HR_HSI, 0., 1.)
    return HR_HSI


def prepare_data_KAIST(path, file_num):
    """Load KAIST dataset from .mat files."""
    HR_HSI = np.zeros((2704, 3376, 28, file_num))
    file_list = os.listdir(path)
    for idx in range(file_num):
        print(f'loading KAIST {idx}')
        path1 = os.path.join(path, file_list[idx])
        data = sio.loadmat(path1)
        HR_HSI[:, :, :, idx] = data['HSI']
    HR_HSI = np.clip(HR_HSI, 0., 1.)
    return HR_HSI


# ===================================================================
# Mask Initialization
# ===================================================================

def init_mask(mask, Phi, Phi_s, mask_type):
    """
    Select mask format for model input.

    Args:
        mask: Raw 3D mask array.
        Phi: Shifted 3D mask tensor.
        Phi_s: Sum-of-squares normalization tensor.
        mask_type: 'Phi', 'Phi_PhiPhiT', 'Mask', or None.

    Returns:
        Selected mask in the requested format.
    """
    if mask_type == 'Phi':
        return Phi
    elif mask_type == 'Phi_PhiPhiT':
        return (Phi, Phi_s)
    elif mask_type == 'Mask':
        return mask
    elif mask_type is None:
        return None


# ===================================================================
# Logging
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