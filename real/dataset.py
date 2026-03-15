"""
Training dataset for Real CASSI reconstruction.

Generates training samples by simulating CASSI measurements from CAVE and
KAIST hyperspectral datasets with random masks and shot noise.
"""

import torch
import torch.utils.data as tud
import random
import numpy as np
import scipy.io as sio


class RealCassiDataset(tud.Dataset):
    """Training dataset that simulates real CASSI measurements.

    Generates (measurement, label, mask) triplets by combining CAVE/KAIST
    HSI ground truth with random mask crops and shot noise simulation.

    Args:
        opt: Configuration object with attributes:
            - isTrain (bool): Training or test mode.
            - size (int): Spatial crop size.
            - trainset_num / testset_num (int): Number of samples per epoch.
            - mask_path (str): Path to physical mask .mat file.
    """

    def __init__(self, opt, CAVE, KAIST):
        super().__init__()
        self.isTrain = opt.isTrain
        self.size = opt.size
        if self.isTrain:
            self.num = opt.trainset_num
        else:
            self.num = opt.testset_num
        self.CAVE = CAVE
        self.KAIST = KAIST

        # Load physical mask
        data = sio.loadmat(opt.mask_path)
        self.mask = data['mask']
        self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 28))

    def __getitem__(self, index):
        if self.isTrain:
            # Randomly choose between CAVE and KAIST datasets
            d = random.randint(0, 1)
            if d == 0:
                index1 = random.randint(0, self.CAVE.shape[-1] - 1)
                hsi = self.CAVE[:, :, :, index1]
            else:
                index1 = random.randint(0, self.KAIST.shape[-1] - 1)
                hsi = self.KAIST[:, :, :, index1]
        else:
            hsi = self.HSI[:, :, :, index]

        shape = np.shape(hsi)

        # Random spatial crop from HSI
        px = random.randint(0, shape[0] - self.size)
        py = random.randint(0, shape[1] - self.size)
        label = hsi[px:px + self.size, py:py + self.size, :]

        # Random spatial crop from mask (mask is 660x660)
        pxm = random.randint(0, 660 - self.size)
        pym = random.randint(0, 660 - self.size)
        mask_3d = self.mask_3d[pxm:pxm + self.size, pym:pym + self.size, :]

        # Compute shifted 3D mask (CASSI dispersion simulation)
        mask_3d_shift = np.zeros((self.size, self.size + (28 - 1) * 2, 28))
        mask_3d_shift[:, 0:self.size, :] = mask_3d
        for t in range(28):
            mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
        mask_3d_shift_s = np.sum(mask_3d_shift ** 2, axis=2, keepdims=False)
        mask_3d_shift_s[mask_3d_shift_s == 0] = 1

        # Data augmentation (training only)
        if self.isTrain:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            for _ in range(rotTimes):
                label = np.rot90(label)
            for _ in range(vFlip):
                label = label[:, ::-1, :].copy()
            for _ in range(hFlip):
                label = label[::-1, :, :].copy()

        # Simulate CASSI measurement
        temp = mask_3d * label
        temp_shift = np.zeros((self.size, self.size + (28 - 1) * 2, 28))
        temp_shift[:, 0:self.size, :] = temp
        for t in range(28):
            temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2 * t, axis=1)
        meas = np.sum(temp_shift, axis=2)
        input_meas = meas / 28 * 2 * 1.2

        # Shot noise simulation
        QE, bit = 0.4, 2048
        input_meas = np.random.binomial((input_meas * bit / QE).astype(int), QE)
        input_meas = np.float32(input_meas) / np.float32(bit)

        # Convert to tensors
        label = torch.FloatTensor(label.copy()).permute(2, 0, 1)
        input_meas = torch.FloatTensor(input_meas.copy())
        mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)
        mask_3d_shift_s = torch.FloatTensor(mask_3d_shift_s.copy())
        return input_meas, label, mask_3d, mask_3d_shift, mask_3d_shift_s

    def __len__(self):
        return self.num
