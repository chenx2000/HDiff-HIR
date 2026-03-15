import option as opt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

import torch
import torch.utils.data as tud
import time
import datetime
import numpy as np

from models import model_generator
from diffusion import create_diffusion
from dataset import RealCassiDataset
from utils import (
    prepare_data_cave, prepare_data_KAIST,
    init_mask, time2file_name, gen_log,
)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

seed = opt.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# ===================================================================
# Setup
# ===================================================================

# Load training data
CAVE = prepare_data_cave(opt.data_path_CAVE, 30)
KAIST = prepare_data_KAIST(opt.data_path_KAIST, 30)

# Output paths
date_time = time2file_name(str(datetime.datetime.now()))
outf = os.path.join(opt.outf, date_time)
os.makedirs(outf, exist_ok=True)

# Model
model = model_generator(opt.method).cuda()

# Diffusion process (DDPM training + DDIM sampling)
diffusion = create_diffusion(
    timesteps=opt.timesteps,
    beta_schedule='linear',
    parameterization='x0',
    ddim_sampling_steps='ddim4',
    loss_type='l1',
)

# Optimizer and scheduler
optimizer = torch.optim.Adam(
    model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999)
)
if opt.scheduler == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=opt.milestones, gamma=opt.gamma
    )
elif opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, opt.max_epoch, eta_min=1e-6
    )


# ===================================================================
# Training
# ===================================================================

if __name__ == "__main__":
    print("Random Seed:", seed)
    logger = gen_log(outf)
    logger.info(
        "Learning rate:{}, batch_size:{}, method:{}, gpu_id:{}\n".format(
            opt.learning_rate, opt.batch_size, opt.method, opt.gpu_id
        )
    )

    for epoch in range(1, opt.max_epoch + 1):
        model.train()
        Dataset = RealCassiDataset(opt, CAVE, KAIST)
        loader_train = tud.DataLoader(
            Dataset, num_workers=8, batch_size=opt.batch_size, shuffle=True
        )

        epoch_loss = 0
        batch_num = int(np.floor(len(Dataset) / opt.batch_size))
        timesteps_sequence = torch.randint(
            0, opt.timesteps, (batch_num, opt.batch_size)
        ).cuda().long()
        start_time = time.time()

        for i, (input_meas, label, Mask, Phi, Phi_s) in enumerate(loader_train):
            input_meas = input_meas.cuda()
            label = label.cuda()
            Phi = Phi.cuda()
            Phi_s = Phi_s.cuda()

            input_mask = init_mask(Mask, Phi, Phi_s, opt.input_mask)
            input_mask = input_mask.cuda().permute(0, 3, 1, 2)

            loss = diffusion.p_losses(
                model, label, timesteps_sequence[i],
                condition=input_meas, mask=input_mask,
            )

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print('%4d %4d / %4d loss = %.10f time = %s' % (
                    epoch, i, len(Dataset) // opt.batch_size,
                    epoch_loss / ((i + 1) * opt.batch_size),
                    datetime.datetime.now(),
                ))

        elapsed_time = time.time() - start_time
        logger.info(
            'epoch = %4d , loss = %.10f , time = %4.2f s' % (
                epoch, epoch_loss / len(Dataset), elapsed_time
            )
        )
        scheduler.step()

        # Save checkpoint (state dict format)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(outf, 'model_%03d.pth' % epoch))
