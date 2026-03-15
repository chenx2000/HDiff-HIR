import option as opt
import os

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from utils import my_summary, load_checkpoint
from models import model_generator
import torch

# Create model and load checkpoint
model = model_generator(opt.method).cuda()

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
    print("Warning: opt.pretrained_model_path is None, evaluating with untrained model!")

# Evaluate FLOPs and parameter count
my_summary(model, 256, 256, 28, 1)
