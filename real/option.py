import argparse

parser = argparse.ArgumentParser(description="HDiff-HIR (Real CASSI)")

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Output directory
parser.add_argument('--outf', type=str, default='./real/diffusion/', help='saving_path')

# Dataset paths
parser.add_argument('--data_path_CAVE', type=str, default='../../datasets/CAVE_512_28/', help='path of CAVE data')
parser.add_argument('--data_path_KAIST', type=str, default='../../datasets/KAIST_CVPR2021/', help='path of KAIST data')
parser.add_argument('--mask_path', type=str, default='../../datasets/TSA_real_data/mask.mat', help='path of mask')

# Model specifications
parser.add_argument('--method', type=str, default='hdiff-l', help='method name (hdiff-s, hdiff-m, hdiff-l)')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument("--input_setting", type=str, default='Y', help='the input measurement of the network')
parser.add_argument("--input_mask", type=str, default='Mask', help='the input mask of the network')

# Diffusion parameters
parser.add_argument("--timesteps", type=int, default=4000)

# Training parameters
parser.add_argument("--size", default=384, type=int, help='cropped patch size')
parser.add_argument("--seed", default=3407, type=int, help='Random_seed')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--isTrain", default=True, type=bool, help='train or test')
parser.add_argument("--max_epoch", type=int, default=2000, help='total epoch')
parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, nargs='+', default=[50, 100, 150, 200, 250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--learning_rate", type=float, default=0.0004)

# Number of training samples per epoch
parser.add_argument("--epoch_sam_num", default=5000, type=int, help='total number of trainset')

opt = parser.parse_args()

opt.trainset_num = 20000 // ((opt.size // 96) ** 2)

for arg in vars(opt):
    val = vars(opt)[arg]
    if val == 'True':
        vars(opt)[arg] = True
    elif val == 'False':
        vars(opt)[arg] = False

# Unpack args to module-level variables for backward compatibility
for arg, val in vars(opt).items():
    globals()[arg] = val
