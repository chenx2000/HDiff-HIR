# HDiff-HIR: Hierarchically Conditional Diffusion Model for Hyperspectral Image Reconstruction
This repo is the implementation of [HDiff-HIR: Hierarchically Conditional Diffusion Model for Hyperspectral Image Reconstruction](https://ieeexplore.ieee.org/abstract/document/11121894) (TCSVT2025).

# Comparison on Simulation Dataset
The performance are reported on 10 scenes of the KAIST dataset. The test size of FLOPS is 256 x 256.
## Quantitative Results

|                                  Method                                   | Params (M) | FLOPS (G) | PSNR  | SSIM  |
|:-------------------------------------------------------------------------:|:----------:|:---------:|:-----:|:-----:|
| [TSA-Net](https://link.springer.com/chapter/10.1007/978-3-030-58592-1_12) |   44.25    |  110.06   | 31.46 | 0.894 |
|                 [HDNet](https://arxiv.org/abs/2203.02149)                 |    2.37    |  154.76   | 34.97 | 0.943 |
|                  [MST](https://arxiv.org/abs/2111.07910)                  |    2.03    |   28.15   | 35.18 | 0.948 |
|                  [CST](https://arxiv.org/abs/2203.04845)                  |    3.00    |   40.10   | 36.12 | 0.957 |
|                             HDiff-HIR (ours)                              |   13.80    |   74.48   | 36.88 | 0.966 |

## Qualitative Results

Download results of HDiff-HIR ([Google Drive](https://drive.google.com/drive/folders/1iAYFQI1ba7YDqVMRvp7g9eaQMreIjk3b?usp=sharing)).

# Usage
## Prepare Dataset
Download cave_1024_28 ([Baidu Disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ), code: `fo0q` | [One Drive](https://bupteducn-my.sharepoint.com/:f:/g/personal/mengziyi_bupt_edu_cn/EmNAsycFKNNNgHfV9Kib4osB7OD4OSu-Gu6Qnyy5PweG0A?e=5NrM6S)), CAVE_512_28 ([Baidu Disk](https://pan.baidu.com/s/1ue26weBAbn61a7hyT9CDkg), code: `ixoe` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EjhS1U_F7I1PjjjtjKNtUF8BJdsqZ6BSMag_grUfzsTABA?e=sOpwm4)), KAIST_CVPR2021 ([Baidu Disk](https://pan.baidu.com/s/1LfPqGe0R_tuQjCXC_fALZA), code: `5mmn` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EkA4B4GU8AdDu0ZkKXdewPwBd64adYGsMPB8PNCuYnpGlA?e=VFb3xP)), TSA_simu_data ([Baidu Disk](https://pan.baidu.com/s/1LI9tMaSprtxT8PiAG1oETA), code: `efu8` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFDwE-7z1fzeWCRDA?e=ofvwrD)), TSA_real_data ([Baidu Disk](https://pan.baidu.com/s/1RoOb1CKsUPFu0r01tRi5Bg), code: `eaqe` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFTpCwLdTi_eSw6ww?e=uiEToT)), and then put them into the corresponding folders of `datasets/` and recollect them as the following form:

```shell
|--HDiff-HIR
    |--real
        |-- dataset.py
        |-- option.py
        |-- train.py
        |-- sample.py
    |--simulation
        |-- dataset.py
        |-- option.py
        |-- train.py
        |-- sample.py
|--datasets
    |--cave_1024_28
        |--scene1.mat
        ：  
    |--CAVE_512_28
        |--scene1.mat
        ：  
    |--KAIST_CVPR2021  
        |--1.mat
        ： 
    |--TSA_simu_data  
        |--mask.mat   
        |--Truth
            |--scene01.mat
            ： 
    |--TSA_real_data  
        |--mask.mat   
        |--Measurements
            |--scene1.mat
            ： 
```

Following [CST](https://arxiv.org/abs/2203.04845) and [DAUHST](https://arxiv.org/abs/2205.10102), we use the CAVE dataset (cave_1024_28) as the simulation training set. Both the CAVE (CAVE_512_28) and KAIST (KAIST_CVPR2021) datasets are used as the real training set. 

## Simulation Experiement
### Training
```shell
cd HDiff-HIR/simulation/
python train.py --method hdiff-l
```
The training log, trained model, and reconstructed HSI will be saved in `HDiff-HIR/simulation/exp/test/` (configurable via `--outf` in `option.py`). 

### Testing (Sampling)
```shell
cd HDiff-HIR/simulation/
python sample.py --method hdiff-l
```
The sampled results will be saved to the directory specified in the `sample.py` script.

## Real Experiement
### Training
```shell
cd HDiff-HIR/real/
python train.py --method hdiff-l
```
The training log, trained model, and reconstructed HSI will be saved in `HDiff-HIR/real/real/diffusion/`.

### Testing (Sampling)
```shell
cd HDiff-HIR/real/
python sample.py --method hdiff-l
```
The sampled results will be output into the corresponding test directory inside `real/`.

### Visualization

For the visualization toolkit, please see [DSMT-main](https://github.com/chenx2000/DSMT-main/tree/main).

# Citation
If this code helps you, please consider citing our work:
```shell
@ARTICLE{luo2026hdiff,
  author={Luo, Fulin and Chen, Xi and Fu, Chuan and Guo, Tan and Du, Bo},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={HDiff-HIR: Hierarchically Conditional Diffusion Model for Hyperspectral Image Reconstruction}, 
  year={2026},
  volume={36},
  number={1},
  pages={777-791},
}
```
