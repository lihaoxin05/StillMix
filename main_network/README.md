# StillMix
This is our implementation of training the main networks. 

This project is built based on [MMAction2](https://github.com/open-mmlab/mmaction2) and [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer).

## Preparing Data
1. Download the training and evaluation data. Please refer to [Datasets](../README.md)
2. Prepare the files containing frame probabilities (produced by the reference networks). We provide the files on [Google Drive](https://drive.google.com/drive/folders/1vL0IckepbfUmZh7vEXlp7LIH4BeHhuzm?usp=drive_link).

## TSM and SlowFast
We use [MMAction2](https://github.com/open-mmlab/mmaction2) for our experiments of TSM and SlowFast. 

### Installation
1. Create environment. Please refer to [MMAction2](https://github.com/open-mmlab/mmaction2). We also provide a conda config file [here](mmaction2-pt1.10.0.yml).
2. Download the folder [mmaction2](mmaction2) and ```pip install -v -e .```.

### Training and inference
Most hyper-parameters are in the configuration files in the path [mmaction2/configs/stillmix](mmaction2/configs/stillmix). You can also specify ```--cfg-options``` to in-place modify the config when submitting the jobs.

The training and inference scripts are in the path [mmaction2/work_dir](mmaction2/work_dir).


## Video Swin Transformer
We use [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) for our experiments of Video Swin Transformer.

### Installation
1. Create environment. Please refer to [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer). We also provide a conda config file [here](swin-pt1.10.0.yml).
2. Download the folder [Video-Swin-Transformer](Video-Swin-Transformer) and ```pip install -v -e .```.

### Training and inference
Most hyper-parameters are in the configuration files in the path [Video-Swin-Transformer/configs/stillmix](Video-Swin-Transformer/configs/stillmix). You can also specify ```--cfg-options``` to in-place modify the config when submitting the jobs.

The training and inference scripts are in the path [Video-Swin-Transformer/work_dir](Video-Swin-Transformer/work_dir).
