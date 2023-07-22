# StillMix
This is our implementation of training the reference networks. 


## Preparing Data
Download the training data. Please refer to [Datasets](../README.md)

## Requirements
- PyTorch >= 1.7.0
- Python >= 3.7.0
- CUDA >= 11.0
- torchvision >= 0.8.0

## Results
We save the probabilities of frames as pkl files. They are released on [Google Drive](https://drive.google.com/drive/folders/1vL0IckepbfUmZh7vEXlp7LIH4BeHhuzm?usp=drive_link).

## Notes
For TSM and Video-Swin-Transformer, we directly use the 2D version of the networks, i.e., ResNet and Swin-Transformer. For SlowFast, we implement the 2D reference network degrading the 3D convolutions into 2D convolutions.