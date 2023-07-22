#!/bin/sh

#SBATCH -o run-%j.output
#SBATCH -p DGXq
#SBATCH -w node20

module load cuda11.2/toolkit/11.2.0

export CUDA_VISIBLE_DEVICES=2

python main_video.py \
--num_class 400 \
--data_dir /export/home/xxx/xxx/datasets/Kinetics_400 \
--train_list /export/home/xxx/xxx/work/dataset_config/Kinetics_400/lists/train.txt \
--val_list /export/home/xxx/xxx/work/dataset_config/Kinetics_400/lists/train.txt \
--train \
--arch resnet50 --pretrain \
--batch_size 256 --lr 0.04 --wd 1e-5 \
--epochs 50 --lr_step 20 40 \
--model_name kinetics400-train-resnet50-pretrain

python main_video.py \
--num_class 400 \
--data_dir /export/home/xxx/xxx/datasets/Kinetics_400 \
--test_list /export/home/xxx/xxx/work/dataset_config/Kinetics_400/lists/train.txt \
--test --save_test_results \
--batch_size 256 \
--model_name kinetics400-train-resnet50-pretrain
