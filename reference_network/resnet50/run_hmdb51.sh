#!/bin/sh

#SBATCH -o run-%j.output
#SBATCH -p DGXq
#SBATCH -w node19

module load cuda11.2/toolkit/11.2.0

export CUDA_VISIBLE_DEVICES=6

python main_frame.py \
--num_class 51 \
--data_dir /export/home/xxx/xxx/datasets/HMDB51/jpegs_256 \
--train_list /export/home/xxx/xxx/work/dataset_config/HMDB51/lists/trainlist01.txt \
--val_list /export/home/xxx/xxx/work/dataset_config/HMDB51/lists/trainlist01.txt \
--train \
--arch resnet50 --pretrain \
--batch_size 256 --lr 0.04 --wd 1e-5 \
--epochs 50 --lr_step 20 40 \
--model_name hmdb51-split01-resnet50-pretrain

python main_frame.py \
--num_class 51 \
--data_dir /export/home/xxx/xxx/datasets/HMDB51/jpegs_256 \
--test_list /export/home/xxx/xxx/work/dataset_config/HMDB51/lists/trainlist01.txt \
--test --save_test_results \
--batch_size 1024 \
--model_name hmdb51-split01-resnet50-pretrain
