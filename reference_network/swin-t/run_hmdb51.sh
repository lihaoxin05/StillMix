#!/bin/sh

#SBATCH -o run-%j.output
#SBATCH -p DGXq
#SBATCH -w node19

module load cuda11.2/toolkit/11.2.0

export CUDA_VISIBLE_DEVICES=7

python main_frame.py \
--num_class 51 \
--data_dir /export/home/xxx/xxx/datasets/HMDB51/jpegs_256 \
--train_list /export/home/xxx/xxx/work/dataset_config/HMDB51/lists/trainlist01.txt \
--val_list /export/home/xxx/xxx/work/dataset_config/HMDB51/lists/trainlist01_val.txt \
--train \
--arch swin_t --pretrain \
--batch_size 64 --lr 0.01 --wd 1e-5 \
--epochs 50 --lr_step 20 40 \
--model_name hmdb51-split01

python main_frame.py \
--num_class 51 \
--data_dir /export/home/xxx/xxx/datasets/HMDB51/jpegs_256 \
--test_list /export/home/xxx/xxx/work/dataset_config/HMDB51/lists/trainlist01.txt \
--test --save_test_results \
--batch_size 256 \
--model_name hmdb51-split01

