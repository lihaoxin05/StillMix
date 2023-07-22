#!/bin/sh

#SBATCH -o run-%j.output
#SBATCH -p DGXq
#SBATCH -w node18

module load cuda11.2/toolkit/11.2.0

export CUDA_VISIBLE_DEVICES=0

python main_frame.py \
--num_class 101 \
--data_dir /export/home/xxx/xxx/datasets/UCF101/jpegs_256 \
--train_list /export/home/xxx/xxx/work/dataset_config/UCF101/lists/trainlist01.txt \
--val_list /export/home/xxx/xxx/work/dataset_config/UCF101/lists/trainlist01.txt \
--train \
--arch resnet50 --pretrain \
--batch_size 256 --lr 0.04 --wd 1e-5 \
--epochs 50 --lr_step 20 40 \
--model_name ucf101-split01

python main_frame.py \
--num_class 101 \
--data_dir /export/home/xxx/xxx/datasets/UCF101/jpegs_256 \
--test_list /export/home/xxx/xxx/work/dataset_config/UCF101/lists/trainlist01.txt \
--test --save_test_results \
--batch_size 256 \
--model_name ucf101-split01
