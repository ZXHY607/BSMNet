#!/usr/bin/env bash
clear

set -x
DATAPATH="/data/zh/data/DrivingStereo/"

CUDA_VISIBLE_DEVICES=3 python main.py --dataset drivingstereo \
    --datapath $DATAPATH --trainlist filenames/Drivingstereo_34888_train_list.txt --testlist filenames/Drivingstereo_6645_test_list.txt \
     --batch_size 1 --test_batch_size 1 \
    --epochs 60 --lrepochs "14,18:2" \
    --model bsmnet --logdir ./checkpoints/test1 \
    # --loadckpt checkpoints/interval/checkpoint_000028.ckpt 
   