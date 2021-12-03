#!/usr/bin/env bash
set -x
DATAPATH="/data/zh/data/DrivingStereo/"
clear

CUDA_VISIBLE_DEVICES=0,1 python test-csl.py --dataset drivingstereo \
    --datapath $DATAPATH --trainlist ./filenames/DrivingStereo_train_list_34000.txt --testlist filenames/ds-test-all.txt \
     --batch_size 4 --test_batch_size 1 \
    --epochs 60 --lrepochs "14,16,18:2" \
    --model gwcnet-gc --logdir ./checkpoints/test \
    --loadckpt checkpoints/uncertainty-driving-scale-att/checkpoint_000034.ckpt > test-all.txt

#uncertainty-driving-scale-att final
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset drivingstereo \
#    --datapath $DATAPATH --trainlist ./filenames/DrivingStereo_train_list_34000.txt --testlist ./filenames/DrivingStereo_test_list.txt \
#     --batch_size 4 --test_batch_size 4 \
#    --epochs 25 --lrepochs "8,10,12,14,16:2" \
#    --model gwcnet-gc --logdir ./checkpoints/sceneflow-driving/gwcnet-gc \
#    --loadckpt ./checkpoints/sceneflow-driving/gwcnet-gc/checkpoint_000011.ckpt