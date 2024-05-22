#!/bin/bash

python train_joint_diffusion_cl.py -p configs/standard_diffusion/continual_learning/diffmatch_pooling/5%labels/cifar10.yaml -t 0 -d CL_CIFAR10_SSL5_TASK1_SEED12 -s 12
python train_joint_diffusion_cl.py -p configs/standard_diffusion/continual_learning/diffmatch_pooling/5%labels/cifar10.yaml -t 1 -d CL_CIFAR10_SSL5_TASK2_SEED12 -s 12
python train_joint_diffusion_cl.py -p configs/standard_diffusion/continual_learning/joint_diffusion_pooling/cifar10/cifar10_kl_t2.yaml -t 1 -o logs/CL_CIFAR10_SSL5_TASK1_SEED12/checkpoints/ckpt__epoch=100_global_step=50000.ckpt -l 0 -n logs/CL_CIFAR10_SSL5_TASK2_SEED12/checkpoints/ckpt__epoch=100_global_step=60000.ckpt -c logs/CL_CIFAR10_SSL5_TASK1_SEED12/checkpoints/ckpt__epoch=100_global_step=60000.ckpt -d CL_CIFAR10_SSL5_TASK1-2_SEED12 -s 12
python train_joint_diffusion_cl.py -p configs/standard_diffusion/continual_learning/diffmatch_pooling/5%labels/cifar10.yaml -t 2 -d CL_CIFAR10_SSL5_TASK3_SEED12 -s 12
python train_joint_diffusion_cl.py -p configs/standard_diffusion/continual_learning/joint_diffusion_pooling/cifar10/cifar10_kl_t3.yaml -t 2 -o logs/CL_CIFAR10_SSL5_TASK1-2_SEED12/checkpoints/ckpt__epoch=120_global_step=60000.ckpt -l 0 1 -n logs/CL_CIFAR10_SSL5_TASK3_SEED12/checkpoints/ckpt__epoch=120_global_step=60000.ckpt -c logs/CL_CIFAR10_SSL5_TASK1-2_SEED12/checkpoints/ckpt__epoch=120_global_step=60000.ckpt -d CL_CIFAR10_SSL5_TASK1-3_SEED12 -s 12
python train_joint_diffusion_cl.py -p configs/standard_diffusion/continual_learning/diffmatch_pooling/5%labels/cifar10.yaml -t 3 -d CL_CIFAR10_SSL5_TASK4_SEED12 -s 12
python train_joint_diffusion_cl.py -p configs/standard_diffusion/continual_learning/joint_diffusion_pooling/cifar10/cifar10_kl_t4.yaml -t 3 -o logs/CL_CIFAR10_SSL5_TASK1-3_SEED12/checkpoints/ckpt__epoch=60_global_step=30000.ckpt -l 0 1 2 -n logs/CL_CIFAR10_SSL5_TASK4_SEED12checkpoints/ckpt__epoch=120_global_step=60000.ckpt -c logs/CL_CIFAR10_SSL5_TASK1-3_SEED12/checkpoints/ckpt__epoch=60_global_step=30000.ckpt -d CL_CIFAR10_SSL5_TASK1-4_SEED12 -s 12
python train_joint_diffusion_cl.py -p configs/standard_diffusion/continual_learning/diffmatch_pooling/5%labels/cifar10.yaml -t 4 -d CL_CIFAR10_SSL5_TASK5_SEED12 -s 12
python train_joint_diffusion_cl.py -p configs/standard_diffusion/continual_learning/joint_diffusion_pooling/cifar10/cifar10_kl_t5.yaml -t 4 -o logs/CL_CIFAR10_SSL5_TASK1-4_SEED12/checkpoints/ckpt__epoch=60_global_step=30000.ckpt -l 0 1 2 3 -n logs/CL_CIFAR10_SSL5_TASK5_SEED12/checkpoints/ckpt__epoch=120_global_step=60000.ckpt -c logs/CL_CIFAR10_SSL5_TASK1-4_SEED12/checkpoints/ckpt__epoch=60_global_step=30000.ckpt -d CL_CIFAR10_SSL5_TASK1-5_SEED12 -s 12