#!/bin/bash

# Suppress TensorFlow and MediaPipe logs
export TF_CPP_MIN_LOG_LEVEL=3
export GLOG_minloglevel=3

# Execute training, redirect noisy logs (stderr) to err.txt
python3 train_attack.py \
    --data celeba \
    --epochs 100 \
    --batch_size 256 \
    --attack_mode HCBsmile \
    2> err.txt
