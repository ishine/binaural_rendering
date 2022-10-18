#!/bin/bash
WORKSPACE=${1:-"./workspace"}  # The first argument is workspace directory.

echo "WORKSPACE=${WORKSPACE}"

TRAIN_CONFIG_YAML="./script3/3_train/table4_3/config.yaml"

# Train & evaluate & save checkpoints.
CUDA_VISIBLE_DEVICES="1" \
python3 moyu/train.py \
    --workspace=$WORKSPACE \
    --gpus=1 \
    --config_yaml=$TRAIN_CONFIG_YAML