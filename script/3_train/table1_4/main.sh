#!/bin/bash
WORKSPACE=${1:-"./workspace"}  # The first argument is workspace directory.

echo "WORKSPACE=${WORKSPACE}"

TRAIN_CONFIG_YAML="./script/4_train/table1_4/config.yaml"

# Train & evaluate & save checkpoints.
CUDA_VISIBLE_DEVICES="0" \
python3 moyu/train.py \
    --workspace=$WORKSPACE \
    --gpus=1 \
    --config_yaml=$TRAIN_CONFIG_YAML