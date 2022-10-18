#!/bin/bash
WORKSPACE=${1:-"./workspace"}  # Default workspace directory

echo "WORKSPACE=${WORKSPACE}"

# Users can modify the following config file.
INDEXES_CONFIG_YAML="./script/2_create_indexes/config.yaml"

# Create indexes for training.
python3 moyu/data/create_indexes/main.py \
    --workspace=$WORKSPACE \
    --config_yaml=$INDEXES_CONFIG_YAML
