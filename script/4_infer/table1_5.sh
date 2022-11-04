#!/bin/bash
WORKSPACE=${1:-"./workspace/ambisonic-binaural"}  # The first argument is workspace directory.

echo "WORKSPACE=${WORKSPACE}"


MODEL="gru"
STEP=100000
DEVICE=0

for CONFIG in "table1_5" # Add more configs...
do
    CONFIG_YAML="./script/3_train/$CONFIG/config.yaml"
    CHECKPOINT_PATH="./workspace/checkpoint/$CONFIG/step=$STEP.pth"

    # infer
    CUDA_VISIBLE_DEVICES=$DEVICE python3 moyu/infer.py  infer_file\
        --input_path="./resource/ambisonic-binaural/ambisonic/test_240s_to_260s.wav" \
        --output_path="./result/ambisounic-binaural/240_260/$CONFIG.wav" \
        --target_path="./resource/ambisonic-binaural/binaural/test_240s_to_260s.wav" \
        --config_yaml=$CONFIG_YAML \
        --checkpoint_path=$CHECKPOINT_PATH
done