#!/bin/bash
# infer
python3 moyu/infer.py  infer_file\
    --input_path="./resource/ambisonic-binaural/ambisonic/test_240s_to_260s.wav" \
    --output_path="./result/ambisounic-binaural/240_260/sp-hrtf.wav" \
    --target_path="./resource/ambisonic-binaural/binaural/test_240s_to_260s.wav" \
    --use_hrir
