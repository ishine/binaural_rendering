#!/bin/bash
AMBISONIC_BINAURAL_DATASET_DIR=${1:-"./dataset/ambisonic-binaural"}  # The first argument is dataset directory.
WORKSPACE=${2:-"./workspace"}  # The second argument is workspace directory.

AMBISONIC_BINAURAL_DATASET_DIR=`realpath ${AMBISONIC_BINAURAL_DATASET_DIR}`
echo "AMBISONIC_BINAURAL_DATASET_DIR=${AMBISONIC_BINAURAL_DATASET_DIR}"

WORKSPACE=$(realpath ${WORKSPACE})
echo $(WORKSPACE=${WORKSPACE})

for SPLIT in "train" "test"
do
    for SOURCE_TYPE in 'ambisonic' 'binaural'
    do
        python3 moyu/data/pack_audios_to_hdf5s/ambisonic-binaural.py \
            --audios_dir="${AMBISONIC_BINAURAL_DATASET_DIR}/${SPLIT}/${SOURCE_TYPE}" \
            --hdf5s_dir="${WORKSPACE}/hdf5/ambisonic-binaural/${SPLIT}/${SOURCE_TYPE}" \
            --sample_rate=48000
    done
done

mkdir "$WORKSPACE/evaluation_audio"
ln -s "$AMBISONIC_BINAURAL_DATASET_DIR" "$WORKSPACE/evalution_audio/ambisonic-binaural"
