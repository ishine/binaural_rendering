#!/bin/bash
mkdir -p dataset
cd dataset

DATASET_URL="https://zenodo.org/record/7212810/files/ambisonic-binaural.zip?download=1"

wget $DATASET_URL -O "ambisonic-binaural.zip"

unzip  "ambisonic-binaural.zip"