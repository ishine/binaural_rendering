# Ambisonic-Based Binaural Rendering

## Prepare
- Download dataset from [https://zenodo.org/record/7212810#.Y2S10exBw-R] and unzip. Alternatively, running the following command:

    `bash script/0_download_and_unzip_dataset/main.sh`

- Pack audios to hdf5 files by running the following command:

    `bash script/1_pack_audios_to_hdf5s/main.sh`

- Create indexes for samping by running the following command:

    `bash script/2_create_indexes/main.sh`
## Train
We provide scripts of all tables shown in our paper. For example, the Table 1 is:

 | Config | SDR | LSD | Q | T | L | I |
 | ------ | --- | --- | - | - | - | - |
Oracle | $\infty$ | 0 | 4.5 (0) | 4.50 (0) | 4.5 (0) | 4.5 (0) 
sp-HRTF | -0.79 | 1.88 | 3.58 (0.45) | 2.67 (0.75) | 3.37 (0.72) | 3.25 (0.69) 
(table1_1) DNN-4 | 5.58 | 0.95 | 1.78 (0.66) | 2.25 (0.99) | 3.08 (1.01) | 2.17 (0.90) 
 GRU-4 | 7.32 | 0.95 | **3.83** (0.37) | **3.58** (0.73) | **3.87** (0.39) | **3.58**(0.45) 
 UNet-41 | 8.03 | 0.93 | 3.5 (0.65) | 3.47 (0.47) | 3.95 (0.21) | 3.5 (0.29)
 GRU-4 (2 mix) | 9.17 | 0.88 | 3.42 (0.19) | 3.33 (0.47) | 3.42 (0.45) | 3.33 (0.37) 
 (table1_5) GRU-4 (4 mix) | **9.86** | **0.85** | 3.42 (0.45) | 3.33 (0.47) | 3.42 (0.53) | 3.20 (0.58) 

To reproduce the GRU-4 (4 mix) model, running the following command:

`bash script/3_train/table1_5/main.sh`

## Infer
We provide two kinds of script for inference. The first is the traditional SP-HRTF method, which can be done by running the following command:

`bash script/4_infer/sp-hrtf.sh`

The second is for evaluating the trained model, which can be done by running the following command:

`bash script/4_infer/table1_5.sh`