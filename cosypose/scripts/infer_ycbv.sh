#!/bin/bash
# Bash script to run CosyPose inference on YCB Video
# Author: Ziqi Lu ziqilu@mit.edu
# Copyright 2023 The Ambitious Folks of the MRG

# NOTE: Don't forget to "conda activate cosypose"
# Whether to use model trained on pbr or real data
model=${1:-'pbr'}
# Path to YCB video data
ycbv_path=${2:-'/media/ziqi/Extreme SSD/data/bop_datasets/ycbv/train_real/'}
# Folder to save prediction results and video
out_path=${3:-'/home/ziqi/Desktop/test/'}
# Make prediction video and save as mp4
vis=${4:-false}

for folder in "$ycbv_path"/*/
do
    out_file="$out_path"/$(basename "$folder").csv
    if [ "$vis" = true ] ; then
        python3 -m cosypose.scripts.run_inference\
            --img_path "$folder"/rgb/ --out "$out_file" --train "$model" --plot
    else
        python3 -m cosypose.scripts.run_inference\
            --img_path "$folder"/rgb/ --out "$out_file" --train "$model"
    fi    
done
