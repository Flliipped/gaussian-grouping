#!/bin/bash

set -e

usage() {
    echo "Usage:"
    echo "  $0 <dataset_name> <scale> [output_name]"
    echo "  $0 <dataset_name> <scale> -m <output_name>"
    exit 1
}

# 至少需要 2 个参数
if [ "$#" -lt 2 ]; then
    usage
fi

dataset_name="$1"
scale="$2"
shift 2

dataset_folder="data/$dataset_name"

if [ ! -d "$dataset_folder" ]; then
    echo "Error: Folder '$dataset_folder' does not exist."
    exit 2
fi

# 默认输出目录
model_path="output/${dataset_name}"

# 解析可选参数
if [ "$#" -gt 0 ]; then
    if [ "$1" = "-m" ]; then
        if [ "$#" -ne 2 ]; then
            usage
        fi
        output_name="$2"
        model_path="output/${output_name}"
    else
        if [ "$#" -ne 1 ]; then
            usage
        fi
        output_name="$1"
        model_path="output/${output_name}"
    fi
fi

echo "Dataset folder: ${dataset_folder}"
echo "Scale: ${scale}"
echo "Model output: ${model_path}"

# Gaussian Grouping training
python train.py \
    -s "${dataset_folder}" \
    -r "${scale}" \
    -m "${model_path}" \
    --config_file config/gaussian_dataset/train.json

# Segmentation rendering using trained model
python render.py \
    -m "${model_path}" \
    --num_classes 256