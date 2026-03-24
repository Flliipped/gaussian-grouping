#!/bin/bash

set -e

usage() {
    echo "Usage:"
    echo "  $0 <dataset_name> <scale> [output_name]"
    echo "  $0 <dataset_name> <scale> -m <output_name>"
    echo "  $0 <dataset_name> <scale> [output_name] [--no-eval]"
    echo "  $0 <dataset_name> <scale> -m <output_name> [--no-eval]"
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

# 默认开启 eval
use_eval=true

# 解析剩余参数
while [ "$#" -gt 0 ]; do
    case "$1" in
        -m)
            if [ -z "$2" ]; then
                usage
            fi
            model_path="output/$2"
            shift 2
            ;;
        --no-eval)
            use_eval=false
            shift
            ;;
        --eval)
            use_eval=true
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            # 位置参数作为 output_name
            model_path="output/$1"
            shift
            ;;
    esac
done

echo "Dataset folder: ${dataset_folder}"
echo "Scale: ${scale}"
echo "Model output: ${model_path}"
echo "Eval enabled: ${use_eval}"

# 训练命令
train_cmd=(
    python train.py
    -s "${dataset_folder}"
    -r "${scale}"
    -m "${model_path}"
    --config_file config/gaussian_dataset/train.json
    --train_split
)

if [ "${use_eval}" = true ]; then
    train_cmd+=(--eval)
fi

echo "Running training command:"
printf ' %q' "${train_cmd[@]}"
echo

"${train_cmd[@]}"

# 渲染命令
render_cmd=(
    python render.py
    -m "${model_path}"
    --num_classes 256
)

echo "Running render command:"
printf ' %q' "${render_cmd[@]}"
echo

"${render_cmd[@]}"