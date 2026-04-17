#!/bin/bash

set -e

usage() {
    echo "Usage:"
    echo "  $0 <dataset_name> <scale> [output_name]"
    echo "  $0 <dataset_name> <scale> -m <output_name>"
    echo "  $0 <dataset_name> <scale> [output_name] [--no-eval] [--wandb] [--config <config_file>]"
    echo "  $0 <dataset_name> <scale> -m <output_name> [--no-eval] [--wandb] [--config <config_file>]"
    exit 1
}

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

model_path="output/${dataset_name}"
use_eval=true
use_wandb=false
config_file="config/gaussian_dataset/train_bcog.json"

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
        --wandb)
            use_wandb=true
            shift
            ;;
        --config)
            if [ -z "$2" ]; then
                usage
            fi
            config_file="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            model_path="output/$1"
            shift
            ;;
    esac
done

echo "Dataset folder: ${dataset_folder}"
echo "Scale: ${scale}"
echo "Model output: ${model_path}"
echo "Eval enabled: ${use_eval}"
echo "Wandb enabled: ${use_wandb}"
echo "Config file: ${config_file}"

if [ ! -f "$config_file" ]; then
    echo "Error: Config file '$config_file' does not exist."
    exit 3
fi

train_cmd=(
    python train.py
    -s "${dataset_folder}"
    -r "${scale}"
    -m "${model_path}"
    --config_file "${config_file}"
    --train_split
)

if [ "${use_eval}" = true ]; then
    train_cmd+=(--eval)
fi
if [ "${use_wandb}" = true ]; then
    train_cmd+=(--use_wandb)
fi

echo "Running training command:"
printf ' %q' "${train_cmd[@]}"
echo
"${train_cmd[@]}"

render_cmd=(
    python render.py
    -m "${model_path}"
    --num_classes 256
)

echo "Running render command:"
printf ' %q' "${render_cmd[@]}"
echo
"${render_cmd[@]}"
