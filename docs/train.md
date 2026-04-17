# Gaussian Grouping: Segment and Edit Anything in 3D Scenes

## 1. Prepare associated SAM masks

### 1.1 Pre-converted datasets
We provide converted datasets in our paper, You can use directly train on datasets from [hugging face link](https://huggingface.co/mqye/Gaussian-Grouping/tree/main)

```
data
|____bear
|____lerf
| |____figurines
|____mipnerf360
| |____counter
```


### 1.2 (Optional) Prepare your own datasets
For your custom dataset, you can follow this step to create masks for training. If you want to prepare masks on your own dataset, you will need [DEVA](../Tracking-Anything-with-DEVA/README.md) python environment and checkpoints.


```
<location>
|---input
    |---<image 0>
    |---<image 1>
    |---...
```
Firstly, convert initial camera pose and point cloud with colmap
```
python convert.py -s <location>
```

Then, convert SAM associated object masks. Note that the quality of converted-mask will largely affect the results of 3D segmentation and editing. And getting the mask is very fast. So it is best to adjust the parameters of anything segment first to make the mask as consistent and reasonable as possible from multiple views.

Example1. Bear dataset
```
bash script/prepare_pseudo_label.sh bear 1
```

Example2. figurines dataset
```
bash script/prepare_pseudo_label.sh lerf/figurines 1
```

Example3. counter dataset
```
bash script/prepare_pseudo_label.sh mipnerf360/counter 2
```

## 2. Training and Masks Rendering

### 2.1 Main BCOG entry

The current research branch is organized around:

`Boundary-Conditioned Object-Aware Grouping (BCOG)`

Recommended training entry:

```bash
bash script/train_bcog.sh <dataset_name> <scale> [output_name]
```

Examples:

```bash
bash script/train_bcog.sh bear 1 bear_bcog
bash script/train_bcog.sh lerf/figurines 1 figurines_bcog
bash script/train_bcog.sh mipnerf360/counter 2 counter_bcog
```

This script will:

1. Train with `train.py`
2. Use `config/gaussian_dataset/train_bcog.json` by default
3. Render masks after training

### 2.2 Direct Python entry

If you prefer to run training manually, the Python entry is:

```bash
python train.py \
    -s data/<dataset_name> \
    -r <scale> \
    -m output/<output_name> \
    --config_file config/gaussian_dataset/train_bcog.json \
    --train_split \
    --eval
```

Example:

```bash
python train.py -s data/lerf/figurines -r 1 -m output/figurines_bcog --config_file config/gaussian_dataset/train_bcog.json --train_split --eval
```

### 2.3 Configs

The branch now keeps a small config hierarchy with `extends` support:

- `config/gaussian_dataset/train_base.json`: shared reconstruction + sugar + reliability-graph defaults
- `config/gaussian_dataset/train.json`: graph-only baseline, used for Stage B style ablations
- `config/gaussian_dataset/train_proto.json`: graph + prototype bank, used for Stage C ablations
- `config/gaussian_dataset/train_bcog.json`: full BCOG recipe with late-stage prototype-disagreement-guided split

You can override the config in the shell script:

```bash
bash script/train_bcog.sh lerf/figurines 1 figurines_graph_only --config config/gaussian_dataset/train.json
```

### 2.4 Legacy scripts

The original repository scripts such as `script/train.sh` and `script/train_lerf.sh` are kept for backward compatibility with the base Gaussian Grouping workflow.
For this branch, prefer `script/train_bcog.sh`.

