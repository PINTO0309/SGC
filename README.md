# SGC
Classification of wearing vs. not wearing sunglasses.

## Install

```bash
uv sync
source .venv/bin/activate
```

## Train

```bash
SIZE=48x48
VAR=p
uv run python -m sgc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/sgc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 64 \
--num_blocks 1 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
```bash
SIZE=48x48
VAR=n
uv run python -m sgc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/sgc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 64 \
--num_blocks 2 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
```bash
SIZE=48x48
VAR=t
uv run python -m sgc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/sgc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 64 \
--num_blocks 3 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
```bash
SIZE=48x48
VAR=s
uv run python -m sgc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/sgc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 64 \
--num_blocks 4 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
```bash
SIZE=48x48
VAR=l
uv run python -m sgc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/sgc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 64 \
--num_blocks 8 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
