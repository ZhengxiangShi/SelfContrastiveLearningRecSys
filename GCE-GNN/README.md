# GCE-GNN
We simply implement our SCL based on the original code of GCE-GNN model for SIGIR 2020 Paper: _Global Context Enhanced Graph Neural Networks for Session-based Recommendation_.

## Requirements
- Python 3
- PyTorch >= 1.3.0
- tqdm

## Usage
Data preprocessing:

The code for data preprocessing can refer to [SR-GNN](https://github.com/CRIPAC-DIG/SR-GNN).

Train and evaluate the model:
```
python build_graph.py --dataset diginetica --sample_num 12
python main.py --dataset diginetica
```

## Running SCL
Increase the item_cl_loss_weight does not improve the result on datasets
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset Tmall \
    --epoch 20 \
    --batch_size 100 \
    --temperature 0.1 \
    --item_cl_loss_weight 0.1 \
    --sampled_item_size 32000 \
    --use_item_cl_loss \
    --saved_models_path output/Tmall_item_cl_loss_w01_t01_ss32000_default

CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset Nowplaying \
    --epoch 20 \
    --batch_size 100 \
    --temperature 0.01 \
    --item_cl_loss_weight 0.5 \
    --sampled_item_size 32000 \
    --use_item_cl_loss \
    --saved_models_path output/Nowplaying_item_cl_loss_w05_t001_ss32000_default

CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset diginetica\
    --epoch 20 \
    --batch_size 100 \
    --temperature 0.1 \
    --item_cl_loss_weight 0.1 \
    --sampled_item_size 32000 \
    --use_item_cl_loss \
    --saved_models_path output/diginetica_item_cl_loss_w01_t01_ss32000_default
```
Please refer to the original repository for reproducing the results of GCE-GNN model.
