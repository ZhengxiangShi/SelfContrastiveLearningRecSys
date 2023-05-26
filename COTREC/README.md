# COTREC
We simply implement our SCL based on the original code of COTREC model for CIKM'21 paper 'Self-Supervised Graph Co-Training for Session-based Recommendation'.

## Requirements: 
- Python 3.7
- Pytorch 1.6.0

## Setting from the original paper
Best Hyperparameter:
+ Tmall: beta=0.01, alpha=0.005, eps=0.2
+ RetailRocket: beta=0.01, alpha=0.005, eps=0.2
+ Diginetica: beta=0.001, alpha=0.005, eps=0.5

Datasets are available at Dropbox: https://www.dropbox.com/sh/j12um64gsig5wqk/AAD4Vov6hUGwbLoVxh3wASg_a?dl=0 The datasets are already preprocessed and encoded by pickle. Some people may encounter a cudaError in line 50 or line 74 when running our codes if your numpy and pytorch version are different with ours. Currently, we haven't found the solution to resolve the version problem. If you have this problem, please try to change numpy and pytorch version same with ours.

### Running SCL
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset Tmall \
    --epoch 30 \
    --batchSize 100 \
    --embSize 100 \
    --l2 1e-5 \
    --lr 0.001 \
    --layer 2 \
    --beta 0.01 \
    --lam 0.005 \
    --eps 0.2 \
    --temperature 0.1 \
    --item_cl_loss_weight 100.0 \
    --sampled_item_size 30000 \
    --use_item_cl_loss \
    --saved_models_path output/Tmall_item_cl_loss_w1000_t01_ss30000_default

CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset diginetica \
    --epoch 30 \
    --batchSize 100 \
    --embSize 100 \
    --l2 1e-5 \
    --lr 0.001 \
    --layer 2 \
    --beta 0.001 \
    --lam 0.005 \
    --eps 0.5 \
    --temperature 0.1 \
    --item_cl_loss_weight 1.0 \
    --sampled_item_size 33000 \
    --use_item_cl_loss \
    --saved_models_path output/diginetica_item_cl_loss_w10_t01_ss33000_default
```
Please refer to the original respository for reproducing the results of COTREC model. You can use `--remove_original_cl_loss` to remove the original contrastive learning loss in the COTREC model.
