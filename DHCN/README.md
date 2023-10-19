# DHCN


### Requirements
Environments: Python3, Pytorch 1.6.0, Numpy 1.18.1, numba

## Setting from the original paper
For Diginetica, the best beta value is 0.01; for Tmall, the best beta value is 0.02. Some people may encounter a cudaError in line 50 or line 74 when running our codes if your numpy and pytorch version are different with ours. Currently, we haven't found the solution to resolve the version problem. If you have this problem, please try to change numpy and pytorch version same with ours.

### Running SCL
```
CUDA_VISIBLE_DEVICES=1 python main.py \
    --dataset Tmall \
    --epoch 30 \
    --batchSize 100 \
    --embSize 100 \
    --l2 1e-5 \
    --lr 0.001 \
    --layer 3 \
    --beta 0.02 \
    --temperature 0.1 \
    --item_cl_loss_weight 100.0 \
    --sampled_item_size 37000 \
    --use_item_cl_loss \
    --remove_original_cl_loss --saved_models_path output/Tmall_results_nocl

CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset Nowplaying \
    --epoch 30 \
    --batchSize 100 \
    --embSize 100 \
    --l2 1e-5 \
    --lr 0.001 \
    --layer 3 \
    --beta 0.02 \
    --temperature 0.1 \
    --item_cl_loss_weight 0.1 \
    --sampled_item_size 30000 \
    --use_item_cl_loss \
    --saved_models_path output/Nowplaying_results

CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset diginetica \
    --epoch 30 \
    --batchSize 100 \
    --embSize 100 \
    --l2 1e-5 \
    --lr 0.001 \·
    --layer 3 \
    --beta 0.01 \
    --temperature 0.1 \
    --item_cl_loss_weight 1.0 \
    --sampled_item_size 37000 \
    --use_item_cl_loss \
    --saved_models_path output/diginetica_results
```
Please refer to the original repository for reproducing the results of DHCN model. You can use `--remove_original_cl_loss` to remove the original contrastive learning loss in the DHCN model.
