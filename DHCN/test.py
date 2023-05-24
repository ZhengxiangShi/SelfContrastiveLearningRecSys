import argparse
import pickle
import time
from util import Data, split_validation
from model import *
import os
import logging


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='dataset name: diginetica/Nowplaying/sample')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=int, default=3, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')
# (zshi)
parser.add_argument('--saved_models_path', type=str, default='output/test')
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--item_cl_loss_weight', type=float, default=0.1)
parser.add_argument('--sess_cl_loss_weight', type=float, default=0.1)
parser.add_argument('--sampled_item_size', type=int, default=None)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--evaluate_k', type=int, default=1)
parser.add_argument('--use_item_cl_loss', action='store_true', default=False)
parser.add_argument('--use_sess_cl_loss', action='store_true', default=False)
parser.add_argument('--random_search', action='store_true', default=False)
parser.add_argument('--remove_original_cl_loss', action='store_true', default=False)
opt = parser.parse_args()

def main():
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))

    if opt.dataset == 'diginetica':
        n_node = 43097
    elif opt.dataset == 'Tmall':
        n_node = 40727
    elif opt.dataset == 'Nowplaying':
        n_node = 60416
    else:
        n_node = 309
    train_data = Data(train_data, shuffle=True, n_node=n_node)
    test_data = Data(test_data, shuffle=True, n_node=n_node)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    model = DHCN(adjacency=train_data.adjacency,
                 n_node=n_node,
                 lr=opt.lr,
                 l2=opt.l2,
                 beta=opt.beta,
                 layers=opt.layer,
                 emb_size=opt.embSize,
                 batch_size=opt.batchSize,
                 dataset=opt.dataset,
                 temperature=opt.temperature,
                 item_cl_loss_weight=opt.item_cl_loss_weight,
                 sess_cl_loss_weight=opt.sess_cl_loss_weight,
                 use_item_cl_loss=opt.use_item_cl_loss,
                 use_sess_cl_loss=opt.use_sess_cl_loss,
                 sampled_item_size=opt.sampled_item_size,
                 random_search=opt.random_search,
                 top_k=opt.top_k,
                 remove_original_cl_loss=opt.remove_original_cl_loss).to(device)
    model.load_state_dict(torch.load(opt.saved_models_path))

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    metrics = {}
    metrics['hit%d' % opt.evaluate_k] = []
    metrics['mrr%d' % opt.evaluate_k] = []

    for i in tqdm.tqdm(slices):
        tar, scores, con_loss = forward(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = []
        for idd in range(model.batch_size):
            index.append(find_k_largest(20, scores[idd]))
        index = np.array(index)
        tar = trans_to_cpu(tar).detach().numpy()
        
        for prediction, target in zip(index[:, :opt.evaluate_k], tar):
            metrics['hit%d' %opt.evaluate_k].append(np.isin(target, prediction))
            if len(np.where(prediction == target)[0]) == 0:
                metrics['mrr%d' %opt.evaluate_k].append(0)
            else:
                metrics['mrr%d' %opt.evaluate_k].append(1 / (np.where(prediction == target)[0][0]+1))

    metrics['hit%d' % opt.evaluate_k] = np.mean(metrics['hit%d' % opt.evaluate_k]) * 100
    metrics['mrr%d' % opt.evaluate_k] = np.mean(metrics['mrr%d' % opt.evaluate_k]) * 100                    
    print(metrics)

if __name__ == '__main__':
    main()
