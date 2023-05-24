import time
from util import Data
from model import *
import os
import argparse
import pickle
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Tmall', help='dataset name: retailrocket/diginetica/Nowplaying/sample')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=int, default=2, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.005, help='ssl task maginitude')
parser.add_argument('--lam', type=float, default=0.005, help='diff task maginitude')
parser.add_argument('--eps', type=float, default=0.2, help='eps')
# 
parser.add_argument('--saved_models_path', type=str, default='output/test')
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--item_cl_loss_weight', type=float, default=0.1)
parser.add_argument('--sess_cl_loss_weight', type=float, default=0.1)
parser.add_argument('--sampled_item_size', type=int, default=5000)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--use_item_cl_loss', action='store_true', default=False)
parser.add_argument('--use_sess_cl_loss', action='store_true', default=False)
parser.add_argument('--random_search', action='store_true', default=False)
parser.add_argument('--remove_original_cl_loss', action='store_true', default=False)

opt = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# torch.cuda.set_device(1)


def main():
    if not os.path.exists(opt.saved_models_path):
        os.mkdir(opt.saved_models_path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] [%(levelname)s] %(message)s", datefmt="%H:%M:%S %a %b %d %Y")

    sHandler = logging.StreamHandler()
    sHandler.setLevel(logging.INFO)
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    fHandler = logging.FileHandler(os.path.join(opt.saved_models_path, 'output.log'), mode='w')
    fHandler.setLevel(logging.INFO)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)

    logger.info(opt)

    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))
    all_train = pickle.load(open('./datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    if opt.dataset == 'diginetica':
        n_node = 43097
    elif opt.dataset == 'Tmall':
        n_node = 40727
    elif opt.dataset == 'retailrocket':
        n_node = 36968
    else:
        n_node = 309
    train_data = Data(train_data, all_train, shuffle=True, n_node=n_node)
    test_data = Data(test_data, all_train, shuffle=True, n_node=n_node)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = COTREC(adjacency=train_data.adjacency,
                   n_node=n_node,
                   lr=opt.lr,
                   l2=opt.l2,
                   beta=opt.beta,
                   lam=opt.lam,
                   eps=opt.eps,
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
    # num_gpus = torch.cuda.device_count()
    # model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    for epoch in range(opt.epoch):
        logger.info('-------------------------------------------------------')
        logger.info('epoch: {}'.format(epoch))
        metrics, total_loss = train_test(model, train_data, test_data, epoch, opt.batchSize, opt.use_item_cl_loss, opt.use_sess_cl_loss)
        full_metrics = metrics.copy()
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
                if K == 10:
                    logger.info("Saving model at epoch {} ...".format(epoch))
                    torch.save(model.state_dict(), os.path.join(opt.saved_models_path, "model.pt"))
                    pickle.dump(full_metrics, open(os.path.join(opt.saved_models_path, "metrics.pkl"), 'wb'))
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
        logger.info(metrics)
        for K in top_K:
            logger.info('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                        (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                         best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))


if __name__ == '__main__':
    main()
