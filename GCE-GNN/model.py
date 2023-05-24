import datetime
import math
import random
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GlobalAggregator
from torch.nn import Module, Parameter
import torch.nn.functional as F
import pickle
import os


class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()

        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
        self.global_agg = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def compute_scores(self, hidden, mask, use_item_cl_loss):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        if use_item_cl_loss:
            b = F.normalize(b, dim=1)  # 
        scores = torch.matmul(select, b.transpose(1, 0))
        return scores, b

    def forward(self, inputs, adj, mask_item, item):
        # 
        # inputs.shape = [bsz, seq_len]
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)

        # local
        h_local = self.local_agg(h, adj, mask_item)  # [bsz, seq_len, hidden_size]

        # global
        item_neighbors = [inputs]
        weight_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)
        
        # mean 
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        
        # sum
        # sum_item_emb = torch.sum(item_emb, 1)
        
        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vector=entity_vectors[hop+1].view(shape),
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim)

        # combine
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)
        output = h_local + h_global  # [bsz, seq_len, hidden_size]

        return output


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data, use_item_cl_loss):
    alias_inputs, adj, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()

    hidden = model(items, adj, mask, inputs)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    # return targets, model.compute_scores(seq_hidden, mask)
    scores, item_embeddings = model.compute_scores(seq_hidden, mask, use_item_cl_loss)  # 
    return targets, scores, item_embeddings  # 


def train_test(model, train_data, test_data, cl_loss_function, use_item_cl_loss=False, sampled_item_size=30000, temperature=0.1, item_cl_loss_weight=0.1, saved_models_path=None):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        # targets, scores = forward(model, data)
        targets, scores, item_embeddings = forward(model, data, use_item_cl_loss)  # 
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
                
        # : compute contrastive loss among item embeddings
        if use_item_cl_loss:
            # print(len(item_embeddings_i))
            bs, _ = item_embeddings.shape
            # start = random.randint(0, len(item_embeddings) - sampled_item_size - 1)
            # sampled_item_embeddings_i = item_embeddings[range(start, start + sampled_item_size), :]
            logits = torch.div(
                torch.matmul(item_embeddings, item_embeddings.T), temperature)
            if torch.cuda.is_available():
                cl_item_loss = cl_loss_function(logits, torch.tensor(range(bs)).cuda())
            else:
                cl_item_loss = cl_loss_function(logits, torch.tensor(range(bs)))
            loss += item_cl_loss_weight * cl_item_loss
        
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    # result = []
    # hit, mrr = [], []
    # for data in test_loader:
    #     targets, scores = forward(model, data)
    #     sub_scores = scores.topk(20)[1]
    #     sub_scores = trans_to_cpu(sub_scores).detach().numpy()
    #     targets = targets.numpy()
    #     for score, target, mask in zip(sub_scores, targets, test_data.mask):
    #         hit.append(np.isin(target - 1, score))
    #         if len(np.where(score == target - 1)[0]) == 0:
    #             mrr.append(0)
    #         else:
    #             mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    # result.append(np.mean(hit) * 100)
    # result.append(np.mean(mrr) * 100)

    # return result

    # 
    metrics = {}
    top_K = [5, 10, 20]
    for k in top_K:
        metrics['hit%d'%k] = []
        metrics['mrr%d'%k] = []

    for data in test_loader:
        targets, scores, _ = forward(model, data, use_item_cl_loss)
        targets = targets.numpy()
        for k in top_K:
            sub_scores = scores.topk(k)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target in zip(sub_scores, targets):
                metrics['hit%d'%k].append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    metrics['mrr%d'%k].append(0)
                else:
                    metrics['mrr%d'%k].append(1 / (np.where(score == target - 1)[0][0] + 1))

    result = {k: np.mean(v) * 100 for k, v in metrics.items()}

    full_metrics = metrics.copy()
    best_results = {}
    for K in top_K:
        best_results['metric%d' % K] = [0, 0]
    for K in top_K:
        metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
        metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
        if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
            best_results['metric%d' % K][0] = metrics['hit%d' % K]
            if K == 10:
                torch.save(model.state_dict(), os.path.join(saved_models_path, "model.pt"))
                pickle.dump(full_metrics, open(os.path.join(saved_models_path, "metrics.pkl"), 'wb'))
        if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
            best_results['metric%d' % K][1] = metrics['mrr%d' % K]

    return result
