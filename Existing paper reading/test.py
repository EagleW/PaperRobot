from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import os, sys, math, pickle, gc

from utils.utils import convert_index, get_subgraph, adjust_sent_order, load_dict, mean_rank, convert_idx2name, write_triples
from utils.data_loader import LinkPredictionDataset, LinkTestTotal, LinkTestDataset
from model.GATA import GATA
from collections import OrderedDict


parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir', type=str, default='paper_reading/'
)
parser.add_argument(
    '--gpu', default='1', type=int, help='default is 1. set 0 to disable use gpu.'
)
parser.add_argument(
    '--batch_size', type=int, default=400, help='Size of a single batch'
)
parser.add_argument(
    "--model", default="models/GATA/best_dev_model.pth.tar",
    help="Model location"
)
args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available()  and args.gpu == 1 else 'cpu')

# Initialize
# load previously saved data
state = pickle.load(open('dataset.pth', 'rb'))
parameters = state['parameters']
graph = state['graph']
text_f = state['text_f']
mappings = state['mappings']
num_ent = state['num_ent']
num_rel = state['num_rel']
id2ent = state['id2ent']
num_ent = state['num_ent']
print("finish load")


rel_f = os.path.join(args.data_dir, 'relation2id.txt')
id2rel, _ = load_dict(rel_f)
name_f = os.path.join(args.data_dir, 'term.pth')
ent2name = pickle.load(open(name_f,"rb"))

# Load Positive and Negative Examples
params = {'batch_size': args.batch_size, 'shuffle': True, 'collate_fn': adjust_sent_order}
train_set = LinkPredictionDataset(os.path.join(args.data_dir, 'train2id.txt'), text_f, id2ent, num_ent)
train_triple_dict = train_set.get_triple_dict()
train_generator = data.DataLoader(train_set, **params)
print('Finish loading train')

params_test = {'batch_size': args.batch_size, 'shuffle': False, 'collate_fn': adjust_sent_order}
test_set = LinkTestTotal(os.path.join(args.data_dir, 'test2id.txt'), num_ent)
print('Finish loading test')

y = torch.FloatTensor([-1])
y = y.to(device)

# Initialize Model
model = GATA(**parameters)
model.to(device)


def test():
    print('Testing...')
    # model.cpu()
    model.eval()

    output_file = open('test_top10.txt', 'w')
    hitk_all = 0
    mean_rank_head = []
    mean_rank_tail = []
    all_named_triples = set()
    for batch_idx, (new_head_triple, new_tail_triple, correct) in enumerate(test_set):
        t_current = time.time()
        print("Current: ", batch_idx, "Total: ", len(test_set))
        test = LinkTestDataset(new_head_triple, new_tail_triple, text_f, id2ent)
        test_generator = data.DataLoader(test, **params_test)
        scores_heads = []
        scores_tails = []

        for current_idx, instance in enumerate(test_generator):
            head, tail, hht_bef, htt_bef, tht_bef, ttt_bef = instance
            head = head.to(device)
            tail = tail.to(device)
            # text information
            hht = list(map(lambda x:x.to(device),hht_bef[0:3]))
            htt = list(map(lambda x:x.to(device),htt_bef[0:3]))
            tht = list(map(lambda x:x.to(device),tht_bef[0:3]))
            ttt = list(map(lambda x:x.to(device),ttt_bef[0:3]))
            batch_nodes, batch_adj = get_subgraph(head, train_triple_dict, graph)
            # get relative location according to the batch_nodes
            shifted_head = convert_index([head], batch_nodes)
            batch_nodes = torch.LongTensor(batch_nodes.tolist()).to(device)
            batch_adj = torch.from_numpy(batch_adj).to(device)
            shifted_head = torch.LongTensor(shifted_head[0]).to(device)
            score_head = model(batch_nodes, batch_adj, head, shifted_head, hht[0], hht[1], hht[2],
                        htt[0], htt[1], htt[2])
            scores_heads.append(score_head.detach())
            del batch_nodes, batch_adj
            batch_nodes, batch_adj = get_subgraph(tail, train_triple_dict, graph)
            # get relative location according to the batch_nodes
            shifted_tail = convert_index([tail], batch_nodes)
            shifted_tail = torch.LongTensor(shifted_tail[0]).to(device)
            batch_nodes = torch.LongTensor(batch_nodes.tolist()).to(device)
            batch_adj = torch.from_numpy(batch_adj).to(device)
            score_tail = model(batch_nodes, batch_adj, tail, shifted_tail, tht[0], tht[1], tht[2],
                        ttt[0], ttt[1], ttt[2])
            scores_tails.append(score_tail.detach())
            del batch_nodes, batch_adj, head, shifted_head, hht, htt, tail, shifted_tail, tht, ttt
            sys.stdout.write(
                '%d batches processed.\r' %
                (current_idx)
            )

        # get head scores
        scores_head = torch.cat(scores_heads, 0)
        scores_head = torch.sum(scores_head, 1).squeeze()
        assert scores_head.size(0) == num_ent
        sorted_head_idx = np.argsort(scores_head.tolist())
        topk_head = new_head_triple[sorted_head_idx][:10]

        #get tail socres
        scores_tail = torch.cat(scores_tails, 0)
        scores_tail = torch.sum(scores_tail, 1).squeeze()
        sorted_tail_idx = np.argsort(scores_tail.tolist())
        topk_tail = new_tail_triple[sorted_tail_idx][:10]

        # predict and output top 10 triples
        named_triples_head = convert_idx2name(topk_head, id2ent, ent2name, id2rel)
        named_triples_tail = convert_idx2name(topk_tail, id2ent, ent2name, id2rel)
        write_triples(named_triples_head, output_file)
        write_triples(named_triples_tail, output_file)

        mean_rank_result_head = mean_rank(new_head_triple, sorted_head_idx, correct, 0)
        mean_rank_result_tail = mean_rank(new_tail_triple, sorted_tail_idx, correct, 1)
        if mean_rank_result_head <= 10:
            hitk_all += 1
        if mean_rank_result_tail <= 10:
            hitk_all += 1
        mean_rank_head.append(mean_rank_result_head)
        mean_rank_tail.append(mean_rank_result_tail)
        del test
    gc.collect()
    output_file.close()

    print('Final mean rank for head is %f'%(np.mean(mean_rank_head)))
    print('Final median rank for head is %f' % np.median(mean_rank_head))
    print('Final mean rank for tail is %f' % (np.mean(mean_rank_tail)))
    print('Final median rank for tail is %f' % np.median(mean_rank_tail))
    print('Final hit10 is %f'%(hitk_all/(len(mean_rank_tail)+1)/2))
    return hitk_all, mean_rank_head, mean_rank_tail



t_total = time.time()
print('loading model from:', args.model)
if args.gpu:
    state = torch.load(args.model)
else:
    state = torch.load(args.model, map_location=lambda storage, loc: storage)
state_dict = state['state_dict']
model.load_state_dict(state_dict)
start_epoch = state['epoch']
best_dev = state['best_prec1']
test()
print('Test Finished!')
print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
