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

from utils.utils import convert_index, get_subgraph, adjust_sent_order, load_dict, load_graph, load_text, mean_rank
from utils.data_loader import LinkPredictionDataset, LinkTestTotal, LinkTestDataset
from model.GATA import GATA
from collections import OrderedDict


parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir', type=str, default='paper_reading/'
)
parser.add_argument(
    '--epochs', type=int, default=1000, help='Number of epochs to train.'
)
parser.add_argument(
    '--lr', type=float, default=0.001, help='Initial learning rate.'
)
parser.add_argument(
    '--gpu', default='1', type=int, help='default is 1. set 0 to disable use gpu.'
)
parser.add_argument(
    '--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).'
)
parser.add_argument(
    '--batch_size', type=int, default=200, help='Size of a single batch'
)
parser.add_argument(
    '--hidden', type=int, default=8, help='Number of hidden units.'
)
parser.add_argument(
    '--nb_heads', type=int, default=8, help='Number of head attentions.'
)
parser.add_argument(
    '--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).'
)
parser.add_argument(
    '--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.'
)
parser.add_argument(
    '--freq', default='3',  type=int, help='Min freq.'
)
parser.add_argument(
    '--max_len', default='100', type=int, help='Max length of context text'
)
parser.add_argument(
    '--margin', type=int, default=1, help='Margin Value'
)
parser.add_argument(
    '--patience', type=int, default=300, help='Patience'
)
parser.add_argument(
    '--load', action='store_true', help='Load dataset.'
)
parser.add_argument(
    '--cont', action='store_true', help='Continue training.'
)
parser.add_argument(
    "--model", default="models/GATA/best_dev_model.pth.tar",
    help="Model location"
)
parser.add_argument(
    "--model_dp", default="models/",
    help="model directory path"
)
args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available()  and args.gpu == 1 else 'cpu')

# Initialize
# load previously saved data
if args.load:
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
else:
    embedding_dim = args.nb_heads * args.hidden
    num_rel = int(open(os.path.join(args.data_dir, 'relation2id.txt')).readline())
    text_file = os.path.join(args.data_dir, 'entity_text_title_tokenized.json')

    mappings, text_f = load_text(text_file, args.freq, args.max_len)
    ent_f = os.path.join(args.data_dir, 'entity2id.txt')
    id2ent, num_ent = load_dict(ent_f)
    # Load Graph Data
    graph, _ = load_graph(os.path.join(args.data_dir, 'train2id.txt'), num_ent)

    # Parse parameters
    parameters = OrderedDict()
    parameters['emb_dim'] = embedding_dim
    parameters['hid_dim'] = args.hidden
    parameters['out_dim'] = args.hidden*args.nb_heads
    parameters['num_voc'] = len(mappings['idx2word'])
    parameters['num_heads'] = args.nb_heads
    parameters['num_ent'] = num_ent
    parameters['num_rel'] = num_rel
    parameters['dropout'] = args.dropout
    parameters['alpha'] = args.alpha
    parameters['margin'] = args.margin
    state = {
        'parameters': parameters,
        'graph': graph,
        'text_f': text_f,
        'id2ent': id2ent,
        'mappings': mappings,
        'num_ent': num_ent,
        'num_rel': num_rel
    }
    pickle.dump(state, open('dataset.pth', "wb"))
    print("finish_dump")

model_dir = args.model_dp
model_name = ['GATA']
for k, v in parameters.items():
    if v == "":
        continue
    model_name.append('='.join((k, str(v))))
model_dir = os.path.join(model_dir, ','.join(model_name[:-1]))
os.makedirs(model_dir, exist_ok=True)

# Load Positive and Negative Examples
params = {'batch_size': args.batch_size, 'shuffle': True, 'collate_fn': adjust_sent_order}
train_set = LinkPredictionDataset(os.path.join(args.data_dir, 'train2id.txt'), text_f, id2ent, num_ent)
train_triple_dict = train_set.get_triple_dict()
train_generator = data.DataLoader(train_set, **params)
print('Finish loading train')

valid_set = LinkPredictionDataset(os.path.join(args.data_dir, 'valid2id.txt'), text_f, id2ent, num_ent)
valid_generator = data.DataLoader(valid_set, **params)
print('Finish loading valid')

params_test = {'batch_size': args.batch_size, 'shuffle': False, 'collate_fn': adjust_sent_order}
test_set = LinkTestTotal(os.path.join(args.data_dir, 'test2id.txt'), num_ent)
print('Finish loading test')

y = torch.FloatTensor([-1])
y = y.to(device)

# Initialize Model
model = GATA(**parameters)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
model.to(device)

# Train
def train(epoch):
    print("Epoch", epoch)
    t = time.time()
    model.train(True)
    torch.set_grad_enabled(True)
    eloss = 0
    for batch_idx, instance in enumerate(train_generator):
        pos, neg, pht_bef, ptt_bef, nht_bef, ntt_bef = instance
        pos = pos.to(device)
        neg = neg.to(device)
        # text information
        pht = list(map(lambda x:x.to(device),pht_bef[0:3]))
        ptt = list(map(lambda x:x.to(device),ptt_bef[0:3]))
        nht = list(map(lambda x:x.to(device),nht_bef[0:3]))
        ntt = list(map(lambda x:x.to(device),ntt_bef[0:3]))
        batch_nodes, batch_adj = get_subgraph(pos, train_triple_dict, graph)
        # get relative location according to the batch_nodes
        shifted_pos, shifted_neg = convert_index([pos, neg], batch_nodes)
        batch_nodes = torch.LongTensor(batch_nodes.tolist()).to(device)
        batch_adj = torch.from_numpy(batch_adj).to(device)
        shifted_pos = torch.LongTensor(shifted_pos).to(device)
        shifted_neg = torch.LongTensor(shifted_neg).to(device)
        score_pos = model(batch_nodes, batch_adj, pos, shifted_pos, pht[0], pht[1], pht[2],
                    ptt[0], ptt[1], ptt[2])
        score_neg = model(batch_nodes, batch_adj, neg, shifted_neg, nht[0], nht[1], nht[2],
                    ntt[0], ntt[1], ntt[2])
        loss_train = F.margin_ranking_loss(score_pos, score_neg, y, margin=args.margin)
        sys.stdout.write(
            '%d batches processed. current batch loss: %f\r' %
            (batch_idx, loss_train.item())
        )
        eloss += loss_train.item()
        loss_train.backward()
        del batch_nodes, batch_adj, shifted_pos, shifted_neg, pos, neg, pht_bef, ptt_bef, nht_bef, ntt_bef
        optimizer.step()
        if batch_idx%500==0:
            gc.collect()
    print('\n')
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(eloss/(batch_idx+1)),
          'time: {:.4f}s'.format(time.time() - t))

    return eloss


# Valid
def validate(epoch):
    t = time.time()
    model.eval()
    torch.set_grad_enabled(False)
    eloss = 0
    for batch_idx, instance in enumerate(valid_generator):
        pos, neg, pht_bef, ptt_bef, nht_bef, ntt_bef = instance
        pos = pos.to(device)
        neg = neg.to(device)
        # text information
        pht = list(map(lambda x:x.to(device),pht_bef[0:3]))
        ptt = list(map(lambda x:x.to(device),ptt_bef[0:3]))
        nht = list(map(lambda x:x.to(device),nht_bef[0:3]))
        ntt = list(map(lambda x:x.to(device),ntt_bef[0:3]))
        batch_nodes, batch_adj = get_subgraph(pos, train_triple_dict, graph)
        # get relative location according to the batch_nodes
        shifted_pos, shifted_neg = convert_index([pos, neg], batch_nodes)
        batch_nodes = torch.LongTensor(batch_nodes.tolist()).to(device)
        batch_adj = torch.from_numpy(batch_adj).to(device)
        shifted_pos = torch.LongTensor(shifted_pos).to(device)
        shifted_neg = torch.LongTensor(shifted_neg).to(device)
        score_pos = model(batch_nodes, batch_adj, pos, shifted_pos, pht[0], pht[1], pht[2],
                    ptt[0], ptt[1], ptt[2])
        score_neg = model(batch_nodes, batch_adj, neg, shifted_neg, nht[0], nht[1], nht[2],
                    ntt[0], ntt[1], ntt[2])
        loss_train = F.margin_ranking_loss(score_pos, score_neg, y, margin=args.margin)
        sys.stdout.write(
            '%d batches processed. current batch loss: %f\r' %
            (batch_idx, loss_train.item())
        )
        eloss += loss_train.item()
        del batch_nodes, batch_adj, shifted_pos, shifted_neg, pos, neg, pht_bef, ptt_bef, nht_bef, ntt_bef
        if batch_idx%500==0:
            gc.collect()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_valid: {:.4f}'.format(eloss/(batch_idx+1)),
          'time: {:.4f}s'.format(time.time() - t))

    return eloss


t_total = time.time()
best_dev = math.inf
bad_counter = 0
start_epoch = 0
if args.cont:
    print('loading model from:', args.model)
    if args.gpu:
        state = torch.load(args.model)
    else:
        state = torch.load(args.model, map_location=lambda storage, loc: storage)
    state_dict = state['state_dict']
    model.load_state_dict(state_dict)
    state_dict = state['optimizer']
    optimizer.load_state_dict(state_dict)
    start_epoch = state['epoch']
    best_dev = state['best_prec1']
for epoch in range(args.epochs):
    train(start_epoch+epoch)
    current_valid = validate(start_epoch+epoch)
    torch.cuda.empty_cache()
    if current_valid < best_dev:
        best_dev=current_valid
        print('new best score on dev: %.4f' % best_dev)
        print('saving the current model to disk...')

        state = {
            'epoch': start_epoch+epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_dev,
            'optimizer': optimizer.state_dict(),
            'parameters': parameters,
            'optimizer': optimizer.state_dict(),
            'mappings': mappings
        }
        torch.save(state, os.path.join(model_dir, 'best_dev_model.pth.tar'))
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

print('Optimization Finished!')
print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
