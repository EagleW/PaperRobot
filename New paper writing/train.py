import gc
import os
import sys
import time
import torch
import pickle
import argparse
import torch.nn as nn
from collections import OrderedDict

from eval import Evaluate
from loader.logger import Tee
from loader.loader import load_file_with_terms
from loader.preprocessing import prepare_mapping, AssembleMem

from utils.optim import get_optimizer
from memory_generator.seq2seq import Seq2seq
from memory_generator.Encoder import EncoderRNN
from memory_generator.Encoder import TermEncoder
from memory_generator.predictor import Predictor
from memory_generator.Decoder import DecoderRNN

# Read parameters from command line
parser = argparse.ArgumentParser()
parser.add_argument(
    "--lower", default='0',
    type=int, help="Lowercase words (this will not affect character inputs)"
)
parser.add_argument(
    "--word_dim", default="128",
    type=int, help="Token embedding dimension"
)
parser.add_argument(
    "--h", default="8",
    type=int, help="No of attention heads"
)
parser.add_argument(
    "--hop", default="3",
    type=int, help="No of Memory layers"
)
parser.add_argument(
    "--dropout", default="0.2",
    type=float, help="Dropout on the embeddings (0 = no dropout)"
)
parser.add_argument(
    "--layer_dropout", default="0.2",
    type=float, help="Dropout on the layer (0 = no dropout)"
)
parser.add_argument(
    "--lr_method", default="adam",
    help="Learning method (SGD, Adadelta, Adam..)"
)
parser.add_argument(
    "--lr_rate", default="0.001",
    type=float, help="Learning method (SGD, Adadelta, Adam..)"
)
parser.add_argument(
    "--model_dp", default="models/",
    help="model directory path"
)
parser.add_argument(
    "--pre_emb", default="",
    help="Location of pretrained embeddings"
)
parser.add_argument(
    "--gpu", default="1",
    type=int, help="default is 1. set 0 to disable use gpu."
)
parser.add_argument(
    "--num_epochs", default="100",
    type=int, help="Number of training epochs"
)
parser.add_argument(
    "--batch_size", default="50",
    type=int, help="Batch size."
)
parser.add_argument(
    "--max_len", default="150",
    type=int, help="Max length."
)
parser.add_argument(
    "--freq", default="5",
    type=int, help="Min freq."
)
parser.add_argument(
    "--cont", action='store_true', help="Continue training."
)
parser.add_argument(
    "--model", default="models/memory/best_dev_model.pth.tar",
    help="Model location"
)
parser.add_argument(
    "--load", action='store_true', help="Load dataset."
)
parser.add_argument(
    "--data_path", default="data",
    help="data directory path"
)
args = parser.parse_args()

# Parse parameters
parameters = OrderedDict()
parameters['lower'] = args.lower == 1
parameters['freq'] = args.freq
parameters['word_dim'] = args.word_dim
parameters['h'] = args.h
parameters['hop'] = args.hop
parameters['pre_emb'] = args.pre_emb
parameters['input_dropout_p'] = args.dropout
parameters['layer_dropout'] = args.layer_dropout
parameters['gpu'] = args.gpu == 1
parameters['batch_size'] = args.batch_size
parameters['max_len'] = args.max_len
parameters['gpu'] = args.gpu == 1
parameters['lr_method'] = args.lr_method
parameters['lr_rate'] = args.lr_rate
parameters['data_path'] = args.data_path
# Check parameters validity
assert os.path.isdir(args.data_path)
assert 0. <= parameters['input_dropout_p'] < 1.0
assert 0. <= parameters['layer_dropout'] < 1.0
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])
model_dir = args.model_dp

model_name = ['memory']
for k, v in parameters.items():
    if v == "":
        continue
    if k == 'pre_emb':
        v = os.path.basename(v)
    model_name.append('='.join((k, str(v))))
model_dir = os.path.join(model_dir, ','.join(model_name[:-1]))
os.makedirs(model_dir, exist_ok=True)

# register logger to save print(messages to both stdout and disk)
training_log_path = os.path.join(model_dir, 'training_log.txt')
if os.path.exists(training_log_path):
    os.remove(training_log_path)
f = open(training_log_path, 'w')
sys.stdout = Tee(sys.stdout, f)

# print model parameters
print("Model location: %s" % model_dir)
print('Model parameters:')
for k, v in parameters.items():
    print('%s=%s' % (k, v))

# Data parameters
lower = parameters['lower']

# load previously saved data
if args.load:
    state = pickle.load(open(args.data_path + '/dataset.pth', 'rb'))
    words = state['words']
    r_dataset = state['r_dataset']
    v_dataset = state['v_dataset']
    t_dataset = state['t_dataset']
else:
    words = []
    r_words, r_dataset = load_file_with_terms(args.data_path + '/train.txt')
    words.extend(r_words)
    v_words, v_dataset = load_file_with_terms(args.data_path + '/valid.txt')
    t_words, t_dataset = load_file_with_terms(args.data_path + '/test.txt')
    state = {
        'words': words,
        'r_dataset': r_dataset,
        'v_dataset': v_dataset,
        't_dataset': t_dataset
    }
    pickle.dump(state, open(args.data_path + '/dataset.pth', "wb"))

mappings, words_freq = prepare_mapping(words, lower, args.freq)
parameters['unk_id'] = mappings['word2id']['<unk>']
parameters['sos_id'] = mappings['word2id']['<sos>']
parameters['eos_id'] = mappings['word2id']['<eos>']

# Index data
r_dataset = AssembleMem(r_dataset, mappings['word2id'], lower, args.batch_size, args.max_len, parameters['gpu'])
v_dataset = AssembleMem(v_dataset, mappings['word2id'], lower, args.batch_size, args.max_len, parameters['gpu'])
print("%i / %i pairs in train / dev." % (r_dataset.len, v_dataset.len))

word2id = mappings['word2id']
id2word = mappings['id2word']
vocab_size = len(mappings['id2word'])

device = torch.device("cuda:0" if torch.cuda.is_available() and parameters['gpu'] else "cpu")
# model initialization
embedding = nn.Embedding(r_dataset.vocab_size, args.word_dim, padding_idx=0)
ref_encoder = EncoderRNN(vocab_size, embedding, parameters['word_dim'], parameters['input_dropout_p'])
term_encoder = TermEncoder(embedding, parameters['input_dropout_p'])
decoder = DecoderRNN(vocab_size, embedding, **parameters)
model = Seq2seq(ref_encoder, term_encoder, decoder)
model = model.to(device)

optimizer = get_optimizer(model, parameters['lr_method'], parameters['lr_rate'])
#
# training starts
#
since = time.time()
best_dev = 0.0
num_epochs = args.num_epochs
epoch_examples_total = r_dataset.len
train_loader = r_dataset.corpus
len_batch_t = len(train_loader)
print('train batches', len_batch_t)
start_epoch = 0
# continue training
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

for epoch in range(num_epochs):
    print('-' * 10)
    print('Epoch {}/{}'.format(epoch + start_epoch, num_epochs - 1))
    # epoch start time
    time_epoch_start = time.time()

    # train
    model.train(True)
    torch.set_grad_enabled(True)
    epoch_loss = 0
    for batch_idx in range(len_batch_t):
        batch_s, batch_o_s, source_len, max_source_oov, batch_term, batch_o_term, batch_t, \
         batch_o_t = r_dataset.get_batch(batch_idx)
        losses = model(batch_s, batch_o_s, source_len, max_source_oov, batch_term, batch_o_term, batch_t,
                       batch_o_t, teacher_forcing_ratio=1)
        batch_loss = losses.mean()
        # print(losses)

        model.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        num_examples = batch_s.size(0)
        loss = batch_loss.item()
        epoch_loss += num_examples * loss
        sys.stdout.write(
            '%d batches processed. current batch loss: %f\r' %
            (batch_idx, loss)
        )
        sys.stdout.flush()
        del batch_s, batch_o_s, batch_t, batch_o_t, source_len, batch_term, batch_o_term
    gc.collect()
    # torch.cuda.empty_cache()
    epoch_loss_avg = epoch_loss / float(epoch_examples_total)
    log_msg = "Finished epoch %d: Train %s: %.4f" % (epoch + start_epoch, "Avg NLLLoss", epoch_loss_avg)
    print()
    print(log_msg)

    predictor = Predictor(model, id2word, vocab_size)
    eval_f = Evaluate()
    print("Start Evaluating")
    cand, ref, titles, terms = predictor.preeval_batch(v_dataset)
    final_scores = eval_f.evaluate(live=True, cand=cand, ref=ref)
    final_scores['Bleu_4'] *= 10.0
    epoch_score = 2*final_scores['ROUGE_L']*final_scores['Bleu_4']/(final_scores['Bleu_4']+ final_scores['ROUGE_L'])
    if epoch_score > best_dev:
            best_dev = epoch_score
            print('new best score on dev: %.4f' % best_dev)
            print('saving the current model to disk...')

            state = {
                'epoch': epoch + 1,
                'parameters': parameters,
                'state_dict': model.state_dict(),
                'best_prec1': best_dev,
                'optimizer': optimizer.state_dict(),
                'mappings': mappings
            }
            torch.save(state, os.path.join(model_dir, 'best_dev_model.pth.tar'))

            print("Examples")
            print("Output:", cand[1])
            print("Refer:", ref[1])
    # epoch end time
    time_epoch_end = time.time()
    # torch.cuda.empty_cache()
    print('epoch training time: %f seconds' % round(
        (time_epoch_end - time_epoch_start), 2))
    print('best dev: ', best_dev)
