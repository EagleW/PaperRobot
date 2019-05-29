import gc
import os
import time
import torch
import pickle
import argparse
import torch.nn as nn

from eval_final import Evaluate
from loader.preprocessing import prepare_mapping, AssembleMem, printcand, filter_stopwords
from loader.loader import load_file_with_terms

from memory_generator.seq2seq import Seq2seq
from memory_generator.Encoder import EncoderRNN
from memory_generator.Encoder import TermEncoder
from memory_generator.predictor import Predictor
from memory_generator.Decoder import DecoderRNN

# Read parameters from command line
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="models/memory/best_dev_model.pth.tar",
    help="Model location"
)
parser.add_argument(
    "--gpu", default="1",
    type=int, help="default is 1. set 0 to disable use gpu."
)
parser.add_argument(
    "--max_len", default="150",
    type=int, help="Max length."
)
parser.add_argument(
    "--data_path", default="data",
    help="data directory path"
)
parser.add_argument(
    "--output", default="data.txt",
    help="data directory path"
)
parser.add_argument(
    "--stop_words", default="1",
    type=int, help="default is 1. set 0 to disable use stopwords."
)
args = parser.parse_args()

print('loading model from:', args.model)
if args.gpu:
    state = torch.load(args.model)
else:
    state = torch.load(args.model, map_location=lambda storage, loc: storage)

parameters = state['parameters']
# Data parameters
lower = parameters['lower']
parameters['gpu'] = args.gpu == 1

data = pickle.load(open(args.data_path + '/dataset.pth', 'rb'))
words = data['words']
_, t_dataset = load_file_with_terms(args.data_path + '/test.txt')
try:
    mappings = state['mappings']
except:
    mappings, words_freq = prepare_mapping(words, lower, parameters['freq'])
state_dict = state['state_dict']

# print model parameters
print('Model parameters:')
for k, v in parameters.items():
    print('%s=%s' % (k, v))

# Index data
t_dataset = AssembleMem(t_dataset, mappings['word2id'], lower, 1, args.max_len, parameters['gpu'])
print("Vocabulary size", t_dataset.vocab_size)
print("%i sentences in test." % (t_dataset.len))

word2id = mappings['word2id']
id2word = mappings['id2word']
vocab_size = len(mappings['id2word'])

device = torch.device("cuda:0" if torch.cuda.is_available() and parameters['gpu'] else "cpu")
embedding = nn.Embedding(t_dataset.vocab_size, parameters['word_dim'], padding_idx=0)


ref_encoder = EncoderRNN(vocab_size, embedding, parameters['word_dim'], parameters['input_dropout_p'])
term_encoder = TermEncoder(embedding, parameters['input_dropout_p'])
decoder = DecoderRNN(vocab_size, embedding, **parameters)
model = Seq2seq(ref_encoder, term_encoder, decoder)
model.load_state_dict(state_dict)
model = model.to(device)
stopwords = filter_stopwords(word2id)
#
# training starts
#
since = time.time()
best_dev = 0.0
epoch_examples_total = t_dataset.len
train_loader = t_dataset.corpus
len_batch_t = len(train_loader)


predictor = Predictor(model, id2word, vocab_size)
eval_f = Evaluate()
print("Start computing")
cands, refs, titles, terms = predictor.preeval_batch_beam(t_dataset, False, stopwords, args.stop_words)
print("Start Evaluating")
final_scores = eval_f.evaluate(live=True, cand=cands, ref=refs)


printcand(args.output, cands)
