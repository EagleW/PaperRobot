import torch
import pickle
import argparse
import torch.nn as nn
from loader.preprocessing import prepare_mapping, filter_stopwords

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
    "--batch_size", default="100",
    type=int, help="Batch size."
)
parser.add_argument(
    "--max_len", default="150",
    type=int, help="Max length."
)
parser.add_argument(
    "--stop_words", default="1",
    type=int, help="default is 1. set 0 to disable use stopwords."
)
parser.add_argument(
    "--data_path", default="data",
    help="data directory path"
)
args = parser.parse_args()

# input()
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
word2id = mappings['word2id']
id2word = mappings['id2word']
vocab_size = len(mappings['id2word'])

device = torch.device("cuda:0" if torch.cuda.is_available() and parameters['gpu'] else "cpu")
embedding = nn.Embedding(vocab_size, parameters['word_dim'], padding_idx=0)


ref_encoder = EncoderRNN(vocab_size, embedding, parameters['word_dim'], parameters['input_dropout_p'])
term_encoder = TermEncoder(embedding, parameters['input_dropout_p'])
decoder = DecoderRNN(vocab_size, embedding, **parameters)
model = Seq2seq(ref_encoder, term_encoder, decoder)
model.load_state_dict(state_dict)
model = model.to(device)
stopwords = filter_stopwords(word2id)
# print(stopwords)
#
# training starts
#
while (1):
    batch_s, batch_o_s, source_len, batch_term, batch_o_term, term_len = [], [], [], [], [], []
    seq_str = input("Source:\n")
    seq = seq_str.strip().split(' ')
    _source, source_oov,_o_source, list_oovs = [],[],[],[]
    for _s in seq:
        try:
            _source.append(word2id[_s])
        except KeyError:
            _source.append(word2id['<unk>'])
            if _s not in source_oov:
                _o_source.append(len(source_oov) + vocab_size)
                source_oov.append(_s)
            else:
                _o_source.append(source_oov.index(_s) + vocab_size)
        else:
            _o_source.append(word2id[_s])
    terms = input("terms:\n")
    terms = terms.strip().split(' ')
    _term, _o_term = [], []
    for _t in terms:
        try:
            _term.append(word2id[_t])
        except KeyError:
            _term.append(word2id['<unk>'])
            if _t not in source_oov:
                _o_term.append(len(source_oov) + vocab_size)
                source_oov.append(_t)
            else:
                _o_term.append(source_oov.index(_t) + vocab_size)
        else:
            _o_term.append(word2id[_t])
    max_source_oov = len(source_oov)
    source_len.append(len(_o_source))
    oovidx2word = {idx: word for idx, word in enumerate(source_oov)}
    list_oovs.append(oovidx2word)
    batch_s = torch.LongTensor([_source]).cuda()
    batch_o_s = torch.LongTensor([_o_source]).cuda()

    batch_term = torch.LongTensor([_term]).cuda()
    batch_o_term = torch.LongTensor([_o_term]).cuda()

    source_len = torch.LongTensor(source_len).cuda()

    predictor = Predictor(model, id2word, vocab_size)
    print("Output:")
    predictor.predict_beam(batch_s, batch_o_s, source_len, max_source_oov, batch_term, batch_o_term, list_oovs,
                           stopwords, args.stop_words)
    print()
