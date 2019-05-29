import torch.nn as nn
from .baseRNN import BaseRNN


class EncoderRNN(BaseRNN):

    def __init__(self, vocab_size, embedding, hidden_size, input_dropout_p,
                 n_layers=1, bidirectional=True, rnn_type='gru'):
        super(EncoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, n_layers, rnn_type)

        self.embedding = embedding
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional)

    def forward(self, source, input_lengths=None):

        # get mask for location of PAD
        mask = source.eq(0).detach()

        embedded = self.embedding(source)
        embedded = self.input_dropout(embedded)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)

        output, hidden = self.rnn(embedded)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # print('o', output.size())
        return output, hidden, mask


class TermEncoder(nn.Module):

    def __init__(self, embedding, input_dropout_p):
        super(TermEncoder, self).__init__()

        self.embedding = embedding
        self.input_dropout = nn.Dropout(input_dropout_p)

    def forward(self, term):
        mask = term.eq(0).detach()
        embedded = self.embedding(term)
        embedded = self.input_dropout(embedded)
        return embedded, mask
