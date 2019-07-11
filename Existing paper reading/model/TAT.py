import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TAT(nn.Module):
    """
    A Bi-LSTM layer with attention
    """
    def __init__(self, embedding_dim, voc_size):
        super(TAT, self).__init__()
        self.hidden_dim = embedding_dim
        self.word_embeddings = nn.Embedding(voc_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim//2, bidirectional=True)
        self.attF = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, sentence, orders, lengths, ent_emb):
        embedded = self.word_embeddings(sentence)
        padded_sent = pack_padded_sequence(embedded, lengths, batch_first=True)
        output = padded_sent
        output, hidden = self.lstm(output)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output[orders]
        att = torch.unsqueeze(self.attF(ent_emb), 2)
        att_score = F.softmax(torch.bmm(output, att), dim=1)
        o = torch.squeeze(torch.bmm(output.transpose(1,2), att_score))
        return o
