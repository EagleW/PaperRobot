import torch
import copy
import math
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MemoryComponent(nn.Module):

    def __init__(self, hop, h, d_model, dropout_p):
        super(MemoryComponent, self).__init__()
        self.max_hops = hop
        self.h = h
        vt = nn.Linear(d_model, 1)
        self.vt_layers = clones(vt, hop)
        Wih = nn.Linear(d_model, d_model)
        self.Wih_layers = clones(Wih, hop)
        Ws = nn.Linear(d_model, d_model)
        self.Ws_layers = clones(Ws, hop)
        self.Wc = nn.Linear(1, d_model)

    def forward(self, query, src, src_mask, cov_mem=None):
        u = query.transpose(0, 1)
        batch_size, max_enc_len = src_mask.size()
        for i in range(self.max_hops ):
            enc_proj = self.Wih_layers[i](src.view(batch_size * max_enc_len, -1)).view(batch_size, max_enc_len, -1)
            dec_proj = self.Ws_layers[i](u).expand_as(enc_proj)
            if cov_mem is not None:
                cov_proj = self.Wc(cov_mem.view(-1, 1)).view(batch_size, max_enc_len, -1)
                e_t = self.vt_layers[i](torch.tanh(enc_proj + dec_proj + cov_proj).view(batch_size * max_enc_len, -1))
            else:
                e_t = self.vt_layers[i](torch.tanh(enc_proj + dec_proj).view(batch_size * max_enc_len, -1))
            term_attn = e_t.view(batch_size, max_enc_len)
            del e_t
            term_attn.data.masked_fill_(src_mask.data.byte(), -float('inf'))
            term_attn = F.softmax(term_attn, dim=1)
            term_context = term_attn.unsqueeze(1).bmm(src)
            u = u + term_context
        return term_context.transpose(0, 1), term_attn
