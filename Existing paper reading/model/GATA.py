# --------- Link Prediction Model with both TAT and GAT contained -----------
import torch.nn as nn
import torch
from .GAT import GAT
from .TAT import TAT


class GATA(nn.Module):
    def __init__(self, emb_dim, hid_dim, out_dim, num_voc, num_heads, num_ent, num_rel, dropout, alpha, **kwargs):
        super(GATA, self).__init__()
        self.ent_embedding = nn.Embedding(num_ent, emb_dim)
        self.rel_embedding = nn.Embedding(num_rel, emb_dim)
        self.graph = GAT(nfeat=emb_dim, nhid=hid_dim, dropout=dropout, nheads=num_heads, alpha=alpha)
        self.text = TAT(emb_dim, num_voc)
        self.gate = nn.Embedding(num_ent, out_dim)

    def forward(self, nodes, adj, pos, shifted_pos, h_sents, h_order, h_lengths, t_sents, t_order, t_lengths):
        node_features = self.ent_embedding(nodes)
        graph = self.graph(node_features, adj)
        head_graph = graph[[shifted_pos[:, 0].squeeze()]]
        tail_graph = graph[[shifted_pos[:, 1].squeeze()]]

        head_text = self.text(h_sents, h_order, h_lengths, node_features[[shifted_pos[:, 0].squeeze()]])
        tail_text = self.text(t_sents, t_order, t_lengths, node_features[[shifted_pos[:, 1].squeeze()]])

        r_pos = self.rel_embedding(pos[:, 2].squeeze())

        gate_head = self.gate(pos[:, 0].squeeze())
        gate_tail = self.gate(pos[:, 1].squeeze())

        score_pos = self._score(head_graph, head_text, tail_graph, tail_text, r_pos, gate_head, gate_tail)
        return score_pos

    def _score(self, hg, ht, tg, tt, r, gh, gt):
        gate_h = torch.sigmoid(gh)
        gate_t = torch.sigmoid(gt)
        head = gate_h * hg + (1-gate_h) * ht
        tail = gate_t * tg + (1-gate_t) * tt
        s = torch.abs(head + r - tail)
        return s
