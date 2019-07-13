import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .utils import generate_corrupt_triples, load_triple_dict
from torch.utils import data


class LinkPredictionDataset(Dataset):
    def __init__(self, kg_file, txt_file, id2ent, num_ent):
        self.triples, self.triple_dict, self.triple_dict_rev = load_triple_dict(kg_file)
        self.texts = txt_file
        self.id2ent = id2ent
        self.num_ent = num_ent
        self.negative = generate_corrupt_triples(self.triples, self.num_ent, self.triple_dict, self.triple_dict_rev)
        self.num_neg = len(self.negative)
        self.triples = np.array(self.triples)
        self.negative = np.array(self.negative)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        positive = self.triples[idx,:]
        negative = self.negative[idx,:]
        pos_h_text = self.texts[self.id2ent[positive[0].item()]]
        pos_t_text = self.texts[self.id2ent[positive[1].item()]]
        neg_h_text = self.texts[self.id2ent[negative[0].item()]]
        neg_t_text = self.texts[self.id2ent[negative[1].item()]]

        return positive, negative, pos_h_text, pos_t_text, neg_h_text, neg_t_text

    def get_triples(self):
        return self.triples

    def get_num_ent(self):
        return self.num_ent

    def get_triple_dict(self):
        return self.triple_dict

    def get_triple_dict_rev(self):
        return self.triple_dict_rev

class LinkTestTotal:
    def __init__(self, kg_file, num_ent):
        self.triples, _, _ = load_triple_dict(kg_file)
        self.num_ent = num_ent
        self.triples = np.array(self.triples)
        self.new_head = np.expand_dims(np.array(list(range(self.num_ent))), axis=1)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        t = self.triples[idx,:]
        np_tri = np.repeat([t], self.num_ent, axis=0)
        old_tail = np_tri[:, 1:]
        old_head = np_tri[:, 0]
        new_head_triple = np.concatenate([self.new_head, old_tail], axis=1)
        new_tail_triple = np_tri
        new_tail_triple[:, 1] = self.new_head.squeeze()
        return new_head_triple, new_tail_triple, np.array(t)


class LinkTestDataset(Dataset):
    def __init__(self, head_triple, tail_triple, txt_file, id2ent):
        self.head_triple = head_triple
        self.tail_triple = tail_triple
        self.texts = txt_file
        self.id2ent = id2ent

    def __len__(self):
        return len(self.head_triple)

    def __getitem__(self, idx):
        head = self.head_triple[idx,:]
        tail = self.tail_triple[idx,:]
        head_h_text = self.texts[self.id2ent[head[0].item()]]
        head_t_text = self.texts[self.id2ent[head[1].item()]]
        tail_h_text = self.texts[self.id2ent[tail[0].item()]]
        tail_t_text = self.texts[self.id2ent[tail[1].item()]]

        return head, tail, head_h_text, head_t_text, tail_h_text, tail_t_text
