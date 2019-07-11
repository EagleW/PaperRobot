import numpy as np
from sys import getsizeof
import torch
import math
import networkx as nx
import json
import pickle
import codecs
from collections import defaultdict, Counter


class KnowledgeGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.triples = []

    def load_file(self, fn, delimiter, threshold):
        fo = open(fn)
        for line in fo:
            line = line.strip()
            if line:
                ent1, ent2, weight = line.split(delimiter)
                if int(weight) >= threshold:
                    self.triples.append((ent1, ent2, weight))

    def load_triple_noweight(self, triples):
        for t in triples:
            ent1, ent2, weight = t
            self.triples.append((ent1, ent2, {'label':weight}))

    def load_file_noweight(self, fn, delimiter, threshold):
        fo = open(fn)
        _ = fo.readline()
        for line in fo:
            line = line.strip()
            if line:
                ent1, ent2, weight = line.split(delimiter)
                self.triples.append((int(ent1), int(ent2), {'label': weight}))

    def filter_small_nodes(self, threshold):
        # this will filter out all the nodes that have less outgoing edges
        to_delete = []
        for node in self.G.nodes():
            if self.G.degree()[node] < threshold:
                to_delete.append(node)
        for n in to_delete:
            self.G.remove_node(n)

    def node_info(self, s):
        seg1 = s.find('<')
        label1 = s[:seg1]
        type1 = s[seg1 + 1:-1]
        return label1, type1

    def triple2graph(self):
        self.G.add_weighted_edges_from(self.triples)

    def triple2graph_noweight(self):
        self.G.add_edges_from(self.triples)


def new_KG(fn):
    KG = KnowledgeGraph()
    KG.load_file_noweight(fn, '\t', 0)
    KG.triple2graph_noweight()
    return KG


def load_triples(kg_f):
    fo = open(kg_f)
    triples = []
    for line in fo:
        line = line.strip()
        ele = line.split('\t')
        if len(ele)==3:
            ele = list(map(int, ele))
            triples.append(ele)


def load_dict(f):
    fo = open(f)
    d = {}
    num = int(fo.readline().strip())
    for line in fo:
        line = line.strip()
        name, idd = line.split('\t')
        d[int(idd)] = name
    return d, num


def load_graph(kbf, num_ent):
    graph = new_KG(kbf)
    adj = torch.FloatTensor(nx.adjacency_matrix(graph.G, nodelist=range(num_ent)).todense())
    return graph, adj


def load_kg_embeddings(f):
    fo = open(f)
    embeddings = json.loads(fo.read())
    return embeddings


def load_triple_dict(f):
    fo = open(f)
    triples = []
    triple_dict = defaultdict(lambda: defaultdict(set))
    triple_dict_rev = defaultdict(lambda: defaultdict(set))
    for line in fo:
        line = line.strip()
        ele = line.split('\t')
        if len(ele) == 3:
            ele = list(map(int, ele))
            triples.append(ele)
            triple_dict[ele[0]][ele[1]].add(ele[2])
            triple_dict_rev[ele[1]][ele[0]].add(ele[2])
    return triples, triple_dict, triple_dict_rev


def create_mapping(freq, min_freq=0, max_vocab=50000):
    freq = freq.most_common(max_vocab)
    item2id = {
        '<pad>': 0,
        '<unk>': 1
    }
    offset = len(item2id)
    for i, v in enumerate(freq):
        if v[1] > min_freq:
            item2id[v[0]] = i + offset
    id2item = {i: v for v, i in item2id.items()}
    return item2id, id2item


def create_dict(item_list):
    assert type(item_list) is list
    freq = Counter(item_list)
    return freq


def prepare_mapping(words, min_freq):
    words = [w.lower() for w in words]
    words_freq = create_dict(words)
    word2id, id2word = create_mapping(words_freq, min_freq)
    print("Found %i unique words (%i in total)" % (
        len(word2id), sum(len(x) for x in words)
    ))

    mappings = {
        'word2idx': word2id,
        'idx2word': id2word
    }

    return mappings


def load_text(f, min_freq, max_len):
    with open(f) as jsf:
        txt = json.load(jsf)
    words = []
    new_txt = {}
    for key in txt:
        tmp = []
        for sent in txt[key]:
            tmp.extend(sent)
            tmp.append('<eos>')
        tmp = tmp[:max_len]
        new_txt[key] = tmp
        words.extend(tmp)
    mappings = prepare_mapping(words, min_freq)
    word2idx = mappings["word2idx"]

    vectorize_txt = defaultdict(list)
    for key in new_txt:
        for w in new_txt[key]:
            try:
                vectorize_txt[key].append(word2idx[w])
            except:
                vectorize_txt[key].append(word2idx['<unk>'])
    return mappings, vectorize_txt


def bern(triple_dict, triple_dict_rev, tri):
    h = tri[0]
    t = tri[2]
    tph = len(triple_dict[h])
    hpt = len(triple_dict_rev[t])
    deno = tph+hpt
    return tph/float(deno), hpt/float(deno)


def adjust_single_sent_order(t):
    batch_size = len(t)
    list_t = t
    sorted_r = sorted([(len(r), r_n, r) for r_n,r in enumerate(list_t)], reverse=True)
    lr, r_n, ordered_list_rev = zip(*sorted_r)
    max_sents = lr[0]
    lr = torch.LongTensor(list(lr))
    r_n = torch.LongTensor(list(r_n))
    batch_t = torch.zeros(batch_size, max_sents).long()                         # (sents ordered by len)
    for i, s in enumerate(ordered_list_rev):
        batch_t[i,0:len(s)] = torch.LongTensor(s)
    return batch_t, r_n, lr # mainly to record batchified text, sentence order, length of sentence,


def adjust_sent_order(l):
    pos,neg,pos_h_text, pos_t_text, neg_h_text, neg_t_text = zip(*l)
    ph = adjust_single_sent_order(pos_h_text)
    pt = adjust_single_sent_order(pos_t_text)
    nh = adjust_single_sent_order(neg_h_text)
    nt = adjust_single_sent_order(neg_t_text)
    return torch.LongTensor(pos),torch.LongTensor(neg),ph,pt,nh,nt

def convert_index(blist, nodeslist):
    new_blist = []
    for b in blist:
        b = b.cpu().numpy()
        index = np.argsort(nodeslist)
        sorted_nodeslist = nodeslist[index]
        found_index = np.searchsorted(sorted_nodeslist, b[:,0:2])
        bindex = np.take(index, found_index, mode='clip')
        bindex_exp = np.hstack((bindex, b[:, 2][:, np.newaxis]))
        new_blist.append(np.array(bindex_exp))
    return new_blist


def generate_corrupt_triples(pos, num_ent, triple_dict, triple_dict_rev):
    neg = []
    for p in pos:
        sub = np.random.randint(num_ent)
        tph, hpt = bern(triple_dict, triple_dict_rev, p)
        n = [sub, p[1], p[2]]
        chose = np.random.choice(2,1,p=[tph, hpt])
        if chose[0] == 1:
            n = [p[0], sub, p[2]]
        neg.append(n)
    return neg

def get_subgraph(triples, triple_dict, whole_graph):
    # Only handle 1-hop for now
    # Data Types for Nodes are INT
    in_graph = set()
    for triple in triples:
        head = triple[0]
        tail = triple[1]
        rel = triple[2]
        in_graph.add(tuple(triple))
        for tri in triple_dict[head]:
            single1 = (head, tri[0], tri[1])
            in_graph.add(single1)
        for tri in triple_dict[tail]:
            single2 = (tail, tri[0], tri[1])
            in_graph.add(single2)
    in_kg = KnowledgeGraph()
    in_kg.load_triple_noweight(in_graph)
    in_kg.triple2graph_noweight()
    included_nodes = list(in_kg.G)
    adj_ingraph = nx.adjacency_matrix(whole_graph.G, nodelist=included_nodes).todense()
    return np.array(included_nodes), adj_ingraph


def mean_rank(triples, sorted_idx, correct, loc):
    reordered_triples = triples[sorted_idx]
    rank = np.argwhere(reordered_triples[:, loc] == correct[loc])[0][0]
    # print("rank is",rank)
    return rank

def convert_idx2name(triples, id2ent, ent2name, id2rel):
    result = []
    for t in triples:
        e1 = ent2name[id2ent[t[0].item()]]
        e2 = ent2name[id2ent[t[1].item()]]
        rel = id2rel[t[2].item()]
        result.append([e1,e2,rel])
    return result

def write_triples(triple_list, wf):
    for t in triple_list:
        wf.write('\t'.join(list(map(str, t)))+'\n')
