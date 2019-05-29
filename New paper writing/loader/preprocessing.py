from collections import Counter
import torch
import json
import string


# Mask variable
def _mask(prev_generated_seq, device, eos_id):
    prev_mask = torch.eq(prev_generated_seq, eos_id)
    lengths = torch.argmax(prev_mask, dim=1)
    max_len = prev_generated_seq.size(1)
    mask = []
    for i in range(prev_generated_seq.size(0)):
        if lengths[i] == 0:
            mask_line = [0] * max_len
        else:
            mask_line = [0] * lengths[i].item()
            mask_line.extend([1] * (max_len - lengths[i].item()))
        mask.append(mask_line)
    mask = torch.ByteTensor(mask)
    mask = mask.to(device)
    return prev_generated_seq.data.masked_fill_(mask, 0)


def create_mapping(freq, min_freq=0, max_vocab=50000):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    char is an indicator whether the mapping is for char/word or the
    """
    freq = freq.most_common(max_vocab)
    item2id = {
        '<pad>': 0,
        '<unk>': 1,
        '<sos>': 2,
        '<eos>': 3
    }
    offset = len(item2id)
    for i, v in enumerate(freq):
        if v[1] > min_freq:
            item2id[v[0]] = i + offset
    id2item = {i: v for v, i in item2id.items()}
    return item2id, id2item


def create_dict(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    freq = Counter(item_list)
    return freq


def prepare_mapping(words, lower, min_freq):
    """
    prepare word2id
    :param words: words corpus
    :param lower: whether lower char
    :return mappings
    """
    if lower:
        words = [w.lower() for w in words]
    words_freq = create_dict(words)
    word2id, id2word = create_mapping(words_freq, min_freq)
    print("Found %i unique words (%i in total)" % (
        len(word2id), sum(len(x) for x in words)
    ))

    mappings = {
        'word2id': word2id,
        'id2word': id2word
    }

    return mappings, words_freq


class AssembleMem:
    def __init__(self, dataset, word2id, lower=True, batch_size=64, max_len=30,
                 cuda=torch.cuda.is_available(), pmid=False):
        if pmid:
            sources, terms, targets, pmids = dataset
            data = list(zip(sources, terms, targets, pmids))
        else:
            sources, terms, targets = dataset
            data = list(zip(sources, terms, targets))
        self.pmid = pmid
        self.batch_size = batch_size
        self.len = len(sources)
        self.word2id = word2id
        self.lower = lower
        self.max_len = max_len
        self.device = torch.device("cuda:0" if cuda and torch.cuda.is_available() else "cpu")
        self.vocab_size = len(word2id)
        self.corpus = self.batchfy(data)

    def batchfy(self, data):
        self.len = len(data)
        data = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        corpus = [self.vectorize(sample) for sample in data]
        return corpus

    def add_start_end(self, vector):
        vector.append(self.word2id['<eos>'])
        return [self.word2id['<sos>']] + vector

    def pad_vector(self, vector, max_len):
        padding = max_len - len(vector)
        vector.extend([0] * padding)
        return vector

    def vectorize(self, sample):
        sample.sort(key=lambda x: len(x[0]), reverse = True)
        list_oovs, targets, sources, terms = [], [], [], []
        batch_s, batch_o_s, source_len, batch_term, batch_o_term, term_len = [], [], [], [], [], []
        batch_t, batch_o_t, target_len = [], [], []
        max_source_oov = 0
        if self.pmid:
            pmids = []
        for data in sample:
            # title
            source_oov = []
            _o_source, _source = [], []
            sources.append(data[0])
            if self.pmid:
                pmids.append(data[3])
            for _s in data[0]:
                if self.lower:
                    _s = _s.lower()
                try:
                    _source.append(self.word2id[_s])
                except KeyError:
                    _source.append(self.word2id['<unk>'])
                    if _s not in source_oov:
                        _o_source.append(len(source_oov) + self.vocab_size)
                        source_oov.append(_s)
                    else:
                        _o_source.append(source_oov.index(_s) + self.vocab_size)
                else:
                    _o_source.append(self.word2id[_s])
            # terms
            _o_term, _term = [], []
            terms.append(data[1])
            for _t in data[1]:
                if self.lower:
                    _t = _t.lower()
                try:
                    _term.append(self.word2id[_t])
                except KeyError:
                    _term.append(self.word2id['<unk>'])
                    if _t not in source_oov:
                        _o_term.append(len(source_oov) + self.vocab_size)
                        source_oov.append(_t)
                    else:
                        _o_term.append(source_oov.index(_t) + self.vocab_size)
                else:
                    _o_term.append(self.word2id[_t])

            if max_source_oov < len(source_oov):
                max_source_oov = len(source_oov)
            batch_s.append(_source)
            batch_o_s.append(_o_source)
            source_len.append(len(_o_source))
            oovidx2word = {idx: word for idx, word in enumerate(source_oov)}
            list_oovs.append(oovidx2word)

            batch_term.append(_term)
            batch_o_term.append(_o_term)
            term_len.append(len(_o_term))

            _o_target, _target = [], []
            targets.append(data[2])
            for _t in data[2]:
                if self.lower:
                    _t = _t.lower()
                try:
                    _target.append(self.word2id[_t])
                except KeyError:
                    _target.append(self.word2id['<unk>'])
                    if _t in source_oov:
                        _o_target.append(source_oov.index(_t) + self.vocab_size)
                    else:
                        _o_target.append(self.word2id['<unk>'])
                else:
                    _o_target.append(self.word2id[_t])

            _target = self.add_start_end(_target)
            batch_t.append(_target)
            batch_o_t.append(self.add_start_end(_o_target))
            target_len.append(len(_target))

        if len(source_len) == 1:
            batch_s = torch.LongTensor(batch_s)
            batch_o_s = torch.LongTensor(batch_o_s)
            batch_term = torch.LongTensor(batch_term)
            batch_o_term = torch.LongTensor(batch_o_term)
            batch_t = torch.LongTensor(batch_t)
            batch_o_t = torch.LongTensor(batch_o_t)
        else:
            batch_s = [torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_s]
            batch_o_s = [torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_o_s]
            batch_s = torch.stack(batch_s, dim=0)
            batch_o_s = torch.stack(batch_o_s, dim=0)

            batch_term = [torch.LongTensor(self.pad_vector(i, max(term_len))) for i in batch_term]
            batch_o_term = [torch.LongTensor(self.pad_vector(i, max(term_len))) for i in batch_o_term]
            batch_term = torch.stack(batch_term, dim=0)
            batch_o_term = torch.stack(batch_o_term, dim=0)

            batch_t = [torch.LongTensor(self.pad_vector(i, max(target_len))) for i in batch_t]
            batch_o_t = [torch.LongTensor(self.pad_vector(i, max(target_len))) for i in batch_o_t]
            batch_t = torch.stack(batch_t, dim=0)
            batch_o_t = torch.stack(batch_o_t, dim=0)
        source_len = torch.LongTensor(source_len)
        if self.pmid:
            return batch_s, batch_o_s, source_len, max_source_oov, batch_term, batch_o_term, batch_t, batch_o_t, list_oovs, \
            targets, sources, terms, pmids
        else:
            return batch_s, batch_o_s, source_len, max_source_oov, batch_term, batch_o_term, batch_t, batch_o_t, list_oovs, \
            targets, sources, terms

    def get_batch(self, i, train=True):
        if self.pmid:
            batch_s, batch_o_s, source_len, max_source_oov, batch_term, batch_o_term, batch_t, batch_o_t, list_oovs, \
             targets, sources, terms, pmids = self.corpus[i]
        else:
            batch_s, batch_o_s, source_len, max_source_oov, batch_term, batch_o_term, batch_t, batch_o_t, list_oovs, \
             targets, sources, terms = self.corpus[i]
        if train:
            return batch_s.to(self.device), batch_o_s.to(self.device), source_len.to(self.device),\
                   max_source_oov, batch_term.to(self.device), batch_o_term.to(self.device), batch_t.to(self.device),\
                   batch_o_t.to(self.device)
        else:
            if self.pmid:
                return batch_s.to(self.device), batch_o_s.to(self.device), source_len.to(self.device),\
                       max_source_oov, batch_term.to(self.device), batch_o_term.to(self.device), list_oovs, targets,\
                       sources, terms, pmids
            else:
                return batch_s.to(self.device), batch_o_s.to(self.device), source_len.to(self.device),\
                       max_source_oov, batch_term.to(self.device), batch_o_term.to(self.device), list_oovs, targets,\
                       sources, terms


def filter_stopwords(word2id):
    nltk_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    stop_words = set()
    for w in set(nltk_stopwords) | set(string.punctuation):
        if w in word2id:
            stop_words.add(word2id[w])
        if w.title() in word2id:
            stop_words.add(word2id[w.title()])
    return stop_words


def printcand(path, cand):
    f_output = open(path, 'w')
    for i in cand:
        f_output.write(cand[i] + "\n")
    f_output.close()
