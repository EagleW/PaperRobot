import os
import json
import gzip
import lzma
import torch
import torch.nn as nn

from loader.preprocessing import create_mapping


def load_files(path):
    sources = []
    targets = []
    words = []
    for line in open(path, 'r'):
        line = line.strip()
        file = json.loads(line)
        sources.append(file['title'])
        targets.append(file['abs'])
        words.extend(file['words'])
    return words, [sources, targets]


def load_file_with_terms(path):
    sources = []
    targets = []
    terms = []
    words = []
    for line in open(path, 'r'):
        line = line.strip()
        file = json.loads(line)
        sources.append(file['title'])
        targets.append(file['abs'])
        terms.append(file['terms'])
        words.extend(file['words'])
    return words, [sources, terms, targets]


def load_file_with_pmid(path):
    heart = ['heart', 'cardio', 'cardiac', 'annulus', 'arrhyth', 'atrium', 'cardi', 'coronary', 'pulmonary', 'valve']
    # heart = ['DNA', 'Atomic', 'Genome', 'Monolayer', 'Molecular', 'Polymer', 'Self-assembly', 'Quantum', 'Ontological',
    #          'Autogeny', 'MEMS', 'NEMS', 'Plasmonics', 'Biomimetic', 'nano', 'Molecular', 'Electrospinning']
    pmids = []
    sources = []
    targets = []
    terms = []
    words = []
    for line in open(path, 'r'):
        flag = False
        line = line.strip()
        file = json.loads(line)
        for h in heart:
            if h.lower() in " ".join(file['title']).lower():
                flag = True
                break
            if h.lower() in  " ".join(file['terms']).lower():
                flag = True
                break
            if h.lower() in " ".join(file['abs']).lower():
                flag = True
                break
        if flag:
            sources.append(file['title'])
            targets.append(file['abs'])
            terms.append(file['terms'])
            words.extend(file['words'])
            pmids.append(file['pmid'])
    return words, [sources, terms, targets, pmids]


def load_file_with_pmid_no_filter(path):
    pmids = []
    sources = []
    targets = []
    terms = []
    words = []
    for line in open(path, 'r'):
        line = line.strip()
        file = json.loads(line)
        sources.append(file['title'])
        targets.append(file['abs'])
        terms.append(file['terms'])
        words.extend(file['words'])
        pmids.append(file['pmid'])
    return words, [sources, terms, targets, pmids]


def load_test_data(path, t_dataset):
    osources, oterms, otargets, opmids = t_dataset
    data = list(zip(osources, oterms, otargets, opmids))
    gturth = {}
    pmids = []
    sources = []
    targets = []
    terms = []

    for d in data:
        gturth[d[3]] = [d[0],d[1],d[2]]

    for line in open(path, 'r'):
        line = line.strip()
        file = json.loads(line)
        if file['pmid'] in gturth:
            sources.append(file['Output'])
            targets.append(gturth[file['pmid']][2])
            terms.append(file['Terms'])
            pmids.append(file['pmid'])

    return [sources, terms, targets, pmids]


def load_test_new_data(path):
    gturth = {}
    pmids = []
    sources = []
    targets = []
    terms = []

    for line in open(path, 'r'):
        line = line.strip()
        file = json.loads(line)
        sources.append(file['Output'])
        targets.append(file['Ref'])
        terms.append(file['Terms'])
        pmids.append(file['pmid'])

    return [sources, terms, targets, pmids]


def load_pretrained(word_emb, id2word, pre_emb):
    if not pre_emb:
        return

    word_dim = word_emb.weight.size(1)

    # Initialize with pretrained embeddings
    new_weights = word_emb.weight.data
    print('Loading pretrained embeddings from %s...' % pre_emb)
    pretrained = {}
    emb_invalid = 0
    for i, line in enumerate(load_embedding(pre_emb)):
        if type(line) == bytes:
            try:
                line = str(line, 'utf-8')
            except UnicodeDecodeError:
                continue
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pretrained[line[0]] = torch.Tensor(
                [float(x) for x in line[1:]])
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    # Lookup table initialization
    for i in range(len(id2word)):
        word = id2word[i]
        if word in pretrained:
            new_weights[i] = pretrained[word]
            c_found += 1
        elif word.lower() in pretrained:
            new_weights[i] = pretrained[word.lower()]
            c_lower += 1
    word_emb.weight = nn.Parameter(new_weights)

    print('Loaded %i pretrained embeddings.' % len(pretrained))
    print('%i / %i (%.4f%%) words have been initialized with '
          'pretrained embeddings.' % (
           c_found + c_lower , len(id2word),
           100. * (c_found + c_lower) / len(id2word)
          ))
    print('%i found directly, %i after lowercasing, ' % (
           c_found, c_lower))
    return word_emb


def load_embedding(pre_emb):
    if os.path.basename(pre_emb).endswith('.xz'):
        return lzma.open(pre_emb)
    if os.path.basename(pre_emb).endswith('.gz'):
        return gzip.open(pre_emb, 'rb')
    else:
        return open(pre_emb, 'r', errors='replace')


def augment_with_pretrained(words_freq, ext_emb_path):
    """
    Augment the dictionary with words that have a pretrained embedding.
    """
    print(
        'Augmenting words by pretrained embeddings from %s...' % ext_emb_path
    )
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = []
    if len(ext_emb_path) > 0:
        for line in load_embedding(ext_emb_path):
            if not line.strip():
                continue
            if type(line) == bytes:
                try:
                    pretrained.append(str(line, 'utf-8').rstrip().split()[0].strip())
                except UnicodeDecodeError:
                    continue
            else:
                pretrained.append(line.rstrip().split()[0].strip())

    pretrained = set(pretrained)

    # We add every word in the pretrained file
    for word in pretrained:
        words_freq[word] += 10

    word2id, id2word = create_mapping(words_freq)

    mappings = {
        'word2id': word2id,
        'id2word': id2word
    }

    return mappings

# old


def load_file_with_pred(path_title):
    sources = []
    targets = []
    preds = []
    words = []
    path = '/data/m1/wangq16/End-end_title_generation/data/' + path_title
    for line in open(os.path.abspath(path), 'r'):
        line = line.strip()
        file = json.loads(line)
        sources.append(file['source'])
        targets.append(file['target'])
        preds.append(file['preds'])
        words.extend(file['words'])
    return words, [sources, preds, targets]
