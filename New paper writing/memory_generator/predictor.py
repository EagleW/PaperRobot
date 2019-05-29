import gc
from itertools import groupby
import torch
import statistics


def filter_duplicate(sents):
    sents = sents.split('.')
    used = []
    used_s = []
    tmp = ""
    for ss in sents:
        tttmp = ''
        for s in ss.split(','):
            if s not in used:
                if len(s) < 2:
                    continue
                used.append(s)
                no_dupes = ([k for k, v in groupby(s.split())])
                ns = ' '.join(no_dupes)
                if ns not in used_s:
                    used_s.append(ns)
                    if s[-1] == ',':
                        tttmp += ns + ' '
                    else:
                        tttmp += ns + ' , '
        if len(tttmp) == 0:
            continue
        tttmp = "%s%s" % (tttmp[0].upper(), tttmp[1:])
        if tttmp[-1] == '.':
            tmp += tttmp + ' '
        else:
            if tttmp[-2:] == ', ':
                tmp += tttmp[:-2]
            else:
                tmp += tttmp
            tmp += ' . '
    return tmp


class Predictor(object):
    def __init__(self, model, id2word, vocab_size):
        self.model = model
        self.model.eval()
        self.id2word = id2word
        self.vocab_size = vocab_size

    def predict(self, batch_s, batch_o_s, source_len, max_source_oov, batch_term, batch_o_term, list_oovs):
        torch.set_grad_enabled(False)
        decoded_outputs, lengths = self.model(batch_s, batch_o_s, source_len, max_source_oov, batch_term,
                                            batch_o_term)
        length = lengths[0]
        output = []
        # print(decoded_outputs)
        for i in range(length):
            symbol = decoded_outputs[0][i].item()
            if symbol < self.vocab_size:
                output.append(self.id2word[symbol])
            else:
                output.append(list_oovs[0][symbol-self.vocab_size])
        return self.prepare_for_bleu(output, True)[0]

    def predict_beam(self, batch_s, batch_o_s, source_len, max_source_oov, batch_term, batch_o_term, list_oovs,
                     stopwords, sflag=False):
        torch.set_grad_enabled(False)
        decoded_outputs = self.model(batch_s, batch_o_s, source_len, max_source_oov, batch_term, batch_o_term,
                                     beam=True, stopwords=stopwords, sflag=sflag)
        outputs = []
        for symbol in decoded_outputs:
            if symbol < self.vocab_size:
                outputs.append(self.id2word[symbol])
            else:
                outputs.append(list_oovs[0][symbol - self.vocab_size])
        outputs = self.prepare_for_bleu(outputs, True)[0]
        print(outputs)
        return outputs

    def preeval_batch(self, dataset, pmid=False):
        torch.set_grad_enabled(False)
        refs = {}
        cands = {}
        titles = {}
        new_terms = {}
        new_pmids = {}
        avg_len_ref = []
        avg_len_out = []
        i = 0
        for batch_idx in range(len(dataset.corpus)):
            if pmid:
                batch_s, batch_o_s, source_len, max_source_oov, batch_term, batch_o_term, list_oovs, targets, \
                 sources, terms, pmids = dataset.get_batch(batch_idx, False)
            else:
                batch_s, batch_o_s, source_len, max_source_oov, batch_term, batch_o_term, list_oovs, targets, \
                 sources, terms = dataset.get_batch(batch_idx, False)
            decoded_outputs, lengths = self.model(batch_s, batch_o_s, source_len, max_source_oov, batch_term,
                                                  batch_o_term)
            for j in range(len(lengths)):
                i += 1
                ref, lref = self.prepare_for_bleu(targets[j])
                if pmid:
                    refs[i] = ref.split()
                    titles[i] = sources[j]
                    new_terms[i] = terms[j]
                else:
                    avg_len_ref.append(lref)
                    refs[i] = [ref]
                    titles[i] = " ".join(sources[j])
                    new_terms[i] = " ".join(terms[j])
                out_seq = []
                for k in range(lengths[j]):
                    symbol = decoded_outputs[j][k].item()
                    if symbol < self.vocab_size:
                        out_seq.append(self.id2word[symbol])
                    else:
                        out_seq.append(list_oovs[j][symbol-self.vocab_size])
                out, lout = self.prepare_for_bleu(out_seq, True)
                if pmid:
                    new_pmids[i] = pmids[j]
                    cands[i] = out.split()
                else:
                    avg_len_out.append(lout)
                    cands[i] = out

                if i % 500 == 0:
                    print("Percentages:  %.4f" % (i/float(dataset.len)))

            # del batch_s, batch_o_s, source_len, batch_term, batch_o_term
            # gc.collect()
            # torch.cuda.empty_cache()

        if pmid:
            return cands, refs, titles, new_terms, new_pmids
        else:
            print("Reference length ", statistics.mean(avg_len_ref))
            print("Output length ", statistics.mean(avg_len_out))
            return cands, refs, titles, new_terms

    def preeval_batch_beam(self, dataset, pmid=False, stopwords=None, sflag=True):
        torch.set_grad_enabled(False)
        refs = {}
        cands = {}
        titles = {}
        new_terms = {}
        new_pmids = {}
        avg_len_ref = []
        avg_len_out = []
        i = 0
        for batch_idx in range(len(dataset.corpus)): #
            if pmid:
                batch_s, batch_o_s, source_len, max_source_oov, batch_term, batch_o_term, list_oovs, targets, \
                 sources, terms, pmids = dataset.get_batch(batch_idx, False)
            else:
                batch_s, batch_o_s, source_len, max_source_oov, batch_term, batch_o_term, list_oovs, targets, \
                 sources, terms = dataset.get_batch(batch_idx, False)
            decoded_outputs = self.model(batch_s, batch_o_s, source_len, max_source_oov, batch_term,
                                                  batch_o_term, beam=True, stopwords=stopwords, sflag=sflag)
            i += 1
            ref, lref = self.prepare_for_bleu(targets[0])
            if pmid:
                refs[i] = ref.split()
                titles[i] = sources[0]
                new_terms[i] = terms[0]
            else:
                avg_len_ref.append(lref)
                refs[i] = [ref]
                titles[i] = " ".join(sources[0])
                new_terms[i] = " ".join(terms[0])
            out_seq = []
            for symbol in decoded_outputs:
                if symbol < self.vocab_size:
                    out_seq.append(self.id2word[symbol])
                else:
                    out_seq.append(list_oovs[0][symbol-self.vocab_size])
            out, lout = self.prepare_for_bleu(out_seq, True)
            if pmid:
                new_pmids[i] = pmids[0]
                cands[i] = out.split()
            else:
                avg_len_out.append(lout)
                cands[i] = out

            if i % 10 == 0:
                print("Percentages:  %.4f" % (i/float(dataset.len)))

            # del batch_s, batch_o_s, source_len, batch_term, batch_o_term
            # gc.collect()
            # torch.cuda.empty_cache()

        if pmid:
            return cands, refs, titles, new_terms, new_pmids
        else:
            print("Reference length ", statistics.mean(avg_len_ref))
            print("Output length ", statistics.mean(avg_len_out))
            return cands, refs, titles, new_terms

    def prepare_for_bleu(self, sentence, train=False):
        sent = [x for x in sentence if x != '<pad>' and x != '<eos>' and x != '<sos>']
        l = len(sent)
        sent = ' '.join(sent)
        if train:
            sent = filter_duplicate(sent)
        return sent, l
