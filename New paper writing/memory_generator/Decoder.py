import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .baseRNN import BaseRNN
from .utils import MemoryComponent


class DecoderRNN(BaseRNN):

    def __init__(self, vocab_size, embedding, word_dim, sos_id, eos_id, unk_id,
                max_len=150, input_dropout_p=0, layer_dropout=0, n_layers=1, lmbda=1,
                gpu=torch.cuda.is_available(), rnn_type='gru', h=8, hop=3,
                beam_size=4, **kwargs):
        hidden_size = word_dim
        super(DecoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, n_layers, rnn_type)

        self.rnn = self.rnn_cell(word_dim, hidden_size, n_layers, batch_first=True)
        self.output_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_len
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.unk_id = unk_id
        self.lmbda = lmbda
        self.embedding = embedding
        self.device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
        self.beam_size = beam_size

        # initialization
        self.memory_init = MemoryComponent(hop, h, hidden_size, layer_dropout)
        self.Wh = nn.Linear(hidden_size * 2, hidden_size)
        # params for ref attention
        self.Wih = nn.Linear(hidden_size * 2, hidden_size)  # for obtaining projection from encoder hidden states
        self.Ws = nn.Linear(hidden_size, hidden_size)  # for obtaining e from current state
        self.Wc = nn.Linear(1, hidden_size)  # for obtaining e from context vector
        self.Wr = nn.Linear(hidden_size * 2, hidden_size)
        self.vt = nn.Linear(hidden_size, 1)
        # params for memory
        self.memory = MemoryComponent(hop, h, hidden_size, layer_dropout)
        # output
        self.V = nn.Linear(hidden_size * 3, self.output_size)
        # parameters for p_gen
        # for changing refcontext vector into a scalar
        self.w_p = nn.Linear(hidden_size * 2, 1)
        # for changing context vector into a scalar
        self.w_h = nn.Linear(hidden_size * 2, 1)

    def decode_step(self, sources_ids, _h, enc_proj, batch_size, cov_ref, cov_mem, max_enc_len, enc_mask,
                    encoder_outputs, embed_target, max_source_oov, term_output, term_id, term_mask): #

        # reference attention
        dec_proj = self.Ws(_h).unsqueeze(1).expand_as(enc_proj)
        # print('decoder proj', dec_proj.size())
        cov_proj = self.Wc(cov_ref.view(-1, 1)).view(batch_size, max_enc_len, -1)
        # print('cov proj', cov_proj.size())
        e_t = self.vt(torch.tanh(enc_proj + dec_proj + cov_proj).view(batch_size*max_enc_len, -1))
        # mask to -INF before applying softmax
        ref_attn = e_t.view(batch_size, max_enc_len)
        del e_t
        ref_attn.data.masked_fill_(enc_mask.data.byte(), -float('inf')) # float('-inf')
        ref_attn = F.softmax(ref_attn, dim=1)

        ref_context = self.Wr(ref_attn.unsqueeze(1).bmm(encoder_outputs).squeeze(1))

        # terms attention
        term_context, term_attn = self.memory(_h.unsqueeze(0), term_output, term_mask, cov_mem)
        term_context = term_context.squeeze(0)

        # output proj calculation
        p_vocab = F.softmax(self.V(torch.cat((_h, ref_context, term_context), 1)), dim=1)

        # pgen
        # print(embed_target.size())
        p_gen = torch.sigmoid(self.w_p(torch.cat((_h, embed_target), 1)))
        p_ref = torch.sigmoid(self.w_h(torch.cat((ref_context, term_context), 1)))

        weighted_Pvocab = p_vocab * p_gen
        weighted_ref_attn = (1 - p_gen) * p_ref * ref_attn
        weighted_term_attn = (1 - p_gen) * (1 - p_ref) * term_attn

        if max_source_oov > 0:
            # create OOV (but in-article) zero vectors
            ext_vocab = torch.zeros(batch_size, max_source_oov)
            ext_vocab=ext_vocab.to(self.device)
            combined_vocab = torch.cat((weighted_Pvocab, ext_vocab), 1)
            del ext_vocab
        else:
            combined_vocab = weighted_Pvocab
        del weighted_Pvocab
        # scatter article word probs to combined vocab prob.
        combined_vocab = combined_vocab.scatter_add(1, sources_ids, weighted_ref_attn)
        combined_vocab = combined_vocab.scatter_add(1, term_id, weighted_term_attn)

        return combined_vocab, ref_attn, term_attn

    def forward(self, max_source_oov=0, targets=None, targets_id=None,
                sources_ids=None, enc_mask=None, encoder_hidden=None,
                encoder_outputs=None, term_id=None, term_mask=None,
                term_output=None, teacher_forcing_ratio=None, beam=False,
                stopwords=None, sflag=False):
        if beam:
            return self.decode(max_source_oov, sources_ids, enc_mask, encoder_hidden, encoder_outputs,
               term_id, term_mask, term_output, stopwords, sflag)
        # initialization
        targets, batch_size, max_length, max_enc_len = self._validate_args(targets, encoder_hidden, encoder_outputs,
                            teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)
        cov_ref = torch.zeros(batch_size, max_enc_len)
        cov_ref = cov_ref.to(self.device)
        cov_mem = torch.zeros_like(term_mask, dtype=torch.float)
        cov_mem = cov_mem.to(self.device)
        # memory initialization
        decoder_hidden, _ = self.memory_init(decoder_hidden, term_output, term_mask)
        # print(encoder_outputs.size())
        enc_proj = self.Wih(encoder_outputs.contiguous().view(batch_size * max_enc_len, -1)).view(batch_size,
                                                                                                  max_enc_len, -1)
        if teacher_forcing_ratio:
            embedded = self.embedding(targets)
            embed_targets = self.input_dropout(embedded)
            dec_lens = (targets > 0).float().sum(1)
            lm_loss, cov_loss = [], []  # , cov_loss_mem , []
            hidden, _ = self.rnn(embed_targets, decoder_hidden)

            # step through decoder hidden states
            for _step in range(max_length):
                _h = hidden[:, _step, :]
                target_id = targets_id[:, _step+1].unsqueeze(1)
                embed_target = embed_targets[:, _step, :]
                combined_vocab, ref_attn, term_attn = self.decode_step(sources_ids, _h, enc_proj, batch_size,
                                cov_ref, cov_mem, max_enc_len, enc_mask,
                                encoder_outputs, embed_target,
                                max_source_oov, term_output, term_id, term_mask)
                # mask the output to account for PAD , cov_ref
                target_mask_0 = target_id.ne(0).detach()
                output = combined_vocab.gather(1, target_id).add_(sys.float_info.epsilon)
                lm_loss.append(output.log().mul(-1) * target_mask_0.float())

                cov_ref = cov_ref + ref_attn
                cov_mem = cov_mem + term_attn

                # Coverage Loss
                # take minimum across both attn_scores and coverage
                _cov_loss_ref, _ = torch.stack((cov_ref, ref_attn), 2).min(2)
                _cov_loss_mem, _ = torch.stack((cov_mem, term_attn), 2).min(2)
                cov_loss.append(_cov_loss_ref.sum(1) + _cov_loss_mem.sum(1))
                # print(cov_loss_ref[-1].size())
                # cov_loss_mem.append(_)
            total_masked_loss = torch.cat(lm_loss, 1).sum(1).div(dec_lens) + self.lmbda * torch.stack(cov_loss, 1)\
                .sum(1).div(dec_lens) / 2.0
            return total_masked_loss
        else:
            return self.evaluate(targets, batch_size, max_length, max_source_oov, encoder_outputs, decoder_hidden,
                                 enc_mask, sources_ids, enc_proj, max_enc_len, term_output, term_id, term_mask,
                                 cov_ref, cov_mem)

    def evaluate(self, targets, batch_size, max_length, max_source_oov, encoder_outputs, decoder_hidden, enc_mask,
                 sources_ids, enc_proj, max_enc_len, term_output, term_id, term_mask, cov_ref, cov_mem):

        lengths = np.array([max_length] * batch_size)
        decoded_outputs = []
        embed_target = self.embedding(targets)
        # step through decoder hidden states
        for _step in range(max_length):
            _h, decoder_hidden = self.rnn(embed_target, decoder_hidden)
            combined_vocab, ref_attn, term_attn = self.decode_step(sources_ids, _h.squeeze(1), enc_proj, batch_size,
            cov_ref, cov_mem, max_enc_len, enc_mask,
            encoder_outputs, embed_target.squeeze(1),
            max_source_oov, term_output, term_id, term_mask)
            # not allow decoder to output UNK
            combined_vocab[:, self.unk_id] = 0
            symbols = combined_vocab.topk(1)[1]
            decoded_outputs.append(symbols.clone())
            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > _step) & eos_batches) != 0
                lengths[update_idx] = len(decoded_outputs)
            symbols.masked_fill_((symbols > self.vocab_size - 1), self.unk_id)
            embed_target = self.embedding(symbols)
            cov_ref = cov_ref + ref_attn
            cov_mem = cov_mem + term_attn
        return torch.stack(decoded_outputs, 1).squeeze(2), lengths.tolist()

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        else:
            if isinstance(encoder_hidden, tuple):
                encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
            else:
                encoder_hidden = self._cat_directions(encoder_hidden)
            encoder_hidden = self.Wh(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, targets, encoder_hidden, encoder_outputs, teacher_forcing_ratio):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")
        else:
            max_enc_len = encoder_outputs.size(1)
        # inference batch size
        if targets is None and encoder_hidden is None:
            batch_size = 1
        else:
            if targets is not None:
                batch_size = targets.size(0)
            else:
                batch_size = encoder_hidden.size(1)

        # set default targets and max decoding length
        if targets is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no targets is provided.")
            # torch.set_grad_enabled(False)
            targets = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            targets = targets.to(self.device)
            max_length = self.max_length
        else:
            max_length = targets.size(1) - 1     # minus the start of sequence symbol

        return targets, batch_size, max_length, max_enc_len

    def getOverallTopk(self, vocab_probs, ref_attn, term_attn, cov_ref, cov_mem,
                        all_hyps, results, decoder_hidden, stopwords):
        new_decoder_hidden, new_cov_ref, new_cov_mem = [], [], []
        new_vocab_probs = []
        for i, hypo in enumerate(all_hyps):
            curr_vocab_probs = vocab_probs[i]
            curr_vocab_probs[hypo.used_words] = 0
            # print(hypo.used_words)
            new_vocab_probs.append(curr_vocab_probs.unsqueeze(0))
        vocab_probs = torch.cat(new_vocab_probs, 0)
        cov_ref += ref_attn
        cov_mem += term_attn
        # return top-k values i.e. top-k over all beams i.e. next step input ids
        # return hidden, cell states corresponding to topk
        probs, inds = vocab_probs.topk(k=self.beam_size, dim=1)
        probs = probs.log()
        candidates = []
        assert len(all_hyps) == probs.size(0), '# Hypothesis and log-prob size dont match'
        # cycle through all hypothesis in full beam
        for i, hypo in enumerate(probs.tolist()):
            for j, _ in enumerate(hypo):
                new_cand = all_hyps[i].extend(token_id=inds[i,j].item(),
                                              hidden_state=decoder_hidden[i].unsqueeze(0),
                                              cov_ref=cov_ref[i].unsqueeze(0),
                                              cov_mem=cov_mem[i].unsqueeze(0),
                                              log_prob= probs[i,j],
                                              stopwords=stopwords)
                candidates.append(new_cand)
        # sort in descending order
        candidates = sorted(candidates, key=lambda x:x.survivability, reverse=True)
        new_beam, next_inp = [], []
        # prune hypotheses and generate new beam
        for h in candidates:
            if h.full_prediction[-1] == self.eos_id:
                # weed out small sentences that likely have no meaning
                if len(h.full_prediction) >= 5:
                    results.append(h)
            else:
                new_beam.append(h)
                next_inp.append(h.full_prediction[-1])
                new_decoder_hidden.append(h.hidden_state)
                new_cov_ref.append(h.cov_ref)
                new_cov_mem.append(h.cov_mem)
            if len(new_beam) >= self.beam_size:
                break
        assert len(new_beam) >= 1, 'Non-existent beam'
        # print(next_inp)
        return new_beam, torch.LongTensor(next_inp).to(self.device).view(-1, 1), results, \
               torch.cat(new_decoder_hidden, 0).unsqueeze(0), torch.cat(new_cov_ref, 0), torch.cat(new_cov_mem, 0)

    # Beam Search Decoding
    def decode(self, max_source_oov=0, sources_ids=None, enc_mask=None, encoder_hidden=None, encoder_outputs=None,
               term_id=None, term_mask=None, term_output=None, stopwords=None, sflag=False):
        # print(encoder_outputs.size())
        # print(term_output.size())
        max_length = self.max_length
        if not sflag:
            stopwords = set(range(max_source_oov + self.vocab_size))
        targets = torch.LongTensor([[self.sos_id]]).to(self.device)
        decoder_hidden = self._init_state(encoder_hidden)
        max_enc_len = encoder_outputs.size(1)
        max_term_len = term_id.size(1)
        cov_ref = torch.zeros(1, max_enc_len)
        cov_ref = cov_ref.to(self.device)
        cov_mem = torch.zeros(1, max_term_len)
        cov_mem = cov_mem.to(self.device)
        # memory initialization
        decoder_hidden, _ = self.memory_init(decoder_hidden, term_output, term_mask)

        decoded_outputs = []
        # all_hyps --> list of current beam hypothesis. start with base initial hypothesis
        all_hyps = [Hypothesis([self.sos_id], decoder_hidden, cov_ref, cov_mem, 0, stopwords)]
        # start decoding
        enc_proj = self.Wih(encoder_outputs.contiguous().view(max_enc_len, -1)).view(1, max_enc_len, -1)
        # print(enc_proj.size())
        embed_target = self.embedding(targets)
        # print(embed_target.size())
        for _step in range(max_length):
            # print(_step)
            # after first step, input is of batch_size=curr_beam_size
            # curr_beam_size <= self.beam_size due to pruning of beams that have terminated
            # adjust enc_states and init_state accordingly
            _h, decoder_hidden = self.rnn(embed_target, decoder_hidden)
            # print(decoder_hidden.size())
            curr_beam_size = embed_target.size(0)
            # print('curr_beam_size', curr_beam_size)

            combined_vocab, ref_attn, term_attn = self.decode_step(sources_ids, _h.squeeze(1), enc_proj, curr_beam_size,
                                                                   cov_ref, cov_mem, max_enc_len, enc_mask,
                                                                   encoder_outputs, embed_target.squeeze(1),
                                                                   max_source_oov, term_output, term_id, term_mask)
            combined_vocab[:, self.unk_id] = 0

            # does bulk of the beam search
            # decoded_outputs --> list of all ouputs terminated with stop tokens and of minimal length
            all_hyps, symbols, decoded_outputs, decoder_hidden, cov_ref, cov_mem = self.getOverallTopk(combined_vocab,
                                ref_attn, term_attn, cov_ref, cov_mem,
                                all_hyps, decoded_outputs,
                                decoder_hidden.squeeze(0), stopwords)

            symbols.masked_fill_((symbols > self.vocab_size - 1), self.unk_id)
            embed_target = self.embedding(symbols)
            # print('embed_target', embed_target.size())
            curr_beam_size = embed_target.size(0)
            # print('curr_beam_size', curr_beam_size)
            if embed_target.size(0) > encoder_outputs.size(0):
                encoder_outputs = encoder_outputs.expand(curr_beam_size, -1, -1).contiguous()
                enc_mask = enc_mask.expand(curr_beam_size, -1).contiguous()
                sources_ids = sources_ids.expand(curr_beam_size, -1).contiguous()
                term_id = term_id.expand(curr_beam_size, -1).contiguous()
                term_mask = term_mask.expand(curr_beam_size, -1).contiguous()
                term_output = term_output.expand(curr_beam_size, -1, -1).contiguous()
                enc_proj = self.Wih(encoder_outputs.contiguous().view(curr_beam_size * max_enc_len, -1))\
                    .view(curr_beam_size, max_enc_len, -1)
                # print('encoder proj', enc_proj.size())
        if len(decoded_outputs) > 0:
            candidates = decoded_outputs
        else:
            candidates = all_hyps
        all_outputs = sorted(candidates, key=lambda x:x.survivability, reverse=True)
        return all_outputs[0].full_prediction #


class Hypothesis(object):
    def __init__(self, token_id, hidden_state, cov_ref, cov_mem, log_prob, stopwords):
        self._h = hidden_state
        self.log_prob = log_prob
        self.hidden_state = hidden_state
        self.cov_ref = cov_ref.detach()
        self.cov_mem = cov_mem.detach()
        self.full_prediction = token_id # list
        self.used_words = list(set(token_id) - stopwords)
        self.survivability = self.log_prob/ float(len(self.full_prediction))

    def extend(self, token_id, hidden_state, cov_ref, cov_mem, log_prob, stopwords):
        return Hypothesis(token_id= self.full_prediction + [token_id],
                          hidden_state=hidden_state,
                          cov_ref=cov_ref.detach(),
                          cov_mem=cov_mem.detach(),
                          log_prob= self.log_prob + log_prob,
                          stopwords=stopwords)
