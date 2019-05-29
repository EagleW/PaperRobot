import torch.nn as nn


class Seq2seq(nn.Module):

    def __init__(self, ref_encoder, term_encoder, decoder):
        super(Seq2seq, self).__init__()
        self.ref_encoder = ref_encoder
        self.term_encoder = term_encoder
        self.decoder = decoder

    def forward(self, batch_s, batch_o_s, source_len, max_source_oov, batch_term, batch_o_term, batch_t=None,
                batch_o_t=None, teacher_forcing_ratio=0, beam=False, stopwords=None, sflag=False): # w2fs=None
        encoder_outputs, encoder_hidden, enc_mask = self.ref_encoder(batch_s, source_len)
        term_output, term_mask = self.term_encoder(batch_term)
        result = self.decoder(max_source_oov, batch_t, batch_o_t, batch_o_s, enc_mask, encoder_hidden,
                              encoder_outputs, batch_o_term, term_mask, term_output, teacher_forcing_ratio, beam,
                              stopwords, sflag)
        return result
