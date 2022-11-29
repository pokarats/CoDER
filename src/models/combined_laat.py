#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Python template with an argument parser and logger. Put all the "main" logic into the method called "main".
             Only use the true "__main__" section to add script arguments. The logger writes to a hidden folder './log/'
             and uses the name of this file, followed by the date (by default). The argument parser comes with a default
             option --quiet to keep the stdout clean.

@copyright: Copyright 2018 Deutsches Forschungszentrum fuer Kuenstliche
            Intelligenz GmbH or its licensors, as applicable.

@author: Noon Pokaratsiri Goldstein, adapted from Saadullah Amin's LAAT implementation
(https://github.com/suamin/P4Q_Guttmann_SCT_Coding/blob/main/laat.py)

LAAT Model as proposed by Vu et al. 2020 (https://www.ijcai.org/proceedings/2020/461)

"""
from laat import LAAT
import torch
import torch.nn as nn
import numpy as np

# not used
class Aggregator(nn.Module):
    def __init__(self, combined_n, out_dim):
        """
        Output to the specified out_dim from a concatenated layer
        """
        super(Aggregator, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.txt_cui_aggregator = nn.Linear(combined_n, out_dim, bias=True)
        self.init()

    def init(self, mean=0.0, std=0.03, xavier=False):
        if xavier:
            torch.nn.init.xavier_uniform_(self.txt_cui_aggregator.weight)
        else:
            # follow LAAT paper fc layer initialization
            torch.nn.init.normal_(self.txt_cui_aggregator.weight, mean, std)
            if self.txt_cui_aggregator.bias is not None:
                self.txt_cui_aggregator.bias.data.fill_(0)

    def forward(self, x):
        # x shape: b x n_cui + n_txt x 2u
        # combined_n == n_cui + n_txt, i.e. x.size at dim 1
        return self.txt_cui_aggregator(x)


class CombinedLAAT(LAAT):
    def __init__(self,
                 n_txt,
                 n_cui,
                 de,
                 L,
                 u=256,
                 da=256,
                 dropout=0.3,
                 pad_idx=0,
                 txt_pre_trained_weights=None,
                 cui_pre_trained_weights=None,
                 trainable=False):
        """
        parameter names follow the variables in the LAAT paper, unless otherwise explained

        :param n_txt: vocab size of text corpus, should be the same as (len(wv.vocab.keys()) + 2; 2 for unk and pad, i.e.
        this should be == loaded_np_array.shape[0] ) (for text input)
        :type n_txt: int
        :param n_cui: vocab size of cui corpus, should be the same as (len(wv.vocab.keys()) + 2; 2 for unk and pad, i.e.
        this should be == loaded_np_array.shape[0] (for cui input)
        :param de: word embedding size for each w_i/cui_i token in doc D (both embeddings MUST be of the same size
        :type de: int
        :param L: number of unique labels in the dataset, i.e. label set
        :type L: int
        :param u: LSTM hidden_size (opt: 512 for full, 256 for 50) (same for both text/cui)
        :type u: int
        :param da: tunable hyperparam, W's out_features, U's in_features (opt: 512 for full, 256 for 50)
        :type da: int
        :param dropout:
        :type dropout:
        """
        super().__init__(n=n_txt,
                         de=de,
                         L=L,
                         u=u,
                         da=da,
                         dropout=dropout,
                         pad_idx=pad_idx,
                         pre_trained_weights=txt_pre_trained_weights,
                         trainable=trainable)
        self.cui_embed = nn.Embedding.from_pretrained(cui_pre_trained_weights,
                                                      padding_idx=pad_idx,
                                                      freeze=trainable) if cui_pre_trained_weights is not None else \
            nn.Embedding(n_cui, de, padding_idx=pad_idx)
        self.aggregator = nn.Linear(4 * u, 2 * u, bias=True)
        self.init_aggregator()

    def init_aggregator(self, mean=0.0, std=0.03, xavier=False):
        if xavier:
            torch.nn.init.xavier_uniform_(self.aggregator.weight)
        else:
            # LAAT paper initialization
            torch.nn.init.normal_(self.aggregator.weight, mean, std)
            if self.aggregator.bias is not None:
                self.aggregator.bias.data.fill_(0)

    def forward(self, x_txt, x_cui, y=None):
        # get txt and cui sequence lengths
        txt_seq_lengths = torch.count_nonzero(x_txt, dim=1).cpu()
        cui_seq_lengths = torch.count_nonzero(x_cui, dim=1).cpu()

        # get txt and cui batch sizes (should be the same for both)
        txt_batch_size = x_txt.size()[0]
        cui_batch_size = x_cui.size()[0]
        assert txt_batch_size == cui_batch_size

        # text input embedding part
        txt_embedded = self.word_embed(x_txt)  # b x txt seq len x de
        txt_embedded = self.dropout(txt_embedded)  # per LAAT paper, dropout applied to embedding step

        # pack padded sequence
        txt_embedded = nn.utils.rnn.pack_padded_sequence(txt_embedded, txt_seq_lengths, batch_first=True,
                                                         enforce_sorted=False)
        txt_lstm_hidden = self.init_lstm_hidden(txt_batch_size)
        self.bilstm.flatten_parameters()
        H_txt, _ = self.bilstm(txt_embedded, txt_lstm_hidden)  # b x seq txt n x 2u <-- dim of unpacked H

        # pad packed output H
        H_txt, txt_unpacked_lengths = nn.utils.rnn.pad_packed_sequence(H_txt, batch_first=True)
        assert torch.equal(txt_seq_lengths, txt_unpacked_lengths)

        # cui input embedding part
        cui_embedded = self.cui_embed(x_cui)  # b x cui seq len x de
        cui_embedded = self.dropout(cui_embedded)

        # pack padded cui sequence
        cui_embedded = nn.utils.rnn.pack_padded_sequence(cui_embedded, cui_seq_lengths, batch_first=True,
                                                         enforce_sorted=False)
        cui_lstm_hidden = self.init_lstm_hidden(cui_batch_size)
        H_cui, _ = self.bilstm(cui_embedded, cui_lstm_hidden)  # b x seq cui n x 2u <-- dim of unpacked H

        # cui pad packed output H
        H_cui, cui_unpacked_lengths = nn.utils.rnn.pad_packed_sequence(H_cui, batch_first=True)
        assert torch.equal(cui_seq_lengths, cui_unpacked_lengths)

        # LAAT layers for both cui and txt input types
        combined_V = []
        for H in [H_txt, H_cui]:
            Z = torch.tanh(self.W(H))  # b x n x da
            A = torch.softmax(self.U(Z), dim=1)  # b x n x L, softmax along dim=1 so that each col in L sums to 1!!
            V = H.transpose(1, 2).bmm(A)  # b x 2u x L
            combined_V.append(V)

        combined_V = torch.cat(combined_V, dim=1)  # b x 4u x L
        # print(combined_V.size())
        V = self.aggregator(combined_V.transpose(1, 2))  # b x L x 2u
        # print(V.size())

        # LAAT implementation of attention layer weighted sum, bias added after summing
        # resultant dim: b x L
        labels_output = self.labels_output.weight.mul(V).sum(dim=2).add(self.labels_output.bias)
        # print(labels_output.size())
        output = (labels_output,)

        if y is not None:
            loss = self.labels_loss_fct(labels_output, y)  # .sum(-1).mean()
            output += (loss,)

        return output


if __name__ == '__main__':
    # test that weights from stored .npy would still work with nn.Embedding.from_pretrained
    #try:
    #    random_np_weights = np.load('testing_np_saved_weights.npy')
    #except FileNotFoundError:
    #    random_np_weights = 10 + 2.5 * np.random.randn(32, 100)

    #if random_np_weights.min() < 0:
    #    random_np_weights += abs(random_np_weights.min()) + 1
    # do NOT use torch.from_numpy here if original np array is float64, will get dtype error later on!
    # torch.Tensor == torch.FloatTensor, gets dtype torch.float32 even if np dtype is float64
    #weights_from_np = torch.Tensor(random_np_weights)

    # testing pre-trained weights, torch.FloatTensor dtype torch.float32
    random_weights = torch.FloatTensor(32, 100).random_(1, 16)
    random_weights_two = torch.FloatTensor(32, 100).random_(1, 16)

    model = CombinedLAAT(n_txt=32, n_cui=32, de=100, L=5, u=256, da=256, dropout=0.3,
                         txt_pre_trained_weights=random_weights,
                         cui_pre_trained_weights=random_weights_two, trainable=True)

    x_txt = torch.LongTensor(8, 16).random_(1, 31)
    x_cui = torch.LongTensor(8, 16).random_(1, 31)

    # simulate padded sequences
    for x in [x_txt, x_cui]:
        for row in range(x.shape[0]):
            eos = np.random.randint(10, 15)
            x[row, eos:] = 0

    y = torch.LongTensor(8, 5).random_(2, 9)
    labels_logits, labels_loss = model(x_txt, x_cui, y.float())
    print(f"labels_loss: {labels_loss}")
    print('LAAT working!')
