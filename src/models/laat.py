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
import torch
import torch.nn as nn
import numpy as np


class LAAT(nn.Module):

    def __init__(self, n, de, L, u=256, da=256, dropout=0.3, pad_idx=0, pre_trained_weights=None, trainable=False):
        """
        parameter names follow the variables in the LAAT paper

        :param n: vocab size of corpus, should be the same as (len(wv.vocab.keys()) + 2; 2 for unk and pad, i.e.
        this should be == loaded_np_array.shape[0]
        :type n: int
        :param de: word embedding size for each w_i token in doc D
        :type de: int
        :param L: number of unique labels in the dataset, i.e. label set
        :type L: int
        :param u: LSTM hidden_size (opt: 512 for full, 256 for 50)
        :type u: int
        :param da: tunable hyperparam, W's out_features, U's in_features (opt: 512 for full, 256 for 50)
        :type da: int
        :param dropout:
        :type dropout:
        """
        super(LAAT, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.word_embed = nn.Embedding.from_pretrained(pre_trained_weights,
                                                       padding_idx=pad_idx,
                                                       freeze=trainable) if pre_trained_weights is not None else \
            nn.Embedding(n,
                         de,
                         padding_idx=pad_idx)
        self.hidden_size = u
        self.bilstm = nn.LSTM(input_size=de, hidden_size=u, bidirectional=True, batch_first=True, num_layers=1)
        self.W = nn.Linear(2 * u, da, bias=False)
        self.U = nn.Linear(da, L, bias=False)
        # changed labels_output dim to 2u x L from 2u x 1 per LAAT implementation
        self.labels_output = nn.Linear(2 * u, L, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.labels_loss_fct = nn.BCEWithLogitsLoss()
        self.init()

    def init(self, mean=0.0, std=0.03, xavier=False):
        if xavier:
            torch.nn.init.xavier_uniform_(self.W.weight)
            torch.nn.init.xavier_uniform_(self.U.weight)
            torch.nn.init.xavier_uniform_(self.labels_output.weight)
            for name, param in self.bilstm.named_parameters():
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
        else:
            # LAAT paper initialization
            torch.nn.init.normal_(self.W.weight, mean, std)
            if self.W.bias is not None:
                self.W.bias.data.fill_(0)
            torch.nn.init.normal_(self.U.weight, mean, std)
            if self.U.bias is not None:
                self.U.bias.data.fill_(0)
            torch.nn.init.normal_(self.labels_output.weight, mean, std)

    def init_lstm_hidden(self, batch_size):
        """
        Initialise the lstm hidden layer per LAAT paper
        :param batch_size: int
            The batch size
        :return: Variable
            The initialised hidden layer
        """
        # [(n_layers x n_directions) x batch_size x hidden_size]
        h = torch.zeros(2, batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(2, batch_size, self.hidden_size).to(self.device)

        return h, c

    def forward(self, x, y=None):
        # get sequence lengths
        seq_lengths = torch.count_nonzero(x, dim=1).cpu()

        # get batch size
        batch_size = x.size()[0]

        embedded = self.word_embed(x)  # b x n x de
        embedded = self.dropout(embedded)  # per LAAT paper, dropout applied to embedding step

        # pack padded sequence
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, seq_lengths, batch_first=True, enforce_sorted=False)
        lstm_hidden = self.init_lstm_hidden(batch_size)
        self.bilstm.flatten_parameters()
        H, _ = self.bilstm(embedded, lstm_hidden)  # b x n x 2u <-- dim of unpacked H

        # pad packed output H
        H, unpacked_lengths = nn.utils.rnn.pad_packed_sequence(H, batch_first=True)
        assert torch.equal(seq_lengths, unpacked_lengths)

        Z = torch.tanh(self.W(H))  # b x n x da
        A = torch.softmax(self.U(Z), -1)  # b x n x L
        V = H.transpose(1, 2).bmm(A)  # b x 2u x L

        # b x L
        # LAAT implementation of attention layer weighted sum, bias added after summing
        labels_output = self.labels_output.weight.mul(V.transpose(1, 2)).sum(dim=2).add(self.labels_output.bias)

        # labels_output = self.labels_output(V.transpose(1, 2))  # b x L x 1
        # labels_output = labels_output.squeeze(dim=2)  # b x L, specify dim or b dropped if batch has 1 sample!
        # labels_output = self.dropout(labels_output) # no dropout in LAAT paper

        output = (labels_output,)

        if y is not None:
            loss = self.labels_loss_fct(labels_output, y)  # .sum(-1).mean()
            output += (loss,)

        return output


if __name__ == '__main__':
    # test that weights from stored .npy would still work with nn.Embedding.from_pretrained
    try:
        random_np_weights = np.load('testing_np_saved_weights.npy')
    except FileNotFoundError:
        random_np_weights = 10 + 2.5 * np.random.randn(32, 100)

    if random_np_weights.min() < 0:
        random_np_weights += abs(random_np_weights.min()) + 1
    # do NOT use torch.from_numpy here if original np array is float64, will get dtype error later on!
    # torch.Tensor == torch.FloatTensor, gets dtype torch.float32 even if np dtype is float64
    weights_from_np = torch.Tensor(random_np_weights)

    # testing pre-trained weights, torch.FloatTensor dtype torch.float32
    random_weights = torch.FloatTensor(32, 100).random_(1, 16)
    model = LAAT(n=32, de=100, L=5, u=256, da=256, dropout=0.3, pre_trained_weights=weights_from_np, trainable=True)
    x = torch.LongTensor(8, 16).random_(1, 31)

    # simulate padded sequences
    for row in range(x.shape[0]):
        eos = np.random.randint(10, 15)
        x[row, eos:] = 0

    y = torch.LongTensor(8, 5).random_(2, 9)
    labels_logits, labels_loss = model(x, y.float())
    print(f"labels_loss: {labels_loss}")
    print('LAAT working!')