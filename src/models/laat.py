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

    def __init__(self, n, de, L, u=256, da=256, dropout=0.3, pad_idx=0, pre_trained_weights=None, trainable=True):
        """
        parameter names follow the variables in the LAAT paper

        :param n: sequence length for the input doc D (max num tokens)
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
        self.word_embed = nn.Embedding.from_pretrained(pre_trained_weights,
                                                       padding_idx=pad_idx,
                                                       freeze=trainable) if pre_trained_weights is not None else \
            nn.Embedding(n,
                         de,
                         padding_idx=pad_idx)
        self.bilstm = nn.LSTM(input_size=de, hidden_size=u, bidirectional=True, batch_first=True, num_layers=1)
        self.W = nn.Linear(2 * u, da, bias=False)
        self.U = nn.Linear(da, L, bias=False)
        self.labels_output = nn.Linear(2 * u, 1)
        self.dropout = nn.Dropout(dropout)
        self.labels_loss_fct = nn.BCEWithLogitsLoss()
        self.init()

    def init(self):
        torch.nn.init.xavier_uniform_(self.W.weight)
        torch.nn.init.xavier_uniform_(self.U.weight)
        torch.nn.init.xavier_uniform_(self.labels_output.weight)
        for name, param in self.bilstm.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_normal_(param)

    def forward(self, x, y=None):
        x = self.word_embed(x)  # b x n x de
        H, _ = self.bilstm(x)  # b x n x 2u
        Z = torch.tanh(self.W(H))  # b x n x da
        A = torch.softmax(self.U(Z), -1)  # b x n x L
        V = H.transpose(1, 2).bmm(A)  # b x 2u x L

        labels_output = self.labels_output(V.transpose(1, 2))  # b x L x 1
        labels_output = labels_output.squeeze()  # b x L
        labels_output = self.dropout(labels_output)

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
    # do NOT use torch.from_numpy here if original np array is float64, will get dtype array later on!
    # torch.Tensor == torch.FloatTensor, gets dtype torch.float32 even if np dtype is float64
    weights_from_np = torch.Tensor(random_np_weights)

    # testing pre-trained weights, torch.FloatTensor dtype torch.float32
    random_weights = torch.FloatTensor(32, 100).random_(1, 16)
    model = LAAT(n=32, de=100, L=5, u=256, da=256, dropout=0.3, pre_trained_weights=weights_from_np, trainable=True)
    x = torch.LongTensor(8, 16).random_(1, 30)
    y = torch.LongTensor(8, 5).random_(2, 9)
    labels_logits, labels_loss = model(x, y.float())
    print(f"labels_loss: {labels_loss}")
    print('LAAT working!')