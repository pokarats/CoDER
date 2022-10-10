#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: prepare Dataloader for LAAT model for MIMIC-III text and umls input versions.

@author: Noon Pokaratsiri Goldstein

Adapted from code base from: https://github.com/suamin/P4Q_Guttmann_SCT_Coding/blob/main/utils.py
"""

import sys
import os
import numpy as np
import json
import torch
import torch.utils.data as data
import itertools
from torch.nn.utils.rnn import pad_sequence

from pathlib import Path
from src.utils.corpus_readers import MimicDocIter, MimicCuiDocIter
from sklearn.preprocessing import MultiLabelBinarizer


def convert_by_vocab(vocab, items, max_seq_length=None, blank_id=None, unk_id=1):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        # any word token not in vocab gets assigned unk_id by default
        output.append(vocab.get(item, unk_id))
    if max_seq_length is not None:
        if len(output) > max_seq_length:
            output = output[:max_seq_length]
        elif blank_id is not None:
            while len(output) < max_seq_length:
                output.append(blank_id)
        else:
            pass

    return output


def convert_tokens_to_ids(vocab, tokens, max_seq_length=None, blank_id=None, unk_id=1):
    return convert_by_vocab(vocab, tokens, max_seq_length, blank_id, unk_id)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


class Features:

    def __init__(self, vocab, unk_token="<UNK>", pad_token="<PAD>"):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.unk_token = unk_token
        self.pad_token = pad_token

    def convert_tokens_to_features(self, tokens, max_seq_length=None):
        ids = self.convert_tokens_to_ids(tokens, max_seq_length)
        return ids

    def convert_tokens_to_ids(self, tokens, max_seq_length=None, blank_id=None, unk_id=1):
        return convert_by_vocab(self.vocab, tokens, max_seq_length, blank_id, unk_id)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


class DataReader:

    def __init__(self,
                 data_dir="../../data/mimic3",
                 version="full",
                 input_type="text",
                 prune_cui=False,
                 cui_prune_file=None,
                 vocab_fn="processed_train_full.json",
                 max_seq_length=4000,
                 doc_iterator=None,
                 umls_iterator=None):
        self.data_dir = Path(data_dir) / f"{version}"
        self.linked_data_dir = self.data_dir.parent.parent / "linked_data" / f"{version}" \
            if input_type == "umls" else None
        self.w2v_dir = (Path(data_dir) / "model") if input_type == "text" else (self.linked_data_dir.parent / "model")
        self.input_type = input_type

        # data file paths
        # if input_type == "text":
        # TODO: extend this to also cover clef file format
        self.train_file = self.data_dir / f"train_{version}.csv"
        self.dev_file = self.data_dir / f"dev_{version}.csv"
        self.test_file = self.data_dir / f"test_{version}.csv"
        self.doc_iterator = MimicDocIter if doc_iterator is None else doc_iterator
        self.doc_split_path = dict(train=self.train_file, dev=self.dev_file, test=self.test_file)

        # get labels from all partitions and fit MultiLabelBinarizer
        all_labels_iter = itertools.chain(self.doc_iterator(self.train_file, slice_pos=3),
                                          self.doc_iterator(self.dev_file, slice_pos=3),
                                          self.doc_iterator(self.test_file, slice_pos=3))
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(all_labels_iter)

        # if cui as input, get umls file paths for getting doc texts, id, len
        # id and labels will still come from .csv text file
        if self.input_type == "umls":
            self.prune_cui = prune_cui
            self.cui_prune_file = self.linked_data_dir / (f"{version}_cuis_to_discard.pickle" if cui_prune_file is None
                                                          else cui_prune_file)
            self.umls_train_file = self.linked_data_dir / f"train_{version}_{input_type}.txt"
            self.umls_dev_file = self.linked_data_dir / f"dev_{version}_{input_type}.txt"
            self.umls_test_file = self.linked_data_dir / f"test_{version}_{input_type}.txt"
            self.umls_doc_iterator = MimicCuiDocIter if umls_iterator is None else umls_iterator
            self.umls_doc_split_path = dict(train=self.umls_train_file,
                                            dev=self.umls_dev_file,
                                            test=self.umls_test_file)

        # load input to feature word 2 id vocab json saved from word_embedding step
        # file name convention for .json from word_embedding step same for umls and text versions
        try:
            vocab_fname = self.w2v_dir / (f"processed_full_{input_type}_pruned.json" if vocab_fn is None else vocab_fn)
        except FileNotFoundError:
            print(f"No vocab file, NEED to make word to index dict from vocab...")
            # TODO: make_vocab_dict function from vocab.csv or cui vocab file
            # for now error exit
            sys.exit(1)

        # load json vocab mapping word to indx, tokens not in vocab will be replaced with unk token
        self.featurizer = Features(json.load(open(f"{vocab_fname}")))
        self.max_seq_length = max_seq_length

        # store doc_id to labels and split stats: min, max, mean num tokens/doc
        self.doc2labels = dict()
        self.split_stats = dict(train=dict(), dev=dict(), test=dict())

    def _fit_transform(self, split):
        if self.input_type == "text":
            text_doc_iter = self.doc_iterator(self.doc_split_path[split])
            for doc_id, doc_sents, doc_labels, doc_len in text_doc_iter:
                tokens = itertools.chain.from_iterable(doc_sents)
                input_ids = self.featurizer.convert_tokens_to_features(tokens, self.max_seq_length)
                self.doc2labels[doc_id] = doc_labels
                yield doc_id, input_ids, self.mlb.transform([doc_labels])

        elif self.input_type == "umls":
            umls_doc_iter = self.umls_doc_iterator(self.umls_doc_split_path[split],
                                                   threshold=0.7,
                                                   pruned=self.prune_cui,
                                                   discard_cuis_file=self.cui_prune_file)
            text_id_iter = self.doc_iterator(self.doc_split_path[split], slice_pos=0)
            text_lab_iter = self.doc_iterator(self.doc_split_path[split], slice_pos=3)
            for doc_id, (umls_data), doc_labels in zip(text_id_iter, umls_doc_iter, text_lab_iter):
                u_id, u_sents, u_len = umls_data
                tokens = itertools.chain.from_iterable(u_sents)
                input_ids = self.featurizer.convert_tokens_to_features(tokens, self.max_seq_length)
                self.doc2labels[doc_id] = doc_labels
                yield doc_id, input_ids, self.mlb.transform([doc_labels])
        else:
            raise ValueError(f"Invalid input_type option!")

    def get_dataset_stats(self, split):
        if self.split_stats[split].get('mean') is not None:
            return self.split_stats[split]

        if self.input_type == "text":
            doc_lens = list(map(int, self.doc_iterator(self.doc_split_path[split], slice_pos=4)))
        elif self.input_type == "umls":
            doc_lens = list(map(int, [doc_data[2] for doc_data in
                                      self.umls_doc_iterator(self.umls_doc_split_path[split],
                                                             pruned=self.prune_cui,
                                                             discard_cuis_file=self.cui_prune_file)]))
        else:
            raise ValueError(f"Invalid input_type option!")

        self.split_stats[split]['min'] = np.min(doc_lens)
        self.split_stats[split]['max'] = np.max(doc_lens)
        self.split_stats[split]['mean'] = np.mean(doc_lens)

        return self.split_stats[split]

    def get_dataset(self, split):
        """
        Get indexable iterable of dataset split
        :param split:
        :type split:
        :return: List of (doc_id, input_ids, binarized_labels)
        :rtype:
        """

        return list(self._fit_transform(split))


class Dataset(data.Dataset):

    def __init__(self, dataset, mlb):
        self.data = dataset
        # self.id2label = {k: v for k, v in enumerate(self.mlb.classes_)}
        self.mlb = mlb  # has already been fit with all label classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        doc_id, input_ids, labels_bin = self.data[index]
        return input_ids, labels_bin

    @staticmethod
    def mimic_collate_fn(dataset_batch):
        input_ids, label_ids = list(zip(*dataset_batch))
        input_ids = list(map(torch.LongTensor, input_ids))  # dtype: torch.int64
        label_ids = list(map(torch.Tensor, label_ids))  # dtype: torch.float32

        padded_input_ids = pad_sequence(input_ids, batch_first=True)  # shape: batch_size x max_seq_len in batch
        label_ids = torch.cat(label_ids, dim=0)  # shape: batch_size x num label classes

        return padded_input_ids, label_ids


def get_dataloader(dataset, batch_size, shuffle, collate_fn=Dataset.mimic_collate_fn, num_workers=8):
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return data_loader


def get_data(batch_size=8, **kwargs):
    dr = DataReader(**kwargs)
    train_data_loader = get_dataloader(Dataset(dr.get_dataset('train'), dr.mlb), batch_size, True)
    dev_data_loader = get_dataloader(Dataset(dr.get_dataset('dev'), dr.mlb), batch_size, False)
    test_data_loader = get_dataloader(Dataset(dr.get_dataset('test'), dr.mlb), batch_size, False)
    return dr, train_data_loader, dev_data_loader, test_data_loader


if __name__ == '__main__':
    check_data_reader = False
    if check_data_reader:
        data_reader = DataReader(data_dir="../../data/mimic3",
                                 version="50",
                                 input_type="umls",
                                 prune_cui=True,
                                 cui_prune_file=None,
                                 vocab_fn="processed_full_umls_pruned.json")
        d_id, x, y = data_reader.get_dataset('train')[0]
        print(f"id: {d_id}, x: {x}\n, y: {y}")
    check_data_loader = True
    if check_data_loader:
        dr, trd, dvd, ted = get_data(batch_size=8,
                                     data_dir="../../data/mimic3",
                                     version="50",
                                     input_type="umls",
                                     prune_cui=True,
                                     cui_prune_file=None,
                                     vocab_fn="processed_full_umls_pruned.json")
        temp = iter(trd)
        x, y = next(temp)
        print(f"x shape: {x.shape}, type: {x.dtype}\n")
        print(x)
        print(f"y shape: {y.shape}, type: {y.dtype}\n")
        print(y)

    # dr = DataReader(data_dir="../../data/mimic3", version="full", input_type="text")
        print(dr.get_dataset_stats('train'))

        print(np.transpose(np.nonzero(y)))
