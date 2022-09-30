# -*- coding: utf-8 -*-

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


def mimic_collate_fn(dataset_batch):
    input_ids, label_ids = list(zip(*dataset_batch))
    input_ids = list(map(torch.LongTensor, input_ids))  # dtype: torch.int64
    label_ids = list(map(torch.Tensor, label_ids))  # dtype: torch.float32

    padded_input_ids = pad_sequence(input_ids, batch_first=True)  # shape: batch_size x max_seq_len in batch
    label_ids = torch.cat(label_ids, dim=0)  # shape: batch_size x num label classes

    return padded_input_ids, label_ids


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
                 vocab_fn="processed_train_full.json",
                 doc_iterator=MimicDocIter,
                 max_seq_length=4000):
        self.data_dir = Path(data_dir) / f"{version}"
        self.w2v_dir = self.data_dir / "model"

        # data file paths
        # TODO: extend this to also cover clef file format
        self.train_file = self.data_dir / f"train_{version}.csv"
        self.dev_file = self.data_dir / f"dev_{version}.csv"
        self.test_file = self.data_dir / f"test_{version}.csv"
        self.doc_split_path = dict(train=self.train_file, dev=self.dev_file, test=self.test_file)

        # get all labels and fit MultiLabelBinarizer
        self.doc_iterator = doc_iterator
        all_labels_iter = itertools.chain(self.doc_iterator(self.train_file, slice_pos=3),
                                          self.doc_iterator(self.dev_file, slice_pos=3),
                                          self.doc_iterator(self.test_file, slice_pos=3))
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(all_labels_iter)
        # self.id2label = {k: v for k, v in enumerate(self.mlb.classes_)}

        # input to feature id
        if input_type == "text":
            try:
                vocab_fname = self.w2v_dir / ("processed_full.json" if vocab_fn is None else vocab_fn)
            except FileNotFoundError:
                print(f"No vocab file, making word to index dict from vocab...")
                # TODO: make_vocab_dict function from vocab.csv
                # for now error exit
                sys.exit(1)

        elif input_type == "cui":
            try:
                vocab_fname = self.w2v_dir / ("processed_full_umls.json" if vocab_fn is None else vocab_fn)
            except FileNotFoundError:
                print(f"No vocab file, making word to index dict from vocab...")
                # TODO: make_vocab_dict function from cui_vocab file
                # for now error exit
                sys.exit(1)
        else:
            raise ValueError(f"Invalid input_type option!")
        self.featurizer = Features(json.load(open(f"{vocab_fname}")))
        self.max_seq_length = max_seq_length

        self.doc2labels = dict()
        self.split_stats = dict(train=dict(), dev=dict(), test=dict())

    def _fit_transform(self, split):
        doc_iter = self.doc_iterator(self.doc_split_path[split])
        for doc_id, doc_sents, doc_labels, doc_len in doc_iter:
            tokens = list()
            for each_sent in doc_sents:
                tokens.extend(each_sent)
            input_ids = self.featurizer.convert_tokens_to_features(tokens, self.max_seq_length)
            self.doc2labels[doc_id] = doc_labels
            yield doc_id, input_ids, self.mlb.transform([doc_labels])

    def get_dataset_stats(self, split):
        doc_lens = list(map(int, self.doc_iterator(self.doc_split_path[split], slice_pos=4)))
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
        dataset = list()
        for doc_id, input_ids, labels_bin in self._fit_transform(split):
            dataset.append((doc_id, input_ids, labels_bin))
        return dataset


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


def get_dataloader(dataset, batch_size, shuffle, collate_fn=mimic_collate_fn, num_workers=1):
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return data_loader


def get_data(data_dir, version, input_type, doc_iterator=MimicDocIter, batch_size=8, max_seq_length=4000):
    dr = DataReader(data_dir, version, input_type, "processed_train_full.json", doc_iterator, max_seq_length)
    train_data_loader = get_dataloader(Dataset(dr.get_dataset('train'), dr.mlb), batch_size, True)
    dev_data_loader = get_dataloader(Dataset(dr.get_dataset('dev'), dr.mlb), batch_size, False)
    test_data_loader = get_dataloader(Dataset(dr.get_dataset('test'), dr.mlb), batch_size, False)
    return dr, train_data_loader, dev_data_loader, test_data_loader


if __name__ == '__main__':
    check_data_reader = False
    if check_data_reader:
        dr = DataReader()
        iter_train = iter(dr.get_dataset('train'))
        print(next(iter_train))
    check_data_loader = True
    if check_data_loader:
        _, trd, dvd, ted = get_data("../../data/mimic3", "full", "text")
        temp = iter(trd)
        print(next(temp))
