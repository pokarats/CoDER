#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Prepare Dataloader for LAAT model for MIMIC-III text, UMLS CUI, and combined input versions.

@author: Noon Pokaratsiri Goldstein

Adapted from code base from Saadullah Amin's Dataloader code:
https://github.com/suamin/P4Q_Guttmann_SCT_Coding/blob/main/utils.py
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
from src.utils.corpus_readers import MimicDocIter, MimicCuiDocIter, MimicCuiSelectedTextIter
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
        """

        :param vocab: json dict mapping token to ids
        :type vocab: dict
        :param unk_token: unknown token
        :type unk_token: str
        :param pad_token: padding token
        :type pad_token: str
        """
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
                 vocab_fn="processed_full_text_pruned.json",
                 max_seq_length=4000,
                 doc_iterator=None,
                 umls_iterator=None,
                 second_txt_vocab_fn=None):
        """

        :param data_dir:
        :type data_dir:
        :param version: full or 50 version of the MIMIC-III dataset
        :type version: str
        :param input_type: text, umls, or combined
        :type input_type: str
        :param prune_cui: param for MimicCuiDocIter - True if prune cui according to set of cuis in prune file
        :type prune_cui: bool
        :param cui_prune_file: name of {version}_cuis_to_discard.pickle file containing set of cuis to discard
        (NOT file path) from cui pruning step; should be in data/linked_data/{version}/
        :type cui_prune_file: str
        :param vocab_fn: name of the .json file containing the w2v vocab mapping str/cui to id (NOT file path) generated
        from word_embeddings step; should be in data/{linked_data|mimic3}/model/ (for combined input_type, this MUST be
        for UMLS vocab mapping file)
        :type vocab_fn: str
        :param max_seq_length: 4000 (per LAAT paper) or other threshold for max number of tokens in input text/cuis
        :type max_seq_length: int
        :param doc_iterator: txt input MimicDocIter
        :param umls_iterator: umls input MimicCuiDocIter
        :param second_txt_vocab_fn: only for combined input_type, txt w2v .json vocab mapping str to id file name
        (NOT path)
        :type second_txt_vocab_fn: str
        """
        self.data_dir = Path(data_dir) / f"{version}"
        self.linked_data_dir = self.data_dir.parent.parent / "linked_data" / f"{version}" \
            if ("umls" in input_type or input_type == "combined" or "MimicCuiSelectedTextIter" in str(doc_iterator)) \
            else None
        self.w2v_dir = (Path(data_dir) / "model") if input_type == "text" else (self.linked_data_dir.parent / "model")
        self.txt_w2v_dir = (Path(data_dir) / "model")
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
        if 'MimicCuiSelectedTextIter' in str(self.doc_iterator):
            print(type(self.doc_iterator), isinstance(self.doc_iterator, MimicCuiSelectedTextIter))
            print(self.doc_iterator)
            all_labels_iter = itertools.chain(MimicDocIter(self.train_file, slice_pos=3),
                                              MimicDocIter(self.dev_file, slice_pos=3),
                                              MimicDocIter(self.test_file, slice_pos=3))
        else:
            all_labels_iter = itertools.chain(self.doc_iterator(self.train_file, slice_pos=3),
                                              self.doc_iterator(self.dev_file, slice_pos=3),
                                              self.doc_iterator(self.test_file, slice_pos=3))
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(all_labels_iter)

        # if cui as input, get umls file paths for getting doc texts, id, len
        # id and labels will still come from .csv text file
        if "umls" in self.input_type or self.input_type == "combined" or \
                "MimicCuiSelectedTextIter" in str(self.doc_iterator):
            self.prune_cui = prune_cui
            self.cui_prune_file = self.linked_data_dir / (f"{version}_cuis_to_discard.pickle" if cui_prune_file is None
                                                          else cui_prune_file)
            self.umls_train_file = self.linked_data_dir / f"train_{version}_umls.txt"
            self.umls_dev_file = self.linked_data_dir / f"dev_{version}_umls.txt"
            self.umls_test_file = self.linked_data_dir / f"test_{version}_umls.txt"
            self.umls_doc_iterator = MimicCuiDocIter if umls_iterator is None else umls_iterator
            self.umls_doc_split_path = dict(train=self.umls_train_file,
                                            dev=self.umls_dev_file,
                                            test=self.umls_test_file)

        # load input to feature word 2 id vocab json saved from word_embedding step
        # file name convention for .json from word_embedding step same for umls and text versions
        # for combined input version, need 2 .json filenames, 1 for each version (umls and text)
        if self.input_type == "combined":
            umls_vocab_fname = self.w2v_dir / (f"processed_full_umls_pruned.json" if vocab_fn is None else
                                               vocab_fn)
            txt_vocab_fname = self.txt_w2v_dir / (f"processed_full_text_pruned.json" if second_txt_vocab_fn is None
                                                  else second_txt_vocab_fn)
            self.featurizer = Features(json.load(open(f"{umls_vocab_fname}")))
            self.txt_featurizer = Features(json.load(open(f"{txt_vocab_fname}")))
        else:
            try:
                # if input_type is NOT "umls" e.g. umls_kge etc. User MUST provide vocab_fn param!!
                # this is even if using the same .json file as "umls" w2v embedding model
                # e.g. processed_full_umls_pruned.json for input_type == "umls_kge"
                vocab_fname = self.w2v_dir / (f"processed_full_{input_type}_pruned.json" if vocab_fn is None else
                                              vocab_fn)
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
            if "MimicCuiSelectedTextIter" in str(self.doc_iterator):
                text_doc_iter = self.doc_iterator(self.umls_doc_split_path[split],
                                                  self.doc_split_path[split],
                                                  True,
                                                  self.cui_prune_file)
            else:
                text_doc_iter = self.doc_iterator(self.doc_split_path[split])
            for doc_id, doc_sents, doc_labels, doc_len in text_doc_iter:
                tokens = itertools.chain.from_iterable(doc_sents)
                input_ids = self.featurizer.convert_tokens_to_features(tokens, self.max_seq_length)
                self.doc2labels[doc_id] = doc_labels
                yield doc_id, input_ids, self.mlb.transform([doc_labels])

        elif "umls" in self.input_type:
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
        elif self.input_type == "combined":
            umls_doc_iter = self.umls_doc_iterator(self.umls_doc_split_path[split],
                                                   threshold=0.7,
                                                   pruned=self.prune_cui,
                                                   discard_cuis_file=self.cui_prune_file)
            text_doc_iter = self.doc_iterator(self.doc_split_path[split])
            for (text_data), (umls_data) in zip(text_doc_iter, umls_doc_iter):
                doc_id, txt_doc_sents, doc_labels, txt_doc_len = text_data
                u_id, u_sents, u_len = umls_data
                txt_tokens = itertools.chain.from_iterable(txt_doc_sents)
                txt_input_ids = self.txt_featurizer.convert_tokens_to_features(txt_tokens, self.max_seq_length)
                umls_tokens = itertools.chain.from_iterable(u_sents)
                umls_input_ids = self.featurizer.convert_tokens_to_features(umls_tokens, self.max_seq_length)
                self.doc2labels[doc_id] = doc_labels
                yield doc_id, txt_input_ids, umls_input_ids, self.mlb.transform([doc_labels])
        else:
            raise NotImplementedError(f"Invalid input_type option!")

    def get_dataset_stats(self, split):
        """
        min, max, mean token count for the given split
        For combined input_type the stats are given as a tuple with the first number being txt input stats
        :param split: train, dev, test
        :type split: str
        :return: value of dict[split]
        """
        if self.split_stats[split].get('mean') is not None:
            return self.split_stats[split]

        if self.input_type == "text":
            if "MimicCuiSelectedTextIter" in str(self.doc_iterator):
                doc_lens = list(map(int, self.doc_iterator(self.umls_doc_split_path[split],
                                                           self.doc_split_path[split],
                                                           True,
                                                           self.cui_prune_file,
                                                           slice_pos=4)))
            else:
                doc_lens = list(map(int, self.doc_iterator(self.doc_split_path[split], slice_pos=4)))
        elif "umls" in self.input_type:
            doc_lens = list(map(int, [doc_data[2] for doc_data in
                                      self.umls_doc_iterator(self.umls_doc_split_path[split],
                                                             pruned=self.prune_cui,
                                                             discard_cuis_file=self.cui_prune_file)]))
        elif self.input_type == "combined":
            txt_doc_lens = list(map(int, self.doc_iterator(self.doc_split_path[split], slice_pos=4)))
            umls_doc_lens = list(map(int, [doc_data[2] for doc_data in
                                           self.umls_doc_iterator(self.umls_doc_split_path[split],
                                                                  pruned=self.prune_cui,
                                                                  discard_cuis_file=self.cui_prune_file)]))
        else:
            raise NotImplementedError(f"Invalid input_type option!")

        if self.input_type == "combined":
            self.split_stats[split]['min'] = (np.min(txt_doc_lens), np.min(umls_doc_lens))
            self.split_stats[split]['max'] = (np.max(txt_doc_lens), np.max(umls_doc_lens))
            self.split_stats[split]['mean'] = (np.mean(txt_doc_lens), np.max(umls_doc_lens))
        else:
            self.split_stats[split]['min'] = np.min(doc_lens)
            self.split_stats[split]['max'] = np.max(doc_lens)
            self.split_stats[split]['mean'] = np.mean(doc_lens)

        return self.split_stats[split]

    def get_dataset(self, split):
        """
        Get indexable iterable of dataset split
        :param split: train, dev, or test
        :type split: str
        :return: List of (doc_id, input_ids, binarized_labels) for text or umls input_type or List of
        (doc_id, txt_input_ids, cui_input_ids, binarized_labels) for combined input_type
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


class CombinedDataset(Dataset):
    """
    Same as Dataset class, but self.data will come from a combined dataset. Re-define __getitem__ to return both txt and
    umls input tokens based on DataReader.get_dataset(split) function in line 220.

    mimic_collate_fn is also re-defined to return padded txt and umls input ids along with labrl_ids
    """

    def __getitem__(self, index):
        doc_id, txt_input_ids, umls_input_ids, labels_bin = self.data[index]
        return txt_input_ids, umls_input_ids, labels_bin

    @staticmethod
    def mimic_collate_fn(dataset_batch):
        txt_input_ids, umls_input_ids, label_ids = list(zip(*dataset_batch))
        txt_input_ids = list(map(torch.LongTensor, txt_input_ids))  # dtype: torch.int64
        umls_input_ids = list(map(torch.LongTensor, umls_input_ids))  # dtype: torch.int64
        label_ids = list(map(torch.Tensor, label_ids))  # dtype: torch.float32

        padded_txt_input_ids = pad_sequence(txt_input_ids,
                                            batch_first=True)  # shape: batch_size x max_seq_len in txt batch
        padded_umls_input_ids = pad_sequence(umls_input_ids,
                                             batch_first=True)  # shape: batch_size x max_seq_len in umls batch
        label_ids = torch.cat(label_ids, dim=0)  # shape: batch_size x num label classes

        return padded_txt_input_ids, padded_umls_input_ids, label_ids


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


def get_data(batch_size=8, dataset_class=Dataset, collate_fn=Dataset.mimic_collate_fn, **kwargs):
    dr = DataReader(**kwargs)
    train_data_loader = get_dataloader(dataset_class(dr.get_dataset('train'), dr.mlb), batch_size, True, collate_fn)
    dev_data_loader = get_dataloader(dataset_class(dr.get_dataset('dev'), dr.mlb), batch_size, False, collate_fn)
    test_data_loader = get_dataloader(dataset_class(dr.get_dataset('test'), dr.mlb), batch_size, False, collate_fn)
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
        train_stats = data_reader.get_dataset_stats("train")
    check_data_loader = True
    # checking snomed prune file
    if check_data_loader:
        dr, trd, dvd, ted = get_data(batch_size=8, dataset_class=Dataset, collate_fn=Dataset.mimic_collate_fn,
                                     data_dir="../../data/mimic3", version="50", input_type="text", prune_cui=True,
                                     cui_prune_file="50_cuis_to_discard_snomedcase4.pickle",
                                     doc_iterator=MimicCuiSelectedTextIter,
                                     max_seq_length=2500)  # max sequence length from train set is 2012
        temp = iter(trd)
        x, y = next(temp)
        print(f"x shape: {x.shape}, type: {x.dtype}\n")
        print(x)
        print(f"y shape: {y.shape}, type: {y.dtype}\n")
        print(y)

        # dr = DataReader(data_dir="../../data/mimic3", version="full", input_type="text")
        print(dr.get_dataset_stats('train'))

        print(np.transpose(np.nonzero(y)))

    # for KGE embedding testing
    check_data_loader_KGE = True
    # checking snomed prune file
    if check_data_loader_KGE:
        dr, trd, dvd, ted = get_data(batch_size=8, dataset_class=Dataset, collate_fn=Dataset.mimic_collate_fn,
                                     data_dir="../../data/mimic3", version="50", input_type="umls_kge", prune_cui=True,
                                     cui_prune_file="50_cuis_to_discard_snomednorel.pickle",
                                     vocab_fn="processed_full_umls_pruned.json")
        temp = iter(trd)
        x, y = next(temp)
        print(f"x shape: {x.shape}, type: {x.dtype}\n")
        print(x)
        print(f"y shape: {y.shape}, type: {y.dtype}\n")
        print(y)

    check_combined_data_loader = False
    if check_combined_data_loader:
        dr, trd, dvd, ted = get_data(batch_size=8, dataset_class=CombinedDataset,
                                     collate_fn=CombinedDataset.mimic_collate_fn, data_dir="../../data/mimic3",
                                     version="50", input_type="combined", prune_cui=True,
                                     cui_prune_file="50_cuis_to_discard_snomedbase.pickle",
                                     vocab_fn=None)
        temp = iter(trd)
        x_txt, x_umls, y = next(temp)
        print(f"x_txt shape: {x_txt.shape}, type: {x_txt.dtype}\n")
        print(x_txt)
        print(f"x_umls shape: {x_umls.shape}, type: {x_umls.dtype}\n")
        print(x_umls)
        print(f"y shape: {y.shape}, type: {y.dtype}\n")
        print(y)
