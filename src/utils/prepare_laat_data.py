# -*- coding: utf-8 -*-

import nltk
import re
import sys
import unicodedata
import os
import pandas as pd
import json
import torch
import torch.utils.data as data

from pathlib import Path
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

WORD_TOKENIZER = RegexpTokenizer(r'\w+')
nltk.download('punkt', download_dir='/netscratch/samin/dev/miniconda3/envs/dre/nltk_data')
SENT_TOKENIZER = nltk.data.load('tokenizers/punkt/spanish.pickle')
PUNCT_TRANSLATE_UNICODE = dict.fromkeys(
    (i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')), u' ')


def read_doc(fname):
    """

    :param fname:
    :type fname:
    :return: [['tok1', 'tok2', ..., 'tokn'], ['tok1', 'tok2', ..., 'tokn'], ...]
    :rtype: List of Lists of str
    """
    sents = list()
    with open(fname, 'r', encoding='utf-8', errors='ignore') as rf:
        sents = process_doc(rf.read().strip())
    return sents


def strip_extra_whitespaces(s):
    return re.sub(r'[^\S\r\n]{2,}', ' ', s)


def strip_numeric(s):
    return re.sub(r'[0-9]+', '', s)


def strip_punct(s):
    return s.translate(PUNCT_TRANSLATE_UNICODE)


def sent_postprocess(s):
    return re.sub(r'\s+', ' ', s.strip())


def process_doc(doc):
    sents = list()
    for sent in SENT_TOKENIZER.tokenize(doc.lower()):
        sent = sent_postprocess(strip_extra_whitespaces(strip_numeric(strip_punct(sent))))
        if sent:
            sents.append(WORD_TOKENIZER.tokenize(sent))
    return sents


def convert_by_vocab(vocab, items, max_seq_length=None, blank_id=0, unk_id=1):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        # any word token not in vocab gets assigned unk_id by default
        output.append(vocab.get(item, default=unk_id))
    if max_seq_length is not None:
        if len(output) > max_seq_length:
            output = output[:max_seq_length]
        else:
            while len(output) < max_seq_length:
                output.append(blank_id)
    return output


def convert_tokens_to_ids(vocab, tokens, max_seq_length=None, blank_id=0, unk_id=1):
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

    def convert_tokens_to_ids(self, tokens, max_seq_length=None, blank_id=0, unk_id=1):
        return convert_by_vocab(self.vocab, tokens, max_seq_length, blank_id, unk_id)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


class DataReader:

    def __init__(self, max_seq_length=4000, use_focus_concept=True):
        data_dir = 'data/clef_format'
        # train_dev
        self.anns_train_dev_file = Path(data_dir) / 'train_dev' / 'anns_train_dev.txt'
        self.ids_dev_file = Path(data_dir) / 'train_dev' / 'ids_development.txt'
        self.ids_train_file = Path(data_dir) / 'train_dev' / 'ids_training.txt'
        self.train_dev_docs_dir = Path(data_dir) / 'train_dev' / 'docs-training'
        # test
        self.anns_test_file = Path(data_dir) / 'test' / 'anns_test.txt'
        self.ids_test_file = Path(data_dir) / 'test' / 'ids_test.txt'
        self.test_docs_dir = Path(data_dir) / 'test' / 'docs'

        # focus concepts
        self.focus_ids = set()
        with open(Path('data') / 'focus_concepts.txt') as rf:
            for line in rf.readlines():
                line = line.strip()
                if not line:
                    continue
                self.focus_ids.add(line)
        self.use_focus_concept = use_focus_concept

        self.doc2labels = dict()
        self.label2id = dict()

        for file in [self.anns_train_dev_file, self.anns_test_file]:
            with open(file) as rf:
                for line in rf:
                    line = line.strip()
                    if not line:
                        continue
                    items = line.split('\t')
                    if len(items) == 1:
                        doc_id = items[0]
                        labels = ['NONE', ]
                    else:
                        doc_id, labels = items
                        labels = labels.split('|')
                    self.doc2labels[doc_id] = labels
                    for label in labels:
                        if self.use_focus_concept:
                            _label = label.split('_')[0]
                            if _label not in self.focus_ids:
                                continue
                        if _label not in self.label2id:
                            self.label2id[_label] = len(self.label2id)

        self.id2label = {v: k for k, v in self.label2id.items()}
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([[j for l in self.label2id.keys() for j in [f'{l}_0', f'{l}_1']]])

        self.doc_ids = dict(train=list(), dev=list(), test=list())
        for split, split_file in [
            ('train', self.ids_train_file),
            ('dev', self.ids_dev_file),
            ('test', self.ids_test_file)
        ]:
            with open(split_file) as rf:
                for line in rf:
                    line = line.strip()
                    if not line:
                        continue
                    doc_id = line
                    self.doc_ids[split].append(doc_id)

        # labels metadata
        meta_df = pd.read_csv('data/metadata.tsv', sep='\t')
        meta_df.set_index('SCT_ID')

        self.sct2cui = meta_df.set_index('SCT_ID')['CUI'].to_dict()
        self.sct2tui = meta_df.set_index('SCT_ID')['TUI'].to_dict()
        self.sct2sg = meta_df.set_index('SCT_ID')['SG'].to_dict()
        self.sct2desc = meta_df.set_index('SCT_ID')['DESC'].to_dict()

        word2id = json.load(open(os.path.join('models/word2vec', 'word2vec.guttmann.100d_word2id.json')))
        self.featurizer = Features(word2id)
        self.max_seq_length = max_seq_length

    def _fit_transform(self, split):
        for doc_id in self.doc_ids[split]:
            if split == 'test':
                fname = os.path.join(self.test_docs_dir, f'{doc_id}.txt')
            else:
                fname = os.path.join(self.train_dev_docs_dir, f'{doc_id}.txt')
            sents = read_doc(fname)
            tokens = list()
            for sent in sents:
                tokens.extend(sent)
            x = self.featurizer.convert_tokens_to_features(tokens, self.max_seq_length)
            labels = self.doc2labels[doc_id]
            y = [0] * len(self.id2label)
            _ = [2] * len(self.id2label)
            for label in labels:
                if self.use_focus_concept:
                    _label, _flag = label.split('_')
                    if _label not in self.focus_ids:
                        continue
                lid = self.label2id[_label]
                y[lid] = 1
                if _flag == "1":
                    flag = 1
                else:
                    flag = 0
                _[lid] = flag
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor(y, dtype=torch.float)
            _ = torch.tensor(_, dtype=torch.long)
            yield doc_id, x, y, _

    def get_dataset(self, split):
        """
        Get indexable iterable of dataset split
        :param split:
        :type split:
        :return: List of (doc_id, input_ids, label_ids)
        :rtype:
        """
        dataset = list()
        for doc_id, input_ids, label_ids, _ in self._fit_transform(split):
            dataset.append((doc_id, input_ids, label_ids, _))
        return dataset


class Dataset(data.Dataset):

    def __init__(self, dataset, id2label, mlb):
        self.data = dataset
        self.id2label = id2label
        self.mlb = mlb

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        doc_id, input_ids, label_ids, _ = self.data[index]
        return input_ids.unsqueeze(0), label_ids.unsqueeze(0), _.unsqueeze(0)

    def collate_fn(data):
        input_ids, label_ids, _ = list(zip(*data))
        input_ids = torch.cat(input_ids, 0)
        label_ids = torch.cat(label_ids, 0)
        _ = torch.cat(_, 0)
        return input_ids, label_ids, _


def get_dataloader(dataset, batch_size, shuffle, num_workers=1):
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=Dataset.collate_fn
    )
    return data_loader


def get_data(batch_size=8, max_seq_length=128, use_focus_concept=True):
    dr = DataReader(max_seq_length, use_focus_concept)
    train_data_loader = get_dataloader(Dataset(dr.get_dataset('train'), dr.id2label, dr.mlb), batch_size, True)
    dev_data_loader = get_dataloader(Dataset(dr.get_dataset('dev'), dr.id2label, dr.mlb), batch_size, False)
    test_data_loader = get_dataloader(Dataset(dr.get_dataset('test'), dr.id2label, dr.mlb), batch_size, False)
    return dr, train_data_loader, dev_data_loader, test_data_loader


if __name__ == '__main__':
    train_word2vec = False
    if train_word2vec:
        with open('data/raw/sentences.txt', 'r', encoding='utf-8', errors='ignore') as rf, \
                open('data/raw/processed_sentences.txt', 'w', encoding='utf-8', errors='ignore') as wf:
            for line in tqdm(rf.readlines()):
                sent = line.strip().lower()
                if not sent:
                    continue
                sent = sent_postprocess(strip_extra_whitespaces(strip_numeric(strip_punct(sent))))
                sent = ' '.join(WORD_TOKENIZER.tokenize(sent)).strip()
                if sent:
                    wf.write(sent + '\n')
    check_data_reader = False
    if check_data_reader:
        dr = DataReader()
        iter_train = iter(dr.get_X_y_ids('train'))
        print(next(iter_train))
    check_data_loader = True
    if check_data_loader:
        trd, dvd, ted = get_data()
        temp = iter(trd)
        print(next(temp))