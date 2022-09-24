#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Sentence iterator classes for reading sentences from dataset/corpus files


@copyright: Copyright 2018 Deutsches Forschungszentrum fuer Kuenstliche
            Intelligenz GmbH or its licensors, as applicable.

@author: Noon Pokaratsiri Goldstein; this is a modification from the code base obtained from:

https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network/blob/master/preprocess_mimic3.py
https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network/blob/master/utils.py
and,
https://github.com/suamin/P4Q_Guttmann_SCT_Coding/blob/main/word2vec.py

"""

from abc import ABC, abstractmethod
import csv
import pickle
import json


class BaseIter(ABC):
    filename = None

    @abstractmethod
    def __iter__(self):
        pass


class CorpusIter(BaseIter):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename, encoding='utf-8', errors='ignore') as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                sentence_tokens = line.split()
                yield sentence_tokens


class ProcessedIter(BaseIter):
    """
    Sentence iterator class for processing .csv file; this is the version from Multi-Res CNN
    """
    def __init__(self, filename, slice_pos=3):
        """

        :param filename: path to corpus file
        :type filename: str or Path
        :param slice_pos: column position of the data to read e.g. for MIMIC-III texts are in 2
        :type slice_pos: int
        """
        self.filename = filename
        self.slice_pos = slice_pos

    def __iter__(self):
        with open(self.filename) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                yield row[self.slice_pos].split()


class MimicIter(ProcessedIter):
    """
    .csv file sentence iterator class; use this for MIMIC-III dataset to remove sep and cls tokens
    """
    def __init__(self, filename, slice_pos=2, sep='[SEP]', cls='[CLS]'):
        """
        :param sep: sentence separator token e.g. [SEP]
        :type sep: str
        :param cls: beginning sentence separator token e.g. [CLS]
        :type cls: str
        """
        super().__init__(filename, slice_pos)
        self.sep = sep
        self.cls = cls

    def __iter__(self):
        with open(self.filename) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                for sent_tokens in [[w for w in sent.split() if w != self.cls] for sent in
                                    row[self.slice_pos].split(self.sep) if sent]:
                    yield sent_tokens


class MimicCuiIter(BaseIter):
    """
    Sentence iterator for MIMIC-III linked_data set where each doc is represented by UMLS CUI entities
    """
    def __init__(self, filename, threshold=0.7, pruned=False, discard_cuis_file=None):
        """
        :param threshold: confidence threshold for UMLS CUIS, default == 0.7 from scispacy
        :type threshold: float
        :param pruned: True to discard CUIs pruned in concepts_pruning.py step
        :type pruned: bool
        :param discard_cuis_file: path to pickle file of CUIs to discard from concepts_pruning.py step
        :type discard_cuis_file: str or Path
        """
        self.filename = filename
        self.confidence_threshold = threshold if threshold is not None else 0.7
        self.prune = pruned
        self.cuis_to_discard = None

        if discard_cuis_file is not None:
            with open(discard_cuis_file, 'rb') as handle:
                self.cuis_to_discard = pickle.load(handle)

    def __iter__(self):
        with open(self.filename) as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                line = json.loads(line)
                uid = list(line.keys())[0]
                cui_sent_tokens = [ents[0] for item in line[uid] for ents in item['umls_ents'] if float(ents[-1]) >
                                   self.confidence_threshold]
                if not cui_sent_tokens:
                    continue
                if self.prune and self.cuis_to_discard is not None:
                    yield [cui_token for cui_token in cui_sent_tokens if cui_token not in self.cuis_to_discard]
                else:
                    yield cui_sent_tokens
