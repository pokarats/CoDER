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
from collections import deque, OrderedDict


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
    def __init__(self, filename, slice_pos=None):
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
                if self.slice_pos:
                    yield row[self.slice_pos].split()
                else:
                    yield row


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


class MimicDocIter(MimicIter):
    """
    Doc iterator class for processing .csv file; assuming file follows MultiResCNN format
    if slice_pos is None, yield: doc_id, text, labels, doc_len
    Use slice_pos to specify a particular field
    slice_pos 0 or 1 --> 'doc_id'
    slice_pos == 2 --> text as List of Lists of str tokens
    slice_pos == 3 --> [labels]
    slice_pos == 4 --> 'doc_len'
    """
    def __init__(self, filename, slice_pos=None, sep='[SEP]', cls='[CLS]'):
        super().__init__(filename, slice_pos, sep, cls)

    def __iter__(self):
        with open(self.filename) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                doc_id = f"{row[0]}_{row[1]}"
                doc_sents = [[w for w in sent.split() if w != self.cls] for sent in
                                    row[2].split(self.sep) if sent]
                doc_labels = row[3].split(';')
                doc_len = row[-1]
                if self.slice_pos is None:
                    yield doc_id, doc_sents, doc_labels, doc_len
                else:
                    # slice_pos specified, return just that column
                    if self.slice_pos == 0 or self.slice_pos == 1:
                        yield doc_id
                    elif self.slice_pos == 2:
                        yield doc_sents
                    elif self.slice_pos == 3:
                        yield doc_labels
                    elif self.slice_pos == 4:
                        yield doc_len
                    else:
                        raise IndexError(f"MIMIC-III *.csv only has 5 columns; double check your file!")


class MimicCuiDocIter(MimicCuiIter):
    """
    Doc iterator for MIMIC-III linked_data set where each doc is represented by UMLS CUI entities
    each yield contains: doc_id, List of Lists of CUIs, num CUIs in doc
    """
    def __init__(self, filename, threshold=0.7, pruned=False, discard_cuis_file=None):
        super().__init__(filename, threshold, pruned, discard_cuis_file)
        # attributes to facilitate generator, not meant to be accessible
        self._doc_ids = deque()
        self._doc_sents = None
        self._doc_len = 0

    def __iter__(self):
        with open(self.filename) as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                line = json.loads(line)
                uid = list(line.keys())[0]
                doc_id, sent_id = list(map(int, uid.split("_")))
                if doc_id not in self._doc_ids:
                    # first sent in doc
                    self._doc_ids.append(doc_id)

                    # if this is the start of a new doc, yield the doc sentences in the last doc
                    if self._doc_sents is not None:
                        doc_sents = self._doc_sents
                        doc_len = str(self._doc_len)

                        # reset for next doc
                        self._doc_sents = None
                        self._doc_len = 0

                        # yield last doc id, text, len
                        yield self._doc_ids.popleft(), doc_sents, doc_len

                    # start of a new doc (this could also be first doc in dataset)
                    self._doc_sents = []
                cui_sent_tokens = [ents[0] for item in line[uid] for ents in item['umls_ents'] if float(ents[-1]) >
                                   self.confidence_threshold]

                # skip empty sentences
                if not cui_sent_tokens:
                    continue
                if self.prune and self.cuis_to_discard is not None:
                    pruned_cui_sent_tokens = [cui_token for cui_token in cui_sent_tokens if cui_token not in
                                           self.cuis_to_discard]
                    # if empty after pruning, skip
                    if not pruned_cui_sent_tokens:
                        continue
                    self._doc_len += len(pruned_cui_sent_tokens)
                    self._doc_sents.append(pruned_cui_sent_tokens)
                else:
                    self._doc_sents.append(cui_sent_tokens)
                    self._doc_len += len(cui_sent_tokens)
            # after end of last sentence in file, yield the last doc
            if self._doc_sents is not None and self._doc_len > 0 and self._doc_ids:
                yield self._doc_ids.popleft(), self._doc_sents, self._doc_len
