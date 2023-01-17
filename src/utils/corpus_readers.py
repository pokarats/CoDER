#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Sentence iterator classes for reading sentences from dataset/corpus files

@author: Noon Pokaratsiri Goldstein; this is a modification from the code base obtained from:

https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network/blob/master/preprocess_mimic3.py
https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network/blob/master/utils.py
and,
https://github.com/suamin/P4Q_Guttmann_SCT_Coding/blob/main/word2vec.py

"""
import itertools
from abc import ABC, abstractmethod
import csv
import platform

if platform.python_version() < "3.8":
    import pickle5 as pickle
else:
    import pickle
import json
from collections import deque


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


class ProcessedIterExtended(BaseIter):
    """
    Sentence iterator class for processing .csv file; this is the version from Multi-Res CNN
    """

    def __init__(self, filename, header=False, delimiter=",", slice_pos=None):
        """

        :param filename: path to corpus file
        :type filename: str or Path
        :param header: whether or not file contains a header row, if True skip this header row
        :type header: bool
        :param delimiter: char that each row is delimited by, e.g. default is ",", use "\t" for a .tsv file
        :type delimiter: str
        :param slice_pos: column position of the data to read e.g. for MIMIC-III texts are in 2
        :type slice_pos: int
        """
        self.filename = filename
        self.header = header
        self.delimiter = delimiter
        self.slice_pos = slice_pos

    def __iter__(self):
        with open(self.filename) as f:
            r = csv.reader(f, delimiter=self.delimiter)
            if self.header:
                next(r)
            for row in r:
                if self.slice_pos:
                    yield row[self.slice_pos]
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
                doc_len = row[4]
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


class MimicDocWholeSentIter(MimicDocIter):
    """
    Doc iterator class for processing .csv file; assuming file follows MultiResCNN format
    if slice_pos is None, yield: doc_id, text, labels, doc_len
    Use slice_pos to specify a particular field
    slice_pos 0 or 1 --> 'doc_id'
    slice_pos == 2 --> text as List of str sentences
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
                doc_sents = [sent.lstrip("[CLS] ") for sent in row[2].split(self.sep) if sent]
                doc_labels = row[3].split(';')
                doc_len = row[4]
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

    def __init__(self, filename, threshold=0.7, pruned=False, discard_cuis_file=None, store_sent_cui_span=False):
        super().__init__(filename, threshold, pruned, discard_cuis_file)
        # attributes to facilitate generator, not meant to be accessible
        self.store_sent_cui_span = store_sent_cui_span

    def __iter__(self):
        with open(self.filename) as rf:
            doc_ids = deque()
            doc_sents = None
            doc_len = 0
            # store spans for each cui for each doc, reset the dict after each doc
            cuis_to_doc_spans = dict()

            for line in rf:
                line = line.strip()
                if not line:
                    continue
                line = json.loads(line)
                uid = list(line.keys())[0]
                doc_id, sent_id = list(map(int, uid.split("_")))
                if doc_id not in doc_ids:
                    # first sent in doc
                    doc_ids.append(doc_id)

                    # if this is the start of a new doc, yield the doc sentences in the last doc
                    if doc_sents is not None:
                        temp_doc_sents = doc_sents
                        temp_doc_len = str(doc_len)
                        if self.store_sent_cui_span:
                            temp_cuis_to_doc_span = cuis_to_doc_spans

                        # reset for next doc
                        doc_sents = None
                        doc_len = 0
                        cuis_to_doc_spans = dict()

                        # yield last doc id, text, len
                        if self.store_sent_cui_span:
                            yield doc_ids.popleft(), temp_doc_sents, temp_doc_len, temp_cuis_to_doc_span
                        else:
                            yield doc_ids.popleft(), temp_doc_sents, temp_doc_len

                    # start of a new doc (this could also be first doc in dataset)
                    doc_sents = []
                cui_sent_tokens = [ents[0] for item in line[uid] for ents in item['umls_ents'] if float(ents[-1]) >
                                   self.confidence_threshold]

                if self.store_sent_cui_span:
                    for item in line[uid]:
                        for ents in item['umls_ents']:
                            if float(ents[-1]) > self.confidence_threshold:
                                cuis_to_doc_spans[ents[0]] = (sent_id, item['s'], item['e'])

                # skip empty sentences
                if not cui_sent_tokens:
                    continue
                if self.prune and self.cuis_to_discard is not None:
                    pruned_cui_sent_tokens = [cui_token for cui_token in cui_sent_tokens if cui_token not in
                                              self.cuis_to_discard]
                    # if empty after pruning, skip
                    if not pruned_cui_sent_tokens:
                        continue
                    doc_len += len(pruned_cui_sent_tokens)
                    doc_sents.append(pruned_cui_sent_tokens)
                else:
                    doc_sents.append(cui_sent_tokens)
                    doc_len += len(cui_sent_tokens)
            # after end of last sentence in file, yield the last doc
            if doc_sents is not None and doc_len > 0 and doc_ids:
                if self.store_sent_cui_span:
                    yield doc_ids.popleft(), doc_sents, doc_len, cuis_to_doc_spans
                else:
                    yield doc_ids.popleft(), doc_sents, doc_len


class MimicCuiSelectedTextIter(BaseIter):
    """
    Doc iterator for MIMIC-III text input_type dataset where each doc is represented by text spans corresponding to
    UMLS CUI entities
    each yield contains: doc_id, List of Lists of str word tokens, num str word tokens in doc
    """
    def __init__(self, cui_filename,
                 txt_filename,
                 pruned=False,
                 discard_cuis_file=None,
                 slice_pos=None,
                 sep='[SEP]',
                 cls='[CLS]',
                 threshold=0.7,
                 store_sent_cui_span=True):
        self.txt_iter = MimicDocWholeSentIter(txt_filename, slice_pos, sep, cls)
        self.cui_doc_iter = MimicCuiDocIter(cui_filename, threshold, pruned, discard_cuis_file, store_sent_cui_span)

    def __iter__(self):
        for (text_data), (cui_data) in zip(self.txt_iter, self.cui_doc_iter):
            selected_doc_sents = []
            selected_doc_len = 0
            doc_id, doc_sents_str, doc_labels, txt_og_doc_len = text_data
            u_doc_id, u_sents, u_len, cui_span_dict = cui_data
            for cui_sent in u_sents:
                text_sent = []
                for cui in cui_sent:
                    sent_idx, start_char, end_char = cui_span_dict.get(cui)
                    text_sent.append(doc_sents_str[sent_idx][start_char:end_char])
                selected_doc_len += len(text_sent)
                if text_sent:
                    selected_doc_sents.append(text_sent)
            yield doc_id, selected_doc_sents, doc_labels, selected_doc_len


if __name__ == '__main__':
    cui_doc_iter = MimicCuiDocIter("../../data/linked_data/50/dev_50_umls.txt")
    mimic_doc_iter = MimicDocIter("../../data/mimic3/50/dev_50.csv")

    dev_set_cui_docs = list(cui_doc_iter)
    dev_set_docs = list(mimic_doc_iter)
    print(f"num cui docs: {len(dev_set_cui_docs)}")
    print(f"num csv docs: {len(dev_set_docs)}")

    cui_id, cui_sents, cui_len = dev_set_cui_docs[-1]
    print(f"cui last doc id: {cui_id}"
          f"cui last docs: {cui_sents}\n"
          f"cui las doc len: {cui_len}\n")

    dev_id, sents, labels, dev_len = dev_set_docs[-1]
    print(f"dev last doc id: {dev_id}"
          f"dev last docs: {sents}\n"
          f"dev las doc len: {dev_len}\n")
