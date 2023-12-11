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
from abc import ABC, abstractmethod
import csv
import platform
import json
from collections import deque
import dgl
from dgl.dataloading import GraphDataLoader
from torch.utils import data
import logging

if platform.python_version() < "3.8":
    import pickle5 as pickle
else:
    import pickle


logger = logging.getLogger(__name__)


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
                # doc_sents = [sent.lstrip(f"{self.cls} ") for sent in row[2].split(self.sep) if sent]
                doc_sents = [" ".join([w for w in sent.split() if w != self.cls]) for sent in row[2].split(self.sep) if sent]
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

    def __str__(self):
        return f"MimicCuiDocIter(filename={self.filename}, " \
               f"threshold={self.confidence_threshold}, " \
               f"pruned={self.prune}, " \
               f"store_sent_cui_span={self.store_sent_cui_span})"

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
        self.slice_pos = slice_pos
        self.txt_iter = MimicDocWholeSentIter(txt_filename, None, sep, cls)
        self.cui_doc_iter = MimicCuiDocIter(cui_filename, threshold, pruned, discard_cuis_file, store_sent_cui_span)

    def __str__(self):
        return "MimicCuiSelectedTextIter"

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

            if self.slice_pos is None:
                yield doc_id, selected_doc_sents, doc_labels, selected_doc_len
            else:
                # slice_pos specified, return just that column
                if self.slice_pos == 0 or self.slice_pos == 1:
                    yield doc_id
                elif self.slice_pos == 2:
                    yield selected_doc_sents
                elif self.slice_pos == 3:
                    yield doc_labels
                elif self.slice_pos == 4:
                    yield selected_doc_len
                else:
                    raise IndexError(f"MIMIC-III *.csv only has 5 columns; double check your file!")


def get_dataloader(dataset, batch_size, shuffle, collate_fn, num_workers=8):
    """


    :param dataset: data.Dataset class for LAAT experiments or dgl.data.DGLDataset for GNN experiments with partitions
    specified
    :param batch_size:
    :param shuffle: True/False
    :type shuffle: bool
    :param collate_fn: collate_fn for respective Dataset Class e.g. Dataset.mimic_collate_fn, DGLDataset.collate_fn etc
    :return: data_loader for the specified dataset partition for LAAT experimenmts, for DGLDataset a tuple of dataloader
    graph_embedding_size, number of label classes
    :param num_workers: torch num_workers
    :type num_workers: int
    """
    if isinstance(dataset, data.Dataset):
        loader_func = data.DataLoader
        logger.info(f"using LAAT data.DataLoader for LAAT Dataset...")
    elif isinstance(dataset, dgl.data.DGLDataset):
        loader_func = GraphDataLoader
        logger.info(f"using DGL GraphDataLoader for GNN Dataset...")
    else:
        raise NotImplementedError(f"Invalid DataLoader/Dataset Option!!!")
    data_loader = loader_func(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    if isinstance(dataset, dgl.data.DGLDataset):
        # dim_nfeats is the embedding size in GNNDataset, gclasses == number of labels (50 or 8000+ for full)
        graph_emb_size, num_label_classes = dataset.dim_nfeats, dataset.gclasses
        logger.info(f"get_dataloader for DGLDataset returns: \n"
                    f"data_loader, graph embedding size ({graph_emb_size}), and total num labels: {num_label_classes}")
        return data_loader, graph_emb_size, num_label_classes

    logger.info(f"get_dataloader for LAAT Dataset returns data_loader only!")
    return data_loader


def get_data(batch_size, dataset_class, collate_fn, reader, **kwargs):
    """
    For DGLDataset, train/dev/test dataloader will be a tuple of dataloader, embedding_size, and num_label_classes

    :param batch_size:
    :type batch_size: int
    :param dataset_class: Dataset or DGLDataset
    :param collate_fn: Dataset.collate_fn or GNNDataset.collate_fn
    :param reader: DataReader or GNNDataReader
    :param kwargs: kwargs for DataReader/GNNDataReader Class, embedding_type, graph edge mode and self_loop options
    for GNNDataset can be passed here
    :return: datareader, train_data_loader, dev_data_loader, test_data_loader

    """

    dataset_class_attr = {k: kwargs.pop(k) for k in ["embedding_type", "mode", "self_loop", "raw_dir", "verbose",
                                                     "force_reload", "ehr_min_prob"]
                          if k in kwargs}

    if not dataset_class_attr:
        # in case attr dict is empty, then use default values
        dataset_class_attr = {"embedding_type": "snomedcase4",  # default == snomedcase4
                              "mode": 'base',  # -->  default == 'base'
                              "self_loop": True,  # --> default == True
                              # "raw_dir" if None --> default == PROJ_FOLDER / 'data'
                              "verbose": True,  # default == False
                              "ehr_min_prob": 0.3,  # default == 0.3, only used if for ehr_prob_mode GNNDataset
                              "force_reload": False}  # default == False

    dataset_class_attr["version"] = kwargs.get("version")  # this attr is also used in DataReader, do not pop

    if "laat_data" in str(dataset_class) or (str(dataset_class) == "Dataset"):
        logger.info(f"should be laat dataset_class: {dataset_class}")
        _ = kwargs.pop("pos_encoding", None)  # remove pos_encoding from laat DataReader if in kwargs

        # initialize datareader class after popping non-relevant keys
        logger.info(f"kwargs for DataReader from dr_params: {kwargs}")
        dr = reader(**kwargs)
        train_data_loader = get_dataloader(dataset_class(dr.get_dataset('train'),
                                                         dr.mlb),
                                           batch_size,
                                           True,
                                           collate_fn)
        dev_data_loader = get_dataloader(dataset_class(dr.get_dataset('dev'),
                                                       dr.mlb),
                                         batch_size,
                                         False,
                                         collate_fn)
        test_data_loader = get_dataloader(dataset_class(dr.get_dataset('test'),
                                                        dr.mlb),
                                          batch_size,
                                          False,
                                          collate_fn)
    elif "gnn_data" in str(dataset_class) or "GNNDataset" in str(dataset_class):
        logger.info(f"should be GNN dataset_class: {dataset_class}")
        dataset_class_attr["pos_encoding"] = kwargs.get("pos_encoding", False)  # need in both DataSet and DataReader
        dataset_class_attr["cui_prune_file"] = kwargs.get("cui_prune_file", None)
        logger.info(f"dataset_class_attr after updates: {dataset_class_attr}")

        # initialize datareader class after popping non-relevant keys
        logger.info(f"kwargs for DataReader from dr_params: {kwargs}")
        dr = reader(**kwargs)
        train_data_loader = get_dataloader(dataset_class(dr.get_dataset('train'), dr.mlb, 'train', **dataset_class_attr),
                                           batch_size,
                                           True,
                                           collate_fn)
        dev_data_loader = get_dataloader(dataset_class(dr.get_dataset('dev'), dr.mlb, 'dev', **dataset_class_attr),
                                         batch_size,
                                         False,
                                         collate_fn)
        test_data_loader = get_dataloader(dataset_class(dr.get_dataset('test'), dr.mlb, 'test', **dataset_class_attr),
                                          batch_size,
                                          False,
                                          collate_fn)
    else:
        raise NotImplementedError(f"{dataset_class} is an invalid Dataset Class option!!!")

    return dr, train_data_loader, dev_data_loader, test_data_loader


if __name__ == '__main__':
    cui_doc_iter = MimicCuiDocIter("../../data/linked_data/50/dev_50_umls.txt")
    mimic_doc_iter = MimicDocIter("../../data/mimic3/50/dev_50.csv")
    mimic_doc_sent_iter = MimicDocWholeSentIter("../../data/mimic3/50/dev_50.csv")
    mimic_selected_txt_iter = MimicCuiSelectedTextIter("../../data/linked_data/50/dev_50_umls.txt",
                                                       "../../data/mimic3/50/dev_50.csv",
                                                       True,
                                                       "../../data/linked_data/50/50_cuis_to_discard_snomedcase4.pickle")

    dev_set_cui_docs = list(cui_doc_iter)
    dev_set_docs = list(mimic_doc_iter)
    dev_sent_docs = list(mimic_selected_txt_iter)
    print(f"num cui docs: {len(dev_set_cui_docs)}")
    print(f"num csv docs: {len(dev_set_docs)}")
    print(f"num csv docs: {len(dev_sent_docs)}")

    cui_id, cui_sents, cui_len = dev_set_cui_docs[0]
    print(f"cui last doc id: {cui_id}\n"
          f"cui last docs: {cui_sents}\n"
          f"cui las doc len: {cui_len}\n")

    dev_id, sents, labels, dev_len = dev_sent_docs[0]
    print(f"dev last doc id: {dev_id}\n"
          f"dev last docs: {sents}\n"
          f"dev las doc len: {dev_len}\n")



