#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Python template with an argument parser and logger. Put all the "main" logic into the method called "main".
             Only use the true "__main__" section to add script arguments. The logger writes to a hidden folder './log/'
             and uses the name of this file, followed by the date (by default). The argument parser comes with a default
             option --quiet to keep the stdout clean.


@copyright: Copyright 2018 Deutsches Forschungszentrum fuer Kuenstliche
            Intelligenz GmbH or its licensors, as applicable.

@author: Noon Pokaratsiri Goldstein
"""

import sys
import time
from datetime import date
import logging
import argparse
import traceback
import pandas as pd
import pickle

from sklearn.preprocessing import MultiLabelBinarizer
from src.utils.utils import read_from_json
from src.utils.concepts_pruning import ConceptCorpusReader
from pathlib import Path
from tqdm import tqdm


logger = logging.getLogger(__name__)


class PrepareData:
    def __init__(self, cl_arg):
        self.icd9_umls_fname = Path(cl_arg.data_dir) / 'ICD9_umls2020aa'
        self.linked_data_dir = Path(cl_arg.data_dir) / 'linked_data' / cl_arg.version
        self.preprocessed_data_dir = self.linked_data_dir / 'preprocessed'
        self.cuis_to_discard = None
        self.rule_based_name = None
        self.all_icd9 = set()
        self.all_partitions_icd9 = set()
        self.dataset_cuis = dict()
        self.dataset_raw_labels = dict()
        self.dataset_binarized_labels = dict()
        self.mlbinarizer = None

        self._get_all_icd9()

    def _get_all_icd9(self):
        logger.info(f"Get all icd9 codes from file: {self.icd9_umls_fname}...")
        with open(self.icd9_umls_fname) as fname:
            for line in tqdm(fname):
                line = line.strip()
                if not line:
                    continue
                # ICD9_umls2020aa is \t separated
                try:
                    icd9, *_ = line.split('\t')
                except ValueError:
                    print(f"icd9 code in {line} missing!")
                    continue
                self.all_icd9.add(icd9)

    def reset_partitions_label_set(self):
        self.all_partitions_icd9 = set()

    def get_all_partitions_icd9(self, partitions=['train', 'dev', 'test'],
                                include_rule_based=False,
                                filename=None,
                                add_name='rule-based'):
        if not self.dataset_raw_labels:
            for partition in partitions:
                self.get_partition_labels(partition)
            if include_rule_based and filename:
                self.add_predicted_labels(filename, name=add_name)

        for key in self.dataset_raw_labels.keys():
            if key != self.rule_based_name:
                for labels in self.dataset_raw_labels[key]:
                    self.all_partitions_icd9.update(labels)
            if include_rule_based:
                for labels in self.dataset_raw_labels[self.rule_based_name]:
                    self.all_partitions_icd9.update(labels)

        return self.all_partitions_icd9

    def load_cuis_to_discard(self, filepath):
        logger.info(f"Loading cuis to discard from pickle fie: {filepath}...")
        with open(filepath, 'rb') as handle:
            self.cuis_to_discard = pickle.load(handle)

    def init_mlbinarizer(self, labels=None):
        self.mlbinarizer = MultiLabelBinarizer(classes=tuple(self.all_icd9) if not labels else tuple(labels))

    def get_partition_data(self, partition, version, pruning_file="50_cuis_to_discard.pickle"):
        corpus_reader = ConceptCorpusReader(self.linked_data_dir, partition, version)
        corpus_reader.read_umls_file()

        pruning_file_path = self.linked_data_dir / pruning_file
        if not self.cuis_to_discard:
            self.load_cuis_to_discard(pruning_file_path)

        logger.info(f"Prune cuis from {partition} samples...")
        pruned_samples = []
        for _, cuis in corpus_reader.docidx_to_ordered_concepts.items():
            a_sample = {'cui': [cui for cui in cuis if cui not in self.cuis_to_discard]}
            pruned_samples.append(a_sample)
        pruned_samples_df = pd.DataFrame(pruned_samples)
        self.dataset_cuis[partition] = pruned_samples_df['cui']

        return self.dataset_cuis[partition]

    def get_partition_labels(self, partition):
        data_path = self.preprocessed_data_dir / f"{partition}.json"
        logger.info(f"Get partition icd9 codes from file: {data_path}...")
        json_data = read_from_json(data_path)
        partition_labels_df = pd.DataFrame(json_data).drop(labels=['id', 'doc'], axis=1)
        self.dataset_raw_labels[partition] = partition_labels_df['labels_id']

        return self.dataset_raw_labels[partition]

    def add_predicted_labels(self, filename, name='rule_based'):
        """

        :param filename: results from rule-based model from previous step or saved json file
        :type filename: Path, str, or actual obj
        :param name: name to save the results under
        :type name: str
        :return: raw labels
        :rtype: pd.Series
        """
        self.rule_based_name = name
        if isinstance(filename, Path) or isinstance(filename, str):
            if isinstance(filename, Path) and filename.exists():
                data_path = filename
            elif isinstance(filename, str):
                data_path = self.linked_data_dir / f"{filename}"
            else:
                raise FileNotFoundError
            logger.info(f"Get predicted icd9 codes from file: {data_path}...")
            json_data = read_from_json(data_path)
        else:
            json_data = filename

        partition_labels_df = pd.DataFrame(json_data).drop(labels=['id'], axis=1)
        self.dataset_raw_labels[name] = partition_labels_df['labels_id']

        return self.dataset_raw_labels[name]

    def get_binarized_labels(self, partition):
        if not self.mlbinarizer:
            raise AttributeError(f"MLBinarizer has NOT been initialized!!!")
        if not self.dataset_raw_labels:
            raise AttributeError(f"No raw labels loaded!!")
        if partition in self.dataset_raw_labels.keys():
            self.dataset_binarized_labels[partition] = \
                self.mlbinarizer.fit_transform(self.dataset_raw_labels[partition])
        else:
            raise KeyError(f"No raw labels from {partition} to binarize!")

        return self.dataset_binarized_labels[partition]


def main(cl_args):
    """Main loop"""
    start_time = time.time()

    logger.info(f"Preparing data for baselin and/or eval...")
    prep = PrepareData(cl_args)
    partitions = ['test', 'dev', 'train']
    all_partitions_labels = prep.get_all_partitions_icd9(partitions, include_rule_based=True, filename=cl_args.filename)
    prep.init_mlbinarizer(labels=all_partitions_labels)

    binarized_labels = dict()
    dataset_cuis = dict()

    for split in partitions:
        binarized_labels[split] = prep.get_binarized_labels(split)
        dataset_cuis[split] = prep.get_partition_data(split, cl_args.version, cl_args.misc_pickle_file)

    binarized_labels['rule_based'] = prep.get_binarized_labels('rule_based')

    assert len(binarized_labels['test']) == len(binarized_labels['rule_based'])
    assert len(binarized_labels['train'] == len(dataset_cuis['train']))

    logger.info(f"Sample from train data: \n{dataset_cuis['train'][:3]}")

    lapsed_time = (time.time() - start_time)
    time_minute = int(lapsed_time // 60)
    time_secs = int(lapsed_time % 60)
    logger.info(f"Module took {time_minute}:{time_secs} minutes!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", action="store", type=str, default="data",
        help="Path to data directory containing both the ICD9_umls2020aa file and the pickle file from concept_pruning"
    )
    parser.add_argument(
        "--version", action="store", type=str, default="50",
        help="Name of mimic-III dataset version to use: full vs 50 (for top50) or 1 for a 1 sample version"
    )
    parser.add_argument(
        "--split", action="store", type=str, default="test",
        help="Partition name: train, dev, test"
    )
    parser.add_argument(
        "--filename", action="store", type=str, default="50_baseline_model_output.json",
        help="Partition name: train, dev, test"
    )
    parser.add_argument(
        "--misc_pickle_file", action="store", type=str, default="50_cuis_to_discard.pickle",
        help="Path to miscellaneous pickle file e.g. for set of unseen cuis to discard"
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False,
        help="Do not print to stdout (log only)."
    )

    args = parser.parse_args(sys.argv[1:])

    # Setup logging and start timer
    basename = Path(__file__).stem
    proj_folder = Path(__file__).resolve().parent.parent.parent
    log_folder = proj_folder / f"scratch/.log/{date.today():%y_%m_%d}"
    log_file = log_folder / f"{time.strftime('%Hh%Mm%Ss')}_{basename}.log"

    if not log_folder.exists():
        log_folder.mkdir(parents=True, exist_ok=False)

    logging.basicConfig(format="%(asctime)s - %(filename)s:%(funcName)s %(levelname)s: %(message)s",
                        filename=log_file,
                        level=logging.INFO)

    # Manage the LOG and where to pipe it (log file only or log file + STDOUT)
    if not args.quiet:
        fmtr = logging.Formatter(fmt="%(funcName)s %(levelname)s: %(message)s")
        stderr_handler = logging.StreamHandler()
        stderr_handler.formatter = fmtr
        logging.getLogger().addHandler(stderr_handler)
        logging.info("Printing activity to the console")
    logger = logging.getLogger(__name__)
    logger.info(f"Running parameter \n{str(args.__dict__)}")

    try:
        main(args)
    except Exception as exp:
        if not args.quiet:
            print(f"Unhandled error: {repr(exp)}")
        logger.error(f"Unhandled error: {repr(exp)}")
        logger.error(traceback.format_exc())
        sys.exit(-1)
    finally:
        print(f"All Done, logged to {log_file}).")
