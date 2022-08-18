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
import random
from datetime import date
import logging
import argparse
import traceback
import pandas as pd
import json

from sklearn.preprocessing import MultiLabelBinarizer
from src.utils.concepts_pruning import ConceptCorpusReader, pickle_obj
from src.models.baseline_models import RuleBasedClassifier
from pathlib import Path
from tqdm import tqdm


logger = logging.getLogger(__name__)

# TODO: get tags for train, dev, test and binarize them
#   binarize a given set of tags (e.g. output from rulebased model)
#   all formats in pandas Datafram
#   save as json afterward


def read_from_json(path):
    """
    :param path:
    :return:
    """
    with open(path, mode="r", encoding="utf-8") as in_file:
        data = json.load(in_file)

    return data


class PrepareDataset:
    def __init__(self, cl_arg, version):
        """

        :param mimic3_dir: directory where mimic3 data files are
        :param split: dev, test, or train partition
        :param version: full vs 50
        :param threshold: confidence level threshold to include concept
        """

        self.icd9_umls_fname = Path(cl_arg.data_dir) / 'ICD9_umls2020aa'
        self.preprocessed_data_dir = Path(cl_arg.data_dir) / 'linked_data' / f'top{version}' / 'preprocessed'
        self.partition_json_file = self.preprocessed_data_dir / f'{cl_arg.partition}.json'
        self.cui_to_discard = None
        self.mlbinarizer = MultiLabelBinarizer()
        self.all_icd9 = set()
        self.partition_raw_labels = None
        self.partition_binarized_labels = None

    def _get_raw_labels(self):
        logger.info(f"Get partition icd9 codes from file: {self.icd9_umls_fname}...")
        json_data = read_from_json(self.partition_json_file)
        partition_df = pd.DataFrame(json_data).drop(labels=['id', 'doc'], axis=1)
        self.partition_raw_labels = partition_df['labels_id']

    def _get_all_icd9(self):
        logger.info(f"Get all icd9 codes from file: {self.icd9_umls_fname}...")
        with open(self.icd9_umls_fname) as rfname:
            for line in tqdm(rfname):
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



def main(cl_args):
    """Main loop"""
    start_time = time.time()

    logger.info(f"Initialize RuleBasedClassifier and ConceptCorpusReader...")
    rule_based_model = RuleBasedClassifier(cl_args, cl_args.version, cl_args.extension)
    corpus_reader = ConceptCorpusReader(cl_args.mimic3_dir, cl_args.split, cl_args.version)
    corpus_reader.read_umls_file()

    results = dict()
    num_samples = len(corpus_reader.docidx_to_ordered_concepts)
    logger.info(f"fitting rule-based mode on {cl_args.version} {cl_args.split} partition...")
    for sample_idx in tqdm(corpus_reader.docidx_to_ordered_concepts.keys(),
                           total=num_samples,
                           desc=f"test_{cl_args.version}_cuis"):
        a_sample = corpus_reader.docidx_to_ordered_concepts[sample_idx]
        predicted_icd9 = rule_based_model.fit(a_sample, similarity_threshold=cl_args.min)
        results[sample_idx] = predicted_icd9

    sample_results_idx = random.sample(range(0, num_samples), 5)
    for idx in sample_results_idx:
        logger.info(f"sample idx {idx} results: \n {results[idx]}")

    pickle_obj(results, cl_args, cl_args.dict_pickle_file)

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
        "--mimic3_dir", action="store", type=str, default="data/linked_data/top50",
        help="Path to MIMIC-III data directory containing processed versions with linked_data"
             "of the top-50 and full train/dev/test splits."
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
        "--min", action="store", type=int, default=0.7,
        help="Min threshold for similarity"
    )
    parser.add_argument(
        "--min_num_labels", action="store", type=int, default=7,
        help="Min threshold for num of predicted labels before extending"
    )
    parser.add_argument(
        "--dict_pickle_file", action="store", type=str, default="extension_baseline_model_output_dict",
        help="Path to pickle file for dict mapping sample idx to output"
    )
    parser.add_argument(
        "--misc_pickle_file", action="store", type=str, default="misc_pickle",
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
