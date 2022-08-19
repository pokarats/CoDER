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

from utils.concepts_pruning import ConceptCorpusReader
from utils.prepare_data import PrepareData
from utils.eval import all_metrics, log_metrics, simple_score
from utils.utils import write_to_json
from models.baseline_models import RuleBasedClassifier, TFIDFBasedClassifier
from pathlib import Path
from tqdm import tqdm
import numpy as np


# TODO: write wrapper class for model selection between baseline models

def main(cl_args):
    """Main loop"""
    start_time = time.time()

    """
    logger.info(f"\n==========START RULE-BASED MODEL==============\n")
    logger.info(f"Initialize RuleBasedClassifier and ConceptCorpusReader...")
    rule_based_model = RuleBasedClassifier(cl_args, cl_args.version, cl_args.extension)
    corpus_reader = ConceptCorpusReader(cl_args.mimic3_dir, cl_args.split, cl_args.version)
    corpus_reader.read_umls_file()

    results = []
    num_samples = len(corpus_reader.docidx_to_ordered_concepts)
    logger.info(f"fitting rule-based mode on {cl_args.version} {cl_args.split} partition...")
    for sample_idx in tqdm(corpus_reader.docidx_to_ordered_concepts.keys(),
                           total=num_samples,
                           desc=f"test_{cl_args.version}_cuis"):
        a_sample = corpus_reader.docidx_to_ordered_concepts[sample_idx]
        predicted_icd9 = rule_based_model.fit(a_sample, similarity_threshold=cl_args.min)
        predicted = {'id': sample_idx,
                     'labels_id': list(predicted_icd9)}
        results.append(predicted)

    sample_results_idx = random.sample(range(0, num_samples), 5)
    for idx in sample_results_idx:
        logger.info(f"sample idx {idx} results: \n {results[idx]}")

    data_folder = Path(cl_args.mimic3_dir)
    result_file = data_folder / f"{cl_args.version}_{cl_args.dict_pickle_file}.json"
    write_to_json(results, result_file)
    """
    logger.info(f"\n==========START DATA PREP FOR TFIDF MODEL==============\n")
    logger.info(f"Preparing data for baseline and/or eval...")
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


    logger.info(f"\n==========START EVAL ON RULE-BASED MODEL==============\n")
    metrics = all_metrics(binarized_labels['rule_based'],
                          binarized_labels[cl_args.split],
                          k=[1, 3, 5],
                          yhat_raw=None,
                          calc_auc=False)
    log_metrics(metrics)

    scores = pd.DataFrame()
    rule_based_eval = simple_score(binarized_labels['rule_based'],
                                   binarized_labels[cl_args.split],
                                   'rule_based')
    scores = pd.concat([scores, rule_based_eval])
    logger.info(f"rule based results eval by sklearn: \n{scores.head()}")

    logger.info(f"\n==========START TFIDF MODELS PIPELINES==============\n")
    logger.info(f"Re-binarize labels for train, test, val without rule-based labels...")

    prep.reset_partitions_label_set()
    all_partitions_labels = prep.get_all_partitions_icd9(partitions, include_rule_based=False)
    prep.init_mlbinarizer(labels=all_partitions_labels)

    for split in partitions:
        binarized_labels[split] = prep.get_binarized_labels(split)
        assert len(binarized_labels[split] == len(dataset_cuis[split]))

    logger.info(f"Trying LogisticRegresion, SGD, and SVM...")
    tfidf_clf = TFIDFBasedClassifier(cl_args)
    tfidf_clf.execute_pipeline(dataset_cuis['train'],
                               binarized_labels['train'],
                               dataset_cuis['test'],
                               binarized_labels['test'])
    scores = pd.concat([scores, tfidf_clf.eval_scores])
    logger.info(f"TFIDF-based results eval by sklearn: \n{scores.head(7)}")
    for i in range(len(tfidf_clf.eval_metrics)):
        log_metrics(tfidf_clf.eval_metrics[i])

    logger.info(f"Trying stack model...")
    tfidf_clf.execute_stack_pipeline(dataset_cuis['train'],
                                     binarized_labels['train'],
                                     dataset_cuis['test'],
                                     binarized_labels['test'])
    scores = pd.concat([scores, tfidf_clf.eval_scores])
    logger.info(f"TFIDF-based stack model results eval by sklearn: \n{scores.head(10)}")
    log_metrics(tfidf_clf.eval_metrics[-1])

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
        "--mimic3_dir", action="store", type=str, default="data/linked_data/50",
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
        "--extension", action="store", default=None,
        help="Extension type for when cui not matching any icd9, options: best or all"
    )
    parser.add_argument(
        "--seed", action="store", type=int, default=23,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--scispacy_model_name", action="store", type=str, default="en_core_sci_lg",
        help="SciSpacy model to use for UMLS concept linking. e.g. en_core_sci_lg"
    )
    parser.add_argument(
        "--linker_name", action="store", type=str, default="scispacy_linker",
        help="SciSpacy UMLS Entity Linker name. e.g. scispacy_linker"
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
        "--cache_dir", action="store", type=str,
        default="/Users/noonscape/Documents/msc_thesis/projects/CoDER/scratch/scispacy",
        help="Path to SciSpacy cache directory. Optionally, set the environment "
             "variable ``SCISPACY_CACHE``."
    )
    parser.add_argument(
        "--dict_pickle_file", action="store", type=str, default="baseline_model_output",
        help="Path to pickle file for dict mapping sample idx to output"
    )
    parser.add_argument(
        "--filename", action="store", type=str, default="50_baseline_model_output.json",
        help="Rule-based result output file, should be in json format"
    )
    parser.add_argument(
        "--misc_pickle_file", action="store", type=str, default="50_cuis_to_discard.pickle",
        help="Path to miscellaneous pickle file e.g. for set of unseen cuis to discard"
    )
    parser.add_argument(
        "--n_process", action="store", type=int, default=48,
        help="Number of processes to run in parallel with spaCy multi-processing."
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, default=4096,
        help="Batch size to use in combination with spaCy multi-processing."
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False,
        help="Do not print to stdout (log only)."
    )

    args = parser.parse_args(sys.argv[1:])

    # Setup logging and start timer
    basename = Path(__file__).stem
    proj_folder = Path(__file__).resolve().parent.parent
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
