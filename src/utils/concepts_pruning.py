#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Python template with an argument parser and logger. Put all the "main" logic into the method called "main".
             Only use the true "__main__" section to add script arguments. The logger writes to a hidden folder './log/'
             and uses the name of this file, followed by the date (by default). The argument parser comes with a default
             option --quiet to keep the stdout clean.

@copyright: Copyright 2018 Deutsches Forschungszentrum fuer Kuenstliche
            Intelligenz GmbH or its licensors, as applicable.

@author: Noon Pokaratsiri Goldstein, adapted from code by Saadullah Amin
"""

import sys
import time
from datetime import date
import logging
import argparse
import traceback
import collections
import json


from src.utils.utils import get_freq_distr_plots, pickle_obj, get_dataset_semantic_types, prune_dfs_dict
from scispacy.umls_linking import UmlsEntityLinker
from pathlib import Path
from tqdm import tqdm


logger = logging.getLogger(__name__)

ICD9_SEMANTIC_TYPES = [
    'T017', 'T019', 'T020', 'T021', 'T022',
    'T023', 'T024', 'T029', 'T030', 'T031',
    'T032', 'T033', 'T034', 'T037', 'T038',
    'T039', 'T040', 'T041', 'T042', 'T046',
    'T047', 'T048', 'T049', 'T058', 'T059',
    'T060', 'T061', 'T062', 'T063', 'T065',
    'T067', 'T068', 'T069', 'T070', 'T074',
    'T075', 'T077', 'T078', 'T079', 'T080',
    'T081', 'T082', 'T102', 'T103', 'T109',
    'T114', 'T116', 'T121', 'T123', 'T125',
    'T126', 'T127', 'T129', 'T130', 'T131',
    'T169', 'T170', 'T171', 'T184', 'T185',
    'T190', 'T191', 'T192', 'T195', 'T196',
    'T197', 'T200', 'T201', 'T203'
]

def get_dataset_icd9_sem_types(file_path):
    """
    Keep only the semantic types in the dataset at file_path that correspond to possible ICD9 codes' semantic types

    :param file_path: Path to file containing dataset (mimic3) semantic types
    :return: intersection between semantic types in mimic3 and those that correspond to ICD9 cpdes
    :rtype: set
    """
    dataset_sem_types = set(get_dataset_semantic_types(file_path))
    return dataset_sem_types.intersection(set(ICD9_SEMANTIC_TYPES))


class ConceptCorpusReader:
    """
    Read linked_data corpus file for a specified split and obtain counts for each UMLS entity in each doc/sample
    """

    def __init__(self, mimic3_dir, split, version, threshold=None):

        """

        :param mimic3_dir: directory where mimic3 data files are
        :param split: dev, test, or train partition
        :param version: full vs 50
        :param threshold: confidence level threshold to include concept
        """

        self.umls_fname = Path(mimic3_dir) / f'{split}_{version}_umls.txt'
        self.docidx_to_concepts = dict()
        # [doc idx][sent id]: [((s1, e1), [concept1, concept2, ...]),(s2, e2,), [concept1, concept2, ...]]
        #self.docidx_to_concepts_simple = dict()
        # [doc idx]: {concept1, concept2, concept3, ....}
        self.docidx_to_ordered_concepts = dict()
        # [doc idx]: [list of concepts in the order of how they appear in the sample, can repeat]
        self.confidence_threshold = threshold if threshold is not None else 0.7
        # 0.7 is the default used in the concepts_linking.py

    def read_umls_file(self):
        """
        Extract entities from each sentence in each doc and store in a dict mapping doc_id, sent_id to list of
        UMLS entities



        :return:

        """
        with open(self.umls_fname) as rf:
            for line in tqdm(rf):
                line = line.strip()
                if not line:
                    continue
                line = json.loads(line)
                uid = list(line.keys())[0]
                doc_id, sent_id = list(map(int, uid.split("_")))
                if doc_id not in self.docidx_to_concepts:
                    self.docidx_to_concepts[doc_id] = dict()
                    # self.docidx_to_concepts_simple[doc_id] = set()
                    self.docidx_to_ordered_concepts[doc_id] = []
                self.docidx_to_concepts[doc_id][sent_id] = [
                    ((item['s'], item['e']), [ents[0] for ents in item['umls_ents']
                                              if float(ents[-1]) > self.confidence_threshold]) for item in line[uid]
                ]
                #self.docidx_to_concepts_simple[doc_id].update([ents[0] for item in line[uid] for ents in item['umls_ents']
                                              #if float(ents[-1]) > self.confidence_threshold])
                self.docidx_to_ordered_concepts[doc_id].extend([ents[0] for item in line[uid] for ents in item['umls_ents']
                                              if float(ents[-1]) > self.confidence_threshold])
                # each 'umls_ents' is a list of lists --> [[cui1, confidence score1],[cui2, confidence score2], ...]
                # ents[0] gets the cui, ents[-1] gets the confidence score

    def concept_dfs(self):
        """
        Concept frequency. Count of each unique concept for all docs in the corpus

        :return: Counter mapping each concept to its frequency
        """
        # dfs doc frequency, count of concepts/doc
        dfs = collections.Counter()
        for doc_id in self.docidx_to_concepts.keys():
            concepts = list()
            for sent_id in self.docidx_to_concepts[doc_id].keys():
                concepts.extend([concept for item in self.docidx_to_concepts[doc_id][sent_id] for concept in item[1]])
                # each item is a tuple, item[0] == (start, end) positions, item[1] == [concept1, concept2, ...]
            dfs += collections.Counter(concepts)
        return dfs


def get_partition_concept_freq(data_dir, split, version, threshold=None):
    """

    :param data_dir: Path to directory containing data files
    :param split: dev/test/train
    :param version: 1, full, 50
    :param threshold: min confidence threshold for cuis/entities to consider (0.0-1.0)
    :return: Counter of cui:count
    :rtype: Counter
    """
    partition_reader = ConceptCorpusReader(data_dir, split, version, threshold)

    logger.info(f'Reading {split} annotated UMLS file for {version} ...')
    partition_reader.read_umls_file()

    logger.info(f'Counting concepts in documents in {split}_{version}')
    partition_dfs = partition_reader.concept_dfs()
    logger.info(f'No. of unique concepts in {split} before pruning: {len(partition_dfs)}')  # e.g. ~90955 for train
    logger.info(f'Top-10 most common concepts in {split} before pruning: {partition_dfs.most_common(10)}')

    return partition_dfs


def get_dataset_dfs(data_dir, list_of_splits=["train", "dev", "test"], version="1", threshold=None):
    """

    :param data_dir: Path to directory containing data files
    :param list_of_splits: ["train", "dev", "test"]
    :type list_of_splits: List of str
    :param version: 1, full, 50
    :param threshold: min confidence threshold for cuis/entities to consider (0.0-1.0)
    :return: dict of Counters {split:Counter for that split}
    :rtype: dict
    """
    partitions_dfs = {split: get_partition_concept_freq(data_dir, split, version, threshold)
                     for split in list_of_splits}

    total_train_dev_test_dfs = partitions_dfs["train"] + partitions_dfs["dev"] + partitions_dfs["test"]
    # ~93137
    logger.info(f'No. of unique concepts in {list_of_splits} before pruning: {len(total_train_dev_test_dfs)}')

    return partitions_dfs


def get_unseen_cuis_to_discard(partition_dfs):
    """
    Make a set of unseen cuis to be discarded. i.e. We want to keep only entities/concepts that are found in train set

    :param partition_dfs: Dict of each partition (train, dev, test) concept freq, which is a Counter
    :type partition_dfs: dict
    :return: Set of cuis found in ONLU dev and test, but NOT in train partition
    :rtype: set
    """
    only_in_dev = set(partition_dfs["dev"].keys()) - partition_dfs["train"].keys()
    only_in_test = set(partition_dfs["test"].keys()) - partition_dfs["train"].keys()

    logger.info(f'No. of unique concepts in only in dev: {len(only_in_dev)}')
    logger.info(f'No. of unique concepts in only in test: {len(only_in_test)}')

    return only_in_dev.union(only_in_test)


def get_min_max_threshold(partition_dfs_counters, partition="train"):
    """

    :param partition_dfs_counters: Dict of Counters for all partitions
    :param partition: name of partition, train, dev, or test
    :return: min threshold, max threshold
    :rtype: tuple
    """
    top_100 = partition_dfs_counters[partition].most_common(100)
    lowest = partition_dfs_counters[partition].most_common()[::-1]

    logger.info(f"Top 100 concepts freq in {partition}: \n{top_100}")
    #logger.info(f"Least frequent 1000 concepts in {partition}: \n{lowest_100}")

    # dict of normalized frequency
    total_freq = sum(partition_dfs_counters[partition].values())
    num_unique_cuis = len(partition_dfs_counters[partition])
    logger.info(f"Total concept counts in {partition}: {total_freq}")
    logger.info(f"Total unique concepts in {partition}: {num_unique_cuis}")

    # find out absolute freq for max and min where max is > 1500x/1million and min is 0.1x/1million
    top_100_normalized = [(cui, freq * 1000000.00 / total_freq) for (cui, freq) in top_100 if freq * 1000000.00 / total_freq >= 1500]
    lowest_normalized = [(cui, freq * 1000000.00 / total_freq) for (cui, freq) in lowest if freq * 1000000.00 / total_freq >= 0.1]
    logger.info(f"Normalized top 100 cuis freq: in {partition}: \n{top_100_normalized[:100]}")
    logger.info(f"Normalzied lowest cuis freq: in {partition}: \n{lowest_normalized[:100]}")

    # find the freq of the cui in the lowest_normalized
    min_threshold = partition_dfs_counters[partition][lowest_normalized[0][0]]
    logger.info(f"min frequency threshold: {min_threshold}")
    num_unique_cuis_to_discarded = len([cui for cui, freq in partition_dfs_counters[partition].items() if freq < min_threshold])
    percent_cuis_to_discard = num_unique_cuis_to_discarded * 100.00 / num_unique_cuis
    logger.info(f"Discarding based on min frequency threshold amonts to: {percent_cuis_to_discard} % of cuis")

    # find the freq of the cui in the top normalized
    max_threshold = partition_dfs_counters[partition][top_100_normalized[-1][0]]
    logger.info(f"max frequency threshold: {max_threshold}")
    num_unique_cuis_above_max = len([cui for cui, freq in partition_dfs_counters[partition].items() if freq > max_threshold])
    percent_cuis_above_max = num_unique_cuis_above_max * 100.00 / num_unique_cuis
    logger.info(f"Discarding based on max frequency threshold amonts to: {percent_cuis_above_max} % of cuis")
    logger.info(f"total discarded cuis amount to: {percent_cuis_to_discard + percent_cuis_above_max} % of cuis")

    return min_threshold, max_threshold


def add_rare_and_freq_cuis_to_discard(partition_dfs, split, min_threshold=5, max_threshold=4000):
    """
    Make a set of cuis that are either too rare or too frequent according to specified min/max thresholds

    :param partition_dfs: multi-split dict of Counters from get_dataset_dfs
    :type partition_dfs: dict
    :param split: train, dev, test
    :type split: str
    :param min_threshold: min frequency
    :param max_threshold: max frequency
    :return: set of cuis to be discarded
    :rtype: set
    """
    logger.info(f"Pruning concepts in {split} with counts outside {min_threshold, max_threshold}")
    cuis_to_discard = set()
    for cui, freq in tqdm(partition_dfs[split].items()):
        if freq < min_threshold:
            cuis_to_discard.add(cui)
            continue
        if freq > max_threshold:
            cuis_to_discard.add(cui)
            continue

    logger.info(f"No. of unique concepts too rare/frequent to discard: {len(cuis_to_discard)}")
    return cuis_to_discard


def add_non_icd9_cuis_to_discard(partition_dfs, split, dataset_icd9_sem_types, spacy_umls_linker):
    """
    Make a set of cuis to be discarded for the split that do not correspond to icd9 tuis

    :param partition_dfs: multi-split dict of Counters from get_dataset_dfs
    :type partition_dfs: dict
    :param split: train, dev, test
    :type split: str
    :param dataset_icd9_sem_types: set of tuis for the dataset that are icd9 possible
    :type dataset_icd9_sem_types: set
    :param spacy_umls_linker: scispacy umls entity linker obj
    :return: cuis to be discarded
    :rtype: set
    """

    logger.info(f"Pruning concepts to keep only icd9 types:\n{dataset_icd9_sem_types}")
    cuis_to_discard = set()
    for cui in tqdm(partition_dfs[split].keys()):
        if any(tui not in dataset_icd9_sem_types for tui in spacy_umls_linker.kb.cui_to_entity[cui].types):
            # kb.cui_to_entity[cui] maps to scispacy.linking_utils.Entity class, which is a NamedTuple
            # see https://github.com/allenai/scispacy/blob/main/scispacy/linking_utils.py from commit@583e35e
            # spacy_umls_linker.kb.cui_to_entity[cui].types can also be index accessed:
            # spacy_umls_linker.kb.cui_to_entity[cui][3]
            # spacy_umls_linker.kb.cui_to_entity[cui].types is a List of Str as 1 cui can have multiple tui's
            cuis_to_discard.add(cui)

    logger.info(f"No. of unique concepts in {split} to discard: {len(cuis_to_discard)}")
    return cuis_to_discard


def main(cl_args):
    """Main loop"""
    start_time = time.time()

    # TUIs in mimic && icd9
    mimic_icd9_tuis = get_dataset_semantic_types(cl_args.semantic_type_file)
    logger.info(f"TUIs in MIMIC corresponding to ICD9 codes:\n{mimic_icd9_tuis}")

    # get cui freq for all splits
    all_partitions_dfs = get_dataset_dfs(cl_args.mimic3_dir, ["train", "dev", "test"], "50")
    pickle_obj(all_partitions_dfs, cl_args, cl_args.dict_pickle_file)

    # prune out unseen, too rare/frequent cuis, and cuis whose types not in icd9 types (for a specific partition/split)
    logger.info('Determining cuis to discard...')
    unseen_cuis = get_unseen_cuis_to_discard(all_partitions_dfs)
    pickle_obj(unseen_cuis, cl_args, cl_args.misc_pickle_file)

    prune_dfs_dict(all_partitions_dfs, unseen_cuis)
    logger.info(f'Number of unique concepts in train without unseen cuis: {len(all_partitions_dfs["train"])}')
    logger.info(f'Number of unique concepts in dev without unseen cuis: {len(all_partitions_dfs["dev"])}')
    logger.info(f'Number of unique concepts in test without unseen cuis: {len(all_partitions_dfs["test"])}')

    pickle_obj(all_partitions_dfs, cl_args, "dict_pickle_file_no_unseen")
    min_threshold, max_threshold = get_min_max_threshold(all_partitions_dfs)

    unseen_rare_freq_cuis = unseen_cuis.union(add_rare_and_freq_cuis_to_discard(all_partitions_dfs,
                                                                                cl_args.split,
                                                                                min_threshold,
                                                                                max_threshold))

    logger.info('Loading SciSpacy UmlsEntityLinker ...')
    linker = UmlsEntityLinker(name=cl_args.linker_name)
    cuis_to_discard = unseen_rare_freq_cuis.union(add_non_icd9_cuis_to_discard(all_partitions_dfs,
                                                                               cl_args.split,
                                                                               mimic_icd9_tuis,
                                                                               linker))

    logger.info(f"No. of all unique concepts to discard: {len(cuis_to_discard)}")
    prune_dfs_dict(all_partitions_dfs, cuis_to_discard)
    logger.info(f'Number of unique concepts in train after pruning: {len(all_partitions_dfs["train"])}')
    logger.info(f'Number of unique concepts in dev after pruning: {len(all_partitions_dfs["dev"])}')
    logger.info(f'Number of unique concepts in test after pruning: {len(all_partitions_dfs["test"])}')

    # pickle cuis to discard for version
    pickle_obj(cuis_to_discard, cl_args, cl_args.pickle_file)
    pickle_obj(all_partitions_dfs, cl_args, cl_args.dict_pickle_file)

    # plot freq distribution for the pruned train parition
    get_freq_distr_plots(all_partitions_dfs, "train", save_fig=True)


    lapsed_time = (time.time() - start_time) // 60
    logger.info(f"Took {lapsed_time} minutes!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        "--split", action="store", type=str, default="train",
        help="Partition name: train, dev, test"
    )
    parser.add_argument(
        "--split_file", action="store", type=str, default="train_50",
        choices=[
            "train_full", "dev_full", "test_full",
            "train_50", "dev_50", "test_50", "dev_1"
        ],
        help="Path to data split file."
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
        "--min", action="store", type=int, default=4,
        help="Min frequency threshold to include cuis"
    )
    parser.add_argument(
        "--max", action="store", type=int, default=8000,
        help="Max frequency threshold to include cuis"
    )
    parser.add_argument(
        "--cache_dir", action="store", type=str,
        default="/Users/noonscape/Documents/msc_thesis/projects/CoDER/scratch/scispacy",
        help="Path to SciSpacy cache directory. Optionally, set the environment "
             "variable ``SCISPACY_CACHE``."
    )
    parser.add_argument(
        "--semantic_type_file", action="store", type=str, default="data/mimic3/semantic_types_mimic.txt",
        help="Path to file containing semantic types in the MIMIC-III dataset"
    )
    parser.add_argument(
        "--pickle_file", action="store", type=str, default="cuis_to_discard",
        help="Path to pickle file for set of cuis to discard"
    )
    parser.add_argument(
        "--dict_pickle_file", action="store", type=str, default="pruned_partitions_dfs_dict",
        help="Path to pickle file for partitions dfs dict of counters"
    )
    parser.add_argument(
        "--misc_pickle_file", action="store", type=str, default="unseen_cuis",
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
    proj_folder = Path(__file__).resolve().parent.parent.parent
    log_folder = proj_folder / f"scratch/.log/{date.today():%y_%m_%d}"
    log_file = log_folder / f"{time.strftime('%Hh%Mm%Ss')}_{basename}.log"

    if not log_folder.exists():
        log_folder.mkdir(parents=True, exist_ok=False)

    logging.basicConfig(format="%(asctime)s - %(filename)s:%(funcName)s %(levelname)s: %(message)s",
                        filename=log_file,
                        disable_existing_loggers=False,
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
        print(f"All Done in (logged to {log_file}).")
