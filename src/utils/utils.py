#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: utitly functions used in concepts_pruning.py, baseline_models.py, sacred_word_embeddings.py, etc.
             The logger writes to a hidden folder './log/'
             and uses the name of this file, followed by the date (by default). The argument parser comes with a default
             option --quiet to keep the stdout clean. (if run from main with logger config, otherwise cout.txt in
             Sacred logging)

@author: Noon Pokaratsiri Goldstein
"""
import json
import sys
import os
import platform
if platform.python_version() < "3.8":
    import pickle5 as pickle
else:
    import pickle
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
import scipy
if platform.system() != 'Darwin':
    sys.path.append(os.getcwd())  # only needed for slurm
from src.utils.corpus_readers import MimicDocIter


logger = logging.getLogger(__name__)


def write_to_json(data, path):
    """
    :param data: [{id:"doc id", "labels_id": ["label1", "label2"]}, {...}]
    :param path:
    :return: None
    """
    with open(path, mode="w", encoding="utf-8") as out_file:
        json.dump(data, out_file, indent=4, ensure_ascii=False)
    logger.info(f"Object saved at: {path}")


def read_from_json(path):
    """
    :param path:
    :return:
    """
    logger.info(f"Reading from json file saved at: {path}")
    with open(path, mode="r", encoding="utf-8") as in_file:
        data = json.load(in_file)

    return data


def lines_from_file(file_path, delimiter="|"):
    """
    Yield line from file path with trailing whitespaces removed

    :param file_path: path to file
    :param delimiter: token type on which to split each line
    :return: each line with trailing whitespaces removed and split on delimiter
    """
    with open(file_path) as f:
        for line in f:
            yield line.rstrip().split(delimiter)


def whole_lines_from_file(file_path):
    """
    Yield line from file path with whitespaces removed

    :param file_path: path to file
    :param delimiter: token type on which to split each line
    :return: each line with whitespaces removed
    """
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def pickle_obj(obj_to_pickle, args_cl, which_pickle):
    """

    :param which_pickle: the cl_args arg for the desired pickle filename e.g. pickle_file
    :param obj_to_pickle:
    :param args_cl: parse args dict
    :return: None
    """
    data_folder = Path(args_cl.mimic3_dir)
    pickle_file = data_folder / f"{args_cl.version}_{which_pickle}.pickle"
    with open(pickle_file, 'wb') as handle:
        pickle.dump(obj_to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(pickle_file, 'rb') as handle:
        pickled = pickle.load(handle)

    assert(obj_to_pickle == pickled), f"Pickled object not the same as original!!"
    logger.info(f"Object pickled saved at: {pickle_file}")


def get_dataset_semantic_types(file_path):
    """
    File should contain semantic types in the dataset (e.g. MIMIC-III)

    :param file_path:
    :return: List of semantic types from dataset to be included
    :rtype: List
    """

    # each line has this format: DEVI|Devices|T074|Medical Device, 3rd item is the semantic type
    return [sem_type[2] for sem_type in lines_from_file(file_path)]


def get_dataset_icd9_codes(data_dir,
                           filename_pattern="*.json",
                           drop_column_names=['id', 'doc'],
                           label_column_name='labels_id'):
    """

    :param data_dir: data directory where all pre-processed data files with labels are, either .json files (if clef
    pre-processed, or the directory must be that of the text input .csv data files)
    :type data_dir: str or Path
    :param filename_pattern: pattern for filenames of the data files
    :type filename_pattern: str
    :param drop_column_names: names of columns to drop (if passing .json files from clef pre-processing)
    :type drop_column_names: List of str
    :param label_column_name: name of the column for the labels (if passing in .json files from clef pre-processing)
    :type label_column_name: str
    :return: set of all labels across partitions in the dataset in data_dir
    :rtype: set
    """
    icd9_code_set = set()
    if ".json" in filename_pattern:
        for data_file in Path(data_dir).iterdir():
            if data_file.is_file() and data_file.match(filename_pattern):
                data = read_from_json(data_file)
                if drop_column_names:
                    data_df = pd.DataFrame(data).drop(labels=drop_column_names, axis=1)
                else:
                    data_df = pd.DataFrame(data)

                for labels in data_df[label_column_name]:
                    icd9_code_set.update(labels)
    elif ".csv" in filename_pattern:
        for data_file in Path(data_dir).iterdir():
            if data_file.is_file() and data_file.match(filename_pattern):
                logger.info(f"Reading from .csv file saved at: {data_file}")
                label_iter = MimicDocIter(data_file, slice_pos=3)
                for labels in label_iter:
                    icd9_code_set.update(labels)
    else:
        raise NotImplementedError

    return icd9_code_set


def get_freq_distr_plots(partition_dfs_counters, partition, save_fig=False):
    """

    :param partition_dfs_counters: Dict of Counters for all partitions
    :param partition: name of partition, train, dev, or test
    :type partition: Str
    :param save_fig: whether the plot will be saved to file or not
    :return: None
    """

    logger.info(f'Plotting {partition} cui frequency distribution')

    cui_freq = pd.DataFrame(partition_dfs_counters[partition].most_common(), columns=['cuis', 'count'])
    fig, ax = plt.subplots(figsize=(18, 12))

    # Plot horizontal bar graph
    cui_freq.sort_values(by='count').plot.barh(x='cuis', y='count', ax=ax, color="brown")
    ax.set_title(f'Cuis Frequency Distribution in {partition}')
    ax.set_xlabel(f'Count')
    ax.set_ylabel(f'Cui')

    if save_fig:
        proj_folder = Path(__file__).resolve().parent.parent.parent
        log_folder = proj_folder / f"scratch/.log/{date.today():%y_%m_%d}"
        fig_path = log_folder / f"{partition}_cuis_freq.png"
        logger.info(f'Saving {partition} cui frequency distribution bar plot to {fig_path}')
        fig.savefig(fig_path, bbox_inches='tight')

    else:
        fig.tight_layout()
        plt.show()


def prune_dfs_dict(partition_dfs_counters, cuis_to_discard):
    """

    :param partition_dfs_counters: dict of Counters for each partition
    :param cuis_to_discard: set of cuis to discard
    :return: None, dict of partition df counters pop the keys in place
    """
    for key in partition_dfs_counters.keys():
        for cui in cuis_to_discard:
            partition_dfs_counters[key].pop(cui, None)


def token_to_token(token):
    """
    Function for TFIDFVectorizer to process data file
    :param token: a word/cui token
    :type token: str
    :return:
    :rtype:
    """
    return token


def sparse_to_array(array_like_obj):
    """

    :param array_like_obj: array or sparse
    :type array_like_obj: csr or ndarray
    :return: ndarray of the sparse or itself unchanged if already an ndarray
    :rtype: ndarray
    """
    if isinstance(array_like_obj, scipy.sparse.csr_matrix):
        return array_like_obj.toarray()
    else:
        return array_like_obj
