#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: utitly functions used in concepts_pruning.py, baseline_models.py, etc.
             The logger writes to a hidden folder './log/'
             and uses the name of this file, followed by the date (by default). The argument parser comes with a default
             option --quiet to keep the stdout clean.

@copyright: Copyright 2018 Deutsches Forschungszentrum fuer Kuenstliche
            Intelligenz GmbH or its licensors, as applicable.

@author: Noon Pokaratsiri Goldstein
"""
import json
import pickle
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date


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


def lines_from_file(file_path, delimiter="|"):
    """
    Yield line from file path with trailing whitespaces removed

    :param file_path: path to file
    :param delimiter: token type on which to split each line
    :return: each line with trailing whitespaces removed
    """
    with open(file_path) as f:
        for line in f:
            yield line.rstrip().split(delimiter)


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
