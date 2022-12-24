#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Wrapper code for checking coverage of SNOMED_CT KGE and pruned MIMIC CUIs.
             The logger writes to a hidden folder './log/'
             and uses the name of this file, followed by the date (by default). The argument parser comes with a default
             option --quiet to keep the stdout clean.


@author: Noon Pokaratsiri Goldstein
"""
import sys
import os
import platform
if platform.system() != 'Darwin':
    sys.path.append(os.getcwd())
else:
    sys.path.extend(['/Users/noonscape/Documents/msc_thesis/projects/CoDER'])
if platform.python_version() < "3.8":
    import pickle5 as pickle
else:
    import pickle
import time
from datetime import date
import logging
import argparse
import traceback
from pathlib import Path
import numpy as np
from src.utils.utils import lines_from_file, read_from_json, write_to_json, pickle_obj
from src.utils.corpus_readers import ProcessedIterExtended


def check_overlap(snomed_ent_file, mimic_w2v_json):
    """

    :param snomed_ent_file:
    :type snomed_ent_file:
    :param mimic_w2v_json:
    :type mimic_w2v_json:
    :return:
    :rtype:
    """

    mimic_cui2idx = read_from_json(mimic_w2v_json)
    mimic_cui2snomed_idx = dict()
    mimic_idx2snomed_idx = dict()
    snomed_cui2idx = {cui: idx for idx, cui in ProcessedIterExtended(snomed_ent_file, header=False, delimiter="\t")}
    mimic_cui_no_rel = set()
    num_cui_no_rel = 0
    for cui, idx in mimic_cui2idx.items():
        try:
            snomed_idx = int(snomed_cui2idx[cui])
            mimic_cui2snomed_idx[cui] = snomed_idx
            mimic_idx2snomed_idx[idx] = snomed_idx
        except KeyError:
            logger.warning(f"CUI {cui} NOT in snomed KGE entities!")
            mimic_cui_no_rel.add(cui)
            num_cui_no_rel += 1
    try:
        ratio_no_rel = num_cui_no_rel / len(mimic_cui2idx)
    except ZeroDivisionError:
        logger.error(f"Check W2V .json vocab file; did not load into dict!!!")
        ratio_no_rel = 0.0
    logger.info(f"{num_cui_no_rel}\t({ratio_no_rel * 100}%) CUIs in Mimic NOT in SNOMED KGE entities!!")

    return ratio_no_rel, mimic_cui2snomed_idx, mimic_idx2snomed_idx, mimic_cui_no_rel


def update_embeddings(src_embeddings_fpath, tgt_embeddings_fpath, mapping, save_fname=None):
    """

    :param src_embeddings_fpath:
    :type src_embeddings_fpath:
    :param tgt_embeddings_fpath:
    :type tgt_embeddings_fpath:
    :param mapping:
    :type mapping:
    :param save_fname:
    :type save_fname:
    :return:
    :rtype:
    """
    src_embeddings_to_update = np.load(src_embeddings_fpath)
    tgt_embeddings = np.load(tgt_embeddings_fpath)

    logger.info(f"Existing embedding from {src_embeddings_fpath}\n"
                f"Size: {src_embeddings_to_update.shape}")

    # both src and tgt embedding have to be of the same dim
    assert tgt_embeddings.shape[1] == src_embeddings_to_update.shape[1]

    for mimic_idx in mapping.keys():
        # "<PAD>": 0, "<UNK>": 1
        if mimic_idx != 0 and mimic_idx != 1:
            snomed_ke_idx = mapping.get(mimic_idx, 0)
            src_embeddings_to_update[int(mimic_idx)] = tgt_embeddings[int(snomed_ke_idx)]

    if save_fname is None:
        outfname = Path(src_embeddings_fpath).stem.replace("umls", "snomedke")
    else:
        outfname = save_fname

    # new .npy will be saved in the same dir as the existing w2v UMLS CUI embedding's npy file
    output_dir = Path(src_embeddings_fpath).parent
    mat_fname = output_dir / f"{outfname}.npy"

    logger.info(f"Keeping exisisting word2id in {output_dir} and saving numpy KE weight matrix at {mat_fname} ...")
    logger.info(f"Updated weight npy file Shape: {src_embeddings_to_update.shape}")
    np.save(str(mat_fname), src_embeddings_to_update)

    return mat_fname


def main(cl_args):
    snomed_fpath = Path(cl_args.ent_file)
    mimic_fpath = Path(cl_args.w2v_json)
    cuimap_fpath = Path(cl_args.cui2idx_path)
    idxmap_fpath = Path(cl_args.idx2idx_path)
    discarded_cuis_fpath = Path(cl_args.mimic3_dir) / f"{cl_args.version}_cuis_to_discard.pickle"

    snomed_rel_coverage, mimic_2snomedidx_map, mimic_idx2idx, no_rel_cuis_set = check_overlap(snomed_fpath, mimic_fpath)
    logger.info(f"{1 - snomed_rel_coverage} of Mimic CUIs have SNOMED CT relations!")
    if cl_args.update_model:
        write_to_json(mimic_2snomedidx_map, cuimap_fpath, indent=None)
        write_to_json(mimic_idx2idx, idxmap_fpath, indent=None)

    if len(no_rel_cuis_set) > 0:
        logger.info(f"{len(no_rel_cuis_set)} CUIs added to discard set.")
        with open(discarded_cuis_fpath, 'rb') as handle:
            existing_discard_cuis = pickle.load(handle)
            logger.info(f"{len(existing_discard_cuis)} CUIs in existing cuis_to_discard set.")
        combined_discard_cuis = no_rel_cuis_set.union(existing_discard_cuis)
        pickle_obj(combined_discard_cuis, cl_args, cl_args.which_pickle)

    if cl_args.update_model:
        updated_npy_fpath = update_embeddings(cl_args.w2v_npy, cl_args.kge_npy, mimic_idx2idx, cl_args.save_fname)
        logger.info(f".npy weights where entities have SNOMED CT relations saved to {updated_npy_fpath}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ent_file", action="store", type=str, default="/Users/noonscape/Documents/msc_thesis/projects/CoDER/data/umls/entities.tsv",
        help="Path to snomed ct entities (entities.tsv) with rel from dglkge training"
    )
    parser.add_argument(
        "--w2v_json", action="store", type=str, default="/Users/noonscape/Documents/msc_thesis/projects/CoDER/data/linked_data/model/processed_full_umls_pruned.json",
        help="Path to MIMIC-III cuis to idx json mapping file from w2v training"
    )
    parser.add_argument(
        "--w2v_npy", action="store", type=str,
        default="/Users/noonscape/Documents/msc_thesis/projects/CoDER/data/linked_data/model/processed_full_umls_pruned.npy",
        help="Path to MIMIC-III embedding weights (.npy file) from w2v training"
    )
    parser.add_argument(
        "--kge_npy", action="store", type=str,
        default="/Users/noonscape/Documents/msc_thesis/projects/CoDER/data/umls/umls_25000_TransE_l2_entity.npy",
        help="Path to KGE embedding weights (.npy file) from DGL_KE training"
    )
    parser.add_argument(
        "--cui2idx_path", action="store", type=str, default="/Users/noonscape/Documents/msc_thesis/projects/CoDER/data/linked_data/model/mimic_snomed_cuiidx_base.json",
        help="Path to write/save mapping json file cuis in mimic3 dataset to dglkge idx."
    )
    parser.add_argument(
        "--idx2idx_path", action="store", type=str,
        default="/Users/noonscape/Documents/msc_thesis/projects/CoDER/data/linked_data/model/mimic_snomed_idxidx_base.json",
        help="Path to write/save mapping json file for mimic3 idx to dglkge idx in entity.tsv file."
    )
    parser.add_argument(
        "--mimic3_dir", action="store", type=str,
        default="/Users/noonscape/Documents/msc_thesis/projects/CoDER/data/linked_data/50",
        help="Path to mimic3_dir where an existing <version>_cuis_to_discard.pickle file is stored (pickle of a set)"
    )
    parser.add_argument(
        "--version", action="store", type=str,
        default="50",
        help="50 vs full version of the dataset"
    )
    parser.add_argument(
        "--save_fname", action="store", type=str,
        default="processed_full_snomedke_pruned",
        help="filename keyword for saving new <save_fname>.npy file; obj to save is a np matrix"
    )
    parser.add_argument(
        "--which_pickle", action="store", type=str,
        default="cuis_to_discard_snomedke",
        help="filename keyword for saving new <version>_<which_pickle>.pickle file; obj to pickle is a set"
    )
    parser.add_argument(
        "--update_model", action="store_true",
        help="Update/Rewrite .json and .npy files in model; files will be overwritten!"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Do not print to stdout (log only)."
    )

    args = parser.parse_args(sys.argv[1:])

    # Setup logging and start timer
    basename = Path(__file__).stem
    proj_folder = Path(__file__).parent.parent.parent
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
