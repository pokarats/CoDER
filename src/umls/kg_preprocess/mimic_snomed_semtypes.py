#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Wrapper code for making an exclusion file containing the semantic type ID and descriptions of Semantic
             Types to EXCLUDE from SNOMED_CT when making the train/dev/test splits of CUIs for training KGE.
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
import time
from datetime import date
import logging
import argparse
import traceback
from pathlib import Path
from src.utils.utils import lines_from_file



ICD9_SEMANTIC_TYPES = {
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
}


def main(cl_args):
    snomed_fpath = Path(cl_args.snomed_path)
    mimic_fpath = Path(cl_args.mimic3_path)
    out_fpath = Path(cl_args.out_path)

    sct_tui_desc = dict()
    mimic_tui_desc = dict()

    for _, tui, sem_desc in lines_from_file(snomed_fpath, delimiter="\t"):
        sct_tui_desc[tui] = sem_desc
    for _, _, tui, sem_desc in lines_from_file(mimic_fpath, delimiter="|"):
        mimic_tui_desc[tui] = sem_desc

    mimic_or_icd9_sem_types = ICD9_SEMANTIC_TYPES.union(set(mimic_tui_desc.keys()))
    tuis_not_in_mimic = set(sct_tui_desc.keys()).difference(mimic_or_icd9_sem_types)

    # write tui and description of the TUIS NOT in Mimic or ICD9 SET
    with open(out_fpath, mode="w+") as out_f:
        for tui in tuis_not_in_mimic:
            print(f"{tui}\t{sct_tui_desc[tui]}", file=out_f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--snomed_path", action="store", type=str, default="data/umls/snomed_semantic_types.txt",
        help="Path to snomed semantic types file"
    )
    parser.add_argument(
        "--mimic3_path", action="store", type=str, default="data/mimic3/semantic_types_mimic.txt",
        help="Path to MIMIC-III semantic types file."
    )
    parser.add_argument(
        "--out_path", action="store", type=str, default="data/umls/sem_types_desc_exclusion.txt",
        help="Path to semantic types description to exclude file."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Do not print to stdout (log only)."
    )

    args = parser.parse_args(sys.argv[1:])

    # Setup logging and start timer
    basename = Path(__file__).stem
    proj_folder = Path(__file__).parent.parent.parent.parent
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
