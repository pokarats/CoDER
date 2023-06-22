#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Updating ICD9_descriptions file in no_umls_icd9 directory after re-querying


@author: Noon Pokaratsiri Goldstein
"""
import sys
import os
import platform
if platform.system() != 'Darwin':
    sys.path.append(os.getcwd())
else:
    sys.path.extend(['/Users/noonscape/Documents/msc_thesis/projects/CoDER'])

import pandas as pd
import argparse
import logging


from pathlib import Path


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    stream=sys.stdout,
    level=logging.INFO
)
logger = logging.getLogger(__file__)


UMLS_VERSION = "2020AA"


def main(args):
    base_input_file = Path(args.data_dir) / "ICD9_descriptions"
    umls2020_input_file = Path(args.data_dir) / f"ICD9_umls{UMLS_VERSION.lower()}"
    output_file = Path(args.data_dir) / f"ICD9_umls{UMLS_VERSION.lower()}_descriptions"
    error_file = Path(args.data_dir) / f"ICD9_umls{UMLS_VERSION.lower()}_nonexisting"

    icd9_to_info = dict()

    if args.base_data_dir:
        old_icd9_desc_file = Path(args.base_data_dir) / "ICD9_descriptions"
        old_umls2020_input_file = Path(args.base_data_dir) / f"ICD9_umls{UMLS_VERSION.lower()}"
        extended_out_desc_file = Path(args.base_data_dir) / "ICD9_descriptions_extended"
        extended_output_file = Path(args.base_data_dir) / f"ICD9_umls{UMLS_VERSION.lower()}_extended"

        base_icd9_to_info = dict()
        base_icd9_to_umls2020 = dict()

        if old_icd9_desc_file.exists():
            read_icd9 = pd.read_csv(old_icd9_desc_file, header=None, sep='\t')
            for icd9, desc in zip(read_icd9[0], read_icd9[1]):
                base_icd9_to_info[icd9] = desc
        if old_umls2020_input_file.exists():
            read_icd9 = pd.read_csv(old_umls2020_input_file, header=None, sep='\t')
            for icd9, *items in zip(read_icd9[0], read_icd9[1], read_icd9[2], read_icd9[3]):
                base_icd9_to_umls2020[icd9] = items


    # existing codes from UMLS queries
    icd9_to_desc = dict()
    if umls2020_input_file.exists():
        queried = pd.read_csv(umls2020_input_file, header=None, sep='\t')
        queried_icd9 = {code for code in set(queried[0].unique())}
    else:
        queried_icd9 = set()

    logger.info(f"# Codes queried before with found UMLS info: {len(queried_icd9)}")

    if base_input_file.exists():
        base_icd9 = pd.read_csv(base_input_file, header=None, sep='\t')
        base_icd9 = {code for code in set(base_icd9[0].unique())}
    else:
        base_icd9 = set()

    code_not_found = base_icd9.difference(queried_icd9)

    logger.info(f"# Codes in base file before NOT found in UMLS: {len(code_not_found)}")
    if code_not_found:
        if (error_file.exists() and args.overwrite) or not error_file.exists():
            with open(error_file, 'w' if args.overwrite else 'a') as ewf:
                for icd9_code in code_not_found:
                    ewf.write(f"{icd9_code}\tN/A\n")
            logger.info(f"Unknown ICD9 codes written to: {error_file}")

    if (output_file.exists() and args.overwrite) or not output_file.exists():
        with open(output_file, 'w' if args.overwrite else 'a') as wf:
            for icd9_code, cui, tui, desc in zip(queried[0], queried[1], queried[2], queried[3]):
                icd9_to_info[icd9_code] = [cui, tui, desc]
                wf.write(f"{icd9_code}\t{desc}\n")
                if args.base_data_dir:
                    try:
                        base_icd9_to_info.update({icd9_code: desc})
                        base_icd9_to_umls2020.update({icd9_code: [cui, tui, desc]})
                    except NameError:
                        print("Something went wrong reading old icd9 description and UMLS files!!!")
                        continue
        logger.info(f"Queried ICD9 codes with description updates written to: {output_file}")

    # extending existing ICD9_descriptions file (sorted by code)
    if args.base_data_dir:
        with open(extended_out_desc_file, 'a') as wdf:
            for icd9_code in sorted(base_icd9_to_info.keys()):
                wdf.write(f"{icd9_code}\t{base_icd9_to_info.get(icd9_code)}\n")
        # extending ICD9_umls2020aa file (sorted by code)
        with open(extended_output_file, 'a') as wof:
            for icd9_code in sorted(base_icd9_to_umls2020.keys()):
                cui, tui, desc = base_icd9_to_umls2020.get(icd9_code)
                wof.write(f"{icd9_code}\t{cui}\t{tui}\t{desc}\n")
        logger.info(f"Updated/Extended files written to: {extended_out_desc_file} and {extended_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", action="store", required=True, type=str,
        help="Path to data directory containing ICD9_descriptions file"
    )
    parser.add_argument(
        "--base_data_dir", action="store", required=False, type=str,
        help="Path to old/base data directory containing old/base ICD9_descriptions file"
    )
    parser.add_argument(
        "--overwrite", action="store_true", required=False,
        help="Overwrite/append to existing file(s) if there, otherwise skip action over existing file"
    )

    args = parser.parse_args()

    import pprint

    pprint.pprint(vars(args))

    main(args)
