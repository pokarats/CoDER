# -*- coding: utf-8 -*-

import requests
import json
import pandas as pd
import argparse
import logging
import time
import sys

from tqdm import tqdm
from pathlib import Path
from authentication import *
from src.utils.config import DEV_UMLS_API_KEY

API_KEY = DEV_UMLS_API_KEY  # place your own UMLS API KEY in src/configs/.env
UMLS_VERSION = "2020AA"
BASE_URL = f"https://uts-ws.nlm.nih.gov/subsets/{UMLS_VERSION}/source/ICD9CM"


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    stream=sys.stdout, 
    level=logging.INFO
)
logger = logging.getLogger(__file__)


def main(args):
    input_file = Path(args.data_dir) / "ICD9_descriptions"
    output_file = Path(args.data_dir) / f"ICD9_umls{UMLS_VERSION.lower()}"
    
    # UMLS returns 401 code after some thousand queries,
    # this basically helps to rerun a job with making
    # existing codes' queries again
    if output_file.exists():
        queried = pd.read_csv(output_file, header=None, sep='\t')
        queried = {code for code in set(queried[0].unique())}
    else:
        queried = set()
    
    logger.info(f"  Codes queried before {len(queried)}")
    
    icd9_mimic3 = pd.read_csv(input_file, header=None, sep='\t')
    icd9_codes = {
        code for code in set(icd9_mimic3[0].unique()) 
        if (code != "@" or "-" not in code) and code not in queried
    }
    
    logger.info(f"  Found {len(icd9_codes)} ICD9 codes from MIMIC3  ")
    
    auth_client = Authentication(API_KEY)
    tgt = auth_client.getTgt()
    
    t = time.time()
    code2item = dict()
    
    with open(output_file, 'a') as wf:
        for idx, code in enumerate(icd9_codes):
            r = requests.get(f"{BASE_URL}/{code}", params={'ticket': auth_client.getServiceTicket(tgt)})
            logger.info(f"... {idx + 1} / {len(icd9_codes)} : query = {code} | status = {r.status_code}")
            if r.status_code == 200:
                items  = json.loads(r.text)
                cui = items['result']['ui']
                logger.info(f"    ... response = {cui}")
                # now query type
                r = requests.get(
                    f"https://uts-ws.nlm.nih.gov/rest/content/{UMLS_VERSION}/CUI/{cui}", 
                    params={'ticket': auth_client.getServiceTicket(tgt)}
                )
                logger.info(f"... query = {cui} | status = {r.status_code}")
                if r.status_code == 200:
                    items  = json.loads(r.text)
                    tuis = [item['uri'].split('/')[-1] for item in items['result']['semanticTypes']]
                    logger.info(f"    ... response = {tuis}")
                    name = items['result']['name']
                else:
                    tuis = []
                    name = ""
                code2item[code] = (cui, ",".join(tuis), name)
                cui, tuis, name = code2item[code]
                wf.write(f"{code}\t{cui}\t{tuis}\t{name}\n")
            else:
                logger.info(f"... ERROR")
    
    t = (time.time() - t) // 60
    
    logger.info(f"Took {t} minutes !")
    logger.info(f"Saving output to `{output_file}` ...")
    
    # with open(output_file, 'a') as wf:
    #     for code in sorted(code2item.keys()):
    #         cui, tuis, name = code2item[code]
    #         wf.write(f"{code}\t{cui}\t{tuis}\t{name}\n")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_dir", action="store", required=True, type=str,
        help="Path to data directory containing."
    )
    
    args = parser.parse_args()
    
    import pprint
    pprint.pprint(vars(args))
    
    main(args)
