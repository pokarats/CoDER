#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Link UMLS CUIs entity to doc text via SciSpacy.


@author: Saadullah Amin
"""

import spacy
import os
import pandas as pd
import numpy as np
import time
import logging
import warnings
import itertools
import json
import argparse

from pathlib import Path
from scispacy.umls_linking import UmlsEntityLinker
from scispacy.abbreviation import AbbreviationDetector


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__file__)


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def load_cui_tagger(scispacy_model_name="en_core_sci_lg", cache_dir=None):
    """
    Args:
        scispacy_model_name : (str), scispacy model name.
        
        cache_dir : (str), SCISPACY_CACHE directory.
    
    """
    if cache_dir is not None:
        os.environ['SCISPACY_CACHE'] = cache_dir
    nlp = spacy.load(scispacy_model_name)
    
    # We suggest to not add abb. detector to have more textual information
    # logger.info('Loadding and adding ``AbbreviationDetector`` to ``nlp.pipe`` ...')
    # nlp.add_pipe('abbreviation_detector')
    
    # We use the defaults set in scispacy see 
    # https://github.com/allenai/scispacy/blob/main/scispacy/linking.py#L67
    logger.info('Loadding and adding ``UmlsEntityLinker`` to ``nlp.pipe`` ...')
    nlp.add_pipe('scispacy_linker')
    
    return nlp


def iter_doc_sents(text, doc_id):
    """
    Args:
        text : (str), preprocessed text.
    
    """
    sents = text.split(' [SEP] [CLS] ')
    # fix boundary sentences
    sents[0] = sents[0].lstrip('[CLS] ')
    sents[-1] = sents[-1].rstrip(' [SEP]')
    for sent_id, sent in enumerate(sents):
        yield f'{doc_id}_{sent_id}', sent


def process_tagged_doc(doc):
    """
        Args
            doc : (spacy Doc), scispacy tagged doc.
    
    """
    return [
        {
            's': ent.start_char, 'e': ent.end_char,
            'umls_ents': ent._.umls_ents
        } for ent in doc.ents if ent._.umls_ents
    ]


def count_total(csv_file):
    """
    Args:
        csv_file : (str), processed csv file
    
    """
    df = pd.read_csv(csv_file)
    return sum([len(list(iter_doc_sents(df.iloc[i]['TEXT'], i))) for i in range(len(df))])


def iter_sents_from_csv_docs(csv_file):
    """
    Args:
        csv_file : (str), processed csv file
    
    """
    df = pd.read_csv(csv_file)
    docs = [df.iloc[i]['TEXT'] for i in range(len(df))]
    for doc_id, doc in enumerate(docs):
        yield from iter_doc_sents(doc, doc_id)


def main(args):
    nlp = load_cui_tagger(args.scispacy_model_name, args.cache_dir)
    fname = Path(args.mimic3_dir) / f'{args.split_file}.csv'
    
    # this trick is attributed to cf. 
    # https://github.com/explosion/spaCy/issues/172#issuecomment-183963403
    gen1, gen2 = itertools.tee(iter_sents_from_csv_docs(fname))
    uids = (uid for (uid, text) in gen1)
    texts = (text for (uid, text) in gen2)
    
    total = count_total(fname)
    logger.info(f'Total number of sentences: {total}')
    
    idx = 0
    jsonls = list()
    
    t = time.time()
    for uid, sent in zip(uids, nlp.pipe(texts, n_process=48, batch_size=4096)):
        if idx % 1000 == 0 and idx > 0:
            speed = idx // ((time.time() - t) / 60)
            logger.info(f'Processed {idx} / {total} sents @ {speed} sents/min ...')
        jsonls.append({f'{uid}': process_tagged_doc(sent)})
        idx += 1
    t = (time.time() - t) // 60
    
    logger.info(f'Took {t} minutes !')
    output_file = Path(args.mimic3_dir) / f'{args.split_file}_umls.txt'
    logger.info(f'Saving output to `{output_file}` ...')
    
    with open(output_file, 'w') as wf:
        for line in jsonls:
            wf.write(json.dumps(line) + '\n')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--mimic3_dir", action="store", required=True, type=str,
        help="Path to MIMIC-III data directory containing processed versions "
        "of the top-50 and full train/dev/test splits."
    )
    parser.add_argument(
        "--split_file", action="store", required=True, type=str,
        choices=[
            "train_full", "dev_full", "test_full",
            "train_50", "dev_50", "test_50"
        ],
        help="Path to data split file."
    )
    parser.add_argument(
        "--scispacy_model_name", action="store", type=str, default="en_core_sci_lg",
        help="SciSpacy model to use for UMLS concept linking."
    )
    parser.add_argument(
        "--cache_dir", action="store", type=str, default="/scratch/cache/scispacy",
        help="Path to SciSpacy cache directory. Optionally, set the environment "
        "variable ``SCISPACY_CACHE``."
    )
    parser.add_argument(
        "--n_process", action="store", type=int, default=48,
        help="Number of processes to run in parallel with spaCy multi-processing."
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, default=4096,
        help="Batch size to use in combination with spaCy multi-processing."
    )
    
    args = parser.parse_args()
    
    import pprint
    pprint.pprint(vars(args))
    
    main(args)
