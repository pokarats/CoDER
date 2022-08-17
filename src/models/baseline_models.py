#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
from datetime import date
import logging
import argparse
import traceback

import pickle
import spacy
from operator import itemgetter
from src.utils.concepts_pruning import ConceptCorpusReader
from scispacy.umls_linking import UmlsEntityLinker
from pathlib import Path
from tqdm import tqdm

"""
rule-base model
input: doc as concepts
process: concepts in each sample  that are also in the icd9 concepts == labels
1. for each sample input, make a set of all cuis
2. remove cuis that do not have icd9 equivalent --> need a set of icd9 cuis or icd9-cui and cui-icd9 dict
3. dict of doc indx (0, 1, 2 etc.) : list of cuis
need to encode all possible labels, i.e. icd9 codes present in the test partition
output: labels

tfidf + classifier
input: vectorized concepts found in each doc -->tfidf input features
linear classifier output to |unique labels in semantic_types_mimic|
output: labels

label cluster (optional add to baseline) by
1) semantic types
2)
"""

class RuleBasedClassifier:
    """
    Classify an input sample according to possible cuis that correspond to ICD9 labels
    """

    def __init__(self, data_dir, version, extension=None):

        """

        :param mimic3_dir: directory where mimic3 data files are
        :param split: dev, test, or train partition
        :param version: full vs 50
        :param threshold: confidence level threshold to include concept
        """

        self.icd9_umls_fname = Path(data_dir) / 'ICD9_umls2020aa'
        self.cui_discard_set_pfile = Path(data_dir) / 'linked_data' / 'top50' / f'{version}_cuis_to_discard.pickle'
        self.cui_to_discard = None
        self.cui_to_icd9 = dict()
        self.icd9_to_cui = dict()
        self.tui_icd9_to_desc = dict()  # dict[tui][icd9] = desc
        self.extension = extension
        self.nlp = spacy.load("en_core_sci_lg") if self.extension is not None else None
        self.linker = UmlsEntityLinker(name="scispacy_linker") if self.extension is not None else None

        self._load_cuis_to_discard()
        self._load_icd9_mappings()


    def _load_cuis_to_discard(self):
        with open(self.cui_discard_set_pfile, 'rb') as handle:
            self.cui_to_discard = pickle.load(handle)

    def _load_icd9_mappings(self):
        with open(self.icd9_umls_fname) as rfname:
            for line in tqdm(rfname):
                line = line.strip()
                if not line:
                    continue
                # ICD9_umls2020aa is \t separated
                try:
                    icd9, cui, tui, desc = line.split('\t')
                except ValueError:
                    print(f"icd9 code in {line} missing tui and desc")
                    ic9, cui = line.split('\t')
                    tui = ""
                    desc = ""
                if tui not in self.tui_icd9_to_desc:
                    self.tui_icd9_to_desc[tui] = dict()
                self.icd9_to_cui[icd9] = cui
                self.cui_to_icd9[cui] = icd9
                self.tui_icd9_to_desc[tui][icd9] = desc


    def _get_similarity_score(self, target_sent, candidate_sent):
        if self.extension:
            if not isinstance(target_sent, spacy.tokens.doc.Doc):
                target_sent = self.nlp(target_sent)
            if not isinstance(candidate_sent, spacy.tokens.doc.Doc):
                candidate_sent = self.nlp(candidate_sent)
            return target_sent.similarity(candidate_sent)
        else:
            print(f"Not possible, no extension option, pass!")
            return 0.0

    def _get_all_icd9_from_cui(self, cui):
        tuis = self.linker.kb.cui_to_entity[cui].types
        icd9_codes = set()
        if len(tuis) < 1:
            return icd9_codes
        for tui in tuis:
            try:
                icd9_codes.update(self.tui_icd9_to_desc[tui].keys())
            except KeyError:
                print(f"{tui} is NOT an ICD9 TUI!!! Should not be possible after pruning!!!")
                continue
        return icd9_codes

    def _get_most_similar_icd9_from_cui(self, cui, similarity_threshold):
        _, _, definitions, tuis, *_ = self.linker.kb.cui_to_entity[cui]
        icd9_codes = []
        max_sim_score = similarity_threshold
        if len(tuis) < 1:
            return icd9_codes
        for tui, definition in zip(tuis, definitions):
            try:
                for icd9 in self.tui_icd9_to_desc[tui].keys():
                    score = self._get_similarity_score(definition, self.tui_icd9_to_desc[tui][icd9])
                    if score < max_sim_score:
                        continue
                    # only add icd9 whose similarity score is higher than the previously added icd9 codes
                    max_sim_score = score
                    icd9_codes.append((icd9, score))
            except KeyError:
                print(f"{tui} is NOT an ICD9 TUI!!! Should not be possible after pruning!!!")
                continue

        # choose 1 icd9 codes with highest similarity score, unless more multiple icd9 share the same score
        sorted_icd9 = sorted(icd9_codes, key=itemgetter(1))
        if len(sorted_icd9) > 0 and len(sorted_icd9) < 2:
            return [sorted_icd9[0][0]]
        elif len(sorted_icd9) == 0:
            return []
        else:
            top_icd9s = []
            for idx in range(len(sorted_icd9) - 1):
                if sorted_icd9[idx][1] == sorted_icd9[idx + 1][1]:
                    top_icd9s.append(sorted_icd9[idx][0])
                    top_icd9s.append(sorted_icd9[idx + 1][0])
                else:
                    break
            return top_icd9s

    def fit(self, input_sample, similarity_threshold=0.4):
        if not isinstance(input_sample, set):
            input_sample = set(input_sample)
        if not self.cui_to_discard:
            self._load_cuis_to_discard()

        pruned_input_cuis = input_sample.difference(self.cui_to_discard)
        icd9_labels = set()
        icd9_labels.update({self.cui_to_icd9.get(cui) for cui in pruned_input_cuis
                            if self.cui_to_icd9.get(cui) is not None})
        if self.extension:
            additional_icd9 = set()
            if self.extension == "all":
                # add all icd9 codes corresponding to the TUI of the cui that doesn't correspond to an icd9 code
                for cui in pruned_input_cuis:
                    if self.cui_to_icd9.get(cui) is None:
                        additional_icd9.update(self._get_all_icd9_from_cui(cui))

            elif self.extension == "best":
                # add icd9 code whose description is most similar to the cui without a corresponding icd9 code
                for cui in pruned_input_cuis:
                    if self.cui_to_icd9.get(cui) is None:
                        additional_icd9.update(self._get_most_similar_icd9_from_cui(cui, similarity_threshold))
            else:
                print(f"Invalid option! Do Nothing!")
                pass

            icd9_labels.update(additional_icd9)

        return icd9_labels

def main(cl_args):
    """Main loop"""
    start_time = time.time()

    logger.info(f"Initialize RuleBasedClassifier and ConceptCorpusReader...")
    rule_based_model = RuleBasedClassifier(cl_args.data_dir, cl_args.version, cl_args.extension)
    corpus_reader = ConceptCorpusReader(cl_args.mimic3_dir, cl_args.split, "1")
    corpus_reader.read_umls_file()

    test_sample_0 = corpus_reader.docidx_to_concepts_simple[0]
    predicted_icd9 = rule_based_model.fit(test_sample_0)

    logger.info(f"Predicted icd9 for test sample: {predicted_icd9}")

    lapsed_time = (time.time() - start_time) // 60
    logger.info(f"Module took {lapsed_time} minutes!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", action="store", type=str, default="../../data",
        help="Path to data directory containing both the ICD9_umls2020aa file and the pickle file from concept_pruning"
    )
    parser.add_argument(
        "--mimic3_dir", action="store", type=str, default="../../data/linked_data/1",
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
        "--extension", action="store", type=str, default="best",
        help="Extension type for when cui not matching any icd9, options: best or all"
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
        "--min", action="store", type=int, default=0.4,
        help="Min threshold for similarity"
    )
    parser.add_argument(
        "--cache_dir", action="store", type=str,
        default="/Users/noonscape/Documents/msc_thesis/projects/CoDER/scratch/scispacy",
        help="Path to SciSpacy cache directory. Optionally, set the environment "
             "variable ``SCISPACY_CACHE``."
    )
    parser.add_argument(
        "--dict_pickle_file", action="store", type=str, default="model_output_dict",
        help="Path to pickle file for dict mapping sample idx to output"
    )
    parser.add_argument(
        "--misc_pickle_file", action="store", type=str, default="misc_pickle",
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
    log_folder = Path(f"../../scratch/.log/{date.today():%y_%m_%d}")
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

