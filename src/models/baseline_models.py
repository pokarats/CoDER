#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Python template with an argument parser and logger. Put all the "main" logic into the method called "main".
             Only use the true "__main__" section to add script arguments. The logger writes to a hidden folder './log/'
             and uses the name of this file, followed by the date (by default). The argument parser comes with a default
             option --quiet to keep the stdout clean.

@copyright: Copyright 2018 Deutsches Forschungszentrum fuer Kuenstliche
            Intelligenz GmbH or its licensors, as applicable.

@author: Noon Pokaratsiri Goldstein, sklearn pipeline adapted from code from Albers Uzila
(https://towardsdatascience.com/multilabel-text-classification-done-right-using-scikit-learn-and-stacked-generalization-f5df2defc3b5)
"""

import sys
import time
from datetime import date
import logging
import argparse
import traceback

import pickle

import pandas as pd
import spacy
from operator import itemgetter
from src.utils.concepts_pruning import ConceptCorpusReader
from src.utils.utils import token_to_token
from src.utils.eval import simple_score, all_metrics
from scispacy.umls_linking import UmlsEntityLinker
from pathlib import Path
from tqdm import tqdm

from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid

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

logger = logging.getLogger(__name__)


class ClfSwitcher(BaseEstimator):
    """
     A Custom BaseEstimator that can switch between classifiers.
    """
    def __init__(self, estimator=MultiOutputClassifier(LinearSVC())):
        """

        :param estimator: the classifier
        :type estimator: sklearn object
        """

        self.estimator = estimator

    def fit(self, train, train_labels):
        self.estimator.fit(train, train_labels)
        return self

    def predict(self, train):
        return self.estimator.predict(train)

    def predict_proba(self, train):
        return self.estimator.predict_proba(train)

    def score(self, train, train_labels):
        return self.estimator.score(train, train_labels)


class TFIDFBasedClassifier:
    def __init__(self, cl_arg):
        self.seed = cl_arg.seed
        self.stacked_clf = MultiOutputClassifier(
            StackingClassifier([('logreg2', LogisticRegression(class_weight='balanced',
                                                               random_state=self.seed)),
                                ('sgd2', SGDClassifier(class_weight='balanced',
                                                       random_state=self.seed,
                                                       loss='modified_huber')),
                                ('svm2', LinearSVC(class_weight='balanced',
                                                   random_state=self.seed))]), n_jobs=-1)
        self.pipeline = Pipeline([('tfidf', TfidfVectorizer()),
                                  ('clf', ClfSwitcher())])
        self.stack_pipeline = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',
                                                                  tokenizer=token_to_token,
                                                                  preprocessor=token_to_token,
                                                                  ngram_range=(1, 2),
                                                                  pattern=None)),
                                        ('stack', self.stacked_clf)])
        self.grid = ParameterGrid({'clf__estimator': (
            MultiOutputClassifier(LogisticRegression(class_weight='balanced',
                                                     random_state=self.seed), n_jobs=-1),
            MultiOutputClassifier(SGDClassifier(class_weight='balanced',
                                                random_state=self.seed,
                                                loss='modified_huber'), n_jobs=-1),
            MultiOutputClassifier(LinearSVC(class_weight='balanced', random_state=self.seed), n_jobs=-1)),
            'tfidf__ngram_range': ((1, 1), (1, 2)),
            'tfidf__analyzer': ('word',),
            'tfidf__tokenizer': (token_to_token,),
            'tfidf__preprocessor': (token_to_token,),
            'tfidf__pattern': (None,)})
        self.models = ['logreg1', 'logreg2', 'sgd1', 'sgd2', 'svm1', 'svm2']
        self.eval_scores = pd.DataFrame()
        self.eval_metrics = []

    def execute_pipeline(self, x_train, y_train, x_val, y_val):
        logger.info(f"Executing sklearn tfidf pipeline for LogisticRegression, SGDClassifier, and SGDClassifier...")
        for model, params in tqdm(zip(self.models, self.grid), total=len(self.models), desc=f"training pipeline"):
            self.pipeline.set_params(**params)
            self.pipeline.fit(x_train, y_train)
            y_pred = self.pipeline.predict(x_val)
            self.eval_metrics.append(all_metrics(y_pred, y_val, k=[1, 3, 5], yhat_raw=None, calc_auc=False))
            tfidf_score = simple_score(y_pred, y_val, model)
            self.eval_scores = pd.concat([self.eval_scores, tfidf_score])

    def execute_stack_pipeline(self, x_train, y_train, x_val, y_val):
        logger.info(f"Executing sklearn tfidf pipeline for stacked classifier")
        self.stack_pipeline.fit(x_train, y_train)
        stack_pred = self.stack_pipeline.predict(x_val)
        self.eval_metrics.append(all_metrics(stack_pred, y_val, k=[1, 3, 5], yhat_raw=None, calc_auc=False))
        stack_model_score = simple_score(stack_pred, y_val, 'stack_model')
        self.eval_scores = pd.concat([self.eval_scores, stack_model_score])


class RuleBasedClassifier:
    """
    Classify an input sample according to possible cuis that correspond to ICD9 labels
    """

    def __init__(self, cl_arg, version, extension=None):

        """

        :param mimic3_dir: directory where mimic3 data files are
        :param split: dev, test, or train partition
        :param version: full vs 50
        :param threshold: confidence level threshold to include concept
        """

        self.icd9_umls_fname = Path(cl_arg.data_dir) / 'ICD9_umls2020aa'
        self.cui_discard_set_pfile = Path(cl_arg.data_dir) / 'linked_data' / version / f'{version}_cuis_to_discard.pickle'
        self.cui_to_discard = None
        self.cui_to_icd9 = dict()
        self.icd9_to_cui = dict()
        self.tui_icd9_to_desc = dict()  # dict[tui][icd9] = desc
        self.extension = extension
        self.nlp = spacy.load(cl_arg.scispacy_model_name) if self.extension else None
        self.linker = UmlsEntityLinker(name=cl_arg.linker_name) if self.extension else None
        self.min_num_labels = cl_arg.min_num_labels

        self._load_cuis_to_discard()
        self._load_icd9_mappings()

    def _load_cuis_to_discard(self):
        logger.info(f"Loading cuis to discard from pickle fie: {self.cui_discard_set_pfile}...")
        with open(self.cui_discard_set_pfile, 'rb') as handle:
            self.cui_to_discard = pickle.load(handle)

    def _load_icd9_mappings(self):
        logger.info(f"Creating cui icd9 mapping from file: {self.icd9_umls_fname}...")
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
        logger.info(f"Getting all icd9 codes related to {cui}")
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
        logger.debug(f"Getting most similar icd9 from {cui}")
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
        if self.extension and len(icd9_labels) < self.min_num_labels:
            additional_icd9 = set()
            if self.extension == "all":
                # add all icd9 codes corresponding to the TUI of the cui that doesn't correspond to an icd9 code
                for cui in pruned_input_cuis:
                    if self.cui_to_icd9.get(cui) is None:
                        additional_icd9.update(self._get_all_icd9_from_cui(cui))

            elif self.extension == "best":
                # add icd9 code whose description is most similar to the cui without a corresponding icd9 code
                for cui in tqdm(pruned_input_cuis, total=len(pruned_input_cuis), desc=f"pruned input cuis"):
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
    rule_based_model = RuleBasedClassifier(cl_args, cl_args.version, cl_args.extension)
    corpus_reader = ConceptCorpusReader(cl_args.mimic3_dir, cl_args.split, "1")
    corpus_reader.read_umls_file()

    test_sample_0 = corpus_reader.docidx_to_ordered_concepts[0]
    predicted_icd9 = rule_based_model.fit(test_sample_0)

    logger.info(f"Predicted icd9 for test sample: {predicted_icd9}")

    lapsed_time = (time.time() - start_time) // 60
    logger.info(f"Module took {lapsed_time} minutes!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", action="store", type=str, default="data",
        help="Path to data directory containing both the ICD9_umls2020aa file and the pickle file from concept_pruning"
    )
    parser.add_argument(
        "--mimic3_dir", action="store", type=str, default="data/linked_data/1",
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
        "--min_num_labels", action="store", type=int, default=5,
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
    proj_folder = Path(__file__).resolve().parent.parent.parent
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

