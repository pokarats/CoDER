#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Python template with an argument parser and logger. Put all the "main" logic into the method called "main".
             Only use the true "__main__" section to add script arguments. The logger writes to a hidden folder './log/'
             and uses the name of this file, followed by the date (by default).


@copyright: Copyright 2018 Deutsches Forschungszentrum fuer Kuenstliche
            Intelligenz GmbH or its licensors, as applicable.

@author: Noon Pokaratsiri Goldstein; this is a modification from the code base obtained from:

https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network/blob/master/preprocess_mimic3.py
https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network/blob/master/utils.py
and,
https://github.com/suamin/P4Q_Guttmann_SCT_Coding/blob/main/word2vec.py

"""

import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from datetime import date
import logging
import json
import sys
import os
import platform
from pathlib import Path

from sacred import Experiment
from sacred.observers import FileStorageObserver

if platform.system() != 'Darwin':
    sys.path.append(os.getcwd())  # only needed for slurm
from corpus_readers import ProcessedIter, MimicIter, MimicCuiIter


PROJ_FOLDER = Path(__file__).resolve().parent.parent.parent
SAVED_FOLDER = PROJ_FOLDER / f"scratch/.log/{date.today():%y_%m_%d}"


# creating sacred experiment
ex = Experiment()
ex.observers.append(FileStorageObserver(SAVED_FOLDER))


@ex.capture
def gensim_to_npy(w2v_model_file, _log, normed=False, outfile=None, embedding_dim=100):

    # gensim can load from PosixPath!!
    if Path(w2v_model_file).suffix == '.wordvectors':
        wv = KeyedVectors.load(str(w2v_model_file), mmap='r')
    else:
        loaded_model = Word2Vec.load(str(w2v_model_file))
        wv = loaded_model.wv  # this is just the KeyedVectors parts
        # free up memory
        del loaded_model

    assert embedding_dim == wv.vector_size, f"specified embedding_dim ({embedding_dim}) and " \
                                            f"loaded model vector_size ({wv.vector_size}) mismatch!"

    word2id = {"<PAD>": 0, "<UNK>": 1}
    mat = np.zeros((len(wv.key_to_index.keys()) + len(word2id), embedding_dim))
    # initialize UNK embedding with random normal
    mat[1] = np.random.randn(100)

    _log.info(f"re_ordering word2id based on adding PAD and UNK tokens...")
    for word in sorted(wv.key_to_index.keys()):
        vector = wv.get_vector(word, norm=normed)
        mat[len(word2id)] = vector
        word2id[word] = len(word2id)

    if outfile is None:
        outfile = Path(w2v_model_file).stem

    output_dir = Path(w2v_model_file).parent
    mat_fname = output_dir / f'{outfile}.npy'
    map_fname = output_dir / f'{outfile}.json'

    _log.info(f'Saving word2id at {map_fname} and numpy matrix at {mat_fname} ...')

    np.save(str(mat_fname), mat)

    with open(map_fname, 'w', encoding='utf-8', errors='ignore') as wf:
        json.dump(word2id, wf)

    return map_fname, map_fname


@ex.capture
def word_embeddings(dataset_vers,
                    notes_file,
                    _log,
                    embedding_size=100,
                    min_count=0,
                    n_iter=5,
                    n_workers=4,
                    save_wv_only=False,
                    data_iterator=ProcessedIter):
    """
    Updated for Gensim >= 4.0, train and save Word2Vec Model

    :param _log: logging for Sacred Experiment
    :type _log: logging
    :param save_wv_only: True if saving only the lightweight KeyedVectors object of the trained model
    :type save_wv_only: bool
    :param dataset_vers: full or 50
    :type dataset_vers: str
    :param notes_file: corpus file path
    :type notes_file: Path or str
    :param embedding_size: vector_size
    :type embedding_size: int
    :param min_count: min frequency
    :type min_count: int
    :param n_iter: how many epochs to train
    :type n_iter: int
    :param n_workers: how many processes/cpu's to use
    :type n_workers: int
    :param data_iterator: type of corpus iterator to use, depending on corpus file: ProcessedIter, CorpusIter
    :type data_iterator: initialized class with __iter__
    :return: path to saved model file

    """

    model_name = f"processed_{Path(notes_file).stem}.model"
    sentences = data_iterator

    model = Word2Vec(vector_size=embedding_size, min_count=min_count, epochs=n_iter, workers=n_workers)

    _log.info(f"building word2vec vocab on {notes_file}...")
    model.build_vocab(sentences)

    _log.info(f"training on {model.corpus_count} sentences over {model.epochs} iterations...")
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

    embedding_dir = Path(notes_file).parent / f"{model_name.split('.')[-1]}"
    if not embedding_dir.exists():
        embedding_dir.mkdir(parents=True, exist_ok=False)

    out_file = embedding_dir / model_name
    _log.info(f"writing embeddings to {out_file}")

    if save_wv_only:
        out_file = embedding_dir / f"{out_file.stem}.wordvectors"
        _log.info(f"only KeyedVectors are saved to {out_file}!! This is no longer trainable!!")
        model_wv = model.wv
        model_wv.save(out_file)
        return out_file

    model.save(str(out_file))
    return out_file


@ex.config
def default_cfg():
    """Default Configs for Processing MIMIC-III text"""

    # data directory and file organization
    data_dir = PROJ_FOLDER / "data"
    version = '50'

    mimic_dir = Path(data_dir) / "mimic3" / version

    # corpus_readers Iterator params
    iter_params = {"filename": str(mimic_dir / f"train_{version}.csv"),
                   "slice_pos": 2,
                   "threshold": 0.7,
                   "prune": False,
                   "discard_cuis_file": None}

    # gensim.models.Word2Vec params, captured func
    dataset_vers = version
    notes_file = iter_params["filename"]
    embedding_size = 100
    min_count = 0
    n_iter = 5
    n_workers = 4
    save_wv_only = False
    data_iterator = MimicIter(notes_file)
    normed = False  # whether to also save L2-normed vectors as embeddings


@ex.named_config
def cui():
    """Baseline configs for Embedding CUIs as input tokens"""

    # data directory and file organization
    data_dir = PROJ_FOLDER / "data"
    version = '50'

    mimic_dir = Path(data_dir) / "linked_data" / version

    # corpus_readers Iterator params
    iter_params = {"filename": str(mimic_dir / f"train_{version}_umls.txt"),
                   "slice_pos": 2,
                   "threshold": 0.7}

    # gensim.models.Word2Vec params
    notes_file = notes_file = iter_params["filename"]
    data_iterator = MimicCuiIter(notes_file)


@ex.named_config
def cui_pruned():
    """Configs for Pruned Version of Embedding CUIs"""

    # data directory and file organization
    data_dir = PROJ_FOLDER / "data"
    version = '50'
   
    mimic_dir = Path(data_dir) / "linked_data" / version

    # corpus_readers Iterator params
    iter_params = {"filename": str(mimic_dir / f"train_{version}_umls.txt"),
                   "slice_pos": 2,
                   "threshold": 0.7,
                   "prune": True,
                   "discard_cuis_file": str(mimic_dir / f"{version}_cuis_to_discard.pickle")}

    # gensim.models.Word2Vec params
    notes_file = iter_params["filename"]
    data_iterator = MimicCuiIter(notes_file,
                                 iter_params["threshold"],
                                 iter_params["prune"],
                                 iter_params["discard_cuis_file"])


@ex.main
def run_word_embeddings(normed, _log, _run):
    _log.info(f"\n=========START W2V EMBEDDING TRAINING==========\n")

    w2v_file = word_embeddings()
    npy_fp, mapping_fp = gensim_to_npy(w2v_file)

    if normed:
        _log.info(f"saving normed vectors as well...")
        out_fname = f"{w2v_file.stem}_normed"
        npy_fp_norm, mapping_fp_norm = gensim_to_npy(w2v_file, normed=normed, outfile=out_fname)

        # log model files to Sacred
        ex.add_artifact(filename=f"{npy_fp_norm}")
        ex.add_artifact(filename=f"{mapping_fp_norm}")

    # Log model files to Sacred
    ex.add_artifact(filename=f"{w2v_file}")
    ex.add_artifact(filename=f"{npy_fp}")
    ex.add_artifact(filename=f"{mapping_fp}")
    _log.info(f"\n=========FINISHED W2V EMBEDDING TRAINING==========\n")


if __name__ == "__main__":
    ex.run_commandline()
