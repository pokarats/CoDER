#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Train and save Word2Vec or FastText model. Command line options, logging, and configurations are through
the Sacred library

@author: Noon Pokaratsiri Goldstein; this is a modification from the code base obtained from:

https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network/blob/master/preprocess_mimic3.py
https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network/blob/master/utils.py
and,
https://github.com/suamin/P4Q_Guttmann_SCT_Coding/blob/main/word2vec.py

"""
import itertools
import numpy as np
from gensim.models import Word2Vec, KeyedVectors, FastText
from datetime import date
import logging
import json
import sys
import os
import platform
from pathlib import Path

from sacred import Experiment
from sacred.observers import FileStorageObserver

if platform.python_version() < "3.8":
    import pickle5 as pickle
else:
    import pickle

if platform.system() != 'Darwin':
    sys.path.append(os.getcwd())  # only needed for slurm
from corpus_readers import ProcessedIter, MimicIter, MimicCuiIter
from utils import whole_lines_from_file

PROJ_FOLDER = Path(__file__).resolve().parent.parent.parent
SAVED_FOLDER = PROJ_FOLDER / f"scratch/.log/{date.today():%y_%m_%d}/{Path(__file__).stem}"


# creating sacred experiment
ex = Experiment()
ex.observers.append(FileStorageObserver(SAVED_FOLDER))


@ex.capture
def gensim_to_npy(w2v_model_file, _log, prune=False, prune_file=None, normed=False, embedding_dim=100, outfile=None):

    # gensim can load from PosixPath!!
    if Path(w2v_model_file).suffix == '.wordvectors':
        wv = KeyedVectors.load(str(w2v_model_file), mmap='r')
    else:
        if "fastText" in str(w2v_model_file):
            loaded_model = FastText.load(str(w2v_model_file))
        else:
            loaded_model = Word2Vec.load(str(w2v_model_file))

        wv = loaded_model.wv  # this is just the KeyedVectors parts
        # free up memory
        del loaded_model

    assert embedding_dim == wv.vector_size, f"specified embedding_dim ({embedding_dim}) and " \
                                            f"loaded model vector_size ({wv.vector_size}) mismatch!"

    if prune and prune_file is not None:
        wv_vocab = set(wv.key_to_index.keys())
        if "text" in str(w2v_model_file.stem):
            # prune file contains the vocab to keep
            vocab = set(whole_lines_from_file(prune_file))
            _log.info(f"vocab file has {len(vocab)} types (including CLS and SEP tokens)\n")

            # ensure that end vocab only has words in the embedding json
            vocab = vocab.intersection(wv_vocab)

        elif "umls" in str(w2v_model_file.stem):
            # prune file contains vocab to prune
            with open(prune_file, 'rb') as handle:
                prune_vocab = pickle.load(handle)
                vocab = wv_vocab.difference(prune_vocab)
        else:
            raise ValueError(f"Invalid input_type option!")

        # get rid of CLS and SEP tokens too
        try:
            vocab.remove("[CLS]")
        except KeyError:
            _log.info(f"[CLS] not in vocab")
        try:
            vocab.remove("[SEP]")
        except KeyError:
            _log.info(f"[SEP] not in vocab")
        _log.info(f"Pruned vocab has {len(vocab)} token types\n")

    else:
        # unpruned vocab from loaded wv
        vocab = wv.key_to_index.keys()
        _log.info(f"Using whole vocab of {len(vocab)} token types\n")

    _log.info(f"re_ordering word2id based on adding PAD and UNK tokens...\n")
    word2id = {"<PAD>": 0, "<UNK>": 1}
    mat = np.zeros((len(vocab) + len(word2id), embedding_dim))
    _log.info(f"npy matrix is of shape: {mat.shape}\n")

    # initialize UNK embedding with random normal
    mat[1] = np.random.randn(100)
    for word in sorted(vocab):
        vector = wv.get_vector(word, norm=normed)
        mat[len(word2id)] = vector
        word2id[word] = len(word2id)

    assert mat.shape[0] == len(word2id)

    if outfile is None:
        if prune:
            outfile = f"{Path(w2v_model_file).stem}_pruned"
        else:
            outfile = Path(w2v_model_file).stem

    output_dir = Path(w2v_model_file).parent
    mat_fname = output_dir / f"{outfile}.npy"
    map_fname = output_dir / f"{outfile}.json"

    _log.info(f'Saving word2id at {map_fname} and numpy matrix at {mat_fname} ...')

    np.save(str(mat_fname), mat)

    with open(map_fname, 'w', encoding='utf-8', errors='ignore') as wf:
        json.dump(word2id, wf)

    return mat_fname, map_fname


@ex.capture
def word_embeddings(_log,
                    version,
                    input_type,
                    data_iterator,
                    notes_file,
                    slice_pos,
                    save_wv_only=False,
                    model_type="w2v",
                    **kwargs):
    """
    Updated for Gensim >= 4.0, train and save Word2Vec Model

    :param _log: logging for Sacred Experiment
    :type _log: logging
    :param version: 50 or full
    :type version: str
    :param input_type: text or umls
    :type input_type: str
    :param data_iterator: type of corpus iterator to use, depending on corpus file: ProcessedIter, CorpusIter
    :type data_iterator: initialized class with __iter__
    :param notes_file: corpus file path if input_type is umls, this should be the dir with paths to all 3 partitions
    :type notes_file: Path or str
    :param slice_pos: which column of the read file from corpus reader to access
    :type slice_pos: int
    :param model_type: w2v vs fastText
    :type model_type: str
    :param save_wv_only: True if saving only the lightweight KeyedVectors object of the trained model
    :type save_wv_only: bool

    :return: path to saved model file

    """
    notes_file_path = Path(notes_file)
    if model_type == "fastText":
        model_name = f"{model_type}_processed_{version}_{input_type}.model"
    else:
        model_name = f"processed_{version}_{input_type}.model"
    if input_type == "text":
        sentences = data_iterator(notes_file_path, slice_pos)
    elif input_type == "umls":
        assert notes_file_path.is_dir(), f"{notes_file} has to be a dir for 'umls' input_type!!"
        tr_fp = notes_file_path / f"train_{version}_{input_type}.txt"
        dev_fp = notes_file_path / f"dev_{version}_{input_type}.txt"
        test_fp = notes_file_path / f"test_{version}_{input_type}.txt"
        tr_iter, dev_iter, test_iter = data_iterator(tr_fp), data_iterator(dev_fp), data_iterator(test_fp)
        sentences = itertools.chain(tr_iter, dev_iter, test_iter)
    else:
        raise ValueError(f"Invalid input_type option!")

    if model_type == "w2v":
        model = Word2Vec(**kwargs)
    elif model_type == "fastText":
        model = FastText(**kwargs)
    else:
        raise NotImplementedError

    _log.info(f"Building {model_type} vocab on {input_type} files from {notes_file}...")
    model.build_vocab(sentences)

    _log.info(f"Training on {model.corpus_count} sentences over {model.epochs} iterations...")
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

    if notes_file_path.is_file():
        embedding_dir = notes_file_path.parent.parent / f"{model_name.split('.')[-1]}"
    elif notes_file_path.is_dir():
        embedding_dir = notes_file_path.parent / f"{model_name.split('.')[-1]}"
    else:
        raise FileNotFoundError(f"{notes_file_path} is not a valid path!")

    if not embedding_dir.exists():
        embedding_dir.mkdir(parents=True, exist_ok=False)

    out_file = embedding_dir / model_name
    _log.info(f"writing embeddings to {out_file}")

    if save_wv_only:
        out_file = embedding_dir / f"{out_file.stem}.wordvectors"
        _log.info(f"only KeyedVectors are saved to {out_file}!! This is no longer trainable!!")
        model.wv.save(str(out_file))
        return out_file

    model.save(str(out_file))
    return out_file


@ex.config
def default_cfg():
    """Default Configs for Processing MIMIC-III text"""

    # data directory and file organization
    data_dir = PROJ_FOLDER / "data"
    version = "full"
    mimic_dir = Path(data_dir) / "mimic3" / f"{version}"

    # word_embedding func params
    wem_params = dict(version=version,
                      input_type="text",
                      data_iterator=MimicIter,
                      notes_file=str(mimic_dir / f"disch_{version}.csv"),
                      slice_pos=3,
                      save_wv_only=False,
                      model_type="w2v")  # w2v for Word2Vec or fastText for FastText model

    # gensim.models.Word2Vec params
    w2v_params = dict(vector_size=100,
                      min_count=0,
                      epochs=5,
                      workers=4,
                      shrink_windows=True)

    # gensim_to_npy func params
    gen_npy_params = dict(prune=True,
                          prune_file=mimic_dir / "vocab.csv",
                          normed=False,  # whether to also save L2-normed vectors as embeddings
                          embedding_dim=w2v_params["vector_size"])


@ex.named_config
def train_only():
    """Configs for Processing MIMIC-III text with only train partition"""

    # data directory and file organization
    data_dir = PROJ_FOLDER / "data"
    version = "full"
    mimic_dir = Path(data_dir) / "mimic3" / f"{version}"

    # word_embedding func params
    wem_params = dict(notes_file=str(mimic_dir / f"train_{version}.csv"),
                      slice_pos=2)

    # gensim_to_npy func params, use default values


@ex.named_config
def cui():
    """Baseline configs for Embedding CUIs as input tokens"""

    # data directory and file organization
    data_dir = PROJ_FOLDER / "data"
    version = "full"
    mimic_dir = Path(data_dir) / "linked_data" / f"{version}"

    # word_embedding func params
    wem_params = dict(input_type="umls",
                      data_iterator=MimicCuiIter,
                      notes_file=str(mimic_dir))

    # gensim_to_npy func params
    gen_npy_params = dict(prune_file=mimic_dir / f"{version}_cuis_to_discard.pickle")


@ex.main
def run_word_embeddings(wem_params, w2v_params, gen_npy_params, _log, _run):
    _log.info(f"\n=========START W2V EMBEDDING TRAINING==========\n")

    w2v_file = word_embeddings(_log, **wem_params, **w2v_params)
    npy_fp, mapping_fp = gensim_to_npy(w2v_file, _log, **gen_npy_params)

    normed = gen_npy_params["normed"]
    if normed:
        _log.info(f"saving normed vectors as well...")
        out_fname = f"{w2v_file.stem}_normed"
        npy_fp_norm, mapping_fp_norm = gensim_to_npy(w2v_file, normed=normed, outfile=out_fname)

        # log model files to Sacred
        ex.add_artifact(filename=f"{mapping_fp_norm}")

    # Log model files to Sacred
    ex.add_artifact(filename=f"{w2v_file}")
    ex.add_artifact(filename=f"{mapping_fp}")
    _log.info(f"\n=========FINISHED W2V EMBEDDING TRAINING==========\n")


if __name__ == "__main__":
    ex.run_commandline()
