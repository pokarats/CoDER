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
import argparse
import time
import traceback

import numpy as np
from gensim.models import Word2Vec, FastText, KeyedVectors
from datetime import date
import logging
import struct
import codecs
import re
import json
import sys
import os
import platform
from pathlib import Path
from tqdm import tqdm


if platform.system() != 'Darwin':
    sys.path.append(os.getcwd())  # only needed for slurm
from src.utils.corpus_readers import CorpusIter, ProcessedIter, MimicIter, MimicCuiIter


PROJ_FOLDER = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)


def train_and_dump_word2vec(
        medline_entities_linked_fname,
        output_dir,
        n_workers=4,
        n_iter=10
):
    # fix embed dim = 100 and max vocab size to 50k
    model = Word2Vec(vector_size=100, workers=n_workers, epochs=n_iter, max_final_vocab=50000)
    sentences = CorpusIter(medline_entities_linked_fname)

    logger.info(f'Building word2vec vocab on {medline_entities_linked_fname}...')
    model.build_vocab(sentences)

    logger.info('Training ...')
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

    os.makedirs(output_dir, exist_ok=True)
    logger.info('Saving word2vec model ...')
    model.save(os.path.join(output_dir, 'word2vec.pubmed2019.50d.gz'))

    wv = model.wv
    del model  # free up memory

    word2id = {"<PAD>": 0, "<UNK>": 1}
    mat = np.zeros((len(wv.vocab.keys()) + 2, 100))
    # initialize UNK embedding with random normal
    mat[1] = np.random.randn(100)

    for word in sorted(wv.vocab.keys()):
        vocab_item = wv.vocab[word]
        vector = wv.vectors[vocab_item.index]
        mat[len(word2id)] = vector
        word2id[word] = len(word2id)

    mat_fname = Path(output_dir) / f'word2vec.guttmann.100d_mat.npy'
    map_fname = Path(output_dir) / f'word2vec.guttmann.100d_word2id.json'

    logger.info(f'Saving word2id at {map_fname} and numpy matrix at {mat_fname} ...')

    np.save(str(mat_fname), mat)
    with open(map_fname, 'w', encoding='utf-8', errors='ignore') as wf:
        json.dump(word2id, wf)


def gensim_to_npy(w2v_model_file, normed=False, outfile=None, embedding_dim=100):
    if Path(w2v_model_file).suffix == '.wordvectors':
        wv = KeyedVectors.load(w2v_model_file, mmap='r')
    else:
        loaded_model = Word2Vec.load(w2v_model_file)
        wv = loaded_model.wv  # this is just the KeyedVectors parts
        # free up memory
        del loaded_model

    assert embedding_dim == wv.vector_size, f"specified embedding_dim ({embedding_dim}) and " \
                                            f"loaded model vector_size ({wv.vector_size}) mismatch!"

    word2id = {"<PAD>": 0, "<UNK>": 1}
    mat = np.zeros((len(wv.key_to_index.keys()) + len(word2id), embedding_dim))
    # initialize UNK embedding with random normal
    mat[1] = np.random.randn(100)

    for word in sorted(wv.key_to_index.keys()):
        vector = wv.get_vector(word, norm=normed)
        mat[len(word2id)] = vector
        word2id[word] = len(word2id)

    if outfile is None:
        outfile = Path(w2v_model_file).stem

    output_dir = Path(w2v_model_file).parent
    mat_fname = output_dir / f'{outfile}.npy'
    map_fname = output_dir / f'{outfile}.json'

    logger.info(f'Saving word2id at {map_fname} and numpy matrix at {mat_fname} ...')

    np.save(str(mat_fname), mat)

    with open(map_fname, 'w', encoding='utf-8', errors='ignore') as wf:
        json.dump(word2id, wf)


def gensim_to_embeddings(wv_file, vocab_file, outfile=None):
    model = Word2Vec.load(wv_file)
    wv = model.wv
    # free up memory
    del model

    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i, line in enumerate(vocabfile):
            line = line.strip()
            if line != '':
                vocab.add(line)
    ind2w = {i + 1: w for i, w in enumerate(sorted(vocab))}

    W, words = build_matrix(ind2w, wv)

    if outfile is None:
        suffix = Path(wv_file).suffix
        outfile = wv_file.replace(suffix, '.embed')

    # smash that save button
    save_embeddings(W, words, outfile)


def gensim_to_fasttext_embeddings(wv_file, vocab_file, Y, outfile=None):
    model = FastText.load(wv_file)
    wv = model.wv
    # free up memory
    del model

    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i, line in enumerate(vocabfile):
            line = line.strip()
            if line != '':
                vocab.add(line)
    ind2w = {i + 1: w for i, w in enumerate(sorted(vocab))}

    W, words = build_matrix(ind2w, wv)

    if outfile is None:
        outfile = wv_file.replace('.fasttext', '.fasttext.embed')

    # smash that save button
    save_embeddings(W, words, outfile)


def build_matrix(ind2w, wv):
    """
        Go through vocab in order. Find vocab word in wv.index2word, then call wv.word_vec(wv.index2word[i]).
        Put results into 1 big matrix.
        Note: ind2w starts at 1 (saving 0 for the pad character), but gensim word vectors starts at 0
    """
    W = np.zeros((len(ind2w) + 1, len(wv.word_vec(wv.index2word[0]))))
    words = ["**PAD**"]
    W[0][:] = np.zeros(len(wv.word_vec(wv.index2word[0])))
    for idx, word in tqdm(ind2w.items()):
        if idx >= W.shape[0]:
            break
        W[idx][:] = wv.word_vec(word)
        words.append(word)
    return W, words


def save_embeddings(W, words, outfile):
    with open(outfile, 'w') as o:
        # pad token already included
        for i in range(len(words)):
            line = [words[i]]
            line.extend([str(d) for d in W[i]])
            o.write(" ".join(line) + "\n")


def load_embeddings(embed_file):
    # also normalizes the embeddings
    W = []
    with open(embed_file) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W.append(vec)
        # UNK embedding, gaussian randomly initialized
        logger.info("adding unk embedding")
        vec = np.random.randn(len(W[-1]))
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        W.append(vec)
    W = np.array(W)
    return W


def word_embeddings(dataset_vers,
                    notes_file,
                    embedding_size=100,
                    min_count=0,
                    n_iter=5,
                    n_workers=4,
                    save_wv_only=False,
                    data_iterator=ProcessedIter):
    """
    Updated for Gensim >= 4.0, train and save Word2Vec Model


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

    logger.info(f"building word2vec vocab on {notes_file}...")
    model.build_vocab(sentences)

    logger.info(f"training on {model.corpus_count} sentences over {model.epochs} iterations...")
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

    embedding_dir = Path(notes_file).parent / f"{model_name.split('.')[-1]}"
    if not embedding_dir.exists():
        embedding_dir.mkdir(parents=True, exist_ok=False)

    out_file = embedding_dir / model_name
    logger.info(f"writing embeddings to {out_file}")

    if save_wv_only:
        out_file = embedding_dir / f"{out_file.stem}.wordvectors"
        logger.info(f"only KeyedVectors are saved to {out_file}!! This is no longer trainable!!")
        model_wv = model.wv
        model_wv.save(out_file)
        return out_file

    model.save(out_file)
    return out_file


def fasttext_embeddings(Y, notes_file, embedding_size, min_count, n_iter):
    modelname = "processed_%s.fasttext" % (Y)
    sentences = ProcessedIter(notes_file)

    model = FastText(vector_size=embedding_size, min_count=min_count, epochs=n_iter)
    logger.info("building fasttext vocab on %s..." % (notes_file))

    model.build_vocab(sentences)
    logger.info("training...")
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    out_file = '/'.join(notes_file.split('/')[:-1] + [modelname])
    logger.info("writing embeddings to %s" % out_file)
    model.save(out_file)
    return out_file


def _readString(f, code):
    # s = unicode()
    s = str()
    c = f.read(1)
    value = ord(c)

    while value != 10 and value != 32:
        if 0x00 < value < 0xbf:
            continue_to_read = 0
        elif 0xC0 < value < 0xDF:
            continue_to_read = 1
        elif 0xE0 < value < 0xEF:
            continue_to_read = 2
        elif 0xF0 < value < 0xF4:
            continue_to_read = 3
        else:
            raise RuntimeError("not valid utf-8 code")

        i = 0

        temp = bytes()
        temp = temp + c

        while i < continue_to_read:
            temp = temp + f.read(1)
            i += 1

        temp = temp.decode(code)
        s = s + temp

        c = f.read(1)
        value = ord(c)

    return s


def _readFloat(f):
    bytes4 = f.read(4)
    f_num = struct.unpack('f', bytes4)[0]
    return f_num


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()

    # emb_debug = []
    if embedding_path.find('.bin') != -1:
        with open(embedding_path, 'rb') as f:
            wordTotal = int(_readString(f, 'utf-8'))
            embedd_dim = int(_readString(f, 'utf-8'))

            for i in range(wordTotal):
                word = _readString(f, 'utf-8')
                # emb_debug.append(word)

                word_vector = []
                for j in range(embedd_dim):
                    word_vector.append(_readFloat(f))
                word_vector = np.array(word_vector, np.float)

                f.read(1)  # a line break

                embedd_dict[word] = word_vector

    else:
        with codecs.open(embedding_path, 'r', 'UTF-8') as file:
            for line in file:
                # logging.info(line)
                line = line.strip()
                if len(line) == 0:
                    continue
                # tokens = line.split()
                tokens = re.split(r"\s+", line)
                if len(tokens) == 2:
                    continue  # it's a head
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    # assert (embedd_dim + 1 == len(tokens))
                    if embedd_dim + 1 != len(tokens):
                        continue
                embedd = np.zeros([1, embedd_dim])
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd

    return embedd_dict, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def build_pretrain_embedding(embedding_path, word_alphabet, norm):
    embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.zeros([len(word_alphabet) + 2, embedd_dim], dtype=np.float32)  # add UNK (last) and PAD (0)
    perfect_match = 0
    case_match = 0
    digits_replaced_with_zeros_found = 0
    lowercase_and_digits_replaced_with_zeros_found = 0
    not_match = 0
    for word, index in word_alphabet.items():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1

        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1

        elif re.sub(r'\d', '0', word) in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[re.sub(r'\d', '0', word)])
            else:
                pretrain_emb[index, :] = embedd_dict[re.sub(r'\d', '0', word)]
            digits_replaced_with_zeros_found += 1

        elif re.sub(r'\d', '0', word.lower()) in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[re.sub(r'\d', '0', word.lower())])
            else:
                pretrain_emb[index, :] = embedd_dict[re.sub(r'\d', '0', word.lower())]
            lowercase_and_digits_replaced_with_zeros_found += 1

        else:
            if norm:
                pretrain_emb[index, :] = norm2one(np.random.uniform(-scale, scale, [1, embedd_dim]))
            else:
                pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

    # initialize pad and unknown
    pretrain_emb[0, :] = np.zeros([1, embedd_dim], dtype=np.float32)
    if norm:
        pretrain_emb[-1, :] = norm2one(np.random.uniform(-scale, scale, [1, embedd_dim]))
    else:
        pretrain_emb[-1, :] = np.random.uniform(-scale, scale, [1, embedd_dim])

    logger.info("pretrained word emb size {}".format(len(embedd_dict)))
    logger.info(
        "prefect match:%.2f%%, case_match:%.2f%%, dig_zero_match:%.2f%%, "
        "case_dig_zero_match:%.2f%%, not_match:%.2f%%"
        % (
            perfect_match * 100.0 / len(word_alphabet),
            case_match * 100.0 / len(word_alphabet),
            digits_replaced_with_zeros_found * 100.0 / len(word_alphabet),
            lowercase_and_digits_replaced_with_zeros_found * 100.0 / len(word_alphabet),
            not_match * 100.0 / len(word_alphabet))
    )

    return pretrain_emb, embedd_dim


def main(cl_args):
    logger.info(f"\n=========START W2V EMBEDDING TRAINING==========\n")
    mimic_dir = Path(cl_args.data_dir) / cl_args.version
    notes_fp = mimic_dir / cl_args.filename

    if notes_fp.suffix == '.csv':
        data_iterator = MimicIter(notes_fp)
    elif notes_fp.suffix == '.txt':
        if cl_args.prune:
            cui_fp = mimic_dir / f"{cl_args.version}_cuis_to_discard.pickle"
            data_iterator = MimicCuiIter(notes_fp,
                                         threshold=0.7,
                                         pruned=cl_args.prune,
                                         discard_cuis_file=cui_fp)
        else:
            data_iterator = MimicCuiIter(notes_fp)
    else:
        logger.error(f"Invalid file type to reade!!")
        raise ValueError

    w2v_file = word_embeddings(dataset_vers=cl_args.version,
                               notes_file=notes_fp,
                               embedding_size=100,
                               min_count=cl_args.min_count,
                               n_iter=5,
                               n_workers=cl_args.n_process,
                               save_wv_only=False,
                               data_iterator=data_iterator)
    gensim_to_npy(w2v_file, cl_args.normed)

    if cl_args.normed:
        out_fname = f"{w2v_file.stem}_normed"
        gensim_to_npy(w2v_file, normed=cl_args.normed, outfile=out_fname)

    # gensim_to_embeddings('%s/processed_full.w2v' % mimic_3_dir, '%s/vocab.csv' % mimic_3_dir)
    # fasttext_file = fasttext_embeddings(Y, '%s/disch_full.csv' % MIMIC_3_DIR, 100, 0, 5)
    # gensim_to_fasttext_embeddings('%s/processed_full.fasttext' % MIMIC_3_DIR, '%s/vocab.csv' % MIMIC_3_DIR, Y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", action="store", type=str, default="data/mimic3",
        help="Path to data directory containing either MIMIC-III or linked_data"
    )
    parser.add_argument(
        "--version", action="store", type=str, default="50",
        help="Name of mimic-III dataset version to use: full vs 50 (for top50) or 1 for a 1 sample version"
    )
    parser.add_argument(
        "--seed", action="store", type=int, default=23,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--min_count", action="store", type=int, default=0,
        help="Min count for vocab gensim Word2Vec training"
    )
    parser.add_argument(
        "--threshold", action="store", type=float, default=0.7,
        help="Min threshold for similarity"
    )
    parser.add_argument(
        "--filename", action="store", type=str, default="train_50.csv",
        help="notes_filename to process, just the filename NOT whole path"
    )
    parser.add_argument(
        "--n_process", action="store", type=int, default=4,
        help="Number of processes to run in parallel multi-processing."
    )
    parser.add_argument(
        "--prune", action="store_true",
        help="Whether to prune CUIs before w2vec embeddings"
    )
    parser.add_argument(
        "--normed", action="store_true",
        help="Whether to get L2 normed vector for w2v embeddings"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Do not print to stdout (log only)."
    )

    args = parser.parse_args(sys.argv[1:])

    # Setup logging and start timer
    basename = Path(__file__).stem
    log_folder = PROJ_FOLDER / f"scratch/.log/{date.today():%y_%m_%d}"
    log_file = log_folder / f"{time.strftime('%Hh%Mm%Ss')}_{basename}_{args.version}_{args.comment}.log"

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

