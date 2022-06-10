# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# Slight changes from:
# 
# https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network/blob/master/preprocess_mimic3.py
#
# and
#
# https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network/blob/master/utils.py
# ----------------------------------------------------------------------------------------------------------------

import numpy as np
import gensim.models
import gensim.models.word2vec as w2v
import gensim.models.fasttext as fasttext
import logging
import csv
import struct
import codecs

from tqdm import tqdm


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__file__)


def gensim_to_embeddings(wv_file, vocab_file, Y, outfile=None):
    model = gensim.models.Word2Vec.load(wv_file)
    wv = model.wv
    # free up memory
    del model
    
    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.strip()
            if line != '':
                vocab.add(line)
    ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}
    
    W, words = build_matrix(ind2w, wv)
    
    if outfile is None:
        outfile = wv_file.replace('.w2v', '.embed')
    
    # smash that save button
    save_embeddings(W, words, outfile)


def gensim_to_fasttext_embeddings(wv_file, vocab_file, Y, outfile=None):
    model = gensim.models.FastText.load(wv_file)
    wv = model.wv
    # free up memory
    del model
    
    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.strip()
            if line != '':
                vocab.add(line)
    ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}
    
    W, words = build_matrix(ind2w, wv)
    
    if outfile is None:
        outfile = wv_file.replace('.fasttext', '.fasttext.embed')
    
    # smash that save button
    save_embeddings(W, words, outfile)


def build_matrix(ind2w, wv):
    """
        Go through vocab in order. Find vocab word in wv.index2word, then call wv.word_vec(wv.index2word[i]).
        Put results into one big matrix.
        Note: ind2w starts at 1 (saving 0 for the pad character), but gensim word vectors starts at 0
    """
    W = np.zeros((len(ind2w)+1, len(wv.word_vec(wv.index2word[0])) ))
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


def word_embeddings(Y, notes_file, embedding_size, min_count, n_iter):
    modelname = "processed_%s.w2v" % (Y)
    sentences = ProcessedIter(Y, notes_file)
    
    model = w2v.Word2Vec(size=embedding_size, min_count=min_count, workers=4, iter=n_iter)
    logger.info("building word2vec vocab on %s..." % (notes_file))
    
    model.build_vocab(sentences)
    logger.info("training...")
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    out_file = '/'.join(notes_file.split('/')[:-1] + [modelname])
    logger.info("writing embeddings to %s" % (out_file))
    model.save(out_file)
    return out_file


def fasttext_embeddings(Y, notes_file, embedding_size, min_count, n_iter):
    modelname = "processed_%s.fasttext" % (Y)
    sentences = ProcessedIter(Y, notes_file)
    
    model = fasttext.FastText(size=embedding_size, min_count=min_count, iter=n_iter)
    logger.info("building fasttext vocab on %s..." % (notes_file))
    
    model.build_vocab(sentences)
    logger.info("training...")
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    out_file = '/'.join(notes_file.split('/')[:-1] + [modelname])
    logger.info("writing embeddings to %s" % (out_file))
    model.save(out_file)
    return out_file


class ProcessedIter(object):
    
    def __init__(self, Y, filename):
        self.filename = filename
    
    def __iter__(self):
        with open(self.filename) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                yield (row[3].split())


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
        
        while i<continue_to_read:
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
                    continue # it's a head
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
    pretrain_emb = np.zeros([len(word_alphabet)+2, embedd_dim], dtype=np.float32)  # add UNK (last) and PAD (0)
    perfect_match = 0
    case_match = 0
    digits_replaced_with_zeros_found = 0
    lowercase_and_digits_replaced_with_zeros_found = 0
    not_match = 0
    for word, index in word_alphabet.items():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        
        elif re.sub(r'\d', '0', word) in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[re.sub(r'\d', '0', word)])
            else:
                pretrain_emb[index,:] = embedd_dict[re.sub(r'\d', '0', word)]
            digits_replaced_with_zeros_found += 1
        
        elif re.sub(r'\d', '0', word.lower()) in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[re.sub(r'\d', '0', word.lower())])
            else:
                pretrain_emb[index,:] = embedd_dict[re.sub(r'\d', '0', word.lower())]
            lowercase_and_digits_replaced_with_zeros_found += 1
        
        else:
            if norm:
                pretrain_emb[index, :] = norm2one(np.random.uniform(-scale, scale, [1, embedd_dim]))
            else:
                pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
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
            perfect_match * 100.0/len(word_alphabet), 
            case_match * 100.0/len(word_alphabet), 
            digits_replaced_with_zeros_found * 100.0/len(word_alphabet),
            lowercase_and_digits_replaced_with_zeros_found * 100.0/len(word_alphabet), 
            not_match*100.0/len(word_alphabet))
        )
    
    return pretrain_emb, embedd_dim


def main():
    MIMIC_3_DIR = '../../data/mimic3'
    Y = 'full'
    
    w2v_file = word_embeddings(Y, '%s/disch_full.csv' % MIMIC_3_DIR, 100, 0, 5)
    gensim_to_embeddings('%s/processed_full.w2v' % MIMIC_3_DIR, '%s/vocab.csv' % MIMIC_3_DIR, Y)
    
    fasttext_file = fasttext_embeddings(Y, '%s/disch_full.csv' % MIMIC_3_DIR, 100, 0, 5)
    gensim_to_fasttext_embeddings('%s/processed_full.fasttext' % MIMIC_3_DIR, '%s/vocab.csv' % MIMIC_3_DIR, Y)


if __name__=="__main__":
    main()
