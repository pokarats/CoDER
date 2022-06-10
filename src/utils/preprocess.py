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

import operator
import csv
import numpy as np
import logging
import nltk
import pandas as pd

from collections import Counter, defaultdict
from tqdm import tqdm
from scipy.sparse import csr_matrix
from nltk.tokenize import RegexpTokenizer


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__file__)


nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer = RegexpTokenizer(r'\w+')


def build_vocab(vocab_min, infile, vocab_filename):
    """
        INPUTS:
            vocab_min: how many documents a word must appear in to be kept
            infile: (training) data file to build vocabulary from
            vocab_filename: name for the file to output
    """
    with open(infile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # header
        next(reader)

        # 0. read in data
        logger.info("reading in data...")
        # holds number of terms in each document
        note_numwords = []
        # indices where notes start
        note_inds = [0]
        # indices of discovered words
        indices = []
        # holds a bunch of ones
        data = []
        # keep track of discovered words
        vocab = {}
        # build lookup table for terms
        num2term = {}
        # preallocate array to hold number of notes each term appears in
        note_occur = np.zeros(400000, dtype=int)
        i = 0
        for row in reader:
            text = row[2]
            numwords = 0
            for term in text.split():
                # put term in vocab if it's not there. else, get the index
                index = vocab.setdefault(term, len(vocab))
                indices.append(index)
                num2term[index] = term
                data.append(1)
                numwords += 1
            # record where the next note starts
            note_inds.append(len(indices))
            indset = set(indices[note_inds[-2]:note_inds[-1]])
            # go thru all the word indices you just added, and add to the note occurrence count for each of them
            for ind in indset:
                note_occur[ind] += 1
            note_numwords.append(numwords)
            i += 1
        # clip trailing zeros
        note_occur = note_occur[note_occur > 0]

        # turn vocab into a list so indexing doesn't get fd up when we drop rows
        vocab_list = np.array([word for word, ind in sorted(vocab.items(), key=operator.itemgetter(1))])

        # 1. create sparse document matrix
        C = csr_matrix((data, indices, note_inds), dtype=int).transpose()
        # also need the numwords array to be a sparse matrix
        note_numwords = csr_matrix(1. / np.array(note_numwords))

        # 2. remove rows with less than 3 total occurrences
        logger.info("removing rare terms")
        # inds holds indices of rows corresponding to terms that occur in < 3 documents
        inds = np.nonzero(note_occur >= vocab_min)[0]
        logger.info(str(len(inds)) + " terms qualify out of " + str(C.shape[0]) + " total")
        # drop those rows
        C = C[inds, :]
        note_occur = note_occur[inds]
        vocab_list = vocab_list[inds]

        logger.info("writing output")
        with open(vocab_filename, 'w') as vocab_file:
            for word in vocab_list:
                vocab_file.write(word + "\n")


def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits,
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code


def write_discharge_summaries(out_file, min_sentence_len, notes_file):
    logger.info("processing notes file")
    with open(notes_file, 'r') as csvfile:
        with open(out_file, 'w') as outfile:
            logger.info("writing to %s" % (out_file))
            outfile.write(','.join(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']) + '\n')
            notereader = csv.reader(csvfile)
            next(notereader)
            
            for line in tqdm(notereader):
                subj = int(line[1])
                category = line[6]
                if category == "Discharge summary":
                    note = line[10]
                    
                    all_sents_inds = []
                    generator = nlp_tool.span_tokenize(note)
                    for t in generator:
                        all_sents_inds.append(t)
                    
                    text = ""
                    for ind in range(len(all_sents_inds)):
                        start = all_sents_inds[ind][0]
                        end = all_sents_inds[ind][1]
                        
                        sentence_txt = note[start:end]
                        
                        tokens = [t.lower() for t in tokenizer.tokenize(sentence_txt) if not t.isnumeric()]
                        if ind == 0:
                            text += '[CLS] ' + ' '.join(tokens) + ' [SEP]'
                        else:
                            text += ' [CLS] ' + ' '.join(tokens) + ' [SEP]'
                    
                    text = '"' + text + '"'
                    outfile.write(','.join([line[1], line[2], line[4], text]) + '\n')
    
    return out_file


def next_labels(labelsfile):
    """
        Generator for label sets from the label file
    """
    labels_reader = csv.reader(labelsfile)
    # header
    next(labels_reader)
    
    first_label_line = next(labels_reader)
    
    cur_subj = int(first_label_line[0])
    cur_hadm = int(first_label_line[1])
    cur_labels = [first_label_line[2]]
    
    for row in labels_reader:
        subj_id = int(row[0])
        hadm_id = int(row[1])
        code = row[2]
        # keep reading until you hit a new hadm id
        if hadm_id != cur_hadm or subj_id != cur_subj:
            yield cur_subj, cur_labels, cur_hadm
            cur_labels = [code]
            cur_subj = subj_id
            cur_hadm = hadm_id
        else:
            # add to the labels and move on
            cur_labels.append(code)
    
    yield cur_subj, cur_labels, cur_hadm


def next_notes(notesfile):
    """
        Generator for notes from the notes file
        This will also concatenate discharge summaries and their addenda, which have the same subject and hadm id
    """
    nr = csv.reader(notesfile)
    # header
    next(nr)
    
    first_note = next(nr)
    
    cur_subj = int(first_note[0])
    cur_hadm = int(first_note[1])
    cur_text = first_note[3]
    
    for row in nr:
        subj_id = int(row[0])
        hadm_id = int(row[1])
        text = row[3]
        
        # keep reading until you hit a new hadm id
        if hadm_id != cur_hadm or subj_id != cur_subj:
            yield cur_subj, cur_text, cur_hadm
            cur_text = text
            cur_subj = subj_id
            cur_hadm = hadm_id
        else:
            # concatenate to the discharge summary and move on
            cur_text += " " + text
    
    yield cur_subj, cur_text, cur_hadm


def concat_data(labelsfile, notes_file, outfilename):
    """
        INPUTS:
            labelsfile: sorted by hadm id, contains one label per line
            notes_file: sorted by hadm id, contains one note per line
    """
    with open(labelsfile, 'r') as lf:
        logger.info("CONCATENATING")
        with open(notes_file, 'r') as notesfile:
            
            with open(outfilename, 'w') as outfile:
                w = csv.writer(outfile)
                w.writerow(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS'])
                
                labels_gen = next_labels(lf)
                notes_gen = next_notes(notesfile)
                
                for i, (subj_id, text, hadm_id) in enumerate(notes_gen):
                    if i % 10000 == 0:
                        logger.info(str(i) + " done")
                    cur_subj, cur_labels, cur_hadm = next(labels_gen)

                    if cur_hadm == hadm_id:
                        w.writerow([subj_id, str(hadm_id), text, ';'.join(cur_labels)])
                    else:
                        logger.info("couldn't find matching hadm_id. data is probably not sorted correctly")
                        break
    
    return outfilename


def split_data(labeledfile, base_name, mimic_dir):
    logger.info("SPLITTING")
    # create and write headers for train, dev, test
    train_name = '%s_train_split.csv' % (base_name)
    dev_name = '%s_dev_split.csv' % (base_name)
    test_name = '%s_test_split.csv' % (base_name)
    train_file = open(train_name, 'w')
    dev_file = open(dev_name, 'w')
    test_file = open(test_name, 'w')
    train_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")
    dev_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")
    test_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")
    
    hadm_ids = {}
    
    # read in train, dev, test splits
    for splt in ['train', 'dev', 'test']:
        hadm_ids[splt] = set()
        with open('%s/%s_full_hadm_ids.csv' % (mimic_dir, splt), 'r') as f:
            for line in f:
                hadm_ids[splt].add(line.rstrip())
    
    with open(labeledfile, 'r') as lf:
        reader = csv.reader(lf)
        next(reader)
        i = 0
        cur_hadm = 0
        for row in reader:
            # filter text, write to file according to train/dev/test split
            if i % 10000 == 0:
                logger.info(str(i) + " read")

            hadm_id = row[1]
            
            if hadm_id in hadm_ids['train']:
                train_file.write(','.join(row) + "\n")
            elif hadm_id in hadm_ids['dev']:
                dev_file.write(','.join(row) + "\n")
            elif hadm_id in hadm_ids['test']:
                test_file.write(','.join(row) + "\n")
            
            i += 1
        
        train_file.close()
        dev_file.close()
        test_file.close()
    
    return train_name, dev_name, test_name


def main():
    MIMIC_3_DIR = '../../data/mimic3'
    Y = 'full'
    notes_file = '%s/NOTEEVENTS.csv' % MIMIC_3_DIR
    
    # ----------------------------------------------------------
    # Step 1: Process code-related files
    # ----------------------------------------------------------
    dfproc = pd.read_csv('%s/PROCEDURES_ICD.csv' % MIMIC_3_DIR)
    dfdiag = pd.read_csv('%s/DIAGNOSES_ICD.csv' % MIMIC_3_DIR)
    
    dfdiag['absolute_code'] = dfdiag.apply(lambda row: str(reformat(str(row[4]), True)), axis=1)
    dfproc['absolute_code'] = dfproc.apply(lambda row: str(reformat(str(row[4]), False)), axis=1)
    
    dfcodes = pd.concat([dfdiag, dfproc])
    dfcodes.to_csv(
        '%s/ALL_CODES.csv' % MIMIC_3_DIR, index=False,
        columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'absolute_code'],
        header=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE']
    )

    df = pd.read_csv(
        '%s/ALL_CODES.csv' % MIMIC_3_DIR, dtype={"ICD9_CODE": str}
    )
    logger.info("unique ICD9 code: {}".format(len(df['ICD9_CODE'].unique())))

    # ----------------------------------------------------------
    # Step 2: Process notes
    # ----------------------------------------------------------
    min_sentence_len = 3
    disch_full_file = write_discharge_summaries(
        "%s/disch_full.csv" % MIMIC_3_DIR, min_sentence_len, 
        '%s/NOTEEVENTS.csv' % (MIMIC_3_DIR)
    )
    
    df = pd.read_csv('%s/disch_full.csv' % MIMIC_3_DIR)
    df = df.sort_values(['SUBJECT_ID', 'HADM_ID'])
    
    # ----------------------------------------------------------
    # Step 3: Filter out the codes that not emerge in notes
    # ----------------------------------------------------------
    hadm_ids = set(df['HADM_ID'])
    with open('%s/ALL_CODES.csv' % MIMIC_3_DIR, 'r') as lf:
        with open('%s/ALL_CODES_filtered.csv' % MIMIC_3_DIR, 'w') as of:
            w = csv.writer(of)
            w.writerow(['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'ADMITTIME', 'DISCHTIME'])
            r = csv.reader(lf)
            # header
            next(r)
            for i,row in enumerate(r):
                hadm_id = int(row[2])
                # print(hadm_id)
                # break
                if hadm_id in hadm_ids:
                    w.writerow(row[1:3] + [row[-1], '', ''])

    dfl = pd.read_csv('%s/ALL_CODES_filtered.csv' % MIMIC_3_DIR, index_col=None)
    dfl = dfl.sort_values(['SUBJECT_ID', 'HADM_ID'])
    dfl.to_csv('%s/ALL_CODES_filtered.csv' % MIMIC_3_DIR, index=False)
    
    sorted_file = '%s/disch_full.csv' % MIMIC_3_DIR
    df.to_csv(sorted_file, index=False)
    
    # ----------------------------------------------------------
    # Step 4: Link notes with their code
    # ----------------------------------------------------------
    labeled = concat_data(
        '%s/ALL_CODES_filtered.csv' % MIMIC_3_DIR, sorted_file, 
        '%s/notes_labeled.csv' % MIMIC_3_DIR
    )
    dfnl = pd.read_csv(labeled)
    
    # ----------------------------------------------------------
    # Step 5: Statistic unique word, total word, HADM_ID number
    # ----------------------------------------------------------
    types = set()
    num_tok = 0
    for row in dfnl.itertuples():
        for w in row[3].split():
            types.add(w)
            num_tok += 1
    
    logger.info("num types: {}, num tokens: {}".format(len(types), num_tok))
    logger.info("HADM_ID: {}".format(len(dfnl['HADM_ID'].unique())))
    logger.info("SUBJECT_ID: {}".format(len(dfnl['SUBJECT_ID'].unique())))
    
    # ----------------------------------------------------------
    # Step 6: Split data into train dev test
    # ----------------------------------------------------------
    fname = '%s/notes_labeled.csv' % MIMIC_3_DIR
    base_name = "%s/disch" % MIMIC_3_DIR #for output
    tr, dv, te = split_data(fname, base_name, MIMIC_3_DIR)

    vocab_min = 3
    vname = '%s/vocab.csv' % MIMIC_3_DIR
    build_vocab(vocab_min, tr, vname)
    
    # ----------------------------------------------------------
    # Step 7: Sort data by its note length, add length to the 
    #         last column
    # ----------------------------------------------------------
    for splt in ['train', 'dev', 'test']:
        filename = '%s/disch_%s_split.csv' % (MIMIC_3_DIR, splt)
        df = pd.read_csv(filename)
        df['length'] = df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)
        df = df.sort_values(['length'])
        df.to_csv('%s/%s_full.csv' % (MIMIC_3_DIR, splt), index=False)
    
    # ----------------------------------------------------------
    # Step 9: Statistic the top 50 code
    # ----------------------------------------------------------
    Y = 50

    counts = Counter()
    dfnl = pd.read_csv('%s/notes_labeled.csv' % MIMIC_3_DIR)
    for row in dfnl.itertuples():
        for label in str(row[4]).split(';'):
            counts[label] += 1
    
    codes_50 = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    codes_50 = [code[0] for code in codes_50[:Y]]
    
    with open('%s/TOP_%s_CODES.csv' % (MIMIC_3_DIR, str(Y)), 'w') as of:
        w = csv.writer(of)
        for code in codes_50:
            w.writerow([code])
    
    # ----------------------------------------------------------
    # Step 10: Split data according to train_50_hadm_ids dev... 
    #          and test...
    # ----------------------------------------------------------
    for splt in ['train', 'dev', 'test']:
        logger.info(splt)
        hadm_ids = set()
        with open('%s/%s_50_hadm_ids.csv' % (MIMIC_3_DIR, splt), 'r') as f:
            for line in f:
                hadm_ids.add(line.rstrip())
        with open('%s/notes_labeled.csv' % MIMIC_3_DIR, 'r') as f:
            with open('%s/%s_%s.csv' % (MIMIC_3_DIR, splt, str(Y)), 'w') as of:
                r = csv.reader(f)
                w = csv.writer(of)
                # header
                w.writerow(next(r))
                i = 0
                for row in r:
                    hadm_id = row[1]
                    if hadm_id not in hadm_ids:
                        continue
                    codes = set(str(row[3]).split(';'))
                    filtered_codes = codes.intersection(set(codes_50))
                    if len(filtered_codes) > 0:
                        w.writerow(row[:3] + [';'.join(filtered_codes)])
                        i += 1
    
    # ----------------------------------------------------------
    # Step 11: Sort data by its note length, add length to the 
    #          last column
    # ----------------------------------------------------------
    for splt in ['train', 'dev', 'test']:
        filename = '%s/%s_%s.csv' % (MIMIC_3_DIR, splt, str(Y))
        df = pd.read_csv(filename)
        df['length'] = df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)
        df = df.sort_values(['length'])
        df.to_csv('%s/%s_%s.csv' % (MIMIC_3_DIR, splt, str(Y)), index=False)


if __name__=="__main__":
    main()
