# -*- coding: utf-8 -*-

import logging
import collections
import json

from scispacy.umls_linking import UmlsEntityLinker
from pathlib import Path
from tqdm import tqdm


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("src")

# replace these with semantic types in "semantic_types_mimic.txt" --> this has all the possible semantic types for
# the entire ICD9 code set found in ICD_umls2020aa.txt
SEMANTIC_TYPES_TO_INCLUDE = [
    'T017', 'T019', 'T020', 'T021', 'T022', 
    'T023', 'T024', 'T029', 'T030', 'T031', 
    'T032', 'T033', 'T034', 'T037', 'T038', 
    'T039', 'T040', 'T041', 'T042', 'T046', 
    'T047', 'T048', 'T049', 'T058', 'T059', 
    'T060', 'T061', 'T062', 'T063', 'T065', 
    'T067', 'T068', 'T069', 'T070', 'T074', 
    'T075', 'T077', 'T078', 'T079', 'T080', 
    'T081', 'T082', 'T102', 'T103', 'T109', 
    'T114', 'T116', 'T121', 'T123', 'T125', 
    'T126', 'T127', 'T129', 'T130', 'T131', 
    'T169', 'T170', 'T171', 'T184', 'T185', 
    'T190', 'T191', 'T192', 'T195', 'T196', 
    'T197', 'T200', 'T201', 'T203'
]


class ConceptCorpusReader:
    
    def __init__(self, mimic3_dir, split, Y):
        self.csv_fname = Path(mimic3_dir) / f'{split}_{Y}.csv'
        # originally preprocessed data file, before clef format preprocessing
        self.umls_fname = Path(mimic3_dir) / f'{split}_{Y}_umls.txt'
        # json file containing umls entities/doc_sent e.g. dev_50_umls.txt, Y == full vs 50
        self.index = dict()
    
    def read_umls_file(self):
        with open(self.umls_fname) as rf:
            for line in tqdm(rf):
                line = line.strip()
                if not line:
                    continue
                line = json.loads(line)
                uid = list(line.keys())[0]
                doc_id, sent_id = list(map(int, uid.split("_")))
                if doc_id not in self.index:
                    self.index[doc_id] = dict()
                self.index[doc_id][sent_id] = [
                    ((item['s'], item['e']), item['umls_ents']) for item in line[uid]
                ]
        # modify to have threshold of max confidence score instead of all (confidence score [umls_ents, "### > threshold"])
    def concept_dfs(self):
        # dfs doc frequency, count of concepts/doc
        dfs = collections.Counter()
        for doc_id in self.index.keys():
            concepts = list()
            for sent_id in self.index[doc_id].keys():
                concepts.extend([concept[0] for item in self.index[doc_id][sent_id] for concept in item[1]])
            dfs += collections.Counter(concepts)
        return dfs


mimic3_dir = "data/mimic3"
Y = "full"

train_reader = ConceptCorpusReader(mimic3_dir, 'train', Y)
logger.info(f'Reading train annotated UMLS file for {Y} ...')
train_reader.read_umls_file()
logger.info(f'Counting concepts with documents ...')
train_dfs = train_reader.concept_dfs()
logger.info(f'No. of unique concepts in train before pruning: {len(train_dfs)}')  # ~90955
logger.info(f'Top-10 most common concepts in train before pruning: {train_dfs.most_common(100)}')

dev_reader = ConceptCorpusReader(mimic3_dir, 'dev', Y)
logger.info(f'Reading dev annotated UMLS file for {Y} ...')
dev_reader.read_umls_file()
logger.info(f'Counting concepts with documents ...')
dev_dfs = dev_reader.concept_dfs()
logger.info(f'No. of unique concepts in dev before pruning: {len(dev_dfs)}') # ~41992
logger.info(f'Top-10 most common concepts in dev before pruning: {dev_dfs.most_common(100)}')

test_reader = ConceptCorpusReader(mimic3_dir, 'test', Y)
logger.info(f'Reading test annotated UMLS file for {Y} ...')
test_reader.read_umls_file()
logger.info(f'Counting concepts with documents ...')
test_dfs = test_reader.concept_dfs()
logger.info(f'No. of unique concepts in test before pruning: {len(test_dfs)}') # ~51112
logger.info(f'Top-10 most common concepts in test before pruning: {test_dfs.most_common(100)}')

total_train_dev_test_dfs = train_dfs + dev_dfs + test_dfs
# ~93137
logger.info(f'No. of unique concepts in train + dev + test before pruning: {len(total_train_dev_test_dfs)}')

only_in_dev = set(dev_dfs.keys()) - train_dfs.keys()
only_in_test = set(test_dfs.keys()) - train_dfs.keys()
cuis_to_discard = only_in_dev.union(only_in_test)
# keep only entities that are found in train set

logger.info(f'No. of unique concepts in only in dev: {len(only_in_dev)}')
logger.info(f'No. of unique concepts in only in test: {len(only_in_test)}')

logger.info('Loading SciSpacy UmlsEntityLinker ...')
linker = UmlsEntityLinker(name='scispacy_linker')


min_count, max_count = 5, 40000
logger.info(f'Pruning concepts with counts outside {min_count, max_count}, '
            f'and the ones which are not unseen in train split, '
            f'and keeping only those with Semantic Types: {SEMANTIC_TYPES_TO_INCLUDE}')
# i.e. keep only the semantic types that correspond to possible ICD9 codes' semantic types

# sci spacy knowledge base obj == umls2020aa cui_to_entity is a dict, 3rd element of this tuple, idx 3 == tui

tuis = {
    tui for cui in train_dfs.keys() for tui in linker.kb.cui_to_entity[cui][3] 
    if tui in set(SEMANTIC_TYPES_TO_INCLUDE) 
}
for cui, count in list(train_dfs.items()):
    if count < min_count:
        cuis_to_discard.add(cui)
        continue
    if count > max_count:
        cuis_to_discard.add(cui)
        continue
    for tui in linker.kb.cui_to_entity[cui][3]:
        if tui not in tuis:
            cuis_to_discard.add(cui)
            break

logger.info(f'No. of unique concepts to discard: {len(cuis_to_discard)}')


# mapping ICD9 codes to snomed CT, ignore below


# https://download.nlm.nih.gov/umls/kss/mappings/ICD9CM_TO_SNOMEDCT/ICD9CM_DIAGNOSIS_MAP_202012.zip
# https://download.nlm.nih.gov/umls/kss/mappings/ICD9CM_TO_SNOMEDCT/ICD9CM_PROCEDURE_MAP_202012.zip

'''
ICD9CM (D) -> SNOMED

1-1 Maps    7,762 (65.5%)   69.8%
1-M Maps    3,387 (28.6%)   25.3%

ICD9CM (P) -> SNOMED

1-1 Maps    1,740 (47.7%)   59.8%
1-M Maps    507 (13.9%)     27.8%

'''

"""
rule-base model
input: doc as concepts
process: concepts found that are also in the icd9 concepts == labels
output: labels

tfidf + classifier
input: vectorized concepts found in each doc -->tfidf input features
linear classifier output to |unique labels in semantic_types_mimic|
output: labels

label cluster (optional add to baseline) by 
1) semantic types
2) 
"""