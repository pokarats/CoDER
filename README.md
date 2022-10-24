# WIP: Graph Structures in Knowledge-Aware Multi-Label Classification for Healthcare Data
This project contains code and documentation for all experiments related to thesis work on exploring graph structures
in knowledge-aware ICD coding task on the MIMIC-III dataset. The task is set up as a multi-label classification
problem.

**WIP:DISCLAIMER:** 
- While Work is in progress (WIP) not all information is up-to-date or accurate.
- Ignore any instructions in here until further notice.
- the only things being updated at the moment are the todo's/developer's notes

## Requirements

- python>=3.7
- pickle5==0.0.12 (not required if python >= 3.8)
- torch==1.11.0
- scikit-learn==0.24.0
- numpy==1.22.2
- scipy==1.6.3
- pandas==1.3.5
- tqdm>=4.62.3,<5.0.0
- nltk==3.7
- gensim==4.1.2
- scispacy==0.5.1
- spacy==3.4.1
- transformers==4.22.0
- python-dotenv==0.21.0
- sacred==0.8.2
- neptune-client==0.16.8
- neptune-sacred==0.10.0

Run `pip install -r requirements.txt` to install the required libraries

Unless you already have the `en_core_sci_lg` model for the specified version (0.5.1) of SciSpacy, make sure to 
run:  

`pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz` 
to download the model.   

If you use a different version of SciSpacy, change `0.5.1` in the script above to your version.

## Data preparation

We experiment with two types of input data: text tokens and UMLS CUI tokens. The instructions in the 
[data directory](data/README.md) will prepare the data files for both types of input.

## Pre-trained Word Embeddings 

Following [Vu et al. (2020)](https://github.com/aehrc/LAAT) and 
[Li and Yu (2020)](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network), we use 
[GENSIM](https://radimrehurek.com/gensim/models/word2vec.html) to train a [Word2Vec Model](https://arxiv.org/abs/1310.4546) 
using the entire MIMIC-III discharge summary data. 

### Scripts

The following scripts generate word embeddings for the text and CUI input types respectively:

With `default` configs and the specified data_dir param:

`python src/utils/sacred_word_embeddings.py with data_dir=data`

With `cui` configs and the specified data_dir param:  

`python src/utils/sacred_word_embeddings.py with cui data_dir=data`

See [src/utils/sacred_word_embeddings.py](src/utils/sacred_word_embeddings.py) for 
[Sacred Configs](https://sacred.readthedocs.io/en/stable/configuration.html) options.

Before running these scripts, make sure your data directory organization follows the scheme described in 
[data/README](data/README.md)

The scripts save trained model files (`.model`, `.npy`, and `.json`) in the `data/[mimic3|linked_data]/model/` 
directories.

There is an option in our code to support [fastText subword embeddings](https://arxiv.org/abs/1607.04606), however we 
have not tested our experiments with this type of embeddings.

## Baseline Experiments

We conducted pre-liminary experiments on the following baseline models.

### Rule-Based

The most basic baseline for the UMLS CUIs input type. The model takes as input for each sample CUI tokens and output
ICD-9 codes that also map to CUIs in the input sample.

There are **3 variations** to this model:

- **No-Extension (None):** if no ICD-9 codes are found, predicted labels are left blank
- **Extension == All:** if no ICD-9 codes are found, output ALL ICD-9 codes belonging to the Semantic Type(s) of the CUIs 
in the input
- **Extension == Best:** if no ICD-9 codes are found, output only the ICD-9 code of the Semantic Type(s) whose definition 
has the 
highest similarity score (among scores above a specified threshold, e.g. 0.7) to the CUIs in the input

### Non Deep Learning Models

Using TFIDF for feature embedding, we experimented with the following models:

- Logistic Regression (LR)
  - 1-gram
  - up to 2-gram
- Support Vector Machine (SVM)
  - 1-gram
  - up to 2-gram
- Stochastic Gradient
- Descent (SGD)
  - 1-gram
  - up to 2-gram
- Stacked (up to 2-gram only)

### LAAT

We implemented the [Vu et al (2020) Label Attention Model for ICD Coding from Clinical Text (LAAT)](https://arxiv.org/abs/2007.06351) 
following the implementation codes in [LAAT GitHub](https://github.com/aehrc/LAAT) and [P4Q_Guttmann_SCT_Coding](https://github.com/suamin/P4Q_Guttmann_SCT_Coding).

### Baseline Results
#### Rule-Based
#### Non Deep Learning Models
#### LAAT

Our implementation reproduced the results for the **Top-50** and **Full** versions of the dataset for the text input type
as reported in [Vu et al. (2020)](https://arxiv.org/abs/2007.06351). We also experimented with using UMLS CUIs as input
tokens.

| Model      | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro) | AUC (Macro) | AUC (Micro) |  P@5  |
|------------|:---------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|:-----:|
| Text Top50 |   75.60   |   66.95   |   66.55    |   71.01    |    92.79    |    94.6     | 67.28 |
| CUI Top50  |   68.75   |   47.38   |   50.55    |   56.10    |    86.16    |    89.26    | 57.50 |
| Text Full  |   65.70   |   50.64   |    9.87    |   57.20    |    89.84    |    98.56    | 80.91 |
| CUI Full   |   64.93   |   36.59   |    6.25    |    46.8    |    84.38    |    97.74    | 73.90 |

(Results from 10/15/2022 run on git commit: 36dda76d)

## Proposed Extensions
### KGE
### GNN
### Extension Results

## TODO's + Notes

1. add CLEF file format capability for running baseline and future experiments for 
   processing input text and label IDs
   1. possible to experiment with other datasets that have been clef-formatted
2. accommodate multi-gpu runs?
3. KGE, GNN implementation
4. Hierachical eval metrics, need ICD-9 Tree structure for HEMKIT


- KGE 
  - instead of w2v, try this for the CUI-input model with the LAAT model
  - [SNOMED KGE](https://github.com/dchang56/snomed_kge)-->use [DGL implementation](https://github.com/awslabs/dgl-ke)
- GNN
  - [GNN-XML paper](http://arxiv.org/abs/2012.05860) --> dig into how they initialize GIN and build their graph/Adj Matrix??
  - [DFGN paper](https://aclanthology.org/P19-1617) --> for graph/Adj Matrix building
  - [GCT paper](https://ojs.aaai.org/index.php/AAAI/article/view/5400) --> for graph/Adj Matrix idea
  - [HyperCore](https://aclanthology.org/2020.acl-main.282) --> for how to aggregate graph with doc text
  


