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
- torch==1.12.0
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
- dgl>=0.5.0,<=0.9.1
- dglke

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

We conducted preliminary experiments on a rule-based and non-deep-learning models with the **UMLS CUI** input type in 
order to get a low baseline performance threshold for this type of input. 

Non-DL results as shown in [Baseline Results](/res/README.md#non-deep-learning-models) for LR and SVM models with 
**UMLS CUIS** input appear comparable to reported results in [Vu et al (2020)](https://arxiv.org/abs/2007.06351) 
with **text input** type.

Based on these promising preliminary experiments, we compared **UMLS CUI** input type with **text input** on one of the
current top-performing DL models for MIMIC-III ICD Coding Task: 
[Label Attention Model for ICD Coding from Clinical Text (LAAT)](https://arxiv.org/abs/2007.06351). Results are reported
in the [LAAT Baseline Results](/res/README.md#laat-results)

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

### Non Deep Learning (DL) Models

Using TFIDF for feature extraction, we experimented with the following Non-DL Models:

- **Logistic Regression (LR)**
  - 1-gram
  - up to 2-gram
- **Support Vector Machine (SVM)**
  - 1-gram
  - up to 2-gram
- **Stochastic Gradient Descent (SGD)**
  - 1-gram
  - up to 2-gram
- **Stacked** (`1-2-gram` only; only for top-50 version)
  - LR + SGD + SVM


We also experimented with `1-gram` only vs `1-2-gram` tokenization range options for feature extraction in these 
models, see [TfidfVectorizer Doc](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). 
Models and TFIDF Vectorization implemented with [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)


### DL Baseline: LAAT

We implemented the [LAAT Model](https://arxiv.org/abs/2007.06351) following their implementation code in 
[LAAT GitHub](https://github.com/aehrc/LAAT), with modifications from
[P4Q_Guttmann_SCT_Coding](https://github.com/suamin/P4Q_Guttmann_SCT_Coding). We used all hyperameters as reported in
their publication.


## Proposed Extensions

From the baseline results, the CUI input models are within a few percentage points in precision score (both the top-50
and full versions). The F1 scores are also within 10-15 percentage points of the text-input models. Recall scores show
the largest difference between the text and CUI models.

We experimented with Knowledge Graph Embeddings (KGE) on the existing baseline LAAT model for the CUI input type. Our
goal is to investigate if KGE pre-training and its relational information improve ICD-9 classification using CUIs
as input in comparison to the baseline W2V embedding. This aims to answer one of our research questions: whether graph/
hierachical/relational information in the embedding (e.g. KGE) space is beneficial to this task.

Another extension is to investigate graph structure in the encoder part of the pipeline. We experimented with using a
GNN model instead of the LSTM-based encoder in the LAAT baseline model.

For results of the experiments in this section, please see [Extension Results](/res/README.md#extension-results)

### KGE

We follow [Benchmark and Best Practices for Biomedical Knowledge Graph Embeddings](https://aclanthology.org/2020.bionlp-1.18)
and trained KGE from the [UMLS SNOMED-CT Knowledge Base](https://www.nlm.nih.gov/research/umls/index.html) to represent
CUIs in the MIMIC dataset. Training is done using [DGL-KE Library](https://doi.org/10.1145/3397271.3401172).

We experimented with 2 pre-training approaches:

- case4 embeddings from the data released by [Benchmark and Best Practices for Biomedical Knowledge Graph Embeddings](https://aclanthology.org/2020.bionlp-1.18)
- base embeddings from our own installation of the [UMLS SNOMED-CT Knowledge Base](https://www.nlm.nih.gov/research/umls/index.html)

Preliminary results show that case4 KGE achieve higher P,R, and F1 scores than base embeddings, using the baseline
LAAT encoder for classification. Since not all CUIs in the MIMIC dataset have relations in the KGE pre-training,
we had to prune out CUIs without relations. It seems that case4 KGE represent **~50%** and base KGE only **35%** of the 
CUIS in the MIMIC dataset. This reduced representation correlates with the lower performance by the base KGE.

### Combined Text and CUI Input Model

We also experimented with training the baseline LAAT model with both types of input: text and CUI. We used W2V embeddings
for the text input type and the case4 KGE for the CUI input type. The idea is to investigate if adding hierachical or 
relational information in the KGE-represented CUIs to the text-input data improves model's performance.

### GNN

From the results of the Combined and KGE models, the text-input LAAT model shows the highest P, R, and F1 scores.
We hypothesized that the LSTM-based encoder in the baseline LAAT model favors sequential embeddings and input type.
As a result, any gain in hierachical or relational information from KGE pretraining would have been negated by the
encoder model. 

To verify this, we experimented with a GNN-based encoder model with pre-trained KGE.


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
  


