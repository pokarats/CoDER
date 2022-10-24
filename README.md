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

We conducted preliminary experiments on a rule-based and non-deep-learning models with the **UMLS CUI** input type in 
order to get a low baseline performance threshold for this type of input. 

Non DL results as shown in [Baseline Results](#non-deep-learning-models-results) for LR and SVM models with 
**UMLS CUIS** input appear comparable to reported results in [Vu et al (2020)](https://arxiv.org/abs/2007.06351) 
with **text input** type.

Based on these promising preliminary experiments, we compared **UMLS CUI** input type with **text input** on one of the
current top-performing DL models for MIMIC-III ICD Coding Task: 
[Label Attention Model for ICD Coding from Clinical Text (LAAT)](https://arxiv.org/abs/2007.06351). Results are reported
[below.](#laat-results)

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

- Logistic Regression (LR)
  - 1-gram
  - up to 2-gram
- Support Vector Machine (SVM)
  - 1-gram
  - up to 2-gram
- Stochastic Gradient Descent (SGD)
  - 1-gram
  - up to 2-gram
- Stacked (`1-2-gram` only; only for top-50 version)
  - LR + SGD + SVM


We also experimented with `1-gram` only vs `1-2-gram` tokenization range options for feature extraction in these 
models, see [TfidfVectorizer Doc](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). 
Models and TFIDF Vectorization implemented with [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)


### DL Baseline: LAAT

We implemented the [LAAT Model](https://arxiv.org/abs/2007.06351) following their implementation code in 
[LAAT GitHub](https://github.com/aehrc/LAAT), with modifications from
[P4Q_Guttmann_SCT_Coding](https://github.com/suamin/P4Q_Guttmann_SCT_Coding). We used all hyperameters as reported in
their publication.

## Baseline Results
### Rule-Based Results

#### Top-50 Version
| Model               | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro) | Acc (Macro) | Acc (Micro) |
|---------------------|:---------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|
| No Extension (None) | **45.02** |   18.88   |   19.58    |   26.60    |    12.57    |    15.34    |
| ALL                 |   16.86   | **26.54** |   18.11    |   20.62    |    9.89     |    11.50    |
| Best                |   40.71   |   20.34   | **23.75**  | **27.13**  |  **13.07**  |  **15.69**  |

(09/16/2022)

#### Full Version
| Model               | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro) | Acc (Macro) | Acc (Micro) |
|---------------------|:---------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|
| No Extension (None) |   8.83    |   14.47   |    2.19    |   10.97    |    1.13     |    5.80     |
| ALL                 |   8.83    |   14.47   |    2.19    |   10.97    |    1.13     |    5.80     |
| Best                |   8.83    |   14.47   |    2.19    |   10.97    |    1.13     |    5.80     |

(09/16/2022)

**NOTE:** that for the Full Version, _the extension criteria (finding < 1 ICD label corresponding to input CUIs)_ were
never met. Hence identical results: i.e. No extensions were triggered.

### Non Deep Learning Models Results

#### Top-50 Version

| Model               | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro) | AUC (Macro) | AUC (Micro) |  P@5  |
|---------------------|:---------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|:-----:|
| LR[^lrparam] 1-gram |   75.60   |   66.95   |   66.55    |   71.01    |    92.79    |    94.6     | 67.28 |
| LR[^lrparam] 2-gram |   68.75   |   47.38   |   50.55    |   56.10    |    86.16    |    89.26    | 57.50 |
| SGD 1-gram          |   65.70   |   50.64   |    9.87    |   57.20    |    89.84    |    98.56    | 80.91 |
| SGD 2-gram          |   64.93   |   36.59   |    6.25    |    46.8    |    84.38    |    97.74    | 73.90 |
| SVM 1-gram          |           |           |            |            |             |             |       |
| SVM 2-gram          |           |           |            |            |             |             |       |
| Stacked 2-gram      |           |           |            |            |             |             |       |
(09/17/2022)  


[^lrparam]: Used default settings for [scikit-learn 1.0.2](https://scikit-learn.org/1.0/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regression#sklearn.linear_model.LogisticRegression) 
except `class_weight=='balanced'`, `C==1000` and `max_iter==1000`.


#### Full Version

| Model          | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro) | AUC (Macro) | AUC (Micro) |  P@5  |
|----------------|:---------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|:-----:|
| LR 1-gram      |   75.60   |   66.95   |   66.55    |   71.01    |    92.79    |    94.6     | 67.28 |
| LR 2-gram      |   68.75   |   47.38   |   50.55    |   56.10    |    86.16    |    89.26    | 57.50 |
| SGD 1-gram     |   65.70   |   50.64   |    9.87    |   57.20    |    89.84    |    98.56    | 80.91 |
| SGD 2-gram     |   64.93   |   36.59   |    6.25    |    46.8    |    84.38    |    97.74    | 73.90 |
| SVM 1-gram     |           |           |            |            |             |             |       |
| SVM 2-gram     |           |           |            |            |             |             |       |

(09/17/2022, 09/20/2022)

### LAAT Results

Our implementation reproduced the results for the **Top-50** and **Full** versions of the dataset for the **text input** 
type as reported in [Vu et al. (2020)](https://arxiv.org/abs/2007.06351). We report the results from using **UMLS CUIs** 
as input tokens below.

| Model      | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro) | AUC (Macro) | AUC (Micro) |  P@5  |
|------------|:---------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|:-----:|
| Text Top50 |   75.60   |   66.95   |   66.55    |   71.01    |    92.79    |    94.6     | 67.28 |
| CUI Top50  |   68.75   |   47.38   |   50.55    |   56.10    |    86.16    |    89.26    | 57.50 |
| Text Full  |   65.70   |   50.64   |    9.87    |   57.20    |    89.84    |    98.56    | 80.91 |
| CUI Full   |   64.93   |   36.59   |    6.25    |    46.8    |    84.38    |    97.74    | 73.90 |

(Results from 10/15/2022 run on [Git Commit@36dda76](https://github.com/pokarats/CoDER/commit/36dda76d28e2a9606688016a770d0bf1129104fe))

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
  


