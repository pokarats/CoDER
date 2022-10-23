## Data preparation

We experiment with two types of input data: text tokens and UMLS CUI tokens. The instructions below will prepare the 
data files for **both** types of input.

### MIMIC-III-full and MIMIC-III-top50 experiments

- We follow [MultiResCNN](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network)
with slight modifications to accommodate for different versions of library packages. 
- To obtain the MIMIC-III database, follow 
[PhysioNet access instructions.](https://mimic.mit.edu/)
- ID files and ICD9 descriptions (`*_hadm_ids.csv` and `ICD9_descriptions` can be obtained 
from [CAML](https://github.com/jamesmullenbach/caml-mimic))
- Put the files of MIMIC-III into the `data` directory as follows:

```
data
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
|   ICD9_descriptions
└───mimic3/
         |   NOTEEVENTS.csv
         |   DIAGNOSES_ICD.csv
         |   PROCEDURES_ICD.csv
         |   *_hadm_ids.csv (id files; get from CAML)
```

#### Steps

1. Generate the train/valid/test sets (filenames: `train.csv`, `valid.csv`, and `test.csv`) for both the `full` and 
`top50` versions of the dataset using this script: `python src/utils/preprocess.py`
2. Generate UMLS CUIs linked data for the train/valid/test data files using this script:
   (See command line options in [src/utils/concept_linking.py](../src/utils/concept_linking.py) for more details)

```angular2html
python src/utils/concept_linking.py \
  --mimic3_dir data/mimic3 \
  --split_file train_50 \
  --scispacy_model_name en_core_sci_lg \
  --cache_dir [Path to SciSpacy cache directory or set the SCISPACY_CACHE environment variable] \
  --n_process [number of cpus to use] \
  --batch_size 4096
```

*NOTE:* You will have to repeat the script above for all the splits required.
   - Alternatively, you can download the SciSpacy linked data from the links below:
     - [for 'full' verion](https://drive.google.com/drive/folders/188KS5fphBLqJM_-LaSCes49sl4IWaJ9o?usp=sharing)
     - [for 'top-50' verion](https://drive.google.com/drive/folders/1ujrpnAiz2voOW8r9eQEn2H6iu608Fl-q?usp=sharing)
3. Create a directory named `linked_data`
4. Create the `full` and `50` version directories within the `data/mimic3` and `data/linked_data` directories and move 
the `[train/valid/test]_[full/50].csv` and `[train/valid/test]_[full/50]_umls.txt` to their respective folders.
5. Also move the `vocab.csv` and `disch_full.csv` files to the `data/mimic3/full` directory.
6. Your `data` folder structure should resemble the following:

```
data
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
|   ICD9_descriptions
|   ICD9_umls2020aa
└───linked_data/
         └───50/
               |  dev_50_umls.txt
               |  test_50_umls.txt
               |  train_50_umls.txt
         └───full/
               |  dev_full_umls.txt
               |  test_full_umls.txt
               |  train_full_umls.txt
└───mimic3/
         |   NOTEEVENTS.csv
         |   DIAGNOSES_ICD.csv
         |   PROCEDURES_ICD.csv
         |   *_hadm_ids.csv (id files; get from CAML)
         |   TOP_50_CODES.csv
         |   ...some other files generated during Step 1
         └───50/
                  |  dev_50.csv
                  |  test_50.csv
                  |  train_50.csv
         └───full/
                  |  dev_full.csv
                  |  test_full.csv
                  |  train_full.csv
                  |  vocab.csv
                  |  disch_full.csv
```

6. The `ICD9_umls2020aa` file contains the codes CUIs, TUIs, and definitions extracted from UMLS2020AA. It is provided in 
this repo `data`, but can also be obtained by running `python src/umls/query_icd9_cuis.py --data_dir data`. 
(UMLS Account is required, see [UTS Account Sign-Up](https://uts.nlm.nih.gov/uts/signup-login) and 
[UMLS Quickstart Guide](https://www.nlm.nih.gov/research/umls/quickstart.html) for more info)

### UMLS Concepts (CUIs) Pruning

Just as we pre-process the text input to remove too rare and too frequent word tokens. We prune concepts (CUIs) in
each sample that are too rare and too frequent. We determine the minimum and maximum frequency thresholds as follows:

- normalized max threshold is > **1500x/1 million**
- normalized min threshold is is **0.1x/1 million**

(TODO: make this an adjustable hyperparameter)

We also prune out CUIs that do not belong to the Semantic Types (TUIs) of the ICD9 codes of the MIMIC-III dataset and
CUIs in the `dev` or `test` sets not seen in `train` set. (i.e. no zero-shot CUIs). 

#### Pruning Script

An example run command below is for pruning the `train` split of the `top-50` version. See 
[src/utils/concepts_pruning.py](../src/utils/concepts_pruning.py) for command line options.

```angular2html
python src/utils/concepts_pruning.py \
  --mimic3_dir data/linked_data/50 \
  --version 50 \
  --split train \
  --split_file train_50 \
  --scispacy_model_name en_core_sci_lg \
  --linker_name scispacy_linker \
  --cache_dir scratch/cache/scispacy \
  --semantic_type_file data/mimic3/semantic_types_mimic.txt \
  --pickle_file cuis_to_discard \
  --dict_pickle_file pruned_partitions_dfs_dict \

```
A set of discarded CUIs and dictionary of pruned partitions (test/val/train) are saved in the respective version's 
directory (or as specified) after running the script as pickle files. 

**IMPORTANT:** the `{full/50}_cuis_to_discard.pickle` file is **used in other scripts**. If you specify a non-default 
filename, make sure to use the **same filename** when a pruning file option is to be specified. The subsequent scripts 
assume the file is in **the version directory (e.g. `data/mimic3/full/`) where it is generated.**

The Semantic Types of the MIMIC-III dataset is provided at 
[data/mimic3/semantic_types_mimic.txt](mimic3/semantic_types_mimic.txt)

More information about [Semantic Types and Groups](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/documentation/SemanticTypesAndGroups.html)

### Additional Pre-Processing for Rule-Based Models

