# Obtaining UMLS Data Files 

The required files listed below can be obtained from the 
[UMLS Knowledge Sources](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html). 

We use the "2020AA" release files and subset for the SNOMED_CT_US Knowledge Source.

Required files:
 - MRCONSO.RFF
 - MRREL.RRF
 - MRSTY.RRF

We store these files in `data/umls`

## UMLS RRF Files Preprocessing

Steps:

1. run `./src/umls/kg_preprocess/preprocess_umls.sh`
   1. this generates the following files in `data/umls`:
      - `snomed_active_concepts.txt`
      - `snomed_active_relations.txt`
      - `snomed_semantic_types.txt`
2. run `./src/umls/kg_preprocess/kg_transductive.sh`
   1. this generates the necessary train/dev/test files for training KGE based on SNOMED_CT Knowldge Base
   2. All generated data files are in `data/umls`
      - **NOTE:** `semantic_info.csv` in this directory will be used later for SEMANTIC TYPE/GROUP info for GNNDataset


### File Description

For information related to each file type, please refer to the 
[UMLS Reference Manual](https://www.ncbi.nlm.nih.gov/books/NBK9685/#ch03.sec3.3)