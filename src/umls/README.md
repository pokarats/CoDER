## UMLS CUI, TUI, ICD-9 Mapping

The `query_icd9_cuis.py` generates the `ICD9_umls2020aa` file, mapping the ICD-9 codes to CUIs, TUIs, and definitions 
extracted from **UMLS2020AA**. It is provided in the repo's `data` directory, but can also be obtained by running:  

`python src/umls/query_icd9_cuis.py --data_dir data`  

(**NOTE:** A UMLS Account is required, see [UTS Account Sign-Up](https://uts.nlm.nih.gov/uts/signup-login) and 
[UMLS Quickstart Guide](https://www.nlm.nih.gov/research/umls/quickstart.html) for more info.)

Prior to running the code, create an environment variable: UMLS_API_KEY in a `.env` file in the `/src/configs` 
directory.

Some scripts in this directory are written by [Saadullah Amin](https://github.com/suamin).

## SNOMED KGE for UMLS CUIs

Following the steps in `src/umls/kg_preprocess` to obtain the necessary files for training KGE from SNOMED CT ontology
with [DGL KE](https://github.com/awslabs/dgl-ke).

We experiment with both the pre-processed files obtained from [Benchmark and Best Practices for Biomedical Knowledge 
Graph Embeddings](https://github.com/dchang56/snomed_kge) (case4 train/valid/test splits) and our own pre-processed
splits following their steps. We use [DGL KE](https://github.com/awslabs/dgl-ke) for training KGE.

We check the coverage in terms of the % of CUIs in the MIMIC-III dataset from our baseline experiments that are also 
present in SNOMED CT and obtain the embedding weights for those CUI entities (.npy file) with this script: 
`mimic_snomed_kge.py`.