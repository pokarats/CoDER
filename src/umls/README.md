## UMLS CUI, TUI, ICD-9 Mapping

The `query_icd9_cuis.py` generates the `ICD9_umls2020aa` file, mapping the ICD-9 codes to CUIs, TUIs, and definitions 
extracted from **UMLS2020AA**. It is provided in the repo's `data` directory, but can also be obtained by running:  

`python src/umls/query_icd9_cuis.py --data_dir data`  

(**NOTE:** A UMLS Account is required, see [UTS Account Sign-Up](https://uts.nlm.nih.gov/uts/signup-login) and 
[UMLS Quickstart Guide](https://www.nlm.nih.gov/research/umls/quickstart.html) for more info.)

Prior to running the code, create an environment variable: UMLS_API_KEY in a `.env` file in the `/src/configs` 
directory.

The code in this directory is written by [Saadullah Amin](https://github.com/suamin).