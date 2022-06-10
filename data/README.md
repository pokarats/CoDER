**Note**: Adopted from [MultiResCNN](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network/tree/master/data)

Our process of preparing data just follows [CAML](https://github.com/jamesmullenbach/caml-mimic) with slight modifications. 
For example, we add sentence splitting, BERT support and fasttext embedding.
Put the files of MIMIC III into the dir as below:
```
data
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
|   ICD9_descriptions
|   ICD9_umls2020aa
└───mimic3/
|   |   NOTEEVENTS.csv
|   |   DIAGNOSES_ICD.csv
|   |   PROCEDURES_ICD.csv
|   |   *_hadm_ids.csv (get from CAML)
```
Download `D_ICD_PROCEDURES.csv.gz` and `D_ICD_DIAGNOSES.csv.gz` from MIMIC-III database and unzip them to get `*.csv` files.


`ICD9_umls2020aa` contains the codes CUIs, TUIs, and definitions extracted from UMLS2020AA.

## Download Linked Data

Please also download the SciSpacy entity linked data.

```
# for full split (download all files under mimic3/)
https://drive.google.com/drive/folders/188KS5fphBLqJM_-LaSCes49sl4IWaJ9o?usp=sharing

# for top-50 split (download all files under mimic3/)
https://drive.google.com/drive/folders/1ujrpnAiz2voOW8r9eQEn2H6iu608Fl-q?usp=sharing
```
