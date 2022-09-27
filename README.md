## WIP

While WIP, ignore any instructions in here until further notice. Meanwhile, the only things being updated at the moment
are the todo's/developer's notes

Usage
-----

1. Prepare data `cd data` and follow the instructions.

2. Preprocess the data `python src/utils/preprocess.py` (see `srun/preprocess.sh` for cluster run).

3. Train word embeddings `python src/utils/word_embeddings.py` (see `srun/word_embeddings.sh` for cluster run).

### TODO's + Notes
LAAT experiments

1. finish prepare data for running LAAT experiments
   1. plan for modular refactorability to accomodate BOTH standard MIMIC-III (.csv files) and CLEF file format for 
      processing input text and label IDs
   2. all future source codes re: LAAT model runs should be sacred+neptune integrated
   3. sacred ex.add_artifact is buggy with slurm and Neptune connection; don't do this

2. finish train/eval codes, accommodate multi-gpu runs
3. run and debug prn

GNN

1. GNN-XML paper --> dig into how they initialize GIN and build their graph/Adj Matrix??

EVAL

1. revisit HEMKIT


CLEAN UP

1. go back from the beginning, refactor pipeline to accommodate standard and clef-formatted input
2. clean up comments, DOCUMENTATION!!!
3. update readme files for each step
