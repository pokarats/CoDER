#!/bin/sh

# SLURM environment arguments
export SCISPACY_CACHE=/netscratch/pokarats/cache/scispacy
MYDATA=/netscratch/pokarats/ds
IMAGE=/netscratch/pokarats/nvcr.io_nvidia_pytorch_22.02-py3_base1.sqsh
NUM_CPUS=32
MEM_PER_CPU=16GB

# variables for srun and python
vers=full
mimic3_dir=$MYDATA/linked_data/"$vers"
split=test
cache=/netscratch/pokarats/cache/scispacy
pickle_file="$vers"_cuis_to_discard

srun -K -p RTXA6000-MLT \
  --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  --container-workdir="$(pwd)" \
  --container-image=$IMAGE \
  --job-name=tfidf_"$vers"_extra_logreg_only \
  --cpus-per-task=$NUM_CPUS \
  --mem-per-cpu=$MEM_PER_CPU \
  --nodes=1 \
  --mail-type=END,FAIL \
  --mail-user=noon.pokaratsiri@dfki.de \
python src/baseline_exp.py \
  --data_dir $MYDATA \
  --mimic3_dir $mimic3_dir \
  --version $vers \
  --split $split \
  --model tfidf \
  --to_skip sgd_1 sgd_2 svm_1 svm_2 \
  --extra \
  --cache_dir $cache \
  --misc_pickle_file "$pickle_file".pickle \
  --add_name tfidf_extra_logreg_only \
  --filename "$vers"_cuis_to_discard_None \
  --n_process $NUM_CPUS