#!/bin/sh

# SLURM environment arguments
SCISPACY_CACHE=/netscratch/pokarats/cache/scispacy
MYDATA=/netscratch/pokarats/ds
IMAGE=/netscratch/pokarats/nvcr.io_nvidia_pytorch_22.02-py3_base1.sqsh
NUM_CPUS=32
MEM_PER_CPU=8GB

# variables for srun and python
vers=full
mimic3_dir=$MYDATA/linked_data/$vers
split=test
cache=/netscratch/pokarats/cache/scispacy
pickle_file="$vers"_cuis_to_discard

srun -K -p batch \
  --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)"\
  --container-workdir="$(pwd)" \
  --container-image=$IMAGE \
  --cpus-per-task=$NUM_CPUS \
  --mem-per-cpu=$MEM_PER_CPU \
  --nodes=1 \
python src/baseline_exp.py \
  --data_dir $MYDATA \
  --mimic3_dir $mimic3_dir \
  --version $vers \
  --split $split \
  --model rule-based \
  --stacked False \
  --cache_dir $cache \
  --misc_pickle_file "$pickle_file".pickle \
  --add_name rule-based-None \
  --filename "$vers"_cuis_to_discard_None\
  --n_process $NUM_CPUS \


for ext in best all
do
  srun -K -p batch \
    --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)"\
    --container-workdir="$(pwd)" \
    --container-image=$IMAGE \
    --cpus-per-task=$NUM_CPUS \
    --mem-per-cpu=$MEM_PER_CPU \
    --nodes=1 \
  python src/baseline_exp.py \
    --data_dir $MYDATA \
    --mimic3_dir $mimic3_dir \
    --version $vers \
    --split $split \
    --model rule-based \
    --extension $ext \
    --stacked False \
    --cache_dir $cache \
    --misc_pickle_file "$pickle_file".pickle \
    --add_name rule-based-$ext \
    --filename "$vers"_cuis_to_discard_"$ext"\
    --n_process $NUM_CPUS
done

srun -K -p batch \
  --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)"\
  --container-workdir="$(pwd)" \
  --container-image=$IMAGE \
  --cpus-per-task=$NUM_CPUS \
  --mem-per-cpu=$MEM_PER_CPU \
  --nodes=1 \
python src/baseline_exp.py \
  --data_dir $MYDATA \
  --mimic3_dir $mimic3_dir \
  --version $vers \
  --split $split \
  --model tfidf \
  --stacked True \
  --cache_dir $cache \
  --misc_pickle_file "$pickle_file".pickle \
  --add_name tfidf \
  --filename "$vers"_cuis_to_discard_None \
  --n_process $NUM_CPUS \
