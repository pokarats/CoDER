#!/bin/sh

# SLURM environment arguments
IMAGE=/netscratch/pokarats/nvcr.io_nvidia_pytorch_22.02-py3_base1.sqsh
NUM_CPUS=8
MEM_PER_CPU=8GB

# Change anaconda environment

mimic3_dir=$MYDATA/linked_data/50
split=train
split_file=train_50
model=en_core_sci_lg
linker=scispacy_linker
cache=/netscratch/pokarats/cache/scispacy
pickle_file=cuis_to_discard_50
dict_pickle=pruned_partitions_dfs_dict_50
batch_size=4096



srun -K -p batch \
  --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,`pwd`:`pwd` \
  --container-workdir=`pwd` \
  --container-image=$IMAGE \
  --cpus-per-task=$NUM_CPUS \
  --mem-per-cpu=$MEM \
  --nodes=1 \
  python src/utils/concepts_pruning.py \
  --mimic3_dir $mimic3_dir \
  --version 50\
  --split $split \
  --split_file $split_file \
  --scispacy_model_name $model \
  --linker_name $linker \
  --cache_dir $cache \
  --pickle_file $pickle_file \
  --dict_pickle_file $dict_pickle \
  --n_process $NUM_CPUS \
  --batch_size $batch_size
