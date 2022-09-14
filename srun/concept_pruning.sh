#!/bin/sh

# SLURM environment arguments
IMAGE=/netscratch/enroot/dlcc_pytorch_20.07.sqsh
NUM_CPUS=32
MEM_PER_CPU=8GB

# Change anaconda environment
ENV=multirescnn
export python=/netscratch/pokarats/anaconda3/envs/$ENV/bin/python3.7

mimic3_dir=data/mimic3
split=train
split_file=train_50
model=en_core_sci_lg
linker=scispacy_linker
cache=/netscratch/pokarats/cache/scispacy
pickle_file=cuis_to_discard_50
dict_pickle=pruned_partitions_dfs_dict_50
batch_size=4096


srun -K -p batch \
  --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
  --container-workdir=`pwd` \
  --container-image=$IMAGE \
  --cpus-per-task=$NUM_CPUS \
  --mem-per-cpu=$MEM \
  --nodes=1 \
  $python src/utils/concepts_pruning.py \
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
