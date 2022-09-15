#!/bin/sh

# SLURM environment arguments
SCISPACY_CACHE=/netscratch/pokarats/cache/scispacy
MYDATA=/netscratch/pokarats/ds
IMAGE=/netscratch/enroot/dlcc_pytorch_20.07.sqsh
NUM_CPUS=32
MEM_PER_CPU=8GB

# Change anaconda environment
ENV=multirescnn
export python=/netscratch/pokarats/anaconda3/envs/$ENV/bin/python

mimic3_dir=$MYDATA/linked_data/"$1"
split=train
split_file=train_"$1"
model=en_core_sci_lg
linker=scispacy_linker
cache=/netscratch/pokarats/cache/scispacy
sem_file=$MYDATA/mimic3/semantic_types_mimic.txt
pickle_file=cuis_to_discard_"$1"
dict_pickle=pruned_partitions_dfs_dict_"$1"
batch_size=4096



srun -K -p batch \
  --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)"\
  --container-workdir="$(pwd)" \
  --container-image=$IMAGE \
  --cpus-per-task=$NUM_CPUS \
  --mem-per-cpu=$MEM_PER_CPU \
  --nodes=1 \
$python src/utils/concepts_pruning.py \
  --mimic3_dir $mimic3_dir \
  --version "$1"\
  --split $split \
  --split_file $split_file \
  --scispacy_model_name $model \
  --linker_name $linker \
  --cache_dir $cache \
  --semantic_type_file $sem_file \
  --pickle_file $pickle_file \
  --dict_pickle_file $dict_pickle \
  --n_process $NUM_CPUS \
  --batch_size $batch_size \
$@
