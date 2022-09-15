#!/bin/sh

# SLURM environment arguments
SCISPACY_CACHE=/netscratch/pokarats/cache/scispacy
MYDATA=/netscratch/pokarats/ds
IMAGE=/netscratch/pokarats/nvcr.io_nvidia_pytorch_22.02-py3_base1.sqsh
NUM_CPUS=32
MEM_PER_CPU=8GB

# variables for srun and python
vers=full
mimic3_dir="$MYDATA"/linked_data/$vers
split=train
split_file=train_$vers
model=en_core_sci_lg
linker=scispacy_linker
cache=/netscratch/pokarats/cache/scispacy
sem_file="$MYDATA"/mimic3/semantic_types_mimic.txt
pickle_file=cuis_to_discard
dict_pickle=pruned_partitions_dfs_dict
batch_size=4096


srun -K -p batch \
  --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)"\
  --container-workdir="$(pwd)" \
  --container-image=$IMAGE \
  --cpus-per-task=$NUM_CPUS \
  --mem-per-cpu=$MEM_PER_CPU \
  --nodes=1 \
python src/utils/concepts_pruning.py \
  --mimic3_dir $mimic3_dir \
  --version $vers \
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

