#!/bin/sh

# SLURM environment arguments
IMAGE=/netscratch/enroot/dlcc_pytorch_20.07.sqsh
NUM_CPUS=32
MEM_PER_CPU=8GB

# Change anaconda environment
ENV=multirescnn
export python=/netscratch/samin/dev/miniconda3/envs/$ENV/bin/python3.7

mimic3_dir=data/mimic3
split=train_50
model=en_core_sci_lg
cache=/netscratch/samin/cache/scispacy
batch_size=4096


srun -K -p batch \
  --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
  --container-workdir=`pwd` \
  --container-image=$IMAGE \
  --cpus-per-task=$NUM_CPUS \
  --mem-per-cpu=$MEM \
  --nodes=1 \
  $python src/utils/concept_linking.py \
  --mimic3_dir $mimic3_dir \
  --split_file $split \
  --scispacy_model_name $model \
  --cache_dir $cache \
  --n_process $NUM_CPUS \
  --batch_size $batch_size
