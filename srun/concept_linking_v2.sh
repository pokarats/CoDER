#!/bin/sh

# SLURM environment arguments
IMAGE=/netscratch/enroot/dlcc_pytorch_20.07.sqsh
NUM_CPUS=32
MEM_PER_CPU=8GB

# Change anaconda environment
ENV=multirescnn
export python=/netscratch/samin/dev/miniconda3/envs/$ENV/bin/python3.7

mimic3_dir=data/mimic3
split=test_50
model=en_core_sci_lg
export SCISPACY_CACHE=/netscratch/samin/cache/scispacy
batch_size=4096


srun -K -p batch \
  --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
  --container-workdir=`pwd` \
  --container-image=$IMAGE \
  --cpus-per-task=$NUM_CPUS \
  --mem-per-cpu=$MEM_PER_CPU \
  --nodes=1 \
  $python utils/concept_linking.py \
  --mimic3_dir $mimic3_dir \
  --split_file $split \
  --scispacy_model_name $model \
  --n_process 24 \
  --batch_size $batch_size
