#!/bin/sh

# SLURM environment arguments
MYDATA=/netscratch/pokarats/ds
IMAGE=/netscratch/pokarats/nvcr.io_nvidia_pytorch_22.02-py3_neptune2.sqsh
NUM_CPUS=4
MEM_PER_CPU=8GB

# variables for srun and python

vers=text
echo "Submitting version: $vers input_type"

srun -K -p RTXA6000-MLT \
  --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  --container-workdir="$(pwd)" \
  --container-image=$IMAGE \
  --job-name=w2v_"$vers" \
  --cpus-per-task=$NUM_CPUS \
  --mem-per-cpu=$MEM_PER_CPU \
  --nodes=1 \
  --mail-type=END,FAIL \
  --mail-user=noon.pokaratsiri@dfki.de \
srun/install_pretask.sh python src/utils/sacred_word_embeddings.py with data_dir="$MYDATA"

sleep 2

vers=umls
cfg=cui
echo "Submitting version: $vers $cfg input_type"

srun -K -p RTXA6000-MLT \
  --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  --container-workdir="$(pwd)" \
  --container-image=$IMAGE \
  --job-name=w2v_"$cfg"_"$vers" \
  --cpus-per-task=$NUM_CPUS \
  --mem-per-cpu=$MEM_PER_CPU \
  --nodes=1 \
  --mail-type=END,FAIL \
  --mail-user=noon.pokaratsiri@dfki.de \
srun/install_pretask.sh python src/utils/sacred_word_embeddings.py with $cfg data_dir="$MYDATA"