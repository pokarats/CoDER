#!/bin/sh

# SLURM environment arguments
MYDATA=/netscratch/pokarats/ds
IMAGE=/netscratch/pokarats/nvcr.io_nvidia_pytorch_22.02-py3_neptune2.sqsh
NUM_CPUS=8
MEM_PER_CPU=4GB

# variables for srun and python
for vers in "50" full
do
  echo "Submitting version: $vers base"

  srun -K -p RTXA6000-MLT \
    --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
    --container-workdir="$(pwd)" \
    --container-image=$IMAGE \
    --job-name=w2v_base_"$vers" \
    --cpus-per-task=$NUM_CPUS \
    --mem-per-cpu=$MEM_PER_CPU \
    --nodes=1 \
    --mail-type=END,FAIL \
    --mail-user=noon.pokaratsiri@dfki.de \
  python src/utils/sacred_word_embeddings.py with data_dir="$MYDATA" 'version=$vers' n_workers="$NUM_CPUS"

  sleep 2

  for cfg in cui cui_pruned
  do
    echo "Submitting version: $vers $cfg"

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
    python src/utils/sacred_word_embeddings.py with $cfg data_dir="$MYDATA" 'version=$vers' n_workers="$NUM_CPUS"

    sleep 1
  done
done
