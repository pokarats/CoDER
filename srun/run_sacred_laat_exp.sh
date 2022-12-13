#!/bin/sh

# SLURM environment arguments
MYDATA=/netscratch/pokarats/ds
IMAGE=/netscratch/pokarats/nvcr.io_nvidia_pytorch_22.02-py3_neptune2.sqsh
NUM_CPUS=16
MEM=64GB

# variables for srun and python

vers="top50 cui snomed case4"
echo "Submitting version: $vers input_type"

srun -K -p $1 \
  --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  --container-workdir="$(pwd)" \
  --container-image=$IMAGE \
  --job-name=laat_"$vers" \
  --cpus-per-task=$NUM_CPUS \
  --gpus=1 \
  --mem=$MEM \
  --nodes=1 \
  --mail-type=END,FAIL \
  --mail-user=noon.pokaratsiri@dfki.de \
srun/install_pretask.sh python src/laat_exp.py with data_dir="$MYDATA" input_type=umls embedding_type=snomedcase4 \
dr_params.prune_cui=True dr_params.cui_prune_file=50_cuis_to_discard_snomedcase4.pickle