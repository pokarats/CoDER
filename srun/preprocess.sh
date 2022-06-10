#!/bin/sh

# SLURM environment arguments
IMAGE=/netscratch/enroot/dlcc_pytorch_20.07.sqsh

# Change anaconda environment
ENV=multirescnn
export python=/netscratch/samin/dev/miniconda3/envs/$ENV/bin/python3.7

srun -K \
  --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
  --container-workdir=`pwd` \
  --container-image=$IMAGE \
  --nodes=1 \
  $python src/utils/preprocess.py
