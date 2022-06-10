#!/bin/sh

IMAGE=/netscratch/enroot/dlcc_pytorch_20.07.sqsh

export python=/netscratch/samin/dev/miniconda3/envs/multirescnn/bin/python3.7

srun -K -p RTXA6000-MLT \
    --container-mounts=/netscratch:/netscratch,/ds:/ds \
    --container-workdir=/netscratch/samin/projects/CoDER \
    --container-image=$IMAGE \
    --pty bash
