#!/bin/bash

srun -K -p RTXA6000-MLT \
  --container-image=/netscratch/pokarats/nvcr.io_nvidia_pytorch_22.02-py3_"$1".sqsh \
  --container-save=/netscratch/pokarats/nvcr.io_nvidia_pytorch_22.02-py3_"$2".sqsh \
  --pty /bin/bash