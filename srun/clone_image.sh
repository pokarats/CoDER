#!/bin/bash

srun \
  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.02-py3.sqsh \
  --container-save=/netscratch/pokarats/nvcr.io_nvidia_pytorch_22.02-py3_"$1".sqsh \
  --pty /bin/bash