#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/netscratch/pokarats/tmp/install_done_${SLURM_JOBID}"
PYVERSION="$(python -V 2>&1 | grep -o '3\.\d*')"
if [[ $SLURM_LOCALID == 0 ]]; then
    if [[ $PYVERSION < 3.8 ]]; then
      # only install this for python version older than 3.8
      echo "$PYVERSION needs to install pickle5"
      pip install pickle5
    else
      echo "$PYVERSION no need to install pickle5, already supported"
    fi
    # install python-dotenv regardless
    echo "Will install python-dotenv"
    pip install python-dotenv
    # Tell other tasks we are done installing
    touch "${DONEFILE}"
else
# Wait until packages are installed
    while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi
# This runs your wrapped command
"$@"
