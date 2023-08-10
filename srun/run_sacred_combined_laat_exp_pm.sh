#!/bin/bash

# SLURM environment arguments
MYDATA=/netscratch/pokarats/ds
IMAGE=/netscratch/pokarats/nvcr.io_nvidia_pytorch_22.02-py3_neptune2.sqsh
NUM_CPUS=16
MEM="$2"GB

# variables for srun and python
for ver in 50 full
do
  for embedding in snomedbase snomednoex snomedcase4
  do
    desc="$ver cui $embedding"
    echo "Submitting: $desc experiment"
    if [[ "$ver" == "50" ]]; then
      echo "$ver version u and da is default 256"
      if [[ "$3" == "umls" ]]; then
        echo "w2v umls embedding type specified"
        if [[ "$4" == "prune" ]]; then
          echo "prune $3 input cuis to KGE $embedding embedding entities"
          srun -K -p $1 \
            --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
            --container-workdir="$(pwd)" \
            --container-image=$IMAGE \
            --job-name=laat_"$desc"_"$3"_"$4" \
            --cpus-per-task=$NUM_CPUS \
            --gpus=1 \
            --mem=$MEM \
            --nodes=1 \
            --mail-type=END,FAIL \
            --mail-user=noon.pokaratsiri@dfki.de \
          srun/install_pretask.sh python src/laat_exp.py with data_dir="$MYDATA" input_type=combined version="$ver" \
          embedding_type="$3" dr_params.prune_cui=True dr_params.vocab_fn=processed_full_"$3"_pruned.json \
          dr_params.cui_prune_file="$ver"_cuis_to_discard_"$embedding".pickle
        else
          echo "use all $3 input cuis as in baseline model without pruning"
          srun -K -p $1 \
            --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
            --container-workdir="$(pwd)" \
            --container-image=$IMAGE \
            --job-name=laat_"$desc"_"$3"_"$4" \
            --cpus-per-task=$NUM_CPUS \
            --gpus=1 \
            --mem=$MEM \
            --nodes=1 \
            --mail-type=END,FAIL \
            --mail-user=noon.pokaratsiri@dfki.de \
          srun/install_pretask.sh python src/laat_exp.py with data_dir="$MYDATA" input_type=combined version="$ver" \
          embedding_type="$3" dr_params.vocab_fn=processed_full_"$3"_pruned.json
        fi
      else
        srun -K -p $1 \
          --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
          --container-workdir="$(pwd)" \
          --container-image=$IMAGE \
          --job-name=laat_"$desc" \
          --cpus-per-task=$NUM_CPUS \
          --gpus=1 \
          --mem=$MEM \
          --nodes=1 \
          --mail-type=END,FAIL \
          --mail-user=noon.pokaratsiri@dfki.de \
        srun/install_pretask.sh python src/laat_exp.py with data_dir="$MYDATA" input_type=combined version="$ver" \
        embedding_type=$embedding dr_params.prune_cui=True dr_params.vocab_fn=processed_full_umls_pruned.json \
        dr_params.cui_prune_file="$ver"_cuis_to_discard_"$embedding".pickle laat_params.separate_encoder=True
      fi
    else
      echo "$ver version u and da need to be 512"
      if [[ "$3" == "umls" ]]; then
        echo "w2v umls embedding type specified"
        if [[ "$4" == "prune" ]]; then
          echo "prune $3 input cuis to KGE $embedding embedding entities"
          srun -K -p $1 \
            --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
            --container-workdir="$(pwd)" \
            --container-image=$IMAGE \
            --job-name=laat_"$desc"_"$3"_"$4" \
            --cpus-per-task=$NUM_CPUS \
            --gpus=1 \
            --mem=$MEM \
            --nodes=1 \
            --mail-type=END,FAIL \
            --mail-user=noon.pokaratsiri@dfki.de \
          srun/install_pretask.sh python src/laat_exp.py with data_dir="$MYDATA" input_type=combined version="$ver" \
          embedding_type="$3" dr_params.prune_cui=True dr_params.vocab_fn=processed_full_"$3"_pruned.json \
          dr_params.cui_prune_file="$ver"_cuis_to_discard_"$embedding".pickle laat_params.u=512 laat_params.da=512
        else
          echo "use all $3 input cuis as in baseline model without pruning"
          srun -K -p $1 \
            --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
            --container-workdir="$(pwd)" \
            --container-image=$IMAGE \
            --job-name=laat_"$desc"_"$3"_"$4" \
            --cpus-per-task=$NUM_CPUS \
            --gpus=1 \
            --mem=$MEM \
            --nodes=1 \
            --mail-type=END,FAIL \
            --mail-user=noon.pokaratsiri@dfki.de \
          srun/install_pretask.sh python src/laat_exp.py with data_dir="$MYDATA" input_type=combined version="$ver" \
          embedding_type="$3" dr_params.vocab_fn=processed_full_"$3"_pruned.json laat_params.u=512 laat_params.da=512
        fi
      else
        srun -K -p $1 \
          --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
          --container-workdir="$(pwd)" \
          --container-image=$IMAGE \
          --job-name=laat_"$desc" \
          --cpus-per-task=$NUM_CPUS \
          --gpus=1 \
          --mem=$MEM \
          --nodes=1 \
          --mail-type=END,FAIL \
          --mail-user=noon.pokaratsiri@dfki.de \
        srun/install_pretask.sh python src/laat_exp.py with data_dir="$MYDATA" input_type=combined version="$ver" \
        embedding_type=$embedding dr_params.prune_cui=True dr_params.vocab_fn=processed_full_umls_pruned.json \
        dr_params.cui_prune_file="$ver"_cuis_to_discard_"$embedding".pickle laat_params.u=512 laat_params.da=512 \
        laat_params.separate_encoder=True
      fi
    fi
    sleep 3
  done
  sleep 1
done
