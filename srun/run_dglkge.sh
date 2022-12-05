#!/bin/sh

# SLURM environment arguments
export DGLBACKEND=pytorch
export CUDA_VISIBLE_DEVICES=0
MYDATA=/netscratch/pokarats/ds
IMAGE=/netscratch/pokarats/nvcr.io_nvidia_pytorch_22.03-py3_dglkgesource.sqsh
NUM_CPUS=8
MEM=32GB

# variables for srun and python

vers="snomed_ct_transel2_base"
echo "Submitting version: $vers kge"

srun -K -p RTX3090-MLT \
  --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  --container-workdir="$(pwd)" \
  --container-image=$IMAGE \
  --job-name=dglke_"$vers" \
  --cpus-per-task=$NUM_CPUS \
  --gpus=1 \
  --mem=$MEM \
  --nodes=1 \
  --mail-type=END,FAIL \
  --mail-user=noon.pokaratsiri@dfki.de \
dglke_train --model_name TransE_l2 --batch_size 1024 --log_interval 100 --neg_sample_size 60 \
        --regularization_coef 1e-09 --hidden_dim 100 --gamma 8 --lr 5e-4 --batch_size_eval 16 \
        --test -adv -a 1.0 --gpu $CUDA_VISIBLE_DEVICES --mix_cpu_gpu --valid --max_step 2000 --dataset 'raw_udd_umls' \
        --format 'raw_udd_hrt' --data_path "$MYDATA"/umls --delimiter '\t' --save_path scratch/umls_kge \
        --data_files train.tsv dev.tsv test.tsv