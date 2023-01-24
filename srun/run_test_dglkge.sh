#!/bin/sh

# SLURM environment arguments
export DGLBACKEND=pytorch
export CUDA_VISIBLE_DEVICES=0
MYDATA=/netscratch/pokarats/ds
OUT_DIR=/netscratch/pokarats/scratch/dglke_test
IMAGE=/netscratch/pokarats/nvcr.io_nvidia_pytorch_22.03-py3_dglkgesource.sqsh
NUM_CPUS=8
MEM=32GB

# variables for srun and python

vers="test_dglke_base"
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
dglke_train --model_name TransE_l2 --batch_size 2 --log_interval 10 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 100 --gamma 20.0 --lr 0.14 --batch_size_eval 1 \
        --test -adv -a 1.0 --gpu $CUDA_VISIBLE_DEVICES --mix_cpu_gpu --max_step 100 --dataset 'raw_udd_test' \
        --format 'raw_udd_hrt' --data_path "$MYDATA"/raw_udd_2 --delimiter ';' --save_path $OUT_DIR \
        --data_files train.tsv valid.tsv test.tsv