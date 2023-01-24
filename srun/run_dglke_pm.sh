#!/bin/sh

# SLURM environment arguments
export DGLBACKEND=pytorch
export CUDA_VISIBLE_DEVICES=0
MYDATA=/netscratch/pokarats/ds
IMAGE=/netscratch/pokarats/nvcr.io_nvidia_pytorch_22.03-py3_dglkgesource.sqsh
NUM_CPUS=8
MEM="$2"GB

# variables for srun and python
for neg in 64
do
  vers="transel2_base_$neg"
  mkdir -p scratch/"$vers"
  echo "dir made at $vers"
  for neval in 90000
  do
    echo "Submitting version: $vers"
    echo "neg eval: $neval"
    srun -K -p $1 \
      --container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
      --container-workdir="$(pwd)" \
      --container-image=$IMAGE \
      --job-name=dglke_"$vers"_"$neval" \
      --cpus-per-task=$NUM_CPUS \
      --gpus=1 \
      --mem=$MEM \
      --nodes=1 \
      --mail-type=END,FAIL \
      --mail-user=noon.pokaratsiri@dfki.de \
      --output=scratch/"$vers"/"$vers"_"$neval".out \
    dglke_train --model_name TransE_l2 --batch_size 1024 --log_interval 1000 --neg_sample_size $neg \
            --regularization_coef 1e-07 --hidden_dim 100 --gamma 10 --lr 0.1 --batch_size_eval 1000 \
            --test -adv --valid --gpu $CUDA_VISIBLE_DEVICES --mix_cpu_gpu --max_step 60000 \
	    --neg_sample_size_eval $neval \
            --dataset case4_"$neval" --format raw_udd_hrt --data_path "$MYDATA"/umls --save_path scratch/$vers \
            --data_files train.tsv dev.tsv test.tsv
    sleep 2
  done
done
