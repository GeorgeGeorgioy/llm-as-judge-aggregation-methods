#!/bin/bash
#BSUB -J LLM_gen
#BSUB -q gpul40s
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=28GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
#BSUB -o stdout/LLM_gen_copy_%J.out
#BSUB -e stderr/LLM_gen_copy_%J.err

# Load conda
#cd /work3/s233559
#export HF_HOME=/work3/s233559/.cache/huggingface
#unset TRANSFORMERS_CACHE
set -e

module purge
module load cuda/12.8.1

source /work3/s233559/.venv/bin/activate

#export CUDA_VISIBLE_DEVICES=0

echo "JOB INFO"
echo "LSB_JOBID=$LSB_JOBID"
echo "LSB_HOSTS=$LSB_HOSTS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
hostname

echo "NVIDIA-SMI"
nvidia-smi

echo "PYTORCH CUDA TEST"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"

# Go to project folder
cd /work3/s233559/Thesis

#export CUDA_VISIBLE_DEVICES=1
 # or
#export CUDA_VISIBLE_DEVICES=0

# for online inference
# python test_n_i.py \
#   --model_name qwen7 \
#   --role judge \
#   --promptroot /work3/s233559/Thesis/prompts/judge 

python testOnSh.py \
 --promptroot /work3/s233559/Thesis2/prompts/generator/generator_HaluEval_favoritism.jsonl\
 --model_name LLAMA_8 \
 --results_dir /work3/s233559/Thesis2/results \
 --aggregation_method oneshot \
 --dataset_name arena \
 --role generator  