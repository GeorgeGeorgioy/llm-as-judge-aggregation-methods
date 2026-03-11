#!/bin/bash
#BSUB -J LLM_gen
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o stdout/cyclegan_train_%J.out
#BSUB -e stderr/cyclegan_train_%J.err

# Load conda
cd /work3/s233559
export HF_HOME=/work3/s233559/.cache/huggingface
unset TRANSFORMERS_CACHE
module load cuda/12.8
source /work3/s233559/.venv/bin/activate

# Go to project folder
cd /work3/s233559/Thesis

export CUDA_VISIBLE_DEVICES=1
 # or
export CUDA_VISIBLE_DEVICES=0

# for online inference

export CUDA_VISIBLE_DEVICES=0
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Πες γεια στα ελληνικά σε μία πρόταση."}
    ],
    "max_tokens": 40,
    "temperature": 0
  }'

#for ofline inference 
#export VLLM_USE_V1=0