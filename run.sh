#!/bin/bash
#SBATCH --job-name=mav       # Job name
#SBATCH -c 4                               # 4 cores per GPU
#SBATCH -t 03-00:00 # 0.7 min per batch for 4 GPUs, so 1008 batches per day, can change this to 00-12:00 for 12 hours
#SBATCH -o logs/output_%j.log              # Output log file (%j will be replaced with the job ID)
#SBATCH -e logs/error_%j.log               # Error log file 
#SBATCH -p seas_gpu                        # Partition name
#SBATCH --account=hankyang_lab
#SBATCH --ntasks-per-node=4                # Run 4 tasks (one per GPU)
#SBATCH --mem=64GB                         # Memory per node (16GB per GPU)
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4 # Request 4 H100 GPUs

# Load necessary modules
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh

# Activate virtual environment
conda activate mav

module load cudnn/9.5.1.17_cuda12-fasrc01 
module load cuda/12.4.1-fasrc01

# export MASTER_ADDR=$(hostname -i)   # ✅ or manually: export MASTER_ADDR=10.31.147.40
# export MASTER_PORT=29500

# # These are good already
# export NCCL_SOCKET_IFNAME=ib0
# export GLOO_SOCKET_IFNAME=em4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



# Roughly 8 GB of memory per GPU is enough for Gemma 4b with 2048 max_new_tokens and batch size 4 and 2 generations
accelerate launch --multi_gpu --num_processes=4 src/main.py

