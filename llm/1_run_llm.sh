#!/bin/bash
#SBATCH --job-name=igem_llama_70b
#SBATCH --output=llm_%j.out
#SBATCH --error=llm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4              # Request 4 GPUs
#SBATCH --cpus-per-task=32        # 4 GPUs * 8 cores = 32 
#SBATCH --time=04:00:00           # hours limit
#SBATCH --hint=nomultithread      # Hyperthreading disabled

#A 100 CONFIGURATION - use the specific A100 partition and our account name
#SBATCH --partition=gpu_p5
#SBATCH --account=duf@a100
#SBATCH --constraint=a100

# Clean modules
module purge

# Load environment
module load pytorch-gpu/py3/2.2.0

# Ensure DSDIR is accessible
echo "Using model from: $DSDIR/HuggingFace_Models/meta-llama/Meta-Llama-3-70B-Instruct"

# Run the python script
# srun python -u 2_llm_extraction_with_username.py
srun python -u 4_llm_task_classification.py