#!/bin/bash
#SBATCH --job-name=igem_task_classifier
#SBATCH --output=bert_inference_%j.out
#SBATCH --error=bert_inference_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread

#SBATCH --partition=gpu_p5
#SBATCH --account=duf@a100
#SBATCH --constraint=a100

module purge
module load pytorch-gpu/py3/2.6.0

export GIT_PYTHON_REFRESH=quiet # to hush git warnings

echo "Using model from: $DSDIR/HuggingFace_Models/bert-base-uncased" 

# srun python -u 2_bert_task_classification_training.py
srun python -u 3_bert_inference.py