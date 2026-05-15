#!/bin/bash
#SBATCH -n 10
#SBATCH --mem-per-cpu=32g
#SBATCH --time=01:10:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g

module purge
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.1.1          
module load python/3.10.13
module load eth_proxy       

cd "$SLURM_SUBMIT_DIR"

python3 CNN.py
