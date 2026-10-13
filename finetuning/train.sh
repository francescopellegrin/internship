#!/bin/bash
#SBATCH --job-name=gemma27b_ft
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16        # 16 CPU sono sufficienti per 1 GPU
#SBATCH --gres=gpu:1              # 1 A100 è sufficiente per 27B QLoRA!
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_F01_260
#SBATCH --qos=normal

export WORK="/leonardo_work/AIFAC_F01_260"
PROJECT_DIR="$WORK/ft_codes"

# Offline mode is essential
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=$WORK/.cache/huggingface

# Fix comune su Leonardo: sposta la cache di Triton su WORK per evitare errori di permessi
export TRITON_CACHE_DIR=$WORK/.cache/triton
mkdir -p $TRITON_CACHE_DIR

module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.1
module load cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.2

# ATTENZIONE: Qui attivo ft_env (l'ambiente che stiamo usando ora)
source $WORK/ft_env/bin/activate

export LD_LIBRARY_PATH="$WORK/ft_env/lib:$LD_LIBRARY_PATH"

echo "Job running on $(hostname)"
echo "GPUs available:"
nvidia-smi

echo "Changing directory to: $PROJECT_DIR"
cd $PROJECT_DIR || { echo "Directory not found! Check PROJECT_DIR"; exit 1; }

# Esegue lo script Python creato poco fa
python gemma-27b.py