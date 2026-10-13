#!/bin/bash
#SBATCH --job-name=inf_gemma-3-27b_0_ft
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=AIFAC_F01_260
#SBATCH --partition=boost_usr_prod
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# ==========================================
# ENVIRONMENT SETUP
# ==========================================

# 1. Force immediate flushing
export PYTHONUNBUFFERED=1

# 2. Offline Mode (CRITICAL)
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 3. Load modules
module purge
module load python/3.11
module load cuda/12.1

# 4. Activate environment
source $WORK/ft_env/bin/activate

# ==========================================
# EXECUTION
# ==========================================

echo "Job started on $(hostname) at $(date)"

# Ci spostiamo nella cartella dove risiede lo script per comodità
cd $WORK/inference_scripts

echo "Working directory: $(pwd)"

# Esecuzione script
python -u gemma-3-27b_0_ft.py

echo "Job finished at $(date)"