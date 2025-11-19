#!/bin/bash
#SBATCH --job-name=Modeling_ants
#SBATCH --account=xujie
#SBATCH --qos=xujie
#SBATCH --mail-type=NONE
#SBATCH --mail-user=NONE
#SBATCH --cpus-per-task=16        
#SBATCH --nodes=1   
#SBATCH --mem=70g
#SBATCH --time=50:00:00
#SBATCH --output=/blue/xujie/fanz/logs/task_%j.out
#SBATCH --partition=hpg-default
pwd; hostname; date

module load conda
conda activate test_env

echo "Launching job for script"
python eval_classic_registration.py --method ants --out_csv /orange/xujie/fanz/Brain_images/ants/checkpoints/eval_metrics.csv