#!/bin/bash
#SBATCH --job-name=Modeling_brainVIT
#SBATCH --account=xujie
#SBATCH --qos=xujie
#SBATCH --mail-type=NONE
#SBATCH --mail-user=NONE
#SBATCH --cpus-per-gpu=8         
#SBATCH --nodes=1   
#SBATCH --gpus=1
#SBATCH --mem=70g
#SBATCH --time=180:00:00
#SBATCH --output=/blue/xujie/fanz/logs/task_%j.out
#SBATCH --partition=hpg-b200
pwd; hostname; date

module load conda
conda activate test_env

echo "Launching job for script"
python train.py --num_workers 4 --batch_size 8 --checkpoint_dir /orange/xujie/fanz/Brain_images/cnn/checkpoints


