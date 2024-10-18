#! /bin/sh

#SBATCH --job-name=run
#SBATCH --output=./logs/run.out
#SBATCH --error=./errors/run.out
#SBATCH --partition=killable
#SBATCH --time=1000 # max time (minutes)
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1


python run_task.py