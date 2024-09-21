#! /bin/sh

#SBATCH --job-name=make_pythia_small
#SBATCH --output=./logs/make_pythia_small_logs.out
#SBATCH --error=./errors/make_pythia_small_errors.out
#SBATCH --partition=killable
#SBATCH --time=1000 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total

python pythia_small.py