#! /bin/sh

#SBATCH --job-name=make_pythia
#SBATCH --output=./logs/make_pythia_logs.out
#SBATCH --error=./errors/make_pythia_errors.out
#SBATCH --partition=killable
#SBATCH --time=1000 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total

python make_model_pythia.py --original_model_name pythia-70m-deduped