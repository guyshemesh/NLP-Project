#! /bin/sh

#SBATCH --job-name=run_task_test
#SBATCH --output=/home/joberant/NLP_2324b/adishani1/main.out # redirect stdout
#SBATCH --error=/home/joberant/NLP_2324b/adishani1/main.err # redirect stderr
#SBATCH --partition=studentkillable
#SBATCH --time=1000 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total


python run_task.py