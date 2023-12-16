#!/bin/sh
#SBATCH --job-name=PytonMultiprocessing
#SBATCH --cpus-per-task=8   # Number of logical CPUs
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=3GB # Memory per logical CPU
#SBATCH -o ./Slurm_output/%A_%a.out # STDOUT

module load Python/3.7.3-gimkl-2018b
python run_collapsar.py
