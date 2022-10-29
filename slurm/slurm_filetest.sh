#!/bin/bash
#SBATCH --job-name=file-2node-test
#SBATCH --ntasks=2

#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=300MB
#SBATCH --time=0:01:00

python filecontext.py