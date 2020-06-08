#!/bin/bash
#SBATCH --job-name=dark          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=25G                # total memory per node
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # TIGER
# SBATCH --gres=gpu:tesla_v100:1  # ADROIT
# SBATCH --reservation=hackathon  # ADROIT, REMOVE THIS AFTER THE HACKATHON

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load anaconda3/2020.2
conda activate dark-env-v2

python -u _run_graph_net_nv.py
#kernprof -l _run_graph_net_nv.py
#nsys profile -o profile_${SLURM_JOBID} --stats=false -t cuda,nvtx,osrt,cublas python -u _run_graph_net_nv.py
