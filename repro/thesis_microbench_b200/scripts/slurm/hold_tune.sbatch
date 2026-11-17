#!/bin/bash
#SBATCH -J coreai_libraries_nccl-microbench.b200hold_tune
#SBATCH -A coreai_libraries_nccl
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 01:30:00
#SBATCH -o /home/siyshen/workspace/b200-microbench/slurm/holdtune_%j.out
# Persistent hold: keep the 2 nodes allocated; benchmark steps attach with srun --jobid=$JOBID --overlap.
scontrol show hostnames $SLURM_JOB_NODELIST
sleep infinity
