#!/bin/bash

#SBATCH -p hawkcpu,engc,eng,enge
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -J expt
#SBATCH --cpus-per-task 10

module load miniconda3

conda activate for_ipopt

# https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time


XLA_FLAGS="--xla_force_host_platform_device_count=5" python -m oven.inverse_mapping \
--msg "testing with multiple samples" \
--optimize 1 --iterations 3000 --lr 0.001 --recirculation 0 --samples 4 \
--atol 1e-6 --rtol 1e-4 --mxstep 10000 \
--id $SLURM_JOB_ID --partition $SLURM_JOB_PARTITION --cpus $SLURM_JOB_CPUS_PER_NODE


wait
exit
