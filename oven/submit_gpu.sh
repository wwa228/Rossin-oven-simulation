#!/bin/bash

#SBATCH --partition=hawkgpu
#SBATCH -t 24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=expt
#SBATCH --gres=gpu:2

module load anaconda3

conda activate for_jax_gpu

# nvidia-smi

export LD_LIBRARY_PATH=/home/scp220/.conda/envs/for_jax_gpu/lib:$LD_LIBRARY_PATH

XLA_FLAGS="--xla_force_host_platform_device_count=20" python -m oven.pe_sshooting_event \
--products_train "R13SSR1, R13SSR4, R13NSR1" --products_test "R13SSR2" \
--msg "testing on gpu, transfer learning, added constant, decreased lr" \
--optimize 1 --iterations 1000 --lr 0.001 --recirculation 0 \
--atol 1e-6 --rtol 1e-4 --mxstep 10000 \
--id $SLURM_JOB_ID --partition $SLURM_JOB_PARTITION --cpus $SLURM_JOB_CPUS_PER_NODE

wait
exit
