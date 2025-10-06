#!/bin/bash

#SBATCH -p hawkcpu
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -J expt
#SBATCH --cpus-per-task 20

module load miniconda3
conda activate for_ipopt

# https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time


XLA_FLAGS="--xla_force_host_platform_device_count=20" python -m oven.parameter_estimation \
--products_train "R13SSR1, R13NSR1, R13SSR4" --products_test "R13SSR2" \
--msg "Testing forward mode sensitivities" \
--optimize 1 --iterations 20 --lr 0.001 --recirculation 1 \
--atol 1e-6 --rtol 1e-4 --mxstep 10000 \
--id $SLURM_JOB_ID --partition $SLURM_JOB_PARTITION --cpus $SLURM_JOB_CPUS_PER_NODE

# 
# --load_dir "log/R13SSR4/2024-12-27 20:47:09.835171" # optimized parameters
# transfer learning, added constant heat transfer rate for constant region, changed initial guess for heat transfer coefficient constant, MAP estimation of initial moisture
# XLA_FLAGS="--xla_force_host_platform_device_count=20" python -m oven.pe_sshooting_event --products_train "R13SSR1, R13SSR2, R13SSR4, R13NSR1" --msg "mean + sigma * normal (different for all expts), added expectation samples" --iterations 1000 --id $SLURM_JOB_ID --lr 0.001 --expectation 4

# interactive version
# srun -p hawkcpu -n 1 --cpus-per-task 10 -t 120 --pty /bin/bash

wait
exit
