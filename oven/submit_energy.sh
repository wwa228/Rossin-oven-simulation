#!/bin/bash

#SBATCH -p hawkcpu
#SBATCH -t 48:00:00
#SBATCH -n 1
#SBATCH -J expt
#SBATCH --cpus-per-task 15


module load miniconda3
conda activate for_ipopt

# https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time


XLA_FLAGS="--xla_force_host_platform_device_count=5" python -m oven.energy_optimization \
--msg "testing fwd(rev) hessian without recirculation" \
--recirculation 0 --atol 1e-8 --rtol 1e-6 --mxstep 10000 --tol 1e-3 --iterations 1000 --samples 5 \
--id $SLURM_JOB_ID --partition $SLURM_JOB_PARTITION --cpus $SLURM_JOB_CPUS_PER_NODE 


# --msg "lower tolerance, updated discretized points in tau dimensions (ntimes), multiple samples, each zone recirculation, high tolerance, changed ode tol" 

wait
exit
