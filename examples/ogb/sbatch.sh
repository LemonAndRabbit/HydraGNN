#!/bin/bash
#SBATCH -A m4133
#SBATCH -J HydraGNN
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -N 1


## remove write permission for others in terms of newly created files and dirs
umask 002

## Module
module reset
module load pytorch/2.0.1

HYDRAGNN_DIR=/global/cfs/cdirs/m4133/c8l/HydraGNN_fix_writer
module use -a /global/cfs/cdirs/m4133/jyc/perlmutter/sw/modulefiles
module load hydragnn/pytorch2.0.1-v2
echo "python:" `which python`
export PYTHONPATH=$HYDRAGNN_DIR:$PYTHONPATH

## Envs
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MPICH_GPU_SUPPORT_ENABLED=0

export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1

set -x

srun -N1 -n4 -c32 --ntasks-per-node=4 --gpus-per-task=1 \
    python train_gap.py --adios gap \
    2>&1 | tee run-ogb-gap-adios.log

set +x
