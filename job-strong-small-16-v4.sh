#!/bin/bash
#SBATCH -A LRN031
#SBATCH -J HydraGNN-DebugRun
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 16
#SBATCH -S 1

ulimit -n 65536

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=0 
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1

## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA
env | grep ^NCCL

source /lustre/orion/cph161/world-shared/mlupopa/module-to-load-frontier.sh

source /lustre/orion/cph161/world-shared/mlupopa/max_conda_envs_frontier/bin/activate
conda activate hydragnn

export PYTHONPATH=/lustre/orion/cph161/world-shared/mlupopa/ADIOS_frontier/install/lib/python3.8/site-packages/:$PYTHONPATH

export PYTHONPATH=$PWD:$PYTHONPATH

set -x

export HYDRAGNN_TRACE_LEVEL=2

for MS in SMALL; do
for NN in 16; do
	srun -N$NN -n$((NN*8)) -c7 --gpus-per-task=1 --gpu-bind=closest \
		python -u ./examples/multidataset/train.py --multi_model_list="OC2020" --multi --num_epoch=1000 \
		--everyone --ddstore --log=exp-strongfull-$MS-$SLURM_JOB_ID-NN$NN --inputfile=${MS}_MTL.json \
		--num_samples 10000
    sleep 5
done
done
echo "Done."
