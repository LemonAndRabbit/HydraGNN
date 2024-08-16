#!/bin/bash
#SBATCH -A CPH161
#SBATCH -J strong
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 2048
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

#export PYTHONPATH=/dir/to/HydraGNN:$PYTHONPATH
export PYTHONPATH=/lustre/orion/world-shared/cph161/jyc/frontier/HydraGNN-gb24:$PYTHONPATH

# declare -a modelcases=("SMALL" "MEDIUM" "LARGE") #"XLARGE"  "XXLARGE")
declare -a modelcases=("LARGE") #"XLARGE"  "XXLARGE")

set -x

export HYDRAGNN_TRACE_LEVEL=2

for MS in XSMALL SMALL MEDIUM LARGE; do
for NN in $((SLURM_JOB_NUM_NODES)); do
	srun -N$NN -n$((NN*8)) -c7 --gpus-per-task=1 --gpu-bind=closest \
		python -u ./examples/multidataset/train.py --multi_model_list="OC2020" --multi --num_epoch=4 \
		--everyone --ddstore --log=exp-strongfull-$MS-$SLURM_JOB_ID-NN$NN --inputfile=${MS}_MTL.json
    sleep 5
done
done
echo "Done."
