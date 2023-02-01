#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
CHECKPOINT=$4
GPUS=${GPUS:-1}
#GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GPUS_PER_NODE=$GPUS
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=${@:5}  # Arguments starting from the fifth one are captured
SRUN_ARGS=${SRUN_ARGS:-""}
MEM=25000

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --mem=${MEM} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/customed_test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" ${PY_ARGS}
