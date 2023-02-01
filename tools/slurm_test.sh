#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
CHECKPOINT=$4
GPUS=${GPUS:-1}
#GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GPUS_PER_NODE=$GPUS
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
PY_ARGS=${@:5}  # Arguments starting from the fifth one are captured
if [ ${PARTITION} == "a100" ]
then
    MEM=30000
    GPUS_PER_NODE="a100:$GPUS"
else
    MEM=12000
fi
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --mem=${MEM} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" ${PY_ARGS}
