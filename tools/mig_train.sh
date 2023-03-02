#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
GPUS=${GPUS:-1}
#GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GPUS_PER_NODE=$GPUS
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}

if [ ${PARTITION} == "a100" ]
then
    MEM=30000
    MEM=$(($MEM * $GPUS))
    GPUS_PER_NODE="a100.10gb:$GPUS"
else
    MEM=24000
    MEM=$(($MEM * $GPUS))
fi
PY_ARGS=${@:4}  # Any arguments from the forth one are captured by this

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE}\
    --ntasks=${GPUS} \
    --mem=${MEM} \
    --ntasks-per-node=${GPUS} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}
