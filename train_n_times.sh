#! bin/bash

for ((i=0 ;i<3 ;i++ ))
do
   GPUS=1 tools/slurm_train.sh ${1:-a100} ${2:-ocsampler} $3 --validate
done
