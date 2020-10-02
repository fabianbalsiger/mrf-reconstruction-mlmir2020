#!/usr/bin/env bash

# Usage: $ run.sh -d=0
# or: $ run.sh -device=0

for i in "$@"
do
case $i in
    -d=*|--device=*)
    DEVICE="${i#*=}"
    shift # past argument=value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument test_without no value
    ;;
    *)
          # unknown option
    ;;
esac
done

if [ -z ${DEVICE+x} ]; then
    echo "argument -d or -device specifying CUDA visible device required!"
    exit 1
fi

echo "CUDA=${DEVICE}"

CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=".." python "./training.py" --config_file="./configs_final/invfwdbwd_lr=1e-4_bs=50.yaml"
CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=".." python "./training_baseline.py" --config_file="./configs_final/cohen_lr=1e-3_bs=50.yaml"
CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=".." python "./training_baseline.py" --config_file="./configs_final/hoppe_lr=1e-4_bs=50.yaml"
CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=".." python "./training_baseline.py" --config_file="./configs_final/invbwd_lr=1e-4_bs=50.yaml"
CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=".." python "./training_baseline.py" --config_file="./configs_final/oksuz_lr=1e-4_bs=50.yaml"
CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=".." python "./training_baseline.py" --config_file="./configs_final/song_lr=1e-3_bs=200.yaml"

exit 0