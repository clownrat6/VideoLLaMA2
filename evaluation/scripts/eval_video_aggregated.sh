#!/bin/bash

# Environment Variables
MODEL_PATH=${1:-"work_dirs/videollama2qwen2.5_vllava/finetune_siglip_tcv35_7b_16f"}
BENCHMARKS=${2:-"mvbench,videomme"}
SAVE_DIR=${3:-"eval_output"}

ARG_WORLD_SIZE=${4:-1}
ARG_NPROC_PER_NODE=${5:-8}

ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16667
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi


echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MODEL_PATH: $MODEL_PATH"
echo "BENCHMARKS: $BENCHMARKS"
echo "SAVE_DIR: $SAVE_DIR"


DATA_ROOT=/mnt/damovl/EVAL_BENCH/VIDEO
declare -A DATA_ROOTS
DATA_ROOTS["mvbench"]="$DATA_ROOT/mvbench"
DATA_ROOTS["videomme"]="$DATA_ROOT/videomme"


IFS=',' read -ra BENCHMARK_LIST <<< "$BENCHMARKS"
for BENCHMARK in "${BENCHMARK_LIST[@]}"; do
    DATA_ROOT=${DATA_ROOTS[$BENCHMARK]}
    if [ -z "$DATA_ROOT" ]; then
        echo "Error: Data root for benchmark '$BENCHMARK' not defined."
        continue
    fi
    torchrun --nnodes $WORLD_SIZE \
        --nproc_per_node $NPROC_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --node_rank $RANK \
        evaluation/evaluate.py \
        --model_path ${MODEL_PATH} \
        --benchmark ${BENCHMARK} \
        --data_root ${DATA_ROOT} \
        --save_path ${SAVE_DIR}/${MODEL_PATH##*/}/${BENCHMARK}.json
done
