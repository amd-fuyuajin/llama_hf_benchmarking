#!/bin/bash

# setup your HUGGING_FACE_HUB_TOKEN in secrets.env
if [ -f "./secrets.env" ]; then
    source ./secrets.env
fi
export CONDA_PREFIX=/home/amd/miniforge3/envs/llama_benchmark_env
export HF_HOME=/home/amd/dataset/hf_home
export INPUT_DIR=/home/amd/dataset/wikipedia/prompts
export RESULTS_ROOT_DIR=/home/amd/workspace/llm_benchmark_results
export CPU_NAME="TurinC-llama3.1-GCP"
export LD_PRELOAD=${CONDA_PREFIX}/lib/libgomp.so
#export LD_PRELOAD=${CONDA_PREFIX}/lib/libgomp.so:${CONDA_PREFIX}/lib/libtcmalloc.so
#export VLLM_CPU_KVCACHE_SPACE=40


#export ONEDNN_VERBOSE=1
