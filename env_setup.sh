#!/bin/bash

export CONDA_PREFIX=/home/amd/anaconda3/envs/vllm_cpu_env
export HF_HOME=/home/amd/dataset/hf_home
export HUGGING_FACE_HUB_TOKEN=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
export INPUT_DIR=/home/amd/dataset/wikipedia/prompts
export RESULTS_ROOT_DIR=/home/amd/workspace/llm_benchmark_results
export CPU_NAME="Genoa-96C"
export LD_PRELOAD=${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libgomp.so
export VLLM_CPU_KVCACHE_SPACE=40
