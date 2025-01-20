#!/bin/bash
set -e 

source env_setup.sh

# get number of CPU cores on the machine
cpu_count=$(nproc)
test_name="test_script"
for num_instances in 12; do
    cores_per_instance=$((cpu_count/num_instances))
    total_batches=$((num_instances*2))
    for input_length in 128; do
        for output_length in 128; do
            folder_name="P${num_instances}_IN${input_length}_OUT${output_length}"
            ./llm_benchmark.sh --test-name $test_name --folder-name $folder_name \
            --input_length $input_length --output_length $output_length \
            --num_instances $num_instances --cores_per_instance $cores_per_instance \
            --total_batches $total_batches
        done
    done
done