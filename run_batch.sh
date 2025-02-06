#!/bin/bash
set -e 

source env_setup.sh

# get number of CPU cores on the machine
# cpu_count=$(nproc)
# test_name="instance_scaling"
# for num_instances in 1 2 4 8 12 24 48; do
#     cores_per_instance=$((cpu_count/num_instances))
#     total_batches=$((num_instances*10))
#     for input_length in 128 1024; do
#         for output_length in 128 1024; do
#             folder_name="P${num_instances}_IN${input_length}_OUT${output_length}"
#             ./llm_benchmark.sh --test-name $test_name --folder-name $folder_name \
#             --input_length $input_length --output_length $output_length \
#             --num_instances $num_instances --cores_per_instance $cores_per_instance \
#             --total_batches $total_batches
#         done
#     done
# done


# test_name="test_batchsize"
# cpu_count=$(nproc)
# num_instances=1
# cores_per_instance=96
# total_batches=2
# input_length=128
# output_length=128
# for bs in 1 2 4 8 12 16 24 32; do
#     folder_name="P${num_instances}_BS${bs}_IN${input_length}_OUT${output_length}"
#     ./llm_benchmark.sh --test-name $test_name --folder-name $folder_name --batch_size $bs \
#             --input_length $input_length --output_length $output_length \
#             --num_instances $num_instances --cores_per_instance $cores_per_instance \
#             --total_batches $total_batches
# done

# get number of CPU cores on the machine
# cpu_count=$(nproc)
# test_name="collect_uprof_bs"
# 
# for num_instances in 1; do
#     cores_per_instance=$((cpu_count/num_instances))
#     total_batches=$((num_instances*2))
#     for batch_size in 8 16 32 64 128 256; do
#         for input_length in 128; do
#             for output_length in 128; do
#                 folder_name="P${num_instances}_BS${batch_size}_IN${input_length}_OUT${output_length}"
#                 ./llm_benchmark.sh --test-name $test_name --folder-name $folder_name --batch_size $batch_size \
#                 --input_length $input_length --output_length $output_length \
#                 --num_instances $num_instances --cores_per_instance $cores_per_instance \
#                 --total_batches $total_batches --uprof
#             done
#         done
#     done
# done
# 
# 
# cpu_count=$(nproc)
# test_name="collect_uprof_input"
# 
# for num_instances in 1; do
#     cores_per_instance=$((cpu_count/num_instances))
#     total_batches=$((num_instances*2))
#     for batch_size in 1 8; do
#         for input_length in 32 64 128 256 512 1024; do
#             for output_length in 128; do
#                 folder_name="P${num_instances}_BS${batch_size}_IN${input_length}_OUT${output_length}"
#                 ./llm_benchmark.sh --test-name $test_name --folder-name $folder_name --batch_size $batch_size \
#                 --input_length $input_length --output_length $output_length \
#                 --num_instances $num_instances --cores_per_instance $cores_per_instance \
#                 --total_batches $total_batches --uprof
#             done
#         done
#     done
# done
# 
# 
# cpu_count=$(nproc)
# test_name="collect_uprof_output"
# 
# for num_instances in 1; do
#     cores_per_instance=$((cpu_count/num_instances))
#     total_batches=$((num_instances*2))
#     for batch_size in 1 8; do
#         for input_length in 128; do
#             for output_length in 32 64 128 256 512 1024; do
#                 folder_name="P${num_instances}_BS${batch_size}_IN${input_length}_OUT${output_length}"
#                 ./llm_benchmark.sh --test-name $test_name --folder-name $folder_name --batch_size $batch_size \
#                 --input_length $input_length --output_length $output_length \
#                 --num_instances $num_instances --cores_per_instance $cores_per_instance \
#                 --total_batches $total_batches --uprof
#             done
#         done
#     done
# done


cpu_count=$(nproc)
test_name="zentorch_test_new_script"
compile_backend="zentorch"

for num_instances in 12; do
    cores_per_instance=$((cpu_count/num_instances))
    total_batches=$((num_instances*2))
    for batch_size in 1 2 4 8 16 32 64 128 256; do
	if (( num_instances == 4 )) && (( batch_size >= 128 )); then
            continue
	fi
        for input_length in 128 1024; do
            for output_length in 128 1024; do
                folder_name="P${num_instances}_BS${batch_size}_IN${input_length}_OUT${output_length}"
                ./llm_benchmark.sh --test-name $test_name --folder-name $folder_name --batch_size $batch_size \
                --input_length $input_length --output_length $output_length \
                --num_instances $num_instances --cores_per_instance $cores_per_instance \
                --total_batches $total_batches --compile_backend $compile_backend \
		--uprof

            done
        done
    done
done
