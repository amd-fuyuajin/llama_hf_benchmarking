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
cpupower frequency-set -r -g performance
cpupower idle-set -d 2

cpu_count=$(nproc)
#cpu_count=144
#cpu_id_list="0-11,16-27,32-43,48-59,64-75,80-91,96-107,112-123,128-139,144-155,160-171,176-187"
#cpu_count=192
#cpu_id_list="0-191"
model_name="meta-llama/Llama-3.1-8B-Instruct"
test_name="06202025-llama3.1-8B-zentorch5.0.2"
compile_backend="zentorch"
model_copies=1

for rep in 1 2 3; do
for num_instances in 4 8 16; do
    cores_per_instance=$((cpu_count/num_instances))
    total_batches=$((num_instances*1))
    for batch_size in 128; do
	if (( num_instances * batch_size > 2500 )); then
            continue
	fi
        for input_length in 128 1024; do
            for output_length in 128 1024; do
                folder_name="P${num_instances}_BS${batch_size}_IN${input_length}_OUT${output_length}_REP${rep}"
                ./llm_benchmark.sh --test-name $test_name --folder-name $folder_name --batch_size $batch_size \
                --model_name $model_name --model_copies $model_copies \
                --input_length $input_length --output_length $output_length \
                --num_instances $num_instances --cores_per_instance $cores_per_instance \
                --total_batches $total_batches --compile_backend $compile_backend \
		#--cpu_id_list $cpu_id_list
		        #--uprof
#		exit 0

            done
        done
    done
done

done
