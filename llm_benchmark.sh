#!/bin/bash
set -e
# set -x

CPU_MODEL="$(lscpu | grep "Model name" | awk -F 'name: ' '{print $2}' | awk '{$1=$1}1' | tr -d ':' | tr ' ' '_')"
echo $CPU_MODEL
MAIN_DIR=$(realpath $(dirname "$0"))
echo "the main directory is $MAIN_DIR"

# get number of CPU cores on the machine
cpu_count=$(nproc)

# default values
device="cpu"
cpu_id_list="0-$((cpu_count-1))"
uprof=false
batch_size=1
model_name="meta-llama/Llama-3.1-8B-Instruct"
input_length=128
output_length=128
model_copies=1

# parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test-name) test_name=$2; shift;;
        --folder-name) folder_name=$2; shift;;
        --model_name) model_name="$2"; shift ;;
        --input_length) input_length="$2"; shift ;;
        --output_length) output_length="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --device) device="$2"; shift ;;
        --cpu_id_list) cpu_id_list="$2"; shift ;;
        --num_instances) num_instances="$2"; shift ;;
        --cores_per_instance) cores_per_instance="$2"; shift ;;
        --total_batches) total_batches="$2"; shift ;;
        --output_dir) output_dir="$2"; shift ;;
        --compile_backend) compile_backend="$2"; shift ;;
        --model_copies) model_copies="$2"; shift ;;
        --uprof) uprof=true;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Example usage of the parsed arguments
echo "Model Name: $model_name"
echo "Input Length: $input_length"
echo "Output Length: $output_length"
echo "Batch Size: $batch_size"
echo "Device: $device"
echo "CPU ID List: $cpu_id_list"
echo "Number of Instances: $num_instances"
echo "Cores per Instance: $cores_per_instance"
echo "Total Batches: $total_batches"
echo "Output Directory: $output_dir"
echo "Compile Backend: $compile_backend"
echo "Model Copies: $model_copies"

# Add your command to run the Python script here, for example:
# python main.py --model_name "$model_name" --input_length "$input_length" --output_length "$output_length" --batch_size "$batch_size" --device "$device" --cpu_id_list "$cpu_id_list" --num_instances "$num_instances" --cores_per_instance "$cores_per_instance" --total_batches "$total_batches" --output_dir "$output_dir" --compile_backend "$compile_backend"

PYTHON="${CONDA_PREFIX}/bin/python"
if $uprof; then
    if [ ! -f "/opt/AMDuProf_Internal_Linux_x64_5.0.845/bin/AMDuProfPcm" ]; then
        echo "uProf 5.0 is not installed. Please install uProf first."
        exit 1
    else
        # prerequisite for uProf to work
        su -c 'echo 0 > /proc/sys/kernel/nmi_watchdog; echo -1 > /proc/sys/kernel/perf_event_paranoid; modprobe msr'
        UPROFPCM="/opt/AMDuProf_Internal_Linux_x64_5.0.845/bin/AMDuProfPcm -m ipc,fp,l1,l2,l3,dc,memory,dma,xgmi,pcie,pipeline_util -a -s --msr --html"
        # UPROF="/opt/AMDuProf_Internal_Linux_x64_5.0.845/bin/AMDPerf/AMDuProfSys --config core,l3,df,umc -a"
        POWER_PROF="/opt/AMDuProf_Internal_Linux_x64_5.0.845/bin/AMDuProfCLI timechart -e power -a"
    fi
fi


OUTPUT_DIR="${RESULTS_ROOT_DIR}/${CPU_NAME}/${test_name}/${folder_name}"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

./platform_info.sh $OUTPUT_DIR

if $uprof; then
    $POWER_PROF -o "$OUTPUT_DIR" \
    $UPROFPCM -o "${OUTPUT_DIR}/${folder_name}.csv" -- \
        $PYTHON -u "${MAIN_DIR}/python/main.py" --model_name "$model_name" \
        --input_length "$input_length" --output_length "$output_length" \
        --batch_size "$batch_size" --device "$device" --cpu_id_list "$cpu_id_list" \
        --num_instances "$num_instances" --cores_per_instance "$cores_per_instance" \
        --total_batches "$total_batches" --output_dir "$OUTPUT_DIR" \
        --compile_backend "$compile_backend" --model_copies "$model_copies"

else
    $PYTHON -u "${MAIN_DIR}/python/main.py" --model_name "$model_name" \
        --input_length "$input_length" --output_length "$output_length" \
        --batch_size "$batch_size" --device "$device" --cpu_id_list "$cpu_id_list" \
        --num_instances "$num_instances" --cores_per_instance "$cores_per_instance" \
        --total_batches "$total_batches" --output_dir "$OUTPUT_DIR" \
        --compile_backend "$compile_backend" --model_copies "$model_copies"

fi


