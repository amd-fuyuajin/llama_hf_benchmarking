# Benchmarking package for Llama inference on CPU

This package provides a robust framework for benchmarking Large Language Model (LLM) inference performance on CPU architectures. It is designed to maximize hardware utilization, support various optimization backends, and automate the collection of performance metrics.

## Key Features

*   **Multi-Instance Execution**: Leveraging Python's `multiprocessing` library, the tool can spawn multiple inference instances to fully saturate high-core-count CPU architectures.
*   **Model Weight Sharing**: To enable running multiple instances of large models (e.g., Llama-3-70B) without exhausting RAM, the framework supports model sharing. A single loaded model copy can serve multiple inference workers.
*   **Multiple Optimization Backends**: Supports compiling models to optimize performance for specific hardware. Supported backends include:
    *   `zentorch` (AMD ZenDNN)
    *   `ipex` (Intel Extension for PyTorch)
    *   `torchinductor` (PyTorch compiler)
    *   `ipex_llm`
*   **Automated Sweeps**: Includes shell scripts to automate sweeping through different configurations, such as:
    *   Batch sizes
    *   Input/Output token lengths
    *   Number of instances
*   **Comprehensive Metrics**: Automatically extracts and calculates key performance indicators (KPIs) like:
    *   Time To First Token (TTFT)
    *   Token generation latency
    *   Throughput (Tokens/sec) at both instance and SoC levels
*   **Hardware Profiling**: Captures detailed platform information (CPU, Memory, BIOS, Environment) alongside results for reproducibility.

## Downloads package and dataset
To get started with the benchmarking package, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/amd-fuyuajin/llama_hf_benchmarking.git
cd llama_hf_benchmarking
```
The dataset used for benchmarking is `wikipedia_20220301_en_240614`. You can download this dataset and generate input prompt files using the provided script.
The generated prompt files can also be downloads from the following location.

## Setup Conda Environment

Ensure you have a Conda distribution installed (e.g., Miniforge or Anaconda). Create the environment using the provided [environment.yml](environment.yml) file.

```bash
conda env create -f environment.yml
conda activate llama_benchmark_env
```

## Configure Environment Variables

The package relies on specific environment variables for paths and API tokens.

#### Step 1: Create secrets.env file

Create a `secrets.env` file to store your Hugging Face access token:

```bash
# secrets.env
export HUGGING_FACE_HUB_TOKEN="your_huggingface_token_here"
```

> **Note**: You can obtain a Hugging Face token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Make sure you have accepted the license agreement for the Llama models you want to benchmark.

#### Step 2: Edit env_setup.sh

Edit the `env_setup.sh` file to configure the required paths for your system:

```bash
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
```
Environment variables explained:
- `CONDA_PREFIX`: Path to your Conda environment
- `HF_HOME`: Directory for Hugging Face model cache
- `INPUT_DIR`: This is the directory to store generated prompt files, which has a format like `$INPUT_DIR/prompts_{length}.txt`
- `RESULTS_ROOT_DIR`: Directory to save benchmark results
- `CPU_NAME`: A label for the CPU being benchmarked, used in result organization
- `LD_PRELOAD`: Preload necessary libraries for performance optimization. Different OMP and memory allocators can be tested here.

## Data Preparation
Pre-generated prompt files can be downloaded. They are for input lengths: 8, 16, 32, 64, 128, 256, 512, 1024, and 2048. But if you need different prompt lenghts other than those existing ones, you can generate your own prompt files using the `prepare_data.py` script.

```bash
python python/prepare_data.py \
    --data_dir data/path/to/wikipedia_20220301_en_240614 \
    --output_dir output/dir/for/prompts \
    --input_lengths 128 256 512 1024 2048 \
    --num_prompts 1000
```
This generates txt files in the format `prompts_{length}.txt` that will be used as inputs during benchmarking.

## Usage

### 1. Running Batch Sweeps
Edit `run_batch.sh` script to configure your desired benchmark parameters.
Here is an example of how to run the benchmark sweep on a single socket server:
```bash
cpu_count=$(nproc)
model_name="meta-llama/Llama-3.1-8B-Instruct"
test_name="06202025-llama3.1-8B-zentorch5.0.2"
compile_backend="zentorch"
model_copies=1

for rep in 1 2 3; do
for num_instances in 4 8 16; do
    cores_per_instance=$((cpu_count/num_instances))
    total_batches=$((num_instances*1))
    for batch_size in 1 2 4 8 16; do
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
            done
        done
    done
done

done
```
Key variables explained:
- `model_name`: Hugging Face model identifier
- `test_name`: A label for the benchmark run
- `compile_backend`: Optimization backend to use, support `zentorch`, `ipex`, `torchinductor`
- `model_copies`: Number of model copies to load into memory for sharing. **set it to 1 for single socket, and set to 2 for dual socket**
- `num_instances`: Number of parallel inference instances to run
- `cores_per_instance`: CPU cores allocated per instance. This is calculated based on total CPU cores divided by `num_instances`
- `batch_size`: Number of inputs processed per inference call
- `input_length`: Number of input tokens per prompt
- `output_length`: Number of output tokens to generate
- `total_batches`: Total number of batches for all instances to process. This is calculated as `num_instances * 1` in the example above, meaning each instance processes 1 batch.
If you want to use AMD uProf to collect hardware profile data, use the `--uprof` flag to enable it. Make sure you have installed AMD uProf and set up the environment correctly.

To run a comprehensive benchmark sweep across multiple configurations, use the `run_batch.sh` script:
**You need to use sudo run**, because `sudo` previlege is required to collect some hardware metrics.

```bash
sudo ./run_batch.sh
```

### 2. Output Files

Benchmark results are saved to the directory specified by `RESULTS_ROOT_DIR`. The output structure is organized as follows:

```
$RESULTS_ROOT_DIR/
├── <CPU_NAME>/
│   ├── <test_name>/
│   │   ├── <folder_name_for run 1>/          # e.g., P8_BS4_IN128_OUT128_REP1
│   │   ├── <folder_name_for run 2>/          # e.g., P8_BS4_IN128_OUT128_REP2
│   │   ├── <folder_name_for run 3>/          # e.g., P8_BS4_IN128_OUT128_REP3
│   │   └── <folder_name_for run 4>/          # e.g., P8_BS4_IN128_OUT128_REP4
│   └── ...
└── ...
```
Here is an example of the output files in each run folder:
```text
-rw-r--r-- 1 amd amd 1.5K Jun 21  2025 bios_info.txt
-rw-r--r-- 1 amd amd 3.3K Jun 21  2025 cpu_info.txt
-rw-r--r-- 1 amd amd  22K Jun 21  2025 dmidecode_memory.txt
-rw-r--r-- 1 amd amd 2.2K Jun 21  2025 env_vars.txt
-rw-r--r-- 1 amd amd 7.3K Jun 21  2025 memory_info.txt
-rw-r--r-- 1 amd amd  520 Jun 21  2025 numa_nodes.txt
-rw-r--r-- 1 amd amd 4.4K Jun 21  2025 output_seq_BS8_IN128_OUT128.txt
-rw-r--r-- 1 amd amd 4.4K Jun 21  2025 output_seq_BS8_IN128_OUT1.txt
-rw-r--r-- 1 amd amd 2.0K Jun 21  2025 packages.txt
-rw-r--r-- 1 amd amd 3.2K Jun 21  2025 perf_dataframe_BS8_IN128_OUT128.csv
-rw-r--r-- 1 amd amd 3.2K Jun 21  2025 perf_dataframe_BS8_IN128_OUT1.csv
-rw-r--r-- 1 amd amd 2.4K Jun 21  2025 performance_metrics_BS8_IN128_OUT128.txt
-rw-r--r-- 1 amd amd 2.4K Jun 21  2025 performance_metrics_BS8_IN128_OUT1.txt
-rw-r--r-- 1 amd amd  136 Jun 21  2025 uname.txt
```
You can find the OS, BIOS, CPU, memory, NUMA, and library information from the respective text files. The performance metrics and detailed performance dataframes are saved in the corresponding files. The files containing `*_OUT1` are for measuring Time To First Token (TTFT), while those with `*_OUT{output_length}` are for measuring throughput and token generation latency.

### 3. Extract Results

To aggregate all benchmark results into a structured CSV file for analysis, use the results extractor:

```bash
python python/extract_results.py <results_directory>
```
The output CSV file will be saved in the `<results_directory>` with the name `llm_benchmark_extracted_results_{timestamp}.csv`.

The generated CSV includes the following performance metrics:
| Metric | Description |
|--------|-------------|
| `model_name` | Name of the model benchmarked |
| `batch_size` | Batch size used |
| `input_length` | Number of input tokens |
| `output_length` | Number of output tokens generated |
| `num_instances` | Number of parallel inference instances |
| `cores_per_instance` | CPU cores allocated per instance |
| `backend` | Compilation backend used |
| `TTFT` | Time To First Token (**The unit is second**) |
| `TPOT` | Time Per Output Token (**The unit is minisecond**) |
| `throughput_instance` | end-to-end throughput (Tokens/second) per instance |
| `throughput_soc` | Aggregate end-to-end throughput (tokens/second) across all instances |
| `throughput_instance_generation` | Generation-only throughput (Tokens/second) per instance |
| `throughput_soc_generation` | Aggregate generation-only throughput (Tokens/second) across all instances |
| `concurrent_prompts` | Total number of concurrent prompts being processed (num_instances * batch_size) |

## Supported Models

This framework has been tested with the following models:
- Llama 3.1 8B / 70B (Instruct and Base)
- Llama 3 8B / 70B
- Llama 2 7B / 13B / 70B

Other Hugging Face compatible models may work but it depends on the version of `transformers`.

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `num_instances` or `batch_size` or enable model weight sharing for large models
2. **Model Download Fails**: Verify your `HUGGING_FACE_HUB_TOKEN` is valid and you've accepted the model license
3. **Backend Not Found**: Ensure the optimization backend (zentorch/ipex) is properly installed in your conda environment


