import argparse
from datetime import datetime
import os
import re
from glob import glob
import pandas as pd
from pathlib import Path
import json

def extract_runtime_info(results_root_dir):
    results = []
    file_list = glob(os.path.join(results_root_dir, '**', 'performance_metrics_*_OUT1.txt'), recursive=True)  # Use glob to find all runtime.txt files recursively
    file_list = [Path(p) for p in sorted(file_list)]
    for file_path in file_list:
        entry = {}
        # Extract the directory name as the test case identifier
        entry["file_name"] = str(file_path)
        entry["platform"] = file_path.parent.parent.parent.name
        entry["test_type"] = file_path.parent.parent.name
        entry["test_name"] = file_path.parent.name
        output_length = re.search(r"OUT(\d+)", entry["test_name"])
        if output_length is None:
            continue
        else:
            entry["output_length"] = int(output_length.group(1))

        # replace the OUT1.txt with f"OUT{output_len}.txt" in the file path
        result_file = re.sub(r"OUT1.txt", f"OUT{entry['output_length']}.txt", file_path.name)
        result_file_path = file_path.parent / result_file
        # extract throughput and other information from result_file_path
        with open(result_file_path, 'r') as file:
            lines = file.readlines()
        arguments = json.loads(lines[0])
        entry["model_name"] = arguments["model_name"]
        entry["input_length"] = arguments["input_length"]
        assert entry["output_length"] == arguments["output_length"]
        entry["batch_size"] = arguments["batch_size"]
        entry["device"] = arguments["device"]
        entry["num_instances"] = arguments["num_instances"]
        entry["cores_per_instance"] = arguments["cores_per_instance"]
        entry["total_batches"] = arguments["total_batches"]
        entry["compile_backend"] = arguments["compile_backend"]

        runtimes = json.loads(lines[1])
        entry["time_to_encode"] = runtimes["time_to_encode"]["mean"]
        entry["time_to_generate"] = runtimes["time_to_generate"]["mean"]
        entry["time_to_decode"] = runtimes["time_to_decode"]["mean"]
        entry["latency"] = runtimes["latency"]["mean"]
        # the instance throughput was calculated using the total latency including runtime
        # for encoding, generation and decoding
        # the throughput for the generation stage is calcuated below
        entry["throughput_instance"] = runtimes["throughput_instance"]["mean"]

        perf_metrics = json.loads(lines[2])
        entry["throughput_soc"] = perf_metrics["throughput_soc"]
        entry["request_per_second"] = perf_metrics["request_per_second"]

        # read the file_path to get TTFT data
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        runtimes = json.loads(lines[1])
        entry["TTFT"] = runtimes["latency"]["mean"]

        # calculate more metrics
        # calculate the throughput for the generation stage
        # for each instance, it is calculated as number of tokens generated divided by 
        # time to the end of gneration subtracted by TTFT
        entry["throughput_instance_generation"] = \
            entry["batch_size"] * (entry["output_length"] -1) / (entry["time_to_encode"] + entry["time_to_generate"] - entry["TTFT"])

        # calcuated the SoC throughput for the generation stage
        # it is calculated as instance throughput for generation stage multiplied by number of instances
        entry["throughput_soc_generation"] = entry["throughput_instance_generation"] * entry["num_instances"]
        
        # calculate time per output token (TPOT) for the generation stage
        # use miniseconds as the unit
        entry["TPOT"] = 1000 / entry["throughput_instance_generation"]

        results.append(entry)

    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract throughput from test results.")
    parser.add_argument("results_root_dir", type=str, help="Path to the root folder containing test results")
    
    args = parser.parse_args()
    results_root_dir = args.results_root_dir
    if results_root_dir:
        data = extract_runtime_info(results_root_dir)
    else:
        data = extract_runtime_info("/results/") #the default path if no argument is provided

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format the timestamp as YYYYMMDD_HHMMSS
    file_name = f"llm_benchmark_extracted_results_{timestamp}.csv"
    data.to_csv(os.path.join(results_root_dir, file_name), index=False)  # Save the DataFrame to a CSV file
    print(data)

