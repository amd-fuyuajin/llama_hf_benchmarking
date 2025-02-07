import argparse
from collections import defaultdict
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import json
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, JoinableQueue, Value, Lock
from time import time, sleep
import pandas as pd
import numpy as np

def parse_cmd():
    parser = argparse.ArgumentParser(description="Benchmark Llama2-70b model on CPU")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Model name or path")
    parser.add_argument("--input_length", type=int, default=64, help="Length of the input sequence")
    parser.add_argument("--output_length", type=int, default=128, help="Length of the output sequence")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on (cpu or cuda)")
    parser.add_argument("--cpu_id_list", type=str, default=f"0-{os.cpu_count()-1}", help="bind the job to specific cpu ids")
    parser.add_argument("--num_instances", type=int, default=1, help="Number of instances to run")
    parser.add_argument("--cores_per_instance", type=int, default=os.cpu_count(), help="Number of cores per instance")
    parser.add_argument("--total_batches", type=int, default=10, help="Total number of batches to run")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--compile_backend", type=str, help="Backend to compile the model (e.g. ipex, zentorch)")
    # add an argument to specify whether to return the generated text
    parser.add_argument("--return_text", action="store_true", help="Return the generated text")
    # todo: add do_sample argument
    # todo: add temperature argument
    # todo: add top_k argument
    # todo: add top_p argument
    # todo: add number of beams argument
    args = parser.parse_args()
    if not args.cpu_id_list:
        args.cpu_id_list = list(range(os.cpu_count()))
    else:
        cpu_ids = []
        for tk in args.cpu_id_list.split(","):
            if "-" in tk:
                s_idx, e_idx = tk.split("-")
                for i in range(int(s_idx), int(e_idx)+1):
                    cpu_ids.append(i)
            else:
                cpu_ids.append(int(tk))
        args.cpu_id_list = cpu_ids
    
    # check the setting for number of instances and cores per instance
    assert args.num_instances * args.cores_per_instance <= len(args.cpu_id_list), "Number of instances * cores per instance should be less than or equal to the number of cpus"

    print(" ===== Arguments passed: =====\n")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    return args

# create a function to generate the output. This function will be called by multiprocessing.Process
# this function will read the prompts from a input queue and write the outputs to an output queue
def generate_output(model, tokenizer, input_queue, output_queue, cpu_ids, args, lock, instance_ready_counter):
    # get pid of the process
    pid = os.getpid()
    os.sched_setaffinity(pid, cpu_ids)
    torch.set_num_threads(len(cpu_ids))
    print(f"starting process {pid}")
    # load all the prompts into memory.
    # get the input_lenght from the args
    prompts_dict = {}
    input_length = args.input_length
    output_length = args.output_length
    batch_size = args.batch_size
    input_dir = os.getenv("INPUT_DIR")
    input_file = os.path.join(input_dir, f"prompts_{input_length}.txt")
    with open(input_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            prompt = json.loads(line)
            prompts_dict[prompt["id"]] = prompt["text"]
    print(f"data loaded in processed {pid}")

    # increment the instance_ready_counter
    lock.acquire()
    instance_ready_counter.value += 1
    lock.release()

    while True:
        # get the prompt from the input queue
        prompt_ids = input_queue.get()
        # print(f"get prompt ids: {prompt_ids}")
        if prompt_ids is None:
            output_queue.put(None)
            input_queue.task_done()
            print(f"stopping process {pid}")
            break
        # get the prompt from the prompts_dict
        prompt_batch = [prompts_dict[prompt_id] for prompt_id in prompt_ids]
        # print(f"prompt_batch: {prompt_batch}")
        t0 = time()        
        # encode the prompt
        inputs = tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        # print(f"inputs: {inputs}")
        # generate the output
        t1 = time()
        with torch.no_grad():
            outputs_tokens = model.generate(
                **inputs, 
                min_new_tokens=output_length, 
                max_new_tokens=output_length, 
                use_cache=True,
                do_sample=False)
        t2 = time()
        # put the output in the output queue
        outputs = tokenizer.batch_decode(outputs_tokens, skip_special_tokens=True)
        t3 = time()
        # print(f"PID={pid}, outputs: {outputs}")
        # if the args.return_text is not set, then do not return the generated text
        # set outputs to a list of empty strings
        if not args.return_text:
            outputs = [""] * len(outputs)

        output_queue.put({
            "prompt_ids": prompt_ids, 
            "outputs": outputs, 
            "time_to_encode": t1-t0,
            "time_to_generate": t2-t1,
            "time_to_decode": t3-t2,
            "latency": t3-t0,
            "throughput_instance": batch_size * output_length/(t3-t0)})
        input_queue.task_done()
        # print(f"Time to encode: {t1-t0:.4f} seconds, Time to generate: {t2-t1:.4f} seconds")

# create a function to load the model and tokenizer
def load_model(args):
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load the model
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    model.eval().to(args.device)
    compile_backend = args.compile_backend
    if compile_backend == "zentorch":
        import zentorch
        model = zentorch.llm.optimize(model, dtype=torch.bfloat16)
        model = torch.compile(model, backend="zentorch")
        print(f"compiled with backend: {compile_backend}")
    elif compile_backend == "ipex":
        import intel_extension_for_pytorch as ipex
        model = ipex.optimize(model, dtype=torch.bfloat16)
        model = torch.compile(model, backend="ipex")
        print(f"compiled with backend: {compile_backend}")
    elif compile_backend == "torchinductor":
        model = torch.compile(model)
        print(f"compiled with backend: {compile_backend}")
    elif compile_backend == "ipex_llm":
        import ipex_llm.transformers as ipex_transformers
        model = ipex_transformers.AutoModelForCausalLM.from_pretrained(model_path,
                                                 #load_in_4bit=True,
                                                 torch_dtype=torch.bfloat16,
                                                 optimize_model=True,
                                                 trust_remote_code=True,
                                                 use_cache=True)
        print(f"compiled with backend: {compile_backend}")
        model.eval().to(args.device)
    
    return model, tokenizer

# create the main function to run the benchmark. it takes the args as argument
# this function will load the model and tokenizer. It will start multiple processes to run the inference
# each process will read the prompts from a input queue and write the outputs to a output queue
def main(args):
    torch.set_num_threads(1)
    # load the model and tokenizer
    model, tokenizer = load_model(args)
    #share the model and tokenizer with the child processes
    model.share_memory()
    # print(model)
    # print(tokenizer)
    # create input and output queues
    input_queue = JoinableQueue()
    output_queue = JoinableQueue()

    # create a lock to synchronize the instance_ready_counter
    lock = Lock()
    instance_ready_counter = Value("i", 0)

    # create the processes
    processes = []
    for i in range(args.num_instances):
        cpu_ids = args.cpu_id_list[i*args.cores_per_instance:(i+1)*args.cores_per_instance]
        p = Process(target=generate_output, args=(model, tokenizer, input_queue, output_queue, cpu_ids, args, lock, instance_ready_counter))
        p.start()
        processes.append(p)

    batch_size = args.batch_size
    total_prompts = args.batch_size * args.total_batches
    print(f"total_prompts: {total_prompts}")

    while instance_ready_counter.value < args.num_instances:
        print(f"waiting for all instances to be ready.")
        sleep(2)

    start = time()
    print("start sending input")
    # send the prompt ids to the input queue
    for batch_ids in range(0, total_prompts, batch_size):
        end_idex = min(batch_ids+batch_size, total_prompts)
        prompt_ids = list(range(batch_ids, end_idex))
        input_queue.put(prompt_ids)
    
    # send None to the input queue to signal the end of the prompts
    for _ in range(args.num_instances):
        input_queue.put(None)

    # get the outputs from the output queue
    # create a file to write the output sequences
    input_length = args.input_length
    output_legth = args.output_length
    output_file = os.path.join(args.output_dir, f"output_seq_BS{batch_size}_IN{input_length}_OUT{output_legth}.txt")
    f_out = open(output_file, "w")

    # create a dictionary to collect the performance metrics
    performance_metrics = defaultdict(list)

    count = 0
    while True:
        outputs = output_queue.get()
        # print(outputs)
        if outputs is None:
            count += 1
            output_queue.task_done()
            if count == args.num_instances:
                break
        else:
            # prompt_ids, outputs = outputs
            # print(f"Prompt ids: {prompt_ids}, Outputs: {outputs}")
            for i in range(len(outputs["prompt_ids"])):
                f_out.write(json.dumps({
                    "prompt_id": outputs["prompt_ids"][i],
                    "output": outputs["outputs"][i],
                }) + "\n")
            performance_metrics["time_to_encode"].append(outputs["time_to_encode"])
            performance_metrics["time_to_generate"].append(outputs["time_to_generate"])
            performance_metrics["time_to_decode"].append(outputs["time_to_decode"])
            performance_metrics["latency"].append(outputs["latency"])
            performance_metrics["throughput_instance"].append(outputs["throughput_instance"])
            output_queue.task_done()
    
    # wait for the processes to finish
    input_queue.join()
    output_queue.join()
    for p in processes:
        p.join()

    end = time()
    total_runtime = end - start
    throughput_soc = total_prompts * args.output_length / total_runtime
    request_per_second = total_prompts / total_runtime
    print(f"Total time taken: {end-start:.4f} seconds")
    print(f"SoC throughput: {request_per_second:.2f} samples/second, {throughput_soc:.2f} tokens/sec")
    f_out.close()

    perf_df = pd.DataFrame(performance_metrics)
    print(perf_df.describe())

    # write the arguments and performance metrics to a file
    performance_file = os.path.join(args.output_dir, f"performance_metrics_BS{batch_size}_IN{input_length}_OUT{output_legth}.txt")
    with open(performance_file, "w") as f:
        f.write(json.dumps(vars(args)) + "\n")
        f.write(perf_df.describe().to_json() + "\n")
        f.write(json.dumps({
            "total_runtime": total_runtime,
            "throughput_soc": throughput_soc,
            "request_per_second": request_per_second
            }) + "\n")


if __name__ == "__main__":
    args = parse_cmd()
    test_output_length = args.output_length
    # set output_lenght to 1 to measure the TTFT
    args.output_length = 1
    main(args)
    # set output_lenght to the original value to measure the throughput
    args.output_length = test_output_length
    main(args)
