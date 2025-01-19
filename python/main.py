import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import json
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, JoinableQueue
from time import time

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
def generate_output(model, tokenizer, input_queue, output_queue, cpu_ids, args):
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
    input_dir = os.getenv("INPUT_DIR")
    input_file = os.path.join(input_dir, f"prompts_{input_length}.txt")
    with open(input_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            prompt = json.loads(line)
            prompts_dict[prompt["id"]] = prompt["text"]
    print(f"data loaded in processed {pid}")
    while True:
        # get the prompt from the input queue
        prompt_ids = input_queue.get()
        print(f"get prompt ids: {prompt_ids}")
        if prompt_ids is None:
            output_queue.put(None)
            input_queue.task_done()
            print(f"stopping process {pid}")
            break
        # get the prompt from the prompts_dict
        t0 = time()
        prompt_batch = [prompts_dict[prompt_id] for prompt_id in prompt_ids]
        print(f"prompt_batch: {prompt_batch}")
        # encode the prompt
        inputs = tokenizer(prompt_batch, return_tensors="pt", padding=False, truncation=True)
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        # print(f"inputs: {inputs}")
        # generate the output
        t1 = time()
        with torch.no_grad():
            outputs_tokens = model.generate(**inputs, min_new_tokens=output_length, max_new_tokens=output_length, do_sample=False)
        t2 = time()
        # put the output in the output queue
        outputs = tokenizer.batch_decode(outputs_tokens)
        print(f"PID={pid}, outputs: {outputs}")
        output_queue.put((prompt_ids, outputs))
        input_queue.task_done()
        print(f"Time to encode: {t1-t0:.4f} seconds, Time to generate: {t2-t1:.4f} seconds")

# create a function to load the model and tokenizer
def load_model(args):
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load the model
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
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
    print(model)
    print(tokenizer)
    # tokenizer.share_memory()
    # create input and output queues
    input_queue = JoinableQueue()
    output_queue = JoinableQueue()
    # create the processes
    processes = []
    for i in range(args.num_instances):
        cpu_ids = args.cpu_id_list[i*args.cores_per_instance:(i+1)*args.cores_per_instance]
        p = Process(target=generate_output, args=(model, tokenizer, input_queue, output_queue, cpu_ids, args))
        p.start()
        processes.append(p)
    
    # send the prompt ids to the input queue
    start = time()
    print("start sending input")
    batch_size = args.batch_size
    total_prompts = args.batch_size * args.total_batches
    print(f"total_prompts: {total_prompts}")
    for batch_ids in range(0, total_prompts, batch_size):
        prompt_ids = list(range(batch_ids, batch_ids+batch_size))
        input_queue.put(prompt_ids)
    
    # send None to the input queue to signal the end of the prompts
    for _ in range(args.num_instances):
        input_queue.put(None)

    # get the outputs from the output queue
    count = 0
    while True:
        outputs = output_queue.get()
        print(outputs)
        if outputs is None:
            count += 1
            output_queue.task_done()
            if count == args.num_instances:
                break
        else:
            # prompt_ids, outputs = outputs
            # print(f"Prompt ids: {prompt_ids}, Outputs: {outputs}")
            output_queue.task_done()
    
    # wait for the processes to finish
    input_queue.join()
    output_queue.join()
    for p in processes:
        p.join()

    end = time()
    print(f"Total time taken: {end-start:.4f} seconds")
    print(f"throughput: {total_prompts/(end-start):.2f} samples/second, {total_prompts * args.output_length/(end-start):.2f} tokens/sec")



# this function is not used
def benchmark(model_name, input_length, output_length, batch_size, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.eval()
    model.to(device)
    
    # Generate dummy input
    input_text = " ".join(["Hello"] * input_length)
    inputs = tokenizer([input_text] * batch_size, return_tensors="pt", padding=False, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Measure time to first token
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=output_length, do_sample=False)
    time_to_first_token = time.time() - start_time

    # Measure throughput
    start_time = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for _ in range(10):  # Run multiple iterations to get a stable throughput measurement
            outputs = model.generate(**inputs, max_new_tokens=output_length, do_sample=False)
    throughput = (output_length * batch_size * 10) / (time.time() - start_time)

    print(f"Time to first token: {time_to_first_token:.4f} seconds")
    print(f"Throughput: {throughput:.2f} samples/second")

if __name__ == "__main__":
    args = parse_cmd()
    main(args)
