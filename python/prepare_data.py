from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import os
import json

def prepare_data(data_dir, model_name, output_dir, prompt_lengths, num_prompts):
    """
    This function is used to prepare prompts for benchmarking. It takes the following arguments:
    - data_dir: The directory containing the dataset which can be loaded using datasets.load_from_disk. 
                for example, the wikipedia 20220301.en dataset.
    - model_name: The model name is used to get tokenizer
    - output_dir: The directory to save the prompts to files. 
    - prompt_lengths: A list of integers representing the lengths of the prompts to generate.
    - num_prompts: The number of prompts to generate for each length.
    """
    # Load the dataset
    dataset = load_from_disk(data_dir)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    filtered_prompts = {length:[] for length in prompt_lengths}

    idx = 0
    for val in tqdm(dataset["train"]):
        prompt = val["text"]
        prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
        # print(prompt_tokens)
        if len(prompt_tokens) < max(prompt_lengths):
            continue
        else:
            for length in prompt_lengths:
                truncated_prompt = tokenizer.decode(prompt_tokens[:length])
                filtered_prompts[length].append({"id": idx, "text": truncated_prompt})
            idx += 1
            if idx == num_prompts:
                break
    
    # if output_dir does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for length in prompt_lengths:
        with open(f"{output_dir}/prompts_{length}.txt", "w") as f:
            for prompt in filtered_prompts[length]:
                f.write(json.dumps(prompt) + "\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare prompts for benchmarking")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the dataset")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="prompts", help="Output directory")
    parser.add_argument("--prompt_lengths", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048], help="Lengths of the prompts")
    parser.add_argument("--num_prompts", type=int, default=10, help="Number of prompts to generate for each length")
    args = parser.parse_args()
    print(args)
    prepare_data(args.data_dir, args.model_name, args.output_dir, args.prompt_lengths, args.num_prompts)
