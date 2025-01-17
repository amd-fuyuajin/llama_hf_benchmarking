import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def benchmark(model_name, input_length, output_length, batch_size, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    # Generate dummy input
    input_text = " ".join(["Hello"] * input_length)
    inputs = tokenizer([input_text] * batch_size, return_tensors="pt", padding=False, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Measure time to first token
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=output_length, do_sample=True)
    time_to_first_token = time.time() - start_time

    # Measure throughput
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):  # Run multiple iterations to get a stable throughput measurement
            outputs = model.generate(**inputs, max_new_tokens=output_length, do_sample=True)
    throughput = (batch_size * 10) / (time.time() - start_time)

    print(f"Time to first token: {time_to_first_token:.4f} seconds")
    print(f"Throughput: {throughput:.2f} samples/second")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Llama2-70b model on CPU")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Model name or path")
    parser.add_argument("--input_length", type=int, default=10, help="Length of the input sequence")
    parser.add_argument("--output_length", type=int, default=20, help="Length of the output sequence")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on (cpu or cuda)")
    # todo: add do_sample argument
    # todo: add temperature argument
    # todo: add top_k argument
    # todo: add top_p argument
    # todo: add number of beams argument
    args = parser.parse_args()
    benchmark(args.model_name, args.input_length, args.output_length, args.batch_size, args.device)
