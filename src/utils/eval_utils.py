import json
import argparse
import time
from datetime import datetime
import random
import numpy as np
import os
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.dataset_utils import load_eval_dataset
from utils.model_registry import get_full_model_name
import torch
from utils.gen_utils import generate_solutions
from utils.dataset_utils import check_correct_answer


def evaluate_problem(args, data_with_generated_solutions):
    """
    Evaluates the model on a single problem or batch of problems using the pass@n evaluation metric.

    Args:
        args (argparse.Namespace): The arguments.
        data_with_generated_solutions (list): List of dictionaries containing "generated_answers" and "gt_answer".

    Returns:
        int: The number of correct predictions.
    """
    if args.eval_mode.startswith("pass@"):
        try:
            n = int(args.eval_mode.split("@")[1])
            correct_count = evaluate_problem_pass_at_n(args, data_with_generated_solutions, n)
        except (ValueError, IndexError):
            raise ValueError(f"Invalid eval_mode format: {args.eval_mode}. Expected format: pass@n where n is a number.")
    elif args.eval_mode == "bon-mav":
        correct_count = evaluate_problem_bon_mav(args, data_with_generated_solutions)
    else:
        raise ValueError(f"Invalid eval_mode: {args.eval_mode}")
    
    return correct_count

def evaluate_problem_bon_mav(args, data_with_generated_solutions):
    """
    Evaluates the model on a single problem or batch of problems using the bon-mav evaluation metric.
    """
    correct_count = 0
    raise NotImplementedError("bon-mav evaluation metric not implemented")
    

def evaluate_problem_pass_at_n(args, data_with_generated_solutions, n):
    """
    Evaluates the model on a single problem or batch of problems using the pass@n evaluation metric.

    Args:
        args (argparse.Namespace): The arguments.
        data_with_generated_solutions (list): List of dictionaries containing "generated_answers" and "gt_answer".
        n (int): The number of generated answers to consider for evaluation (e.g., 1 for pass@1, 8 for pass@8).

    Returns:
        int: The number of correct predictions.
    """
    assert args.num_generations >= n, f"num_generations must be {n} or greater for pass@{n} evaluation"
    if args.num_generations > n:
        print(f"Warning: You've set num_generations to {args.num_generations}, but eval_mode is set to pass@{n}. Only the first {n} generated answers will be considered.")

    correct_count = 0
    for d in data_with_generated_solutions:
        is_correct = False
        for i in range(n):
            is_correct = check_correct_answer(d["generated_answers"][i], d["gt_answer"], args.dataset)
            if is_correct:
                correct_count += 1
                break
        if args.verbose:
            print(f"Checking if any of the first {n} generated answers {d['generated_answers'][:n]} is the same as the ground truth answer {d['gt_answer']} | Result: {'Correct' if is_correct else 'Incorrect'}")

    return correct_count



#TODO: Work in progress
def evaluate_model(args, model, tokenizer, eval_dataset, device):
    """
    Evaluates the model on a set of examples and prints detailed results.
    
    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer for encoding inputs and decoding outputs.
        eval_examples (list): List of evaluation examples, each containing "problem" and "solution".
        device: The device (CPU or GPU) to run evaluation on.
        
    Returns:
        float: The accuracy percentage (correct predictions / total examples * 100).
        
    Explanation:
        1. Sets the model to evaluation mode.
        2. For each example in the evaluation set:
           - Encodes the prompt and generates a response using the model.
           - Extracts the predicted answer from the generated response.
           - Compares the predicted answer with the expected answer using multiple methods:
             a. Exact string matching
             b. Single number extraction and comparison
             c. Last number extraction and comparison
           - Prints detailed information about each example.
        3. Calculates and returns the overall accuracy.
        4. Returns the model to training mode.
    """
    model.eval()
    correct = 0
    total = len(eval_dataset)
    print("\n" + "="*50)
    print("EVALUATION ON", total, "EXAMPLES USING", args.eval_mode)
    print("="*50)



    ans = generate_solutions(args, eval_examples, model, tokenizer, accelerator)

    # Save generated answers as backup
    with open(f"backup_generated_answers_{args.run_name}.json", "w") as f:
        json.dump(ans, f)



#TODO: Work in progress
if __name__ == "__main__":
    start_time = time.time()
    start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(description="Self-Improvement via GRPO and MAV")
    parser.add_argument("--model", type=str, default="gemma-3-4b-it", help="Model")
    parser.add_argument("--dataset", type=str, default="math", help="Dataset")
    parser.add_argument("--temp", type=float, default=1.0, help="Model temperature")
    parser.add_argument("--topk", type=int, default=64, help="Top-k")
    parser.add_argument("--topp", type=float, default=0.95, help="Top-p")
    parser.add_argument("--num_generations", type=int, default=8, help="Number of generations to sample")
    parser.add_argument("--log_completions", action="store_true", help="Whether to log prompt-completion pairs")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of tokens for generation")
    parser.add_argument("--num_train_iterations", type=int, default=1, help="Number of training iterations")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--eval_mode", type=str, default="pass@1", help="Evaluation mode")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    accelerator = Accelerator()
    print(f"Using device(s): {accelerator.device}")

    os.makedirs("generated_data", exist_ok=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    if args.output_dir is None:
        output_dir = args.model + "_" + args.dataset + "_MAV_GRPO"
    else:
        output_dir = args.output_dir

    # Initialize wandb
    wandb_run_name = f"{args.model}-MAV-GRPO"
    # wandb.init(project="MAV-GRPO-Training", name=wandb_run_name)

    full_model_name = get_full_model_name(args.model)

    print("Downloading model...")
    # Load the pre-trained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        full_model_name,
        torch_dtype=torch.bfloat16,
        #attn_implementation="flash_attention_2",
        device_map="auto"
    )
    print("Downloaded model: ", args.model)
    # # Move the model to the determined device. #TODO: Might not need this when doing distributed training with accelerate 
    # model = model.to(device)

    model = model.to(accelerator.device)  # ✅ Just move to device, no .prepare()



    tokenizer = AutoTokenizer.from_pretrained(full_model_name, padding_side="left")
    # Set the pad token to be the same as the end-of-sequence token.
    tokenizer.pad_token = tokenizer.eos_token
    # Update the model configuration with the correct token IDs.
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # Load the eval dataset for this dataset
    eval_dataset = load_eval_dataset(args.dataset)

    eval_examples = list(eval_dataset)  # Convert HuggingFace Dataset to a list

    evaluate_model(args, model, tokenizer, eval_dataset, accelerator.device)

    # # Get rank and number of processes
    # rank = accelerator.local_process_index
    # num_procs = accelerator.num_processes
    # total_dataset_length = len(eval_examples)

    # # Compute base steps per process and the remainder
    # problems_per_proc = total_dataset_length // num_procs
    # remainder = total_dataset_length % num_procs

    # # Determine the start and end indices (zero-based)
    # if rank < remainder:
    #     # The first 'remainder' ranks get an extra step
    #     start_problem = rank * (problems_per_proc + 1)
    #     end_problem = start_problem + (problems_per_proc + 1)
    # else:
    #     start_problem = rank * problems_per_proc + remainder
    #     end_problem = start_problem + problems_per_proc

    # problems = eval_examples[start_problem:end_problem]
    # prompts = [p["problem"] for p in problems]
        
    # ans = generate_answers(args, prompts, model, tokenizer, accelerator)
    # solutions = [p["solution"] for p in problems]
    
    # # save generated answers and solutions
    # with open(f"/n/netscratch/hankyang_lab/Lab/alex/SIMAV/ans_{args.model}_{args.dataset}_{start_problem}:{end_problem}_rank{rank}.json", "w") as f:
    #     json.dump(ans, f)
    # with open(f"/n/netscratch/hankyang_lab/Lab/alex/SIMAV/solutions_{args.model}_{args.dataset}_{start_problem}:{end_problem}_rank{rank}.json", "w") as f:
    #     json.dump(solutions, f)

    # now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print(f"[{now}] Rank {rank} completed {start_problem}:{end_problem}")
    

    # end_time = time.time()
    # end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # elapsed = end_time - start_time

    # timing_info = {
    #     "start_time": start_str,
    #     "end_time": end_str,
    #     "elapsed_seconds": round(elapsed, 2),
    #     "elapsed_minutes": round(elapsed / 60, 2),
    # }

    # print(timing_info)

    # if dist.is_available() and dist.is_initialized():
    #     dist.destroy_process_group()

    

    