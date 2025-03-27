from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.model_registry import get_full_model_name
from utils.dataset_utils import load_train_dataset, load_eval_dataset, load_domain_specific_verifiers, check_correct_answer
from utils.gen_utils import generate_solutions
from utils.eval_utils import evaluate_problem


import argparse
import wandb
import torch
import random
import numpy as np
import json
import os
from tqdm import trange
import yaml

from accelerate import Accelerator
import time
from datetime import datetime
import torch.distributed as dist

def main():
    """
    Main function to run the complete training and evaluation pipeline.

    The process consists of:
      1. Loading the pre-trained model and tokenizer.
      2. Evaluating the initial model performance (before any finetuning).
      3. Performing reinforcement learning (GRPO) finetuning with MAV as the reward function, evaluating the model after each epoch.
      4. Saving the finetuned model and tokenizer.

    """
    start_time = time.time()
    start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_str_for_file = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
    parser.add_argument("--steps_per_iteration", type=int, default=500, help="Number of steps per iteration")
    parser.add_argument("--eval_only", action="store_true", default=False, help="Whether to evaluate only")
    parser.add_argument("--eval_mode", type=str, default="pass@1", help="Evaluation mode")
    parser.add_argument("--verbose", action="store_true", default=False, help="Whether to print verbose output")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    accelerator = Accelerator()
    print(f"Using device(s): {accelerator.device}")

    if args.output_dir is None:
        output_dir = f"{start_str_for_file}_{args.model}_{args.dataset}_{args.eval_mode}"
    else:
        output_dir = args.output_dir
    
    output_dirpath = f"/n/netscratch/hankyang_lab/Lab/alex/SIMAV/{output_dir}"
    os.makedirs(output_dirpath, exist_ok=True)

    # Log config to YAML file
    config_dict = vars(args)
    config_dict["num_procs"] = accelerator.num_processes
    print("Logging config to YAML file...")
    with open(f"{output_dirpath}/config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

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
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Update the model configuration with the correct token IDs.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # Load the train and eval datasets for this dataset
    train_dataset = load_train_dataset(args.dataset)
    eval_dataset = load_eval_dataset(args.dataset)
    veras = load_domain_specific_verifiers(args.dataset)

    #TODO: MOVE THIS TO EVALUATE_MODEL

    if args.eval_only:
        print(f"Evaluating {args.dataset} dataset using {args.eval_mode}")
        eval_examples = list(eval_dataset)  # Make it indexable

        model.eval()

        # Get distributed rank info
        rank = accelerator.local_process_index
        num_procs = accelerator.num_processes
        total_eval = len(eval_examples)
        print(f"Evaluating {total_eval} examples")

        # Determine how many examples each process gets
        evals_per_proc = total_eval // num_procs
        remainder = total_eval % num_procs

        # Compute start and end index for this rank
        if rank < remainder:
            start_idx = rank * (evals_per_proc + 1)
            end_idx = start_idx + (evals_per_proc + 1)
        else:
            start_idx = rank * evals_per_proc + remainder
            end_idx = start_idx + evals_per_proc

        rank_eval_examples = eval_examples[start_idx:end_idx]
        batch_size = 16

        correct_count = 0
        # Break into batches of 16 and evaluate
        for i in trange(0, len(rank_eval_examples), batch_size, desc=f"Rank {rank} (eval)"):
            batch = rank_eval_examples[i:i + batch_size]
            data_with_generated_solutions = generate_solutions(args, batch, model, tokenizer, accelerator)
            correct_count += evaluate_problem(args, data_with_generated_solutions)

            data_idx = start_idx + i  # global position for naming
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(f"[{now}] Rank {rank} completed eval {data_idx}-{data_idx+batch_size}")
            with open(f"{output_dirpath}/eval_{data_idx}-{data_idx+batch_size}_rank{rank}.json", "w") as f:
                json.dump(data_with_generated_solutions, f)
            
        # Convert local correct count to a tensor
        correct_tensor = torch.tensor([correct_count], dtype=torch.long, device=accelerator.device)

        # Reduce (sum) across all processes
        total_correct = accelerator.reduce(correct_tensor, reduction="sum").item()

        # Only print from main process
        if accelerator.is_main_process:
            print(f"Evaluation complete. Accuracy: {total_correct / total_eval:.2f} on {args.dataset} dataset using {args.eval_mode}")
    else: 
        train_data = list(train_dataset)  # Convert HuggingFace Dataset to a list

        # Get rank and number of processes
        rank = accelerator.local_process_index
        num_procs = accelerator.num_processes
        total_steps = args.steps_per_iteration

        # Compute base steps per process and the remainder
        steps_per_proc = total_steps // num_procs
        remainder = total_steps % num_procs

        # Determine the start and end indices (zero-based)
        if rank < remainder:
            # The first 'remainder' ranks get an extra step
            start_step = rank * (steps_per_proc + 1)
            end_step = start_step + (steps_per_proc + 1)
        else:
            start_step = rank * steps_per_proc + remainder
            end_step = start_step + steps_per_proc

        for i in trange(start_step, end_step, desc=f"Rank {rank}"):
            batch_data = random.sample(train_data, args.batch_size)
            # batch_prompts = [ex["problem"] for ex in batch_data]
            data_with_generated_solutions = generate_solutions(args, batch_data, model, tokenizer, accelerator)
            # batch_solutions = [ex["solution"] for ex in batch_data]

            breakpoint()

            #TODO: FIX THIS
            with open(f"{output_dirpath}/train_output_step_{i}_rank{rank}.json", "w") as f:
                json.dump(data_with_generated_solutions, f)

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Rank {rank} completed step {i} (batch size = {args.batch_size})")
        

        end_time = time.time()
        end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = end_time - start_time

        timing_info = {
            "start_time": start_str,
            "end_time": end_str,
            "elapsed_seconds": round(elapsed, 2),
            "elapsed_minutes": round(elapsed / 60, 2),
        }

        print(timing_info)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    # breakpoint()
    # ans = generate_answers(args, prompts, model, tokenizer)
    # breakpoint()

    # # -------------------------------
    # # Step 0: INITIAL EVALUATION
    # # -------------------------------

    # # Evaluate the initial performance of the model before any finetuning.
    # print("\nInitial model evaluation before GRPO:")
    # pre_grpo_accuracy = evaluate_model(model, tokenizer, eval_dataset, device)
    # print(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")

    # model = optimize_model_memory(model)

    # # Load the domain-specific verifier set for this dataset
    # verifiers = load_domain_specific_verifiers(args.dataset)

    # breakpoint()

    # wandb.finish()

if __name__ == "__main__":
    main()