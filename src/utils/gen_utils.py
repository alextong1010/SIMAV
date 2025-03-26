import argparse
from prompts.gen_prompts import get_gen_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from accelerate import Accelerator
from typing import Optional
from utils.math_utils.util import remove_boxed, last_boxed_only_string
from termcolor import colored

def generate_solutions(
    args: argparse.Namespace,
    data: dict | list[dict], 
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    accelerator: Accelerator
) -> dict | list[dict]:
    """
    Generate multiple solutions for one or multiple problems.
    
    Args:
        args: Parser arguments
        data: A single data dictionary or a list of data dictionaries
        model: The model to use for generation
        tokenizer: The tokenizer to use
        accelerator: The accelerator to use

    Returns:
        If data is a dict: the input dict with a new "solutions" key containing generated solutions
        If data is a list: the input list of dicts, each with a new "solutions" key
    """
    # Handle single problem or list of problems
    is_batch = isinstance(data, list)
    batch_problems = [d["problem"] for d in data] if is_batch else [data["problem"]]
    
    # Create prompts for each problem
    user_prompts = [get_gen_prompt(args.dataset, p) for p in batch_problems]
    generator_messages = [
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            },
        ] for prompt in user_prompts
    ]
    # Process all prompts in batch
    inputs = tokenizer.apply_chat_template(
        generator_messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", padding="longest",
        pad_to_multiple_of=8
    ).to(accelerator.device)

    print(f"\nGenerating {args.num_generations} different responses for {len(batch_problems)} problem(s)...\n")

    with torch.inference_mode():
        # Adding temperature and top_p for more variation
        outputs = model.generate(
            **inputs, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temp,  # Add some randomness
            top_p=args.topp,        # Nucleus sampling
            top_k=args.topk,
            do_sample=True,    # Enable sampling
            num_return_sequences=args.num_generations
        )
        
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        

    # Assign solutions during grouping
    num = args.num_generations
    for d, i in zip(data, range(0, len(decoded_outputs), num)):
        d['gt_answer'] = extract_answer(d['solution'], args.dataset)
        d["generated_outputs"] = decoded_outputs[i:i+num]
        d["generated_solutions"] = [extract_solutions_from_generated_outputs(output) for output in d["generated_outputs"]]
        d["generated_answers"] = [extract_answer(solution, args.dataset) for solution in d["generated_solutions"]]

    return data if is_batch else data[0]


#TODO: Double check models other than Gemma generates outputs in the same format as Gemma
def extract_solutions_from_generated_outputs(generated_outputs: str) -> str:
    """Extract the solutions from the generated outputs."""
    if "model\n" in generated_outputs:
        generated_solutions = generated_outputs.split("model\n", 1)[1]
        if len(generated_solutions) == 0:
            print(f"Warning: Generated solutions is empty. This is the generated outputs: {generated_outputs}")
        return generated_solutions
    else:
        raise ValueError(f"Generated output does not contain model\n: {generated_outputs}. Double check this behavior, "
                         f"and update the extract_solutions_from_generated_outputs function.")

def extract_answer(solution: str, dataset_name: str, err_msg: Optional[str] = None) -> str:
    """Extract the answer from the solution."""
    if dataset_name == "math":
        # use provided extraction function
        answer = remove_boxed(last_boxed_only_string(solution))
        answer = answer.replace("**", "")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not answer:
        # Answer is None or empty string
        if isinstance(answer, str) and len(answer) == 0:
            # Is empty string, check if '\\boxed{}' is present (if present, extracted answer is empty string)
            if "\\boxed{}" in solution:
                return ""
        print(
            colored(f"\nWARNING in extract_answer, found no answer: {answer=} with {type(answer)=} ({dataset_name=}), "
                    f"and full solution (length {len(solution)}) is: \n{'-' * 30}\n{solution}\n{'-' * 30} (WARNING in extract_answer)\n"
                    f"{('     ERROR MESSAGE: ' + err_msg) if err_msg is not None else ''}", "yellow"))
        return None

    return answer