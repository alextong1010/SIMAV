# from torch.utils.data import DataLoader
# from utils.dataset_utils import collate_fn
# from datasets import Dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from tqdm import tqdm
import argparse

# def generate_answers_from_loader(
#     args: argparse.Namespace,
#     dataloader: DataLoader,
#     model: AutoModelForCausalLM,
#     tokenizer: AutoTokenizer
# ):
#     all_outputs = []

#     print(f"Generating {args.num_generations} response(s) per sample for {len(dataloader)} samples in batches of {args.batch_size}...")

#     with torch.inference_mode():
#         for inputs in tqdm(dataloader, desc="Generating", total=len(dataloader)):
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=args.max_new_tokens,
#                 temperature=args.temp,
#                 top_p=args.topp,
#                 top_k=args.topk,
#                 do_sample=True,
#                 num_return_sequences=args.num_generations
#             )

#             decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

#             batch_size = inputs["input_ids"].size(0)
#             grouped = [
#                 decoded[i * args.num_generations: (i + 1) * args.num_generations]
#                 for i in range(batch_size)
#             ]
#             all_outputs.extend(grouped)

#     return all_outputs


from prompts.gen_prompts import get_gen_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from accelerate import Accelerator

def generate_answers(
    args: argparse.Namespace,
    problem: str | list[str], 
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    accelerator: Accelerator
) -> list[str] | list[list[str]]:
    """
    Generate multiple answers for one or multiple problems.
    
    Args:
        args: Parser arguments
        problem: A single problem string or a list of problem strings
        model: The model to use for generation
        tokenizer: The tokenizer to use

    Returns:
        If problem is a string: list of generated answers
        If problem is a list: list of lists, where each inner list contains generated answers for one problem
    """

    # Handle single problem or list of problems
    is_batch = isinstance(problem, list)
    problems = problem if is_batch else [problem]
    
    # Create prompts for each problem
    user_prompts = [get_gen_prompt(args.dataset, p) for p in problems]
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

    print(f"Generating {args.num_generations} different responses for {len(problems)} problem(s)...\n")

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
        
        # Group the flat list of generations
        grouped_outputs = [
            decoded_outputs[i * args.num_generations: (i + 1) * args.num_generations]
            for i in range(len(problems))
        ]

    return grouped_outputs[0] if not is_batch else grouped_outputs