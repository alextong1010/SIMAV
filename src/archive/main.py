from trl import GRPOConfig, GRPOTrainer
from utils.model_registry import get_full_model_name
from utils.dataset_utils import load_train_dataset, load_eval_dataset, load_domain_specific_verifiers
from src.utils.mavgrpo import MAVGRPOTrainer

import argparse
import wandb

#TODO: SET UP DISTRIBUTED TRAINING

def main():
    parser = argparse.ArgumentParser(description="Self-Improvement via GRPO and MAV")
    parser.add_argument("--model", type=str, default="gemma-3-1b-it", help="Model")
    parser.add_argument("--dataset", type=str, default="math", help="Dataset")
    parser.add_argument("--temp", type=float, default=1.0, help="Model temperature")
    parser.add_argument("--topk", type=int, default=64, help="Top-k")
    parser.add_argument("--topp", type=float, default=0.95, help="Top-p")
    parser.add_argument("--num_generations", type=int, default=8, help="Number of generations to sample")
    parser.add_argument("--log_completions", action="store_true", help="Whether to log prompt-completion pairs")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens for generation")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    args = parser.parse_args()
    
    # Get the full model name from the registry
    full_model_name = get_full_model_name(args.model)

    # Initialize wandb
    wandb_run_name = f"{args.model}-MAV-GRPO"
    # wandb.init(project="MAV-GRPO-Training", name=wandb_run_name)
    
    # Load the train and eval datasets for this dataset
    train_dataset = load_train_dataset(args.dataset)
    eval_dataset = load_eval_dataset(args.dataset)

    # Load the domain-specific verifier set for this dataset
    verifiers = load_domain_specific_verifiers(args.dataset)

    #TODO: Implement a reward function based off the verifiers
    #TODO: Subclass GRPOTrainer and add my functionalities
    # Define the reward function based off the verifiers
    def reward_func(output, **kwargs):
        raise NotImplementedError("Reward function not implemented")

    # Configure training with wandb logging enabled
    training_args = GRPOConfig(
        output_dir=f"{args.model}-MAV-GRPO", 
        logging_steps=10, 
        # use_vllm=True,
        max_prompt_length=None,  # Allow any prompt length
        max_completion_length=args.max_tokens,
        report_to="wandb",  # Enable wandb reporting
        log_completions=args.log_completions,  # Log prompt-completion pairs
        temperature=args.temp,
        top_k=args.topk,
        top_p=args.topp,
        num_generations=args.num_generations,
        num_train_epochs=args.num_train_epochs,  # Add the number of training epochs
        eval_strategy="epoch",
        save_strategy="epoch",
    )
    
    # Use the MAVGRPOTrainer instead of the standard GRPOTrainer
    # trainer = MAVGRPOTrainer(
    #     model=full_model_name,
    #     verifier_names=verifiers,  # Use the domain-specific verifiers
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     dataset_name=args.dataset,  # Add the dataset name parameter
    # )

    trainer = GRPOTrainer(
        model=full_model_name,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # dataset_name=args.dataset,  # Add the dataset name parameter
    )
    
    breakpoint()

    print(trainer.sampling_params)
    trainer.train()
    
    # Close wandb run when done
    wandb.finish()

if __name__ == "__main__":
    main()