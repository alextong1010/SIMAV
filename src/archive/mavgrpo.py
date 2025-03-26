from trl import GRPOTrainer, GRPOConfig
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from typing import Any, Union, Optional, Callable, List
from datasets import Dataset, IterableDataset
# Import the prompt functions
from src.prompts.gen_prompts import get_gen_prompt
from src.prompts.vera_prompts import get_vera_prompt, is_not_direct_approval, VERA_ANSWER_SYMBOL, VERA_ASK_FOR_APPROVAL_ONLY_PROMPT

class MAVGRPOTrainer(GRPOTrainer):
    """
    A version of GRPOTrainer that uses verifier prompts for reward computation.
    
    This trainer uses:
    1. Generation prompts from gen_prompts.py for generating solutions
    2. Verifier prompts from vera_prompts.py for evaluating solutions
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        verifier_names: List[str] = ["general_summarize", "general_edge", "general_mistakes", 
                                    "general_domain", "units_steps"],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        self_verify: bool = True,
        dataset_name: str = "math",
        **kwargs
    ):
        """
        Initialize the VerifierRewardGRPOTrainer.
        
        Args:
            model: The model to use for both generation and verification
            dataset_name: Name of the dataset (e.g., "math")
            verifier_names: List of verifier names to use from vera_prompts.py
            args: Configuration for this trainer
            train_dataset: Dataset to use for training
            eval_dataset: Dataset to use for evaluation
            self_verify: If True, use the same model for verification as for generation.
                        If False, use a separate verifier model (not implemented yet).
            **kwargs: Additional arguments to pass to GRPOTrainer
        """ 
        # Store the dataset name
        self.dataset_name = dataset_name
        
        # Store the self_verify flag
        self.self_verify = self_verify
        self.verifier_names = verifier_names
        
        # Check if self_verify is False and raise NotImplementedError
        if not self_verify:
            raise NotImplementedError("Using a separate verifier model is not implemented yet.")
        else:
            # Create a reward function for each verifier
            reward_funcs = [self._create_verifier_function(name) for name in verifier_names]
        # Initialize the parent class with the reward functions and weights
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            reward_funcs=reward_funcs,
            **kwargs
        )
    
    def _create_verifier_function(self, verifier_name):
        """Create a closure for a specific verifier function"""
        def verifier_reward_function(prompts, completions, **kwargs):
            return self._verifier_reward_function(verifier_name, prompts, completions, **kwargs)
        
        # Set the function name to the verifier name for better logging
        verifier_reward_function.__name__ = verifier_name
        
        return verifier_reward_function
    
    def _verifier_reward_function(self, verifier_name, prompts, completions, **kwargs):
        """
        Use the model to verify the quality of generated completions.
        
        This function formats prompts and completions into a verification prompt
        and uses the model to determine if the solution is correct.
        """
        device = self.accelerator.device
        rewards = []
        
        # Format the verification prompts
        verification_prompts = []
        for prompt, completion in zip(prompts, completions):
            # Extract the question from the prompt
            if isinstance(prompt, list):  # Conversational format
                # For conversational format, extract the last user message
                for message in reversed(prompt):
                    if message["role"] == "user":
                        question = message["content"]
                        break
                else:
                    question = ""
            else:  # Text format
                # For text format, extract the question from the generation prompt
                # This assumes the prompt follows the format in get_gen_prompt
                if "QUESTION:" in prompt:
                    question = prompt.split("QUESTION:")[1].split("\n\n")[0].strip()
                else:
                    question = prompt
            
            # Extract the solution from the completion
            if isinstance(completion, list):  # Conversational format
                solution = completion[0]["content"] if isinstance(completion[0], dict) else completion[0]
            else:  # Text format
                solution = completion
            
            # Get the verification prompt
            verification_prompt = get_vera_prompt(
                dataset_name=self.dataset_name,
                vera_name=verifier_name,
                question=question,
                solution=solution
            )
            
            # For verifiers other than direct approval, add a follow-up prompt
            if is_not_direct_approval(verifier_name):
                verification_prompt += "\n\n" + VERA_ASK_FOR_APPROVAL_ONLY_PROMPT
            
            verification_prompts.append(verification_prompt)
        
        # Use the model to generate verifications
        if self.args.use_vllm:
            # Use vLLM for verification
            all_verification_prompts = self.accelerator.gather_for_metrics(verification_prompts)
            
            if self.accelerator.is_main_process:
                # Generate verifications using vLLM
                verification_outputs = self.vllm_client.generate(
                    prompts=all_verification_prompts,
                    max_tokens=50,  # Enough tokens for the verification answer
                    temperature=0.1,  # Low temperature for more deterministic verification
                )
                
                # Extract verification results
                verification_results = []
                for output in verification_outputs:
                    # Check if the output contains the approval symbol with True
                    if f"{VERA_ANSWER_SYMBOL}True" in output:
                        verification_results.append(1.0)  # Correct solution
                    elif f"{VERA_ANSWER_SYMBOL}False" in output:
                        verification_results.append(0.0)  # Incorrect solution
                    else:
                        # If no clear answer, default to neutral
                        verification_results.append(0.5)
            else:
                verification_results = None
            
            # Broadcast verification results to all processes
            verification_results = self.accelerator.broadcast(verification_results)
            
            # Get the slice for this process
            process_slice = slice(
                self.accelerator.process_index * len(verification_prompts),
                (self.accelerator.process_index + 1) * len(verification_prompts),
            )
            rewards = verification_results[process_slice]
            
        else:
            # Use regular generation for verification
            with torch.no_grad():
                for verification_prompt in verification_prompts:
                    # Tokenize the verification prompt
                    inputs = self.processing_class(
                        text=verification_prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.processing_class.model_max_length - 50,  # Leave room for the verification
                    ).to(device)
                    
                    # Generate verification
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.1,
                        return_dict_in_generate=True,
                        output_scores=False,
                    )
                    
                    # Decode the output
                    verification_text = self.processing_class.batch_decode(
                        outputs.sequences[:, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True
                    )[0]
                    
                    # Check if the output contains the approval symbol with True
                    if f"{VERA_ANSWER_SYMBOL}True" in verification_text:
                        rewards.append(1.0)  # Correct solution
                    elif f"{VERA_ANSWER_SYMBOL}False" in verification_text:
                        rewards.append(0.0)  # Incorrect solution
                    else:
                        # If no clear answer, default to neutral
                        rewards.append(0.5)
        
        return rewards
    
    def _prepare_inputs(self, inputs):
        """
        Override to format generation prompts using get_gen_prompt.
        
        This method is called before generation to prepare the inputs.
        """
        breakpoint()
        # If inputs is a list of dictionaries (batch of examples)
        if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], dict) and "prompt" in inputs[0]:
            # Format the prompts using get_gen_prompt
            for i, example in enumerate(inputs):
                if "problem" in example:
                    problem = example["problem"]
                    inputs[i]["prompt"] = get_gen_prompt(self.dataset_name, problem)
        
        # Call the parent method to handle the rest of the preparation
        return super()._prepare_inputs(inputs)