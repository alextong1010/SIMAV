from utils.gen_utils import generate_answers_from_loader
import json

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
    print("EVALUATION ON", total, "EXAMPLES")
    print("="*50)

    ans = generate_answers_from_loader(eval_dataset, model, tokenizer, 'math', temperature=args.temp, max_new_tokens=args.max_new_tokens, top_p=args.topp, top_k=args.topk, num_generations=args.num_generations)
    
    # Save generated answers as backup
    with open(f"backup_generated_answers_{args.run_name}.json", "w") as f:
        json.dump(ans, f)
    
    

    