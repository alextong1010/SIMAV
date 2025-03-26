"""
Registry of supported models and their full HuggingFace paths.
"""

MODEL_REGISTRY = {
    # Gemma models
    "gemma-3-1b-it": "google/gemma-3-1b-it",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    
    # Llama models
    
    # Mistral models
    
    # Other models
}

def get_full_model_name(short_name):
    """
    Get the full model name from the registry.
    
    Args:
        short_name (str): The short name of the model.
        
    Returns:
        str: The full model name.
        
    Raises:
        ValueError: If the model is not found in the registry.
    """
    if short_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[short_name]
    
    #TODO: Implement a way to check if it's a custom model (i.e. a path to a model)
        
    raise ValueError(f"Model '{short_name}' not found in the registry. Available models: {list(MODEL_REGISTRY.keys())}") 