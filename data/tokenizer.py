"""
Tokenizer wrapper for TinyStories training.
"""
from transformers import AutoTokenizer


def get_tokenizer(name: str = "gpt2"):
    """
    Get a tokenizer for training.
    
    Args:
        name: HuggingFace tokenizer name (default: gpt2)
    
    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    
    # GPT-2 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer
