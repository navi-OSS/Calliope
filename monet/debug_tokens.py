import torch
from transformers import AutoTokenizer
import os
from monet.tokenizer import PrunedTokenizer

def debug_mapping(text):
    base_model_path = "pruned_gemma_3_270m"
    print(f"DEBUG: Analyzing text: '{text}'")
    
    # 1. Standard Tokenizer
    std_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    std_ids = std_tokenizer(text)["input_ids"]
    
    # 2. Pruned Mapping
    indices_path = os.path.join(base_model_path, "keep_indices.pt")
    keep_indices = torch.load(indices_path)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_indices)}
    
    print(f"\n{'ID':<10} | {'Token':<20} | {'In Pruned?':<10} | {'New ID':<10}")
    print("-" * 60)
    
    for tid in std_ids:
        token = std_tokenizer.convert_ids_to_tokens(tid)
        new_id = old_to_new.get(tid, None)
        status = "✅" if new_id is not None else "❌ (Mapped to PAD)"
        print(f"{tid:<10} | {token:<20} | {status:<10} | {str(new_id):<10}")

if __name__ == "__main__":
    debug_mapping("What is the square root of 144?")
    print("\n" + "="*60 + "\n")
    debug_mapping("Once upon a time, in a world made of clockwork,")
