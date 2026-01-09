import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from tqdm import tqdm
import os
import argparse

def find_english_tokens(tokenizer, max_vocab_size=65536):
    """
    Identify tokens in the vocabulary that are English-centric by scanning the ENTIRE vocab.
    Ensures numbers, punctuation, and math symbols are preserved.
    """
    vocab = tokenizer.get_vocab()
    # Sort by ID to try to maintain some order, though we will pick based on content
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    keep_indices = []
    special_tokens = set(tokenizer.all_special_ids)
    
    print(f"🔍 Analyzing all {len(sorted_vocab)} tokens...")
    
    # Priority 1: Special tokens
    essential_indices = set(special_tokens)
    
    # Priority 2: Essential ASCII characters (numbers, punct, symbols)
    # Scan entire vocab for single-char and essential patterns
    print("   Scanning for essential characters (numbers, punctuation)...")
    for token, idx in tqdm(sorted_vocab):
        clean_token = token.replace('\u2581', ' ')
        # Keep single ASCII characters (0-9, a-z, A-Z, symbols)
        if len(clean_token) == 1:
            try:
                clean_token.encode('ascii')
                essential_indices.add(idx)
            except UnicodeEncodeError:
                pass
        # Specific search for numbers/punctuation clusters if they are bigger
        if any(c in "0123456789.,?!:;+-*/=<>^|()[]{}'\"" for c in clean_token):
             try:
                # Still check if it's mostly ASCII
                clean_token.encode('ascii')
                essential_indices.add(idx)
             except UnicodeEncodeError:
                pass

    print(f"   Found {len(essential_indices)} essential tokens.")
    
    # Priority 3: Fill remaining with English-looking tokens from the rest of the vocab
    # We prioritize lower IDs as they are usually the most fundamental merges
    remaining_slots = max_vocab_size - len(essential_indices)
    print(f"   Filling remaining {remaining_slots} slots with fundamental English merges...")
    
    for token, idx in tqdm(sorted_vocab):
        if idx in essential_indices:
            continue
            
        clean_token = token.replace('\u2581', ' ')
        try:
            clean_token.encode('ascii')
            essential_indices.add(idx)
            if len(essential_indices) >= max_vocab_size:
                break
        except UnicodeEncodeError:
            continue
            
    # Final sorted list of indices to keep
    final_indices = sorted(list(essential_indices))
    print(f"✅ Selected {len(final_indices)} English-centric tokens from entire vocab.")
    return final_indices

def prune_gemma_model(base_model_id, save_path, max_vocab_size=65536):
    """
    Full pipeline: Load, prune, and save.
    """
    print(f"📥 Loading tokenizer and model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Prune indices
    keep_indices = find_english_tokens(tokenizer, max_vocab_size=max_vocab_size)
    
    # Load model (CPU is fine for pruning weights)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    )
    
    old_embeddings = model.get_input_embeddings()
    old_lm_head = model.get_output_embeddings()
    
    new_size = len(keep_indices)
    embed_dim = old_embeddings.embedding_dim
    
    # Create new modules
    new_embeddings = nn.Embedding(new_size, embed_dim)
    new_lm_head = nn.Linear(embed_dim, new_size, bias=False)
    
    # Map weights
    print(f"✂️ Pruning embedding layers to {new_size} tokens...")
    vocab_limit = old_embeddings.weight.shape[0]
    with torch.no_grad():
        for new_idx, old_idx in enumerate(keep_indices):
            if old_idx < vocab_limit:
                new_embeddings.weight[new_idx] = old_embeddings.weight[old_idx]
                new_lm_head.weight[new_idx] = old_lm_head.weight[old_idx]
            else:
                print(f"⚠️ Warning: Index {old_idx} is out of bounds for vocab size {vocab_limit}. Filling with zeros.")
                nn.init.zeros_(new_embeddings.weight[new_idx])
                nn.init.zeros_(new_lm_head.weight[new_idx])
            
    model.set_input_embeddings(new_embeddings)
    model.set_output_embeddings(new_lm_head)
    model.config.vocab_size = new_size
    
    # Save the pruned model
    print(f"💾 Saving pruned model to: {save_path}")
    model.save_pretrained(save_path)
    
    # Save a modified tokenizer config or just reuse and filter locally
    # For now, we save the indices so the tokenizer mapping is consistent
    torch.save(keep_indices, os.path.join(save_path, "keep_indices.pt"))
    tokenizer.save_pretrained(save_path)
    
    print("✨ Pruning complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune Gemma-3 vocabulary")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-270m", help="HF Model ID")
    parser.add_argument("--save_path", type=str, default="./pruned_gemma_3_270m", help="Path to save pruned model")
    parser.add_argument("--vocab_size", type=int, default=65536, help="Target vocabulary size")
    
    args = parser.parse_args()
    
    prune_gemma_model(args.model_id, args.save_path, args.vocab_size)
