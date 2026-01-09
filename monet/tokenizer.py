import torch
from transformers import AutoTokenizer
import os

class PrunedTokenizer:
    """
    Adapter for the Gemma-3 tokenizer to work with the pruned vocabulary.
    Maps original token IDs to the new 0..65535 indices.
    """
    def __init__(self, base_model_path):
        print(f"🔄 Loading Pruned Tokenizer from {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        indices_path = os.path.join(base_model_path, "keep_indices.pt")
        if not os.path.exists(indices_path):
            raise FileNotFoundError(f"Could not find keep_indices.pt in {base_model_path}")
            
        self.keep_indices = torch.load(indices_path)
        # Create a mapping from old_id -> new_id
        # Token IDs not in keep_indices will be mapped to the [UNK] token or 0
        self.old_to_new = {old_id: new_id for new_id, old_id in enumerate(self.keep_indices)}
        
        self.unk_token_id = self.tokenizer.unk_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        
    def __call__(self, text, **kwargs):
        # 1. Get original tokenization
        outputs = self.tokenizer(text, **kwargs)
        
        # 2. Map input_ids to pruned indices
        input_ids = outputs["input_ids"]
        
        if isinstance(input_ids, torch.Tensor):
            new_input_ids = input_ids.clone()
            # We need to handle mapping for each element
            # This is slow for large batches but fine for verification/inference
            for i in range(input_ids.shape[0]):
                for j in range(input_ids.shape[1]):
                    old_id = input_ids[i, j].item()
                    new_input_ids[i, j] = self.old_to_new.get(old_id, 0) # Default to 0 (usually <pad>)
            outputs["input_ids"] = new_input_ids
        else:
            # Handle list of lists (standard tokenizer output)
            if isinstance(input_ids[0], list):
                outputs["input_ids"] = [[self.old_to_new.get(tid, 0) for tid in seq] for seq in input_ids]
            else:
                outputs["input_ids"] = [self.old_to_new.get(tid, 0) for tid in input_ids]
                
        return outputs

    def decode(self, token_ids, **kwargs):
        # 1. Map pruned indices back to original IDs
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        if isinstance(token_ids, list):
            if isinstance(token_ids[0], list):
                old_ids = [[self.keep_indices[tid] for tid in seq] for seq in token_ids]
            else:
                old_ids = [self.keep_indices[tid] for tid in token_ids]
        else:
            old_ids = self.keep_indices[token_ids]
            
        return self.tokenizer.decode(old_ids, **kwargs)
