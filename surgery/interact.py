import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class PrunedGemmaWrapper:
    def __init__(self, model_path):
        print(f"Loading pruned model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Use float32 for CPU stability
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # Load the mapping: new_idx -> old_idx
        self.keep_indices = torch.load(os.path.join(model_path, "keep_indices.pt"))
        self.old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(self.keep_indices)}
        self.unk_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
        self.new_unk_id = self.old_to_new.get(self.unk_id, 0)
        
    def generate(self, prompt, max_new_tokens=50):
        # 1. Tokenize (Full vocab)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        
        # 2. Remap input_ids to pruned vocab
        original_ids = inputs["input_ids"][0]
        pruned_ids = []
        for oid in original_ids.tolist():
            new_id = self.old_to_new.get(oid, self.new_unk_id)
            pruned_ids.append(new_id)
        
        print(f"DEBUG: Input Len: {input_len}, Original IDs: {original_ids.tolist()[:5]}..., Pruned IDs: {pruned_ids[:5]}...")
        
        inputs["input_ids"] = torch.tensor([pruned_ids]).to(self.model.device)
        
        # 3. Generate
        output_ids = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        )
        
        # 4. Filter for NEW tokens only
        generated_new_ids = output_ids[0][input_len:].tolist()
        print(f"DEBUG: Generated {len(generated_new_ids)} tokens. IDs: {generated_new_ids[:5]}...")
        
        generated_old_ids = []
        for nid in generated_new_ids:
            if nid < len(self.keep_indices):
                generated_old_ids.append(self.keep_indices[nid])
            else:
                generated_old_ids.append(self.unk_id)
        
        return self.tokenizer.decode(generated_old_ids, skip_special_tokens=True)

if __name__ == "__main__":
    wrapper = PrunedGemmaWrapper("./pruned_gemma_3_270m")
    
    prompts = [
        "Once upon a time in a small village,",
        "The best way to cook a steak is",
        "Explain the concept of a black hole in simple terms:"
    ]
    
    for p in prompts:
        print(f"\n--- Prompt: {p} ---")
        response = wrapper.generate(p)
        print(f"Gemma: {response}")
