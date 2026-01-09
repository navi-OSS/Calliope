import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from bicameral.config import BicameralConfig
from bicameral.surgery_model import BicameralSurgery

class HybridBicameralTester:
    def __init__(self, hybrid_path, base_model_path):
        print(f"🏥 Loading Hybrid Model: {hybrid_path}")
        
        # 1. Load the tokenizer and mapping
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.keep_indices = torch.load(os.path.join(base_model_path, "keep_indices.pt"))
        self.old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(self.keep_indices)}
        self.unk_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
        self.new_unk_id = self.old_to_new.get(self.unk_id, 0)
        
        # 2. Reconstruct the Hybrid Model
        # In PyTorch 2.6+, weights_only=True is forced. We must set it to False
        # to load our custom Config class, or add it to safe_globals.
        checkpoint = torch.load(os.path.join(hybrid_path, "hybrid_model.pt"), map_location="cpu", weights_only=False)
        config = checkpoint['config']
        
        # Load the base pruned model first
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        self.model = BicameralSurgery(base_model, config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("✅ Hybrid Model Loaded.")

    def generate(self, prompt, max_new_tokens=50):
        # 1. Tokenize (Full vocab)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        
        # 2. Remap to pruned indices
        pruned_ids = []
        for oid in inputs["input_ids"][0].tolist():
            pruned_ids.append(self.old_to_new.get(oid, self.new_unk_id))
        inputs["input_ids"] = torch.tensor([pruned_ids])
        
        # 3. Use the underlying gemma.generate (which now uses our wrapped layers)
        with torch.no_grad():
            output_ids = self.model.gemma.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            )
        
        # 4. Remap back to original IDs for decoding
        generated_new_ids = output_ids[0][input_len:].tolist()
        generated_old_ids = []
        for nid in generated_new_ids:
            if nid < len(self.keep_indices):
                generated_old_ids.append(self.keep_indices[nid])
            else:
                generated_old_ids.append(self.unk_id)
        
        return self.tokenizer.decode(generated_old_ids, skip_special_tokens=True)

if __name__ == "__main__":
    HYBRID_PATH = "./hybrid_gemma_3_bicameral"
    BASE_PATH = "./pruned_gemma_3_270m"
    
    tester = HybridBicameralTester(HYBRID_PATH, BASE_PATH)
    
    prompts = [
        "In a secret underground laboratory, the scientists found",
        "The fundamental law of physics states that",
        "To build a successful startup, you must first"
    ]
    
    for p in prompts:
        print(f"\n--- Prompt: {p} ---")
        res = tester.generate(p)
        print(f"Hybrid Gemma: {res}")
