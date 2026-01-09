import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from monet.graft import MonetModel
from monet.tpr import StructuralLobe
from monet.tokenizer import PrunedTokenizer
import os

class MonetEngine:
    """
    High-level API for performing inference with the Monet hybrid model.
    """
    def __init__(self, model_path=None, base_model_id="pruned_gemma_3_270m"):
        print(f"🚀 Initializing Monet Engine (Base: {base_model_id})...")
        
        self.tokenizer = PrunedTokenizer(base_model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load Base
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float32).to(self.device)
        
        # 2. Init 18-layer Bicameral experts (V4.0 State Space)
        hidden_size = base_model.config.hidden_size
        num_layers = len(base_model.model.layers)
        lobes = [StructuralLobe(d_model=hidden_size, tpr_dim=hidden_size).to(self.device) for _ in range(num_layers)]
        gates = [torch.nn.Parameter(torch.tensor(-10.0)) for _ in range(num_layers)]
        
        # 3. Assemble
        self.model = MonetModel(base_model, lobes, gates).to(self.device)
        
        # 4. Optional Load
        if model_path:
            weights_path = os.path.join(model_path, "monet_model.pt")
            if os.path.exists(weights_path):
                print(f"   Loading weights from {weights_path}...")
                try:
                    self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=False), strict=False)
                except Exception as e:
                    print(f"   ⚠️ Weight loading failed (likely Arch mismatch): {e}")
            else:
                print(f"   ⚠️ No weights found at {weights_path}, using initial state.")
            
        self.model.eval()

    def generate(self, prompt, max_new_tokens=50, num_thinking_passes=1):
        print(f"\nPrompt: {prompt}")
        print(f"Thinking passes (K): {num_thinking_passes}")
        print("Generating: ", end="", flush=True)
        
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Run the model with recurrence
                logits = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    num_thinking_passes=num_thinking_passes
                )
                
                # Sample (Greedy for now)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                
                # Append
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones_like(next_token)
                ], dim=1)
                
                # Print token
                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                print(token_text, end="", flush=True)
                
                if next_token.item() == self.tokenizer.tokenizer.eos_token_id:
                    break
                    
        print("\n")
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    args = parser.parse_args()
    
    engine = MonetEngine()
    output = engine.generate(args.prompt)
    print(output)
