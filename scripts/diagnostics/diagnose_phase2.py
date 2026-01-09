import torch
from monet.inference import MonetEngine
import os

def diagnose_phase2():
    engine = MonetEngine(model_path="monet_v4_base")
    
    # Load V4.0 Aligned Weights
    weights_path = "monet_v4_aligned.pt"
    if os.path.exists(weights_path):
        print(f"🔄 Loading Aligned V4.0 weights from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        
        # CPU Cast for local inference
        for k in state_dict:
            if torch.is_tensor(state_dict[k]):
                state_dict[k] = state_dict[k].to(torch.float32)
                
        engine.model.load_state_dict(state_dict, strict=False)
    
    prompt = "The complex relationship between state space models and"
    
    print("\n--- TEST 1: V4.0 Grounded Baseline (Gates Closed: -10.0) ---")
    engine.model.reset_structural_cache()
    for g in engine.model.gates: g.data.fill_(-10.0)
    engine.generate(prompt, max_new_tokens=30)
    
    print("\n--- TEST 2: Gate Opening (G = -2.0 / ~0.12) ---")
    # If S approx Neural Update, this should be X + 1.12*Update
    # Should be fluent, maybe repetitive or sharp.
    engine.model.reset_structural_cache()
    for g in engine.model.gates: g.data.fill_(-2.0)
    engine.generate(prompt, max_new_tokens=20)
    
    print("\n--- TEST 3: Structural Overdrive (G = 0.0 / 0.5) ---")
    # X + 1.5*Update. Likely unstable but should be recognizable text.
    engine.model.reset_structural_cache()
    for g in engine.model.gates: g.data.fill_(0.0)
    engine.generate(prompt, max_new_tokens=20)

    print("\n--- TEST 4: PURE SYSTEM 2 (Structural Partition) ---")
    # In this mode, Neural Updates are COMPLETELY bypassed.
    # We are testing if the Lobe can generate language alone.
    engine.model.reset_structural_cache()
    engine.model.system_mode = "structural"
    engine.generate(prompt, max_new_tokens=20)

if __name__ == "__main__":
    diagnose_phase2()
