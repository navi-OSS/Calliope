import torch
from monet.inference import MonetEngine
import os

def diagnose():
    engine = MonetEngine()
    
    # Load Weights
    weights_path = "monet_v1/monet_model.pt"
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        for k in state_dict:
            if torch.is_tensor(state_dict[k]):
                state_dict[k] = state_dict[k].to(torch.float32)
        engine.model.load_state_dict(state_dict)

    prompt = "The capital of France is"
    
    print("\n--- TEST: Inspecting Learned Gates ---")
    for i, g in enumerate(engine.model.gates):
        val = g.item()
        sig = torch.sigmoid(g).item()
        print(f"Layer {i}: RAW={val:.4f} | SIGMOID={sig:.4f}")

    print("\n--- TEST: Active Co-Processing (Learned Gating) ---")
    # No gate forcing! Let the learned gates run.
    engine.generate(prompt, max_new_tokens=20)

if __name__ == "__main__":
    diagnose()
