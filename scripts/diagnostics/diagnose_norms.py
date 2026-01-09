import torch
from monet.inference import MonetEngine
import os

def diagnose_norms():
    engine = MonetEngine()
    
    # Load Weights
    weights_path = "monet_v1/monet_model.pt"
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        for k in state_dict:
            if torch.is_tensor(state_dict[k]):
                state_dict[k] = state_dict[k].to(torch.float32)
        engine.model.load_state_dict(state_dict)

    prompt = "The capital of France is Paris."
    # We can't rely on manual forwarding because Gemma-3 requires positional embeddings.
    # Instead, we will rely on Hooks to capture the data during a real forward pass.
    
    updates = {} # Layer -> {'conv_in': val, 'neural_update': val, 'struct_update': val}
    
    layer_idx = 5
    lobe = engine.model.lobes[layer_idx]
    
    # We need to monkey-patch the Lobe forward to capture its output
    # And we also need to capture the layer input/output.
    
    # Actually, let's just make a new hook that wraps the existing hook logic?
    # Or just replace the hook on layer 5.
    
    original_hook = engine.model.base_model.model.layers[layer_idx]._forward_hooks[layer_idx]
    
    # Remove original hook
    del engine.model.base_model.model.layers[layer_idx]._forward_hooks[layer_idx]
    
    def diagnostic_hook(module, layer_input, layer_output):
        # Replicate logic but capture norms
        
        # Neural Update Extraction
        # layer_output[0] is X + Neural(X)
        # layer_input[0] is X
        
        x = layer_input[0]
        neural_out = layer_output[0]
        neural_update = neural_out - x
        
        struct_update = lobe(x)
        
        print(f"\n📊 Layer {layer_idx} Analysis:")
        print(f"   Shape: {x.shape}")
        
        n_norm = neural_update.norm(dim=-1).mean().item()
        s_norm = struct_update.norm(dim=-1).mean().item()
        
        print(f"   ||Neural Update||:     {n_norm:.4f}")
        print(f"   ||Structural Update||: {s_norm:.4f}")
        
        if s_norm > 0:
            ratio = s_norm / n_norm
            print(f"   Ratio (Struct/Neural): {ratio:.4f}")
            if ratio > 5.0:
                print("   ⚠️  CRITICAL: Structural update is too large!")
            elif ratio < 0.2:
                print("   ⚠️  CRITICAL: Structural update is too small!")
            else:
                print("   ✅ Scale is aligned.")
        
        # We must return the integrated output to keep the model running (though we only need 1 step)
        # Convex: X + (1-g)N + gS
        gate = engine.model.gates[layer_idx]
        g = torch.sigmoid(gate)
        integrated = (1.0 - g) * neural_update + g * struct_update
        
        if isinstance(layer_output, tuple):
            return (x + integrated,) + layer_output[1:]
        else:
            return x + integrated

    # Register new hook
    engine.model.base_model.model.layers[layer_idx].register_forward_hook(diagnostic_hook)
    
    print("\nRunning forward pass...")
    engine.generate(prompt, max_new_tokens=1)

if __name__ == "__main__":
    diagnose_norms()
