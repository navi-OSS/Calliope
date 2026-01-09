import torch
import os
from monet.inference import MonetEngine

def run_rigorous_tests():
    print("🔬 INITIALIZING RIGOROUS TESTING: Monet V3.2 (202M)\n")
    
    engine = MonetEngine()
    
    # Ensure float32 for MPS
    weights_path = "monet_v1/monet_model.pt"
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        for k in state_dict:
            if torch.is_tensor(state_dict[k]):
                state_dict[k] = state_dict[k].to(torch.float32)
        engine.model.load_state_dict(state_dict)
    
    prompts = [
        "123 + 456 = ",
        "The primary function of a bicameral mind is",
        "Explain the square root of 2.",
        "A logic puzzle: If A, then B. A is true. Therefore,",
        "Once upon a time in a digital garden,"
    ]
    
    results = []
    
    test_configs = [
        {"name": "K=1", "passes": 1, "gate_bias": 0.01},
        {"name": "K=2", "passes": 2, "gate_bias": 0.01},
        {"name": "S2+", "passes": 1, "gate_bias": 1.0},
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        row = {"Prompt": prompt}
        
        for config in test_configs:
            print(f"   Running {config['name']}...", end="", flush=True)
            
            # Temporarily bias gates if needed
            original_gates = []
            if config['gate_bias'] != 0.01:
                for g in engine.model.gates:
                    original_gates.append(g.data.clone())
                    g.data.fill_(config['gate_bias'])
            
            output = engine.generate(prompt, max_new_tokens=15, num_thinking_passes=config['passes'])
            
            # Restore gates
            if original_gates:
                for i, g in enumerate(engine.model.gates):
                    g.data.copy_(original_gates[i])
            
            row[config['name']] = output.replace("\n", " ")
            print(" Done.")
            
        results.append(row)
        
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    for row in results:
        print(f"\nPROMPT: {row['Prompt']}")
        print(f"  [K=1]  : {row['K=1']}")
        print(f"  [K=2]  : {row['K=2']}")
        print(f"  [S2++] : {row['S2+']}")
    
    print("\n✅ RIGOROUS TESTING COMPLETE.")

if __name__ == "__main__":
    run_rigorous_tests()

if __name__ == "__main__":
    run_rigorous_tests()
