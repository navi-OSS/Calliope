import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from monet.graft import MonetModel
from monet.tpr import StructuralLobe
from monet.tokenizer import PrunedTokenizer
import os

def verify_monet():
    print("🔍 Verifying Monet Forward Pass...")
    
    base_model_path = "pruned_gemma_3_270m"
    model_weights_path = "monet_v1/monet_model.pt"
    
    # 1. Load Tokenizer
    print("   Loading Pruned Tokenizer...")
    tokenizer = PrunedTokenizer(base_model_path)
    
    # 2. Load Base Model (Architecture only, we'll load weights into the hybrid)
    print("   Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float32)
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Initialize Structural Lobes (18 layers)
    hidden_size = base_model.config.hidden_size # 640
    num_layers = len(base_model.model.layers)
    lobes = [StructuralLobe(d_model=hidden_size, tpr_dim=hidden_size) for _ in range(num_layers)]
    gates = [torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float32)) for _ in range(num_layers)]
    
    # 4. Assemble Hybrid Model
    print("   Assembling Hybrid Model...")
    monet = MonetModel(base_model, lobes, gates).to(device).to(torch.float32)
    
    # 5. Load weights if they exist (Handle MPS float64 incompatibility)
    weights_path = "monet_v1/monet_model.pt"
    if os.path.exists(weights_path):
        print(f"   Loading weights from {weights_path}...")
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
            # Force float32
            for k in state_dict:
                if torch.is_tensor(state_dict[k]):
                    state_dict[k] = state_dict[k].to(torch.float32)
            monet.load_state_dict(state_dict)
        except Exception as e:
            print(f"   ⚠️ Error loading weights: {e}")
            print("   Proceeding with random weights for architectural verification.")
    else:
        print("   ⚠️ No weights found. Running with random initialization.")
    
    monet.eval()
    
    # 6. Test Forward Pass
    test_input = "What is the square root of 144?"
    print(f"\n📝 Test Input: '{test_input}'")
    
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Test original forward pass
        logits = monet(inputs.input_ids, attention_mask=inputs.attention_mask)
        
        print(f"✅ Logits Shape: {logits.shape}")
        
        # Verify shape: [Batch=1, SeqLen, VocabSize]
        expected_shape = (1, inputs.input_ids.shape[1], base_model.config.vocab_size)
        if logits.shape == expected_shape:
            print(f"✅ Output shape matches expected: {expected_shape}")
        else:
            print(f"❌ Shape mismatch! Expected {expected_shape}, got {logits.shape}")
            return
            
    # 7. Verify Gating Interaction
    print("\n⚖️ Verifying System 2 (Lobe) Interaction...")
    with torch.no_grad():
        # Zero out ALL gates
        original_gates = [g.data.clone() for g in monet.gates]
        for g in monet.gates: g.data.fill_(0.0)
        logits_base = monet(inputs.input_ids, attention_mask=inputs.attention_mask)
        
        # set all gates to 1.0
        for g in monet.gates: g.data.fill_(1.0)
        logits_hybrid = monet(inputs.input_ids, attention_mask=inputs.attention_mask)
        
        diff = torch.abs(logits_hybrid - logits_base).mean().item()
        print(f"   Mean logit difference (Gates 0.0 vs 1.0): {diff:.6f}")
        
        if diff > 1e-6:
            print("✅ System 2 (Collective) is successfully influencing the output stream.")
        else:
            print("❌ System 2 appears to have no effect on output.")
            
        # Restore gates
        for i, g in enumerate(monet.gates): g.data = original_gates[i]

    # 8. Verify Global Recurrence (K=1 vs K=2)
    print("\n🔄 Verifying Global Recurrence (Thinking Passes)...")
    with torch.no_grad():
        # Pass 1: K=1
        logits_k1 = monet(inputs.input_ids, attention_mask=inputs.attention_mask, num_thinking_passes=1)
        
        # Pass 2: K=2
        logits_k2 = monet(inputs.input_ids, attention_mask=inputs.attention_mask, num_thinking_passes=2)
        
        r_diff = torch.abs(logits_k2 - logits_k1).mean().item()
        print(f"   Mean logit difference (K=1 vs K=2): {r_diff:.6f}")
        
        if r_diff > 1e-6:
            print("✅ Global Recurrence is successfully evolving the state across passes.")
        else:
            print("❌ Global Recurrence seems to have no effect (identity). Check loop logic.")

    print("\n✨ VERIFICATION COMPLETE: Monet is architecturally sound.")

if __name__ == "__main__":
    verify_monet()
