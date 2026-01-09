import torch
import torch.nn.functional as F
from monet.inference import MonetEngine
import os

def diagnose_logits():
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
    inputs = engine.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(engine.device)
    
    print(f"\nPrompt: '{prompt}'")
    
    # 1. Run Pure Neural (Force Gates -10)
    for g in engine.model.gates: g.data.fill_(-10.0)
    with torch.no_grad():
        logits_neural = engine.model(input_ids)
        probs_neural = F.softmax(logits_neural[0, -1], dim=-1)
        top_k_neural = torch.topk(probs_neural, 5)
        
    print("\n--- Pure Neural Top-5 ---")
    for prob, idx in zip(top_k_neural.values, top_k_neural.indices):
        token = engine.tokenizer.decode([idx.item()])
        print(f"'{token}': {prob.item():.4f}")
        
    # 2. Run Hybrid (Gates -2.0)
    for g in engine.model.gates: g.data.fill_(-2.0)
    with torch.no_grad():
        logits_hybrid = engine.model(input_ids)
        probs_hybrid = F.softmax(logits_hybrid[0, -1], dim=-1)
        top_k_hybrid = torch.topk(probs_hybrid, 5)
        
    print("\n--- Hybrid (Gate=-2.0) Top-5 ---")
    for prob, idx in zip(top_k_hybrid.values, top_k_hybrid.indices):
        token = engine.tokenizer.decode([idx.item()])
        print(f"'{token}': {prob.item():.4f}")

    # 3. KL Divergence
    kl = F.kl_div(logits_hybrid[0, -1].log_softmax(dim=-1), probs_neural, reduction='sum')
    print(f"\nKL Divergence: {kl.item():.4f}")

if __name__ == "__main__":
    diagnose_logits()
