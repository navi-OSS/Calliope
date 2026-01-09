import torch
import torch.nn as nn
import torch.optim as optim
from monet.graft import MonetModel
from monet.tpr import StructuralLobe
from transformers import AutoModelForCausalLM
import os
import glob
import tqdm

def train_phase1(model_path="monet_v1", base_model_id="pruned_gemma_3_270m", data_dir="data/alignment", epochs=5, lr=1e-4):
    print("🚀 Starting Phase 1 Training: Manifold Alignment...")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f" usando dispositivo: {device}")
    
    # 1. Load Monet
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float32)
    hidden_size = base_model.config.hidden_size
    num_layers = len(base_model.model.layers)
    
    lobes = [StructuralLobe(d_model=hidden_size, tpr_dim=hidden_size).to(torch.float32) for _ in range(num_layers)]
    gates = [nn.Parameter(torch.tensor(0.1, dtype=torch.float32)) for _ in range(num_layers)] # Start with small influence
    
    model = MonetModel(base_model, lobes, gates).to(device).to(torch.float32)
    
    # Load existing weights if available (Load to CPU first, then cast)
    weights_path = os.path.join(model_path, "monet_model.pt")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        # Force float32 for all loaded tensors
        for k in state_dict:
            if torch.is_tensor(state_dict[k]):
                state_dict[k] = state_dict[k].to(torch.float32)
        model.load_state_dict(state_dict)
    
    # Freeze System 1
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    # Initialize Gates to -10.0 (sigmoid(-10) approx 0.0)
    # This starts System 2 as "Silent" (Closed), allowing gradual learned entry.
    for g in model.gates:
        g.data.fill_(-10.0)
        g.requires_grad = True # Learned Gating!
        
    # Optimizer for System 2 LOBES and GATES
    # We want to train the Gating mechanism to find its own equilibrium
    params_to_optimize = list(model.lobes.parameters()) + list(model.gates)
    optimizer = optim.AdamW(params_to_optimize, lr=lr)
    criterion = nn.MSELoss()
    
    # 2. Training Loop
    data_files = glob.glob(os.path.join(data_dir, "*.pt"))
    if not data_files:
        print(f"❌ No data found in {data_dir}. Run make_dataset.py first.")
        return

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm.tqdm(data_files, desc=f"Epoch {epoch}")
        
        for f in pbar:
            # Load cached hidden states [Layers, B, S, H]
            target_h = torch.load(f, weights_only=False).to(device).to(torch.float32)
            
            optimizer.zero_grad()
            
            batch_loss = 0
            for i in range(num_layers):
                x_in = target_h[i]
                
                # Active Synergy Objective:
                # We want S(x) to be non-zero but aligned.
                # Ideally, S(x) should be IDENTITY (x) so that adding it just scales the vector.
                # Output = X + N(X) + G*X = X(1+G) + N(X)
                # This is a safe "Active" state.
                
                struct_pred = model.lobes[i](x_in)
                
                # Identity Loss: S(x) should predict x
                layer_loss = criterion(struct_pred, x_in)
                
                batch_loss += layer_loss
            
            # Update all 18 lobes + gates together
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            pbar.set_postfix({"loss": batch_loss.item()})
            
        print(f"Epoch {epoch} complete. Avg Loss: {total_loss / len(data_files)}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), weights_path)
        print(f"💾 Checkpoint saved to {weights_path}")

if __name__ == "__main__":
    train_phase1()
