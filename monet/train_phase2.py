import torch
import torch.nn as nn
import torch.optim as optim
from monet.graft import MonetModel
from monet.tpr import StructuralLobe
from transformers import AutoModelForCausalLM
import os
import glob
import tqdm

def train_phase2(model_path="monet_v1", base_model_id="pruned_gemma_3_270m", data_dir="data/linguistic", epochs=5, lr=2e-4):
    print("🚀 Starting Phase 2: Linguistic Induction (Silent Pre-Training)...")
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
    
    # Re-instantiate Lobes and Gates
    lobes = [StructuralLobe(d_model=hidden_size, tpr_dim=hidden_size).to(torch.float32) for _ in range(num_layers)]
    # Gates must match the graft.py definition involved in the saved state dict
    gates = [nn.Parameter(torch.tensor(-10.0)) for _ in range(num_layers)]
    
    model = MonetModel(base_model, lobes, gates).to(device).to(torch.float32)
    
    # Load Pre-Trained Weights (Phase 1)
    weights_path = os.path.join(model_path, "monet_model.pt")
    if os.path.exists(weights_path):
        print(f"🔄 Loading Phase 1 weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        for k in state_dict:
            if torch.is_tensor(state_dict[k]):
                state_dict[k] = state_dict[k].to(torch.float32)
        model.load_state_dict(state_dict)
    else:
        print("❌ Phase 1 weights not found! Aborting.")
        return
    
    # 2. Configure Freezing (The Surgeon's Precise Hand)
    
    # A. Freeze System 1 (Neural) - Always frozen
    for param in model.base_model.parameters():
        param.requires_grad = False
        
    # B. Freeze Main Gates (Keep them Closed/Silent at -10.0)
    for g in model.gates:
        g.data.fill_(-10.0) # Ensure they are closed
        g.requires_grad = False # DO NOT OPEN YET
        
    # C. Configure Structural Lobe Weights
    params_to_optimize = []
    
    for layer_idx, lobe in enumerate(model.lobes):
        # Unfreeze Router (Expert Gating) to allow selection of Syntax
        for p in lobe.expert_gating.parameters():
            p.requires_grad = True
            params_to_optimize.append(p)
            
        # Unfreeze Syntax Expert
        for p in lobe.syntax_expert.parameters():
            p.requires_grad = True
            params_to_optimize.append(p)
            
        # Freeze Logic & Formal Experts (Not yet)
        for p in lobe.logic_expert.parameters():
            p.requires_grad = False
        if lobe.nous:
            for p in lobe.nous.parameters(): p.requires_grad = False
        for p in lobe.to_formal.parameters(): p.requires_grad = False
        for p in lobe.from_formal.parameters(): p.requires_grad = False
        
    # Optimizer
    optimizer = optim.AdamW(params_to_optimize, lr=lr)
    criterion = nn.MSELoss()
    
    print(f"🔧 Optimizing {len(params_to_optimize)} parameters (Syntax + Routers). Gates are FROZEN closed.")
    
    # 3. Training Loop
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
                x_in = target_h[i] # Input to layer i
                
                with torch.no_grad():
                    # Calculate Neural Update (The "Teacher" Signal)
                    # N(x) = Output - Input
                    neural_update = target_h[i+1] - target_h[i]
                    
                    # --- NEURAL DROPOUT (Competence Training) ---
                    # With p=0.5, we mask the Neural Update.
                    # Case 1 (Mask=1): Target = N + 0. S learns Residual (0).
                    # Case 2 (Mask=0): Target = 0 + S. S learns N (Full Manifold).
                    # This forces S to be capable of BOTH independence and cooperation.
                    neural_mask = (torch.rand(1, device=device) > 0.5).float()
                
                # Forward pass of Lobe
                struct_pred = model.lobes[i](x_in)
                
                # Prediction = Mask * Neural + Structural
                # Ideally, this should match the Total Update (which is Neural_Update)
                # If Mask=1: N + S ~ N  => S ~ 0
                # If Mask=0: 0 + S ~ N  => S ~ N
                combined_pred = neural_mask * neural_update + struct_pred
                
                # Loss against the true target (Neural Update)
                layer_loss = criterion(combined_pred, neural_update)
                
                batch_loss += layer_loss
            
            # Backprop
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            pbar.set_postfix({"loss": batch_loss.item()})
            
        print(f"Epoch {epoch} complete. Avg Loss: {total_loss / len(data_files)}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), weights_path)
        print(f"💾 Checkpoint saved to {weights_path}")

if __name__ == "__main__":
    train_phase2()
