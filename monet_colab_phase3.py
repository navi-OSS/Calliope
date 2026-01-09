# Monet V4.0 Phase 3: Logic Grounding (Colab Edition)
# Run this script in Google Colab on a T4 GPU.

import os
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from transformers import AutoModelForCausalLM
from datasets import load_dataset
import sys

# --- 1. Environment & Path Setup ---
# Assumes 'Calliope' or 'monet' repo is cloned in /content/
if not os.path.exists("./monet"):
    print("⚠️ 'monet' directory not found. Cloning repository...")
    os.system("git clone https://github.com/thiliimanya/Calliope.git temp_repo")
    os.system("mv temp_repo/* .")
    os.system("rm -rf temp_repo")

# Install missing deps if needed
try:
    import einops
except ImportError:
    os.system("pip install einops accelerate datasets")

from monet.graft import MonetModel
from monet.tpr import StructuralLobe, StateSpaceExpert
from monet.tokenizer import PrunedTokenizer

# --- 2. CRITICAL PATCH: Vectorized State Space Expert ---
# We monkey-patch the class to ensure OOM-free vectorized execution
# This replaces the Python loop with a Parallel Scan (cumsum)

def vectorized_forward(self, x, state=None):
    """
    Vectorized Forward Pass for Linear Recurrent Unit.
    x: [Batch, Seq, Dim]
    """
    u = self.in_proj(x)
    B, L, D = u.shape
    
    if L > 1 and state is None:
        # Training/Batch mode (Vectorized Parallel Scan)
        # Formula: h_t = lambda * h_{t-1} + u_t
        # Vectorized: h_t = lambda^t * cumsum(u_t / lambda^t)
        
        # Use float32 for accumulation stability
        u_f32 = u.to(torch.float32)
        log_lamb_f32 = self.log_lambda.to(torch.float32)
        
        # exponents: [0, 1, ..., L-1]
        exponents = torch.arange(L, device=u.device, dtype=torch.float32)
        
        # log_powers: [L, D]
        log_powers = exponents.unsqueeze(1) * log_lamb_f32.unsqueeze(0)
        
        # Apply scaling: u_t * exp(-t * log_lambda)
        # We unsqueeze for batch dimension
        u_scaled = u_f32 * torch.exp(-log_powers).unsqueeze(0)
        
        # Parallel Cumulative Sum
        h_f32 = torch.cumsum(u_scaled, dim=1)
        
        # Re-apply decay: h_t * exp(t * log_lambda)
        h_f32 = h_f32 * torch.exp(log_powers).unsqueeze(0)
        
        h = h_f32.to(u.dtype)
        next_state = h[:, -1, :]
    else:
        # Step-by-step inference mode (Legacy)
        lamb = torch.exp(self.log_lambda)
        # x is [B, 1, D] or [B, D]
        prev_h = state if state is not None else torch.zeros(B, D, device=u.device, dtype=u.dtype)
        
        u_step = u[:, 0, :] if L > 1 else u.squeeze(1)
        next_state = lamb * prev_h + u_step
        h = next_state.unsqueeze(1) # shape [B, 1, D]
        
    return self.out_proj(h), next_state

# Apply Patch
print("💉 Applying Vectorized SSM Patch to StateSpaceExpert...")
StateSpaceExpert.forward = vectorized_forward

# --- 3. Training Script ---

def train_colab():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Colab GSM8K Finetune on {device}")
    
    # Config for T4 GPU (16GB VRAM)
    # Vectorized SSM is very memory efficient, so we can push BS
    batch_size = 16 
    grad_accum_steps = 4
    max_length = 512
    
    # Load Models
    # Assumes weights are uploaded to /content/
    base_model_path = "./pruned_gemma_3_270m"
    weights_missing = not (os.path.exists(os.path.join(base_model_path, "model.safetensors")) or 
                          os.path.exists(os.path.join(base_model_path, "pytorch_model.bin")))
                          
    if weights_missing:
        print("📥 Downloading Pruned Gemma Base...")
        from huggingface_hub import snapshot_download
        try:
            snapshot_download(repo_id="thiliimanya/pruned_gemma_3_270m", local_dir=base_model_path)
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: Could not download model weights.")
            print(f"   Reason: {e}")
            print("\n👉 ACTION REQUIRED:")
            print("   The script cannot find the base model weights ('model.safetensors').")
            print("   Since the repository 'thiliimanya/pruned_gemma_3_270m' might not be public/exist:")
            print("   1. Open the file browser on the left.")
            print("   2. Navigate to 'pruned_gemma_3_270m' folder.")
            print("   3. Drag and Drop your LOCAL 'model.safetensors' (511MB) into that folder.")
            print("   4. Re-run this cell.")
            sys.exit(1)

    # Double check before loading to prevent ugly Traceback
    if not (os.path.exists(os.path.join(base_model_path, "model.safetensors")) or 
            os.path.exists(os.path.join(base_model_path, "pytorch_model.bin"))):
        print("❌ Error: Directory exists but weights are still missing. Did you upload them?")
        sys.exit(1)

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
    tokenizer = PrunedTokenizer(base_model_path)
    
    hidden_size = base_model.config.hidden_size
    num_layers = len(base_model.model.layers)
    
    # Reconstruct V4.0 Lobe Structure
    lobes = nn.ModuleList([
        StructuralLobe(d_model=hidden_size, tpr_dim=hidden_size).to(torch.float16) 
        for _ in range(num_layers)
    ])
    gates = nn.ParameterList([nn.Parameter(torch.tensor(-10.0)) for _ in range(num_layers)])
    
    model = MonetModel(base_model, lobes, gates).to(device)
    
    # Freeze System 1
    base_model.eval() 
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Optimizer
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4) 
    
    # Load Aligned Weights
    weights_path = "./monet_v4_aligned.pt"
    if os.path.exists(weights_path):
        print(f"🔄 Loading Aligned Substrate: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        
        # Clean keys
        clean_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")
            clean_state_dict[new_key] = v
        
        model.load_state_dict(clean_state_dict, strict=False)
    else:
        print("⚠️ Warning: 'monet_v4_aligned.pt' not found. Starting from scratch (not recommended).")
    
    # Data
    print("🌊 Loading GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    samples = [f"Question: {item['question']}\n\nAnswer: {item['answer']}" for item in dataset]
    
    model.train()
    pbar = tqdm.tqdm(total=len(samples))
    ce_loss = nn.CrossEntropyLoss(ignore_index=0)
    
    running_loss = 0.0
    optimizer.zero_grad()
    
    print("🏁 Starting Training...")
    for batch_idx, text in enumerate(samples):
        try:
            inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            labels = inputs.input_ids.clone()
            
            model.reset_structural_cache()
            
            with torch.cuda.amp.autocast():
                logits = model(inputs.input_ids, attention_mask=inputs.attention_mask)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = ce_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                logit_scale = loss / grad_accum_steps
            
            logit_scale.backward()
            running_loss += logit_scale.item() * grad_accum_steps
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix({"loss": f"{running_loss / grad_accum_steps:.4f}"})
                running_loss = 0.0
            
            pbar.update(1)
            
            if batch_idx > 0 and batch_idx % 500 == 0:
                torch.save(model.state_dict(), "monet_v4_logic_colab_latest.pt")
                
        except Exception as e:
            print(f"⚠️ Error: {e}")
            continue

    print("✅ Finished.")
    torch.save(model.state_dict(), "monet_v4_logic_final.pt")

if __name__ == "__main__":
    train_colab()
