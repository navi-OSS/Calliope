# Monet V4.0 Phase 3: Logic Grounding (Colab Edition - FOURIER PIVOT)
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
base_model_path = "./pruned_gemma_3_270m"
if not os.path.exists("./monet"):
    print("⚠️ 'monet' directory not found. Cloning repository...")
    os.system("git clone https://github.com/navi-OSS/Calliope.git temp_repo")
    os.system("cp -rv temp_repo/* .")
    os.system("rm -rf temp_repo")

# Install missing deps
try:
    import einops
except ImportError:
    os.system("pip install einops accelerate datasets")

from monet.graft import MonetModel
from monet.tpr import StructuralLobe
from monet.tokenizer import PrunedTokenizer

# --- 2. ARCHITECTURE PATCH: Fourier Neural Expert ---
# Since we pivoted from SSM to Fourier for stability (Zero loss=nan)

class FourierExpert(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.weight = nn.Parameter(torch.view_as_complex(torch.randn(max_len // 2 + 1, d_model, 2) * 0.02))
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, state=None):
        u = self.in_proj(x)
        B, L, D = u.shape
        x_freq = torch.fft.rfft(u, n=L, dim=1, norm="ortho")
        w_slice = self.weight[:x_freq.shape[1], :]
        x_freq = x_freq * w_slice
        out = torch.fft.irfft(x_freq, n=L, dim=1, norm="ortho")
        out = self.out_proj(out)
        return self.norm(out), None

# Patching the Lobe to use Fourier instead of SSM
from monet import tpr
tpr.FourierExpert = FourierExpert
print("💉 Fourier Global Expert Patched.")

# --- 3. Training Script ---

def train_colab():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Colab GSM8K Finetune (FOURIER) on {device}")
    
    # Config
    batch_size = 16 
    grad_accum_steps = 4
    max_length = 512
    
    # Ensure folders exist
    os.makedirs(base_model_path, exist_ok=True)
    
    # Check weights
    weights_missing = not (os.path.exists(os.path.join(base_model_path, "model.safetensors")) or 
                          os.path.exists(os.path.join(base_model_path, "pytorch_model.bin")))
                          
    if weights_missing:
        print("📥 Downloading Pruned Gemma Base...")
        from huggingface_hub import snapshot_download
        try:
            snapshot_download(repo_id="thiliimanya/pruned_gemma_3_270m", local_dir=base_model_path)
        except Exception as e:
            print(f"⚠️ HuggingFace download failed. Checking manual uploads...")

    # --- Auto-Rescue: Check root directory for weights ---
    if os.path.exists("model.safetensors"):
        print("📦 Found 'model.safetensors' in root. Moving to folder...")
        import shutil
        shutil.move("model.safetensors", os.path.join(base_model_path, "model.safetensors"))

    # --- Auto-Rescue: Tokenizer Indices ---
    indices_file = "keep_indices.pt"
    indices_target = os.path.join(base_model_path, indices_file)
    if not os.path.exists(indices_target):
        if os.path.exists(indices_file):
            print(f"📦 Found '{indices_file}' in root. Moving...")
            import shutil
            shutil.move(indices_file, indices_target)
        else:
            print(f"🌐 '{indices_file}' missing. Downloading from GitHub...")
            raw_url = f"https://raw.githubusercontent.com/navi-OSS/Calliope/master/pruned_gemma_3_270m/{indices_file}"
            os.system(f"curl -L {raw_url} -o {indices_target}")

    # Load Models
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
    tokenizer = PrunedTokenizer(base_model_path)
    
    hidden_size = base_model.config.hidden_size
    num_layers = len(base_model.model.layers)
    
    # Reconstruct V4.0 Lobe Structure (using Fourier Expert via Patch)
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
        
        # Strict=False because we swapped SSM for Fourier (param names will mismatch)
        model.load_state_dict(clean_state_dict, strict=False)
    
    # Data
    print("🌊 Loading GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    samples = [f"Question: {item['question']}\n\nAnswer: {item['answer']}" for item in dataset]
    import random
    random.shuffle(samples)
    
    model.train()
    pbar = tqdm.tqdm(total=len(samples))
    ce_loss = nn.CrossEntropyLoss(ignore_index=0)
    
    running_loss = 0.0
    optimizer.zero_grad()
    
    print("🏁 Starting Fourier Grounding...")
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
                loss = loss / grad_accum_steps
            
            loss.backward()
            running_loss += loss.item() * grad_accum_steps
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix({"loss": f"{running_loss / grad_accum_steps:.4f}"})
                running_loss = 0.0
            
            pbar.update(1)
            
            if batch_idx > 0 and batch_idx % 500 == 0:
                torch.save(model.state_dict(), "monet_v4_logic_fourier_latest.pt")
                
        except Exception as e:
            print(f"⚠️ Error: {e}")
            continue

    print("✅ Finished.")
    torch.save(model.state_dict(), "monet_v4_logic_fourier_final.pt")

if __name__ == "__main__":
    train_colab()
