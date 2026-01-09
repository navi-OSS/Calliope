import os
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from transformers import AutoModelForCausalLM
from monet.graft import MonetModel
from monet.tpr import StructuralLobe
from monet.tokenizer import PrunedTokenizer
from datasets import load_dataset
import time

def train_final():
    # 1. Setup
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    batch_size = 1 # Minimal for 8GB RAM uniqueness
    grad_accum_steps = 32 # Simulate larger batch size
    max_length = 512
    
    print(f"🚀 Starting Local GSM8K Finetune (Device={device}) [Vectorized SSM]")
    
    # Load Models
    model_path = "./pruned_gemma_3_270m"
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    tokenizer = PrunedTokenizer(model_path)
    
    hidden_size = base_model.config.hidden_size
    num_layers = len(base_model.model.layers)
    
    # Reconstruct V4.0 Lobe Structure
    lobes = nn.ModuleList([
        StructuralLobe(d_model=hidden_size, tpr_dim=hidden_size).to(torch.float32) 
        for _ in range(num_layers)
    ])
    gates = nn.ParameterList([nn.Parameter(torch.tensor(-10.0)) for _ in range(num_layers)])
    
    model = MonetModel(base_model, lobes, gates).to(device)
    
    # Freeze System 1
    base_model.eval() 
    for param in base_model.parameters():
        param.requires_grad = False
    
    # 2. Optimization
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4) 
    
    # 3. Load Aligned Weights
    weights_path = "monet_v4_aligned.pt"
    if os.path.exists(weights_path):
        print(f"🔄 Loading Aligned Substrate: {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        
        # Clean keys (remove _orig_mod prefix from compile)
        clean_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")
            clean_state_dict[new_key] = v
        
        model.load_state_dict(clean_state_dict, strict=False)
    
    # 4. GSM8K Data
    print("🌊 Loading GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    # Formatting: Question -> Answer
    samples = [f"Question: {item['question']}\n\nAnswer: {item['answer']}" for item in dataset]
    
    # 5. Training Loop
    model.train()
    pbar = tqdm.tqdm(total=len(samples))
    ce_loss = nn.CrossEntropyLoss(ignore_index=0)
    
    print(f"🏁 Starting Phase 3 Logic Grounding on {device}...")
    optimizer.zero_grad()
    
    running_loss = 0.0
    
    for batch_idx, text in enumerate(samples):
        try:
            # Memory check: Clear MPS cache occasionally
            if batch_idx % 50 == 0:
                torch.mps.empty_cache()

            inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            labels = inputs.input_ids.clone()
            
            # Forward pass (Base model is already frozen)
            model.reset_structural_cache()
            
            # Vectorized SSM call happens inside model() -> lobes -> experts
            logits = model(inputs.input_ids, attention_mask=inputs.attention_mask)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = ce_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Gradient Accumulation
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
            
            # Save every 500 samples
            if batch_idx > 0 and batch_idx % 500 == 0:
                torch.save(model.state_dict(), "monet_v4_logic_local_latest.pt")
                
        except Exception as e:
            print(f"⚠️ Error: {e}")
            continue

    print("✅ Finished Local GSM8K Grounding.")
    torch.save(model.state_dict(), "monet_v4_logic_final.pt")

if __name__ == "__main__":
    train_final()
