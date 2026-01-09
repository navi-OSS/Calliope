import modal
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

# --- MODAL CONFIGURATION ---
MINUTES = 60
HOURS = 60 * MINUTES

app = modal.App("monet-phase2-chinchilla")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", 
        "transformers", 
        "datasets", 
        "einops", 
        "tqdm", 
        "accelerate",
        "numpy"
    )
    .env({"PYTHONPATH": "/root"})
    .add_local_dir("monet", remote_path="/root/monet")
    .add_local_dir("nous", remote_path="/root/nous")
    .add_local_dir("pruned_gemma_3_270m", remote_path="/root/pruned_gemma_3_270m")
    .add_local_dir("monet_v4_base", remote_path="/root/monet_v4_base")
)

vol = modal.Volume.from_name("monet-brain-v1", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB", # Upgrading to A100 for 3x speedup (~2.5 hours total)
    timeout=5 * HOURS, 
    volumes={"/data": vol},
)
def train():
    import tqdm
    from transformers import AutoModelForCausalLM
    from monet.graft import MonetModel
    from monet.tpr import StructuralLobe
    from monet.tokenizer import PrunedTokenizer
    from datasets import load_dataset, interleave_datasets
    
    # 1. Setup
    device = "cuda"
    batch_size = 16 # Ultra-stable for SSM sequential scan
    max_length = 256 # Focused Logic context
    
    print(f"🚀 Starting Ultra-Stable GSM8K Grounding (BS={batch_size}, L={max_length}) on {device}...")
    
    # Load Models
    model_path = "/root/pruned_gemma_3_270m"
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    tokenizer = PrunedTokenizer(model_path)
    
    hidden_size = base_model.config.hidden_size
    num_layers = len(base_model.model.layers)
    
    # Reconstruct V4.0 Lobe Structure (State Space)
    lobes = nn.ModuleList([
        StructuralLobe(d_model=hidden_size, tpr_dim=hidden_size).to(torch.bfloat16) 
        for _ in range(num_layers)
    ])
    gates = nn.ParameterList([nn.Parameter(torch.tensor(-10.0)) for _ in range(num_layers)])
    
    model = MonetModel(base_model, lobes, gates).to(device)
    
    # Freeze System 1 (Neural Base)
    base_model.eval() 
    for param in base_model.parameters():
        param.requires_grad = False

    # 2. Optimization
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4) 
    
    # 3. Resume Logic
    ckpt_path_v4_latest = "/data/monet_v4_latest.pt"
    ckpt_path_logic_latest = "/data/monet_v4_logic_latest.pt"
    
    if os.path.exists(ckpt_path_logic_latest):
        print(f"🔄 Resuming Logic training: {ckpt_path_logic_latest}")
        checkpoint = torch.load(ckpt_path_logic_latest, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif os.path.exists(ckpt_path_v4_latest):
        print(f"🔄 Grounding in Aligned Base: {ckpt_path_v4_latest}")
        checkpoint = torch.load(ckpt_path_v4_latest, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    # 4. GSM8K Data
    print("🌊 Loading GSM8K...")
    # Using a list instead of a stream for 8k samples - much more stable
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    
    samples = []
    for item in dataset:
        samples.append(f"Question: {item['question']}\n\nAnswer: {item['answer']}")
    
    import random
    random.shuffle(samples)

    # 5. Training Loop
    model.train()
    pbar = tqdm.tqdm(total=len(samples))
    ce_loss = nn.CrossEntropyLoss(ignore_index=0)
    
    for i in range(0, len(samples), batch_size):
        batch_texts = samples[i:i+batch_size]
        try:
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            labels = inputs.input_ids.clone()
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(inputs.input_ids, attention_mask=inputs.attention_mask)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = ce_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pbar.update(len(batch_texts))
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Periodic Save
            if i % (batch_size * 20) == 0:
                torch.save({"model_state_dict": model.state_dict()}, ckpt_path_logic_latest)
                vol.commit()
                
        except Exception as e:
            print(f"⚠️ Error: {e}")
            continue

    print("✅ Finished GSM8K Grounding.")
    torch.save(model.state_dict(), "/data/monet_v4_logic_final.pt")
    vol.commit()

if __name__ == "__main__":
    with app.run():
        train.remote()
