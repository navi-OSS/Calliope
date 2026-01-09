import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from monet.tokenizer import PrunedTokenizer
from datasets import load_dataset
import os
import tqdm

def generate_alignment_data(model_id="pruned_gemma_3_270m", num_samples=10000, max_length=128, save_dir="data/linguistic"):
    print(f"📦 Generating Linguistic Induction data from {model_id}...")
    os.makedirs(save_dir, exist_ok=True)
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f" usando dispositivo: {device}")
    
    # 1. Load Model & Tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    tokenizer = PrunedTokenizer(model_id)
    
    # 2. Load Dataset (FineWeb-Edu: The Gold Standard)
    # We use a larger sample for linguistic induction
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    # 3. Extraction Loop
    samples_saved = 0
    pbar = tqdm.tqdm(total=num_samples)
    
    with torch.no_grad():
        for sample in ds:
            text = sample["text"]
            if len(text.strip()) < 50: continue # Skip short lines
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            if inputs.input_ids.shape[1] < 10: continue
            
            # Forward pass with hidden state extraction
            outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
            
            # Save all hidden states for this sequence
            # shape: [NumLayers+1, Batch, Seq, Hidden]
            all_h = torch.stack(outputs.hidden_states).cpu()
            
            torch.save(all_h, os.path.join(save_dir, f"sample_{samples_saved}.pt"))
            
            samples_saved += 1
            pbar.update(1)
            if samples_saved >= num_samples: break
            
    print(f"\n✅ Saved {samples_saved} samples to {save_dir}")

if __name__ == "__main__":
    generate_alignment_data()
