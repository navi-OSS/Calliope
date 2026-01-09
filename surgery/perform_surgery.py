import torch
import os
from transformers import AutoModelForCausalLM, AutoConfig
from bicameral.config import BicameralConfig
from bicameral.surgery_model import BicameralSurgery

def perform_surgery(model_path, save_path):
    print(f"🏥 Starting Bicameral Surgery on: {model_path}")
    
    # 1. Load the pruned model
    # We use CPU to avoid OOM during surgery if no GPU is active
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    
    # 2. Extract dimensions for BicameralConfig
    h_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    v_size = model.config.vocab_size
    
    print(f"📊 Model Specs: {h_size} hidden, {n_layers} layers, {v_size} vocab")
    
    config = BicameralConfig(
        d_model=h_size,
        n_layers=n_layers,
        n_heads=model.config.num_attention_heads,
        d_ff=model.config.intermediate_size,
        vocab_size=v_size,
        n_roles=64,    # Increased from 32 to reach ~200M total
        d_filler=64,   
        gate_init_bias=-2.0 # Start with Gemma dominance
    )
    
    # 3. Perform Injection
    print("💉 Injecting Symbolic Lobe...")
    hybrid_model = BicameralSurgery(model, config)
    
    # 4. Verify parameter count
    total = hybrid_model.count_total_parameters()
    trainable = hybrid_model.count_trainable_parameters()
    print(f"✅ Surgery Complete!")
    print(f"   - Total Params: {total/1e6:.2f}M")
    print(f"   - Trainable (Symbolic) Params: {trainable/1e6:.2f}M")
    print(f"   - Base (Gemma) Params: {(total-trainable)/1e6:.2f}M")
    
    # 5. Test Forward Pass
    print("🧪 Testing forward pass...")
    test_input = torch.randint(0, v_size, (1, 32)).to(device)
    with torch.no_grad():
        outputs = hybrid_model(test_input)
    print(f"   - Forward Success! Logits shape: {outputs.logits.shape}")
    
    # 6. Save the Hybrid weights
    # We save as a state dict because it's a custom wrapper
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f"💾 Saving Hybrid Model to: {save_path}")
    torch.save({
        'model_state_dict': hybrid_model.state_dict(),
        'config': config
    }, os.path.join(save_path, "hybrid_model.pt"))
    
    print("✨ Surgery finished. Model is ready for Logic Training (bAbI).")

if __name__ == "__main__":
    MODEL_PATH = "./pruned_gemma_3_270m"
    SAVE_PATH = "./hybrid_gemma_3_bicameral"
    perform_surgery(MODEL_PATH, SAVE_PATH)
