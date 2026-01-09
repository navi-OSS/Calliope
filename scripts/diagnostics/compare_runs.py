"""
Compare two Bicameral Transformer checkpoints.
Run A: 12% gate activation (conservative init)
Run B: 40% gate activation (higher init + gate LR)
"""
import torch
import torch.nn.functional as F
from bicameral.config import BicameralConfig
from bicameral.model import BicameralTransformer
from data.tokenizer import get_tokenizer

def load_checkpoint(path, device):
    """Load a checkpoint and return the model."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = BicameralTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint

def compare_gates(model_a, model_b):
    """Compare gate activations between two models."""
    stats_a = model_a.get_gate_statistics()
    stats_b = model_b.get_gate_statistics()
    
    print("=" * 70)
    print("🔀 Gate Activation Comparison")
    print("=" * 70)
    print(f"{'Layer':<12} {'Run A (12%)':<15} {'Run B (40%)':<15} {'Δ (B-A)':<10}")
    print("-" * 70)
    
    total_a, total_b = 0, 0
    for layer in stats_a.keys():
        act_a = stats_a[layer]["mean_activation"]
        act_b = stats_b[layer]["mean_activation"]
        delta = act_b - act_a
        total_a += act_a
        total_b += act_b
        
        arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
        print(f"{layer:<12} {act_a:<15.4f} {act_b:<15.4f} {delta:+.4f} {arrow}")
    
    avg_a = total_a / len(stats_a)
    avg_b = total_b / len(stats_b)
    print("-" * 70)
    print(f"{'Average':<12} {avg_a:<15.4f} {avg_b:<15.4f} {avg_b - avg_a:+.4f}")
    print("=" * 70)

def generate_sample(model, tokenizer, device, prompt, max_tokens=100):
    """Generate text from a prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def compare_generations(model_a, model_b, tokenizer, device, prompts):
    """Compare generations from both models on the same prompts."""
    print("\n" + "=" * 70)
    print("📝 Generation Comparison")
    print("=" * 70)
    
    for prompt in prompts:
        print(f"\n🎯 Prompt: \"{prompt}\"")
        print("-" * 70)
        
        # Set same seed for fair comparison
        torch.manual_seed(42)
        gen_a = generate_sample(model_a, tokenizer, device, prompt)
        
        torch.manual_seed(42)
        gen_b = generate_sample(model_b, tokenizer, device, prompt)
        
        print(f"\n📘 Run A (12% gates):")
        print(gen_a)
        
        print(f"\n📗 Run B (40% gates):")
        print(gen_b)
        
        print("-" * 70)

def compute_perplexity(model, tokenizer, device, texts):
    """Compute average perplexity on a set of texts."""
    total_loss = 0
    total_tokens = 0
    
    for text in texts:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs["loss"]
        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = get_tokenizer("gpt2")
    
    # Load both checkpoints
    print("\n📂 Loading checkpoints...")
    model_a, ckpt_a = load_checkpoint("final_A.pt", device)
    print(f"  Run A loaded: {model_a.count_parameters() / 1e6:.2f}M params")
    
    model_b, ckpt_b = load_checkpoint("final.pt", device)
    print(f"  Run B loaded: {model_b.count_parameters() / 1e6:.2f}M params")
    
    # Compare gates
    compare_gates(model_a, model_b)
    
    # Compare generations
    prompts = [
        "Once upon a time, there was a little girl named Lily who",
        "The boy wanted to help his friend, so he",
        "Mom said it was time to go to bed, but",
        "In the garden, there was a beautiful flower that",
    ]
    compare_generations(model_a, model_b, tokenizer, device, prompts)
    
    # Compute perplexity on test sentences
    test_texts = [
        "The little girl played with her toys in the garden.",
        "Mom gave the boy a cookie and he was very happy.",
        "The dog ran fast and caught the ball.",
        "It was a sunny day and the children went to the park.",
        "The cat slept on the warm bed all day long.",
    ]
    
    print("\n" + "=" * 70)
    print("📊 Perplexity Comparison (lower is better)")
    print("=" * 70)
    
    ppl_a = compute_perplexity(model_a, tokenizer, device, test_texts)
    ppl_b = compute_perplexity(model_b, tokenizer, device, test_texts)
    
    print(f"Run A (12% gates): {ppl_a:.2f}")
    print(f"Run B (40% gates): {ppl_b:.2f}")
    print(f"Difference: {ppl_b - ppl_a:+.2f} ({'B better' if ppl_b < ppl_a else 'A better'})")
    print("=" * 70)

if __name__ == "__main__":
    main()
