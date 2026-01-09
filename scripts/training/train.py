"""
Training script for the Bicameral Transformer on TinyStories.
Optimized for Google Colab T4 GPU with rich training insights.
"""
import os
import math
import time
import argparse
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from bicameral.config import BicameralConfig
from bicameral.model import BicameralTransformer
from data.tokenizer import get_tokenizer
from data.dataset import get_dataloader


# Try to import wandb for optional logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TrainingMetrics:
    """Track and display training metrics with rich insights."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.loss_history = deque(maxlen=window_size)
        self.grad_norm_history = deque(maxlen=window_size)
        self.tokens_per_sec_history = deque(maxlen=window_size)
        self.gate_history = {}  # Per-layer gate activations
        
        self.total_tokens = 0
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def update(self, loss: float, grad_norm: float, tokens: int, step_time: float):
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        self.total_tokens += tokens
        
        tokens_per_sec = tokens / step_time if step_time > 0 else 0
        self.tokens_per_sec_history.append(tokens_per_sec)
    
    def update_gates(self, gate_stats: dict):
        for layer, stats in gate_stats.items():
            if layer not in self.gate_history:
                self.gate_history[layer] = deque(maxlen=self.window_size)
            self.gate_history[layer].append(stats["mean_activation"])
    
    @property
    def avg_loss(self) -> float:
        return sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0
    
    @property
    def avg_grad_norm(self) -> float:
        return sum(self.grad_norm_history) / len(self.grad_norm_history) if self.grad_norm_history else 0
    
    @property
    def avg_tokens_per_sec(self) -> float:
        return sum(self.tokens_per_sec_history) / len(self.tokens_per_sec_history) if self.tokens_per_sec_history else 0
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time if self.start_time else 0
    
    def get_gate_summary(self) -> dict:
        """Get average gate activation per layer."""
        return {
            layer: sum(hist) / len(hist) if hist else 0 
            for layer, hist in self.gate_history.items()
        }
    
    def format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def compute_grad_norm(model: nn.Module) -> float:
    """Compute the L2 norm of gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def generate_sample(model, tokenizer, device, prompt: str = "Once upon a time", max_tokens: int = 50) -> str:
    """Generate a sample during training to monitor quality."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
        )
    
    model.train()
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def print_gate_report(gate_stats: dict, step: int):
    """Print a detailed gate activation report."""
    print(f"\n{'='*60}")
    print(f"🔀 Gate Activation Report (Step {step})")
    print(f"{'='*60}")
    
    activations = [s["mean_activation"] for s in gate_stats.values()]
    avg = sum(activations) / len(activations)
    min_act = min(activations)
    max_act = max(activations)
    
    print(f"Overall: avg={avg:.4f}, min={min_act:.4f}, max={max_act:.4f}")
    print("-" * 60)
    
    # Visual bar chart
    for layer, stats in gate_stats.items():
        act = stats["mean_activation"]
        bar_len = int(act * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  {layer}: [{bar}] {act:.4f}")
    
    print(f"{'='*60}\n")


def train(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    if device.type == "cuda":
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Tokenizer
    tokenizer = get_tokenizer("gpt2")
    
    # Model config
    config = BicameralConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_roles=args.n_roles,
        d_filler=args.d_filler,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )
    
    # Model
    model = BicameralTransformer(config)
    model = model.to(device)
    
    # Print parameter count
    param_count = model.count_parameters()
    print(f"🧠 Model parameters: {param_count / 1e6:.2f}M")
    
    # Initialize wandb if available and requested
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="bicameral-transformer",
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args),
        )
        wandb.watch(model, log="gradients", log_freq=100)
        print("📈 Weights & Biases logging enabled")
    
    # Data loader
    train_loader = get_dataloader(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        batch_size=args.batch_size,
        max_length=args.max_seq_len,
        split="train",
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )
    
    # Optimizer with separate learning rates
    # Gate parameters get higher LR to adapt faster
    gate_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'gate.gate_proj' in name:
            gate_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.learning_rate},
        {'params': gate_params, 'lr': args.learning_rate * args.gate_lr_multiplier, 'weight_decay': 0.0},
    ], betas=(0.9, 0.95), weight_decay=args.weight_decay)
    
    print(f"🔧 Gate LR: {args.learning_rate * args.gate_lr_multiplier:.2e} ({args.gate_lr_multiplier}x base)")
    
    # Mixed precision
    scaler = GradScaler(enabled=args.use_amp)
    
    # Training metrics
    metrics = TrainingMetrics(window_size=100)
    
    # Training loop
    model.train()
    step = 0
    max_steps = args.max_steps
    warmup_steps = args.warmup_steps
    accumulation_steps = args.gradient_accumulation_steps
    
    # Checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting training for {max_steps:,} steps")
    print(f"📦 Batch size: {args.batch_size}, Accumulation: {accumulation_steps}")
    print(f"📦 Effective batch size: {args.batch_size * accumulation_steps}")
    print(f"📏 Sequence length: {args.max_seq_len}")
    print(f"{'='*60}\n")
    
    pbar = tqdm(total=max_steps, desc="Training", dynamic_ncols=True)
    metrics.start()
    
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            
            step_start = time.time()
            
            # Learning rate schedule
            lr = get_lr(step, warmup_steps, max_steps, args.learning_rate, args.min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            # Move to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            batch_tokens = input_ids.numel()
            
            # Forward pass with mixed precision
            # Forward pass with mixed precision
            with autocast(enabled=args.use_amp):
                outputs = model(input_ids, labels=labels)
                lm_loss = outputs["loss"]
                
                # Auxiliary orthogonality loss
                ortho_loss = model.get_orthogonality_loss() * args.ortho_loss_weight
                
                # Total loss
                loss = (lm_loss + ortho_loss) / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            grad_norm = 0.0
            if (step + 1) % accumulation_steps == 0:
                # Compute gradient norm before clipping
                scaler.unscale_(optimizer)
                grad_norm = compute_grad_norm(model)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Update metrics
            step_time = time.time() - step_start
            metrics.update(
                loss=loss.item() * accumulation_steps,
                grad_norm=grad_norm,
                tokens=batch_tokens,
                step_time=step_time,
            )
            
            # Progress bar update
            if step % args.log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{metrics.avg_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "tok/s": f"{metrics.avg_tokens_per_sec:.0f}",
                    "grad": f"{metrics.avg_grad_norm:.2f}",
                })
                
                # Log to wandb
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "loss": metrics.avg_loss,
                        "learning_rate": lr,
                        "grad_norm": metrics.avg_grad_norm,
                        "tokens_per_sec": metrics.avg_tokens_per_sec,
                        "step": step,
                    })
            
            # Detailed gate report
            if step > 0 and step % args.gate_report_interval == 0:
                gate_stats = model.get_gate_statistics()
                metrics.update_gates(gate_stats)
                print_gate_report(gate_stats, step)
                
                if args.use_wandb and WANDB_AVAILABLE:
                    for layer, stats in gate_stats.items():
                        wandb.log({f"gate/{layer}": stats["mean_activation"], "step": step})
            
            # Sample generation
            if step > 0 and step % args.sample_interval == 0:
                print(f"\n{'='*60}")
                print(f"📝 Sample Generation (Step {step})")
                print(f"{'='*60}")
                sample = generate_sample(model, tokenizer, device, 
                                        prompt="Once upon a time, there was a little",
                                        max_tokens=80)
                print(sample)
                print(f"{'='*60}\n")
                
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({"sample": wandb.Html(f"<pre>{sample}</pre>"), "step": step})
            
            # Checkpointing
            if step > 0 and step % args.save_interval == 0:
                checkpoint_path = os.path.join(
                    args.checkpoint_dir, 
                    f"bicameral_step_{step}.pt"
                )
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "metrics": {
                        "loss": metrics.avg_loss,
                        "total_tokens": metrics.total_tokens,
                        "elapsed_time": metrics.elapsed_time,
                    }
                }, checkpoint_path)
                print(f"\n💾 Saved checkpoint to {checkpoint_path}")
            
            pbar.update(1)
            step += 1
    
    pbar.close()
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎉 Training Complete!")
    print(f"{'='*60}")
    print(f"📊 Final Loss: {metrics.avg_loss:.4f}")
    print(f"⏱️  Total Time: {metrics.format_time(metrics.elapsed_time)}")
    print(f"🔢 Total Tokens: {metrics.total_tokens:,}")
    print(f"⚡ Avg Throughput: {metrics.total_tokens / metrics.elapsed_time:.0f} tok/s")
    
    # Final gate summary
    print(f"\n🔀 Final Gate Activations:")
    gate_stats = model.get_gate_statistics()
    for layer, stats in gate_stats.items():
        print(f"  {layer}: {stats['mean_activation']:.4f}")
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "bicameral_final.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "final_metrics": {
            "loss": metrics.avg_loss,
            "total_tokens": metrics.total_tokens,
            "elapsed_time": metrics.elapsed_time,
            "gate_activations": {k: v["mean_activation"] for k, v in gate_stats.items()},
        }
    }, final_path)
    print(f"\n💾 Final model saved to {final_path}")
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Bicameral Transformer")
    
    # Model architecture
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--n_roles", type=int, default=16)
    parser.add_argument("--d_filler", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ortho_loss_weight", type=float, default=0.01,
                       help="Weight for orthogonality loss (harden symbols)")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--gate_lr_multiplier", type=float, default=10.0,
                       help="Learning rate multiplier for gate parameters (default: 10x)")
    
    # Data
    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories",
                       help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default=None,
                       help="Dataset config name (e.g. en-10k for bAbI)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Logging and checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--gate_report_interval", type=int, default=500, 
                       help="Steps between detailed gate activation reports")
    parser.add_argument("--sample_interval", type=int, default=500,
                       help="Steps between sample generation")
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    
    # Wandb
    parser.add_argument("--use_wandb", action="store_true", default=False,
                       help="Enable Weights & Biases logging")
    
    # Mixed precision
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="use_amp")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
