import torch
import torch.nn as nn
from monet.tpr import StateSpaceExpert

def verify_vectorization():
    print("🧪 Verifying StateSpaceExpert Vectorization...")
    
    B, L, D = 2, 8, 16 # Small batch for verification
    device = "cpu"
    
    # Initialize Expert
    expert = StateSpaceExpert(d_model=D).to(device)
    
    # Random input
    x = torch.randn(B, L, D)
    
    # 1. Sequential Pass (Reference)
    # We'll temporarily patch forward to use the loop logic for comparison
    def sequential_forward(u, log_lambda):
        lamb = torch.exp(log_lambda).view(1, 1, -1)
        h = torch.zeros_like(u)
        curr_h = torch.zeros(u.shape[0], u.shape[2])
        for i in range(u.shape[1]):
            curr_h = lamb.squeeze(1) * curr_h + u[:, i, :]
            h[:, i, :] = curr_h
        return h, curr_h

    u = expert.in_proj(x)
    h_seq, next_state_seq = sequential_forward(u, expert.log_lambda)
    ref_out = expert.out_proj(h_seq)
    
    # 2. Vectorized Pass (Current Implementation)
    vec_out, next_state_vec = expert(x)
    
    # Compare
    diff_out = torch.abs(ref_out - vec_out).max().item()
    diff_state = torch.abs(next_state_seq - next_state_vec).max().item()
    
    print(f"   Max Output Diff: {diff_out:.8f}")
    print(f"   Max State Diff:  {diff_state:.8f}")
    
    if diff_out < 1e-5:
        print("✅ Vectorization Correctness Verified!")
    else:
        print("❌ Vectorization Mismatch!")
        exit(1)

if __name__ == "__main__":
    verify_vectorization()
