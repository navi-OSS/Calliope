"""
Core layers for the Bicameral Transformer architecture.

This module contains:
- SwiGLU: The Neural Lobe activation (always active)
- TensorProductRepresentation: The Symbolic Lobe for variable binding
- SymbolicGate: Learned confidence gate for symbolic branch
- BicameralBlock: Full layer combining both branches
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import BicameralConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) for better length generalization."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos/sin cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key tensors."""
    # q/k shape: (batch_size, n_heads, seq_len, d_head)
    # cos/sin: (seq_len, d_head) -> (1, 1, seq_len, d_head)
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with RoPE."""
    
    def __init__(self, config: BicameralConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.rotary = RotaryPositionalEmbedding(config.d_head, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Projects to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Optimized attention using Scaled Dot Product Attention (Flash Attention where available)
        # We pass is_causal=True if no mask is provided to use optimized kernels
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask, 
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=(mask is None)
        )
        
        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network - The Neural Lobe.
    
    Handles natural language fluency, probability, "vibes", narrative flow,
    and intuitive pattern matching. Always active.
    """
    
    def __init__(self, config: BicameralConfig):
        super().__init__()
        self.w_gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w_up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w_down = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down(silu(gate) * up)
        gate = self.w_gate(x)
        up = self.w_up(x)
        return self.dropout(self.w_down(F.silu(gate) * up))


class TensorProductRepresentation(nn.Module):
    """
    Tensor Product Representation (TPR) - The Symbolic Lobe.
    
    Handles variable binding (x=5), state persistence, logical constraints,
    and formal syntax. Decomposes input into Roles (slots) and Fillers (values)
    and binds them into a matrix-valued state.
    
    Binding Operation: State = Σ(Role_i ⊗ Filler_i)
    """
    
    def __init__(self, config: BicameralConfig):
        super().__init__()
        self.n_roles = config.n_roles
        self.d_filler = config.d_filler
        self.d_model = config.d_model
        
        # Project input to roles (attention over role slots)
        self.role_proj = nn.Linear(config.d_model, config.n_roles, bias=False)
        
        # Project input to fillers for each role
        self.filler_proj = nn.Linear(config.d_model, config.n_roles * config.d_filler, bias=False)
        
        # Project bound state back to model dimension
        self.output_proj = nn.Linear(config.d_filler, config.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Compute role attention weights (soft selection of role slots)
        # roles: (batch, seq, n_roles)
        roles = self.role_proj(x)
        role_weights = F.softmax(roles, dim=-1)
        
        # Compute fillers for each role
        # fillers: (batch, seq, n_roles, d_filler)
        fillers = self.filler_proj(x).view(batch_size, seq_len, self.n_roles, self.d_filler)
        
        # Binding: weighted sum of fillers (soft analog of symbolic binding)
        # state = Σ(role_weight_i * filler_i)
        # state: (batch, seq, d_filler)
        state = torch.einsum('bsr,bsrf->bsf', role_weights, fillers)
        
        # Project back to model dimension
        return self.dropout(self.output_proj(state))


class SymbolicGate(nn.Module):
    """
    Learnable gate that determines the confidence of the Symbolic branch.
    
    Initialized with negative bias so the Neural branch dominates early,
    with symbolic contributions growing as the model learns when they're useful.
    """
    
    def __init__(self, config: BicameralConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_model)
        
        # Initialize bias to make gate start near 0 (sigmoid(-2) ≈ 0.12)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, config.gate_init_bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns gate values in (0, 1) for each position and dimension
        return torch.sigmoid(self.gate_proj(x))


class BicameralBlock(nn.Module):
    """
    A single Bicameral Block combining Neural and Symbolic processing.
    
    Output = x + Attn(x) + MLP_Neural(x) + (Gate(x) · TPR_Symbolic(x))
    
    The Neural Lobe (MLP) is always active, providing the substrate of 
    language understanding. The Symbolic Lobe (TPR) is gated, contributing
    when the model learns that structured reasoning is needed.
    """
    
    def __init__(self, config: BicameralConfig):
        super().__init__()
        
        # Layer norms (using RMSNorm for efficiency)
        self.ln_attn = RMSNorm(config.d_model)
        self.ln_neural = RMSNorm(config.d_model)
        self.ln_symbolic = RMSNorm(config.d_model)
        
        # Core components
        self.attention = MultiHeadAttention(config)
        self.neural_mlp = SwiGLU(config)  # Neural Lobe
        self.symbolic_tpr = TensorProductRepresentation(config)  # Symbolic Lobe
        self.gate = SymbolicGate(config)  # Confidence gate for symbolic branch
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention (always active)
        attn_out = self.attention(self.ln_attn(x), mask)
        
        # Neural Lobe - always active
        neural_out = self.neural_mlp(self.ln_neural(x))
        
        # Symbolic Lobe - gated
        symbolic_out = self.symbolic_tpr(self.ln_symbolic(x))
        gate_values = self.gate(x)
        gated_symbolic = gate_values * symbolic_out
        
        # Combine: residual + attention + neural + gated_symbolic
        return x + attn_out + neural_out + gated_symbolic
