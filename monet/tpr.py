import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
try:
    from nous.model import NousModel
except ImportError:
    # Allow running without Nous for structure testing
    NousModel = None

class FourierExpert(nn.Module):
    """
    A Fourier Neural Operator (FNO) Expert for global token mixing.
    Uses FFT to provide $O(L \\log L)$ global interaction without the numerical 
    instability of SSM recurrence or the $O(L^2)$ cost of Attention.
    """
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Learnable filter in the frequency domain
        # We use RFFT, so we need max_len // 2 + 1 complex weights
        self.weight = nn.Parameter(torch.view_as_complex(torch.randn(max_len // 2 + 1, d_model, 2) * 0.02))
        
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, state=None):
        """
        x: [Batch, Seq, Dim]
        """
        u = self.in_proj(x)
        B, L, D = u.shape
        
        # --- Fourier Mixing ---
        # 1. To Frequency Domain
        x_freq = torch.fft.rfft(u, n=L, dim=1, norm="ortho")
        
        # 2. Apply Learnable Filter
        # We slice or interpolate the weights based on current L
        # For simplicity and training stability, we slice the fixed-size weights
        w_slice = self.weight[:x_freq.shape[1], :]
        x_freq = x_freq * w_slice
        
        # 3. Back to Time Domain
        out = torch.fft.irfft(x_freq, n=L, dim=1, norm="ortho")
        
        # 4. Out Projection + Jump-connection style Norm
        out = self.out_proj(out)
        return self.norm(out), None # Stateless for now

class StructuralLobe(nn.Module):
    """
    The Structural Hemisphere (System 2) for Monet.
    
    Implements: Dense Neuro-Symbolic Experts.
    Input: [Batch, Seq, 640] (Gemma hidden state)
    Output: [Batch, Seq, 640] (Integrated structural check)
    """
    def __init__(self, d_model=640, tpr_dim=640, num_roles=8): # tpr_dim defaults to full width
        super().__init__()
        self.d_model = d_model
        # We allow tpr_dim != d_model but default to identity for full fidelity
        self.tpr_dim = tpr_dim 
        
        # --- 1. Interface (Corpus Callosum) ---
        # If dimensions match, these are Identity or light adaptation layers
        if d_model != tpr_dim:
            self.adapter_in = nn.Linear(d_model, tpr_dim)
            self.adapter_out = nn.Linear(tpr_dim, d_model)
        else:
            self.adapter_in = nn.Identity()
            self.adapter_out = nn.Identity()
        
        # --- 2. Cognitive State (Roles) ---
        self.roles = nn.Parameter(torch.randn(num_roles, tpr_dim))
        
        # --- 3. Expert Gating ---
        # 3. Gating Mechanism (Learned Router)
        self.expert_gating = nn.Linear(tpr_dim, 3) # [Syntax, Logic, Formal]
        
        # --- 4. Dense Experts ---
        
        # A. Syntactic Expert (Fourier Global)
        self.syntax_expert = FourierExpert(tpr_dim)
        
        # B. Logical Expert (Fourier Global)
        self.logic_expert = FourierExpert(tpr_dim)
        
        # C. Formal Expert (Nous Branch)
        self.to_formal = nn.Linear(tpr_dim, 3) 
        self.from_formal = nn.Linear(3, tpr_dim)
        
        # EMBED NOUS MODEL AS A SUBMODULE
        if NousModel:
            self.nous = NousModel()
        else:
            self.nous = None

    def forward(self, x, state=None):
        """
        x: [Batch, Seq, D_Model]
        state: Dict[str, Tensor] containing per-expert states
        Returns: (output, next_state_dict)
        """
        # 1. Adapt to Expert Space (Identity if full width)
        expert_input = self.adapter_in(x)
        
        # 2. Compute Expert Relevance (Sigmoid for independent activation)
        relevance = torch.sigmoid(self.expert_gating(expert_input))
        
        # 3. Run Dense Experts (Fourier Mixers)
        # Note: Fourier Experts are global and currently stateless
        
        # Expert 1: Syntax
        out_syntax, _ = self.syntax_expert(expert_input)
        
        # Expert 2: Logic
        out_logic, _ = self.logic_expert(expert_input)
        
        # Expert 3: Formal (Nous Model - currently stateless)
        if self.nous:
            symbolic_input = torch.tanh(self.to_formal(expert_input)) 
            B, S, D = symbolic_input.shape
            flat_input = symbolic_input.view(-1, D).to(torch.float32)
            log_a = torch.log(torch.abs(flat_input) + 1e-15)
            log_a = torch.clamp(log_a, -10, 10) 
            formal_out_flat = torch.exp(2.0 * log_a)
            out_formal = self.from_formal(formal_out_flat.view(B, S, 3).to(expert_input.dtype))
        else:
            out_formal = expert_input
        
        # 4. Weighted Integration
        integrated_signal = (
            relevance[..., 0:1] * out_syntax +
            relevance[..., 1:2] * out_logic +
            relevance[..., 2:3] * out_formal
        )
        
        # 5. Return to Residual Stream
        output = self.adapter_out(integrated_signal)
        
        return output, {}
