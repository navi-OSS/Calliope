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
        # Initialization: 0.001 to ensure System 2 starts with infinitesimal influence.
        self.weight = nn.Parameter(torch.view_as_complex(torch.randn(max_len // 2 + 1, d_model, 2) * 0.001))
        
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
        u_f32 = u.to(torch.float32)
        
        # 1. To Frequency Domain (Safe FFT)
        x_freq = torch.fft.rfft(u_f32, n=L, dim=1, norm="ortho")
        
        # 2. Apply Learnable Filter
        w_slice = self.weight[:x_freq.shape[1], :].to(torch.complex64)
        x_freq = x_freq * w_slice
        
        # 3. Back to Time Domain
        out_f32 = torch.fft.irfft(x_freq, n=L, dim=1, norm="ortho")
        
        # 4. Out Projection
        out = self.out_proj(out_f32.to(u.dtype))
        
        # 5. Stabilization: Cleanse NaNs and Clamp outliers
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        out = torch.clamp(out, -10.0, 10.0) 
        
        return self.norm(out), None

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
        # Gating Mechanism (Learned Router)
        # Cold-Start Init: Initialize bias to -10.0 so System 2 starts INACTIVE.
        self.expert_gating = nn.Linear(tpr_dim, 3) # [Syntax, Logic, Formal]
        with torch.no_grad():
            self.expert_gating.bias.fill_(-10.0)
        
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
        
        # 2. Compute Expert Relevance 
        # Add epsilon to prevent grad nan in sigmoid
        relevance = torch.sigmoid(self.expert_gating(expert_input))
        
        # 3. Run Dense Experts (Fourier Mixers)
        # Force float32 for System 2 logic to prevent float16 overflow NaNs
        expert_input_f32 = expert_input.to(torch.float32)
        
        # Expert 1: Syntax
        out_syntax_f32, _ = self.syntax_expert(expert_input_f32)
        
        # Expert 2: Logic
        out_logic_f32, _ = self.logic_expert(expert_input_f32)
        
        # Expert 3: Formal (Bypassed for NaN Isolation)
        out_formal_f32 = expert_input_f32
        
        # 4. Weighted Integration (Stay in float32)
        rel_f32 = relevance.to(torch.float32)
        integrated_signal_f32 = (
            rel_f32[..., 0:1] * out_syntax_f32 +
            rel_f32[..., 1:2] * out_logic_f32 +
            rel_f32[..., 2:3] * out_formal_f32
        )
        
        # 5. Return to Residual Stream (Cast back to input dtype)
        output = self.adapter_out(integrated_signal_f32.to(expert_input.dtype))
        
        # Final safety cleanse before exiting to Gemma
        output = torch.nan_to_num(output, nan=0.0)
        
        return output, {}
