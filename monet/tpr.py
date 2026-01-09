import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
try:
    from nous.model import NousModel
except ImportError:
    # Allow running without Nous for structure testing
    NousModel = None

class StateSpaceExpert(nn.Module):
    """
    A Linear Recurrent Unit (LRU) for structural sequence modeling.
    Provides context awareness ($O(L)$ complexity) to the Structural Lobe.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Recurrence coefficient (Lambda). Diagonal for efficiency.
        # Initialized to allow long-range memory (near 1.0).
        self.log_lambda = nn.Parameter(torch.log(torch.ones(d_model) * 0.9))
        
        # Input/Output projections
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, state=None):
        """
        x: [Batch, Seq, Dim]
        state: [Batch, Dim] (Optional, for step-by-step inference)
        Returns: (output, next_state)
        """
        u = self.in_proj(x)
        B, L, D = u.shape
        
        if L > 1 and state is None:
            # Training/Batch mode (Vectorized Parallel Scan)
            # Formula: h_t = lambda * h_{t-1} + u_t
            # Vectorized: h_t = lambda^t * cumsum(u_t / lambda^t)
            
            # Use float32 for accumulation stability
            u_f32 = u.to(torch.float32)
            log_lamb_f32 = self.log_lambda.to(torch.float32)
            
            # exponents: [0, 1, ..., L-1]
            exponents = torch.arange(L, device=u.device, dtype=torch.float32)
            
            # log_powers: [L, D]
            log_powers = exponents.unsqueeze(1) * log_lamb_f32.unsqueeze(0)
            
            # Apply scaling: u_t * exp(-t * log_lambda)
            # We unsqueeze for batch dimension
            u_scaled = u_f32 * torch.exp(-log_powers).unsqueeze(0)
            
            # Parallel Cumulative Sum
            h_f32 = torch.cumsum(u_scaled, dim=1)
            
            # Re-apply decay: h_t * exp(t * log_lambda)
            h_f32 = h_f32 * torch.exp(log_powers).unsqueeze(0)
            
            h = h_f32.to(u.dtype)
            next_state = h[:, -1, :]
        else:
            # Step-by-step inference mode
            lamb = torch.exp(self.log_lambda)
            # x is [B, 1, D] or [B, D]
            prev_h = state if state is not None else torch.zeros(B, D, device=u.device, dtype=u.dtype)
            
            # Use only the first token if a sequence is passed in inference mode
            u_step = u[:, 0, :] if L > 1 else u.squeeze(1)
            next_state = lamb * prev_h + u_step
            h = next_state.unsqueeze(1) # shape [B, 1, D]
            
        return self.out_proj(h), next_state

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
        
        # A. Syntactic Expert (State Space)
        self.syntax_expert = StateSpaceExpert(tpr_dim)
        
        # B. Logical Expert (State Space)
        self.logic_expert = StateSpaceExpert(tpr_dim)
        
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
        
        # 3. Run Dense Experts (Stateful SSMs)
        state = state or {}
        
        # Expert 1: Syntax
        out_syntax, next_state_syntax = self.syntax_expert(expert_input, state.get("syntax"))
        
        # Expert 2: Logic
        out_logic, next_state_logic = self.logic_expert(expert_input, state.get("logic"))
        
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
        
        next_state = {
            "syntax": next_state_syntax,
            "logic": next_state_logic
        }
        
        return output, next_state
