import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .layers import TensorProductRepresentation, SymbolicGate, RMSNorm
from .config import BicameralConfig

class BicameralLayerWrapper(nn.Module):
    """
    Wraps a single Gemma layer and adds a parallel Symbolic branch.
    
    Output = Gemma_Layer(x) + (Gate(x) * TPR(x))
    """
    def __init__(self, base_layer: nn.Module, config: BicameralConfig):
        super().__init__()
        self.base_layer = base_layer
        
        # Symbolic components
        self.ln_symbolic = RMSNorm(config.d_model)
        self.symbolic_tpr = TensorProductRepresentation(config)
        self.gate = SymbolicGate(config)

    def __getattr__(self, name):
        """Proxy missing attributes to the underlying Gemma layer."""
        if name in ["base_layer", "ln_symbolic", "symbolic_tpr", "gate"]:
            return super().__getattr__(name)
        return getattr(self.base_layer, name)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs
    ):
        # 1. Base Gemma Layer Output
        # The base layer returns a tuple (hidden_states, optional_outputs...)
        outputs = self.base_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs
        )
        
        gemma_hidden = outputs[0]
        
        # 2. Symbolic Injection
        # We apply the symbolic branch to the original hidden_states 
        # (acting as a parallel path to the transformer block)
        symbolic_out = self.symbolic_tpr(self.ln_symbolic(hidden_states))
        gate_values = self.gate(hidden_states)
        gated_symbolic = gate_values * symbolic_out
        
        # 3. Combine
        # Result = Gemma Output + Gated Symbolic
        new_hidden = gemma_hidden + gated_symbolic
        
        # Return same structure as original layer
        if isinstance(outputs, tuple):
            return (new_hidden,) + outputs[1:]
        return new_hidden

class BicameralSurgery(nn.Module):
    """
    The full Surgery model that wraps a Gemma-3 instance.
    """
    def __init__(self, gemma_model: nn.Module, config: BicameralConfig):
        super().__init__()
        self.gemma = gemma_model
        self.config = config
        
        # Freeze internal Gemma weights
        for param in self.gemma.parameters():
            param.requires_grad = False
            
        # Bicameralize each layer
        # Depending on Gemma version, layers might be in gemma.model.layers
        # Let's assume standard HF structure
        layers = self.gemma.model.layers
        for i in range(len(layers)):
            layers[i] = BicameralLayerWrapper(layers[i], config)
            
    def forward(self, input_ids, labels=None, **kwargs):
        # Forward through the wrapped gemma
        outputs = self.gemma(input_ids=input_ids, labels=labels, **kwargs)
        
        # We can still calculate the auxiliary loss if we want
        # but the model instance is inside. 
        # A better way might be to attach it to the output.
        return outputs

    def get_orthogonality_loss(self):
        """Aggregate orthogonality loss across all injected branches."""
        total_loss = 0
        layers = self.gemma.model.layers
        for layer in layers:
            if isinstance(layer, BicameralLayerWrapper):
                # Calculate Gram matrix for the TPR role projections
                w = layer.symbolic_tpr.role_proj.weight
                gram = torch.matmul(w, w.t())
                eye = torch.eye(gram.size(0), device=gram.device)
                off_diagonal = gram * (1 - eye)
                total_loss += off_diagonal.pow(2).sum()
        return total_loss

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
