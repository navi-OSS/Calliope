"""
Configuration for the Bicameral Transformer architecture.
"""
from dataclasses import dataclass


@dataclass
class BicameralConfig:
    """Configuration for the Bicameral Transformer model."""
    
    # Model dimensions
    d_model: int = 512           # Hidden dimension
    n_layers: int = 8            # Number of Bicameral blocks
    n_heads: int = 8             # Number of attention heads
    d_ff: int = 2048             # Feed-forward hidden dimension (4x d_model)
    
    # TPR (Tensor Product Representation) config
    n_roles: int = 16            # Number of role slots for symbolic binding
    d_filler: int = 64           # Filler vector dimension
    
    # Vocabulary and sequence
    vocab_size: int = 50257      # GPT-2 tokenizer vocab size
    max_seq_len: int = 512       # Maximum sequence length
    
    # Regularization
    dropout: float = 0.1         # Dropout probability
    
    # Training
    tie_embeddings: bool = True  # Tie input/output embeddings
    
    # Gate initialization (sigmoid(-2.0) ≈ 0.12, so logic starts small)
    gate_init_bias: float = -2.0
    
    # Memory optimization
    use_gradient_checkpointing: bool = False
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = self.d_model // self.n_heads
