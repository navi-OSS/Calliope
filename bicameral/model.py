"""
Full Bicameral Transformer model for language modeling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.utils.checkpoint import checkpoint

from .config import BicameralConfig
from .layers import BicameralBlock, RMSNorm


class BicameralTransformer(nn.Module):
    """
    Bicameral Transformer - A hybrid neural-symbolic language model.
    
    Architecture combines:
    - Token embeddings (no separate positional - using RoPE in attention)
    - Stack of BicameralBlocks (each with neural + symbolic branches)
    - Final LayerNorm + output projection
    """
    
    def __init__(self, config: BicameralConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Stack of Bicameral blocks
        self.blocks = nn.ModuleList([
            BicameralBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = RMSNorm(config.d_model)
        
        # Output projection (tied with embeddings if configured)
        if config.tie_embeddings:
            self.lm_head = None  # Will use token_emb.weight
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Re-initialize gate biases (they get zeroed by _init_weights)
        # This ensures gates start LOW so neural branch dominates early
        self._init_gate_biases()
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _init_gate_biases(self):
        """
        Re-initialize gate biases to start LOW.
        
        This ensures the neural branch dominates early in training,
        with symbolic contributions growing as the model learns when they're useful.
        sigmoid(-2.0) ≈ 0.12, so gates start mostly closed.
        """
        for block in self.blocks:
            nn.init.constant_(block.gate.gate_proj.bias, self.config.gate_init_bias)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass of the Bicameral Transformer.
        
        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            labels: Optional target labels for computing loss
            mask: Optional attention mask
        
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        # Token embeddings
        x = self.token_emb(input_ids)
        x = self.drop(x)
        
        # Pass through Bicameral blocks
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                x = checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            # Tied embeddings
            logits = F.linear(x, self.token_emb.weight)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {"logits": logits, "loss": loss}
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs, shape (batch, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated token IDs, shape (batch, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            
            # Forward pass
            outputs = self(idx_cond)
            logits = outputs["logits"][:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_gate_statistics(self) -> dict:
        """
        Get statistics about the symbolic gate activations.
        Useful for understanding how much the model relies on symbolic processing.
        """
        gate_stats = {}
        for i, block in enumerate(self.blocks):
            with torch.no_grad():
                bias = block.gate.gate_proj.bias.clone()
                activation = torch.sigmoid(bias).mean().item()
                gate_stats[f"layer_{i}"] = {
                    "mean_activation": activation,
                    "bias_mean": bias.mean().item(),
                }
        return gate_stats
    
    def get_orthogonality_loss(self) -> torch.Tensor:
        """
        Compute orthogonality loss for TPR role vectors.
        Penalizes overlap between different roles to prevent variable binding bleeding.
        loss = || W @ W.T - I * diag(W @ W.T) ||^2
        """
        total_loss = 0.0
        for block in self.blocks:
            # Get role projection weights: (n_roles, d_model)
            w = block.symbolic_tpr.role_proj.weight
            
            # Compute Gram matrix: (n_roles, n_roles)
            gram = torch.matmul(w, w.t())
            
            # We want off-diagonal elements to be zero
            # Create identity mask (1s on diagonal, 0s elsewhere)
            eye = torch.eye(gram.size(0), device=gram.device)
            
            # Keep diagonal elements as is (we don't force unit norm, just orthogonality)
            # Penalize only off-diagonal elements
            # (gram * (1 - eye)) zeroes out the diagonal
            off_diagonal = gram * (1 - eye)
            
            total_loss += off_diagonal.pow(2).sum()
            
        return total_loss
