"""
Nous Inference Engine
=====================

High-level API for using trained Nous models.
"""

import torch
import math
from pathlib import Path
from typing import List, Tuple, Union, Optional

from .model import NousModel


class NousEngine:
    """
    High-level inference engine for Nous symbolic reasoning.
    
    Example:
        >>> engine = NousEngine.load("nous/exports/nous_v1.pt")
        >>> roots = engine.solve_quadratic(1, -5, 6)
        >>> print(roots)  # [(2.0, 0.0), (3.0, 0.0)]
    """
    
    def __init__(self, model: NousModel, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "NousEngine":
        """
        Load a trained Nous model from disk.
        
        Args:
            path: Path to the .pt file
            device: Device to load the model on ('cpu' or 'cuda')
        
        Returns:
            NousEngine instance
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        model = NousModel()
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return cls(model, device)
    
    def save(self, path: Union[str, Path]):
        """
        Save the model to disk.
        
        Args:
            path: Destination path
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'metadata': {
                'version': '1.0',
                'domains': ['arithmetic', 'calculus', 'algebra', 'logic', 'number_theory'],
                'precision': 'float64'
            }
        }, path)
    
    # ========== Arithmetic ==========
    
    def power(self, base: float, exponent: float) -> float:
        """Compute base^exponent."""
        with torch.no_grad():
            a = torch.tensor([[base]], dtype=torch.float64, device=self.device)
            b = torch.tensor([[exponent]], dtype=torch.float64, device=self.device)
            result = self.model.arithmetic.power(a, b)
            return result.item()
    
    def log(self, x: float) -> float:
        """Natural logarithm."""
        with torch.no_grad():
            a = torch.tensor([[x]], dtype=torch.float64, device=self.device)
            result = self.model.arithmetic.log(a)
            return result.item()
    
    # ========== Calculus ==========
    
    def derivative(self, coeffs: List[float]) -> List[float]:
        """
        Compute derivative of polynomial.
        
        Args:
            coeffs: [a, b, c] for ax^2 + bx + c
        
        Returns:
            [2a, b] for derivative 2ax + b
        """
        with torch.no_grad():
            c = torch.tensor([coeffs], dtype=torch.float64, device=self.device)
            result = self.model.calculus.derivative(c)
            return result.squeeze().tolist()
    
    def integrate(self, coeffs: List[float], x: Optional[float] = None) -> Union[List[float], float]:
        """
        Compute integral of polynomial.
        
        Args:
            coeffs: [a, b, c] for ax^2 + bx + c
            x: If provided, evaluate definite integral from 0 to x
        
        Returns:
            If x is None: [a/3, b/2, c, 0] for indefinite integral
            If x is provided: Definite integral value
        """
        with torch.no_grad():
            c = torch.tensor([coeffs], dtype=torch.float64, device=self.device)
            
            if x is not None:
                x_t = torch.tensor([[x]], dtype=torch.float64, device=self.device)
                result = self.model.calculus.eval_integral(c, x_t)
                return result.item()
            else:
                result = self.model.calculus.integral(c)
                return result.squeeze().tolist()
    
    # ========== Algebra ==========
    
    def solve_quadratic(self, a: float, b: float, c: float) -> List[Tuple[float, float]]:
        """
        Solve ax^2 + bx + c = 0.
        
        Args:
            a, b, c: Coefficients
        
        Returns:
            List of (real, imag) tuples for each root.
            For real roots, imag = 0.0.
        """
        with torch.no_grad():
            a_t = torch.tensor([[a]], dtype=torch.float64, device=self.device)
            b_t = torch.tensor([[b]], dtype=torch.float64, device=self.device)
            c_t = torch.tensor([[c]], dtype=torch.float64, device=self.device)
            roots = self.model.algebra.solve_quadratic(a_t, b_t, c_t)
            return roots[0]  # Return first batch element
    
    # ========== Logic ==========
    
    def logic_and(self, a: float, b: float) -> float:
        """Boolean AND."""
        with torch.no_grad():
            a_t = torch.tensor([[a]], dtype=torch.float64, device=self.device)
            b_t = torch.tensor([[b]], dtype=torch.float64, device=self.device)
            result = self.model.logic.logic_and(a_t, b_t)
            return result.item()
    
    def logic_or(self, a: float, b: float) -> float:
        """Boolean OR."""
        with torch.no_grad():
            a_t = torch.tensor([[a]], dtype=torch.float64, device=self.device)
            b_t = torch.tensor([[b]], dtype=torch.float64, device=self.device)
            result = self.model.logic.logic_or(a_t, b_t)
            return result.item()
    
    def logic_xor(self, a: float, b: float) -> float:
        """Boolean XOR."""
        with torch.no_grad():
            a_t = torch.tensor([[a]], dtype=torch.float64, device=self.device)
            b_t = torch.tensor([[b]], dtype=torch.float64, device=self.device)
            result = self.model.logic.logic_xor(a_t, b_t)
            return result.item()
    
    def logic_not(self, a: float) -> float:
        """Boolean NOT."""
        with torch.no_grad():
            a_t = torch.tensor([[a]], dtype=torch.float64, device=self.device)
            result = self.model.logic.logic_not(a_t)
            return result.item()
    
    def logic_implies(self, a: float, b: float) -> float:
        """Logical implication (a => b)."""
        with torch.no_grad():
            a_t = torch.tensor([[a]], dtype=torch.float64, device=self.device)
            b_t = torch.tensor([[b]], dtype=torch.float64, device=self.device)
            result = self.model.logic.logic_implies(a_t, b_t)
            return result.item()
    
    # ========== Number Theory ==========
    
    def gcd(self, a: int, b: int) -> int:
        """Greatest Common Divisor."""
        with torch.no_grad():
            a_t = torch.tensor([a], dtype=torch.float64, device=self.device)
            b_t = torch.tensor([b], dtype=torch.float64, device=self.device)
            result = self.model.number_theory.gcd(a_t, b_t)
            return int(result.item())
    
    def mod(self, a: int, p: int) -> int:
        """Modular arithmetic: a mod p."""
        with torch.no_grad():
            a_t = torch.tensor([[a]], dtype=torch.float64, device=self.device)
            p_t = torch.tensor([[p]], dtype=torch.float64, device=self.device)
            result = self.model.number_theory.mod(a_t, p_t)
            return int(result.item())
    
    # ========== Utility ==========
    
    def __repr__(self):
        return f"NousEngine(device={self.device})"
