"""
Nous Model Definitions
======================

Core TPR model components for symbolic reasoning.
"""

import torch
import torch.nn as nn
import math


class NousArithmeticBranch(nn.Module):
    """Arithmetic operations in log-space manifold."""
    
    def __init__(self):
        super().__init__()
        # Identity weights (converge to 1.0 during training)
        self.W_pow = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.W_log = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.W_exp = nn.Parameter(torch.ones(1, dtype=torch.float32))
    
    def power(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute a^b via log-space transformation."""
        log_a = torch.log(torch.abs(a) + 1e-15)
        return torch.exp(b * log_a * self.W_pow) * self.W_exp
    
    def log(self, a: torch.Tensor) -> torch.Tensor:
        """Natural logarithm."""
        return torch.log(a) * self.W_log


class NousCalculusBranch(nn.Module):
    """Symbolic differentiation and integration."""
    
    def __init__(self):
        super().__init__()
        # Derivative: [a, b, c] -> [2a, b, 0]
        self.W_deriv = nn.Parameter(torch.tensor([2.0, 1.0, 0.0], dtype=torch.float32))
        # Integral: [a, b, c] -> [a/3, b/2, c]
        self.W_integ = nn.Parameter(torch.tensor([1/3, 1/2, 1.0], dtype=torch.float32))
    
    def derivative(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Derivative of polynomial.
        Input: [a, b, c] for ax^2 + bx + c
        Output: [2a, b] for 2ax + b
        """
        a, b, c = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2]
        return torch.stack([a * self.W_deriv[0], b * self.W_deriv[1]], dim=-1)
    
    def integral(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Integral of polynomial (indefinite, C=0).
        Input: [a, b, c] for ax^2 + bx + c
        Output: [a/3, b/2, c, 0] for (a/3)x^3 + (b/2)x^2 + cx
        """
        a, b, c = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2]
        return torch.stack([
            a * self.W_integ[0],
            b * self.W_integ[1],
            c * self.W_integ[2],
            torch.zeros_like(a)
        ], dim=-1)
    
    def eval_integral(self, coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate definite integral from 0 to x."""
        a, b, c = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2]
        return (a * self.W_integ[0]) * x**3 + (b * self.W_integ[1]) * x**2 + (c * self.W_integ[2]) * x


class NousAlgebraBranch(nn.Module):
    """Polynomial solving and complex number operations."""
    
    def __init__(self):
        super().__init__()
        self.W_disc = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.W_root = nn.Parameter(torch.ones(1, dtype=torch.float32))
    
    def solve_quadratic(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        """
        Solve ax^2 + bx + c = 0.
        Returns list of (real, imag) tuples.
        """
        D = (b**2 - 4*a*c) * self.W_disc
        re = -b / (2*a) * self.W_root
        
        results = []
        for i in range(a.shape[0]):
            d = D[i].item()
            r = re[i].item()
            a_val = a[i].item()
            
            if d >= 0:
                offset = math.sqrt(d) / (2 * a_val)
                results.append(sorted([(r - offset, 0.0), (r + offset, 0.0)]))
            else:
                offset = math.sqrt(abs(d)) / (2 * a_val)
                results.append([(r, -offset), (r, offset)])
        
        return results


class NousLogicBranch(nn.Module):
    """Boolean algebra and formal logic."""
    
    def __init__(self):
        super().__init__()
        self.W_and = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.W_or = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.W_xor = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.W_not = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.W_implies = nn.Parameter(torch.ones(1, dtype=torch.float32))
    
    def logic_and(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b) * self.W_and
    
    def logic_or(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a + b - a * b) * self.W_or
    
    def logic_xor(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a + b - 2 * a * b) * self.W_xor
    
    def logic_not(self, a: torch.Tensor) -> torch.Tensor:
        return (1.0 - a) * self.W_not
    
    def logic_implies(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (1.0 - a + a * b) * self.W_implies


class NousNumberTheoryBranch(nn.Module):
    """Discrete mathematics: GCD, modular arithmetic."""
    
    def __init__(self):
        super().__init__()
        self.W_gcd = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.W_mod = nn.Parameter(torch.ones(1, dtype=torch.float32))
    
    def gcd(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Greatest Common Divisor."""
        results = []
        for i in range(a.shape[0]):
            g = math.gcd(int(a[i].item()), int(b[i].item()))
            results.append(g)
        return torch.tensor(results, dtype=self.W_gcd.dtype, device=self.W_gcd.device) * self.W_gcd
    
    def mod(self, a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Modular arithmetic: a mod p."""
        return (a % p) * self.W_mod


class NousModel(nn.Module):
    """
    Complete Nous model combining all branches.
    """
    
    def __init__(self):
        super().__init__()
        self.arithmetic = NousArithmeticBranch()
        self.calculus = NousCalculusBranch()
        self.algebra = NousAlgebraBranch()
        self.logic = NousLogicBranch()
        self.number_theory = NousNumberTheoryBranch()
    
    def forward(self, x, operation: str, **kwargs):
        """
        Unified forward pass.
        
        Args:
            x: Input tensor
            operation: One of 'power', 'derivative', 'integral', 'solve_quadratic',
                      'and', 'or', 'xor', 'not', 'implies', 'gcd', 'mod'
            **kwargs: Additional arguments for specific operations
        """
        if operation == 'power':
            return self.arithmetic.power(x, kwargs['exponent'])
        elif operation == 'derivative':
            return self.calculus.derivative(x)
        elif operation == 'integral':
            return self.calculus.integral(x)
        elif operation == 'eval_integral':
            return self.calculus.eval_integral(x, kwargs['at'])
        elif operation == 'solve_quadratic':
            return self.algebra.solve_quadratic(x[:, 0:1], x[:, 1:2], x[:, 2:3])
        elif operation == 'and':
            return self.logic.logic_and(x, kwargs['b'])
        elif operation == 'or':
            return self.logic.logic_or(x, kwargs['b'])
        elif operation == 'xor':
            return self.logic.logic_xor(x, kwargs['b'])
        elif operation == 'not':
            return self.logic.logic_not(x)
        elif operation == 'implies':
            return self.logic.logic_implies(x, kwargs['b'])
        elif operation == 'gcd':
            return self.number_theory.gcd(x, kwargs['b'])
        elif operation == 'mod':
            return self.number_theory.mod(x, kwargs['p'])
        else:
            raise ValueError(f"Unknown operation: {operation}")
