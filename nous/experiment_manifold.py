"""
Phase 45d: Exact Discovery via Symbolic Parameterization

Since neural networks can't represent log exactly, we parameterize
the transformation as a SYMBOLIC form that CAN express log:

Approach: Parameterize f as a power-series or use numerical integration
of the defining property.

Key insight: log(x) = integral from 1 to x of (1/t) dt
If we learn that the integrand should be t^p where p = -1, we've
discovered log exactly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

print("🧪 Phase 45d: Exact Discovery via Derivative Property")
print("=====================================================")
print("Key Insight: The derivative of log(x) is 1/x = x^(-1)")
print("            If we discover the exponent p = -1, we've found log.")

# --- Approach: Learn the exponent of the antiderivative ---
# 
# If f'(x) = x^p, then f(x) = x^(p+1)/(p+1) for p ≠ -1
# But if p = -1, we get f(x) = log(x)
#
# We can test: Does the data satisfy f(xy) = f(x) + f(y)?
# This requires f'(x) = c/x, i.e., p = -1

# Let's discover p by checking which power makes the homomorphism work

print("\n1. Searching for the exponent p where ∫x^p dx satisfies homomorphism...")

def test_exponent(p, n_samples=1000):
    """
    If f'(x) = x^p, test how well f satisfies f(xy) = f(x) + f(y)
    
    For p ≠ -1: f(x) = x^(p+1)/(p+1)
    For p = -1: f(x) = log(x)
    """
    x = torch.rand(n_samples) * 4 + 0.5
    y = torch.rand(n_samples) * 4 + 0.5
    xy = x * y
    
    if abs(p + 1) < 1e-10:  # p ≈ -1
        f_x = torch.log(x)
        f_y = torch.log(y)
        f_xy = torch.log(xy)
    else:
        f_x = x**(p+1) / (p+1)
        f_y = y**(p+1) / (p+1)
        f_xy = (xy)**(p+1) / (p+1)
    
    # Homomorphism error: f(xy) - f(x) - f(y) should be 0
    error = torch.mean((f_xy - f_x - f_y)**2).item()
    return error

# Search over exponents
print("\n   Testing various exponents:")
best_p = None
best_error = float('inf')

for p in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]:
    error = test_exponent(p)
    status = "✅" if error < 1e-10 else ""
    print(f"   p = {p:+5.1f}: Homomorphism Error = {error:.2e} {status}")
    if error < best_error:
        best_error = error
        best_p = p

print(f"\n🎯 Best exponent: p = {best_p}")

if abs(best_p + 1) < 0.1:
    print("   This means f'(x) = 1/x, so f(x) = log(x)")
    print("\n✅ EXACT DISCOVERY: Found that log(x) is characterized by derivative 1/x")

# --- Now learn p via gradient descent ---
print("\n2. Learning p via optimization...")

class ExactManifoldModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Instead of learning p directly, learn a mixing weight between
        # log(x) and x^p for various p values.
        # This avoids the singularity at p = -1.
        
        # Actually, simpler: learn in log-space
        # p_param ∈ R, actual p = -exp(p_param) - epsilon
        # This ensures p is always negative and can approach -1
        
        # Simplest fix: Initialize p near -1 and use higher learning rate
        self.p = nn.Parameter(torch.tensor(-0.9, dtype=torch.float64))
    
    def f(self, x):
        """Antiderivative of x^p, handling p=-1 via smooth approximation"""
        p = self.p
        eps = 0.01
        
        # Use log-domain for stability when p is close to -1
        # For p != -1: f(x) = x^(p+1)/(p+1)
        # As p -> -1: this approaches log(x)
        
        # Smooth approximation using limit:
        # lim_{p->-1} x^(p+1)/(p+1) = log(x)
        # We can use: x^h/h ≈ log(x) + h/2 * (log(x))^2 + ...
        
        # For numerical stability, just use a conditional
        if torch.abs(p + 1) < eps:
            # Near the singularity, switch to log
            return torch.log(x + 1e-10)
        else:
            return x**(p+1) / (p+1)
    
    def homomorphism_loss(self, n_samples=500):
        x = torch.rand(n_samples, dtype=torch.float64) * 4 + 0.5
        y = torch.rand(n_samples, dtype=torch.float64) * 4 + 0.5
        xy = x * y
        
        f_x = self.f(x)
        f_y = self.f(y)
        f_xy = self.f(xy)
        
        return torch.mean((f_xy - f_x - f_y)**2)

model = ExactManifoldModel()
# Use L-BFGS for faster convergence
optimizer = torch.optim.LBFGS([model.p], lr=1.0, max_iter=20, line_search_fn="strong_wolfe")

for epoch in range(100):
    def closure():
        optimizer.zero_grad()
        loss = model.homomorphism_loss()
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    if epoch % 20 == 0:
        loss = model.homomorphism_loss()
        print(f"   Epoch {epoch}: p = {model.p.item():.6f}, Loss = {loss.item():.2e}")

# --- Final Result ---
final_p = model.p.item()
print(f"\n✨ Final Result:")
print(f"   Learned p = {final_p:.10f}")
print(f"   Expected p = -1.0")
print(f"   Error = {abs(final_p + 1):.2e}")

if abs(final_p + 1) < 1e-6:
    print("\n✅ EXACT DISCOVERY ACHIEVED!")
    print("   The model discovered that p = -1, meaning:")
    print("   f(x) = ∫x^(-1) dx = log(x)")
    print("   This is EXACT, not an approximation.")
elif abs(final_p + 1) < 0.01:
    print("\n⚠️ NEAR-EXACT DISCOVERY")
    print(f"   p ≈ -1 (within 1%)")
else:
    print("\n❌ FAILED to converge to p = -1")
