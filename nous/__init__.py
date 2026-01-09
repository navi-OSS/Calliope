"""
Nous: The Symbolic Reasoning Core
=================================

A bit-perfect, infinitely extrapolating symbolic reasoning engine
built on Tensor Product Representations (TPR).

Example usage:
    >>> from nous import NousEngine
    >>> engine = NousEngine.load("nous/exports/nous_v1.pt")
    >>> roots = engine.solve_quadratic(1, -5, 6)
    >>> print(roots)  # [(2.0, 0.0), (3.0, 0.0)]
"""

from .model import (
    NousArithmeticBranch,
    NousCalculusBranch,
    NousAlgebraBranch,
    NousLogicBranch,
    NousNumberTheoryBranch,
    NousModel
)
from .inference import NousEngine

__version__ = "1.0.0"
__author__ = "Calliope Project"

__all__ = [
    "NousEngine",
    "NousModel",
    "NousArithmeticBranch",
    "NousCalculusBranch",
    "NousAlgebraBranch",
    "NousLogicBranch",
    "NousNumberTheoryBranch",
]
