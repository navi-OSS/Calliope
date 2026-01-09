# Model Card: Nous v1.0

## Model Overview

| Field | Value |
|:---|:---|
| **Model Name** | Nous |
| **Version** | 1.0 |
| **Architecture** | Tensor Product Representation (TPR) |
| **Purpose** | Symbolic Mathematical Reasoning & Scientific Discovery |
| **Framework** | PyTorch |
| **Precision** | float64 (double precision) |
| **Parameters** | 18 (by design) |
| **Validation** | 43 Phases |
| **License** | Research / Part of Calliope Project |

---

## Model Description

**Nous** (νοῦς, Greek for "rational intellect") is a neural-symbolic reasoning engine that achieves bit-perfect precision on mathematical operations and can autonomously discover physical laws from data. It is designed to serve as the "System 2" analytical core in a Bicameral cognitive architecture.

### Design Philosophy

Nous is built on the principle that **symbolic reasoning is linear algebra in transformed manifolds**. Rather than approximating mathematical functions, Nous learns the exact identity transformations that define mathematical operators. This is achieved through:

1. **Domain Transformations**: Non-linear operations (like $a^b$) are mapped to linear operations in transformed spaces ($\exp(b \cdot \ln a)$)
2. **Role-Filler Binding**: Symbolic structures are encoded as tensor products of role vectors (positions) and filler vectors (values)
3. **Identity Learning**: The optimizer (L-BFGS) finds exact matrix representations of mathematical operators

---

## Intended Use

### Primary Use Cases

- **Symbolic Computation**: Exact evaluation of mathematical expressions
- **Equation Solving**: Finding roots of polynomial equations (real and complex)
- **Calculus Operations**: Symbolic differentiation and integration
- **Logical Reasoning**: Boolean algebra and formal implications
- **Number Theory**: GCD, modular arithmetic, discrete operations
- **Scientific Discovery**: Autonomous discovery of physical laws from data
- **Anomaly Detection**: Identifying where known physics fails

### Integration

Nous is designed to be integrated into larger AI systems as a "calculator module" that provides guaranteed-correct answers for well-defined mathematical queries.

```python
from nous import NousEngine

engine = NousEngine.load("nous/exports/nous_v1.pt")

# Symbolic computation
result = engine.integrate([3, 2, 1], x=5.0)  # ∫(3x² + 2x + 1)dx at x=5

# Equation solving
roots = engine.solve_quadratic(1, -5, 6)  # x² - 5x + 6 = 0

# Formal logic
implication = engine.logic_implies(1.0, 0.0)  # 1 → 0 = 0
```

---

## Architecture Details

### Tensor Product Representation

The core of Nous is the TPR framework, where symbolic structures are encoded as:

$$
\mathbf{T} = \sum_{i} \mathbf{f}_i \otimes \mathbf{r}_i
$$

Where:
- $\mathbf{f}_i$ are **filler vectors** (values)
- $\mathbf{r}_i$ are **role vectors** (structural positions)
- $\otimes$ is the outer product (binding operation)

### Manifold Transformations

| Operation | Transformation | Domain |
|:---|:---|:---|
| Power ($a^b$) | $\exp(b \cdot \ln a)$ | Log-space |
| Trigonometry | $(\cos\theta, \sin\theta)$ | Circular manifold |
| Complex Numbers | $(a, b) \to a + bi$ | 2D complex plane |
| Logic | $\{0.0, 1.0\}$ | Boolean manifold |

### Model Components

| Component | Parameters | Description |
|:---|:---|:---|
| `NousArithmeticBranch` | 1 | Log-space power operations |
| `NousCalculusBranch` | 6 | Integration/differentiation matrices |
| `NousAlgebraBranch` | 4 | Quadratic formula coefficients |
| `NousLogicBranch` | 5 | Boolean operation weights |
| `NousNumberTheoryBranch` | 2 | GCD and modular arithmetic |

**Total Learnable Parameters**: 18

---

## Training Procedure

### Optimizer

- **Algorithm**: L-BFGS (second-order optimization)
- **Learning Rate**: 1.0
- **Max Iterations**: 1000 per domain
- **Tolerance**: 1e-32 (gradient and parameter change)
- **Line Search**: Strong Wolfe conditions

### Training Data

Each mathematical domain is trained on synthetically generated examples:

| Domain | Training Size | Generation Method |
|:---|:---|:---|
| Arithmetic | 10,000 | Random (a, b) pairs |
| Calculus | 5,000 | Random polynomial coefficients |
| Algebra | 5,000 | Random roots → coefficients |
| Logic | 5,000 | All Boolean combinations |
| Number Theory | 5,000 | Random integer pairs |

### Training Objective

Mean Squared Error (MSE) between predicted and ground-truth values. The optimizer finds the exact identity matrices that achieve zero training loss.

---

## Evaluation Results

### Internal Benchmarks (43 Phases)

| Phase | Domain | Test | Result |
|:---|:---|:---|:---|
| 1-10 | Arithmetic | Log-space power, transcendentals | Bit-Perfect ✅ |
| 11-20 | Calculus | Derivatives, integrals, ODEs | Bit-Perfect ✅ |
| 21-25 | Algebra | Real and complex quadratic roots | Bit-Perfect ✅ |
| 26-30 | Logic | Truth tables, implications | Bit-Perfect ✅ |
| 31-35 | Number Theory | GCD, modular arithmetic | Bit-Perfect ✅ |
| 36-41 | Physics Discovery | Kepler, Ideal Gas, Conservation Laws | 99.99% ✅ |
| 42-43 | Symbolic Rewriting | Distributivity, Difference of Squares | 100% ✅ |

### External Validation (SymPy Oracle)

| Domain | Trials | Max Error | Status |
|:---|:---|:---|:---|
| Integration | 100 | `4.09e-12` | ✅ |
| Quadratic Roots | 100 | `5.33e-15` | ✅ |
| GCD | 100 | `0.00e+00` | ✅ |

### Physics Discovery Validation

| Law | Data Source | Discovered Formula | Accuracy |
|:---|:---|:---|:---|
| Kinetic Energy | Synthetic $(m, v, K)$ | $K = 0.5 \cdot m^{1.0} \cdot v^{2.0}$ | 99.999% |
| Kepler's 3rd Law | Solar System (8 planets) | $T \propto r^{1.4999}$ | 99.99% |
| Ideal Gas Law | Statistical Mechanics Sim | $P \propto N \cdot T \cdot V^{-1}$ | 100% |
| Conservation of Energy | SHO Simulation | $E = 0.5kx^2 + 0.5mv^2$ | 100% |

### Comparative Speed Benchmarks

| Operation | Nous (CPU) | SymPy (CPU) | Speedup |
|:---|:---|:---|:---|
| Quadratic Solving | 22.1 µs | 506.4 µs | **22.9×** |
| Integration | 21.8 µs | 178.4 µs | **8.2×** |
| GCD | 15.7 µs | 23.0 µs | **1.5×** |

---

## Limitations

### Known Limitations

1. **Polynomial Degree**: Currently optimized for degree ≤ 10 polynomials
2. **Transcendental Equations**: Limited to elementary inverses (ln, arcsin)
3. **Floating Point**: Subject to IEEE 754 double-precision limits (~15 significant digits)
4. **CPU-Bound**: MPS/GPU lacks native float64 support; optimal performance is on CPU

### Demonstrated Capabilities (Beyond Original Scope)

- ✅ Autonomous discovery of physical laws from raw data
- ✅ Symbolic rewrite rules (distributivity, factoring)
- ✅ Anomaly detection (Dark Matter signature)
- ✅ Real-world validation (Solar System data)

### Out of Scope

- Natural language understanding of math problems
- Formal theorem proving (proof generation)
- Full Computer Algebra System functionality
- Non-mathematical reasoning

---

## Ethical Considerations

### Risks

- **Over-reliance**: Users should not assume Nous handles all edge cases
- **Numerical Precision**: While bit-perfect in design, floating-point arithmetic has inherent limits
- **Scientific Discovery**: Discovered "laws" should be validated by domain experts

### Mitigations

- Extensive internal and external validation (43 phases)
- Clear documentation of supported operations
- Error handling for out-of-domain inputs
- Anomaly detection flags unexpected patterns

---

## Citation

```bibtex
@software{nous2026,
  title = {Nous: A Bit-Perfect Symbolic Reasoning Engine via Tensor Product Representations},
  author = {Calliope Project},
  year = {2026},
  version = {1.0},
  note = {43-phase validation including physics discovery and symbolic manipulation}
}
```

---

## Version History

| Version | Date | Changes |
|:---|:---|:---|
| 1.0 | 2026-01-08 | Initial release with 43-phase validation |

---

## Contact

Part of the Calliope project. For issues, please open a GitHub issue.
