# Monet: The Deep Recurrent Bicameral Model (200M)

**Status**: Draft Specification 3.2 (Scaled Depth)
**Architecture**: Iterative Bicameral Transformer with Residual Expert Blocks.
**Core Concept**: **Structural Depth** — Scaling System 2 to match the non-linear complexity of the neural base.

---

## 1. High-Level Vision

Monet implements a **Bicameral Mind**, integrating two distinct modes of cognition into a single differentiable architecture:

1.  **System 1 (The Right Hemisphere)**:
    *   **Substrate**: Gemma Transformer Layers (Neural)
    *   **Strengths**: Intuition, fluency, ambiguity, semantic association, context.
    *   **Role**: The "Narrator" that generates fluent stream of thought.

2.  **System 2 (The Left Hemisphere)**:
    *   **Substrate**: Nous TPR Module (Symbolic)
    *   **Strengths**: Syntax, logic, hierarchy, exactness, compositionality.
    *   **Role**: The "Structurer" that enforces grammatical, logical, and mathematical consistency.

The goal is not a "calculator" attached to a chatbot, but a **Coordinate System for Thought**: the transformer provides the *content*, while the TPR provides the *structure* (syntax trees, logical forms, proofs) that frames that content.

---

## 2. Architecture: Global Recurrence

Monet is a **Depth-Recurrent Model**. It doesn't just pass through its 18 layers once; it iterates the entire stack $K$ times per token.

### 2.1 The Global Recurrent Equation
For each token, the hidden state $X$ evolves through **$K$ Thinking Passes**:

$$
X_{(k+1)} = \text{BicameralStack}(X_{(k)}, \text{Context})
$$

Where $\text{BicameralStack}$ is the sequence of 18 parallel experts (Neural + Structural). 

- **Thinking Depth**: Total computational depth is $18 \times K$. 
- **Convergence**: As $k \to K$, the model moves from "Intuitive Guess" to "Structured Certainty".

### 2.2 Shared Weights, Infinite Depth
By reusing the same weights across passes, we achieve **Infinite Effective Depth** with a finite parameter count (~158M). This allows the model to process problems of arbitrary structural complexity by simply "thinking" longer.

---

## 3. The Structural Hemisphere (System 2)

System 2 is composed of **18 Residual Expert Blocks** distributed across the depth of the model. This creates a **Structural Lobe** with ~60M parameters (~30% of total model capacity).

| Component | Capacity | Ratio |
| :--- | :--- | :--- |
| **System 1 (Neural)** | 142M | 71% |
| **System 2 (Structural)** | 58M | 29% |
| **Total** | **200M** | **100%** |

### 3.1 Deep Expert Blocks
To handle complex structural reasoning, each facet (Syntax, Logic) is implemented as a **Residual MLP** with 2x expansion ($640 \to 1280 \to 640$).
1.  **Syntactic Facet**: Enforces grammatical hierarchy and narrative state.
2.  **Logical Facet**: Enforces variable binding and propositional consistency.
3.  **Formal Facet (Nous)**: Performs bit-perfect symbolic operations.

### 3.2 Components

#### A. TPR Registers (Working Memory)
Unlike a Transformer's KV cache (which is associative), TPR Registers are **positional**.
- **Roles**: Learnable vectors $r_k$ representing abstract slots (e.g., `[AGENT]`, `[ACTION]`, `[GOAL]`).
- **State**: $S = \sum f_i \otimes r_i$
- **Capacity**: Can hold recursive structures (trees/graphs).

#### B. Execution Units

1.  **Syntactic Unit (Learned)**
    *   **Task**: Maintain grammatical structure, track narrative arcs.
    *   **Mechanism**: Learned linear maps on TPR state.
    *   **Example**: "Active Voice" → "Passive Voice" transform.

2.  **Logic Unit (Mixed)**
    *   **Task**: Boolean reasoning, syllogisms.
    *   **Mechanism**: Soft-logic gates (AND/OR/NOT) defined in TPR space.
    *   **Example**: `P implies Q` + `P` → `Q`.

3.  **Math Unit (Exact Nous)**
    *   **Task**: Precise calculation, physics equations.
    *   **Mechanism**: **The Manifold Transformed Modules**.
        *   Log-space exact multiplication.
        *   Coefficient-shifting differentiation.
    *   **Trigger**: High confidence from the Instruction Head.

### 3.3 The "Bus" (Interface)

To interface 640-dim embeddings with specialized TPR manifolds:

1.  **Encoder (Write)**: $f_{fill} = \text{MLP}_{enc}(X)$
2.  **Decoder (Read)**: $X_{out} = \text{MLP}_{dec}(f_{result})$

For the **Exact Math Unit**, we effectively perform a "context switch":
- The MLP encoder learns to project "number embeddings" (like token '5') into the scalar values expected by the Math Unit.
- The Math Unit computes `5^2 = 25`.
- The MLP decoder projects the scalar `25` back into the embedding closest to token '25'.

---

## 4. Training Strategy

We train end-to-end, but with **Teacher Forcing** for the ALU instructions.

1.  **Phase 1: Autoencoder**: Train the Encoder/Decoder to map Gemma embeddings to/from TPR space.
2.  **Phase 2: Instruction Tuning**: Train the Control Logic to select the right operation (e.g., predict "Math" operation when seeing "3 + 5").
3.  **Phase 3: Joint Training**: Fine-tune the residual integration so Gemma learns to *use* the ALU's output.

---

## 5. Implementation Roadmap

1.  **Data Adapter**: Build specific datasets that exercise each Execution Unit (Narrative, Logic, Math).
2.  **TPR Branch**: Implement the `MonetTPR` PyTorch module (wrapping Nous).
3.  **Surgery**: Graft `MonetTPR` onto the pruned Gemma base.
4.  **Training**: Run the 3-phase training curriculum.
