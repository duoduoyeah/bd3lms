# Mathematical Background: Loss Functions in Diffusion Language Models

This document provides a **high-level introduction** to the mathematical foundations behind the loss functions used in BD3LMs (Block Discrete Denoising Diffusion Language Models).

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [What is KL Divergence?](#2-what-is-kl-divergence)
3. [What is the ELBO?](#3-what-is-the-elbo)
4. [From ELBO to the Diffusion Loss](#4-from-elbo-to-the-diffusion-loss)
5. [The Simplified BD3LM Loss](#5-the-simplified-bd3lm-loss)
6. [Summary](#6-summary)

---

## 1. The Big Picture

### The Goal

We want to train a model that learns the probability distribution of real data (text). Given a sequence of tokens `x`, we want:

```
Maximize: log p_θ(x)   (the log-likelihood of real data)
```

**Problem**: For diffusion models, computing `log p_θ(x)` directly is intractable (too hard to compute).

**Solution**: Use a **lower bound** that we CAN compute. This is called the **ELBO**.

### The Key Insight

```
-log p_θ(x) ≤ ELBO(x; θ)
         ↑              ↑
   What we want    What we can compute
   to minimize     (upper bound on negative log-likelihood)
```

If we minimize the ELBO, we're guaranteed to be improving (or at least not hurting) the true log-likelihood.

---

## 2. What is KL Divergence?

### Intuition

**KL Divergence** measures how "different" two probability distributions are.

```
D_KL[P || Q] = "How much information is lost when Q is used to approximate P"
```

### Properties

| Property | Meaning |
|----------|---------|
| `D_KL[P || Q] ≥ 0` | Always non-negative |
| `D_KL[P || Q] = 0` | If and only if P = Q (distributions are identical) |
| `D_KL[P || Q] ≠ D_KL[Q || P]` | **Not symmetric!** Order matters |

### Formula (Discrete)

```
D_KL[P || Q] = Σ P(x) log(P(x) / Q(x))
```

### Example: Coin Flip

Suppose:
- P = fair coin: P(heads) = 0.5, P(tails) = 0.5
- Q = biased coin: Q(heads) = 0.9, Q(tails) = 0.1

```
D_KL[P || Q] = 0.5 × log(0.5/0.9) + 0.5 × log(0.5/0.1)
             = 0.5 × (-0.85) + 0.5 × (2.32)
             ≈ 0.74 bits
```

This tells us: using the biased coin Q to model the fair coin P loses ~0.74 bits of information.

### Why KL Divergence in Diffusion?

In diffusion models, we use KL divergence to measure:
```
D_KL[q(x_{s} | x_{t}, x) || p_θ(x_{s} | x_{t})]
      ↑                        ↑
  True reverse process    Model's learned reverse
```

We want our model `p_θ` to match the true reverse process `q`.

---

## 3. What is the ELBO?

### ELBO = Evidence Lower BOund

The "evidence" is `p(x)` — the probability of observing the data. The ELBO gives us a **lower bound** on `log p(x)`.

### Why Do We Need It?

For diffusion models, we introduce **latent variables** (the noisy intermediate states `x_t`). The marginal likelihood becomes:

```
p(x) = ∫ p(x, x_{1:T}) dx_{1:T}
```

This integral is intractable. We can't compute it directly.

### The ELBO Derivation (Simplified)

Starting from the log-likelihood, we use Jensen's inequality:

```
log p(x) = log ∫ p(x, z) dz                    # z = latent variables
         = log ∫ q(z|x) × [p(x,z)/q(z|x)] dz  # multiply by q/q
         = log E_q [p(x,z)/q(z|x)]             # expectation under q
         ≥ E_q [log p(x,z)/q(z|x)]             # Jensen's inequality!
         = ELBO
```

**Key point**: Jensen's inequality says `log E[X] ≥ E[log X]` for concave functions like log.

### The ELBO Formula

```
ELBO = E_q [log p(x,z)] - E_q [log q(z|x)]
     = E_q [log p(x|z)] - D_KL[q(z|x) || p(z)]
        ↑                  ↑
   Reconstruction       Regularization
      term                 term
```

| Term | What it does |
|------|--------------|
| Reconstruction | Ensures model can reconstruct x from latent z |
| KL term | Keeps the posterior q close to the prior p |

---

## 4. From ELBO to the Diffusion Loss

### The Diffusion Setup

In diffusion models:
- **Forward process** `q`: Gradually add noise to data `x → x_1 → x_2 → ... → x_T` (known, fixed)
- **Reverse process** `p_θ`: Learn to denoise `x_T → ... → x_1 → x` (what we train)

### The Diffusion ELBO (NELBO)

The negative ELBO for diffusion (from the paper, Eq. 3):

```
L(x; θ) = E_q [ -log p_θ(x | x_{t(1)})                    # Reconstruction
              + Σ D_KL[q(x_s | x_t, x) || p_θ(x_s | x_t)]  # Diffusion steps
              + D_KL[q(x_T | x) || p(x_T)]                 # Prior
            ]
```

### Breaking It Down

| Term | Name | Meaning |
|------|------|---------|
| `-log p_θ(x \| x_{t(1)})` | Reconstruction Loss | How well can we recover x from nearly-clean data? |
| `Σ D_KL[q \|\| p_θ]` | Diffusion Loss | Does our denoising match the true reverse? |
| `D_KL[q(x_T) \|\| p(x_T)]` | Prior Loss | Does the final noisy state match our prior (all [MASK])? |

### Visual

```
Forward (Known):
x_0 ──q──> x_{0.1} ──q──> x_{0.5} ──q──> x_{1.0}
(clean)                                   (all MASK)

Reverse (Learned):
x_0 <──p_θ── x_{0.1} <──p_θ── x_{0.5} <──p_θ── x_{1.0}

Loss = Σ D_KL[ q(previous | current, original) || p_θ(previous | current) ]
```

---

## 5. The Simplified BD3LM Loss

### Masked Diffusion Simplification

For **masked diffusion** (used in BD3LM), several simplifications occur:

1. **Prior Loss → 0**: At t=1, everything is masked. Both q and p give all-[MASK], so KL = 0.

2. **Reconstruction Loss → 0**: At t→0, nothing is masked. The input equals the output.

3. **Only Diffusion Loss remains**!

### The Final Simplified Loss (Paper Eq. 8)

```
L_BD(x; θ) = Σ_{b=1}^{B} E_{t~[0,1]} E_q [ (α'_t)/(1-α_t) × log p_θ(x^b | x_t^b, x^{<b}) ]
```

Where:
- `B` = number of blocks
- `t` = noise level (sampled uniformly from [0,1])
- `α_t` = probability a token is NOT masked at time t
- `α'_t` = derivative of α_t (rate of change)
- `x^b` = tokens in block b
- `x_t^b` = noisy (partially masked) tokens in block b
- `x^{<b}` = clean tokens from all previous blocks

### What is `(α'_t)/(1-α_t)`?

This is the **loss_scale** or weight. For the linear schedule where `α_t = 1-t`:

```
α'_t = -1
1 - α_t = t

loss_scale = α'_t / (1-α_t) = -1/t
```

| t | loss_scale = -1/t |
|---|-------------------|
| 0.1 | -10.0 |
| 0.5 | -2.0 |
| 1.0 | -1.0 |

**Intuition**: When t is small (few tokens masked), each prediction is more "valuable" → higher weight.

### The Loss in Plain English

```
For each block:
    1. Sample a random noise level t
    2. Mask some tokens based on t
    3. Ask the model to predict the original tokens
    4. Weight the cross-entropy loss by (-1/t)
    5. Sum across all blocks
```

---

## 6. Summary

### The Mathematical Journey

```
Goal: Maximize log p_θ(x)
        ↓
Can't compute directly (intractable)
        ↓
Use ELBO as a surrogate (lower bound)
        ↓
ELBO decomposes into KL divergence terms
        ↓
For masked diffusion, simplifies to weighted cross-entropy
        ↓
Final loss: loss_scale × cross_entropy
```

### Key Concepts Recap

| Concept | What it is | Why it matters |
|---------|------------|----------------|
| **Log-likelihood** | `log p_θ(x)` | The "true" objective we want to maximize |
| **KL Divergence** | Distance between distributions | Measures how well model matches target |
| **ELBO** | Lower bound on log-likelihood | Tractable objective we can optimize |
| **loss_scale** | `-1/t` (for linear schedule) | Weights loss based on noise level |

### Connection to code_loss.md

| Math concept | Code concept |
|--------------|--------------|
| `(α'_t)/(1-α_t)` | `loss_scale` in `noise_schedule.py` |
| `log p_θ(x^b \| x_t^b, x^{<b})` | `log_p_theta` gathered from model output |
| `E_{t~[0,1]}` | `_sample_t()` uniform sampling |
| Block conditioning `x^{<b}` | Block-causal attention mask |

---

## Further Reading

For more details:
- **code_loss.md**: Implementation-focused explanation of the loss
- **Paper Appendix B.3**: Full derivation of the simplified NELBO
- **Sahoo et al. (2024a)**: Original MDLM paper with complete proofs

---

## Appendix: Why Minimizing NELBO Works

The gap between negative log-likelihood and NELBO is exactly a KL divergence:

```
-log p(x) = ELBO - D_KL[q(z|x) || p_θ(z|x)]
```

Since `D_KL ≥ 0`:
```
-log p(x) ≤ ELBO  (NELBO is an upper bound on negative log-likelihood)
```

When we minimize NELBO, we're minimizing an upper bound on what we actually care about. If the bound is tight (KL term is small), we're effectively minimizing the true negative log-likelihood.

```
        ┌──────────────────────────┐
        │      NELBO (what we      │
        │       minimize)          │
        └──────────────────────────┘
                    ↓
        ┌──────────────────────────┐
        │   -log p(x) (true NLL)   │ ← Always ≤ NELBO
        └──────────────────────────┘
```
