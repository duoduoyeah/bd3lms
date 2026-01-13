# Loss Functions in BD3LMs

This document explains how the training loss is computed for both AR and BD3LM models.

## Overview

| Model | Loss Type | Weighting |
|-------|-----------|-----------|
| AR (GPT-2) | Cross-entropy | Uniform (all positions equal) |
| BD3LM | Weighted cross-entropy | Weight 1: `loss_scale` from noise schedule (always) |
| | | Weight 2: importance weight (only if mixing t distributions) |

---

## 1. AR Loss (Simple)

For autoregressive models, the loss is straightforward: sum of cross-entropy at each position.

### Formula

```
L_AR = -Σ_{i=1}^{T} log P(x_i | x_{<i})
```

Each position contributes equally to the total loss.

### Code Location

`diffusion.py:874-877`:

```python
if self.parameterization == 'ar':
    output = self.forward(input_tokens, None)  # [batch, seq_len, vocab_size]
    loss = - output.gather(-1, output_tokens[:, :, None])[:, :, 0]
    # loss[i] = -log P(x_{i+1} | x_{0:i})
```

### Visual

```
Position:     0      1      2      3
Input:       BOS    The    cat    sat
Target:             The    cat    sat    down
Loss:              L_0    L_1    L_2    L_3
Weight:             1      1      1      1
                   ──────────────────────────
Total Loss:        L_0 +  L_1 +  L_2 +  L_3
```

---

## 2. BD3LM Loss (Weighted by `loss_scale`)

BD3LM uses a weighted loss where each position's contribution is scaled by `loss_scale`, which depends on the noise level `t`.

### The Key Difference from AR

- **AR**: All positions weighted equally (weight = 1)
- **BD3LM**: Positions weighted by `loss_scale(t)` from noise schedule

### Code Location

`diffusion.py:819-862`:

```python
def _forward_pass_diffusion(self, x0, t=None, ...):
    # 1. Sample t uniformly from [eps_min, eps_max]
    t = self._sample_t(x0.shape, x0.device, sampling_eps_min, sampling_eps_max)

    # 2. Get loss_scale (THE weight) and masking probability from noise schedule
    loss_scale, p = self.noise(t)

    # 3. Create noisy input by masking tokens with probability p
    xt = self.q_xt(x0, p, ...)

    # 4. Model predicts log P(x0 | xt)
    logits = self.forward(x_input, sigma=sigma)

    # 5. Get log probability of true tokens
    log_p_theta = torch.gather(logits, dim=-1, index=x0[:, :, None]).squeeze(-1)

    # 6. Apply the weight: loss_scale
    loss = loss_scale * log_p_theta

    return loss
```

---

## What is `loss_scale`?

`loss_scale` is a weighting factor derived from the noise schedule. It depends on `t`.

### Code Location

`noise_schedule.py`:

**LogLinear schedule:**
```python
class LogLinearNoise(Noise):
    def compute_loss_scaling_and_move_chance(self, t):
        loss_scale = -1 / t
        return loss_scale, t
```

**Cosine schedule:**
```python
class CosineNoise(Noise):
    def compute_loss_scaling_and_move_chance(self, t):
        cos = - (1 - self.eps) * torch.cos(t * torch.pi / 2)
        sin = - (1 - self.eps) * torch.sin(t * torch.pi / 2)
        move_chance = cos + 1
        loss_scaling = sin / (move_chance + self.eps) * torch.pi / 2
        return loss_scaling, move_chance
```

### Values for LogLinear Schedule

| t | loss_scale = -1/t |
|---|-------------------|
| 0.1 | -10.0 |
| 0.3 | -3.33 |
| 0.5 | -2.0 |
| 0.8 | -1.25 |
| 1.0 | -1.0 |

**Pattern**: Smaller `t` → larger `|loss_scale|` → more weight on the loss.

### Why This Weighting?

The `loss_scale` comes from the ELBO derivation for discrete diffusion:

```
loss_scale = w(t) = -d/dt log(1 - p(t))
```

**Intuition:**
- When `t` is small: few tokens are masked, so predicting them is "harder" and more informative → weight more
- When `t` is large: many tokens are masked, prediction is "easier" → weight less

This ensures the ELBO is estimated correctly.

---

## Concrete Example

**Setup:** block_size=4, sampled t=0.3

```
Original x0:  [The]  [cat]  [sat]  [down]
                ↓ mask with probability p=0.3
Noisy xt:     [MASK] [cat]  [MASK] [down]
                ↓ model predicts all positions
Predictions:  [The?] [cat]  [sat?] [down]

Per-position loss (only masked positions contribute):
              L_0     0      L_2     0

loss_scale for t=0.3:
              w(0.3) = -1/0.3 = -3.33

Final loss = loss_scale * (L_0 + L_2)
           = -3.33 * (L_0 + L_2)
```

Note: The negative sign is because `log_p_theta` is negative (log of probability), so `loss_scale * log_p_theta` becomes positive.

---

## What About t Sampling?

You might ask: "How is `t` chosen?"

```python
t = self._sample_t(...)  # Samples t uniformly from [eps_min, eps_max]
```

Each training step gets a random `t`:
- Step 1: t=0.7
- Step 2: t=0.2
- Step 3: t=0.9
- ...

### Code Location

`diffusion.py:768-789`:

```python
def _sample_t(self, batch_dims, device, sampling_eps_min, sampling_eps_max, block_size=None):
    if block_size is None:
        block_size = self.block_size
    n = batch_dims[-1]
    num_blocks = n // block_size
    _eps_b = torch.rand((batch_dims[0], num_blocks), device=device)

    # Antithetic sampling for variance reduction
    if self.antithetic_sampling:
        offset_b = torch.arange(batch_dims[0] * num_blocks, device=device) / (batch_dims[0] * num_blocks)
        _eps_b = (_eps_b / (batch_dims[0] * num_blocks) + offset_b) % 1

    t = _eps_b
    # Scale to [sampling_eps_min, sampling_eps_max]
    t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
    return t
```

### When t Sampling Distribution Matters (Weight 2)

If you use a **fixed** `U[a, b]` distribution throughout training, the sampling distribution is just a constant factor and can be ignored.

**But** if you **mix** different distributions during training, you need an **importance weight** to correct for non-uniform coverage.

#### Example: Mixing Two Distributions

```
Training setup:
- 50% of batches: t ~ U[0.1, 0.7]
- 50% of batches: t ~ U[0.5, 1.0]
```

Problem: `t=0.6` gets sampled from **both** distributions, but `t=0.2` only from the first. Without correction, the `t=0.6` region is over-represented in training.

#### The Fix: Importance Weighting

```python
# If sampling from U[a, b], the density is q(t) = 1/(b-a)
# Importance weight = 1/q(t) = (b-a)

if using_distribution_1:  # U[0.1, 0.7]
    t = uniform(0.1, 0.7)
    importance_weight = 0.7 - 0.1  # = 0.6
elif using_distribution_2:  # U[0.5, 1.0]
    t = uniform(0.5, 1.0)
    importance_weight = 1.0 - 0.5  # = 0.5

# Apply both weights
loss = importance_weight * loss_scale * log_p_theta
#      ^^^^^^^^^^^^^^^^   ^^^^^^^^^^
#      "Weight 2"         "Weight 1"
#      (distribution)     (noise schedule)
```

#### Summary: When Do You Need Weight 2?

| Training Setup | Need Importance Weight? |
|----------------|------------------------|
| Fixed `U[a, b]` throughout training | No (constant, can ignore) |
| Mix multiple `U` distributions | **Yes** |
| Change `U` distribution mid-training | **Yes** |

In this codebase, they use a **fixed** `U[eps_min, eps_max]`, so the importance weight is not explicitly included. The current code effectively has:

```python
# Current code (fixed distribution, no importance weight needed)
loss = loss_scale * log_p_theta
```

If you were to mix distributions, you would modify to:

```python
# Hypothetical code (mixing distributions, importance weight needed)
importance_weight = sampling_eps_max - sampling_eps_min
loss = importance_weight * loss_scale * log_p_theta
```

---

## Visual Comparison

### AR Loss

```
Position:     0      1      2      3
Token:       The    cat    sat    down
Loss:        L_0    L_1    L_2    L_3
Weight:       1      1      1      1
             ────────────────────────
Total:       1·L_0 + 1·L_1 + 1·L_2 + 1·L_3
```

### BD3LM Loss (t=0.3 example)

```
Position:     0      1      2      3
x0:          The    cat    sat    down
xt:          MASK   cat    MASK   down  (masked with p=0.3)
Loss:        L_0     0     L_2     0    (only masked positions)
Weight:     -3.33    -    -3.33    -    (loss_scale = -1/0.3)
             ────────────────────────
Total:      -3.33·L_0 + -3.33·L_2
```

---

## Summary

| Aspect | AR | BD3LM |
|--------|----|----|
| Loss formula | `-Σ log P(x_i \| x_{<i})` | `loss_scale · Σ -log P(x_0 \| x_t)` |
| Weight 1 (always) | 1 (uniform) | `loss_scale = -1/t` (for LogLinear) |
| Weight 2 (conditional) | N/A | `(b-a)` if mixing `U[a,b]` distributions |
| Which positions | All positions | Only masked positions |
| What's predicted | Next token | Masked tokens |

**Key takeaways:**
- BD3LM always has **Weight 1** (`loss_scale`) from the noise schedule
- BD3LM needs **Weight 2** (importance weight) only if mixing different sampling distributions
- In this codebase, a fixed `U[eps_min, eps_max]` is used, so Weight 2 is not explicitly included

---

## Relationship to Perplexity

During **training**: we minimize the loss described above.

During **evaluation**: we compute perplexity from the NLL:
```
PPL = exp(average NLL per token)
```

For BD3LM, the "NLL" used for perplexity is the ELBO estimate. See [perplexity.md](perplexity.md) for details.
