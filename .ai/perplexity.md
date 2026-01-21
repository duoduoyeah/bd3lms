# Perplexity in BD3LMs

This document explains how perplexity is computed in this repository, covering both autoregressive (AR) and block diffusion approaches.

> **Important Distinction:** During **training**, we optimize the **loss** (see [loss.md](loss.md) for details). **Perplexity** is a metric computed during **evaluation** to measure model quality. Perplexity is derived from the loss/NLL but is not directly used as the training objective.

## What is NLL (Negative Log-Likelihood)?

NLL measures how "surprised" the model is by the true token. If the model assigns probability `p` to the correct token:

```
NLL = -log(p)
```

Examples:
- Model is confident (p=0.9): NLL = -log(0.9) ≈ 0.105 (low, good)
- Model is uncertain (p=0.1): NLL = -log(0.1) ≈ 2.3 (high, bad)

**Perplexity** is simply the exponentiated average NLL:

```
PPL = exp(average NLL per token)
```

Lower perplexity = better model. A perplexity of 10 means the model is, on average, as uncertain as if choosing uniformly among 10 tokens.

---

## Two Types of Perplexity in This Repo

### 1. AR Perplexity (GPT-2 Style)

#### How It Works

Predict the next token given all previous tokens (causal/left-to-right):

```
Input:  [BOS] The  cat  sat
Target:       The  cat  sat  down
              ↑    ↑    ↑    ↑
            predict each next token causally
```

#### Code Location

`diffusion.py:874-877`:

```python
if self.parameterization == 'ar':
    output = self.forward(input_tokens, None)  # Causal attention
    loss = - output.gather(-1, output_tokens[:, :, None])[:, :, 0]
    # loss[i] = -log P(token[i+1] | token[0:i])
```

#### The Math

```
NLL_AR = -Σ log P(x_t | x_{<t})

PPL_AR = exp(NLL_AR / T)  where T = sequence length
```

This is the standard language modeling objective.

---

### 2. BD3LM Perplexity (Diffusion Style)

#### How It Works

Randomly mask tokens at different noise levels, then predict the original tokens:

```
Original x0: [The] [cat] [sat] [down]
                    ↓ mask with probability p
Noisy xt:    [The] [MASK] [sat] [MASK]
                    ↓ model predicts
Prediction:  [The] [cat?] [sat] [down?]
```

#### Code Location

`diffusion.py:819-862`:

```python
def _forward_pass_diffusion(self, x0, t=None, ...):
    # 1. Sample noise level t ∈ [ε, 1] for each block
    t = self._sample_t(x0.shape, x0.device, sampling_eps_min, sampling_eps_max)

    # 2. Get masking probability p(t) from noise schedule
    loss_scale, p = self.noise(t)
    # p = probability a token gets masked
    # loss_scale = weighting factor from ELBO derivation

    # 3. Create noisy input (mask tokens with prob p)
    xt = self.q_xt(x0, p, ...)

    # 4. Model predicts log P(x0 | xt)
    logits = self.forward(x_input, sigma=sigma)

    # 5. Get log prob of true token at each position
    log_p_theta = torch.gather(logits, dim=-1, index=x0[:, :, None]).squeeze(-1)

    # 6. Weight by loss_scale (this comes from ELBO math)
    loss = loss_scale * log_p_theta
```

#### The Math (ELBO for Discrete Diffusion)

The true NLL is intractable, so we optimize an upper bound (Evidence Lower BOund):

```
NLL ≤ ELBO = E_t [ w(t) · E_{x_t|x_0} [ -log P_θ(x_0 | x_t) ] ]
```

Where:
- `t` is sampled uniformly (or with antithetic sampling)
- `w(t) = loss_scale` is the importance weight from the noise schedule
- `x_t` is the masked version of `x_0`
- Model predicts `P_θ(x_0 | x_t)` for all positions

The `loss_scale` is derived from the noise schedule:
```python
loss_scale = -d/dt log(1 - p(t))
```

This weighting ensures the Monte Carlo estimate equals the true ELBO in expectation.

---

## Key Differences: AR vs BD3LM

| Aspect | AR (GPT-2) | BD3LM (Diffusion) |
|--------|-----------|-------------------|
| What model sees | All previous tokens | Partially masked sequence |
| Attention pattern | Causal (left-to-right) | Block-diagonal + cross-attention |
| Training signal | Next token prediction | Masked token reconstruction |
| Loss weighting | Uniform across positions | Weighted by noise schedule `w(t)` |
| Metric | Exact NLL | ELBO (upper bound on NLL) |

### Visual Comparison

**AR Model:**
```
Position:    0     1     2     3
Input:      BOS → The → cat → sat
Predict:          The   cat   sat   down
Attention:   [causal: each position sees only left context]
```

**BD3LM:**
```
Position:    0     1     2     3
x0:         The   cat   sat   down
            ↓ sample t, mask with prob p(t)
xt:         The  MASK   sat  MASK
            ↓ predict all positions simultaneously
Predict:    The   cat   sat   down
Attention:  [block-diagonal within noisy blocks]
            [cross-attend to clean context from previous blocks]
```

---

## Generative Perplexity (Sample Quality Evaluation)

Besides training/validation perplexity, this repo also computes **generative perplexity** to evaluate the quality of generated samples.

### How It Works

1. Generate text samples from the model
2. Use an external pretrained LM (e.g., GPT-2) to score these samples
3. Compute perplexity under the external model

### Code Location

`metrics.py:149-219`:

```python
def record_generative_perplexity(self, text_samples, ...):
    # 1. Load external evaluator (e.g., GPT-2)
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
        self.gen_ppl_eval_model_name_or_path)

    # 2. Re-tokenize generated samples for the eval model
    samples = self.tokenizer(text_samples, ...)

    # 3. Compute cross-entropy loss using sliding window
    for each batch:
        logits = eval_model(sample_chunk, attention_mask=...)

        # NLL = cross-entropy between predicted and actual next token
        nlls = F.cross_entropy(logits[..., :-1], sample_chunk[..., 1:], reduction='none')

        # Only count non-EOS tokens
        valid_tokens = (sample_chunk[..., 1:] != eos_token_id)

    # 4. Compute perplexity
    avg_nll = (nlls * valid_tokens).sum() / valid_tokens.sum()
    gen_ppl = avg_nll.exp()
```

### Why Use External Model?

- **Training perplexity**: Measures how well the model fits training data
- **Generative perplexity**: Measures how "natural" generated samples are

A model could overfit (low training PPL) but generate poor samples (high generative PPL).

---

## Perplexity Metric Implementation

`metrics.py:26-33`:

```python
class Perplexity(NLL):
    def compute(self) -> Tensor:
        # mean_value = sum of all NLLs
        # weight = count of valid tokens
        return torch.exp(self.mean_value / self.weight)
```

### Arithmetic Mean vs Geometric Mean

The average used here is the **arithmetic mean** (sum and divide), not geometric mean:

```
PPL = exp( (NLL_1 + NLL_2 + ... + NLL_N) / N )
    = exp( arithmetic_mean(NLLs) )
```

**However, there's an elegant mathematical equivalence:**

Since `NLL_i = -log(p_i)`, we can rewrite:

```
PPL = exp( (1/N) * Σ(-log(p_i)) )
    = exp( (1/N) * (-log(p_1) - log(p_2) - ... - log(p_N)) )
    = exp( -log(p_1 * p_2 * ... * p_N) / N )
    = exp( log( (p_1 * p_2 * ... * p_N)^(-1/N) ) )
    = (p_1 * p_2 * ... * p_N)^(-1/N)
    = 1 / (p_1 * p_2 * ... * p_N)^(1/N)
    = 1 / geometric_mean(probabilities)
```

**Two equivalent interpretations:**

| View | Formula |
|------|---------|
| NLL perspective | `exp(arithmetic_mean(NLLs))` |
| Probability perspective | `1 / geometric_mean(probabilities)` |

**Why use arithmetic mean of NLLs in code?**

Numerical stability. Multiplying many small probabilities (e.g., 0.1 * 0.2 * 0.05 * ...) quickly underflows to zero. Summing log-probabilities avoids this issue.

### Concrete Example

| Position | True Token | Model Prob (p) | NLL = -log(p) |
|----------|-----------|----------------|---------------|
| 0 | "The" | 0.301 | 1.2 |
| 1 | "cat" | 0.449 | 0.8 |
| 2 | "sat" | 0.223 | 1.5 |
| 3 | "down" | 0.607 | 0.5 |

**Method 1: Arithmetic mean of NLLs (what the code does)**

```python
# Total NLL
total_nll = 1.2 + 0.8 + 1.5 + 0.5 = 4.0

# Average NLL
avg_nll = 4.0 / 4 = 1.0

# Perplexity
ppl = exp(1.0) ≈ 2.718
```

**Method 2: Inverse geometric mean of probabilities (equivalent)**

```python
# Product of probabilities
prob_product = 0.301 * 0.449 * 0.223 * 0.607 ≈ 0.0183

# Geometric mean
geo_mean = prob_product^(1/4) ≈ 0.368

# Perplexity
ppl = 1 / geo_mean ≈ 2.718
```

Both methods yield the same result!

**Interpretation:** On average, the model is as uncertain as if it were choosing uniformly among ~2.7 tokens at each position.

---

## Summary

### Training vs Evaluation

| Stage | What We Use | Purpose |
|-------|-------------|---------|
| **Training** | Loss function | Optimize model parameters |
| **Evaluation** | Perplexity (derived from NLL) | Measure model quality |

Perplexity is **not** the training objective. We train by minimizing loss, then compute perplexity to evaluate how well the model performs. See [loss.md](loss.md) for detailed explanation of the loss functions.

### Perplexity Metrics in This Repo

| Metric | What It Measures | Where Computed |
|--------|-----------------|----------------|
| Validation PPL | Model generalization (from ELBO) | `diffusion.py:414-453` |
| Generative PPL | Quality of generated samples | `metrics.py:149-219` |

For BD3LM specifically, the reported "perplexity" is actually derived from **ELBO** (Evidence Lower Bound), not exact NLL. This is standard practice in diffusion models since exact likelihood computation is intractable.
