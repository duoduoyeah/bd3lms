# Alternative Understanding of "Noise" in Language Models

This document presents a different perspective on what "noise" means in diffusion language models, compared to the standard BD3LM view.

---

## 1. Two Views of "Noise"

### Standard BD3LM View: Noise = Masking

In BD3LM, "noise" means replacing tokens with [MASK]:

```
Clean:    [The] [cat] [sat] [down]
Noisy:    [The] [MASK] [sat] [MASK]
```

The denoising process = predict what [MASK] should be.

### Our View: Noise = Information Granularity

We propose that "noise" is better understood as **how specific the information is** at each position:

| Token Type | Mapping | Information Level | Noise Level |
|------------|---------|-------------------|-------------|
| Clean token | 1 → 1 | Exact token known | 0% (no noise) |
| Group token | 1 → N | Token is one of N candidates | Partial noise |
| Mask token | 1 → V | Token could be anything | 100% (max noise) |

Where V = vocabulary size, N = group size.

**Key insight**: A [MASK] token is just a group token where the group = entire vocabulary!

```
Noise spectrum:

Clean token    Group token (1→N)    Mask token
    |________________|__________________|
   0%              partial             100%
(1→1)              noise             (1→V)
```

---

## 2. Reframing BD3LM as a Composite Model

### The Standard Narrative

> "BD3LM does diffusion within blocks, gradually denoising masked tokens."

### Our Narrative

> "BD3LM is a composite model that learns B different prediction tasks simultaneously."

For block size B=4, the model learns to predict:

| Task | What it predicts | Given context |
|------|------------------|---------------|
| k=1 | Token at position i+1 | Clean tokens [0:i] + mixture at [i+1:i+4] |
| k=2 | Token at position i+2 | Clean tokens [0:i] + mixture at [i+1:i+4] |
| k=3 | Token at position i+3 | Clean tokens [0:i] + mixture at [i+1:i+4] |
| k=4 | Token at position i+4 | Clean tokens [0:i] + mixture at [i+1:i+4] |

**Critical difference from AR**: The positions i+1 to i+k-1 are NOT empty — they contain either:
- Mask tokens (indicating "unknown")
- Clean tokens (if already decoded)

This is extra information that standard AR doesn't have!

---

## 3. The Core Question

**Why can a model predict position i+k when it doesn't know positions i+1 to i+k-1?**

### Hypothesis A: It can't do it well

Standard GPT-2 with `target_shift=k` (predicting i+k given [0:i]) should perform worse as k increases, because:
- Less information available
- Longer dependency distance

### Hypothesis B: Mask tokens help

If we fill positions i+1 to i+k-1 with [MASK] tokens instead of nothing, the model might perform better because:
- The model knows HOW MANY unknown tokens are in between
- The positional information is preserved
- [MASK] is learnable — it represents "could be anything"

### Hypothesis C: Group tokens help more

If we give partial information (group tokens) at positions i+1 to i+k-1, the model should perform even better because:
- Reduced uncertainty (N choices instead of V choices)
- Progressive refinement becomes possible

---

## 4. Proposed Experiments

### Experiment A: Baseline Target Shift

**Setup**: Standard GPT-2, varying `target_shift`

| target_shift | Predicts | Given |
|--------------|----------|-------|
| 1 | Token at i+1 | Tokens [0:i] |
| 2 | Token at i+2 | Tokens [0:i] |
| k | Token at i+k | Tokens [0:i] |

**Question**: How does perplexity change as k increases?

**Expected result**: Performance degrades as k increases.

---

### Experiment B: BD3LM Position-Specific Evaluation

**Setup**: BD3LM with block size B, but evaluate each position separately

For each k in [1, B]:
- Only compute loss at position i+k within each block
- Use special attention mask to isolate
- Fill other positions with [MASK]

**Question**: Does BD3LM at position k outperform GPT-2 with target_shift=k?

**Hypothesis**: No significant difference — the mask tokens don't add useful information beyond positional encoding.

---

### Experiment C: Group Tokens

**Setup**: Replace [MASK] with group tokens that represent subsets of vocabulary

Example with vocab_size=4096, num_groups=4:
- Group token 1 → represents tokens [0:1023]
- Group token 2 → represents tokens [1024:2047]
- Group token 3 → represents tokens [2048:3071]
- Group token 4 → represents tokens [3072:4095]

**Question**:
1. Do group tokens improve prediction of i+k?
2. What group size N makes predicting i+k as good as predicting i+1?

**Noise degree**:
- N=1: clean token (0% noise)
- N=V: mask token (100% noise)
- N in between: partial noise

---

### Experiment D: Parallel Denoising LM

**Idea**: Two-stage decoding within each block

```
Stage 1: [MASK] → [GROUP]     (reduce from V choices to N choices)
Stage 2: [GROUP] → [CLEAN]    (reduce from N choices to 1 choice)
```

**Success metric for Stage 1**:
- Transition is "successful" if the predicted group contains the true token

**Potential advantage**:
- If Stage 1 is fast/easy, we can decode multiple positions in parallel
- Higher tokens-per-step throughput than both AR and BD3LM

---

## 5. Visual Comparison

### Standard AR (GPT-2)

```
Position:  0    1    2    3    4    5
Input:    [A]  [B]  [C]   ?    ?    ?
                     ↓
Predict:            [D]  (only next token)
```

### AR with target_shift=3

```
Position:  0    1    2    3    4    5
Input:    [A]  [B]  [C]   ?    ?    ?
                               ↓
Predict:                      [F]  (skip 2 positions)
                              (harder! no info about D, E)
```

### BD3LM

```
Position:  0    1    2    3    4    5
Input:    [A]  [B]  [C]  [M]  [M]  [M]
                     ↓    ↓    ↓
Predict:            [D]  [E]  [F]  (all positions, M=mask)
                    (knows there ARE tokens at 3,4,5)
```

### Our Proposal: Group Token Model

```
Position:  0    1    2    3    4    5
Input:    [A]  [B]  [C]  [G1] [G2] [G3]
                     ↓    ↓    ↓
Predict:            [D]  [E]  [F]  (G=group token)
                    (knows D∈group1, E∈group2, F∈group3)
```

---

## 6. Key Insight: What is [MASK] Really?

### Standard view
> "[MASK] is a special token meaning 'unknown'"

### Our view
> "[MASK] is a group token where group_size = vocab_size"

This reframes the entire diffusion process:

| Diffusion step | Token state | Group size |
|----------------|-------------|------------|
| t=1.0 (start) | [MASK] | V (all vocab) |
| t=0.5 (middle) | [GROUP] | N (subset) |
| t=0.0 (end) | [CLEAN] | 1 (exact) |

**Denoising = progressively shrinking the group size from V → 1**

---

## 7. Why This Matters

### For understanding BD3LM:
- BD3LM's "noise schedule" is really a schedule of information granularity
- The model learns to work with varying levels of certainty

### For model design:
- We can explicitly control the information level with group tokens
- Two-stage decoding (MASK→GROUP→CLEAN) might be more efficient

### For parallelism:
- If GROUP→CLEAN is as easy as standard next-token prediction
- And MASK→GROUP is cheap
- Then parallel decoding becomes viable with good quality

---

## 8. Open Questions

1. **Is there an optimal group size N** that balances:
   - Ease of MASK→GROUP prediction
   - Ease of GROUP→CLEAN prediction
   - Parallelism benefit

2. **How should groups be constructed?**
   - Random partition?
   - Semantic clustering (similar tokens in same group)?
   - Frequency-based?

3. **Can we learn the grouping?**
   - Differentiable group assignment?
   - Hierarchical groups?

4. **Comparison to multi-token prediction heads**:
   - Papers that add extra heads to predict multiple future tokens
   - Why do they fail? Is it the same problem we're addressing?

---

## Summary

| Concept | BD3LM View | Our View |
|---------|------------|----------|
| Noise | Binary: masked or not | Continuous: group size 1 to V |
| [MASK] | Special "unknown" token | Group token with group_size=V |
| Denoising | Predict masked tokens | Shrink group size: V → N → 1 |
| Model | Single denoiser | Composite of B predictors |
| Parallelism | Limited by diffusion steps | Potentially better with 2-stage |
