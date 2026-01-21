# Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models 

Marianne Arriola $^{\dagger} \boldsymbol{*} \quad$ Aaron Kerem Gokaslan ${ }^{\dagger} \quad$ Justin T. Chiu $^{\ddagger} \quad$ Zhihan Yang $^{\dagger}$ Zhixuan $\mathbf{Q i}^{\boldsymbol{\dagger}}$ Jiaqi Han ${ }^{\boldsymbol{q}} \quad$ Subham Sekhar Sahoo ${ }^{\boldsymbol{\dagger}} \quad$ Volodymyr Kuleshov $^{\boldsymbol{\dagger}}$


#### Abstract

Diffusion language models offer unique benefits over autoregressive models due to their potential for parallelized generation and controllability, yet they lag in likelihood modeling and are limited to fixed-length generation. In this work, we introduce a class of block diffusion language models that interpolate between discrete denoising diffusion and autoregressive models. Block diffusion overcomes key limitations of both approaches by supporting flexible-length generation and improving inference efficiency with KV caching and parallel token sampling. We propose a recipe for building effective block diffusion models that includes an efficient training algorithm, estimators of gradient variance, and data-driven noise schedules to minimize the variance. Block diffusion sets a new state-of-the-art performance among diffusion models on language modeling benchmarks and enables generation of arbitrary-length sequences. We provide the code ${ }^{1}$, along with the model weights and blog post on the project page:


https://m-arriola.com/bd31ms

## 1 Introduction

Diffusion models are widely used to generate images (Ho et al., 2020; Dhariwal \& Nichol, 2021; Sahoo et al., 2024b) and videos (Ho et al., 2022; Gupta et al., 2023), and are becoming increasingly effective at generating discrete data such as text (Lou et al., 2024; Sahoo et al., 2024a) or biological sequences (Avdeyev et al., 2023; Goel et al., 2024). Compared to autoregressive models, diffusion models have the potential to accelerate generation and improve the controllability of model outputs (Schiff et al., 2024; Nisonoff et al., 2024; Li et al., 2024; Sahoo et al., 2024c).

Discrete diffusion models currently face at least three limitations. First, in applications such as chat systems, models must generate output sequences of arbitrary length (e.g., a response to a user's question). However, most recent diffusion architectures only generate fixed-length vectors (Austin et al., 2021; Lou et al., 2024). Second, discrete diffusion uses bidirectional context during generation and therefore cannot reuse previous computations with KV caching, which makes inference less efficient (Israel et al., 2025). Third, the quality of discrete diffusion models, as measured by standard metrics such as perplexity, lags behind autoregressive approaches and further limits their applicability (Gulrajani \& Hashimoto, 2024; Sahoo et al., 2024a).
This paper makes progress towards addressing these limitations by introducing Block Discrete Denoising Diffusion Language Models (BD3-LMs), which interpolate between discrete diffusion and autoregressive models. Specifically, block diffusion models (also known as semi-autoregressive models) define an autoregressive probability distribution over blocks of discrete random variables (Si et al., 2022; 2023); the conditional probability of a block given previous blocks is specified by a discrete denoising diffusion model (Austin et al., 2021; Sahoo et al., 2024a).

Developing effective BD3-LMs involves two challenges. First, efficiently computing the training objective for a block diffusion model is not possible using one standard forward pass of a neural

[^0]![](https://cdn.mathpix.com/cropped/2025_10_29_b6105f7bcf5e026ed288g-02.jpg?height=551&width=1440&top_left_y=290&top_left_x=352)
Figure 1: Block diffusion sequentially generates blocks of tokens by performing diffusion within each block and conditioning on previous blocks. By combining strength from autoregressive and diffusion models, block diffusion overcomes the limitations of both approaches by supporting variable-length, higher-quality generation and improving inference efficiency with KV caching and parallel sampling.

network and requires developing specialized algorithms. Second, training is hampered by the high variance of the gradients of the diffusion objective, causing BD3-LMs to under-perform autoregression even with a block size of one (when both models should be equivalent). We derive estimators of gradient variance, and demonstrate that it is a key contributor to the gap in perplexity between autoregression and diffusion. We then propose custom noise processes that minimize gradient variance and make progress towards closing the perplexity gap.

We evaluate BD3-LMs on language modeling benchmarks, and demonstrate that they are able to generate sequences of arbitrary length, including lengths that exceed their training context. In addition, BD3-LMs achieve new state-of-the-art perplexities among discrete diffusion models. Compared to alternative semi-autoregressive formulations that perform Gaussian diffusion over embeddings (Han et al., 2022; 2023), our discrete approach features tractable likelihood estimates and yields samples with improved generative perplexity using an order of magnitude fewer generation steps. In summary, our work makes the following contributions:

- We introduce block discrete diffusion language models, which are autoregressive over blocks of tokens; conditionals over each block are based on discrete diffusion. Unlike prior diffusion models, block diffusion supports variable-length generation and KV caching.
- We introduce custom training algorithms for block diffusion models that enable efficiently leveraging the entire batch of tokens provided to the model.
- We identify gradient variance as a limiting factor of the performance of diffusion models, and we propose custom data-driven noise schedules that reduce gradient variance.
- Our results establish a new state-of-the-art perplexity for discrete diffusion and make progress toward closing the gap to autoregressive models.


## 2 Background: Language Modeling Paradigms

Notation We consider scalar discrete random variables with $V$ categories as 'one-hot' column vectors in the space $\mathcal{V}=\left\{\mathbf{x} \in\{0,1\}^{V}: \sum_{i} \mathbf{x}_{i}=1\right\} \subset \Delta^{V}$ for the simplex $\Delta^{V}$. Let the $V$-th category denote a special [MASK] token, where $\mathbf{m} \in \mathcal{V}$ is its one-hot vector. We define $\mathbf{x}^{1: L}$ as a sequence of $L$ tokens, where $\mathbf{x}^{\ell} \in \mathcal{V}$ for all tokens $\ell \in\{1, \ldots, L\}$, and use $\mathcal{V}^{L}$ to denote the set of all such sequences. Throughout the work, we simplify notation and refer to the token sequence as $\mathbf{x}$ and an individual token as $\mathbf{x}^{\ell}$. Finally, let $\operatorname{Cat}(\cdot ; p)$ be a categorical distribution with probability $p \in \Delta^{V}$.

### 2.1 Autoregressive Models

Consider a sequence of $L$ tokens $\mathbf{x}=\left[\mathbf{x}^{1}, \ldots, \mathbf{x}^{L}\right]$ drawn from the data distribution $q(\mathbf{x})$. Autoregressive (AR) models define a factorized distribution of the form

$$
\begin{equation*}
\log p_{\theta}(\mathbf{x})=\sum_{\ell=1}^{L} \log p_{\theta}\left(\mathbf{x}^{\ell} \mid \mathbf{x}^{<\ell}\right) \tag{1}
\end{equation*}
$$

where each $p_{\theta}\left(\mathbf{x}^{\ell} \mid \mathbf{x}^{<\ell}\right)$ is parameterized directly with a neural network. As a result, AR models may be trained efficiently via next token prediction. However, AR models take $L$ steps to generate $L$ tokens due to the sequential dependencies.

### 2.2 Discrete Denoising Diffusion Probabilistic Models

Diffusion models fit a model $p_{\theta}(\mathbf{x})$ to reverse a forward corruption process $q$ (Sohl-Dickstein et al., 2015; Ho et al., 2020; Sahoo et al., 2024b). This process starts with clean data $\mathbf{x}$ and defines latent variables $\mathbf{x}_{t}=\left[\mathbf{x}_{t}^{1}, \ldots, \mathbf{x}_{t}^{L}\right]$ for $t \in[0,1]$, which represent progressively noisier versions of $\mathbf{x}$. Given a discretization into $T$ steps, we define $s(j)=(j-1) / T$ and $t(j)=j / T$. For brevity, we drop $j$ from $t(j)$ and $s(j)$ below; in general, $s$ denotes the time step preceding $t$.

The D3PM framework (Austin et al., 2021) defines $q$ as a Markov forward process acting independently on each token $\mathbf{x}^{\ell}: q\left(\mathbf{x}_{t}^{\ell} \mid \mathbf{x}_{s}^{\ell}\right)=\operatorname{Cat}\left(\mathbf{x}_{t}^{\ell} ; Q_{t} \mathbf{x}_{s}^{\ell}\right)$ where $Q_{t} \in \mathbb{R}^{V \times V}$ is the diffusion matrix. The matrix $Q_{t}$ can model various transformations, including masking, random token changes, and related word substitutions.

An ideal diffusion model $p_{\theta}$ is the reverse of the process $q$. The D3PM framework defines $p_{\theta}$ as

$$
\begin{equation*}
p_{\theta}\left(\mathbf{x}_{s} \mid \mathbf{x}_{t}\right)=\prod_{\ell=1}^{L} p_{\theta}\left(\mathbf{x}_{s}^{\ell} \mid \mathbf{x}_{t}\right)=\sum_{\mathbf{x}}\left[\prod_{\ell=1}^{L} q\left(\mathbf{x}_{s}^{\ell} \mid \mathbf{x}_{t}^{\ell}, \mathbf{x}^{\ell}\right) p_{\theta}\left(\mathbf{x}^{\ell} \mid \mathbf{x}_{t}\right)\right] \tag{2}
\end{equation*}
$$

where the denoising base model $p_{\theta}\left(\mathbf{x}^{\ell} \mid \mathbf{x}_{t}\right)$ predicts clean token $\mathbf{x}^{\ell}$ given the noisy sequence $\mathbf{x}_{t}$, and the reverse posterior $q\left(\mathbf{x}_{s}^{\ell} \mid \mathbf{x}_{t}^{\ell}, \mathbf{x}\right)$ is defined following Austin et al. (2021) in Suppl. B.3.
The diffusion model $p_{\theta}$ is trained using variational inference. Let KL[ $\cdot$ ] denote the Kullback-Leibler divergence. Then, the Negative ELBO (NELBO) is given by (Sohl-Dickstein et al., 2015):

$$
\begin{equation*}
\mathcal{L}(\mathbf{x} ; \theta)=\mathbb{E}_{q}\left[-\log p_{\theta}\left(\mathbf{x} \mid \mathbf{x}_{t(1)}\right)+\sum_{j=1}^{T} D_{\mathrm{KL}}\left[q\left(\mathbf{x}_{s(j)} \mid \mathbf{x}_{t(j)}, \mathbf{x}\right) \| p_{\theta}\left(\mathbf{x}_{s(j)} \mid \mathbf{x}_{t(j)}\right)\right]+D_{\mathrm{KL}}\left[q\left(\mathbf{x}_{t(T)} \mid \mathbf{x}\right) \| p_{\theta}\left(\mathbf{x}_{t(T)}\right)\right]\right] \tag{3}
\end{equation*}
$$

This formalism extends to continuous time via Markov chain (CTMC) theory and admits score-based generalizations (Song \& Ermon, 2019; Lou et al., 2024; Sun et al., 2022). Further simplifications (Sahoo et al., 2024a; Shi et al., 2024; Ou et al., 2025) tighten the ELBO and enhance performance.

## 3 Block Diffusion Language Modeling

We explore a class of Block Discrete Denoising Diffusion Language Models (BD3-LMs) that interpolate between autoregressive and diffusion models by defining an autoregressive distribution over blocks of tokens and performing diffusion within each block. We provide a block diffusion objective for maximum likelihood estimation and efficient training and sampling algorithms. We show that for a block size of one, the diffusion objective suffers from high variance despite being equivalent to the autoregressive likelihood in expectation. We identify high training variance as a limitation of diffusion models and propose data-driven noise schedules that reduce the variance of the gradient updates during training.

### 3.1 Block Diffusion Distributions and Model Architectures

We propose to combine the language modeling paradigms in Sec. 2 by autoregressively modeling blocks of tokens and performing diffusion within each block. We group tokens in $\mathbf{x}$ into $B$ blocks of
length $L^{\prime}$ with $B=L / L^{\prime}$ (we assume that $B$ is an integer). We denote each block $\mathbf{x}^{(b-1) L^{\prime}: b L^{\prime}}$ from token at positions $(b-1) L^{\prime}$ to $b L^{\prime}$ for blocks $b \in\{1, \ldots, B\}$ as $\mathbf{x}^{b}$ for simplicity. Our likelihood factorizes over blocks as

$$
\begin{equation*}
\log p_{\theta}(\mathbf{x})=\sum_{b=1}^{B} \log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}^{<b}\right) \tag{4}
\end{equation*}
$$

and each $p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}^{<b}\right)$ is modeled using discrete diffusion over a block of $L^{\prime}$ tokens. Specifically, we define a reverse diffusion process as in (2), but restricted to block $b$ :

$$
\begin{equation*}
p_{\theta}\left(\mathbf{x}_{s}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right)=\sum_{\mathbf{x}^{b}} q\left(\mathbf{x}_{s}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{b}\right) p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right) \tag{5}
\end{equation*}
$$

We obtain a principled learning objective by applying the NELBO in (3) to each term in (4) to obtain

$$
\begin{equation*}
-\log p_{\theta}(\mathbf{x}) \leq \mathcal{L}_{\mathrm{BD}}(\mathbf{x} ; \theta):=\sum_{b=1}^{B} \mathcal{L}\left(\mathbf{x}^{b}, \mathbf{x}^{<b} ; \theta\right), \tag{6}
\end{equation*}
$$

where each $\mathcal{L}\left(\mathbf{x}^{b}, \mathbf{x}^{<b} ; \theta\right)$ is an instance of (3) applied to $\log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}^{<b}\right)$. Since the model is conditioned on $\mathbf{x}^{<b}$, we make the dependence on $\mathbf{x}^{<b}, \theta$ explicit in $\mathcal{L}$. We denote the sum of these terms $\mathcal{L}_{\mathrm{BD}}(\mathbf{x} ; \theta)$ (itself a valid NELBO).

Model Architecture Crucially, we parameterize the $B$ base denoiser models $p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right)$ using a single neural network $\mathbf{x}_{\theta}$. The neural network $\mathbf{x}_{\theta}$ outputs not only the probabilities $p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right)$, but also computational artifacts for efficient training. This will enable us to compute the loss $\mathcal{L}_{\mathrm{BD}}(\mathbf{x} ; \theta)$ in parallel for all $B$ blocks in a memory-efficient manner. Specifically, we parameterize $\mathbf{x}_{\theta}$ using a transformer (Vaswani et al., 2017) with a block-causal attention mask. The transformer $\mathbf{x}_{\theta}$ is applied to $L$ tokens, and tokens in block $b$ attend to tokens in blocks 1 to $b$. When $\mathbf{x}_{\theta}$ is trained, $\mathbf{x}_{\theta}^{b}\left(\mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right)$ yields $L^{\prime}$ predictions for denoised tokens in block $b$ based on noised $\mathbf{x}_{t}^{b}$ and clean $\mathbf{x}^{<b}$.
In autoregressive generation, it is normal to cache keys and values for previously generated tokens to avoid recomputing them at each step. Similarly, we use $\mathbf{K}^{b}, \mathbf{V}^{b}$ to denote the keys and values at block $b$, and we define $\mathbf{x}_{\theta}$ to support these as input and output. The full signature of $\mathbf{x}_{\theta}$ is

$$
\begin{equation*}
\mathbf{x}_{\text {logits }}^{b}, \mathbf{K}^{b}, \mathbf{V}^{b} \leftarrow \mathbf{x}_{\theta}^{b}\left(\mathbf{x}_{t}^{b}, \mathbf{K}^{1: b-1}, \mathbf{V}^{1: b-1}\right):=\mathbf{x}_{\theta}^{b}\left(\mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right) \tag{7}
\end{equation*}
$$

where $\mathbf{x}_{\text {logits }}^{b}$ are the predictions for the clean $\mathbf{x}^{b}$, and $\mathbf{K}^{b}, \mathbf{V}^{b}$ is the key-value cache in the forward pass of $\mathbf{x}_{\theta}$, and $\mathbf{K}^{1: b-1}, \mathbf{V}^{1: b-1}$ are keys and values cached on a forward pass of $\mathbf{x}_{\theta}$ over $\mathbf{x}^{<b}$ (hence the inputs $\mathbf{x}^{<b}$ and $\mathbf{K}^{1: b-1}, \mathbf{V}^{1: b-1}$ are equivalent).

### 3.2 Efficient Training and Sampling Algorithms

Ideally, we wish to compute the loss $\mathcal{L}_{\mathrm{BD}}(\mathbf{x} ; \theta)$ in one forward pass of $\mathbf{x}_{\theta}$. However, observe that denoising $\mathbf{x}_{t}^{b}$ requires a forward pass on this noisy input, while denoising the next blocks requires running $\mathbf{x}_{\theta}$ on the clean version $\mathbf{x}^{b}$. Thus every block has to go through the model at least twice.

Training Based on this observation, we propose a training algorithm with these minimal computational requirements (Alg. 1). Specifically, we precompute keys and values $\mathbf{K}^{1: B}, \mathbf{V}^{1: B}$ for the full sequence $\mathbf{x}$ in a first forward pass $\left(\emptyset, \mathbf{K}^{1: B}, \mathbf{V}^{1: B}\right) \leftarrow \mathbf{x}_{\theta}(\mathbf{x})$. We then compute denoised predictions for all blocks using $\mathbf{x}_{\theta}^{b}\left(\mathbf{x}_{t}^{b}, \mathbf{K}^{1: b-1}, \mathbf{V}^{1: b-1}\right)$. Each token passes through $\mathbf{x}_{\theta}$ twice.

Vectorized Training Naively, we would compute the logits by applying $\mathbf{x}_{\theta}^{b}\left(\mathbf{x}_{t}^{b}, \mathbf{K}^{1: b-1}, \mathbf{V}^{1: b-1}\right)$ in a loop $B$ times. We propose a vectorized implementation that computes $\mathcal{L}_{\mathrm{BD}}(\mathbf{x} ; \theta)$ in one forward pass on the concatenation $\mathbf{x}_{\text {noisy }} \oplus \mathbf{x}$ of clean data $\mathbf{x}$ with noisy data $\mathbf{x}_{\text {noisy }}=\mathbf{x}_{t_{1}}^{1} \oplus \cdots \oplus \mathbf{x}_{t_{B}}^{B}$ obtained by applying a noise level $t_{b}$ to each block $\mathbf{x}^{b}$. We design an attention mask for $\mathbf{x}_{\text {nois }} \oplus \mathbf{x}$ such that noisy tokens attend to other noisy tokens in their block and to all clean tokens in preceding blocks (see Suppl. B.6). Our method keeps the overhead of training BD3-LMs tractable and combines with pretraining to further reduce costs.

Sampling We sample one block at a time, conditioned on previously sampled blocks (Alg 2). We may use any sampling procedure $\operatorname{Sample}\left(\mathbf{x}_{\theta}^{b}, \mathbf{K}^{1: b-1}, \mathbf{V}^{1: b-1}\right)$ to sample from the conditional distribution $p_{\theta}\left(\mathbf{x}_{s}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right)$, where the context conditioning is generated using cross-attention with pre-computed keys and values $\mathbf{K}^{1: b-1}, \mathbf{V}^{1: b-1}$. Similar to AR models, caching the keys and values saves computation instead of recalculating them when sampling a new block.
Notably, our block diffusion decoding algorithm enables us to sample sequences of arbitrary length, whereas diffusion models are restricted to fixed-length generation. Further, our sampler admits parallel generation within each block, whereas AR samplers are constrained to generate token-by-token.

```
Algorithm 1 Block Diffusion Training
    Input: datapoint $\mathbf{x}$, \# of blocks $B$, forward
    noise process $q_{t}(\cdot \mid \mathbf{x})$, model $\mathbf{x}_{\theta}$, loss $\mathcal{L}_{\mathrm{BD}}$
    repeat
        Sample $t_{1}, \ldots, t_{B} \sim \mathcal{U}[0,1]$
        $\forall b \in\{1, \ldots, B\}: \mathbf{x}_{t_{b}}^{b} \sim q_{t_{b}}\left(\cdot \mid \mathbf{x}^{b}\right)$
        $\emptyset, \mathbf{K}^{1: B}, \mathbf{V}^{1: B} \leftarrow \mathbf{x}_{\theta}(\mathbf{x}) \quad \triangleright$ KV cache
        $\forall b: \mathbf{x}_{\text {logit }}^{b}, \emptyset, \emptyset \leftarrow \mathbf{x}_{\theta}^{b}\left(\mathbf{x}_{t_{b}}^{b}, \mathbf{K}^{1: b-1}, \mathbf{V}^{1: b-1}\right)$
        Let $\mathbf{x}_{\text {logit }} \leftarrow \mathbf{x}_{\text {logit }}^{1} \oplus \cdots \oplus \mathbf{x}_{\text {logit }}^{B}$
        Take gradient step on $\nabla_{\theta} \mathcal{L}_{\mathrm{BD}}\left(\mathbf{x}_{\text {logit }} ; \theta\right)$
    until converged
```

```
Algorithm 2 Block Diffusion Sampling
    Input: \# blocks $B$, model $\mathbf{x}_{\theta}$, diffusion sam-
    pling algorithm Sample
    $\mathbf{x}, \mathbf{K}, \mathbf{V} \leftarrow \emptyset \quad \triangleright$ output \& KV cache
    for $b=1$ to $B$ do
        $\mathbf{x}^{b} \leftarrow \operatorname{Sample}\left(\mathbf{x}_{\theta}^{b}, \mathbf{K}^{1: b-1}, \mathbf{V}^{1: b-1}\right)$
        $\emptyset, \mathbf{K}^{b}, \mathbf{V}^{b} \leftarrow \mathbf{x}_{\theta}^{b}\left(\mathbf{x}^{b}\right)$
        $\mathbf{x} \leftarrow \mathbf{x}^{1: b-1} \oplus \mathbf{x}^{b}$
        $(\mathbf{K}, \mathbf{V}) \leftarrow\left(\mathbf{K}^{1: b-1} \oplus \mathbf{K}^{b}, \mathbf{V}^{1: b-1} \oplus \mathbf{V}^{b}\right)$
    end for
    return x
```


## 4 Understanding Likelihood Gaps Between Diffusion \& AR Models

### 4.1 Masked BD3-LMs

The most effective diffusion language models leverage a masking noise process (Austin et al., 2021; Lou et al., 2024; Sahoo et al., 2024a), where tokens are gradually replaced with a special mask token. Here, we introduce masked BD3-LMs, a special class of block diffusion models based on the masked diffusion language modeling framework (Sahoo et al., 2024a; Shi et al., 2024; Ou et al., 2025).
More formally, we adopt a per-token noise process $q\left(\mathbf{x}_{t}^{\ell} \mid \mathbf{x}^{\ell}\right)=\operatorname{Cat}\left(\mathbf{x}_{t}^{\ell} ; \alpha_{t} \mathbf{x}^{\ell}+\left(1-\alpha_{t}\right) \mathbf{m}\right)$ for tokens $\ell \in\{1, \ldots, L\}$ where $\mathbf{m}$ is a one-hot encoding of the mask token, and $\alpha_{t} \in[0,1]$ is a strictly decreasing function in $t$, with $\alpha_{0}=1$ and $\alpha_{1}=0$. We employ the linear schedule where the probability of masking a token at time $t$ is $1-\alpha_{t}$. We adopt the simplified objective from Sahoo et al. (2024a); Shi et al. (2024); Ou et al. (2025) (the full derivation is provided in Suppl. B.3):

$$
\begin{equation*}
-\log p_{\theta}(\mathbf{x}) \leq \mathcal{L}_{\mathrm{BD}}(\mathbf{x} ; \theta):=\sum_{b=1}^{B} \mathbb{E}_{t \sim[0,1]} \mathbb{E}_{q} \frac{\alpha_{t}^{\prime}}{1-\alpha_{t}} \log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right) \tag{8}
\end{equation*}
$$

where $\alpha_{t}^{\prime}$ is the instantaneous rate of change of $\alpha_{t}$ under the continuous-time extension of (3) that takes $T \rightarrow \infty$. The NELBO is tight for $L^{\prime}=1$ but becomes a looser approximation of the true negative log-likelihood for $L^{\prime} \rightarrow L$ (see Suppl. B.5).

### 4.2 Case Study: Single Token Generation

Our block diffusion parameterization (8) is equivalent in expectation to the autoregressive NLL (1) in the limiting case where $L^{\prime}=1$ (see Suppl. B.4). Surprisingly, we find a two point perplexity gap between our block diffusion model for $L^{\prime}=1$ and AR when training both models on the LM1B dataset.

Although the objectives are equivalent in expectation, we show that the remaining perplexity gap is a result of high training variance. Whereas AR is trained using the cross-entropy of $L$ tokens, our block diffusion model for $L^{\prime}=1$ only computes the cross-entropy for masked tokens $\mathbf{x}_{t}^{\ell}=\mathbf{m} \forall \ell \in\{1, \ldots L\}$

Table 1: Test perplexities for singletoken generation (PPL; $\downarrow$ ) across 16B tokens on LM1B.
|  | PPL $(\downarrow)$ |
| :--- | :--- |
| AR | $\mathbf{2 2 . 8 8}$ |
| $\quad+$ random batch size | 24.37 |
| BD3-LM $L^{\prime}=1$ | $\leq 25.56$ |
| $\quad+$ tuned schedule | $\mathbf{2 2 . 8 8}$ |


![](https://cdn.mathpix.com/cropped/2025_10_29_b6105f7bcf5e026ed288g-06.jpg?height=632&width=1003&top_left_y=287&top_left_x=542)
Figure 2: Train NLLs for modeling the per-token likelihood on LM1B. Models are trained on 16B tokens. Training under the discrete diffusion NELBO, where half of the tokens in a batch are masked on average, has similar training variance to an AR model with a random batch size.

so that $\mathbb{E}_{t \sim \mathcal{U}[0,1]} q\left(\mathbf{x}_{t}^{\ell}=\mathbf{m} \mid \mathbf{x}^{\ell}\right)=0.5$. Thus, training on the diffusion objective involves estimating loss gradients with 2 x fewer tokens and is responsible for higher training variance compared to AR.
To close the likelihood gap, we train a BD3-LM for $L^{\prime}=1$ by designing the forward process to fully mask tokens, i.e. $q\left(\mathbf{x}_{t}^{\ell}=\mathbf{m} \mid \mathbf{x}^{\ell}\right)=1$. Under this schedule, the diffusion objective becomes equivalent to the AR objective (Suppl. B.4). In Table 1, we show that training under the block diffusion objective yields the same perplexity as AR training. Empirically, we see that this reduces the variance of the training loss in Figure 2. We verify that tuning the noise schedule reduces the variance of the objective by measuring $\operatorname{Var}_{\mathbf{x}, t}\left[\mathcal{L}_{\mathrm{BD}}(\mathbf{x} ; \theta)\right]$ after training on 328 M tokens: while training on the NELBO results in a variance of 1.52 , training under full masking reduces the variance to 0.11 .

### 4.3 Diffusion Gap from High Variance Training

Next, we formally describe the issue of gradient variance in training diffusion models. Given our empirical observations for single-token generation, we propose an estimator for gradient variance that we use to minimize the variance of diffusion model training for $L^{\prime} \geq 1$. While the NELBO is invariant to the choice of noise schedule (Suppl. B.3), this invariance does not hold for our Monte Carlo estimator of the loss used during training. As a result, the variance of the estimator and its gradients are dependent on the schedule. First, we express the estimator of the NELBO with a batch size $K$. We denote a batch of sequences as $\mathbf{X}=\left[\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(K)}\right]$, with each $\mathbf{x}^{(k)} \stackrel{\text { iid }}{\sim} q(\mathbf{x})$. We obtain the batch NELBO estimator below, where $t(k, b)$ is sampled in sequence $k$ and block $b$ :

$$
\begin{equation*}
\mathcal{L}_{\mathrm{BD}}(\mathbf{X} ; \theta):=l(\mathbf{X} ; \theta)=\frac{1}{K} \sum_{k=1}^{K} \sum_{b=1}^{B} \frac{\alpha_{t(k, b)}^{\prime}}{1-\alpha_{t(k, b)}} \log p_{\theta}\left(\mathbf{x}^{(k), b} \mid \mathbf{x}_{t(k, b)}^{(k), b}, \mathbf{x}^{(k),<b}\right) \tag{9}
\end{equation*}
$$

The variance of the gradient estimator over $M$ batches for each batch $\mathbf{X}^{m} \forall m \in\{1, \ldots, M\}$ is:

$$
\begin{equation*}
\operatorname{Var}_{\mathbf{X}, t}\left[\nabla_{\theta} l(\mathbf{X} ; \theta)\right] \approx \frac{1}{M-1} \sum_{m=1}^{M}\left\|\nabla_{\theta} l\left(\mathbf{X}^{m} ; \theta\right)-\frac{1}{M} \sum_{m=1}^{M} \nabla_{\theta} l\left(\mathbf{X}^{m} ; \theta\right)\right\|_{2}^{2} \tag{10}
\end{equation*}
$$

## 5 Low-Variance Noise Schedules for BD3-LMs

### 5.1 Intuition: Avoid Extreme Mask Rates

We aim to identify schedules that minimize the variance of the gradient estimator and make training most efficient. In a masked setting, we want to mask random numbers of tokens, so that the model
learns to undo varying levels of noise, which is important during sampling. However, if we mask very few tokens, reconstructing them is easy and does not provide useful learning signal. If we mask everything, the optimal reconstruction are the marginals of each token in the data distribution, which is easy to learn, and again is not useful. These extreme masking rates lead to poor high-variance gradients: we want to learn how to clip them via a simple and effective new class of schedules.

### 5.2 Clipped Schedules for Low-Variance Gradients

We propose a class of "clipped" noise schedules that sample mask rates $1-\alpha_{t} \sim \mathcal{U}[\beta, \omega]$ for $0 \leq \beta, \omega \leq 1$. We argue that from the perspective of deriving Monte Carlo gradient estimates, these schedules are equivalent to a continuous schedule where the mask probability is approximately 0 before the specified range such that $1-\alpha_{<\beta} \approx \epsilon$ and approximately 1 after the specified range $1-\alpha_{>\omega} \approx 1-\epsilon$. Consequently, $\alpha_{t}^{\prime}$ is linear within the range: $\alpha_{t}^{\prime} \approx 1 /(\beta-\omega)$.

### 5.3 Data-Driven Clipped Schedules Across Block Sizes

As the optimal mask rates may differ depending on the block size $L^{\prime}$, we adaptively learn the schedule during training. While Kingma et al. (2021) perform variance minimization by isolating a variance term using their squared diffusion loss, this strategy is not directly applicable to our variance estimator in Equation 10 since we seek to reduce variance across random batches in addition to random $t_{b}$.
Instead, we optimize parameters $\beta, \omega$ to directly minimize training variance. To limit the computational burden of the optimization, we use the variance of the estimator of the diffusion ELBO as a proxy for the gradient estimator to optimize $\beta, \omega: \min _{\beta, \omega} \operatorname{Var}_{\mathbf{X}, t}[\mathcal{L}(\mathbf{X} ; \theta, \beta, \omega)]$. We perform a grid search at regular intervals during training to find the optimal $\beta, \omega$ (experimental details in Sec. 6).

In Table 2, we show that variance of the diffusion NELBO is correlated with test perplexity. Under a range of "clipped" noise rate distributions, we find that there exists a unique distribution for each block size $L^{\prime} \in\{4,16,128\}$ that minimizes both the variance of the NELBO and the test perplexity.

Table 2: Perplexities (PPLs; $\downarrow$ ) and variances of the NELBO $\operatorname{Var}_{\mathbf{X}, t}\left[\mathcal{L}_{\mathrm{BD}}(\mathbf{X} ; \theta)\right]$ (Var. NELBO; $\downarrow$ ). Models are trained on LM1B using a linear schedule for 65B tokens, then finetuned for 10B tokens.
| $L^{\prime}$ | $\mathcal{U}[0, .5]$ |  | $\mathcal{U}[.3, .8]$ |  | $\mathcal{U}[.5,1]$ |  | $\mathcal{U}[0,1]$ |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  | PPL | Var. NELBO | PPL | Var. NELBO | PPL | Var. NELBO | PPL | Var. NELBO |
| 128 | 31.72 | 1.03 | 31.78 | 1.35 | 31.92 | 1.83 | 31.78 | 3.80 |
| 16 | 31.27 | 7.90 | 31.19 | 3.62 | 31.29 | 3.63 | 31.33 | 7.39 |
| 4 | 29.23 | 32.68 | 29.37 | 10.39 | 29.16 | 8.28 | 29.23 | 23.65 |


## 6 Experiments

We evaluate BD3-LMs across standard language modeling benchmarks and demonstrate their ability to generate arbitrary-length sequences unconditionally. We pre-train a base BD3-LM using the maximum block size $L^{\prime}=L$ for 850 K gradient steps and fine-tune under varying $L^{\prime}$ for 150 K gradient steps on the One Billion Words dataset (LM1B; Chelba et al. (2014)) and OpenWebText (OWT; Gokaslan et al. (2019)). Details on training and inference are provided in Suppl C.
To reduce the variance of training on the diffusion NELBO, we adaptively learn the range of masking rates by optimizing parameters $\beta, \omega$ as described in Section 5.3. In practice, we do so using a grid search during every validation epoch (after $\sim 5 \mathrm{~K}$ gradient

Table 3: Test perplexities (PPL; $\downarrow$ ) of models trained for 65B tokens on LM1B. Best diffusion value is bolded.
|  | PPL $(\downarrow)$ |
| :--- | :---: |
| Autoregressive |  |
| Transformer-X Base (Dai et al., 2019) | 23.5 |
| Transformer (Sahoo et al., 2024a) | 22.83 |
| Diffusion |  |
| D3PM (absorb) (Austin et al., 2021) | $\leq 82.34$ |
| SEDD (Lou et al., 2024) | $\leq 32.68$ |
| MDLM (Sahoo et al., 2024a) | $\leq 31.78$ |
| Block diffusion (Ours) |  |
| BD3-LMs $L^{\prime}=16$ | $\leq 30.60$ |
|  | $L^{\prime}=8$ |
|  | $\leq 29.83$ |
|  | $L^{\prime}=4$ |


updates) to identify $\beta, \omega: \min _{\beta, \omega} \operatorname{Var}_{\mathbf{X}, t}[\mathcal{L}(\mathbf{X} ; \theta, \beta, \omega)]$. During evaluation, we report likelihood under uniformly sampled mask rates (8) as in Austin et al. (2021); Sahoo et al. (2024a).

### 6.1 Likelihood Evaluation

On LM1B, BD3-LMs outperform all prior diffusion methods in Table 3. Compared to MDLM (Sahoo et al., 2024a), BD3-LMs achieve up to $13 \%$ improvement in perplexity. We observe a similar trend on OpenWebText in Table 4.

We also evaluate the ability of BD3-LMs to generalize to unseen datasets in a zero-shot setting, following the benchmark from Radford et al. (2019). We evaluate the likelihood of models trained with OWT on datasets Penn Tree Bank (PTB; (Marcus et al., 1993)), Wikitext (Merity et al., 2016), LM1B, Lambada (Paperno et al., 2016), AG News (Zhang et al., 2015), and Scientific Papers (Pubmed and Arxiv subsets; (Cohan et al., 2018)). In Table 5, BD3-

Table 4: Test perplexities (PPL; $\downarrow$ ) on OWT for models trained for 524 B tokens. Best diffusion value is bolded.
|  | PPL $(\downarrow)$ |
| :--- | :--- |
| AR (Sahoo et al., 2024a) | 17.54 |
| SEDD (Lou et al., 2024) | $\leq 24.10$ |
| MDLM (Sahoo et al., 2024a) | $\leq 22.98$ |
| BD3-LMs $L^{\prime}=16$ | $\leq 22.27$ |
| $L^{\prime}=8$ | $\leq 21.68$ |
| $L^{\prime}=4$ | $\leq \mathbf{2 0 . 7 3}$ |


LM achieves the best zero-shot perplexity on Pubmed, surpassing AR, and the best perplexity among diffusion models on Wikitext, LM1B, and AG News.

Table 5: Zero-shot validation perplexities $(\downarrow)$ of models trained for 524B tokens on OWT. All perplexities for diffusion models are upper bounds.
|  | PTB | Wikitext | LM1B | Lambada | AG News | Pubmed | Arxiv |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AR | $\mathbf{8 1 . 0 7}$ | $\mathbf{2 5 . 3 2}$ | $\mathbf{5 1 . 1 4}$ | 52.13 | $\mathbf{5 2 . 1 1}$ | 48.59 | 41.22 |
| SEDD | 96.33 | 35.98 | 68.14 | 48.93 | 67.82 | 45.39 | 40.03 |
| MDLM | 90.96 | 33.22 | 64.94 | $\mathbf{4 8 . 2 9}$ | 62.78 | 43.13 | $\mathbf{3 7 . 8 9}$ |
| BD3-LM $L^{\prime}=4$ | 96.81 | 31.31 | 60.88 | 50.03 | 61.67 | $\mathbf{4 2 . 5 2}$ | 39.20 |


### 6.2 Sample Quality and Variable-Length Sequence Generation

One key drawback of many existing diffusion language models (e.g., Austin et al. (2021); Lou et al. (2024)) is that they cannot generate full-length sequences that are longer than the length of the output context chosen at training time. The OWT dataset is useful for examining this limitation, as it contains many documents that are longer than the training context length of 1024 tokens.

We record generation length statistics of 500 variable-length samples in Table 6. We continue sampling tokens until an end-of-sequence token [EOS] is generated or sample quality significantly degrades (as measured by sample entropy).

Table 6: Generation length statistics from sampling 500 documents from models trained on OWT.
|  | Median |  |
| :--- | :---: | :---: |
|  | \# tokens | \# tokens |
| OWT train set | 717 | 131 K |
| AR | 4008 | 131 K |
| SEDD | 1021 | 1024 |
| BD3-LM $L^{\prime}=16$ | 798 | 9982 |


BD3-LMs generate sequences up to $\approx 10 \times$ longer than those of SEDD (Lou et al., 2024), which is restricted to the training context size.
We also examine the sample quality of BD3-LMs through quantitative and qualitative analyses. In Table 7, we generate sequences of lengths $L=1024,2048$ and measure their generative perplexity under GPT2-Large. To sample $L=2048$ tokens from MDLM, we use their block-wise decoding technique (which does not feature block diffusion training as in BD3-LMs).

We also compare to SSD-LM (Han et al., 2022), an alternative block diffusion formulation. Unlike our discrete diffusion framework, SSD-LM uses Gaussian diffusion and does not support likelihood estimation. Further, BD3-LM adopts an efficient sampler from masked diffusion, where the number of generation steps (NFEs) is upper-bounded by $L$ since tokens are never remasked (Sahoo et al., 2024a; Ou et al., 2025). For SSD-LM, we compare sample quality using $T=1 \mathrm{~K}$ diffusion steps per block, matching their experimental setting (yielding $\geq 40 \mathrm{~K}$ NFEs), and $T=25$ where NFEs are comparable across methods.

Table 7: Generative perplexity (Gen. PPL; $\downarrow$ ) and number of function evaluations (NFEs; $\downarrow$ ) of 300 samples of lengths $L=1024$, 2048. All models are trained on OWT. AR, SEDD, MDLM, BD3-LMs use 110 M parameters and are trained on 524B tokens, while SSD-LM uses 400 M parameters and is pre-trained on 122B tokens. Best diffusion value is bolded. We provide further details in Suppl. C.5.
| Model | $L=1024$ |  | $L=2048$ |  |
| :--- | :--- | :--- | :--- | :--- |
|  | Gen. PPL | NFEs | Gen. PPL | NFEs |
| AR | 14.1 | 1 K | 13.2 | 2K |
| Diffusion |  |  |  |  |
| SEDD | 52.0 | 1 K | - | - |
| MDLM | 46.8 | 1 K | 41.3 | 2 K |
| Block Diffusion |  |  |  |  |
| SSD-LM $L^{\prime}=25$ | 37.2 | 40 K | 35.3 | 80 K |
|  | 281.3 | 1 K | 281.9 | 2K |
| BD3-LMs $L^{\prime}=16$ | 33.4 | 1 K | 31.5 | 2 K |
| $L^{\prime}=8$ | 30.4 | 1 K | 28.2 | 2 K |
| $L^{\prime}=4$ | 25.7 | 1 K | 23.6 | 2 K |


BD3-LMs achieve the best generative perplexities compared to previous diffusion methods. Relative to SSD-LM, our discrete approach yields samples with improved generative perplexity using an order of magnitude fewer generation steps. We also qualitatively examine samples taken from BD3-LM and baselines (AR, MDLM) trained on the OWT dataset; we report samples in Suppl. D. We observe that BD3-LM samples have higher coherence than MDLM samples and approach the quality of AR.

### 6.3 Ablations

We assess the impact of the design choices in our proposed block diffusion recipes, namely 1) selection of the noise schedule and 2) the efficiency improvement of the proposed training algorithm relative to a naive implementation.

## Selecting Noise Schedules to Reduce Training Variance

Compared to the linear schedule used in Lou et al. (2024); Sahoo et al. (2024a), training under "clipped" noise schedules is the most effective for reducing the training variance which correlates with test perplexity. In Table 8, the ideal "clipped" masking rates, which are optimized during training, are specific to the block size and further motivate our optimization.

Relative to other standard noise schedules (Chang et al., 2022), "clipped" masking achieves the best performance. As heavier masking is effective for the smaller block size $L^{\prime}=4$, we compare with logarithmic and square root schedules that also encourage heavy masking. As lighter masking is optimal for $L^{\prime}=16$, we compare with square and cosine schedules.

## Efficiency of Training Algorithm

In the BD3-LM training algorithm (Sec. 3.2), we compute $\mathbf{x}_{\text {logit }}$ using two options. We may perform two forward passes through the network (precomputing keys and values for the full sequence $\mathbf{x}$, then computing denoised predictions), or combine these passes by concatenating the two inputs into the same attention kernel.
We find that a single forward pass is more efficient as we reduce memory bandwidth bottlenecks by leveraging efficient attention kernels (Dao et al., 2022; Dong et al., 2024), see Suppl. B.7. Instead of paying the cost

Table 8: Effect of the noise schedule on likelihood estimation. We finetune BD3-LMs on 3B tokens from LM1B and evaluate on a linear schedule. For clipped schedules, we compare optimal clipping for $L^{\prime}=4,16$.

| Noise schedule | PPL | Var. NELBO |
| :--- | :---: | :---: |
| $\mathbf{L}=\mathbf{4}$ |  |  |
| Clipped |  |  |
| $\mathcal{U}[0.45,0.95]$ | $\mathbf{2 9 . 2 1}$ | $\mathbf{6 . 2 4}$ |
| $\mathcal{U}[0.3,0.8]$ | 29.38 | 10.33 |
| Linear $\mathcal{U}[0,1]$ | 30.18 | 23.45 |
| Logarithmic | 30.36 | 23.53 |
| Square root | 31.41 | 26.43 |
| $\mathbf{L}=\mathbf{1 6}$ |  |  |
| Clipped |  |  |
| $\mathcal{U}[0.45,0.95]$ | 31.42 | 3.60 |
| $\mathcal{U}[0.3,0.8]$ | $\mathbf{3 1 . 1 2}$ | $\mathbf{3 . 5 8}$ |
| Linear $\mathcal{U}[0,1]$ | 31.72 | 7.62 |
| Square | 31.43 | 13.03 |
| Cosine | 31.41 | 13.00 |

of two passes through the network, we only pay the cost of a more expensive attention operation. Our vectorized approach has $20-25 \%$ speed-up during training relative to performing two forward passes.

## 7 Discussion and Prior Work

Comparison to D3PM Block diffusion builds off D3PM (Austin et al., 2021) and applies it to each autoregressive conditional. We improve over D3PM in three ways: (1) we extend D3PM beyond fixed sequence lengths; (2) we study the perplexity gap of D3PM and AR models, identify gradient variance as a contributor, and design variance-minimizing schedules; (3) we improve over the perplexity of D3PM models. Our work applies to extensions of D3PM (He et al., 2022; Lou et al., 2024) including ones in continuous time (Campbell et al., 2022; Sun et al., 2022).

Comparison to MDLM BD3-LMs further make use of the perplexity-enhancing improvements in MDLM (Sahoo et al., 2024a; Shi et al., 2024; Ou et al., 2025). We also build upon MDLM: (1) while Sahoo et al. (2024a) point out that their NELBO is invariant to the noise schedule, we show that the noise schedule has a significant effect on gradient variance; (2) we push the state-of-the-art in perplexity beyond MDLM. Note that our perplexity improvements stem not only from block diffusion, but also from optimized schedules, and could enhance standard MDLM and D3PM models.

Comparison to Gaussian Diffusion Alternatively, one may perform diffusion over continuous embeddings of discrete tokens (Li et al., 2022; Dieleman et al., 2022; Chen et al., 2022). This allows using algorithms for continuous data (Song et al., 2020; Ho \& Salimans, 2022), but yields worse perplexity (Graves et al., 2023; Gulrajani \& Hashimoto, 2024).

Comparison to Semi-Autoregressive Diffusion Han et al. (2022; 2023) introduced a block formulation of Gaussian diffusion. BD3-LMs instead extend Austin et al. (2021), and feature: (1) tractable likelihood estimates for principled evaluation; (2) faster generation, as our number of model calls is bounded by the number of generated tokens, while SSD-LM performs orders of magnitude more calls; (3) improved sample quality. AR-Diffusion (Wu et al., 2023) extends SSD-LM with a left-to-right noise schedule; Chen et al. (2025); Ye et al. (2024) apply to decision traces and videos; Hao et al. (2024); Kong et al. (2025) extend to latent reasoning. PARD (Zhao et al., 2024) applies discrete block diffusion to graphs. In contrast, we (1) interpolate between AR/diffusion performance; (2) support KV caching; (3) perform attention within noised blocks, whereas PARD injects new empty blocks.
Autoregressive diffusion models (Hoogeboom et al., 2021b;a) extend any-order AR models (AOARMs; Uria et al. (2014)) to support parallel sampling. Zheng et al. (2024) prove equivalence between MDLM and AO-ARM training. Further extensions of ARMs that compete with diffusion include iterative editing (Gu et al., 2019), parallel and speculative decoding (Gu et al., 2017; Santilli et al., 2023; Cai et al., 2024; Gloeckle et al., 2024), consistency training (Kou et al., 2024), guidance (Sanchez et al., 2023), and cross-modal extensions (Liu et al., 2023; Tian et al., 2025).

Limitations Training BD3-LMs is more expensive than regular diffusion training. We propose a vectorized algorithm that keeps training speed within $<2 \mathrm{x}$ of diffusion training speed; in our experiments, we also pre-train with a standard diffusion loss to further reduce the speed gap. Additionally, BD3-LMs generate blocks sequentially, and hence may face the same speed and controllability constraints as AR especially when blocks are small. Their optimal block size is task specific (e.g., larger for greater control). BD3-LMs are subject to inherent limitations of generative models, including hallucinations (Achiam et al., 2023), copyright infringement (Gokaslan et al., 2024), controllability (Schiff et al., 2024; Wang et al., 2023) and harmful outputs (Bai et al., 2022).

## 8 Conclusion

This work explores block diffusion and is motivated by two problems with existing discrete diffusion: the need to generate arbitrary-length sequences and the perplexity gap to autoregressive models. We introduce BD3-LMs, which represent a block-wise extension of the D3PM framework (Austin et al., 2021), and leverage a specialized training algorithm and custom noise schedules that further improve performance. We observe that in addition to being able to generate long-form documents, these models also improve perplexity, setting a new state-of-the-art among discrete diffusion models.



## Contents

1 Introduction ..... 1
2 Background: Language Modeling Paradigms ..... 2
2.1 Autoregressive Models ..... 3
2.2 Discrete Denoising Diffusion Probabilistic Models ..... 3
3 Block Diffusion Language Modeling ..... 3
3.1 Block Diffusion Distributions and Model Architectures ..... 3
3.2 Efficient Training and Sampling Algorithms ..... 4
4 Understanding Likelihood Gaps Between Diffusion \& AR Models ..... 5
4.1 Masked BD3-LMs ..... 5
4.2 Case Study: Single Token Generation ..... 5
4.3 Diffusion Gap from High Variance Training ..... 6
5 Low-Variance Noise Schedules for BD3-LMs ..... 6
5.1 Intuition: Avoid Extreme Mask Rates ..... 6
5.2 Clipped Schedules for Low-Variance Gradients ..... 7
5.3 Data-Driven Clipped Schedules Across Block Sizes ..... 7
6 Experiments ..... 7
6.1 Likelihood Evaluation ..... 8
6.2 Sample Quality and Variable-Length Sequence Generation ..... 8
6.3 Ablations ..... 9
7 Discussion and Prior Work ..... 10
8 Conclusion ..... 10
A Block Diffusion NELBO ..... 17
B Masked BD3-LMs ..... 17
B. 1 Forward Process ..... 18
B. 2 Reverse Process ..... 18
B. 3 Simplified NELBO for Masked Diffusion Processes ..... 18
B. 4 Recovering the NLL from the NELBO for Single Token Generation ..... 19
B. 5 Tightness of the NELBO ..... 20
B. 6 Specialized Attention Masks ..... 20
B. 7 Optimized Attention Kernel with FlexAttention ..... 21
C Experimental Details ..... 24
C. 1 Datasets ..... 24
C. 2 Architecture ..... 24
C. 3 Training ..... 24
C. 4 Likelihood Evaluation ..... 24
C. 5 Inference ..... 25
D Samples ..... 26

## A Block Diffusion NELBO

Below, we provide the Negative ELBO (NELBO) for the block diffusion parameterization. Recall that the sequence $\mathbf{x}^{1: L}=\left[\mathbf{x}^{1}, \ldots, \mathbf{x}^{L}\right]$ is factorized over $B$ blocks, which we refer to as $\mathbf{x}$ for simplicity, drawn from the data distribution $q(\mathbf{x})$. Specifically, we will factorize the likelihood over $B$ blocks of length $L^{\prime}$, then perform diffusion in each block over $T$ discretization steps. Let $D_{\mathrm{KL}}[\cdot]$ to denote the Kullback-Leibler divergence, $t, s$ be shorthand for $t(i)=i / T$ and $s(i)=(i-1) / T \forall i \in[1, T]$. We derive the NELBO as follows:

$$
\begin{align*}
-\log p_{\theta}(\mathbf{x}) & =-\sum_{b=1}^{B} \log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}^{<b}\right) \\
& =-\sum_{b=1}^{B} \log \mathbb{E}_{q} \frac{p_{\theta}\left(\mathbf{x}_{t(1): t(T)}^{b} \mid \mathbf{x}^{<b}\right)}{q\left(\mathbf{x}_{t(1): t(T)}^{b} \mid \mathbf{x}^{b}\right)} \\
& =-\sum_{b=1}^{B} \log \mathbb{E}_{q} \frac{p_{\theta}\left(\mathbf{x}_{t(T)}^{b} \mid \mathbf{x}^{<b}\right) \prod_{i=1}^{T} p_{\theta}\left(\mathbf{x}_{s(i)}^{b} \mid \mathbf{x}_{t(i)}^{b}, \mathbf{x}^{<b}\right)}{\prod_{i=1}^{T} q\left(\mathbf{x}_{t(i)}^{b} \mid \mathbf{x}_{s(i)}^{b}\right)} \\
& \leq \sum_{b=1}^{B}[\underbrace{-\mathbb{E}_{q} \log p_{\theta}\left(\mathbf{x}^{b} \left\lvert\, \mathbf{x}_{t=\frac{1}{T}}^{b}\right., \mathbf{x}^{<b}\right)}_{\mathcal{L}_{\text {recons }}} \\
& \quad+\underbrace{\mathbb{E}_{t \in\left\{\frac{2}{T}, \ldots, \frac{T-1}{T}, 1\right\}} \mathbb{E}_{q} T \mathrm{D}_{K L}\left(q\left(\mathbf{x}_{s}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{b}\right) \| p_{\theta}\left(\mathbf{x}_{s}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right)\right)}_{\mathcal{L}_{\text {diffusion }}} \\
& \quad+\underbrace{\mathrm{D}_{K L}\left(q\left(\mathbf{x}_{t=1}^{b} \mid \mathbf{x}^{b}\right) \| p_{\theta}\left(\mathbf{x}_{t=1}^{b}\right)\right)}_{\mathcal{L}_{\text {prior }}}] \tag{11}
\end{align*}
$$

## B Masked BD3-LMs

We explore a specific class of block diffusion models that builds upon the masked diffusion language modeling framework. In particular, we focus on masking diffusion processes introduced by Austin et al. (2021) and derive a simplified NELBO under this framework as proposed by Sahoo et al. (2024a); Shi et al. (2024); Ou et al. (2025).
First, we define the diffusion matrix $Q_{t}$ for states $i \in\{1, \ldots, V\}$. Consider the noise schedule function $\alpha_{t} \in[0,1]$, which is a strictly decreasing function in $t$ satisfying $\alpha_{0}=1$ and $\alpha_{1}=0$. Denote the mask index as $m=V$. The diffusion matrix is defined by Austin et al. (2021) as:

$$
\left[Q_{t}\right]_{i j}= \begin{cases}1 & \text { if } i=j=m  \tag{12}\\ \alpha_{t} & \text { if } i=j \neq m \\ 1-\alpha_{t} & \text { if } j=m, i \neq m\end{cases}
$$

The diffusion matrix for the forward marginal $Q_{t \mid s}$ is:

$$
\left[Q_{t \mid s}\right]_{i j}= \begin{cases}1 & \text { if } i=j=m  \tag{13}\\ \alpha_{t \mid s} & \text { if } i=j \neq m \\ 1-\alpha_{t \mid s} & \text { if } j=m, i \neq m\end{cases}
$$

where $\alpha_{t \mid s}=\alpha_{t} / \alpha_{s}$.

## B. 1 Forward Process

Under the D3PM framework (Austin et al., 2021), the forward noise process applied independently for each token $\ell \in\{1, \ldots L\}$ is defined using diffusion matrices $Q_{t} \in \mathbb{R}^{V \times V}$ as

$$
\begin{equation*}
q\left(\mathbf{x}_{t}^{\ell} \mid \mathbf{x}^{\ell}\right)=\operatorname{Cat}\left(\mathbf{x}_{t}^{\ell} ; \bar{Q}_{t} \mathbf{x}^{\ell}\right), \quad \text { with } \quad \bar{Q}_{t(i)}=Q_{t(1)} Q_{t(2)} \ldots Q_{t(i)} \tag{14}
\end{equation*}
$$

## B. 2 Reverse Process

Let $Q_{t \mid s}$ denote the diffusion matrix for the forward marginal. We obtain the reverse posterior $q\left(\mathbf{x}_{s}^{\ell} \mid \mathbf{x}_{t}^{\ell}, \mathbf{x}^{\ell}\right)$ using the diffusion matrices:

$$
\begin{equation*}
q\left(\mathbf{x}_{s}^{\ell} \mid \mathbf{x}_{t}^{\ell}, \mathbf{x}^{\ell}\right)=\frac{q\left(\mathbf{x}_{t}^{\ell} \mid \mathbf{x}_{s}^{\ell}, \mathbf{x}^{\ell}\right) q\left(\mathbf{x}_{s}^{\ell} \mid \mathbf{x}^{\ell}\right)}{q\left(\mathbf{x}_{t}^{\ell} \mid \mathbf{x}^{\ell}\right)}=\operatorname{Cat}\left(\mathbf{x}_{s}^{\ell} ; \frac{Q_{t \mid s} \mathbf{x}_{t}^{\ell} \odot Q_{s}^{\top} \mathbf{x}^{\ell}}{\left(\mathbf{x}_{t}^{\ell}\right)^{\top} Q_{t}^{\top} \mathbf{x}^{\ell}}\right) \tag{15}
\end{equation*}
$$

where $\odot$ denotes the Hadmard product between two vectors.

## B. 3 Simplified NELBO for Masked Diffusion Processes

Following Sahoo et al. (2024a); Shi et al. (2024); Ou et al. (2025), we simplify the NELBO in the case of masked diffusion processes. Below, we provide the outline of the NELBO derivation; see the full derivation in Sahoo et al. (2024a); Shi et al. (2024); Ou et al. (2025).

We will first focus on simplifying the diffusion loss term $\mathcal{L}_{\text {diffusion }}$ in Eq. 11. We employ the SUBSparameterization proposed in Sahoo et al. (2024b) which simplifies the denoising model $p_{\theta}$ for masked diffusion. In particular, we enforce the following constraints on the design of $p_{\theta}$ by leveraging the fact that there only exists two possible states in the diffusion process $\mathbf{x}_{t}^{\ell} \in\left\{\mathbf{x}^{\ell}, \mathbf{m}\right\} \forall \ell \in\{1, \ldots, L\}$.

1. Zero Masking Probabilities. We set $p_{\theta}\left(\mathbf{x}^{\ell}=\mathbf{m} \mid \mathbf{x}_{t}^{\ell}\right)=0$ (as the clean sequence $\mathbf{x}$ doesn't contain masks).
2. Carry-Over Unmasking. The true posterior for the case where $\mathbf{x}_{t}^{\ell} \neq \mathbf{m}$ is $q\left(\mathbf{x}_{s}^{\ell}=\mathbf{x}_{t}^{\ell} \mid \mathbf{x}_{t}^{\ell} \neq\right. \mathbf{m})=1$ (if a token is unmasked in the reverse process, it is never remasked). Thus, we simplify the denoising model by setting $p_{\theta}\left(\mathbf{x}_{s}^{\ell}=\mathbf{x}_{t}^{\ell} \mid \mathbf{x}_{t}^{\ell} \neq \mathbf{m}\right)=1$.

As a result, we will only approximate the posterior $p_{\theta}\left(\mathbf{x}_{s}^{\ell}=\mathbf{x}^{\ell} \mid \mathbf{x}_{t}^{\ell}=\mathbf{m}\right)$. Let $\mathbf{x}^{b, \ell}$ denote a token in the $\ell$-th position in block $b \in\{1, \ldots, B\}$. The diffusion loss term becomes:

$$
\begin{aligned}
\mathcal{L}_{\text {diffusion }} & =\sum_{b=1}^{B} \mathbb{E}_{t} \mathbb{E}_{q} T\left[\mathrm{D}_{\mathrm{KL}}\left[q\left(\mathbf{x}_{s}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{b}\right) \| p_{\theta}\left(\mathbf{x}_{s}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right)\right]\right] \\
& =\sum_{b=1}^{B} \mathbb{E}_{t} \mathbb{E}_{q} T\left[\sum_{\ell=1}^{L^{\prime}} \mathrm{D}_{\mathrm{KL}}\left[q\left(\mathbf{x}_{s}^{b, \ell} \mid \mathbf{x}_{t}^{b, \ell}, \mathbf{x}^{b, \ell}\right) \| p_{\theta}\left(\mathbf{x}_{s}^{b, \ell} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right)\right]\right]
\end{aligned}
$$

$\mathrm{D}_{\mathrm{KL}}$ is simply the discrete-time diffusion loss for the block $b$; hence, from Sahoo et al. (2024a) (Suppl. B.1), we get:

$$
\begin{align*}
& =\sum_{b=1}^{B} \mathbb{E}_{t} \mathbb{E}_{q} T\left[\sum_{\ell=1}^{L^{\prime}} \frac{\alpha_{t}-\alpha_{s}}{1-\alpha_{t}} \log p_{\theta}\left(\mathbf{x}^{b, \ell} \mid \mathbf{x}_{t}^{b, \ell}, \mathbf{x}^{<b}\right)\right] \\
& =\sum_{b=1}^{B} \mathbb{E}_{t} \mathbb{E}_{q} T\left[\frac{\alpha_{t}-\alpha_{s}}{1-\alpha_{t}} \log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right)\right] \tag{16}
\end{align*}
$$

Lastly, we obtain a tighter approximation of the likelihood by taking the diffusion steps $T \rightarrow \infty$ (Sahoo et al., 2024a), for which $T\left(\alpha_{t}-\alpha_{s}\right)=\alpha_{t}^{\prime}$ :

$$
\begin{equation*}
\mathcal{L}_{\text {diffusion }}=\sum_{b=1}^{B} \mathbb{E}_{t \sim[0,1]} \mathbb{E}_{q}\left[\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}} \log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right)\right] \tag{17}
\end{equation*}
$$

For the continuous time case, Sahoo et al. (2024a) (Suppl. A.2.4) show the reconstruction loss reduces to 0 as $\mathbf{x}_{t(1)}^{b} \sim \lim _{T \rightarrow \infty} \operatorname{Cat}\left(. ; \mathbf{x}_{t=\frac{1}{T}}^{b}\right)=\operatorname{Cat}\left(. ; \mathbf{x}^{b}\right)$. Using this, we obtain:

$$
\begin{align*}
\mathcal{L}_{\text {recons }} & =-\mathbb{E}_{q} \log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t(1)}^{b}, \mathbf{x}^{<b}\right) \\
& =-\log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t(1)}^{b}=\mathbf{x}^{b}, \mathbf{x}^{<b}\right) \\
& =0 \tag{18}
\end{align*}
$$

The prior loss $\mathcal{L}_{\text {prior }}=\mathrm{D}_{K L}\left(q\left(\mathbf{x}_{t=1}^{b} \mid \mathbf{x}^{b}\right) \| p_{\theta}\left(\mathbf{x}_{t=1}^{b}\right)\right)$ also reduces to 0 because $\alpha_{t=1}=0$ which ensures $q\left(\mathbf{x}_{t=1}^{b} \mid \mathbf{x}^{b}\right)=\operatorname{Cat}(. ; \mathbf{m})$ and $p_{\theta}\left(\mathbf{x}_{t=1}^{b}\right)=\operatorname{Cat}(. ; \mathbf{m})$; see Sahoo et al. (2024a) (Suppl. A.2.4). Finally, we obtain a simple objective that is a weighted average of cross-entropy terms:

$$
\begin{equation*}
\mathcal{L}_{\mathrm{BD}}(\mathbf{x} ; \theta)=\sum_{b=1}^{B} \mathbb{E}_{t \sim[0,1]} \mathbb{E}_{q}\left[\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}} \log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right)\right] \tag{19}
\end{equation*}
$$

The above NELBO is invariant to the choice of noise schedule $\alpha_{t}$; see Sahoo et al. (2024a) (Suppl. E.1.1).

## B. 4 Recovering the NLL from the NELBO for Single Token Generation

Consider the block diffuson NELBO for a block size of 1 where $L^{\prime}=1, B=L$. The block diffusion NELBO is equivalent to the AR NLL when modeling a single token:

$$
\begin{aligned}
-\log p(\mathbf{x}) & \leq \sum_{b=1}^{L} \mathbb{E}_{t \sim[0,1]} \mathbb{E}_{q}\left[\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}} \log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right)\right] \\
& \because \alpha_{t}^{\prime}=-1 \text { and } \alpha_{t}=1-t \\
& =-\sum_{b=1}^{L} \mathbb{E}_{t \sim[0,1]} \mathbb{E}_{q}\left[\frac{1}{t} \log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right)\right] \\
& =-\sum_{b=1}^{L} \mathbb{E}_{t \sim[0,1]} \frac{1}{t} \mathbb{E}_{q}\left[\log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right)\right]
\end{aligned}
$$

Expanding $\mathbb{E}_{q}[$.$] ,$

$$
\begin{align*}
=-\sum_{b=1}^{L} \mathbb{E}_{t \sim[0,1]} \frac{1}{t} & {\left[q\left(\mathbf{x}_{t}^{b}=\mathbf{m} \mid \mathbf{x}^{b}\right) \log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}=\mathbf{m}, \mathbf{x}^{<b}\right)\right.} \\
& \left.+q\left(\mathbf{x}_{t}^{b}=\mathbf{x}^{b} \mid \mathbf{x}^{b}\right) \log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}=\mathbf{x}^{b}, \mathbf{x}^{<b}\right)\right] \tag{20}
\end{align*}
$$

Recall that our denoising model employs the SUBS-parameterization proposed in Sahoo et al. (2024b). The "carry-over unmasking" property ensures that $\log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}=\mathbf{x}^{b}, \mathbf{x}^{<b}\right)=0$, as an unmasked token is simply copied over from from the input of the denoising model to the output. Hence, (20) reduces to following:

$$
\begin{aligned}
-\log p_{\theta}(\mathbf{x}) & \leq-\sum_{b=1}^{L} \mathbb{E}_{t \sim[0,1]} \frac{1}{t} q\left(\mathbf{x}_{t}^{b}=\mathbf{m} \mid \mathbf{x}^{b}\right) \log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}=\mathbf{m}, \mathbf{x}^{<b}\right) \\
& \because q\left(\mathbf{x}_{t}^{b}=\mathbf{m} \mid \mathbf{x}^{b}\right)=t, \text { we get: } \\
& =-\sum_{b=1}^{L} \mathbb{E}_{t \sim[0,1]} \log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}=\mathbf{m}, \mathbf{x}^{<b}\right)
\end{aligned}
$$

$$
\begin{equation*}
=-\sum_{b=1}^{L} \log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{m}, \mathbf{x}^{<b}\right) \tag{21}
\end{equation*}
$$

For single-token generation ( $L^{\prime}=1$ ) we recover the autoregressive NLL.

## B. 5 Tightness of the NELBO

For block sizes $1 \leq K \leq L$, we show that $-\log p(\mathbf{x}) \leq \mathcal{L}_{K} \leq \mathcal{L}_{K+1}$. Consider $K=1$, where we recover the autoregressive NLL (see Suppl B.4):

$$
\begin{align*}
\mathcal{L}_{1} & =\sum_{b=1}^{L} \log \mathbb{E}_{t \sim[0,1]} \mathbb{E}_{q} \frac{\alpha_{t}^{\prime}}{1-\alpha_{t}} p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right) \\
& =-\sum_{b=1}^{L} \log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{m}, \mathbf{x}^{<b}\right) \tag{22}
\end{align*}
$$

Consider the ELBO for block size $K=2$ :

$$
\begin{equation*}
\mathcal{L}_{2}=\sum_{b=1}^{L / 2} \log \mathbb{E}_{t \sim[0,1]} \mathbb{E}_{q} \frac{\alpha_{t}^{\prime}}{1-\alpha_{t}} p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right) \tag{23}
\end{equation*}
$$

We show that $\mathcal{L}_{1} \leq \mathcal{L}_{2}$, and this holds for all $1 \leq K \leq L$ by induction. Let $\mathbf{x}^{b, \ell}$ correspond to the token in position $\ell \in\left[1, L^{\prime}\right]$ of block $b$. We derive the below inequality:

$$
\begin{align*}
-\sum_{b=1}^{L} \log p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{m}, \mathbf{x}^{<b}\right) & =-\sum_{b=1}^{L / 2} \log \mathbb{E}_{t \sim[0,1]} \mathbb{E}_{q} \frac{1}{1-\alpha_{t}} p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right) \\
& =-\sum_{b=1}^{L / 2} \log \mathbb{E}_{t \sim[0,1]} \mathbb{E}_{q} \prod_{i=1}^{2} \frac{1}{1-\alpha_{t}} p_{\theta}\left(\mathbf{x}^{b, \ell} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right) \\
& =-\sum_{b=1}^{L / 2} \log \prod_{i=1}^{2} \mathbb{E}_{t \sim[0,1]} \mathbb{E}_{q} \frac{1}{1-\alpha_{t}} p_{\theta}\left(\mathbf{x}^{b, \ell} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right) \\
& \leq-\sum_{b=1}^{L / 2} \sum_{i=1}^{2} \log \mathbb{E}_{t \sim[0,1]} \mathbb{E}_{q} \frac{1}{1-\alpha_{t}} p_{\theta}\left(\mathbf{x}^{b, \ell} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right) \tag{24}
\end{align*}
$$

## B. 6 Specialized Attention Masks

We aim to model conditional probabilities $p_{\theta}\left(\mathbf{x}^{b} \mid \mathbf{x}_{t}^{b}, \mathbf{x}^{<b}\right)$ for all blocks $b \in[1, B]$ simultaneously by designing an efficient training algorithm with our transformer backbone. However, modeling all $B$ conditonal terms requires processing both the noised sequence $\mathbf{x}_{t}^{b}$ and the conditional context $\mathbf{x}^{<b}$ for all $b$.

Rather than calling the denoising network $B$ times, we process both sequences simultaneously by concatenating them $\mathbf{x}_{\text {full }} \leftarrow \mathbf{x}_{t} \oplus \mathbf{x}$ as input to a transformer. We update this sequence $\mathbf{x}_{\text {full }}$ of length $2 L$ tokens using a custom attention mask $\mathcal{M}_{\text {full }} \in\{0,1\}^{2 L \times 2 L}$ for efficient training.
The full attention mask is comprised of four $L \times L$ smaller attention masks:

$$
\mathcal{M}_{\text {full }}=\left[\begin{array}{cc}
\mathcal{M}_{B D} & \mathcal{M}_{O B C} \\
\mathbf{0} & \mathcal{M}_{B C}
\end{array}\right]
$$

where $\mathcal{M}_{B D}$ and $\mathcal{M}_{O B C}$ are used to update the representation of $\mathbf{x}_{t}$ and $\mathcal{M}_{B C}$ is used to update the representation of $\mathbf{x}$. We define these masks as follows:

- $\mathcal{M}_{B D}$ (Block-diagonal mask): Self-attention mask within noised blocks $\mathbf{x}_{t}^{b}$

$$
\left[\mathcal{M}_{B D}\right]_{i j}= \begin{cases}1 & \text { if } i, j \text { are in the same block } \\ 0 & \text { otherwise }\end{cases}
$$

- $\mathcal{M}_{O B C}$ (Offset block-causal mask): Cross-attention to conditional context $\mathbf{x}^{<b}$

$$
\left[\mathcal{M}_{O B C}\right]_{i j}= \begin{cases}1 & \text { if } j \text { belongs in a block before } i \\ 0 & \text { otherwise }\end{cases}
$$

- $\mathcal{M}_{B C}$ (Block-causal mask): Attention mask for updating $\mathbf{x}^{b}$

$$
\left[\mathcal{M}_{B C}\right]_{i j}= \begin{cases}1 & \text { if } j \text { belongs in the same block as } i, \text { or a block before } i \\ 0 & \text { otherwise }\end{cases}
$$

We visualize an example attention mask for $L=6$ and block size $L^{\prime}=2$ in Figure 3.

![](https://cdn.mathpix.com/cropped/2025_10_29_b6105f7bcf5e026ed288g-21.jpg?height=663&width=1194&top_left_y=750&top_left_x=463)
Figure 3: Example of Specialized Attention Mask

## B. 7 Optimized Attention Kernel with FlexAttention

As Figure 3 demonstrates, our attention matrix is extremely sparse. We can exploit this sparsity to massively improve the efficiency of BD3-LMs.
FlexAttention (Dong et al., 2024) is a compiler-driven programming model that enables efficient implementation of attention mechanisms with structured sparsity in PyTorch. It provides a flexible interface for defining custom attention masks while maintaining high performance comparable to manually optimized attention kernels.
Below in Fig. 4 we define a block-wise attention mask, block_diff_mask, based on its definition as $\mathcal{M}_{\text {full }} \in\{0,1\}^{2 L \times 2 L}$ in Suppl. B.6. We fuse the attention operations into a single FlexAttention kernel designed to exploit the sparsity in our attention matrix to increase computational efficiency. By doing so, we perform the following optimizations:

- Precomputed Block Masking: The create_block_mask utility generates a sparse attention mask at compile-time, avoiding per-step computation of invalid attention entries. Through sparsity-aware execution, FlexAttention kernels reduce the number of FLOPs in the attention computation.
- Reduced Memory Footprint: By leveraging block-level sparsity, the attention mechanism avoids full materialization of large-scale attention matrices, significantly reducing memory overhead. FlexAttention minimizes memory accesses by skipping fully masked blocks.
- Optimized Computation via torch. compile: The integration of torch. compile enables kernel fusion and efficient execution on GPUs by generating optimized Triton-based kernels. This efficiently parallelizes masked attention computations using optimized GPU execution paths.

```
def block_diff_mask(b, h, q_idx, kv_idx, block_size, n):
    """
    Constructs the specialized block diffusion attention mask composed of
                    three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for
                    conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0
    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        block_size: Defines the block structure.
        n: Sequence length of x_0 and x_t
    Returns:
        A boolean attention mask.
    """
    # Indicate whether token belongs to xt (0) or x0 (1)
    x0_flag_q = (q_idx >= n)
    x0_flag_kv = (kv_idx >= n)
    # Compute block indices
    block_q = torch.where(x0_flag_q == 1,
                (q_idx - n) // block_size,
                q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1,
                (kv_idx - n) // block_size,
                kv_idx // block_size)
    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (
        (block_q > block_kv)
        & (x0_flag_q == 0)
        & (x0_flag_kv == 1)
    )
    # **3. Block-Causal Mask (M_BC) **
    block_causal = (
        (block_q >= block_kv)
        & (x0_flag_q == 1)
        & (x0_flag_kv == 1)
    )
    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | block_causal
```

Figure 4: We can adapt the masking strategy from Fig. 3 to a FlexAttention compatible sparse masking function as above. This enables the creation of a customized JIT attention operation that uses significantly less memory with up to $\approx 5 \mathrm{X}$ speedup over the naive native scaled_dot_product_attention implementation in PyTorch ( $\geq 2.5$ ) on a A5000 GPU with $L=1024$ and batch size $B=16$.

```
from torch.nn.attention.flex_attention import flex_attention,
        create_block_mask
from functools import partial
# Define block-wise attention mask
my_block_diff_mask = partial(block_diff_mask, seq_len=seq_len, block_size
        =block_size)
# Generate optimized sparse block mask
block_mask = create_block_mask(my_block_diff_mask, None, None, seq_len*2,
            seq_len*2, device=device)
# Compute attention using FlexAttention
# Use no-cudagraphs to avoid an extra copy on small compile graphs.
# Use max-autotune if compiling a larger model all at once.
@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def single_pass_block_diff_attn(q, k, v, block_mask):
    return flex_attention(q, k, v, block_mask=block_mask)
```

Figure 5: Attention computation using FlexAttention with our proposed custom mask.

This implementation exploits FlexAttention's ability to dynamically optimize execution based on the provided sparsity pattern. By precomputing block-level sparsity and leveraging efficient kernel fusion, it enables scalable attention computation for long sequences.
Overall, this approach provides a principled method to accelerate attention computations while preserving structured dependency constraints. End-to-end, replacing FlashAttention kernels using a custom mask with FlexAttention kernels leads to $\approx 15 \%$ speedup in a model forward pass. We use a single A5000 for $L=1024$ and batch size $B=16$.

## C Experimental Details

We closely follow the same training and evaluation setup as used by Sahoo et al. (2024a).

## C. 1 Datasets

We conduct experiments on two datasets: The One Billion Word Dataset (LM1B; Chelba et al. (2014)) and OpenWebText (OWT; Gokaslan et al. (2019)). Models trained on LM1B use the bert-base-uncased tokenizer and a context length of 128 . We report perplexities on the test split of LM1B. Models trained on OWT use the GPT2 tokenizer Radford et al. (2019) and a context length of 1024. Since OWT does not have a validation split, we leave the last 100 k documents for validation.

In preparing LM1B examples, Sahoo et al. (2024a) pad each example to fit in the context length of $L=128$ tokens. Since most examples consist of only a single sentence, block diffusion modeling for larger block sizes $L^{\prime}>4$ would not be useful for training. Instead, we concatenate and wrap sequences to a length of 128 . As a result, we retrain our autoregressive baseline, SEDD, and MDLM on LM1B with wrapping.
Similarly for OWT, we do not pad or truncate sequences, but concatenate them and wrap them to a length of 1024 similar to LM1B. For unconditional generation experiments in Section 6.2, we wish to generate sequences longer than the context length seen during training. However, Sahoo et al. (2024a) inject beginning-of-sequence and end-of-sequence tokens ([BOS], [EOS] respectively) at the beginning and end of the training context. Thus, baselines from Sahoo et al. (2024a) will generate sequences that match the training context size. To examine model generations across varying lengths in Section 6.2, we retrain our AR, SEDD, and MDLM baselines without injecting [BOS] and [EOS] tokens in the examples. We also adopt this preprocessing convention for training all BD3-LMs on OWT.

## C. 2 Architecture

The model architecture augments the diffusion transformer (Peebles \& Xie, 2023) with rotary positional embeddings (Su et al., 2021). We parameterize our autoregressive baselines, SEDD, MDLM, and BD3-LMs with a transformer architecture from Sahoo et al. (2024a) that uses 12 layers, a hidden dimension of 768 , and 12 attention heads. This corresponds to 110 M parameters. We do not include timestep conditioning as Sahoo et al. (2024a) show it does not affect performance. We use the AdamW optimizer with a batch size of 512 and constant learning rate warmup from 0 to $3 e-4$ for 2.5 K gradient updates.

## C. 3 Training

We train a base BD3-LM using the maximum context length $L^{\prime}=L$ for 850 K gradient steps. Then, we fine-tune under varying $L^{\prime}$ using the noise schedule optimization for 150 K gradient steps on the One Billion Words dataset (LM1B) and OpenWebText (OWT). This translates to 65B tokens and 73 epochs on LM1B, 524B tokens and 60 epochs on OWT. We use 3090, A5000, A6000, and A100 GPUs.

## C. 4 Likelihood Evaluation

We use a single Monte Carlo estimate for sampling $t$ to evaluate the likelihood of a token block. We adopt a low-discrepancy sampler proposed in Kingma et al. (2021) that reduces the variance of this estimate by ensuring the time steps are more evenly spaced across the interval $[0,1]$ following Sahoo et al. (2024a). In particular, we sample the time step for each block $b \in\{1, \ldots, B\}$ and sequence $k \in \{1, \ldots, K\}$ from a different partition of the uniform interval $t(k, b) \sim \mathcal{U}\left[\frac{(k-1) B+b-1}{K B}, \frac{(k-1) B+b}{K B}\right]$.
This low-discrepancy sampler is used for evaluation. For training, each masking probability may be sampled from a "clipped" range $1-\alpha_{t} \sim \mathcal{U}[\beta, \omega]$. During training, we uniformly sample $t \in[0,1]$ under the low-discrepancy sampler. We then apply a linear interpolation to ensure that the masking probability is linear within the desired range: $1-\alpha_{t}=\beta+(\omega-\beta) t$.

When reporting zero-shot likelihoods on benchmark datasets from Radford et al. (2019) using models trained on OWT, we wrap all sequences to 1024 tokens and do not add [EOS] between sequences following Sahoo et al. (2024a).

## C. 5 Inference

Generative Perplexity We report generative perplexity under GPT2-Large from models trained on OWT using a context length of 1024 tokens. Since GPT2-Large uses a context size of 1024, we compute the generative perplexity for samples longer than 1024 tokens using a sliding window with a stride length of 512 tokens.

Nucleus Sampling Following SSD-LM (Han et al., 2022), we employ nucleus sampling for BD3LMs and our baselines. For SSD-LM, we use their default hyperparameters $p=0.95$ for block size $L^{\prime}=25$. For BD3-LMs, AR and MDLM, we use $p=0.9$. For SEDD, we find that $p=0.99$ works best.

Number of Diffusion Steps In Table 7, BD3-LMs and MDLM use $T=5 \mathrm{~K}$ diffusion steps. BD3LMs and MDLM use efficient sampling by caching the output of the denoising network as proposed by Sahoo et al. (2024a); Ou et al. (2025), which ensures that the number of generation steps does not exceed the sample length $L$. Put simply, once a token is unmasked, it is never remasked as a result of the simplified denoising model (Suppl. B.3). We use MDLM's block-wise decoding algorithm for generating variable-length sequences, however these models are not trained with block diffusion. We adopt their default stride length of 512 tokens.
SSD-LM (first row in Table 7) and SEDD use $T=1 \mathrm{~K}$ diffusion steps. Since block diffusion performs $T$ diffusion steps for each block $b \in\{1, \ldots, B\}$, SSD-LM undergoes $B T$ generation steps. Thus to fairly compare with SSD-LM, we also report generative perplexity for $T=25$ diffusion steps so that the number of generation steps does not exceed the sequence length (second row in Table 7).

Improved Categorical Sampling of Diffusion Models We employ two improvements to Gumbelbased categorical sampling of diffusion models as proposed by Zheng et al. (2024).
First, we use the corrected Gumbel-based categorical sampling from Zheng et al. (2024) by sampling 64-bit Gumbel variables. Reducing the precision to 32-bit has been shown to significantly truncate the Gumbel variables, lowering the temperature and decreasing the sentence entropy.
Second, Zheng et al. (2024) show that the MDLM sampling time scales with the diffusion steps $T$, even though the number of generation steps is bounded by the sequence length. For sample length $L$ and vocabulary size $V$, the sampler requires sampling $\mathcal{O}(T L V)$ uniform variables and performing logarithmic operations on them.
We adopt the first-hitting sampler proposed by Zheng et al. (2024) that requires sampling $\mathcal{O}(L V)$ uniform variables, and thus greatly improves sampling speed especially when $T \gg L$. The firsthitting sampler is theoretically equivalent to the MDLM sampler and leverages two observations: (1) the transition probability is independent of the denoising network, (2) the transition probability is the same for all masked tokens for a given $t$. Thus, the first timestep where a token is unmasked can be analytically sampled as follows (assuming a linear schedule where $\alpha_{t}=1-t$ ):

$$
\begin{equation*}
t_{n-1}=t_{n} u^{1 / n} \tag{25}
\end{equation*}
$$

where $n \in\{L, \ldots, 1\}$ denotes the number of masked tokens, $u_{n} \sim \mathcal{U}[0,1]$ and $t_{n-1}$ corresponds to the first timestep where $n-1$ tokens are masked.

Variable-Length Sequence Generation For arbitrary-length sequence generation using BD3-LMs and AR in Table 6, we continue to sample tokens until the following stopping criteria are met:

1. an [EOS] token is sampled
2. the average entropy of the the last 256 -token chunk is below 4
where criterion 2 are necessary to prevent run-on samples from compounding errors (for example, a sequence of repeating tokens). We find that degenerate samples with low entropy result in significantly
low perplexities under GPT2 and lower the reported generative perplexity. Thus, when a sample meets criterion 2, we regenerate the sample when reporting generative perplexity in Table 7.
