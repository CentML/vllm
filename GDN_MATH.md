# Mathematical Analysis of `fused_recurrent_gated_delta_rule_fwd`

## Overview

This kernel implements a **Gated Delta Rule** recurrence, which is a form of fast linear attention with a learnable state update mechanism. It's designed for efficient autoregressive (recurrent) computation in linear attention architectures.

## Dimension Notation

| Symbol | Description | Example from benchmark |
|--------|-------------|----------------------|
| $B$ | Batch size (number of input sequences when not using variable-length) | 1 |
| $T$ | Total number of tokens | 1024 |
| $N$ | Number of sequences (= `len(cu_seqlens) - 1` for variable-length) | 1024 |
| $H$ | Number of key/query heads | 4 (TP=4) or 8 (TP=2) |
| $H_V$ | Number of value heads (can be > $H$ for Grouped Value Attention) | 8 (TP=4) or 16 (TP=2) |
| $K$ | Key/Query head dimension | 128 |
| $V$ | Value head dimension | 128 |

## Input Tensors (for the specific benchmark case)

| Tensor | Shape | Description |
|--------|-------|-------------|
| $\mathbf{Q}$ | $[B, T, H, K]$ | Query tensor |
| $\mathbf{K}$ | $[B, T, H, K]$ | Key tensor |
| $\mathbf{V}$ | $[B, T, H_V, V]$ | Value tensor |
| $\mathbf{g}$ | $[B, T, H_V]$ | Decay gate (log-space) |
| $\boldsymbol{\beta}$ | $[B, T, H_V]$ | Learning rate / mixing coefficient |
| $\mathbf{h}_0$ | $[N, H_V, K, V]$ | Initial hidden state |
| `cu_seqlens` | $[N+1]$ | Cumulative sequence lengths for variable-length batching |
| `ssm_state_indices` | $[N]$ | Indices mapping sequences to states |

## Algorithm

The kernel processes each sequence independently. For a single sequence with $T$ tokens and a single value head, the recurrence is:

### Initialization

The hidden state is initialized from the initial state:

$$
\mathbf{h}_{-1} \in \mathbb{R}^{K \times V}
$$

### Per-Timestep Recurrence (for $t = 0, 1, \ldots, T-1$)

**Step 1: L2 Normalization** (since `use_qk_l2norm_in_kernel=True`)

$$
\bar{\mathbf{q}}_t = \frac{\mathbf{q}_t}{\|\mathbf{q}_t\|_2 + \epsilon} \quad \in \mathbb{R}^{K}
$$

$$
\bar{\mathbf{k}}_t = \frac{\mathbf{k}_t}{\|\mathbf{k}_t\|_2 + \epsilon} \quad \in \mathbb{R}^{K}
$$

where $\epsilon = 10^{-6}$.

**Step 2: Scale Query**

$$
\tilde{\mathbf{q}}_t = s \cdot \bar{\mathbf{q}}_t \quad \in \mathbb{R}^{K}
$$

where $s$ is the scale factor (typically $1/\sqrt{K}$).

**Step 3: Apply Decay Gate**

$$
\mathbf{h}_t \leftarrow \mathbf{h}_{t-1} \cdot e^{g_t} \quad \in \mathbb{R}^{K \times V}
$$

This is an element-wise multiplication with a scalar decay factor. The gate $g_t \in \mathbb{R}$ controls how much of the previous state to retain. Typically $g_t \leq 0$ (e.g., from `log_sigmoid`), so $e^{g_t} \in (0, 1]$.

**Step 4: Compute Delta (Prediction Error)**

$$
\boldsymbol{\delta}_t = \mathbf{v}_t - \bar{\mathbf{k}}_t^\top \mathbf{h}_t \quad \in \mathbb{R}^{V}
$$

This computes the difference between the target value $\mathbf{v}_t$ and what the current state would predict given key $\bar{\mathbf{k}}_t$. The term $\bar{\mathbf{k}}_t^\top \mathbf{h}_t$ is:

$$
\left(\bar{\mathbf{k}}_t^\top \mathbf{h}_t\right)_j = \sum_{i=1}^{K} \bar{k}_{t,i} \cdot h_{t,i,j} \quad \in \mathbb{R}^{V}
$$

**Step 5: Scale Delta by Beta**

$$
\boldsymbol{\delta}'_t = \beta_t \cdot \boldsymbol{\delta}_t \quad \in \mathbb{R}^{V}
$$

The scalar $\beta_t \in \mathbb{R}$ acts as a learning rate controlling the magnitude of the state update.

**Step 6: Update State (Rank-1 Update)**

$$
\mathbf{h}_t \leftarrow \mathbf{h}_t + \bar{\mathbf{k}}_t \otimes \boldsymbol{\delta}'_t \quad \in \mathbb{R}^{K \times V}
$$

where $\otimes$ denotes the outer product:

$$
\left(\bar{\mathbf{k}}_t \otimes \boldsymbol{\delta}'_t\right)_{i,j} = \bar{k}_{t,i} \cdot \delta'_{t,j} \quad \in \mathbb{R}^{K \times V}
$$

**Step 7: Compute Output**

$$
\mathbf{o}_t = \tilde{\mathbf{q}}_t^\top \mathbf{h}_t \quad \in \mathbb{R}^{V}
$$

which expands to:

$$
o_{t,j} = \sum_{i=1}^{K} \tilde{q}_{t,i} \cdot h_{t,i,j} \quad \in \mathbb{R}^{V}
$$

**Step 8: Store Final State**

The updated state $\mathbf{h}_t$ is stored. With `inplace_final_state=True`, this overwrites the initial state tensor at the appropriate index.

---

## Complete Single-Step Recurrence (Summary)

Given inputs at timestep $t$:
- $\mathbf{q}_t \in \mathbb{R}^K$, $\mathbf{k}_t \in \mathbb{R}^K$, $\mathbf{v}_t \in \mathbb{R}^V$
- $g_t \in \mathbb{R}$, $\beta_t \in \mathbb{R}$
- Previous state $\mathbf{h}_{t-1} \in \mathbb{R}^{K \times V}$

The recurrence is:

$$
\boxed{
\begin{aligned}
\bar{\mathbf{q}}_t &= \frac{s \cdot \mathbf{q}_t}{\|\mathbf{q}_t\|_2 + \epsilon} & \in \mathbb{R}^K \\[0.5em]
\bar{\mathbf{k}}_t &= \frac{\mathbf{k}_t}{\|\mathbf{k}_t\|_2 + \epsilon} & \in \mathbb{R}^K \\[0.5em]
\mathbf{h}_t &= e^{g_t} \cdot \mathbf{h}_{t-1} + \bar{\mathbf{k}}_t \otimes \left[\beta_t \left(\mathbf{v}_t - \bar{\mathbf{k}}_t^\top \mathbf{h}_{t-1} \cdot e^{g_t}\right)\right] & \in \mathbb{R}^{K \times V} \\[0.5em]
\mathbf{o}_t &= \bar{\mathbf{q}}_t^\top \mathbf{h}_t & \in \mathbb{R}^V
\end{aligned}
}
$$

---

## Interpretation

### Delta Rule Connection
The name "Delta Rule" comes from the Widrow-Hoff / LMS learning rule in neural networks. The state $\mathbf{h}$ acts as a learnable associative memory (key-value store). The update:

$$
\mathbf{h} \leftarrow \mathbf{h} + \mathbf{k} \otimes (\mathbf{v} - \mathbf{k}^\top \mathbf{h})
$$

is analogous to gradient descent on the squared error $\|\mathbf{v} - \mathbf{k}^\top \mathbf{h}\|^2$ with $\mathbf{k}$ as the input.

### Gated Decay
The exponential decay $e^{g_t}$ provides a **forget gate**, allowing the model to selectively forget old information. This makes the recurrence more expressive than vanilla linear attention.

### Linear Attention Perspective
This can be viewed as a recurrent form of linear attention where:
- $\mathbf{h}_t$ is the "running sum" state (analogous to $\sum_{i \leq t} \mathbf{k}_i \mathbf{v}_i^\top$ in linear attention)
- The delta rule modification prevents unbounded state growth and improves expressivity

---

## Parallelization Strategy

The kernel parallelizes over:
1. **Key dimension blocks** ($NK = \lceil K / BK \rceil$, constrained to 1 in current implementation)
2. **Value dimension blocks** ($NV = \lceil V / BV \rceil$)
3. **Sequences × Value heads** ($N \times H_V$)

Each thread block handles one chunk of the $[K, V]$ state matrix for one sequence and one value head, iterating sequentially over the $T$ tokens in that sequence.
