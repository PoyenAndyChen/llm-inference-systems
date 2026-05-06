# Long-Context Inference

**After reading this chapter, the reader will be able to:**

- Reason from RoPE's relative-position formulation through the lineage of context-extension recipes — Position Interpolation, NTK-aware base scaling, YaRN, LongRoPE, LongRoPE2 — and explain why each successive method is a *frequency-domain* rather than *position-domain* fix.
- Choose a sequence-parallel layout (Megatron-SP, DeepSpeed Ulysses, Ring Attention, Striped Attention, USP) for a given prefill workload and derive the bandwidth and memory tradeoffs that make one layout dominate another.
- Place the 2025–2026 long-context primitives — sliding-window-with-sinks, natively-trained sparse attention (NSA, DSA), query-aware sparse selection (Quest, MInference, DuoAttention), and hybrid Mamba-Transformer architectures — on a single decision map, distinguishing what the inference engine has to do differently for each.

By 2026 "long context" is no longer one problem. It has fragmented into four sub-disciplines the engine must coordinate at once: how positions are encoded across rotation periods the model never saw at training; how a single sequence's compute is distributed across many GPUs when no GPU's HBM holds the full activations; how attention's $O(N^{2})$ scaling is replaced by sub-quadratic patterns that still trade favorably against quality; and how the KV cache is organized when its byte count exceeds the model weights. Earlier chapters cover the *infrastructure* response to KV growth — paged allocation [see §10/02-paged-kv-memory](../10-engine-core/02-paged-kv-memory.md), tiered offload [see §30/02-kv-tiered-offload](../30-kv-cache/02-kv-tiered-offload.md), prefix caching [see §10/07-prompt-prefix-caching](../10-engine-core/07-prompt-prefix-caching.md). This chapter is about the *attention-layer* response: position encoding, sequence parallelism, structural sparsity, and architectural alternatives to dense softmax attention. The KV-shape side — MQA, GQA, MLA, hybrid Mamba caches — is taken up in [§30/03-attention-variants](../30-kv-cache/03-attention-variants.md) and cross-referred rather than repeated.

## 1. RoPE and the geometry of context extension

Every modern open-weights model uses RoPE — rotary position embeddings, introduced by [RoFormer](../papers.md#roformer) (Su et al., 2021). The constraint imposed by RoPE's relative-position formulation is what makes long-context extension a tractable engineering problem.

### 1.1 The rotary mechanism

RoPE encodes position by rotating each query and key vector in $d/2$ two-dimensional subspaces. Group the $d$-dimensional Q and K into pairs $(x_{2i}, x_{2i+1})$, treat each pair as a complex number $z_i = x_{2i} + j x_{2i+1}$, and multiply at position $m$ by a complex phase $e^{j m \theta_i}$ with frequencies

$$\theta_i \;=\; \mathrm{base}^{-2i/d}, \qquad \mathrm{base} \;=\; 10{,}000 \text{ (RoFormer default)}.$$

Equivalently, a $2{\times}2$ rotation:

$$R_{\theta_i, m} \;=\; \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \phantom{-}\cos m\theta_i \end{pmatrix}, \qquad q'_m \;=\; R_{\theta, m}\, q_m, \qquad k'_n \;=\; R_{\theta, n}\, k_n.$$

The defining property is that the inner product after rotation depends only on the *relative* displacement $m - n$, not on $m$ or $n$ separately:

$$\langle q'_m,\, k'_n \rangle \;=\; \langle q_m,\, R_{\theta, m-n}\, k_n \rangle.$$

The model only ever has to learn a function of $m-n$, and that is why RoPE generalizes more gracefully than absolute encodings. The cost is where context extension becomes hard. The frequencies $\theta_i$ span six orders of magnitude: low-index dimensions complete a full rotation in tens of tokens; high-index dimensions take tens of thousands of tokens to complete one. At training context $L_{\text{train}}$, the high-index dimensions have sometimes never completed a single rotation. They are *under-trained*.

### 1.2 Why naive extension fails

Suppose a model is trained at $L_{\text{train}}=4{,}096$ and served at $L_{\text{ext}}=128{,}000$. High-frequency dimensions have seen rotation angles $m \theta_0 \in [0, 4096]$ rad — many full periods, fully covered. Very-low-frequency dimensions have seen $m \theta_{d/2-1} \in [0, 0.4]$ rad — a tiny arc on the unit circle. At inference position $m=128{,}000$, the low-frequency phase is $12.8$ rad, well outside the training distribution; the model has never been shown a key whose relative phase exceeds the trained range. Attention scores in those subspaces blow up and perplexity diverges within a few hundred tokens of the extended region.

The four extension recipes that dominate production in 2026 — PI, NTK-aware, YaRN, LongRoPE / LongRoPE2 — all attack this *out-of-distribution rotation angle* problem. The lineage moves progressively from the position domain (rescale $m$) to the frequency domain (rescale $\theta_i$ per dimension), because the frequency domain is where the under-training actually lives.

### 1.3 Position Interpolation (PI)

[Chen et al., 2023](https://arxiv.org/abs/2306.15595) proposed the simplest fix: linearly compress the position index back into the trained range,

$$m \;\mapsto\; m' \;=\; m \cdot \frac{L_{\text{train}}}{L_{\text{ext}}}, \qquad q'_m \;=\; R_{\theta, m'}\, q_m.$$

Every rotation angle at inference now lies within $[0, L_{\text{train}}\theta_i]$, and a short fine-tune (≈1k steps) adapts the model to the compressed positions. PI made 32K-class windows on Llama-1 cheap. The cost is uniform compression: phase resolution between adjacent tokens drops by the same factor across all frequencies, so short-range patterns become harder to distinguish.

### 1.4 NTK-aware scaling

The community blog post known as *NTK-aware scaling* (bloc97, 2023) was the first frequency-domain fix. The insight: a uniform compression of $m$ corresponds to a uniform compression of all $\theta_i$, but high-frequency dimensions did not need help — they were trained on full-period coverage. Only low-frequency dimensions need rescaling, and they need it more aggressively. The mechanism is base-scaling:

$$\mathrm{base}' \;=\; \mathrm{base} \cdot s^{d/(d-2)}, \qquad \theta'_i \;=\; (\mathrm{base}')^{-2i/d},$$

where $s = L_{\text{ext}}/L_{\text{train}}$. The algebra ensures the highest-frequency dimension is barely affected while the lowest-frequency dimensions are compressed by approximately $s$. NTK-aware scaling required no fine-tuning to extend Llama-1 from 2K to 8K with substantially better quality than PI and remains a respectable baseline; it was contributed by an anonymous community member on Reddit before YaRN canonized it.

### 1.5 YaRN

[YaRN](../papers.md#yarn) (Peng et al., ICLR 2024) generalized NTK-aware into a principled, fine-tunable method with two pieces. First, *NTK-by-parts* partitions frequencies into three regions by how many full rotations they complete over $L_{\text{train}}$: high-frequency dimensions are left untouched, very-low-frequency dimensions are linearly interpolated like PI, and mid-frequency dimensions ramp smoothly between, parameterized by cutoffs $\alpha$ and $\beta$. Second, *attention temperature scaling*: extending context softens the attention distribution because more tokens compete for the same softmax mass, so YaRN scales the attention logits by

$$\sqrt{\log L_{\text{ext}} \,/\, \log L_{\text{train}}},$$

preserving distribution shape at extended length. YaRN is the SOTA RoPE extension recipe through 2024 and the basis of Llama 3.1's 128K extension.

### 1.6 LongRoPE and LongRoPE2

[LongRoPE](../papers.md#longrope) (Ding et al., ICML 2024) replaces the closed-form NTK-by-parts schedule with an *evolutionary search* over per-dimension rescaling factors. Each $\theta_i$'s rescaling factor is a variable, and the search minimizes perplexity on a held-out long-context corpus. Different models prefer different schedules; LongRoPE pushes context extension to 2M tokens and is the basis of Phi-3-128K.

[LongRoPE2](../papers.md#longrope2) (Microsoft, 2025-02) extends the method with an improved training strategy. Its central observation is that high RoPE dimensions are systematically *under-trained*: at $L_{\text{train}}=4{,}096$ those dimensions barely see a full rotation, so the model has no calibrated representation of their phase regardless of how well-chosen the rescaling schedule is. LongRoPE2's mixed-context training exercises all RoPE dimensions across the rotation circle during continued pretraining, producing 128K context with an order of magnitude less data than Meta's Llama 3.1 recipe (≈10B tokens vs ≈800B), with negligible short-context regression. Phi-4 and several Microsoft 2026 models use LongRoPE2.

### 1.7 What the engine does at load time

For the inference engine, the practical consequence is a small set of `rope_scaling` configuration fields the model checkpoint carries. vLLM's `model_runner` reads `rope_scaling.type` ∈ {`linear`, `dynamic` (NTK-aware), `yarn`, `longrope`}, constructs the rotated cos/sin tables once at load time, and applies them inside the fused RoPE kernel. SGLang and TRT-LLM follow the same pattern. The runtime cost of every scheme is identical — one elementwise rotation per Q/K head per token; training-time cost differs by orders of magnitude and dictates which recipe a model author picks.

## 2. Sequence parallelism for prefill

A single GPU's HBM cannot hold the activations of a long prompt at high enough batch to amortize weight reads, and by 1M tokens neither the activations nor the working KV fit on one GPU at all. The response is *sequence parallelism*: split the sequence axis across devices and have them cooperate to compute the attention that requires every position to see every other.

Four canonical layouts have emerged. They differ in *what* is sharded (sequence or heads), *when* they communicate (per-layer or per-attention), and *how* communication is structured (all-to-all, ring, or all-reduce). USP unifies them as special cases of a 2D layout.

### 2.1 Megatron-SP

Megatron-LM's sequence parallelism ([Korthikanti et al., 2022](https://arxiv.org/abs/2205.05198)) is a companion to tensor parallelism, not a standalone layout. It targets the *non-attention* sub-layers — LayerNorm, dropout, residuals — whose activations TP would otherwise duplicate across all TP ranks. Megatron-SP shards the sequence axis of those activations: each TP rank holds $1/P_{\text{TP}}$ of the sequence between attention blocks, with an `all-gather` before the column-parallel projection that opens each attention or FFN and a `reduce-scatter` after the row-parallel projection that closes it. Communication volume matches plain TP, but the activation memory footprint is reduced by $P_{\text{TP}}$ — the part that makes long contexts trainable and prefillable. Megatron-SP is the baseline Ulysses and Ring extend.

### 2.2 DeepSpeed Ulysses

[DeepSpeed Ulysses](../papers.md#ds-ulysses) (Microsoft, 2023-09) separates sequence parallelism from TP and shards the sequence axis end-to-end. Each Ulysses rank holds *all* parameters of all layers but only $1/P_{\text{U}}$ of the sequence. The challenge is the attention block: attention requires every Q to attend to every K across the full sequence, but each rank holds only a sequence shard of Q, K, V. Ulysses solves this with a pair of `all-to-all` collectives that *transpose* the data layout on entry and exit:

```
before attention:   rank r holds Q[seq_r,    head_*]
                                K[seq_r,    head_*]
                                V[seq_r,    head_*]
all-to-all:         rank r holds Q[seq_*,    head_r]
                                K[seq_*,    head_r]
                                V[seq_*,    head_r]
local attention on full sequence, head_r heads
all-to-all:         rank r holds out[seq_r,  head_*]
```

After the first all-to-all, each rank owns a subset of attention heads but the entire sequence for those heads, and computes a local FlashAttention call. The second all-to-all reverses the layout. Communication cost is two all-to-alls per attention block; volume per rank per call is $O(B \cdot S \cdot d / P_{\text{U}})$, scaling independently of $P_{\text{U}}$ for a fixed model — the constant-volume property that motivated the design.

The hard limit is head count: Ulysses cannot scale beyond $H$ ranks because each rank must own at least one full head after the all-to-all. GQA models with $H_{\text{kv}}=8$ are typically capped at $P_{\text{U}} \le 8$.

### 2.3 Ring Attention

[Ring Attention](../papers.md#ring-attn) (Liu, Zaharia, Abbeel, ICLR 2024) takes the opposite approach: each rank holds a contiguous shard of the sequence and *all* heads, with the attention computation structured as a ring around which K and V blocks rotate. At step $t$ of an $N$-rank ring, rank $r$ holds Q for its sequence shard and the K, V block currently in transit; each rank computes a partial output against the local K, V block, then forwards it to the next rank while receiving the next block from the previous one. After $N-1$ steps every Q has been attended against every K, V block, and partial outputs are reduced via the same online-softmax recurrence FlashAttention uses internally.

Compute and communication overlap. Per ring step, each rank performs a $B {\times} B$ attention compute of $T_c \approx 4 B^{2} d / P$ FLOPs against a communication of $T_{\text{comm}} \approx 2 B d / \beta$ bytes, where $\beta$ is the link bandwidth. Compute hides communication when

$$\frac{4 B^{2} d / P}{F_{\text{peak}}} \;>\; \frac{2 B d}{\beta} \;\;\Longleftrightarrow\;\; B \;>\; \frac{P \cdot F_{\text{peak}}}{2 \beta}.$$

For B200 ($F_{\text{peak}} \approx 4.5$ PFLOPs BF16) on NVLink5 ($\beta \approx 1.8$ TB/s), $B \gtrsim 1{,}250 \cdot P$ tokens per rank — so a 16-rank ring needs $\sim$320K tokens before Ring is truly compute-bound. Below that crossover, Ring spends measurable time waiting on P2P. The HBM benefit is unconditional, however: each rank's working KV memory is $1/N$ of the full sequence's, which is what makes 1M-token prefill possible at all.

### 2.4 Striped Attention

Ring Attention has a load-imbalance problem under causal masking. With contiguous sharding, rank $r$'s Q tokens are positions $[r S/N, (r+1) S/N)$. When the rotating K, V block is from a *later* rank, every attention score is zero by the causal mask — the local rank wastes compute on a fully masked block. The earliest rank gets the most work; the latest does almost none.

[Striped Attention](../papers.md#striped-attn) (Brandon et al., 2023) fixes this with a strided permutation. Rank $r$ holds positions $\{r, r+N, r+2N, \ldots\}$ — a stride of $N$. Every rank's Q tokens are now uniformly spread across the full sequence, and every rotating K, V block has roughly half its entries on the live side of the causal mask. Reported speedup is $1.45$–$1.65{\times}$ over Ring at 786K on TPUv4. The cost is a one-time permutation at sequence ingress and its inverse at egress.

### 2.5 Unified Sequence Parallelism (USP)

[USP](../papers.md#usp) (Fang & Zhao, 2024-05) recognizes Ulysses and Ring as two axes of a 2D layout rather than competitors. The outer layout is Ulysses (all-to-all on heads); the inner layout is Ring (P2P on KV blocks). Total parallelism is $P = P_{\text{U}} \cdot P_{\text{R}}$, and the operator picks the split: Ulysses dominates when head count is large and per-rank compute cheap; Ring dominates when sequence length is large and inter-node bandwidth constrained. The repo `feifeibear/long-context-attention` is the de-facto sequence-parallel layer vLLM, SGLang, and TRT-LLM vendor or reimplement.

### 2.6 Tradeoff summary

| Layout | Memory per GPU | Comm pattern | Comm rounds | Key constraint |
|---|---|---|---|---|
| Megatron-SP | $O(S \cdot d / P_{\text{TP}})$ activations only | all-gather + reduce-scatter | $O(L)$ per fwd | TP-companion, no attention scaling |
| Ulysses | $O(S \cdot d)$ params, $O(S \cdot d / P_{\text{U}})$ acts | all-to-all on heads | 2 per attention | $P_{\text{U}} \le H_{\text{kv}}$ |
| Ring | $O(d^{2})$ params, $O(S \cdot d / P_{\text{R}})$ acts | P2P ring | $P_{\text{R}}-1$ per attention | compute hides comm only at large $S$ |
| Striped | same as Ring | P2P ring | $P_{\text{R}}-1$ per attention | same as Ring with balanced causal |
| USP | composition | all-to-all $\times$ ring | varies | tune $P_{\text{U}} {\cdot} P_{\text{R}}$ |

Engines apply these patterns at *prefill*, where the sequence is long and fully present at the start of the request. Decode is a different regime — one new token per step — where these layouts collapse to trivial communication and the question becomes how the KV cache is sharded ([see §20/01-parallelism-strategies](./01-parallelism-strategies.md)). SGLang's chunked pipeline parallelism (early 2026) carves prefill into chunks pipelined across stages, composing with sequence parallelism to keep $\geq$ 900K-token prefill throughput within $\sim$10% of short-prompt baselines on B200/H200.

## 3. Sliding window and attention sinks

The complementary response is to give up on full attention in some layers. The first wave (Longformer, BigBird) used static sparse patterns and is essentially obsolete for serving in 2026. The current generation uses *sliding-window attention* (SWA) restricted to a fixed local window plus a small set of always-attended tokens.

Mistral 7B (2023) was the first widely deployed model with SWA: every layer's attention is restricted to the most recent $W=4{,}096$ tokens. The KV cache is bounded at $W$ tokens regardless of context length. Gemma 3 alternates SWA and full-attention layers in a 5:1 ratio; GPT-OSS uses 1:1. Benefits are immediate — bounded KV per layer, $O(W)$ instead of $O(S)$ attention compute — and the cost is hard: SWA layers cannot retrieve information beyond $W$ tokens. Interleaved full-attention layers carry the long-range information; SWA layers handle local refinement.

[StreamingLLM](../papers.md#streamingllm) (Xiao et al., ICLR 2024) supplied the missing piece. Pure sliding-window attention fails on the first sliding step: the model has been trained to deposit a small amount of attention mass on the *initial* tokens of a sequence, and dropping them as a naive sliding window does corrupts the residual stream. The fix is to keep a small set of *attention sinks* (typically the first 4 tokens) permanently in cache and slide the window after the sink:

$$\text{kept}(i) \;=\; \mathbb{1}[i \le k_{\text{sink}}] \;\lor\; \mathbb{1}[i > t - W],$$

with $k_{\text{sink}} \approx 4$. The cache is bounded at $k_{\text{sink}} + W$ tokens, and generation over arbitrary length proceeds without divergence. GPT-OSS adds a *learned sink bias* — an additional bias vector added to the attention logits at sink positions — that lets the model adapt the sink shape during pretraining; the inference consequence is a kernel modification (FA-3 added a sink-bias path in 2025). The compression-side discussion of attention sinks — H2O, SnapKV, and the modern eviction lineage — lives in [§30/01-kv-compression](../30-kv-cache/01-kv-compression.md).

## 4. Native sparse attention

Static sparse patterns failed because the attention pattern that maximizes quality is *dynamic and content-dependent*. The 2024 generation rediscovered this with *learned sparsity* — patterns selected per-query at runtime. The 2025 generation went one step further by training the model from scratch on the sparse primitive, so the model's representations co-evolve with the sparse attention rather than being post-hoc compressed.

[NSA](../papers.md#nsa) (DeepSeek, 2025-02) — *Native Sparse Attention* — is the lineage paper. It proposes a hardware-aligned, natively trainable sparse attention with three parallel branches whose outputs are summed:

1. *Compressed coarse-grained attention*. Pooled summary tokens — one per coarse block of $B_c$ positions — carry a low-resolution view of the entire sequence. Every query attends to all summary tokens: $O(S/B_c)$ work per query, providing global recall.
2. *Selected fine-grained attention*. The query, scored against summary tokens, selects the top-$K$ blocks at coarse resolution and attends to them at full resolution: $O(K B_c)$ work per query.
3. *Sliding-window attention*. A standard local window of $W$ tokens: $O(W)$ work per query.

Total work is $O(S/B_c + K B_c + W)$ per query, sub-quadratic in $S$. The training signal flows through all three branches so the model learns when each one carries the relevant information. NSA reports up to $11.6{\times}$ decode speedup at 64K context with quality matching dense attention on RULER and LongBench v2.

[DSA](../papers.md#dsa-v32) — *DeepSeek Sparse Attention*, in DeepSeek-V3.2 (2025-09 → 2025-12) — productizes the NSA idea in a frontier-scale model. The mechanism is a *lightning indexer* that scores every cached KV position against the current query at low cost, followed by top-2048 selection and full-attention computation only over the selected positions. Day-zero kernel support shipped in vLLM and SGLang.

The hedge required by the evidence base, restated for emphasis: DeepSeek-V3.2 is the first frontier-scale model to ship a natively-trained sparse-attention primitive in production; downstream-quality parity vs. V3.1 is reported by DeepSeek and not yet independently confirmed at long-context benchmarks beyond their evals. The result suggests the long-running theoretical claim — that learned sparsity at training time is finally tractable at frontier scale — is approaching reality, but it is not yet a settled empirical fact that DSA preserves all relevant capabilities at all context lengths.

## 5. Long-context KV management at runtime

For models that did not bake sparsity into pretraining, the engine still has to manage a multi-hundred-thousand-token KV cache without paying the dense-attention bill at every decode step. Three runtime techniques carry the bulk of production deployments. They overlap with the KV-compression chapter [see §30/01-kv-compression](../30-kv-cache/01-kv-compression.md); the long-context cut emphasizes which methods survive at 128K → 1M, where uniform-budget eviction collapses but query-aware and layer-pyramidal methods hold up.

### 5.1 Quest: query-aware page selection

[Quest](../papers.md#quest) (Tang et al., ICML 2024) frames the problem as page selection rather than eviction. The KV is left intact in HBM, but only the pages most likely to contribute to the current query's attention output are loaded into the kernel. Each KV page maintains a per-channel min and max of its keys, and the per-page criticality for query $q$ is the upper bound

$$c_p \;=\; \max\!\bigl(\langle q, k_p^{\max}\rangle,\; \langle q, k_p^{\min}\rangle\bigr),$$

an admissible heuristic on the largest attention score the page could produce. The engine ranks pages by $c_p$, loads the top-$K$, and attends only over them. Quest reports $7.03{\times}$ decode latency reduction with negligible quality loss on long-context QA at 128K, and composes with KV offload — low-criticality pages live on CPU and stream in only when their bound clears the threshold.

### 5.2 MInference: head-pattern templates

[MInference](../papers.md#minference) (Microsoft, NeurIPS 2024 Spotlight) targets *prefill* and observes that long-context attention patterns fall into three structural templates each head obeys consistently across inputs:

- **A-shape** ("anchor + recent"): heads attending to the first few tokens and the last $W$ recent ones.
- **Vertical-Slash**: heads attending to a small set of vertical columns plus a diagonal slash of recent tokens.
- **Block-Sparse**: heads attending in clustered blocks identified at runtime by a cheap pattern classifier.

The pattern is determined offline per head per layer. At prefill, each head runs a sparse kernel specialized to its template — a $10{\times}$ speedup at 1M-token prefill on A100. The decision is not "sparse vs. dense" but "which sparse template per head" — a pattern NSA generalizes by training the pattern selector jointly with the model.

### 5.3 DuoAttention: retrieval vs. streaming heads

[DuoAttention](../papers.md#duoattention) (Xiao et al., MIT/NVIDIA, ICLR 2025) partitions attention heads into two roles. *Retrieval heads* genuinely need the full long-context KV and are kept dense. *Streaming heads* — the majority — can be served with only a constant-length sink + recent buffer with negligible quality loss. The partition is *learned* via a short fine-tuning pass with a sparsity-inducing objective.

DuoAttention enables 3.3M-token Llama-3-8B serving on a single A100 with 4-bit KV quantization — a setting pure-eviction methods cannot reach because they uniformly compress all heads. The head-specialization lineage (FastGen → Ada-KV → DuoAttention) is developed in [§30/01-kv-compression](../30-kv-cache/01-kv-compression.md).

## 6. Hybrid Mamba-Transformer architectures

The most aggressive attack on long-context cost replaces the attention layer entirely in some fraction of the model. State-space models — Mamba, Mamba-2, and the Mamba-3 generation reaching ICLR 2026 — process tokens through a *recurrent state* of fixed size, $O(1)$ memory and $O(N)$ time per sequence regardless of context length. Pure SSMs trail dense Transformers on fine-grained recall (the "selective copy" failure mode), so the production pattern is *hybrid*: a few full-attention layers carry recall, a majority of SSM layers carry the bulk.

Three hybrid families are deployed in production.

- **Jamba-1.5** (AI21, 2024-08). 1:7 attention:Mamba interleave plus MoE; 256K context with $\sim$4 GB of attention KV plus fixed-size SSM state; reports $\sim$3$\times$ throughput vs Llama-2-70B at 256K.
- **Falcon-H1** (TII, 2025-07). Parallel attention + Mamba heads in the same layer with concatenated outputs; 0.5B–34B; 256K context; the 34B model matches Qwen2.5-72B and Llama-3.3-70B.
- **Granite 4** (IBM, 2025-10). 9:1 Mamba-2 : attention ratio; reports >70% production-RAM reduction on long-context multi-session enterprise workloads — the most aggressive ratio shipped at scale.

[Mamba-3](../papers.md#mamba-3) (ICLR 2026 oral, arXiv:2603.15569) is the latest canonical SSM reference. Its contribution is improved hardware utilization through restructured state-update kernels and a tile-friendly recurrence; the fixed-state-size shape is unchanged. The optimal attention:SSM ratio remains unsettled — Granite's 9:1, Jamba's 1:7, and Falcon-H1's parallel placement each work at scale. Cross-architecture KV-shape consequences are developed in [§30/03-attention-variants](../30-kv-cache/03-attention-variants.md).

The serving implication is *heterogeneous cache*. A pure-attention engine assumes one cache shape: $L$ layers each holding a variable-length $[\text{seq}, H_{\text{kv}}, d_h]$ KV tensor. A hybrid engine has two: variable-length KV for attention layers and a fixed-size $[\text{state}, d_{\text{state}}]$ tensor per Mamba layer, the latter never growing. The Mamba state is typically pre-allocated separately because its size is request-independent — vLLM and SGLang's hybrid-model paths allocate Mamba states from a separate pool to keep PagedAttention's block-based allocator from fragmenting around fixed-size tensors. Mamba-2 and Mamba-3 layers run dedicated chunk-state recurrence kernels rather than the FlashAttention path, and the batching scheduler must coordinate the two kernel families per layer.

## 7. Evaluation and the long-context benchmark gap

Long-context serving cannot be evaluated with short-context recipes. Three benchmarks have become canonical. *Needle-in-a-Haystack* (NIAH, Kamradt 2023) hides a fact in a long context and queries the model for it; *RULER* (NVIDIA, COLM 2024) generalizes NIAH into 13 task types across 4 categories and is the de-facto OSS benchmark; *LongBench v2* (Tsinghua, 2024-12) targets 8K–2M with 503 multiple-choice tasks across six categories. *SCBench* (Microsoft/HKUST, ICLR 2025) reframes the evaluation around the *KV cache lifecycle* — multi-turn, shared-prefix, reuse-across-requests — and finds several sub-$O(n)$ memory methods that look fine on single-shot recall break in multi-turn use.

The gap between recall and reasoning is now well understood. A model that retrieves a needle at 1M is not necessarily one that can synthesize across 100K tokens of reasoning input. Frontier APIs (Claude Opus 4.6 at 1M GA, Gemini 1.5/2.5 Pro at 1–2M) report internal multi-task long-context evals; open-weights models below frontier scale routinely show steep quality drops past 128K–256K on RULER and LongBench v2 even when their advertised context window is much larger. The community consensus is that 1M-class quality is achievable; advertised 10M numbers are not yet supported by independent evaluation.

## Current production state

Across the production engines that dominate 2026 serving, long-context support is now table stakes up to 128K, plausible up to 1M for select models, and exceptional beyond that. Position-extension recipes are picked by the model author and delivered as `rope_scaling` configuration: Llama 3.1+ ships YaRN; Phi-3 and Phi-4 ship LongRoPE / LongRoPE2; Qwen2.5-1M and most other open frontier models ship YaRN with model-specific tuning. NTK-aware scaling remains the no-fine-tune fallback for community extensions but is no longer the production path of choice for natively long models. Engines apply RoPE in the fused kernel and treat the rescaling schedule as load-time configuration; the hot path is identical across schemes.

Sequence parallelism for prefill is a deployed engine feature rather than a research demonstration. vLLM, SGLang, and TRT-LLM each ship a USP-style 2D sequence-parallel layout on top of their tensor-parallel implementations; SGLang's chunked pipeline parallelism (early 2026) extends the design to keep $\geq$ 900K-token prefill throughput within ~10% of short-prompt baselines on B200/H200. Sliding-window attention with sinks is the kernel default for Mistral, Gemma 3, and GPT-OSS layers and ships as a first-class FA-3 / FA-4 path; engines maintain a hybrid KV allocator that bounds SWA-layer caches at the window size while letting full-attention-layer caches grow. Quest, MInference, and DuoAttention have landed in some production configurations but are not yet default-on across engines — they require model-specific calibration and an extra runtime path not every backend carries. DeepSeek-V3.2's DSA is the most prominent production deployment of natively-trained sparse attention; day-zero DSA kernels in SGLang and vLLM make it operationally visible at scale, but quality parity vs. V3.1 rests on DeepSeek's evaluation and is not yet independently confirmed at frontier long-context benchmarks. Hybrid Mamba-Transformer models — Jamba-1.5, Falcon-H1, Granite 4, the Mamba-3 generation — are served by all major engines with separate state pools for SSM layers; the optimal attention-to-SSM ratio is unsettled and heterogeneous caches are now a routine engine concern rather than a research curiosity.

The unresolved question across the field is end-to-end production serving above 1M tokens in OSS engines. Frontier APIs do it. Open-weights checkpoints nominally reach 10M (Llama 4 Scout). No widely deployed open-source serving stack routinely produces frontier-quality output at $>$2M, and the missing primitives are a combination of cluster-level KV migration, sparse-attention integration with prefix caching, and benchmarks that can actually distinguish a working 10M model from a broken one. That gap is where the next 12 months of long-context inference research lands.
