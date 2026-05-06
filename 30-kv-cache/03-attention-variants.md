# Attention Variants as KV-Shape Problems

**After reading this chapter, the reader will be able to:**

- Read every contemporary attention variant — MQA, GQA, MLA, sliding-window, sink, YOCO, gated linear attention, Mamba, and its hybrids — as a design move that reshapes the KV cache, and trace the inference-time consequences (per-token bytes, decode bandwidth, kernel structure) from that shape.
- Derive the MLA absorption identity that lets DeepSeek-V3 store only the latent vector $c_{KV}$ at decode time rather than the full $K$ and $V$, and quantify its footprint against MHA, GQA, and MQA.
- Place hybrid Mamba-Transformer architectures and diffusion language models on the same map: each breaks a different assumption built into autoregressive serving engines, and each forces a distinct engine adaptation.

The previous chapters in this section treated the KV cache as a fixed object to be compressed ([see §30/01-kv-compression](01-kv-compression.md)) or moved across tiers ([see §30/02-kv-tiered-offload](02-kv-tiered-offload.md)). This chapter takes the orthogonal view: every attention variant from the past three years can be read as a change to the *shape* of the cache. Some shrink the per-head footprint (GQA, MQA). One collapses K and V into a shared latent (MLA). Some bound the cache at a fixed window (sliding window, sinks). Some replace the cache with a single global state (YOCO) or a fixed-size recurrent state (Mamba, GLA). One — diffusion language models — abolishes it entirely.

The unifying claim is that architecture and engine are codesigned at the cache. A model that halves the KV footprint per token does not just save memory; it moves the operating point along the decode roofline of [see §00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md), it changes the kernel shape, and it changes which compression and prefix-caching abstractions still apply. The chapter walks the variants in order of how aggressively they reshape the cache, ending with the architectures that no longer have one.

## 1. The KV-shape framing

The canonical bytes-per-token-per-layer formula derived in [see §00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md) is

$$\text{bytes}_{\text{KV/token/layer}} \;=\; 2 \cdot b \cdot H_{\text{kv}} \cdot d_h$$

with $H_{\text{kv}}$ KV heads, $d_h$ head dimension, $b$ bytes per element. Across $L$ layers and $S$ tokens, a single request's cache is $2 b L H_{\text{kv}} d_h S$ bytes; long-context decode bandwidth pulls that whole object across HBM once per generated token. Any shape change that reduces $H_{\text{kv}} d_h$, decouples $K$ from $V$, replaces the expression with a constant in $S$, or eliminates it entirely shows up directly as a decode-time bandwidth win.

The variants cluster into four families:

1. **Same shape, fewer heads.** MHA → MQA → GQA reduce $H_{\text{kv}}$ while leaving the per-head tensors intact. Reduction is exactly $H_{\text{kv}}/H$.
2. **Different shape, latent.** MLA replaces stored $K, V$ with a single shared latent of dimension $d_c \ll d$, plus a small RoPE-decoupled key.
3. **Bounded shape.** Sliding window and attention sinks keep the per-token shape but cap the length at $W$ tokens.
4. **Different state object.** YOCO caches once globally; GLA / GTA and SSMs (Mamba, Mamba-2, Mamba-3) replace the cache with a fixed-size recurrent state; diffusion language models replace it with a re-processed token sequence.

The summary footprint table is the chapter's central artifact:

| Variant | KV per token per layer | Example (70B-class, $H{=}64$, $d_h{=}128$, FP16) | 128k-ctx, 80 layers |
|---|---|---|---|
| MHA | $2 b H d_h$ | 32 KB | $\approx 336$ GB |
| GQA, $G{=}8$ | $2 b G d_h$ | 4 KB | $\approx 42$ GB |
| MQA | $2 b \cdot 1 \cdot d_h$ | 0.5 KB | $\approx 5$ GB |
| MLA, $d_c{=}512$, $d_h^R{=}64$ | $b(d_c + d_h^R)$ | 1.15 KB | $\approx 9$ GB (61 layers) |
| Sliding-window, $W{=}4096$ | unchanged, capped at $W$ | 4 KB/token × 4096 = 16 MB/layer | bounded, $\approx 1.3$ GB |
| YOCO | global $K,V$, not per-layer | $L\times$ smaller than baseline | $\approx 0.5$ GB |
| Mamba SSM state | $b \cdot d \cdot N$, fixed in $S$ | $\sim 64$ KB/layer constant | $\approx 4$ MB |
| Diffusion LM | no KV cache | n/a | n/a |

## 2. MHA, MQA, GQA — same shape, fewer heads

**MHA** is the baseline: each of $H$ query heads has its own $K$ and $V$ head. For 70B-class $H{=}64, d_h{=}128$, MHA at FP16 is 32 KB/token/layer; at 128k context across 80 layers that is ~336 GB for a single request — four times an H100's HBM. MHA is no longer used for new long-context decoders above the 1B scale.

**MQA** ([Shazeer 2019](https://arxiv.org/abs/1911.02150)) keeps one $K$ and one $V$ head, broadcast against all $H$ query heads. Cache drops to $2 b d_h$ — an $H\times$ reduction. MQA was first deployed at scale in PaLM and proved one shared KV head is functional at foundation-model scale, but consistently underperforms MHA on quality-sensitive tasks at smaller scales. MQA is rare in 2024–2026 frontier models.

**GQA** ([Ainslie et al. 2023](https://arxiv.org/abs/2305.13245)) is the production middle ground. Queries are partitioned into $G$ groups of $H/G$ each, and a single $K, V$ pair is shared inside each group:

$$Q^{(g, j)}, \quad K^{(g)}, V^{(g)}, \quad g \in \{1,\ldots,G\}, \; j \in \{1,\ldots,H/G\}.$$

$G{=}H$ recovers MHA; $G{=}1$ recovers MQA. Per-token-per-layer cache is $2 b G d_h$. With short uptraining from MHA checkpoints, quality matches MHA within fractions of a perplexity point at $G{=}8$, while cache memory drops to $G/H$ of MHA.

GQA is the production default for new dense models in 2024–2026: Llama-2 70B and the entire Llama-3 family ($G{=}8$ across 8B, 70B, 405B), Mistral 7B v0.3, Qwen2 / Qwen2.5 / Qwen3, Gemma 2/3, GPT-OSS, and most non-MLA MoE models. The dominant choice has been $G{=}8$ at $d_h{=}128$, yielding the 4 KB/token/layer figure that recurs throughout this book. The pragmatic argument for GQA over MQA: the additional 7 KV heads cost very little memory (4 KB vs 0.5 KB at $H{=}64$) but recover essentially all of MHA's quality on long-context retrieval, where MQA degrades.

Kernel implications are minor: a GQA kernel is structurally an MHA kernel that broadcasts $K, V$ across $H/G$ query heads inside the inner loop. FlashAttention 2 and 3 handle GQA natively; cross-reference [see §10/01-attention-kernels](../10-engine-core/01-attention-kernels.md).

## 3. MLA: latent-space attention

DeepSeek-V2's Multi-head Latent Attention ([MLA-V2](../papers.md#mla-v2)) and its V3 deployment take a structurally different path. Instead of reducing $H_{\text{kv}}$, MLA projects $K$ and $V$ into a low-dimensional shared latent and stores only the latent. The mathematical move and its inference-time *absorption identity* are the central content of this section.

### 3.1 The forward pass

For input hidden state $h_t \in \mathbb{R}^d$, MLA computes a single shared latent

$$c_{KV,t} \;=\; W^{DKV} h_t, \qquad c_{KV,t} \in \mathbb{R}^{d_c}, \quad d_c \ll d,$$

and per-head $K, V$ are upprojected from it:

$$K^{(h)}_t = c_{KV,t} \, W^{UK,(h)}, \qquad V^{(h)}_t = c_{KV,t} \, W^{UV,(h)}, \qquad W^{UK,(h)}, W^{UV,(h)} \in \mathbb{R}^{d_c \times d_h}.$$

A separate decoupled-RoPE branch produces a per-head RoPE-rotated key $k_t^{R,(h)} \in \mathbb{R}^{d_h^R}$, because applying RoPE inside the latent space breaks the rotation's relative-position property. The query splits analogously into a content part $Q_t^{C,(h)}$ and a RoPE part $Q_t^{R,(h)}$, with score

$$s^{(h)}_{t,t'} \;=\; \frac{1}{\sqrt{d_h + d_h^{R}}} \Big( \langle Q_t^{C,(h)}, K_{t'}^{(h)} \rangle + \langle Q_t^{R,(h)}, k_{t'}^{R,(h)} \rangle \Big).$$

So far this is just a low-rank factorization. The savings come at inference time.

### 3.2 The absorption identity

Naively, decode would store $c_{KV,t}$ and then expand to per-head $K^{(h)}_t = c_{KV,t} W^{UK,(h)}$ before each query — paying both the latent-storage and the materialization compute. The absorption observation is:

$$\langle Q_t^{C,(h)}, K_{t'}^{(h)} \rangle \;=\; \langle Q_t^{C,(h)}, c_{KV,t'} W^{UK,(h)} \rangle \;=\; \langle Q_t^{C,(h)} (W^{UK,(h)})^{\top}, c_{KV,t'} \rangle.$$

The matrix $W^{UK,(h)}$ is *absorbed into the query side*: at each decode step, precompute

$$\tilde Q_t^{(h)} \;=\; Q_t^{C,(h)} (W^{UK,(h)})^{\top} \;\in\; \mathbb{R}^{d_c}$$

once per query position, and the score against any cached latent becomes a $d_c$-dimensional dot product against the *stored latent*, not a $d_h$-dimensional dot product against a materialized key. The same absorption applies on the value side, where $W^{UV,(h)}$ is folded into the output projection $W^{O,(h)}$:

$$o^{(h)} \;=\; \big( \sum_{t'} a_{t,t'} c_{KV,t'} W^{UV,(h)} \big) W^{O,(h)} \;=\; \big( \sum_{t'} a_{t,t'} c_{KV,t'} \big) \big( W^{UV,(h)} W^{O,(h)} \big),$$

with the bracketed product fused once at load time.

The net of absorption: at decode time, **only $c_{KV,t}$ (dimension $d_c$) is stored per token per layer, and the per-step attention reads only that latent.** The expanded $K$ and $V$ are never materialized for cached positions.

### 3.3 Arithmetic and adoption

For DeepSeek-V3, $d_c = 512$, $d_h^{R} = 64$, $L = 61$. Per-token-per-layer cache at FP16 is $2 \cdot 576 = 1152$ bytes. A like-for-like dense MHA with $H{=}128, d_h{=}128$ would cost $2 \cdot 2 \cdot 128 \cdot 128 = 65{,}536$ bytes per token per layer — a $\approx 57\times$ ratio. (DeepSeek-V3's own report quotes $\approx 32\times$ against a different baseline; the absolute 1.15 KB/token/layer is the more useful planning number.) Against an $H_{\text{kv}}{=}8$ GQA baseline at the same $d$, MLA's cache is $\approx 28\%$. At 128k context across 61 layers, a single request fits in $\approx 9$ GB versus $\approx 32$ GB for the equivalent GQA design. This is the structural reason DeepSeek-V3 serves long-context queries on commodity 8× H100 nodes at batch sizes competitively-priced GQA alternatives cannot match.

Absorption is not free: $W^{UK,(h)}$ is applied to every query at every step, an extra $H \cdot d_h \cdot d_c \cdot 2$ FLOPs per token per layer. For DeepSeek-V3 this is on the order of a few percent of the per-token FLOP budget; on Hopper/Blackwell with FP8 tensor cores it runs at the same precision as the rest of attention. At the decode-bound batch sizes of typical production, the memory-versus-compute trade is overwhelmingly favorable.

**Adoption is hedged.** MLA is confirmed in production at scale only in DeepSeek's own line (V2, V3, V3.1, V3.2, R1, V3.2-Speciale) and Moonshot AI's Kimi K2 family. Llama-3 / Llama-4, Qwen2.5 / Qwen3, Mistral, Gemma 3, GPT-OSS — none of the other major 2024–2026 frontier model families adopted MLA. Reasons commonly cited: it requires training-from-scratch or substantial uptraining; engine-side kernel support (FlashMLA, ThunderMLA — see [§10/01-attention-kernels](../10-engine-core/01-attention-kernels.md)) lagged GQA kernels by 6–12 months; and GQA at $G{=}8$ already gives ~8× cache reduction at zero engineering cost. MLA is a compelling architectural answer to long-context KV cost, but is not yet a settled industry default.

## 4. Sliding window and attention sinks

### 4.1 Sliding window

A sliding-window-attention (SWA) layer attends only to the last $W$ tokens. Per-token-per-layer cache is unchanged, but per-request cache is bounded by $\min(S, W)$. For Mistral 7B v0.1/v0.2 with $W = 4096$, the cache stops growing past 4096 tokens, capping a single request at $\approx 1$ GB. The trade is sharp: anything beyond $W$ is unrecoverable except through implicit propagation up the layer stack. Pure-SWA models cannot retrieve specific information from beyond their window.

The lineage:

- **Mistral 7B v0.1/v0.2 (2023):** every layer SWA, $W{=}4096$.
- **Mistral 7B v0.3 (2024):** dropped SWA in favor of full GQA, reflecting the shift toward true long-context support.
- **Gemma 3 (2024):** *interleaved* — 5 SWA layers ($W{=}4096$) per 1 global-attention layer. Most positions in most layers attend locally; a few global layers preserve long-range dependency.
- **GPT-OSS (2024–2025):** half the layers SWA, half full attention, with a learned attention-sink bias.
- **Qwen3-Next (2025):** also uses interleaved SWA.

Production engines provide separate allocators for SWA layers: TRT-LLM's circular-buffer KV, vLLM's SWA-aware allocator, SGLang's SWA RadixCache. A hybrid model ends up with two cache shapes coexisting in one engine, a precursor to the heterogeneous-cache problem of §6.

### 4.2 Attention sinks

The "attention sink" phenomenon ([StreamingLLM](https://arxiv.org/abs/2309.17453), Xiao et al., MIT/Meta/CMU/NVIDIA, ICLR 2024) is the empirical observation that a small number of initial tokens consistently absorb a large fraction of attention mass across heads and layers, even when those tokens are semantically uninformative. Softmax attention is non-zero everywhere, and the network learns to use a stable sink to absorb attention mass that has nowhere else to go.

A sliding-window cache that *evicts* these sinks suffers a quality cliff. The fix is the StreamingLLM recipe: keep a few initial tokens as permanent sinks plus the last $W$ tokens. With this small change, SWA generation runs stably to millions of tokens. The recipe was integrated into HuggingFace Transformers and TRT-LLM. A subsequent refinement is *learned sink bias* (GPT-OSS), training a learnable bias that plays the sink role at attention time. The eviction-as-design-knob view of sinks turns out to be a useful precursor for the KV-compression chapter — heuristic eviction methods all implement a sink-plus-something-else rule [see §30/01-kv-compression](01-kv-compression.md).

### 4.3 YOCO

The most aggressive bounded-shape variant is You Only Cache Once ([YOCO](https://arxiv.org/abs/2405.05254), Sun et al., Microsoft, NeurIPS 2024 oral). Instead of caching $K, V$ at every layer, YOCO splits the model into a *self-decoder* (producing one global $K, V$) and a *cross-decoder* (every layer of which attends via cross-attention to that single global pair). Per-request KV is no longer a function of $L$ — only of one logical layer's worth — yielding an $L\times$ memory reduction. The paper demonstrates 1M-context prefill in seconds at scales where dense MHA would not fit. The trade is architectural: YOCO is a different topology, not a drop-in modification, and no frontier model has been trained on it. It remains a research direction rather than a deployed architecture, alongside Mamba and the SSM family.

## 5. Gated linear attention: GLA and GTA

Linear attention replaces softmax with an associative kernel, turning attention into a recurrence with fixed-size state:

$$o_t \;=\; \phi(q_t)^{\top} S_t, \qquad S_t \;=\; S_{t-1} + \phi(k_t) v_t^{\top}, \qquad S_t \in \mathbb{R}^{d_k \times d_v}.$$

The state has fixed size $d_k d_v$ regardless of context length — there is no per-token cache. Vanilla linear attention significantly underperforms softmax attention on long-range tasks because the state has no forgetting mechanism.

**GLA** (Gated Linear Attention, [Yang et al., MIT, ICML 2024](https://arxiv.org/abs/2312.06635)) adds a data-dependent gate that lets the state both write and *forget* per-position:

$$S_t \;=\; G_t \odot S_{t-1} \;+\; k_t v_t^{\top}, \qquad o_t \;=\; q_t^{\top} S_t.$$

With this gate, GLA closes most of the quality gap to softmax attention at fixed parameter budget while keeping the constant-state property.

The serving implications are significant. At decode time:

- **No KV cache reads.** The "cache" is the recurrent state $S_t$, which lives in registers or shared memory. Decode bandwidth is dominated by weight reads, not state reads — the long-context bandwidth-growth problem in [see §00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md) disappears.
- **Constant arithmetic intensity in $S$.** The recurrence is $O(d_k d_v)$ per step, independent of context length.
- **Different prefill kernel.** GLA prefill uses a parallel-scan formulation. FlashAttention does not apply; engines need a separate GLA backend.

**GTA** (Grouped-head, [GLA-GTA arXiv:2505.21487](https://arxiv.org/abs/2505.21487), Princeton, 2025-05) extends GLA with grouped heads in the spirit of GQA. The result roughly halves KV-state size relative to GQA at matched quality on the authors' benchmarks; the GLA-style decode kernel comes within 2× of FlashMLA on speculative-decoding workloads. As of 2026-05, GTA is a research-direction variant; no major OSS engine ships a default GTA backend.

GLA and GTA are the cleanest demonstration that "no KV cache" is a coherent design point: the per-token left-context summary is compressed into a fixed $O(d^2)$ object the decoder rolls forward, rather than an $O(S d)$ object the decoder reads each step.

## 6. Hybrid Mamba-Transformer cache shapes

Selective-state-space models (SSMs) — Mamba ([Gu, Dao, 2023](https://arxiv.org/abs/2312.00752)), Mamba-2 ([Dao, Gu, ICML 2024](https://arxiv.org/abs/2405.21060)), and the latest canonical reference **Mamba-3** ([ICLR 2026 oral, arXiv:2603.15569](https://arxiv.org/abs/2603.15569)) — replace attention with a structured recurrence whose state is a fixed-size matrix per layer:

$$h_t \;=\; A_t h_{t-1} + B_t x_t, \qquad y_t = C_t h_t,$$

with input-dependent $A_t, B_t, C_t$ and state $h_t \in \mathbb{R}^{d \times N}$ (typically $N \in [16, 128]$). Decoding maintains $h_t$ and updates it in $O(d N)$ per step, independent of context length. There is no KV cache. For practical sizes ($d \approx 4096, N{=}16$), per-layer state is roughly 128 KB; across 32 layers that is 4 MB per request — a million-fold reduction relative to MHA at 128k context. Decode arithmetic intensity is constant in $S$.

Pure SSMs have one well-known weakness: in-context retrieval is significantly weaker than attention-based models at matched parameters. Mamba-2 narrowed this gap; Mamba-3 narrows it further with better hardware-efficient training and improved selectivity, but the gap has not closed entirely. The pragmatic 2025–2026 response has been hybrid Mamba-Transformer architectures.

### 6.1 Hybrid lineage

| Model | Org / date | Hybrid pattern |
|---|---|---|
| Jamba | AI21, 2024-03 | 1:7 attn:Mamba interleave + MoE; 256k ctx |
| Jamba-1.5 | AI21, 2024-08 | same pattern at scale |
| Hymba | NVIDIA, 2024-11 | parallel attn+SSM in same layer + meta tokens |
| Falcon-H1 | TII, 2025-07 | parallel attn + SSM, concatenated outputs; 0.5B–34B |
| Granite-4 | IBM, 2025-10 | 9:1 Mamba-2 : attention; >70% RAM reduction at long ctx |

The patterns stabilize around two designs: *interleaved* (some layers attention, some SSM) and *parallel* (within one layer, attention and SSM run in parallel). Interleave ratios vary — Jamba 1:7, Granite-4 1:9, Hymba parallel — and **the optimal ratio is unsettled as of mid-2026.** The benchmarks are workload-dependent: RULER and LongBench v2 favor higher attention ratios; SCBench-style multi-turn reuse favors lower. This is currently a tunable hyperparameter without a defensible first-principles answer.

### 6.2 The heterogeneous-cache problem

A hybrid model has *two distinct cache shapes per request*:

- **Per-attention-layer KV cache.** Standard $2 b H_{\text{kv}} d_h$ per token, growing in $S$. Subject to the usual paging, prefix caching, FP8 quantization, tiered offload.
- **Per-SSM-layer recurrent state.** Fixed-size $b \cdot d \cdot N$ per layer per request, constant in $S$, updated in place each step.

The serving engine must hold both simultaneously, and the consequences fall out of the shape mismatch:

- **Memory accounting splits.** KV-pressure metrics used by autoscalers ([see §50/02-autoscaling-cost-and-sustainability](../50-cluster-systems/02-autoscaling-cost-and-sustainability.md)) measure the attention-layer cache only; the SSM state is a per-request constant.
- **PagedAttention does not apply to SSM layers.** The SSM state is a single fixed-size object per request; there is no paging. Engines maintain a separate per-request state allocator (vLLM V1's SSM-state tracker, SGLang's `MambaCache`, custom paths in TRT-LLM).
- **Prefix caching is partial.** Cross-request reuse is straightforward for attention-layer KV blocks, but the SSM state at the prefix's end must also be reused. SGLang's hybrid radix cache attaches the SSM state to the radix-tree node; vLLM keeps an aligned SSM-state cache. Prefix caching for hybrids is engine-specific and less mature than for pure-attention models.
- **Spec-decoding interactions.** Speculative decoding ([see §10/05-speculative-decoding](../10-engine-core/05-speculative-decoding.md)) requires rolling per-request state back on draft rejection. For attention layers, the KV cache supports this natively. For SSM layers the state must be checkpointed before draft tokens and restored on rejection.

The 2025–2026 OSS engine work has been catching up. None of the engine support is yet on par with pure-attention serving, and long-tail correctness issues (especially around prefix-cache invalidation when SSM state and KV blocks become inconsistent under chunked prefill) are still surfacing.

Mamba-3 (ICLR 2026 oral) is the canonical SSM reference for new training runs as of 2026-05. It improves on Mamba-2 in two operationally relevant ways: better Hopper/Blackwell tensor-core utilization via a chunkwise-parallel formulation that fuses the structured-state-space scan with matmul, and better selectivity that further narrows the in-context-retrieval gap. No major frontier hybrid has yet retrained on Mamba-3 (Granite-4 used Mamba-2; Falcon-H1 a custom SSM parameterization).

## 7. Non-Transformer architectures: diffusion language models

The variants through §6 all preserve the autoregressive, causal-mask, one-token-at-a-time assumption built into the engine architecture of [see §10/03-batching-scheduling](../10-engine-core/03-batching-scheduling.md). Diffusion language models break that assumption.

A diffusion LM generates by *denoising*: starting from a fully-masked or fully-noised token sequence, the model iteratively refines all positions in parallel over a small number of denoising steps (typically 8–32). Each step looks at the entire partially-denoised sequence and produces an updated sequence. There is no "next token" being emitted — every position is updated each step.

Two production-relevant systems as of 2026:

- **Mercury** (Inception Labs, 2024–2025): commercial diffusion LM offered via API with reported throughput substantially higher than autoregressive models of comparable quality. Inception Labs' public statements describe Mercury as running on a *proprietary engine*; standard OSS engines (vLLM, SGLang) do not directly apply.
- **LLaDA** (2024): open masked-diffusion language model, BERT-style — at each step, masked tokens are predicted and a sampled subset is committed, with the rest re-masked for the next step. LLaDA established that a competitive diffusion LM is trainable from open data and provides a public reference architecture.

The serving challenge: there is no KV cache, but there is also no per-token decode regime. The "state" is the full partially-denoised token sequence, re-processed by every layer at every denoising step.

- **Per-step compute resembles prefill.** Every step processes all $N$ positions with full self-attention. Arithmetic intensity per step is $O(N)$ — compute-bound on any current accelerator.
- **No KV growth.** Memory is bounded at the sequence length, not by per-token cache accumulation. Prefix caching as defined in [see §10/07-prompt-prefix-caching](../10-engine-core/07-prompt-prefix-caching.md) does not directly apply.
- **Engine assumptions break.** Continuous batching is built around per-step token emission with KV growth; speculative decoding accelerates next-token prediction with verification; paged KV memory is the central memory abstraction. None of these directly applies to a diffusion LM. Engine adaptation is an open problem.

The acknowledgement of this break is reflected in OSS engine architecture: SGLang ships a *separate* diffusion engine path rather than retrofitting the autoregressive runtime. The same is reportedly true at Inception Labs internally: Mercury's serving engine is described as built specifically for the diffusion regime.

The chapter's takeaway is that **diffusion language models are a qualitatively different serving workload from autoregressive serving**, and the engine assumptions catalogued in the rest of this book are autoregressive-specific. The Mercury and diffusion-LM details in this section are hedged: Inception Labs has been selective about disclosure, and most public information is at the level of high-level reporting rather than technical disclosure. Mercury runs on a proprietary engine — a sourced claim from Inception Labs' public statements; internals are not speculated on here.

## Current production state

As of May 2026, the dense-frontier model lineage is overwhelmingly **GQA at $G{=}8$**: Llama-3 (8B, 70B, 405B), Llama-4, Mistral 7B v0.3, Mixtral, Qwen2.5, Qwen3, Gemma 2/3, GPT-OSS, and most enterprise deployments. The 4 KB/token/layer figure for $H{=}64, d_h{=}128, G{=}8$ is the de-facto baseline in capacity planning for non-MLA dense long-context workloads.

The MLA lineage is centered on DeepSeek's own line (V2, V3, V3.1, V3.2, R1, V3.2-Speciale) and Moonshot's Kimi K2 family. MLA is a compelling architectural answer with confirmed production deployment at scale, but is **not yet a settled industry default**: most non-DeepSeek frontier model families chose GQA or alternative architectural moves (interleaved SWA, native sparse attention) rather than MLA. The kernel ecosystem caught up over 2025: FlashMLA from DeepSeek's open-source week (Feb 2025), ThunderMLA (Hazy Research, March 2025), and MLA-aware paths in vLLM, SGLang, and TRT-LLM. The MLA + chunked-prefill + prefix-caching interaction has been a recurring sharp edge ([see §10/07-prompt-prefix-caching](../10-engine-core/07-prompt-prefix-caching.md)).

Sliding-window attention is **production-deployed at frontier scale in interleaved form**: Gemma 3, GPT-OSS, Qwen3-Next all use SWA layers with full-attention layers in some pattern. Pure-SWA models in the Mistral 7B v0.1/v0.2 style are not the production norm in 2026; the v0.3 transition to GQA reflected the broader community preference for actual long-context support over a hard cache cap. Associated kernel work — circular-buffer KV in TRT-LLM, SWA-aware allocators in vLLM and SGLang, attention-sink kernel support in FA-3 — is settled.

Hybrid Mamba-Transformer architectures crossed into mainstream OSS in 2025: Jamba-1.5 (AI21), Falcon-H1 (TII), Granite-4 (IBM), with Granite-4 in particular targeting enterprise deployments where the >70% RAM reduction at long context translates to operational cost wins. Mamba-3 (ICLR 2026 oral) is the canonical SSM reference for new hybrid training runs as of 2026-05; no major frontier hybrid has yet retrained on Mamba-3. The optimal attn:SSM ratio is unsettled. Engine support for the heterogeneous-cache problem is improving but is not on par with pure-attention serving.

YOCO and pure GLA / GTA remain research directions. Diffusion language models — Mercury and LLaDA — are the qualitatively different regime: no KV cache, iterative denoising of the full sequence, and an engine architecture not derived from autoregressive serving. SGLang ships a separate diffusion engine path acknowledging the regime break. As of 2026-05, the production footprint of diffusion LMs remains small relative to autoregressive serving, but the workload is real and the engine adaptation is an open infrastructure problem worth tracking.

The unifying conclusion is that the cache is not an implementation detail; it is the architectural axis around which every modern attention variant is designed. The KV-shape question is the question the next attention paper will answer too, and an engineer reading that paper should be able to read the bytes-per-token-per-layer formula off its first page.
