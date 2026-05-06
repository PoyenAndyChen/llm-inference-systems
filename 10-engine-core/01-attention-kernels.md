# Attention Kernels

**After reading this chapter, the reader will be able to:**

- Derive the online-softmax recurrence and the IO-complexity argument that anchor every FlashAttention variant, and trace the algorithmic lineage from FA-1 through FA-4 alongside the variant kernels (FlashDecoding, FlashInfer, FlashMLA, ThunderMLA, GLA/GTA, DSA).
- Read a Hopper or Blackwell attention kernel at the level of TMA, WGMMA, warp specialization, tcgen05, and TMEM, and explain why FA-4 needed a CuTe-DSL rewrite to handle Blackwell's asymmetric hardware scaling.
- Predict, for a given engine and (architecture, model-shape, KV-layout, precision) tuple, which backend the engine's attention selector will choose, and why the cross-engine convergence on FlashInfer (NVIDIA-side) and the parallel ROCm AITER/Wave lineage have shaped the production landscape.

Attention is the kernel that defines the field. GEMMs, all-reduces, RMSNorm fusions are shared with training; attention is uniquely an inference problem — the only kernel whose shape changes with sequence position, whose access pattern is set by per-request KV layout, and whose arithmetic intensity differs by an order of magnitude between prefill and decode [see §00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md). It is also where the largest gaps remain between hand-written CUDA and what compilers can produce, which is why attention has its own DSLs (CuTe-DSL, ThunderKittens, TileLang) and its own dedicated kernel collection (FlashInfer) that no other operator type has earned.

The chapter is organized into four parts. **Part 1** is the algorithm: online-softmax, IO complexity, FA-1→FA-4 lineage, FlashDecoding, FlashInfer's BSR unification, and the MLA family. **Part 2** is per-architecture implementation on Hopper and Blackwell — the hardware features the algorithms exploit and the structural reason FA-3 cannot run on Ampere and FA-4 needed a rewrite for Blackwell. **Part 3** covers variant kernels — paged-aware, sparse / sink / sliding-window, and quantized. **Part 4** is the ecosystem: DSLs, per-engine backend selectors, and CUDA-graph interaction. The reader is assumed to be comfortable with `Q · Kᵀ`, the SM memory hierarchy, and the prefill/decode split.

---

## PART 1 — Algorithm

### 1.1 The online-softmax problem

Standard softmax over a length-$N$ vector $x$ is

$$\text{softmax}(x)_i \;=\; \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}.$$

Two facts set the entire structure of attention kernels. First, the denominator $\ell = \sum_j e^{x_j}$ depends on every element — computing $\text{softmax}(x)_i$ exactly requires a *complete* pass over $x$ before any output element can be finalized. Second, $e^{x_j}$ overflows at moderate values ($e^{89}$ already exceeds FP32 range), so production code uses the numerically safe form

$$\text{softmax}(x)_i \;=\; \frac{e^{x_i - m}}{\sum_{j} e^{x_j - m}}, \qquad m = \max_j x_j,$$

mathematically identical because $e^{-m}$ cancels. Safe softmax thus requires a first pass for $m$, a second for the exponentials and $\ell$, and a third to write outputs.

The naive attention kernel matches this literally: compute the full $N \times N$ score matrix $S = QK^\top / \sqrt{d_h}$ in HBM, find row-wise maxima, compute exponentials, sum, divide, multiply by $V$. The score matrix is the bottleneck — at $N=4096$, $S$ is 64 MiB in FP16 per head, far larger than any L2 cache, and it is read and written multiple times. FlashAttention's central observation is that the *score matrix never has to exist* if softmax can be made streaming.

### 1.2 The online-softmax recurrence

Suppose $x$ is split into two contiguous blocks $x = [x^{(1)}, x^{(2)}]$. Compute on block 1 alone:

$$m^{(1)} = \max(x^{(1)}), \quad \ell^{(1)} = \sum_j e^{x_j^{(1)} - m^{(1)}}, \quad o^{(1)} = \sum_j e^{x_j^{(1)} - m^{(1)}} v_j^{(1)}.$$

Now block 2 arrives. The new running maximum is $m_{\text{new}} = \max(m^{(1)}, \max(x^{(2)}))$. The trick is that $\ell^{(1)}$ was computed relative to the *old* maximum $m^{(1)}$, which may now be smaller than $m_{\text{new}}$. Each old exponential picks up a multiplicative correction $e^{m^{(1)} - m_{\text{new}}}$:

$$
\ell_{\text{new}} \;=\; e^{m^{(1)} - m_{\text{new}}} \cdot \ell^{(1)} \;+\; \sum_{j \in \text{block 2}} e^{x_j^{(2)} - m_{\text{new}}}.
$$

Output state rescales the same way:

$$
o_{\text{new}} \;=\; e^{m^{(1)} - m_{\text{new}}} \cdot o^{(1)} \;+\; \sum_{j \in \text{block 2}} e^{x_j^{(2)} - m_{\text{new}}} v_j^{(2)}.
$$

By induction, after $K$ blocks, the running tuple $(m_K, \ell_K, o_K)$ is *exactly* what a single-pass safe softmax would have produced; the final output is $o_K / \ell_K$. The identity is $e^{x - m_{\text{new}}} = e^{x - m_{\text{old}}} \cdot e^{m_{\text{old}} - m_{\text{new}}}$, so the rescale factor $e^{m_{\text{old}} - m_{\text{new}}} \in (0, 1]$ corrects every previously-accumulated exponential to the new reference.

This is the *online-softmax recurrence*, the algorithmic heart of every FlashAttention variant:

$$
\boxed{
\begin{aligned}
m_{\text{new}} &= \max(m_{\text{old}}, \max(x_{\text{new}})), \\
\ell_{\text{new}} &= e^{m_{\text{old}} - m_{\text{new}}} \cdot \ell_{\text{old}} + \sum e^{x_{\text{new}} - m_{\text{new}}}, \\
o_{\text{new}}  &= e^{m_{\text{old}} - m_{\text{new}}} \cdot o_{\text{old}} + \sum e^{x_{\text{new}} - m_{\text{new}}} \cdot v_{\text{new}}.
\end{aligned}
}
$$

The recurrence turns attention from a two-pass algorithm into a streaming one. Tiles of $K$ and $V$ are loaded one at a time, contribute to the running output, and are discarded; only the per-row $(m, \ell, o)$ state survives. The score matrix is never materialized in HBM. The online-softmax derivation itself predates FlashAttention by several years (Milakov and Gimelshein, "Online normalizer calculation for softmax," 2018); FA-1 was the first to wire it into a fused attention kernel with the right tile shapes.

### 1.3 IO complexity

Streaming was motivated by hardware. The arithmetic-intensity argument in [§00/02](../00-foundations/02-transformer-arithmetic-roofline.md) says the binding constraint at long sequences is HBM traffic. Standard attention pays $O(N^2)$ HBM writes for $S$ and another $O(N^2)$ HBM reads for the softmax-and-multiply pass — quadratic in $N$, dwarfing the $O(Nd)$ traffic on $Q, K, V$ once $N \gg d$.

FA-1's IO-complexity result is that, with on-chip SRAM of size $M$, tiled streaming attention requires $\Theta(N^2 d / M)$ HBM accesses — quadratic in $N$ but reduced by SRAM capacity. For Ampere ($M \approx 192$ KB per SM) and $d = 128$, that is roughly $N^2 / 1500$ HBM accesses on the attention sub-layer, a ~1500× reduction. The HBM *write* of the score matrix — the dominant cost in naive attention — vanishes entirely; FA-1 never writes $S$ to HBM. The FLOP count is unchanged; FA avoids paying HBM bandwidth for the $O(N^2 d)$ multiply-adds, not the multiply-adds themselves. On bandwidth-bound hardware the wall-clock speedup is therefore close to the bandwidth-saving ratio.

The tile sizes $B_r$ (query rows) and $B_c$ (KV columns) are chosen so the working set $B_r d + B_c d + B_r B_c$ fits in SRAM; FA-3 on H100 uses 64 or 128 along each axis depending on head dimension. The choice trades occupancy (smaller tiles, more concurrent thread blocks) against data reuse (larger tiles, amortized load latency).

### 1.4 The FlashAttention lineage

The FlashAttention lineage is the most consequential single thread in production attention kernels. Each version was a hardware-aware re-engineering of the same algorithm.

**FA-1** ([FlashAttn-1](../papers.md#flashattn-1), Dao, Fu, Ermon, Rudra, Ré, NeurIPS 2022) introduced tiled streaming attention with the online-softmax recurrence; results on A100 were 2–4× over hand-tuned baselines. The kernel parallelized over `(batch, head)` pairs only; at small batch×head products the SMs starved.

**FA-2** ([FlashAttn-2](../papers.md#flashattn-2), Dao, ICLR 2024) re-partitioned work along the sequence axis so thread blocks could parallelize across queries (the outer loop), and reshuffled warp-fragment ownership to reduce non-matmul FLOPs (the rescale operations on Ampere went through SFU rather than tensor cores). FA-2 reached 50–73% of A100 FP16 peak — the de-facto attention reference for several years.

**FA-3** ([FlashAttn-3](../papers.md#flashattn-3), Shah, Bikshandi, Zhang, Thakkar, Ramani, Ré, Dao, NeurIPS 2024) was the Hopper rewrite. Hopper added TMA (Tensor Memory Accelerator, for async bulk HBM↔SRAM copies) and WGMMA (warp-group MMA, operating on 64×16×16 tiles across all four warps of a warp group). Naive use of either left compute units idle while data was in flight. FA-3's contribution was *warp specialization* — producer warps issue TMA loads, consumer warps issue WGMMA — combined with *ping-pong scheduling* between two warp groups, alternating softmax and WGMMA each iteration so SFU and tensor-core lanes execute concurrently. Together with FP8 support via block-quantization plus incoherent processing (a Hadamard-transform-based variance-reduction trick), FA-3 reached ~740 TFLOP/s FP16 and ~1.2 PFLOP/s FP8 on H100 SXM5 — roughly 2× FA-2. The [FA-2-Hopper](../papers.md#fa-2-hopper) Colfax paper was the architectural blueprint.

**FA-4** ([FlashAttn-4](../papers.md#flashattn-4), Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao, March 2026) is the Blackwell rewrite, the first FA written entirely in CuTe-DSL. The problem on B200 is *asymmetric* hardware scaling: tensor-core throughput grew ~2.25× from H100 (the 5th-generation Tensor Cores plus `tcgen05`), but shared-memory bandwidth and MUFU (which dispatches the hardware `exp2`) did not — MUFU on B200 dispatches at ~16 ops/clock per SM against matmul at ~8192 ops/clock per SM. Naive FA-3 ports stalled in softmax with idle matmul lanes. FA-4's headline trick is a software-emulated `exp2` via a Horner-form cubic polynomial on FMA units, bypassing MUFU — Sollya-tuned coefficients ≈ (1.0, 0.6951, 0.2276, 0.0771) on the fractional part after a Cody-Waite range reduction $2^x = 2^n \cdot 2^f$ with integer $n$ folded into the IEEE-754 exponent for free. The forward kernel pipelines two query tiles per CTA in a ping-pong; the backward uses a 2-CTA MMA mode that halves operand-B traffic. NVIDIA-affiliated and co-author benchmarks report 1605 TFLOP/s BF16 on B200, ~71% utilization, 1.1–1.3× over cuDNN 9.13 fused SDPA, and 2.1–2.7× over Triton — vendor-supplied numbers, not independently replicated outside Together / Modal / Colfax / NVIDIA channels; treat as an upper bound. Replication on consumer Blackwell (SM120/SM121, RTX 5090, RTX 6000 Pro) is incomplete: those parts lack `tcgen05` and TMEM, and the SM100 cubins do not recompile down. FA-4 ships as a separate install path in `Dao-AILab/flash-attention` depending on `nvidia-cutlass-dsl`.

> **Sidebar — FA-3 still dominates the installed base.** Despite FA-4's headline numbers, the median production token in 2026 still flows through FA-3 on H100 or H200. Blackwell ramp is under way at every major hyperscaler, but most enterprise inference clusters and the long tail of OSS deployments are Hopper-default. vLLM and SGLang both dispatch FA-3 on SM90, and FA-3's footprint will outlast its frontier-default status by years.

### 1.5 FlashDecoding

FA-1/2/3 are prefill-shaped: they parallelize across batch×head and (after FA-2) across the query sequence axis. *Decode* is the opposite shape: query length 1 against a long KV history. At batch 1, almost all of an H100's 132 SMs are idle; the kernel is bound on a single SM walking $S$.

**FlashDecoding** ([FlashDecode](../papers.md#flashdecode), Dao, Haziza, Massa, Sizov, October 2023, blog) adds a third parallelism axis: the KV sequence itself. KV is split across SMs; each SM computes a partial $(m^{(i)}, \ell^{(i)}, o^{(i)})$; a final reduction merges them using the online-softmax rescale from §1.2. The merge is one rescale per partial, bit-exact to a single-pass kernel — the same identity as the inter-block update, applied across SMs. This is the canonical decode-side parallelism strategy and powers FlashInfer's paged decode, FlashMLA, TRT-LLM XQA, and the vLLM Triton decode kernel. **FlashDecoding++** ([FlashDecode++](../papers.md#flashdecode), Hong et al., MLSys 2024) added an asynchronized softmax with a unified maximum estimate plus a flat-GEMM optimization for query-length-1 decode; it is the MLSys-publication-of-record for the split-K family.

### 1.6 FlashInfer and the BSR unification

Production attention has more shapes than dense $(B, H, N, d_h)$. Real batches are heterogeneous: requests at different prefill positions, with different KV histories, paged-block lookups, sometimes tree-shaped query masks (speculative decoding), sometimes radix-tree-shaped KV (prefix sharing). Writing a separate kernel for every (batch shape, KV layout, mask topology) tuple is not maintainable.

**FlashInfer** ([FlashInfer](../papers.md#flashinfer), Ye, Chen, Lai, Lin et al., MLSys 2025 best paper) unified them. The move is to express every KV layout — paged, ragged, radix-tree, tree-mask — as a single **block-sparse-row (BSR)** matrix: rows are query positions, columns are key positions, and the sparsity pattern encodes which K and V blocks each query attends to. A flat-paged KV is a BSR with a block-grouping structure; a per-request paged KV is a BSR with permuted columns; an EAGLE tree mask is a BSR with a triangular-within-tree pattern. Once everything is BSR, one family of kernel templates handles all of them.

FlashInfer's templates are JIT-compiled — kernels specialized to `(head_dim, page_size, mask_type, dtype, ...)` are instantiated at startup. An **inspector-executor** scheduler walks per-batch metadata (block tables, query offsets, KV lengths) once, builds a flat work schedule, and emits a fixed-sized work plan that the executor consumes across SMs. The fixed launch shape is what makes FlashInfer CUDA-graph-compatible — per-batch metadata is a tensor argument, not a launch-shape variable.

As of 2026, **FlashInfer is the dominant cross-engine NVIDIA-side kernel collection**. It is the source of truth for paged-attention kernels in vLLM (`FLASHINFER`, `FLASHINFER_MLA`), SGLang (`FlashInferAttnBackend`, `FlashInferMLAAttnBackend`, FlashInfer-MoE dispatchers), MLC, and increasingly TRT-LLM (NVIDIA actively ships TRT-LLM-derived kernels through FlashInfer for engine reuse). NVIDIA stewardship of the project is widely reported — the project is the conduit through which NVIDIA-authored kernels reach OSS engines — but no formal acquisition document is on record as of May 2026. FlashInfer originated at UW, CMU, and OctoAI; OctoAI itself was acquired by NVIDIA. The careful framing: *NVIDIA stewardship is widely reported; no formal acquisition document confirmed*.

The parallel ROCm-side story is different. **AITER** (AMD Inference Tensor Engine Runtime) and **Wave** build on ROCm's Composable Kernel framework and are integrated into vLLM (`ROCM_AITER_FA`, `ROCM_AITER_UNIFIED_ATTN`) and SGLang (`AiterAttnBackend`, `WaveAttnBackend`). The ROCm-side ecosystem lacks a single equivalent to FlashInfer's BSR unification — kernel selection is more bespoke per layout — but AMD MI300/MI355 paths in production engines are functional today.

### 1.7 MLA kernels: FlashMLA, ThunderMLA

Multi-head Latent Attention (MLA), introduced in [DeepSeek-V2](../papers.md#mla-v2) and refined in [DeepSeek-V3](../papers.md#deepseek-v3-fp8), compresses the per-token KV state to a small latent vector $c_{\text{KV}} \in \mathbb{R}^{d_c}$ plus a small RoPE-decoupled key of dimension $d_h^R$. The KV footprint drops by roughly an order of magnitude versus GQA (1152 bytes/token/layer at FP16 vs. 4096 for an equivalent GQA — see [§00/02](../00-foundations/02-transformer-arithmetic-roofline.md) Table 2.2).

The kernel-side challenge is that the latent must be *up-projected* by $W^{UK}$ and $W^{UV}$ to recover per-head K and V before attention compute. Materializing those tensors defeats the bandwidth saving entirely — the up-projections are dense-MHA-sized. The MLA *absorption trick* fuses the up-projection into the surrounding kernels:

- $W^{UK}$ is absorbed into the Q projection: $q^\top k = q^\top W^{UK} c_{\text{KV}} = (W^{UK,\top} q)^\top c_{\text{KV}}$. The factor $W^{UK,\top} q$ is computed once at Q-projection; the dot product runs against the *compressed* latent, with KV bandwidth proportional to $d_c$ rather than $d$.
- $W^{UV}$ is absorbed into the output projection: $\sum p_j v_j = W^{UV} \sum p_j c_{\text{KV}}^{(j)}$ — pulling $W^{UV}$ outside the sum.

Both moves are algebraic identities; the cost is the kernel-engineering work to make them happen *inside* the attention kernel, with tile shapes for the compressed K/V dimensions, the page-table interaction (the latent is what the page table indexes), and the RoPE-decoupled component handling.

**FlashMLA** ([FlashMLA](../papers.md#flashmla), DeepSeek, February 2025 "Open-Source Week" day 1) is the canonical MLA decode kernel for Hopper, targeting BF16 paged KV with block-size 64 and reporting 3000 GB/s memory-bound and 580–660 TFLOP/s compute-bound throughput on H800. Its release was the enabling event for DeepSeek production-stack reproduction in vLLM and SGLang in March–May 2025. **ThunderMLA** ([ThunderMLA](../papers.md#thundermla), Hazy Research, March 2025) is a ThunderKittens reimplementation with an on-device scheduler. Engine integrations also include **CutlassMLA**, **TRTLLM-MLA**, and **FlashInferMLA** — all FlashAttention-style implementations specialized for the $(qk\_head\_dim \neq v\_head\_dim)$ shape.

DeepSeek-V3.2 (September 2025) added **DSA** (DeepSeek Sparse Attention) as a natively-trained sparse primitive: a lightning-indexer (FP8) scores token importance, top-$k$ selection drives attention from $O(L^2)$ to $O(Lk)$. DSA kernels live in FlashMLA's sparse path, reportedly reach 640/410 TFLOP/s prefill/decode on H800, with day-0 support in vLLM (`FLASHMLA_SPARSE`) and SGLang (`NativeSparseAttnBackend`). [DSA-V32](../papers.md#dsa-v32) is the canonical reference; downstream-quality parity vs. V3.1 is reported by DeepSeek and not independently confirmed at long-context benchmarks beyond their evals. Cross-reference: [§20/04-long-context-inference](../20-distributed-inference/04-long-context-inference.md).

### 1.8 GLA, GTA, and the latent-attention frontier

[GLA-GTA](../papers.md#gla-gta) (Zadouri et al., May 2025; same first author as FA-4) introduces **Grouped-Tied Attention (GTA)** — a parameter-sharing variant of GQA that halves KV footprint at parity quality — and **Grouped Latent Attention (GLA)**, a parallelism-friendly cousin of MLA. The GLA kernel reportedly runs ≤2× faster than FlashMLA in speculative-decode regimes. These are model-architecture changes more than kernel changes; the architectural taxonomy is in [§30/03-attention-variants](../30-kv-cache/03-attention-variants.md). From a kernel perspective, GLA inherits MLA's absorption requirements; a generic FlashInfer BSR template handles both.

---

## PART 2 — Per-architecture implementation

The algorithm is hardware-agnostic; the kernel is not. Each NVIDIA datacenter generation has shipped architectural features that demanded a kernel rewrite.

### 2.1 Hopper (H100 / H200): TMA, WGMMA, warp specialization

**Tensor Memory Accelerator (TMA).** A dedicated copy engine for async bulk HBM↔SRAM transfers. One thread issues a TMA descriptor; the engine moves the data without occupying compute lanes or instruction slots. The issuing warp continues executing while the copy is in flight; a barrier waits for completion. TMA decouples data movement from compute and is the precondition for warp specialization. TMA is Hopper-only; Ampere lacks it, which is the structural reason FA-3 cannot run on Ampere — porting the pipeline back would require synchronous `cp.async`, with substantially different scheduling.

**WGMMA.** A tensor-core instruction at *warp-group* granularity (four warps, 128 threads) rather than per-warp. WGMMA expects A in registers and B in SMEM (or both in SMEM), runs async, computes 64×16×16 tiles in BF16/FP16/FP8. The warp-group granularity let Hopper scale peak tensor-core throughput without proportionally scaling per-warp register file — at the cost that programmers reason about warp groups, not warps.

**Warp specialization** partitions warps within a thread block by role: **producer warps** issue TMA loads of upcoming K/V tiles; **consumer warps** (one or two warp groups) execute WGMMA against the current tiles and run softmax. The two communicate through async barriers — producer signals tile ready, consumer signals SMEM slot reusable. With double- or triple-buffering, the producer stays one or two iterations ahead and the consumer never stalls on data.

**Ping-pong scheduling** runs two warp groups simultaneously, alternating roles: while group A runs WGMMA of iteration $i$, group B runs softmax of iteration $i-1$; the next clock pair flips. Softmax goes through SFU / MUFU; WGMMA goes through tensor cores — different hardware units, executable concurrently without contention. This is what unlocked FA-3's 2× over FA-2: one warp group's softmax overlaps with the other's matmul, leaving tensor cores idle a smaller fraction of cycles.

The FA-3 iteration pipeline:

```
producer:    [TMA load K_i+1, V_i+1]  →  [TMA load K_i+2, V_i+2]  →  ...
consumer A:  [WGMMA Q · K_i^T]    [softmax]    [WGMMA P · V_i]
consumer B:                       [WGMMA Q · K_{i+1}^T]    [softmax]    [WGMMA P · V_{i+1}]
                                  ↑ A's softmax overlaps B's first WGMMA
```

FA-3's **FP8** path adds block-quantization (per-block K/V scales applied during dequant inside the kernel) plus *incoherent processing* (a Hadamard transform on Q and K spreads outliers across channels, reducing FP8 error). The combination preserves accuracy while running tensor cores at FP8 peak — ~1.2 PFLOP/s on H100 SXM5.

### 2.2 Blackwell (B200 / GB200 / GB300): tcgen05, TMEM, asymmetric scaling

Blackwell scaled tensor-core throughput sharply but kept other on-chip resources roughly constant — naive FA-3 ports stalled on un-scaled units while matmul lanes were idle.

**5th-generation Tensor Cores and `tcgen05`.** A new family of MMA instructions on *paired CTAs* (two CTAs sharing data through a new on-chip memory tier, TMEM). `tcgen05.mma` produces results into TMEM rather than registers, and is ~2.25× faster per SM than WGMMA at FP16/BF16; NVFP4 and MXFP8 are first-class formats with hardware microscaling. SM100 (datacenter Blackwell — B100/B200/B300, GB200/GB300 NVL72) ships `tcgen05`; SM120/SM121 (consumer Blackwell — RTX 5090, RTX 6000 Pro, DGX Spark) does not, falling back to `mma.sync.aligned.block_scale`. The split matters: SM100 cubins do not run on consumer Blackwell, and FA-4's SM100 cubins are not available on RTX 5090 / 6000 Pro as of mid-2026.

**TMEM (Tensor Memory).** A 512 KB on-chip tier per SM cluster, distinct from shared memory, not visible to ordinary loads/stores; the only legal destination for `tcgen05.mma` outputs. TMEM exists because the per-SM register file did not scale with tensor-core throughput. TMEM is layout-restricted (16-byte rows, 32-row warp lanes); getting tile shapes right for TMEM is the largest source of on-Blackwell kernel-engineering work.

**Asymmetric scaling.** Tensor cores grew ~2.25×; SMEM bandwidth grew sub-2×; the hardware `exp2` in MUFU did not scale at all — it dispatches at ~16 ops/clock per SM on B200 versus ~8192 ops/clock for matmul. Softmax exponentials, a non-bottleneck on Hopper, became the bottleneck on Blackwell. **Polynomial `exp2` on FMA** is FA-4's response: a Horner cubic on FMA units at FMA rate (~8192 ops/clock per SM), bypassing MUFU. Sollya-tuned coefficients (≈ 1.0, 0.6951, 0.2276, 0.0771) plus a Cody-Waite range reduction $2^x = 2^n \cdot 2^f$ — $n$ folded into the IEEE-754 exponent for free, $f$ approximated by the cubic — produces a few-ulps-accurate `exp2` at FMA throughput. The accuracy is below the BF16 attention noise floor; the throughput win is order-of-magnitude. [FA4-RE](../papers.md#fa4-re) (Modal) and [FA4-Colfax](../papers.md#fa4-colfax) are the clearest engineering walkthroughs.

**Asymmetric WGMMA scaling problem at Blackwell precision.** At FP8 / NVFP4, per-tile scaling factor distributions interact non-trivially with the WGMMA accumulator. Naive ports of Hopper FP8 attention observed accuracy regressions on B200 traceable to this; FA-4's mitigations are documented in the paper's accuracy section. The engineering takeaway: Blackwell FP8/FP4 attention needs Blackwell-specific calibration, not a Hopper recipe ported unchanged.

**FA-4 backward.** The backward adds a 2-CTA MMA mode that halves operand-B HBM traffic by sharing operand-B across CTA pairs. Paper reports 3.15× over FA-2 backward; independent replication beyond co-author-affiliated channels (Modal, Together, Lambda, Colfax) is incomplete as of May 2026.

### 2.3 The structural takeaway

Each NVIDIA generation reshapes which architectural features attention kernels can use. Ampere had `cp.async` and warp-level MMA; Hopper added TMA, WGMMA, and warp groups; Blackwell added `tcgen05`, TMEM, and CTA pairing while leaving SFU and SMEM bandwidth lagging. The FA lineage documents how these features map to the online-softmax-tile structure — and the limits of hand-written CUDA C++. FA-4's migration to CuTe-DSL was driven precisely because the template metaprogramming for CTA pairing, TMEM layouts, and polynomial-`exp2` dispatch became unmanageable in raw C++.

The lesson for an engine maintainer: attention is not a static "use FA, it's fastest" choice. It is a hardware-aware dispatch — covered in Part 4.

---

## PART 3 — Variant kernels

The dense kernel is the headline. Production engines dispatch variants for paged KV, sparse / sliding / sink topologies, and quantized attention — the same online-softmax structure with different KV layouts and mask patterns.

### 3.1 Paged-aware kernels

Standard FlashAttention assumes contiguous K and V — single base pointer, single stride, linear walk. Paged attention breaks that: KV is allocated in fixed-size blocks (typically 16 or 32 tokens per block) scattered in HBM, with a per-request *block table* mapping logical positions to physical block indices. The kernel takes the block table as an argument and *gathers* K and V from non-contiguous block addresses [see §10/02-paged-kv-memory](../10-engine-core/02-paged-kv-memory.md).

The kernel-side cost is small: an extra index-table lookup per tile plus indirect K/V loads. The algorithmic cost is zero — the online-softmax recurrence is unchanged. The implementation cost is in the metadata plumbing — `CommonAttentionMetadata`-like structures (block_table, slot_mapping, query_start_loc, seq_lens, …) that every backend consumes. FlashInfer's BSR formulation handles this cleanly: paged KV is a BSR matrix with column blocks for physical KV blocks and per-row block selection from the block table. vLLM's `FLASHINFER` / `FLASHINFER_MLA` use this directly; SGLang's `FlashInferAttnBackend` wraps `flashinfer.BatchPrefillWithPagedKVCacheWrapper` / `BatchDecodeWithPagedKVCacheWrapper`; the cascade-attention path uses `flashinfer.cascade.merge_state` to combine shared-prefix and per-request-suffix partials.

### 3.2 Sparse, sink, and sliding-window kernels

**Sliding-window attention.** Mistral 7B v0.1/v0.2 capped attention to the last $W$ keys; Gemma 3 alternates 5 sliding-window layers with 1 global layer; GPT-OSS uses 1:1 with $W = 128$; Qwen3-Next interleaves similarly. The kernel change is a mask — positions outside the window contribute zero. KV memory is bounded by $\min(S, W)$, decoupling memory from $S$. vLLM's hybrid KV cache manager handles per-layer page-size unification so sliding-window and full-attention layers coexist [see §10/02-paged-kv-memory](../10-engine-core/02-paged-kv-memory.md).

**Attention sinks.** [StreamingLLM](../papers.md#streamingllm) (Xiao et al., ICLR 2024) discovered that LLMs dump unused softmax mass on the first few token positions, treating them as a no-op "sink"; truncating those positions catastrophically degrades quality on long contexts. The fix: always keep the first $K$ tokens, even when older middle-context tokens are evicted. The kernel-side change is a mask combining $W$-token sliding window and $K$-token sink prefix. GPT-OSS and Gemma 3 ship sink-aware kernels by default; vLLM and SGLang support sinks through their FlashInfer / FA backends with a `has_sink` flag. [GPT-OSS-vLLM](../papers.md#gpt-oss-vllm) walks through the 128-window-plus-sink kernel work on Blackwell.

**Block-sparse for NSA / DSA.** Native Sparse Attention (NSA, Feb 2025) and DSA (Sept 2025, DeepSeek-V3.2) are natively-trained sparse primitives: the model trains with a sparse mask selecting "important" KV per query, with a lightning-indexer (FP8) computing the importance score and top-$k$ selection driving complexity from $O(L^2)$ to $O(Lk)$. FlashMLA's sparse path runs the FP8 indexer first, then a BSR-shaped attention kernel against selected blocks. vLLM: `FLASHMLA_SPARSE`. SGLang: `NativeSparseAttnBackend` (`nsa_indexer.py`, `transform_index.py`, `quant_k_cache.py`). DSA integration into FlashInfer's BSR unification or ThunderKittens is incomplete as of mid-2026 — sparse lives on a parallel kernel track. Cross-references: [§30/03-attention-variants](../30-kv-cache/03-attention-variants.md), [§20/04-long-context-inference](../20-distributed-inference/04-long-context-inference.md).

### 3.3 Quantized attention

Quantizing $QK^\top$ scores and the softmax-V product is a kernel track of its own. The motivation is bandwidth: at long context, the dominant cost is reading K and V from HBM; storing them at 8 or 4 bits drops the bandwidth load proportionally.

**FA-3's FP8 path** uses block-quantization (per-block K/V scales) plus incoherent processing (Hadamard on Q and K) to keep FP8 attention numerically accurate at FP8 peak throughput.

**SageAttention 1/2/3** ([SageAttn2](../papers.md#sageattn2), ICML 2025; [SageAttn3](../papers.md#sageattn3), May 2025) is the dominant external lineage. SageAttention-1 was FP8; SageAttention-2 quantized $Q$ and $K$ to INT4 thread-by-thread with outlier smoothing while keeping $P$ and $V$ at FP8 (~3× FA-2 on H100); SageAttention-3 introduced microscaling FP4 (NVFP4, MXFP4) attention — the first plug-and-play NVFP4 attention — reportedly 1038 TOPS on RTX 5090, ~5× FA on the same part.

**KV-cache quantization** is a separate axis from score quantization: K and V can be stored quantized in HBM and dequantized inside the kernel on read. vLLM supports `kv_cache_dtype ∈ {fp8_e5m2, fp8_e4m3, nvfp4}`; SGLang the same plus `auto`. The dequant happens in the K/V load path — not as a separate kernel — because materializing dequantized K/V to HBM would erase the bandwidth saving. The full story (KIVI, KVQuant, TurboQuant) is in [§10/04-quantization](../10-engine-core/04-quantization.md). [BitDecoding](../papers.md#bitdecoding) (March 2025) is an emerging tensor-core-native low-bit KV decode kernel; production status is research-phase as of May 2026.

---

## PART 4 — Ecosystem and selection

Two further layers determine how a kernel reaches a request: the *DSL* (sets who can write kernels, how fast they evolve) and the per-engine *backend selector* (decides at engine startup or per-request which kernel to dispatch).

### 4.1 DSL landscape

The 2017–2024 era wrote attention in CUDA C++, often via CUTLASS. The 2024–2026 generation moved to higher-level DSLs that compile to the same instructions but expose tile primitives directly.

**Triton** (OpenAI, since 2021) is the Python-embedded GPU DSL that became the training-era default. [Liger](../papers.md#liger) (LinkedIn, October 2024) productionized Triton fused kernels (RMSNorm, RoPE, FlashCE, attention variants) into HF Trainer / TRL / Axolotl. On the inference side, [Triton-Anatomy](../papers.md#triton-anatomy) (Ringlein et al., IBM Research, November 2025) documents how a ~800-line Triton attention kernel reached 105.9% of FA-3 on H100 after auto-tuning, Q-block grouping, and persistent kernels — the basis of vLLM's `TRITON_ATTN` backend, default on AMD and universal fallback elsewhere ([vLLM-Triton](../papers.md#vllm-triton)). Triton on Blackwell is functional but reportedly trails CuTe-DSL FA-4 by 2.1–2.7× on headline kernels (vendor-supplied; hedge).

**CuTe-DSL** ([CuTe-DSL](../papers.md#cute-dsl), NVIDIA, 2025) is a Python API on CUTLASS 4 — exposes CuTe's `Tensor` and `Layout` algebra to Python at C++-template performance, with 20–30× faster compile than C++ templates. FA-4 is built entirely in CuTe-DSL; `nvidia-cutlass-dsl` is a hard dependency of the FA-4 install path. CuTe-DSL is Blackwell-targeted (SM100+) by design.

**ThunderKittens** ([ThunderKittens](../papers.md#thunderkittens), Spector et al., Hazy Research, October 2024) is a tile-primitive C++ DSL hitting cuBLAS / FA-grade speeds in a fraction of the source-line count. TK 2.0 ([TK-2.0](../papers.md#tk-2-0), February 2026) added full Blackwell support with NVFP4 / MXFP8 GEMMs at-or-above cuBLAS on B200; powers Cursor Composer training and Together AI inference. ThunderMLA, [HipKittens](../papers.md#hipkittens) (AMD MI300), and ParallelKittens (multi-GPU collective-aware tiles) are the family.

**TileLang** ([TileLang](../papers.md#tilelang), Wang et al., PKU/Microsoft, April 2025; ICLR 2026) reaches ~98% of FlashMLA on H100 in ~70 lines of Python; basis of `QwenLM/FlashQLA` and `deepseek-ai/TileKernels`.

**AITER and Wave** are the AMD ROCm-side DSLs / kernel libraries — AITER wraps tuned Composable Kernel implementations, Wave is a separate DSL. Integration points: SGLang's `AiterAttnBackend` / `WaveAttnBackend`, vLLM's `ROCM_AITER_FA` / `ROCM_AITER_UNIFIED_ATTN`.

| DSL | Hardware target | Production use |
|---|---|---|
| Triton | NVIDIA + AMD via Triton-AMD | vLLM Triton backend (default on AMD); FlexAttention; Liger; SageAttention-2 |
| CuTe-DSL | NVIDIA Blackwell (SM100+); Hopper / Ampere functional | FA-4; CUTLASS 4 examples |
| ThunderKittens | NVIDIA Hopper + Blackwell; AMD via HipKittens | ThunderMLA; Cursor Composer training; Together AI inference |
| TileLang | NVIDIA Hopper + Blackwell | FlashQLA (Qwen); TileKernels (DeepSeek) |
| AITER / Wave | AMD ROCm | vLLM, SGLang on MI300X / MI355X |

The MLSys 2026 NVIDIA Track [FlashInfer Kernel Generation Contest](../papers.md#flashinfer-bench) institutionalizes AI-agent-written GPU kernels (CuTe-DSL / Triton / TileLang / cuTile) as a benchmark.

### 4.2 Per-engine backend selectors

Production engines dispatch at engine startup (and sometimes per-request) based on hardware capability, KV layout (paged vs. ragged), model architecture (MHA vs. GQA vs. MLA vs. NSA), KV dtype (BF16, FP8, NVFP4), and sinks / sliding window / spec-decode trees.

#### vLLM

In vLLM V1, selection lives in `vllm/v1/attention/selector.py::get_attn_backend` → `current_platform.get_attn_backend_cls`. On NVIDIA, `vllm/platforms/cuda.py::CudaPlatformBase.get_attn_backend_cls` iterates `_get_backend_priorities(use_mla, device_capability, num_heads, kv_cache_dtype)` and picks the first valid backend. The `AttentionBackendEnum` registry holds the roster: `FLASH_ATTN`, `FLASH_ATTN_DIFFKV`, `TRITON_ATTN`, `ROCM_ATTN`, `ROCM_AITER_FA`, `ROCM_AITER_UNIFIED_ATTN`, `FLASHINFER`, `FLEX_ATTENTION`, `TREE_ATTN`, `TURBOQUANT`, plus the MLA family in `backends/mla/` (`FLASH_ATTN_MLA`, `FLASHMLA`, `FLASHMLA_SPARSE`, `FLASHINFER_MLA`, `FLASHINFER_MLA_SPARSE`, `CUTLASS_MLA`, `TRITON_MLA`).

Default priority cascades:

| Setting | Priority order |
|---|---|
| Non-MLA, Hopper / Ada / Ampere | FLASH_ATTN → FLASHINFER → TRITON_ATTN → FLEX_ATTENTION → TURBOQUANT |
| Non-MLA, Blackwell | FLASHINFER → FLASH_ATTN → TRITON_ATTN → FLEX_ATTENTION → TURBOQUANT |
| MLA, non-Blackwell | FLASH_ATTN_MLA → FLASHMLA → FLASHINFER_MLA → TRITON_MLA → FLASHMLA_SPARSE |
| MLA, Blackwell | FLASHINFER_MLA → CUTLASS_MLA → FLASH_ATTN_MLA → FLASHMLA → TRITON_MLA → (sparse) |

The FA backend routes by version: `flash_attn.fa_utils.get_flash_attn_version()` returns FA-4 on SM100 (Blackwell), FA-3 on SM90 (Hopper), FA-2 on SM80/89. Operators override with `--attention-backend=...`; backends validate via `validate_configuration(...)`.

#### SGLang

SGLang's selector lives in `srt/layers/attention/attention_registry.py` (`register_attention_backend` + `ATTENTION_BACKENDS`), with defaults wired through `server_args.py::_handle_attention_backend_compatibility` and `model_executor/model_runner.py::init_attention_backend`. As of the May 2026 HEAD:

- **Hopper (H100 / H200).** MHA and MLA default to FA-3.
- **Blackwell (B200).** MHA defaults to TRTLLM-MHA; MLA defaults to FlashInfer-MLA, with TRTLLM-MLA preferred for DeepSeek-V3 paths; multi-modal attention defaults to FA-4.
- **Other.** FlashInfer preferred; Triton as fallback.

The roster spans FlashInfer, FA-3, FA-4, Triton, Torch SDPA, FlexAttention, TRTLLM-MHA, Dual-Chunk-FA, AITER, Wave, Ascend, Intel XPU. MLA family: FlashInfer-MLA, FlashMLA, Cutlass-MLA, TRTLLM-MLA, FA-3, Triton, FA-4, Ascend-MLA — the most-developed MLA family in any production engine. Specialized: GDN (linear attention), NSA (DeepSeek sparse via `NativeSparseAttnBackend`). SGLang's `HybridAttnBackend` is a distinguishing feature: prefill uses one backend (e.g. FlashInfer) and decode uses another (e.g. FA-3) when kernel performance differs at the two query-length regimes — a per-batch dispatch leveraging the prefill/decode asymmetry [see §00/02](../00-foundations/02-transformer-arithmetic-roofline.md). Context-parallel prefill (`is_nsa_enable_prefill_cp`, `nsa_cp_round_robin_split_q_seqs`) is the kernel-side enablement for DeepSeek-V3.2 long-context serving.

#### TensorRT-LLM

TRT-LLM ships **`trtllm-gen`** pre-compiled FMHA cubins per architecture: SM100 cubins use `tcgen05` and TMEM; SM120/SM121 cubins use `mma.sync.aligned.block_scale`. The MQA/GQA-specialized decode kernel is **XQA** ([XQA](../papers.md#xqa)) — 2.4× Llama-70B at iso-latency over the prior MMHA kernel; a Hopper QGMMA path was added in 2025. The vendor-neutral fallback is **cuDNN 9.13+ fused SDPA** ([FA2-cuDNN](../papers.md#fa2-cudnn)) using an FA-2 algorithm baseline; FA-4-style ideas are reportedly back-ported in cuDNN 9.21's generic-GEMM-fusion path. TRT-LLM-Gen-derived kernels are increasingly released through FlashInfer for vLLM / SGLang reuse. TRT-LLM's MLA path uses **FlashMLA** + the broader family (FlashInfer-MLA, Cutlass-MLA, TRTLLM-MLA).

#### Cross-engine convergence summary

| Workload | Hopper default | Blackwell default | AMD MI300X |
|---|---|---|---|
| Dense BF16/FP16 prefill | FA-3 (vLLM, SGLang) | FA-4 (vLLM); TRTLLM-MHA (SGLang); trtllm-gen (TRT-LLM) | AITER / Triton |
| Dense decode | FA-3 + paged-decode (vLLM, SGLang); XQA (TRT-LLM) | FA-4 (vLLM); TRTLLM-MHA (SGLang) | AITER / Wave / Triton |
| MLA decode (DeepSeek) | FlashMLA (universal) | FlashInfer-MLA / TRTLLM-MLA / Cutlass-MLA | AITER MLA |
| DSA (DeepSeek-V3.2) | FlashMLA sparse (vLLM `FLASHMLA_SPARSE`, SGLang `NSA`) | Same | Pending |
| FP8 / NVFP4 attention | FA-3 FP8 | FA-4, SageAttention-3 | Pending |

Engines have converged on a shared NVIDIA-side kernel collection (FlashInfer plus FA-N and TRT-LLM-Gen kernels piped through it), differ in per-architecture defaults and dispatch order, and run a parallel ROCm-side track (AITER, Wave, Composable Kernel) when the workload lands on AMD silicon.

### 4.3 CUDA-graph interaction

CUDA graphs require fixed launch shapes; attention shapes are dynamic (per-batch metadata — block table, slot mapping, sequence lengths — changes every iteration). Two implications:

1. **Attention is typically *not* CUDA-graph-captured directly.** vLLM's piecewise modes (`PIECEWISE`, `FULL_AND_PIECEWISE`) leave attention eager and capture the rest. SGLang has three runners: full-decode, piecewise, and "breakable" piecewise (falls back to eager for specific ops without re-capture). The dispatcher picks at runtime based on `max_query_len` (1 or `1+num_spec_tokens` ⇒ uniform decode, eligible for full capture).
2. **FlashInfer's inspector-executor is CUDA-graph-friendly.** The inspector emits a fixed-sized *plan tensor* that the executor consumes; launch shape is fixed even though work content varies. Per-batch metadata is a tensor argument, not a launch-shape variable.

Full development — `CUDAGraphMode` enum, capture cost model, piecewise compilation, megakernels — is in [§10/08-cuda-graphs-compilation](../10-engine-core/08-cuda-graphs-compilation.md). The kernel-side implication: attention is the operator that constrains how aggressive CUDA-graph strategy can be, and FlashInfer's design is what makes the cost manageable.

---

## Current production state

As of May 2026, the production attention-kernel landscape has settled into a recognizable shape. On the algorithm side, the FlashAttention lineage continues to define the dense path: **FA-3 is the universal default on Hopper** (H100, H200), powering the median production token in 2026, and **FA-4 is the FA-family default on Blackwell** (B200, GB200, GB300 NVL72) where it has shipped, with vendor-reported 1605 TFLOP/s BF16 and ~71% utilization that has not yet been independently replicated outside co-author-affiliated channels. **FlashDecoding** (split-K decode) is the canonical decode-side parallelism strategy across all engines and all architectures. **FlashMLA** is the canonical MLA decode kernel and powers DeepSeek-V3 / V3.2 serving in vLLM, SGLang, and TRT-LLM; **DSA** kernels (in FlashMLA's sparse path) ship with day-0 support in vLLM and SGLang for DeepSeek-V3.2.

On the kernel-collection side, **FlashInfer is the dominant cross-engine NVIDIA-side kernel collection**: integrated into vLLM (`FLASHINFER` and `FLASHINFER_MLA`), SGLang (`FlashInferAttnBackend`, `FlashInferMLAAttnBackend`, FlashInfer-MoE dispatchers), MLC, and increasingly TRT-LLM, with NVIDIA stewardship widely reported but not formally documented. Its BSR-unification of paged, ragged, radix-tree, and tree-mask layouts under one kernel template family is the design that scales to the heterogeneous request mixes of production. The parallel ROCm-side track is led by **AITER** and **Wave**, integrated into both vLLM and SGLang for AMD MI300X / MI355X. Above the kernels, four DSLs compete: **Triton** (vLLM Triton backend's default on AMD; the universal fallback elsewhere), **CuTe-DSL** (FA-4's foundation; Blackwell-targeted), **ThunderKittens** (Cursor Composer training, Together AI inference), and **TileLang** (FlashQLA, TileKernels). The consolidation around FlashInfer for kernels and the four-way DSL competition above it is the structural shape of the field.

Per-engine selection logic distinguishes vLLM's FA-4-on-Blackwell / FA-3-on-Hopper / Triton-on-AMD cascade, SGLang's FA-3-on-Hopper / TRTLLM-MHA-and-FlashInfer-MLA-on-Blackwell defaults with FA-4 on the multi-modal path, and TRT-LLM's `trtllm-gen` cubins plus XQA decode and cuDNN fused-SDPA fallback. None of the three engines defaults to a single kernel; all three implement explicit hardware-aware backend selection, and the cross-engine convergence is on the *kernel collection* (FlashInfer) rather than the *kernel choice*. The forward direction continues toward higher-level DSLs, automatic warp specialization (Tawa and similar compiler work), megakernels, and LLM-agent-written kernels (the MLSys 2026 FlashInfer Kernel Generation Contest); the FA-3-on-Hopper installed base will continue to dominate the production token mix for the foreseeable future, even as the frontier moves to FA-4 and beyond on Blackwell.
