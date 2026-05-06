# LoRA Serving and Multi-Tenant Adapters

**After reading this chapter, the reader will be able to:**

- Reason about the *shape* of a low-rank adapter — LoRA, DoRA, VeRA — at the level required to predict its inference cost, decide whether to merge or keep it unmerged, and tell which kernel path a serving engine will take for a given batch.
- Trace the kernel lineage from Punica's BGMV/SGMV through S-LoRA's Unified Paging to vLLM V1's Triton shrink-and-expand and SGLang's chunked SGMV (csgmv), and explain why a "two skinny matmuls" workload turned into a multi-year systems problem.
- Place the past-twelve-months work — heterogeneous-rank batching, joint LoRA + KV memory management, predictive cold-start, zero-overhead fused adapter forms — onto a single design-space map and read which engines are converging on which patterns.

A serving operator who supports many customers does not, in 2026, build many full fine-tunes. They build one base model and per-tenant adapters. The adapter is small enough to ship over the wire, cheap enough to train on consumer hardware, and structurally compatible with the base model. The economic argument is decisive: $r=8$ adapters on a 70B base add ~0.1% to parameter count; HBM holds one base, the adapters page in on demand, and a single replica serves hundreds to thousands of tenants. Everything in this chapter is a consequence of pushing on that economic argument until the kernels, the scheduler, and the memory allocator all bend to fit.

## 1. The shape of an adapter

### 1.1 LoRA

LoRA [`[LoRA]`, Hu et al. 2021] decomposes the *update* to a frozen weight $W_0 \in \mathbb{R}^{d \times k}$ as a low-rank product:

$$
\Delta W = B A, \qquad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d,k).
$$

At inference, the effective weight is $W_{\text{eff}} = W_0 + \alpha B A$, with scaling $s = \alpha / r$ keeping training dynamics rank-invariant. The forward pass on input $x$ is

$$
y = W_0 x + s \cdot B(A x).
$$

The adapter adds $r(d+k)$ parameters versus $dk$ for the full matrix — for $d=k=4096$, $r=8$, the ratio is $\approx r/d \approx 0.2\%$. Practical ranks span $r \in \{4, 8, 16, 32, 64\}$. Higher rank improves quality on downstream tasks at sub-linear marginal returns; production deployments cluster at $r=8$ and $r=16$. The adapter is typically attached at the four attention projections (q, k, v, o) and the three MLP projections (gate, up, down) — seven matmul sites per transformer block — though minimal recipes attach only q/v.

### 1.2 DoRA and VeRA

DoRA [`[DoRA]`, Liu et al. 2024] decomposes each weight matrix into a magnitude vector $m \in \mathbb{R}^k$ and a unit-norm direction matrix $V$, then applies LoRA to the direction only:

$$
W = m \cdot \frac{V + B A}{\|V + B A\|_c}, \qquad \|\cdot\|_c \text{ = column-wise norm.}
$$

The reported quality gains over LoRA at matched rank are real but modest, and the per-step inference cost is the same as LoRA's once the magnitude vector is folded into the base GEMM scaling. From a serving perspective, DoRA is *LoRA-shaped*: the same kernels apply, with one extra elementwise scale.

VeRA [`[VeRA]`, Kopiczko et al. 2024] takes the opposite direction. The matrices $A$ and $B$ are *frozen and shared across all adapters* (drawn from the same random seed); only two scaling vectors $b \in \mathbb{R}^d$ and $d \in \mathbb{R}^r$ are trained per adapter:

$$
\Delta W = \mathrm{diag}(b) \, B \, \mathrm{diag}(d) \, A.
$$

Per-adapter storage drops by roughly a factor of $r$, which matters when serving thousands of adapters concurrently and the adapter pool itself becomes a memory line item. VeRA is a parameter-efficiency story; the kernel shape is still the same two skinny matmuls, with diagonal scalings absorbed into the inner loop.

For the rest of the chapter, "LoRA" stands in for the full family unless a specific variant matters. The serving infrastructure does not, today, distinguish LoRA from DoRA at the kernel level, and no production engine ships VeRA-specific batching kernels — adapter heterogeneity is solved at the *rank* axis, not the parameterization axis.

### 1.3 Mergeable vs. unmerged

Given $W_0$, $A$, $B$ for a single adapter, the merged form

$$
W_{\text{merged}} = W_0 + \alpha B A
$$

precomputes the effective weight, after which serving is identical to a base-only model — zero runtime overhead, zero kernel changes. This is the path LoRA's original paper takes when claiming "no inference latency cost." It is unworkable for multi-tenant serving: with $N$ adapters and $L$ attached layers, the merged form needs $N$ copies of every base-model matrix in HBM, which destroys the very economics that made adapters interesting.

The unmerged form keeps $W_0$ once in HBM and applies $B(Ax)$ at runtime:

$$
y = W_0 x + s \cdot B(A x).
$$

The two extra matmuls are small — $A x$ is $r \cdot d$ FLOPs, $B(\cdot)$ is $r \cdot k$ FLOPs, so the adapter contributes $r(d+k)/(dk) \approx r/d$ extra FLOPs per attached layer, well under 1% for typical ranks. *All requests sharing the same base can share* $W_0$ *in HBM*, which is the entire point. The per-token math is therefore cheap; the difficulty is structural.

## 2. The 20–50% latency paradox

A naive multi-tenant LoRA implementation — "for each request in the batch, do a small extra matmul against its adapter" — measures 20–50% higher decode latency than the base model in published benchmarks [`[Punica]`, `[S-LoRA]`], even though the FLOP overhead is sub-percent. The arithmetic and the wall-clock disagree by two orders of magnitude. Three structural facts explain the gap.

**Continuous batching is broken at the per-request adapter boundary.** Continuous batching [see [§10/03-batching-scheduling](../10-engine-core/03-batching-scheduling.md)] turns a heterogeneous mix of running requests into a single fused matmul each step, by stacking their hidden states and calling one large GEMM against the shared $W_0$. The instant each request needs a *different* $A_i, B_i$, that single GEMM splinters into $B$ tiny GEMMs — one per request. Each tiny GEMM falls below the Tensor-Core minimum tile size (typically 128×128 for the M dimension on Hopper), launches its own kernel, and serializes against the GPU's launch latency. A decode step that should have been one 32-microsecond kernel becomes thirty-two 4-microsecond kernels, dominated by launch overhead, not arithmetic.

**Memory traffic for the adapters is not free.** The activation $x$ is read once for the base GEMM and again for the shrink. The adapter weights $A_i$ for $B$ distinct requests stream through L2 with no reuse if the adapters are all different. Decode is bandwidth-bound to begin with [see [§00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md)]; adding $B \cdot r \cdot d$ bytes of adapter reads on top of the base weight stream tightens the bandwidth budget at the moment it is already binding.

**The "no overhead" claim assumed merge.** LoRA's original 0%-overhead number assumed merged inference. The unmerged path was never the point of the paper; the multi-tenant serving community inherited the parameterization but had to invent the kernels.

The remedy, in three lineage steps, is the BGMV / SGMV / Triton shrink-and-expand work that occupies the next section.

## 3. Kernel lineage: Punica → S-LoRA → vLLM V1 / SGLang csgmv

### 3.1 Punica: BGMV and SGMV

Punica [`[Punica]`, Chen et al. 2023] introduced the first kernels that batch heterogeneous-adapter requests in a single launch. The key idea is to *gather* the adapter weight inside the inner loop rather than materialize a per-request copy.

For a batch of $B$ tokens (decode-shaped, one token per request), with adapter index $\mathrm{idx}[i] \in \{0, \dots, N-1\}$ and a stacked adapter tensor $\mathcal{A} \in \mathbb{R}^{N \times d \times r}$:

$$
\mathrm{out}[i, :] = s \cdot x[i, :] \cdot \mathcal{A}[\mathrm{idx}[i], :, :].
$$

The Batched Gather Matmul-Vector (BGMV) kernel walks the $i$ axis, gathers the row-block $\mathcal{A}[\mathrm{idx}[i]]$ into registers, and emits one fused launch over the whole batch. A full LoRA addon is two BGMV calls — the *shrink* $d \to r$ and the *expand* $r \to d$. BGMV is decode-shaped: every request contributes one token, and the gather pattern is essentially a permutation-with-broadcast.

For prefill, where multiple tokens of the same request use the same adapter, the gather is wasteful. The Segmented Gather Matmul-Vector (SGMV) kernel groups the batch into segments — contiguous spans where every token uses adapter $a_j$ — and runs a regular GEMM per segment:

$$
\mathrm{out}_{[\,\mathrm{start}_j : \mathrm{end}_j\,]} = s \cdot X_{[\,\mathrm{start}_j : \mathrm{end}_j\,]} \cdot \mathcal{A}[a_j].
$$

Each segment recovers full Tensor-Core utilization on its inner GEMM; the gather happens only at segment boundaries. Prefill batches with a small number of unique adapters, common in chat workloads where many tokens of a system prompt share an adapter, run at near-base-model throughput.

Punica's CUTLASS-implemented BGMV/SGMV reported up to 12× throughput over the prior state of the art at the time. The kernels assumed *uniform rank* across the batch and *uniform attached layer set* across adapters — assumptions the next generation had to relax.

### 3.2 S-LoRA: Unified Paging

S-LoRA [`[S-LoRA]`, Sheng et al. 2023] generalized Punica along three axes. First, *heterogeneous ranks*: the kernel rounds each segment's rank up to a fixed bucket size (typically 8, 16, 32, 64) and pads, so a batch can contain $r=8$ and $r=64$ adapters in the same launch. Second, *tensor-parallel adapter sharding*: the adapter weights are split across TP ranks the same way the base weights are, so multi-GPU inference does not centralize the adapter on one rank. Third, and most consequentially for the rest of the field, *Unified Paging*: the GPU memory pool that holds the KV cache also holds adapter weights, with the same paging mechanism.

Unified Paging is the conceptual unlock. Before S-LoRA, KV memory and adapter memory were separate allocators with separate eviction policies; either could OOM while the other had headroom. S-LoRA makes them one resource pool indexed by the same block-table machinery that PagedAttention introduced [see [§10/02-paged-kv-memory](../10-engine-core/02-paged-kv-memory.md)]. Adapters become "another tenant" of the block pool; eviction is global; loading and unloading respects the same fragmentation analysis.

S-LoRA reports up to 4× throughput over a naive vLLM-then-prevailing baseline and serves orders of magnitude more concurrent adapters than the merged-form alternative.

### 3.3 vLLM V1: Triton shrink-and-expand

vLLM V1, the architecture that became default in 2025 [see [§80/01-vllm](../80-oss-deep-dives/01-vllm.md)], retired the SGMV/BGMV CUTLASS kernels and replaced them with tuned Triton kernels for the shrink and expand steps. The relevant code lives in `vllm/lora/punica_wrapper/` (the package name preserves the lineage even though the kernels do not) and `vllm/lora/ops/triton_ops/lora_shrink_op.py`, `lora_expand_op.py`. The `PunicaWrapperGPU` class dispatches `add_shrink`, `add_expand`, `add_lora_embedding`, `add_lora_logits`, `add_lora_fused_moe`, and the gate/up/down variants for MLP attachment points.

Triton's flexibility wins on three dimensions. Mixed-rank batches are natural: the same kernel covers $r=8$ and $r=64$ without the bucket-and-pad gymnastics of CUTLASS templates. AMD ROCm support is a kernel rewrite, not a vendor library port. And LoRA-on-MoE — adapters that target *expert* weights — composes with the fused MoE GEMM through `add_lora_fused_moe` and the `lora_experts_mixin.py` machinery in the FusedMoE path. The trade is some peak throughput on Hopper for fixed-rank workloads; the V1 maintainers judged the flexibility worth it, and the `PunicaWrapperBase` interface preserves the BGMV/SGMV vocabulary even though the implementations are now Triton.

### 3.4 SGLang: csgmv (chunked SGMV)

SGLang's [see [§80/02-sglang](../80-oss-deep-dives/02-sglang.md)] LoRA stack is built around a chunked SGMV kernel, written lowercase as **csgmv** in the codebase to match the file naming (`lora/triton_ops/chunked_sgmv_*`, `lora/backend/chunked_backend.py`). The naming is intentional and is preserved here. The chunking insight is that the largest segments in a real prefill batch are often too small to saturate a Hopper SM but too large to fit in registers without spilling; cutting each segment into chunks of `max_lora_chunk_size` (default 16) tokens lets the kernel pipeline shrink, expand, and writeback within an SM's resident tile budget. SGLang's docs claim a 20–80% latency improvement over the basic Triton path under high concurrency [`[SGLang-LoRA-docs]`, vendor-reported, workload-dependent]; the underlying mechanism is recovering the Tensor-Core utilization that small heterogeneous segments would otherwise leave on the floor.

SGLang also exposes LoRA-specific scheduling primitives that vLLM does not. `LoRADrainer` (`lora/lora_drainer.py`) is a fairness mechanism: when an adapter is being drained, no new requests for it enter the running batch; existing requests finish; then the adapter is evicted. This bounds the tail latency of adapter churn under thrashing workloads. `LoRAOverlapLoader` (`lora/lora_overlap_loader.py`) overlaps adapter weight transfer with the base-model forward pass to hide host-to-device PCIe time. The scheduler treats `running_loras` as a first-class set and guards admission with `Scheduler._can_schedule_lora_req` against `max_loras_per_batch`.

The csgmv-and-overlap-loader combination is, at this writing, the highest-throughput open-source multi-LoRA path under heavy adapter heterogeneity; the V1 Triton path is the most mature and ships LoRA-on-MoE.

### 3.5 Cost model

For a batch with $T$ total tokens, $S$ unique adapters, average rank $\bar r$, and $L$ LoRA-attached layers:

$$
\mathrm{FLOPs}_{\mathrm{LoRA}} \approx 2 \cdot T \cdot \bar r (d + k) \cdot L,
$$

versus base-model FLOPs $\approx 2 \cdot T \cdot d k \cdot L$. The ratio is $\bar r (d+k) / (dk) \approx \bar r / d$, which for $\bar r = 8, d = 4096$ is $\approx 0.2\%$. The byte-traffic side is the binding one:

$$
\mathrm{Bytes}_{\mathrm{LoRA}} \approx \underbrace{S \cdot \bar r (d+k) \cdot L \cdot \mathrm{dtype}}_{\text{adapter weights streamed}} + \underbrace{2 \cdot T \cdot \bar r \cdot L \cdot \mathrm{dtype}}_{\text{intermediate activations}}.
$$

The first term grows with the number of *distinct* adapters in the batch; the second with the number of tokens. SGMV's segmenting amortizes the first term over segment size — the adapter weight is loaded once per segment, then reused across $T_j$ tokens — driving arithmetic intensity from $O(1)$ in the BGMV decode case to $O(T_j)$ in the SGMV prefill case. This is the structural reason why prefill LoRA is cheap and decode LoRA is expensive: in prefill, segments are long; in decode, every request is a segment of one.

## 4. Tiered adapter cache and cold-start

A production multi-tenant deployment cannot keep every adapter resident in HBM. A 70B base running with $r=16$, seven attach points, FP16 adapters is $\sim 60$ MB per adapter; an H100 with $\sim 50$ GB of headroom after weights and KV could hold roughly 800. Real catalogs run to tens of thousands of adapters, with a long-tailed access distribution.

The standard architecture is a three-tier cache:

```
HBM   (hot)    ← active in batch
DRAM  (warm)   ← loaded recently, evicted from HBM
Object store  ← cold, on demand
```

LRU eviction governs the boundary between tiers; engine-specific knobs (`max_loras` for HBM, `max_cpu_loras` for DRAM in vLLM; `max_loras_per_batch` and a separate eviction policy in SGLang's `LoRAMemoryPool`) cap each tier's residency. TensorRT-LLM exposes a two-level cache through `PeftCacheManager` with `task_id`-keyed LRU, separate from the KV cache pool. NVIDIA NIM packages the same idea into a "swarm of LoRAs" microservice: one base-model NIM, many adapters indexed by name, dynamic load via `POST /v1/load_lora_adapter`.

**Cold-start latency.** The first request to touch a cold adapter pays the loading cost. For an adapter of size $M_a$ over a host-to-device link of effective bandwidth $B_{H2D}$:

$$
T_{\text{load}} \approx \frac{M_a}{B_{H2D}}.
$$

For a 60 MB Llama-3-8B LoRA over PCIe Gen5 (~50 GB/s effective), the bound is ~1.2 ms — small relative to a prefill but visible at the tail when an entire request's TTFT lands on a single cold-load synchronization. Worse, the latency is *bursty*: an idle adapter that suddenly receives a flurry of requests pays the load once, but every queued request behind that load waits.

**Predictive prefetch.** P-LoRA [`[P-LoRA]`, 2025] adds an LSTM traffic predictor on top of a page-based adapter memory manager: hot adapters are prefetched into HBM before the first request lands. With prefetch hit-rate $h$, expected cold-start drops to $(1-h) \cdot T_{\text{load}}$. Reported gains are 1.52× throughput over S-LoRA and 35% TTFT reduction at the tail [vendor-reported, workload-dependent].

**ServerlessLoRA** [`[ServerlessLoRA]`, 2025] targets the serverless LLM-API regime, where individual function instances would otherwise re-load the *base* model along with the adapter on every cold start. The system shares a backbone model across isolated functions through a secured-memory protocol, removing ~99% of the duplicated weight transfer; the adapter still needs paging in, but the dominant cold-start cost — the base — disappears. Reported numbers are up to 86% TTFT reduction and 89% cost reduction [vendor-reported, serverless-regime-specific].

**CPU-assisted prefill.** CaraServe [`[CaraServe]`, 2024] and its follow-up Toppings [`[Toppings]`, USENIX ATC 2025] hide cold-start latency by starting the adapter's *prefill* on the CPU while the GPU copy is in flight. The CPU computes the LoRA contribution to the first chunk of prefill at low throughput but in parallel with the host-to-device DMA; once the adapter lands on GPU, the GPU takes over for the remainder. The CPU prefill is wasted work in steady state, but it converts a hard cold-start stall into a latency-hidden ramp. Toppings extends this with a rank-aware scheduler that penalizes potential SLO violations directly in the cost function.

## 5. Heterogeneous-rank batching

S-LoRA's bucket-and-pad approach to mixed ranks is correct but not optimal: an $r=8$ adapter padded to $r=64$ wastes 8× the adapter compute. The vLLM V1 Triton kernels handle mixed ranks natively without padding, but the *scheduler* still sees a performance skew: a batch with one $r=64$ adapter and ten $r=8$ adapters runs at near $r=64$ cost because the kernel must launch the full-rank tile.

**Toppings** [`[Toppings]`, 2025] addresses this with rank-aware scheduling: the admission control prefers batches whose rank-distribution is concentrated, deferring outlier-rank requests to a later batch when peers are available. The penalty function explicitly weights potential SLO violations, so a starved high-rank request eventually wins admission even if it temporarily breaks the rank-concentration heuristic.

**LoRAServe** [`[LoRAServe]`, 2025] takes the heterogeneity problem to a distributed setting. In a multi-replica deployment, adapters are not uniformly placed — some replicas hold $r=64$ adapters, others hold $r=8$. LoRAServe's scheduler is workload-aware: it places adapters where current load and future demand suggest they will batch well, and uses GPUDirect RDMA to pull a remote adapter into a busy replica when local placement would force a cross-rank rank mix. The reported numbers — up to 2× throughput, 9× lower TTFT, 50% fewer GPUs at SLO — are vendor-reported on the paper's specific benchmark and have not been independently reproduced; the directional claim, that rank-aware placement beats rank-aware batching in distributed regimes, is consistent with the small body of follow-up work.

## 6. Joint LoRA + KV cache

S-LoRA's Unified Paging treats adapters and KV blocks as one memory pool, but its allocation policy is still local — each request gets its KV blocks first, the adapter loads next. In a regime where adapter memory is a non-trivial fraction of total HBM, the two compete: a high-concurrency adapter with hundreds of small requests wants its weights resident; the KV pool wants the same bytes for additional concurrent requests.

**FastLibra / ELORA** [`[ELORA]`, HPCA 2026] jointly optimizes the two as a single dependency-aware pool with one cost model for swap-in/swap-out. The dependency is structural: if the adapter is evicted, all requests using it must either be paused or have their adapter reloaded; if KV is evicted, only the affected request must recompute. The cost model accounts for both and produces a joint admission/eviction policy. Reported numbers — 63% TTFT reduction, 40% TPOT reduction over an unspecified S-LoRA-class baseline — are paper-internal and workload-dependent. The conceptual contribution is solid: LoRA and KV are not separable in a fully utilized engine.

**LRAgent** [`[LRAgent]`, 2025], a tree-based KV-cache + adapter co-management scheme aimed at multi-LoRA agentic workloads, makes a similar point on a different axis: agent loops re-enter the same prompts repeatedly, and prefix-cache reuse must be aware of the adapter ID to avoid leaking KV across tenants. Production engines already enforce this — vLLM hashes the LoRA ID into the prefix-cache block-key, and SGLang's `RadixKey` carries `extra_key` namespacing per LoRA — but the joint optimization across both pools is still nascent.

## 7. Zero-overhead and fused adapter forms

A separate research direction asks whether adapter overhead can be made structurally zero, rather than amortized through batching.

**zFLoRA** [`[zFLoRA]`, EMNLP 2025] designs structured LoRA variants that compile into the base model's existing matmul shapes, so the adapter is simply absorbed into the per-tile work. The reported overhead is near zero on H100 and on Samsung's Galaxy S25 NPU. The caveat is that zFLoRA assumes batch-size-1 / homogeneous-task: it is an *on-device* technique, not a multi-tenant kernel. It belongs on the design-space map as a complementary point — when the workload is one tenant per device, zFLoRA-style fusion eliminates the kernel work that BGMV/SGMV exists to amortize.

**Fused-FLoRA** [`[Fused-FLoRA]`, 2025] fuses the adapter projections directly into the adjacent base GEMM via a fused kernel. The fusion is per-shape, so it does not compose cleanly with heterogeneous batches — but for replicas dedicated to a single adapter, it removes the launch overhead entirely.

**AdaFuse** [`[AdaFuse]`, 2026] takes the fused approach further with token-level pre-gating: a custom CUDA kernel selects which adapters apply to which tokens and merges their parameters into the backbone in one pass. Reported adapter overhead drops from 250–950% (a naive multi-adapter switching baseline) to ~29% [paper-reported, on the paper's specific benchmark]. The 29% residual is within the same regime as the BGMV/SGMV approaches; the contribution is on the *dynamic adapter switching* side rather than the steady-state batch.

These three lines all answer the same architectural question — *can the kernel be the optimization rather than the obstacle?* — but they trade flexibility for performance in different proportions, and none has yet displaced the SGMV/Triton path as the production default for heterogeneous-tenant batches.

## 8. LoRA and speculative decoding

Speculative decoding [see [§10/05-speculative-decoding](../10-engine-core/05-speculative-decoding.md)] composes awkwardly with multi-tenant LoRA. The modified rejection sampling that gives spec-dec its acceptance guarantee assumes the drafter and target distributions are aligned; if the target uses a per-request adapter and the drafter does not, the drafter is effectively drawing from the *base* distribution and the target from the *adapted* one. Acceptance rates fall in proportion to the divergence the adapter introduces.

Three production paths exist. *Turbo-LoRA* [`[Turbo-LoRA]`, Predibase] jointly trains the LoRA and the draft head, so the drafter inherits the adapter; the cost is per-adapter draft training. *Per-tenant draft adapters* require a draft model with its own adapter pool, which doubles the multi-tenant kernel work. *Adapter-blind drafting* — the drafter ignores the LoRA — works when the adapter is mild (e.g., style or formatting tweaks) but degrades on adapters that change vocabulary or distribution sharply (domain-specific tuning). vLLM's V0 spec-dec + LoRA path (`PR #11966`) supports the third pattern; the V1 stack does not, as of this writing, ship a stable spec-dec + LoRA combination. The interaction is an open systems question more than an open algorithmic one.

## Current production state

The production landscape in 2026 has converged on a small number of architectural patterns. vLLM V1 ships Triton shrink-and-expand kernels (`vllm/lora/punica_wrapper/`, `lora/ops/triton_ops/`) with native mixed-rank support, LoRA-on-MoE through `add_lora_fused_moe`, and an LRU adapter cache governed by `max_loras` (HBM) and `max_cpu_loras` (DRAM). Cross-replica adapter consistency in K8s deployments remains an open RFC (#12174) — the canonical workaround is AIBrix [`[AIBrix]`, ByteDance/vLLM], which adds a Kubernetes control plane with `ModelAdapter` CRDs and LoRA-aware routing on top of vLLM. SGLang ships csgmv as the default high-concurrency backend with `LoRADrainer` for fairness and `LoRAOverlapLoader` for cold-start hiding; its `RadixKey.extra_key` machinery namespaces the prefix cache per adapter. TensorRT-LLM's `PeftCacheManager` drives a two-level adapter cache with batched-GEMM + splitK kernels and is the largest enterprise production path. NVIDIA NIM packages the swarm-of-LoRAs pattern; Cloudflare Workers AI runs Punica kernels at edge scale; LoRAX (Predibase) was the first production-grade multi-LoRA server and remains the reference open-source comparison. Frontier serverless deployments — Together AI's serverless multi-LoRA, AWS SageMaker's managed multi-tenant LoRA endpoints — sit on top of these engines with cold-start mitigations layered on. The unresolved frontiers are heterogeneous-rank scheduling under fairness constraints, joint LoRA + KV admission policy, speculative decoding under per-request adapters, and serving multi-LoRA *compositions* (LoraHub-style) without re-fragmenting the kernel work that BGMV/SGMV exists to amortize.
