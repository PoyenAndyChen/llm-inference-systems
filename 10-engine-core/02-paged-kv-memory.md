# Paged KV Memory

**After reading this chapter, the reader will be able to:**

- Recover the per-token KV-cache footprint of any modern decoder-only model from first principles, and read off the memory budget — and therefore the achievable batch size at given context length — of a (model, hardware, precision) tuple from a small reference table.
- Explain how PagedAttention turns the KV cache from a contiguous per-request reservation into a pool of fixed-size physical blocks indexed by a logical-to-physical block table, what fragmentation guarantees that buys, and where the kernel cost shows up.
- Place the two competing memory abstractions — PagedAttention (custom block table, kernel-side scatter-gather) and vAttention (system virtual memory, contiguous-looking address space) — and the production-engine block managers that implement them on a single design map; in particular, distinguish the *memory-management* problem from the *kernel-side paged-attention computation* problem, which are too often conflated.

The previous chapter ([§00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md)) derived the bytes-per-token formula and showed that for long contexts the KV cache dominates memory. That arithmetic sets how much state has to live in HBM. *Where* it lives, and *how* the engine recycles those bytes across thousands of concurrent requests of unknown future length, is what this chapter is about. The original PagedAttention paper ([PagedAttention](../papers.md#pagedattention)) reported that pre-PagedAttention engines reserved roughly 60–80% more memory than they ever used; replacing contiguous reservation with paged allocation is the structural reason vLLM and its successors run batch sizes that early-2023 systems could not. Every modern engine — vLLM, SGLang, TensorRT-LLM, TGI, Dynamo — has converged on some variant of this design.

## 1. The KV math, in one table

The full derivation is in [§00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md). The reference identity, repeated here for fluency, is

$$\text{bytes/token} \;=\; 2 \cdot L \cdot H_{\text{kv}} \cdot d_h \cdot b,$$

with $L$ layers, $H_{\text{kv}}$ key/value heads (equal to $H$ for MHA, fewer for GQA, one for MQA), head dimension $d_h$, and $b$ bytes per element ($b{=}2$ at FP16/BF16, $b{=}1$ at FP8, $b{=}0.5$ at NVFP4). MLA replaces the $2 \cdot H_{\text{kv}} \cdot d_h$ term with a small latent of dimension $d_c$ plus a per-layer decoupled-RoPE component $d_h^R$, so $\text{bytes/token}_{\text{MLA}} = b \cdot L \cdot (d_c + d_h^R)$. The factor of 2 collapses because the latent is shared between K and V rather than stored separately.

A working table for capacity planning, at FP16 unless otherwise noted:

| Model | $L$ | KV layout | bytes/token | KV at $S{=}4{,}096$ | KV at $S{=}32{,}768$ | KV at $S{=}131{,}072$ |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 32 | GQA-8, $d_h{=}128$ | 131 072 | 0.50 GB | 4.0 GB | 16.0 GB |
| Llama-3.1-70B | 80 | GQA-8, $d_h{=}128$ | 327 680 | 1.25 GB | 10.0 GB | 40.0 GB |
| Llama-3.1-405B | 126 | GQA-8, $d_h{=}128$ | 516 096 | 1.97 GB | 15.8 GB | 63.0 GB |
| Mistral-7B v0.3 | 32 | GQA-8, $d_h{=}128$ | 131 072 | 0.50 GB | 4.0 GB | 16.0 GB |
| Qwen3-32B (dense) | 64 | GQA-8, $d_h{=}128$ | 262 144 | 1.00 GB | 8.0 GB | 32.0 GB |
| GPT-OSS-120B (MoE active) | 36 | GQA-8, $d_h{=}128$ + sliding-window | 147 456 | 0.56 GB | 4.5 GB | 18.0 GB |
| DeepSeek-V3 (MLA) | 61 | $d_c{=}512$, $d_h^R{=}64$ | 70 272 | 0.27 GB | 2.15 GB | 8.6 GB |
| DeepSeek-V3 (MLA, FP8) | 61 | $d_c{=}512$, $d_h^R{=}64$ | 35 136 | 0.13 GB | 1.07 GB | 4.3 GB |

Two operational facts fall out. Dense GQA-8 70B-class models put roughly 10 GB of KV in HBM per concurrent 32k-context request — half of an H100 across just eight in-flight long-context requests. MLA's per-token state is small enough that DeepSeek-V3's 61-layer cache is *cheaper per token* than Llama-3.1-8B's GQA cache. Quantization scales the table linearly: FP8 KV halves every entry, INT4 (KIVI / KVQuant) quarters them. The absolute bytes-per-token figure varies by 5–15× across the production model zoo, and a memory manager has to handle that range with one set of data structures.

## 2. Why naïve allocation fails

Pre-PagedAttention engines (FasterTransformer, early HuggingFace `text-generation-inference`, pre-V0 vLLM) allocated KV as a contiguous tensor per request, sized to a worst-case sequence length: $\text{reserved} = N_{\text{requests}} \cdot S_{\max} \cdot \text{bytes/token}$. Two failure modes follow. *Internal reservation waste*: a chat request that generates 200 tokens is billed for $S_{\max}$ — 2k, 4k, or 8k. The original PagedAttention paper measures this at 60–80% of per-request reservation on realistic traces. *External fragmentation*: each reservation must be contiguous, so freeing a finished short request leaves a hole that cannot host a longer one — the textbook fragmentation pathology, in HBM rather than on disk. The two compose. The directly observable symptom is a low `kv_cache_usage` metric while admission backpressure rejects new requests: the engine is "out of memory" while most of HBM is empty. Fixing this requires giving up on contiguous per-request allocation.

## 3. PagedAttention

PagedAttention ([Kwon et al., SOSP 2023](https://arxiv.org/abs/2309.06180), UC Berkeley / Stanford / UCSD) borrows the design directly from operating-system virtual memory. The KV state is partitioned into fixed-size **physical blocks**: each block holds $B$ tokens' worth of KV across all heads of one layer (or layer group), and physical blocks are pooled across all requests rather than reserved per request. Each request maintains a **block table**, a small array of physical block indices, that records which physical block currently backs each of its logical blocks. To attend to position $i$, the kernel looks up logical block $\lfloor i/B \rfloor$ in the block table to find the physical block, then offsets by $i \bmod B$ within that block to find the K and V vectors.

The mapping is intentionally trivial:

```
Logical               Physical
block 0  --------->   block_7    [tokens 0..15]
block 1  --------->   block_23   [tokens 16..31]
block 2  --------->   block_4    [tokens 32..47]   <- partially filled (47 % 16 == 15)
block 3  --------->   (unallocated until token 48)
```

When a request is admitted, the block manager allocates exactly enough physical blocks to hold the prompt: $\lceil L_P / B \rceil$ blocks. As decode emits tokens, additional physical blocks are allocated lazily, one at a time, when the previously-last block fills. When a request finishes, its physical blocks are returned to the global pool and become candidates for the next request.

The headline design properties:

- **Zero external fragmentation.** All physical blocks have the same size, so any free block can serve any logical position of any request. Allocation either succeeds — when a free block exists — or admission stalls; there is no case where total free memory exceeds the request's need but no contiguous run is available.
- **Bounded internal fragmentation.** The only waste is in the *last* block of each request, which is partially filled. With block size $B$ and request length $S$, wasted slots are $(B - (S \bmod B)) \bmod B$, bounded above by $B-1$ tokens. The pathological case is $S = B+1$: two blocks allocated, one slot used in the second. For $B{=}16$ and $S{=}17$, waste is 15 of 32 reserved slots — a 47% loss on that one request. For $S{=}1024$ at the same $B$, worst-case waste is 15 of 1040, under 2%. Average waste across realistic length mixes is $(B-1)/2$ tokens per request — independent of $S$.
- **Memory sharing.** Two requests with a shared prefix can point early logical-block-table entries at the same physical block. Until divergence, the prefix's KV state is computed once and read by both; on divergence, the engine performs a copy-on-write (COW) split of the divergence block. This primitive — physical pages shared across logical address spaces — is what lets paged engines support beam search, parallel sampling, and prefix caching with one mechanism.

The block table is materialized in HBM as a 2-D integer tensor of shape `[max_active_requests, max_logical_blocks_per_request]`, indexed at attention time as a per-request gather. vLLM's block table lives at `vllm/v1/worker/block_table.py`; SGLang carries an analogous structure under `RadixCache`; TensorRT-LLM's `KVCacheManager` exposes it through `setBlockOffsets`. An 8k-context request with $B{=}16$ has 512 logical blocks, so the table is 2 KB at INT32. At realistic batch sizes the whole block-table tensor fits in L2.

The cost of paged allocation is paid on the kernel side. A FlashAttention-2 kernel that assumed contiguous K and V tensors cannot run unmodified against paged KV — it has to take the block table as input and gather K/V vectors from the physical pool, an extra level of indirection inside the inner loop. **The kernel-side problem of computing attention against paged KV is a separate problem from the memory-management contribution of PagedAttention**, and the two are frequently conflated. The PagedAttention *paper* contributes the block-table abstraction, the COW-share primitive, and the fragmentation analysis. The PagedAttention *kernel* — and its descendants in FlashInfer ([FlashInfer](../papers.md#flashinfer), MLSys 2025), FA-3, FA-4, FlashMLA — is the engineering needed to make that abstraction performant. The kernel side is treated in [§10/01-attention-kernels](./01-attention-kernels.md); this chapter is about the manager.

### 3.1 Block size as a tuning knob

Block size $B$ trades fragmentation against metadata overhead and kernel efficiency. Smaller $B$ means more kernel iterations, more index-arithmetic per attended position, and a larger block-table tensor; the gather granularity wastes L1. Larger $B$ amortizes the kernel-side gather (a single TMA descriptor on Hopper or tcgen05 LD/ST on Blackwell loads a whole block at once) but raises average internal waste — which scales as $\propto B$ — and degrades sharing granularity, since prefix matches only score at $B$-token boundaries. At $B$ near the worst-case reservation, paged allocation degenerates back to the contiguous baseline.

vLLM defaults to $B{=}16$ and accepts the trade-off; SGLang's RadixCache is page-aligned at 16; TensorRT-LLM ships 32 and 64 as alternatives for prefill-bandwidth-prioritized builds. MLA kernels (FlashMLA) often use larger blocks because the per-token MLA state is smaller, so a single block holds more useful K/V mass. There is no first-principles answer to "what is the right $B$" across model sizes, contexts, and offload tiers; vLLM's 16 is a local optimum on H100/H200 for GQA-8 dense models. For mixed-attention models — Gemma 3, GPT-OSS, hybrid Mamba — vLLM V1's hybrid coordinator normalizes across multiple block sizes by computing a uniform **page size** = `block_size × num_layers_in_group × kv_hidden_size` and bundling sub-blocks per layer.

### 3.2 Two residual costs

Two residual costs deserve naming. First, when a single long-context request holds 1000+ logical blocks, the block-table tensor lookups become a non-trivial part of the per-step latency budget; vLLM and SGLang both invest in compact int32 block tables and persistent batches to keep this in check. Second, the COW divergence event — when two requests sharing a prefix block diverge mid-block — requires copying the partially-shared block before the divergent write. The cost is one memcpy of a single block per divergence, amortized across subsequent decodes on that branch.

## 4. vAttention

vAttention ([Prabhu et al., MSR India, ASPLOS'25](https://arxiv.org/abs/2405.04437)) is the serious counter-proposal. The premise: PagedAttention solves a memory-management problem by introducing an application-layer abstraction — the block table — that every kernel must respect. Modern accelerators already have memory-management hardware underneath: the GPU virtual-memory subsystem and the CUDA Virtual Memory Management (VMM) API. Why not use it?

vAttention reserves a large *virtual* address range per request — sized to the worst case — but does not back it with physical memory at admission time. Physical pages are allocated on demand and mapped into the request's virtual range as decode advances. The allocation unit is a CUDA VMM page (typically 2 MB on Hopper/Blackwell with huge-pages enabled), much larger than a 16-token block but much smaller than a full sequence reservation. The driver handles virtual→physical translation transparently; the kernel sees a flat, contiguous K and V tensor.

The advantages are kernel-side. A standard FlashAttention or FA-3 kernel — written without block-table awareness — runs unmodified on vAttention's KV. No scatter-gather, no block-table tensor, no kernel-rewrite tax when a new attention variant ships. The paper reports up to 1.97× over vLLM on long-context decode, attributable to the eliminated gather overhead. The disadvantages are operational. vAttention requires CUDA VMM support (Hopper+); the 2 MB page granularity is much coarser than 16-token blocks, so per-request internal fragmentation can exceed PagedAttention's; and prefix sharing across requests is harder — pages live in a single virtual address space, so sharing a prefix requires mapping the same physical page into multiple virtual ranges (operationally fiddly) or forgoing shared state.

The headline contrast:

| Property | PagedAttention | vAttention |
|---|---|---|
| Allocation unit | 16-token block (configurable) | CUDA VMM page (typically 2 MB) |
| Per-request layout | Logical→physical block table in HBM | Contiguous virtual address space |
| Kernel API | Block-table-aware kernel | Standard contiguous kernel |
| External fragmentation | Zero | Zero (handled by CUDA VMM) |
| Internal fragmentation per request | $\le B-1$ tokens | $\le$ (page size − unused tail) |
| Prefix sharing | Native via block-table COW | Requires explicit page sharing |
| Pre-allocation | KV pool allocated once at startup | Virtual reservation only; physical lazy |
| Runtime dependency | Block-aware attention kernel | CUDA VMM (Hopper+) |

Production adoption settled on PagedAttention for OSS engines: vLLM, SGLang, TensorRT-LLM, and TGI all ship the block-table design as default. vAttention exists in vLLM as an open RFC (`#17612`) but is not in main; it is more visible in research forks and in specialized deployments where contiguous-kernel compatibility is the binding constraint. The trade-off is ongoing: every new attention kernel that ships first as a contiguous-K/V implementation pays a port cost to become block-table-aware, and that ongoing tax is the strongest argument for vAttention's approach. The distinction is between an *application-layer* memory manager (PagedAttention: the engine decides; pays kernel cost; gets prefix sharing for free) and a *systems-layer* manager (vAttention: the driver decides; gets kernel simplicity; pays for sharing).

## 5. The vLLM V1 block manager

vLLM V1's block manager ([§80/01-vllm](../80-oss-deep-dives/01-vllm.md)) is the most-deployed PagedAttention implementation. The relevant code lives in `vllm/v1/core/`: `kv_cache_manager.py::KVCacheManager` (request-level allocation: `get_computed_blocks`, `allocate_slots`, `free_blocks`, `cache_blocks`), `block_pool.py::BlockPool` (physical-block free list, prefix-cache hash map, LRU eviction), and `kv_cache_coordinator.py` (the hybrid coordinator that handles models mixing attention types).

The KV cache itself is one or more 4-D GPU tensors of shape `(num_blocks, block_size, num_kv_heads, head_dim)`, allocated once at engine startup. An HBM-profiling pass runs a dummy forward, measures peak activation and overhead, and assigns the remainder to the KV pool. **vLLM V1 pre-allocates the full pool**; it does not grow on demand. This is the operationally significant contrast with vAttention's lazy physical-page allocation. The benefit is determinism — HBM budget is fully committed at startup, so decode-time OOM is a scheduling bug rather than a system surprise. The cost is that the pool must be sized correctly up front and cannot adapt to an unanticipated long-context request.

### 5.1 Hash-chained block IDs

V1's prefix cache is built on a content hash chained per block. The relevant invariant: each *full* (i.e., $B$-token-aligned) block carries a `BlockHash` computed as

```
BlockHash(i) = hash(BlockHash(i-1), block_tokens(i), extra_keys)
```

where `extra_keys` covers LoRA adapter ID, multimodal placeholder hashes, and an optional tenant `cache_salt`. The hash function is configurable (`sha256` default, `sha256_cbor` for cross-process determinism, `xxhash` for speed). Two requests whose first $k$ blocks all hash identically share their first $k$ blocks of KV state — the classic prefix-cache-hit case, scored at $k$-block granularity.

This is *hash chaining*, not a Merkle tree — a distinction worth making because the design often gets misdescribed as "Merkle-style." A Merkle tree organizes hashes hierarchically: internal nodes hash their children, a root authenticates the whole tree, and lookups follow inclusion proofs. V1 does none of that. It computes a linear chain of per-block hashes; each block authenticates only the prefix up to and including itself; lookups walk blocks in token order until the first miss. The structure is closer to a content-addressed linked list than a Merkle tree, and the chained construction is what makes prefix-cache lookup an $O(\text{logical-blocks})$ ordered walk rather than a tree traversal.

`KVCacheManager.get_computed_blocks` computes block hashes for the prompt up to `max_cache_hit_length = num_tokens − 1` (the last token always recomputes for logits) and probes the global `cached_block_hash_to_block` multi-map. Hits return existing physical blocks; only the suffix needs fresh allocation. As decode fills new blocks, their hashes are computed and registered. The prefix cache is treated as a free-list optimization, not a separate data structure: a cached but unreferenced block sits at the tail of the free queue and is reclaimed by LRU only under allocation pressure. The detailed treatment of prefix caching is in [§10/07-prompt-prefix-caching](./07-prompt-prefix-caching.md).

### 5.2 Allocation layout and the V1 KV connector

Per-step allocation in `KVCacheManager.allocate_slots` reserves blocks across four logical regions of a request:

```
| comp     | new_comp        | ext_comp     | new + lookahead |
^prefix-cache hit            ^connector hit  ^to compute
            ^cached but newly bumped to head
```

`comp` is the prefix-cache hit; `new_comp` is recently-evicted-but-still-recoverable cache; `ext_comp` is what an external connector can fetch from a tiered store; `new + lookahead` reserves slack for new K/V plus drafter tokens for speculative decoding (essential for EAGLE / Medusa / MTP — without it, the drafter writes into unallocated blocks).

The **KV connector V1** interface (`vllm/distributed/kv_transfer/kv_connector/v1/`) is the boundary at which the block manager talks to external KV stores. Pluggable backends include NIXL ([ai-dynamo/nixl](https://github.com/ai-dynamo/nixl)), Mooncake, LMCache, MoriIO, peer-to-peer NCCL, and a multi-connector facade; downstream projects (llm-d, AIBrix) supply more. Block hashes pass through unchanged — a LMCache-stored block fetched into a new GPU's pool retains its original cache identity. This is the integration point for HBM→DRAM→SSD→remote tiered offload; architectural treatment in [§30/02-kv-tiered-offload](../30-kv-cache/02-kv-tiered-offload.md), engine view in [§80/01-vllm](../80-oss-deep-dives/01-vllm.md).

### 5.3 Edge cases

Three sharp edges deserve naming. *MLA + chunked prefill* ([§10/03-batching-scheduling](./03-batching-scheduling.md)): the MLA kernel's attention-absorption optimization required dedicated paths in vLLM 2025 releases, and prefix caching with MLA was conditional on `enable_chunked_prefill=True` in some versions. *LoRA-aware caching*: two requests with the same prompt but different LoRA adapters must not share prefix state, so the LoRA ID is folded into the per-block hash via `extra_keys`. *Multi-tenant isolation*: production deployments include a per-tenant `cache_salt` in the hash, so tenant A's cache cannot be observed by tenant B on identical prompts. Without the salt, automatic prefix caching would be a covert side channel.

## 6. Block managers in other engines

OSS engines have converged on the paged-block design but differ in cache semantics. **SGLang** builds its prefix cache as a **token-level radix tree** ([RadixAttention](../papers.md#radixattention), NeurIPS 2024) with LRU eviction over tree nodes; the back-end physical memory is still paged blocks. The two abstractions — radix tree of *which prefixes are cached*, paged blocks of *where their KV lives* — compose: the tree is the prefix-identity index, the block pool is the storage. **TensorRT-LLM**'s `KVCacheManager` ships paged blocks with an `enableBlockReuse` flag (gated by `paged_context_fmha` for the kernel side); under pressure, cached-but-unreferenced blocks are offloaded to host memory rather than evicted, treating host as a second tier — closer to the LMCache-style tiered model than to vLLM's pure LRU. **TGI** uses a vLLM-derived paged manager and is in feature-freeze as of March 2026; HF officially recommends vLLM/SGLang for new deployments. **NVIDIA Dynamo**'s `KVBM` is the paged manager of a *cluster*, maintaining a fleet-wide block index and routing requests to GPUs whose tables already hold a maximal prefix overlap, transported through NIXL.

The convergence point is the data structure: a 2-D logical-to-physical block table, a global free-block pool, a content-addressed prefix index. The variation is in (a) the prefix-index structure (radix vs hash chain), (b) the eviction policy (pure LRU vs priority-with-host-offload), and (c) whether the manager extends to a cluster view. None of the major OSS engines ships vAttention as default.

## 7. A request, end to end

A worked trace on vLLM V1 — block size 16, a 35-token prompt with a prefix hit on the first 16 tokens, a 50-token generation — shows the pieces composing. The 35-token prompt splits into two full blocks (tokens 0..15, 16..31) and a partial block (32..34). `get_computed_blocks` hashes the two full blocks; the first hash matches a cached entry, so physical block 7 is reused; the second does not, so physical block 23 is carved from the free pool. `allocate_slots` reserves block 4 for the partial trailer plus the first decode tokens, yielding block table `[7, 23, 4]`. Prefill recomputes only tokens 16..34 — block 7's KV is reused unchanged. Once block 23 fills, its chained hash is registered, available for future requests sharing this 32-token prefix. Decode fills block 4 to position 47, then allocates a new physical block, and so on. At termination, ref-counts drop; cached-but-unreferenced blocks return to the free queue's tail and are reclaimed by LRU only under pressure. Three design properties showed up at once: cross-request prefix sharing via the block table, zero external fragmentation at allocation, and internal fragmentation bounded to the partial trailing block.

## Current production state

As of mid-2026, every major open-source LLM serving engine ships PagedAttention or a close descendant by default. vLLM V1's block manager (`vllm/v1/core/kv_cache_manager.py` + `block_pool.py`) is the most-deployed implementation; SGLang's RadixCache + paged-block backend is the closest competitor and leads the highest-throughput-at-scale leaderboards (notably DeepSeek-V3 long-context serving). TensorRT-LLM's `KVCacheManager` with `enableBlockReuse=true` and `paged_context_fmha` is the dominant enterprise-installed-base path through NVIDIA's NIM/TRT-LLM stack. TGI inherits a vLLM-style paged manager but entered feature-freeze in March 2026. Block sizes converged near 16 tokens for GQA on Hopper-class hardware; MLA-specific paths and sliding-window-interleaved models (Gemma 3, GPT-OSS) drove the V1 *hybrid* coordinator that lets one engine host multiple block-size regimes.

vAttention remains the serious counter-proposal. Its principal advantage — kernel-side compatibility with contiguous-K/V attention implementations — is most valuable when a new attention variant ships first as a contiguous kernel not yet ported to a block-table-aware form. As of mid-2026 it lives in research forks and specialized deployments, with vLLM RFC `#17612` tracking a possible main-line integration. The two designs are not mutually exclusive: future engines may combine a vAttention-style contiguous virtual layout for the active KV with a PagedAttention-style content-hashed block index for cross-request prefix sharing — kernel simplicity of one, sharing primitive of the other. No production engine has shipped that synthesis.

The cluster-level extension — KV-aware routing, tiered offload, KV connectors — has converged on a common pattern: a paged manager per engine, exposing block-hash identity through a connector ([§30/02-kv-tiered-offload](../30-kv-cache/02-kv-tiered-offload.md)), with a cluster-wide router (Dynamo Smart Router, AIBrix KV index, llm-d EPP) assigning requests to GPUs whose block tables already hold a maximal prefix. The bytes-per-token arithmetic of §1 ultimately determines how aggressive that routing has to be and how much HBM the cluster has to budget for KV; the block manager in this chapter is the local data structure those cluster decisions ride on top of.
