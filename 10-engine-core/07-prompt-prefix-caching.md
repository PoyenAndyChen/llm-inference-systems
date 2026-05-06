# Prompt and Prefix Caching

**After reading this chapter, the reader will be able to:**

- Distinguish the three reuse modes that production systems lump under the
  label "prompt caching" — system-prompt-only reuse, automatic prefix caching
  over a token-indexed structure, and chunk-level reuse — and reason about
  when each is correct and how it scales with workload shape.
- Read the two canonical implementations side-by-side: SGLang's RadixAttention
  (a token-level radix tree with reference-counted leaves and cache-aware
  scheduling) and vLLM V1's hash-chained block table (a content-addressed
  hash chain over fixed-size blocks). Resist the common mischaracterization
  of vLLM's scheme as a Merkle tree.
- Place the surrounding research and product surface — Hydragen,
  ChunkAttention, CacheBlend, the Anthropic prompt-caching API,
  cross-instance KV reuse via LMCache and Mooncake — onto a single map and
  identify which idea is being deployed and which is still in research.

Prefix caching is the cheapest big win in modern LLM serving. It delivers
a multiplicative TTFT reduction without changing the model, the kernel,
or the hardware — it just notices that the same tokens have already been
computed and reuses the work. A workload with a 1000-token shared system
prompt and a 200-token unique suffix sees attention cost collapse from
$O((1000 + 200)^2)$ to $O(200 \cdot (1000 + 200))$, an order of magnitude
on attention alone. Reasoning, RAG, and agentic workloads in 2025–2026
made prefix caching default-on across every major engine.

## 1. Three reuse modes

The phrase "prompt cache" appears in product documentation, engine release
notes, and research papers, and refers to three distinct mechanisms with
different correctness conditions and different performance profiles.

### 1.1 System-prompt-only reuse

The simplest mode. A fixed prefix — typically a system prompt and a
tool schema — is prepended to every request a tenant issues. The KV for
this prefix is computed once and reused across all requests that share
it. Correctness conditions are tight: the prefix must be
**byte-identical** at the token level, the KV blocks must be aligned to
a block boundary (since attention kernels read whole blocks), and
per-tenant isolation prevents one tenant's KV from being served to
another's request.

This is the model behind Anthropic's `cache_control` API and similar
products [Anthropic-PC]: a user-marked break point in the prompt tells
the server where the cacheable prefix ends, and subsequent requests
hitting that prefix at exactly that position pay reduced TTFT and
reduced input billing. Internals are not public; published numbers
should be treated as an API contract, not a description of backing
storage.

### 1.2 Automatic prefix caching

The richer mode generalizes system-prompt reuse to *any* shared prefix.
A new request arrives, the engine compares its tokens against a
structure indexed by token sequences, and any matching prefix — system
prompt, RAG preamble, multi-turn conversation history, prior reasoning
prefix — contributes already-computed KV blocks. Only the unmatched
suffix is prefilled.

Two indexing structures dominate: a **radix tree** (trie) over token
sequences with leaves pointing at KV blocks [RadixAttention], and a
**hash chain** over fixed-size blocks with a hash table mapping block
hashes to KV blocks (vLLM V1). Both achieve the same goal and are
roughly equivalent in expressive power for this use case; they differ
in how cleanly they expose partial prefix information to the scheduler,
how they handle non-prefix metadata like LoRA identity, and how they
bound lookup cost. Sections 3 and 4 develop both.

Automatic prefix caching is what most engines mean today when they
ship "prefix caching": vLLM V1, SGLang, TensorRT-LLM, Dynamo, and their
LMCache and Mooncake-backed variants run it default-on.

### 1.3 Chunk-level reuse

The third mode breaks the prefix-only constraint. A prompt is
partitioned into chunks — for example, the documents in a RAG context —
and each chunk is cached independently. A request that shares *any*
chunk with a prior request reuses that chunk's KV, even if the order
or position differs. CacheBlend (Yao et al., 2024) [CacheBlend] is
the canonical paper; it addresses the correctness problem that arises
when KV blocks from different contexts are combined. Section 7 covers
this in detail.

The three modes form a containment hierarchy in expressiveness:
system-prompt-only ⊂ automatic-prefix ⊂ chunk-level. They form a
*reverse* containment in deployment maturity: system-prompt-only ships
everywhere as an API; automatic-prefix is default-on inside engines;
chunk-level lives in specialized layers and research.

## 2. The savings model

Whichever mode is deployed, the underlying arithmetic is the same.
Consider a request whose token sequence has length $n = p + s$, with $p$
tokens of shared prefix already cached and $s$ tokens of unique suffix to
prefill. The dominant cost in prefill is attention over the full token
range, quadratic in the absence of caching:

$$
T_{\text{no cache}} \sim O\!\left( (p+s)^2 \right) \cdot c_{\text{attn}} + O(p+s) \cdot c_{\text{ff}}
$$

With a cache hit on the $p$-token prefix, only the $s$ suffix queries
attend over the full $p+s$ context — the prefix's self-attention was
already done:

$$
T_{\text{prefix cache}} \sim O\!\left( s \cdot (p+s) \right) \cdot c_{\text{attn}} + O(s) \cdot c_{\text{ff}}
$$

The savings ratio in the long-prefix limit is

$$
\frac{T_{\text{no cache}}}{T_{\text{prefix cache}}} \approx \frac{(p+s)^2}{s \cdot (p+s)} = \frac{p+s}{s} \;\xrightarrow[p \gg s]{}\; \frac{p}{s}
$$

For a 16k-token RAG context with 256 tokens of unique question, the
asymptotic speedup on attention is $64\times$. End-to-end TTFT savings
are smaller because the feed-forward portion of prefill is linear in $n$
and is *not* skipped. Decode is unaffected: each new token attends over
the full KV regardless of how that KV was produced. The win is entirely
in TTFT.

A few practical adjustments. The **last token always recomputes**:
engines reserve the final token for fresh prefill so logits are
produced for sampling (vLLM enforces
`max_cache_hit_length = num_tokens − 1`). **Block alignment caps the
hit**: a prefix of length $p$ caches at most
$\lfloor p / B \rfloor \cdot B$ tokens with block size $B$. **Chunked
prefill interaction**: when a request has a partial cache hit under
chunked prefill, only the first chunk benefits from the hit in vLLM's
V1 design
[see §10/03-batching-scheduling](03-batching-scheduling.md).

## 3. RadixAttention deep dive

RadixAttention is the prefix-cache abstraction introduced by SGLang
[RadixAttention] and the most fully developed example of a token-level
radix-tree cache.

### 3.1 The data structure

The cache is a **radix tree** (compressed trie) whose path labels are
token sequences and whose leaves point at KV blocks in the device
pool. Two properties are core. **Shared structure**: the tree is
global to the engine, so requests with overlapping prefixes share
nodes — two requests sharing 1024 tokens and diverging on token 1025
traverse the same path and split at the divergence point. This is
what makes "system prompt + many users" essentially free in compute.
**Reference counting**: each node carries a `lock_ref` counter,
incremented when an in-flight request depends on it and decremented on
completion; eviction skips any node with `lock_ref > 0`, guaranteeing
a request's prefix is never evicted from under it. Eviction policy
(LRU by default) then applies over reference-zero leaves.

A schematic of three requests sharing prefixes:

```
                root
                 |
        "you are a helpful"        <- shared system prompt
                 |
              assistant.\n
              /        |       \
  "Translate"     "Summarize"   "<tool>"
       |              |              |
   "to French"     "this article"   schema...
       |              |              |
   user query A   user query B   tool call C
   [leaf, KV]     [leaf, KV]    [leaf, KV]
   lock_ref=1    lock_ref=0     lock_ref=2
```

A new request prefixed with
`"you are a helpful assistant.\nTranslate to French\n..."` walks from
the root, matches the system-prompt path, descends the
`Translate to French` branch, and reuses every node up to its
divergence point. The unmatched tail becomes a new leaf.

In SGLang (`python/sglang/srt/mem_cache/radix_cache.py`), a `TreeNode`
holds `children`, `parent`, `key: RadixKey`, `value` (a tensor of KV
indices), `lock_ref`, and a `host_value` / `host_ref_counter` pair for
the L2 host tier. The `RadixKey` packs `token_ids`, an optional
`extra_key` (namespacing LoRA adapters and sampling salts so different
adapters do not accidentally share KV), and an `is_bigram` flag used
by EAGLE-style speculative decoding.

### 3.2 Operations

Three core operations. **`match_prefix`** walks from `root_node` along
`child_key`-keyed children, calling `child.key.match(...)` at each
child to find the longest prefix that matches; a match ending inside a
stored segment splits the node via `_split_node` to expose a clean
boundary. **`insert`** walks the same way but appends rather than
splits; after the forward pass, the request's full token sequence is
inserted, with new leaves for the unmatched suffix.
**`evict`** maintains a heap of leaves with `lock_ref == 0` ordered by
the configured strategy (LRU, LFU, FIFO, MRU, FILO, priority, SLRU);
when KV memory pressure rises, leaves are popped, KV indices returned
to the allocator, and parents whose children all evicted become new
candidates. Eviction propagates upward naturally.

### 3.3 Cache-aware scheduling

Because the tree is shared with the scheduler, SGLang exposes prefix
overlap as a scheduling signal. The `SchedulePolicy` enum
(`python/sglang/srt/managers/schedule_policy.py`) supports two
cache-aware options: **`LPM` (longest-prefix-match)** sorts the
waiting queue by the size of the cache hit each request would receive,
prioritizing requests whose work is mostly cached; **`DFS_WEIGHT`**
traverses the radix tree depth-first, weighting subtrees by the number
of waiting requests beneath each node, admitting batches that maximize
cache locality. LPM downgrades to FCFS when the queue exceeds 128
requests to bound sort cost. This tight cache-scheduler coupling is
what SGLang's paper means by "co-design."

### 3.4 Variants and cluster routing

The "RadixAttention" name now refers to a *family*: `RadixCache`,
`HiRadixCache` (host and storage tiers), `SWARadixCache`
(sliding-window-attention, where nodes can be partially valid),
`MambaRadixCache` (Mamba/SSM and hybrid models, where state is
non-additive), and a 2026 `UnifiedRadixCache` consolidating the family
into one tree of `UnifiedTreeNode` objects with per-component data
(full attention, SWA, Mamba) and cascade eviction. A C++ implementation
in `cpp_radix_tree/` exists for the very-fast path where Python overhead
in the cache hot loop matters. The proliferation reflects that
attention-variant models break some simple radix-tree assumptions.

The radix tree is per-instance. To preserve hits across an instance
fleet, SGLang's Rust gateway (`sgl-model-gateway`) maintains an
*approximate* radix tree per worker in `policies/tree.rs`, used for
**cache-aware load balancing**: a request is hashed by its prefix and
routed to the worker most likely to have it cached, via consistent
hashing or power-of-two-choices over cache-fitness scores. The gateway
tree is approximate in that it does not see exact engine evictions; the
engine's KV-event stream (`BlockStored`/`BlockRemoved`) bounds the
drift. Dynamo's Smart Router and AIBrix's prefix index implement
structurally similar gateway-level caches
[see §50/01-router-gateway](../50-cluster-systems/01-router-gateway.md).

## 4. vLLM V1: hash-chained block table

vLLM V1's prefix cache is structurally different. Instead of a radix
tree, vLLM uses a **hash chain over fixed-size blocks** plus a hash table
mapping block hashes to KV blocks. The hash chain is the source of a
recurring mischaracterization in the literature.

### 4.1 What it is — and what it is not

It is **not a Merkle tree**. A Merkle tree's hash structure is
*bottom-up*: a parent's hash is the hash of its children's hashes,
propagating recursively up to a root, and exists to support
*authentication* — a verifier checks leaf membership by walking a small
path of hashes up to the root.

vLLM's structure is a **linked hash chain along the prefix dimension**.
Each block's hash is a function of the block's contents and the
*immediately preceding* block's hash:

$$
H(b_i) = h\!\left( \text{tokens}(b_i),\; H(b_{i-1}),\; \text{extra\_keys}(b_i) \right), \quad H(b_0) = h(\text{tokens}(b_0))
$$

This is a linear chain, not a tree. There is no internal node whose
hash summarizes a subtree; there is no recursive bottom-up
summarization. The structure is correct and efficient for prefix
lookup — two requests share a prefix iff their first $k$ block hashes
match, and "longest cached prefix?" is a sequence of $O(n/B)$
hash-table probes. But it provides no Merkle-style integrity
properties; the goal is fast lookup, not cryptographic verification,
and naming it after Merkle is a category error that some early
documentation propagated.

### 4.2 Implementation

The block pool (`vllm/v1/core/block_pool.py::BlockPool`) holds the
physical block array, a `FreeKVCacheBlockQueue` ordered for LRU
eviction (free blocks that are also prefix-cached sit at the tail as
eviction candidates), and a `cached_block_hash_to_block` multi-map from
block hash to physical block (multi because hash collisions are
tolerated and disambiguated by full-token comparison). Hash algorithms
include `sha256` (default), `sha256_cbor` (deterministic across Python
versions), `xxhash`, and `xxhash_cbor`; the default moved to `sha256`
for cache-poisoning safety in vLLM 0.11.

The lookup path: for each waiting request, the manager computes block
hashes along the request's prefix using the chain $H(b_i)$ and probes
`cached_block_hash_to_block` block by block; the walk stops at the
first miss. Matched block ids are returned; new blocks are allocated for
the unmatched suffix plus any lookahead slots for speculative draft
tokens.

### 4.3 Correctness extensions

The hash function takes `extra_keys` to mix in:

- **`cache_salt`**, a per-tenant or per-organization salt preventing
  cross-tenant prefix sharing — two tenants whose system prompts
  tokenize identically do not collide because their salts differ.
- **LoRA id**, since requests with different active LoRA adapters
  cannot share KV.
- **Multimodal placeholder hashes**, when an image or audio token in the
  prompt corresponds to embeddings that depend on input bytes.

The same correctness machinery exists in SGLang via the `extra_key`
field on `RadixKey`; namespace separation is universal, and the choice
of structure (radix tree vs. hash chain) is orthogonal to it.

### 4.4 Cascade attention and connector composition

vLLM also exposes a kernel-side optimization that benefits
common-prefix batches. In
`GPUModelRunner._compute_cascade_attn_prefix_lens`, when a batch shares
a long common prefix, attention is split into a shared part (computed
once) and a per-request suffix part — the same insight as Hydragen (§5),
applied at the engine level on a batch with a detected common prefix.

The prefix cache lives behind the `KVConnectorBase_V1` interface so
that local misses can fall through to a host or remote tier without
changing the scheduler. A request's `get_computed_blocks` returns the
local hit length; the connector reports additional externally cached
tokens via `get_num_new_matched_tokens`. LMCache, NIXL, Mooncake, and a
CPU-offload connector all sit in this slot.

## 5. Hydragen: shared-prefix attention

Hydragen (Juravsky et al., 2024) [Hydragen] is often introduced
alongside prefix caching, but it is a *kernel-side* technique that
complements rather than replaces a cache. Some prior write-ups
describe Hydragen as a "shared-suffix" approach. **That is incorrect:
Hydragen shares the prefix.**

Consider a batch of $B$ requests sharing prefix $P$ and having
distinct suffixes $s^{(1)}, \dots, s^{(B)}$. Attention is linear in
keys and values under softmax renormalization, so attention over
$P \oplus s^{(r)}$ splits as:

$$
\text{Attn}(q^{(r)},\, K_P \oplus K_{s^{(r)}},\, V_P \oplus V_{s^{(r)}}) \;=\; \text{Combine}\!\Big( \text{Attn}(q^{(r)}, K_P, V_P),\; \text{Attn}(q^{(r)}, K_{s^{(r)}}, V_{s^{(r)}}) \Big)
$$

where `Combine` does softmax renormalization across the two parts.

The key observation: $K_P, V_P$ are the same for every $r$. Stacking
queries $\{q^{(r)}\}_r$ into a $(B, d)$ matrix turns prefix-attention
into a single batched matmul against $(K_P, V_P)$ — a GEMM rather than
$B$ separate GEMVs. Tensor cores are GEMM-shaped and underutilized on
per-request GEMVs; a single GEMM saturates them. Compute for the
prefix portion drops from $O(B \cdot |P|)$ to $O(|P|)$ plus
$O(B \cdot d)$ to combine; reported speedups reach $32\times$.

Hydragen does not replace a prefix cache — it requires one to have
produced $K_P, V_P$. Its idea is realized in production under different
names: vLLM's cascade attention (above) and SGLang's
`flashinfer.cascade.merge_state`. The original paper's standalone
implementation is not in heavy production use, but the technique is
everywhere.

## 6. ChunkAttention

ChunkAttention (Ye et al., 2024) [ChunkAttention] applies the same
shared-prefix idea at *block* granularity: a prefix tree of KV chunks
replaces per-token edges, and a two-phase attention pass processes
shared prefix chunks once across the batch and unique suffix chunks per
request, combining outputs via the same softmax-merge as Hydragen.
ChunkAttention is largely subsumed in production by PagedAttention's
block layout combined with cascade-style prefix attention; the paper
remains a clean reference for the shared-prefix kernel-side
optimization at chunk granularity.

## 7. CacheBlend: chunk-level reuse without a prefix

CacheBlend [CacheBlend] is the canonical paper for chunk-level reuse —
the third mode from §1. It targets RAG workloads where many requests
share retrieved documents but no two requests share a common *prefix*:
documents arrive in different orders, with different queries, prefixed
by different system messages.

Suppose a request's prompt is the concatenation of chunks
$C_1, \dots, C_k$, and KV for each $C_i$ is precomputed and cached.
Naively concatenating the cached KV blocks produces wrong results for
two reasons. First, **position information**: each chunk's KV was
computed at some reference position, and concatenating at new positions
(e.g., $C_i$ at offset $\sum_{j<i} |C_j|$) makes RoPE or absolute-
position embeddings inconsistent. Second, **cross-attention**: when
$C_i$ was prefilled in isolation, its tokens could only attend to
themselves; in a fresh prefill of $C_1 \oplus \dots \oplus C_k$, tokens
in $C_2$ would attend back to $C_1$, and those cross-chunk attention
contributions are missing from the cached KV.

CacheBlend's recipe is to **selectively recompute** a small fraction of
tokens at chunk boundaries — informed by the empirical observation that
attention mass concentrates on a relatively small set of high-attention
tokens, so recomputing the KV for those repairs most of the missing
cross-attention. The choice of which tokens to recompute is guided by a
fast predictive pass over the cached chunks; the recompute folds into
the prefill of the unique suffix. The end-to-end claims are
2.2–3.3$\times$ TTFT and 2.8–5$\times$ throughput on RAG workloads.
CacheBlend ships in LMCache as a serializer/deserializer plugin and is
exercised in RAG-heavy deployments that route through LMCache; it is
not the default in any general-purpose engine, because the correctness
story requires per-model tuning of which tokens to recompute. The
broader pattern of non-prefix reuse with selective recompute also
appears in Prompt Cache [Prompt-Cache] (modular reuse with a schema
language) and CacheGen [CacheGen] (network-side compression with
recompute on receive); CacheBlend is the most production-ready
realization.

## 8. The Anthropic prompt-caching API

The Anthropic API [Anthropic-PC] is the canonical example of a
*productized* prompt cache. From the user's perspective: a
`cache_control: { type: "ephemeral" }` marker can be attached to
system blocks, tool definitions, or content blocks; the cached prefix
extends through the marker. Default TTL is 5 minutes, with a 1-hour
`ttl: "1h"` extension. Pricing: 5-minute cache write at $1.25\times$
the input rate, 1-hour write at $2\times$, cache read at $0.1\times$.
Cache entries are isolated between organizations.

Internals beyond the contract are not public. The *shape* of the
abstraction matches the system-prompt-only mode in §1.1, with a
user-controlled cache boundary; the write multiplier reflects that the
first request pays for the prefill that subsequent requests are
spared.

## 9. Cross-request and cross-instance reuse

At cluster scale, two complications arise: a request may land on an
instance that does not hold its prefix (a routing problem, addressed by
the gateway-level approximate prefix-aware routing of §3.4); and even
if routed correctly, the prefix may have been evicted from the
instance's GPU pool (a tiering problem — the prefix should be offloaded
to host memory, NVMe, or a remote object store, and re-loaded on
demand).

LMCache [LMCache] and Mooncake [Mooncake] are the dominant open-source
substrates for cross-instance reuse. Both expose a KV connector
interface; the engine's prefix-cache misses fall through, the connector
checks a cluster-wide prefix store, transfers KV blocks back over RDMA
/ NIXL / TCP, and surfaces them as a "longer hit" to the scheduler.
The two systems differ in lineage — LMCache is a Python-first cache
controller with many backend connectors (Redis, S3, Mooncake Store,
EIC, file system, NIXL); Mooncake is a C++-core RDMA transfer engine
with a distributed object store layered on top — and they integrate
through each other (LMCache has a Mooncake Store connector).

Quantitative impact varies dramatically with workload shape. LMCache's
own technical report quotes "up to ~15$\times$ throughput in
shared-prefix workloads," and Mooncake's papers quote large speedups
for Kimi's production traffic; both numbers are workload-dependent and
should be read as upper bounds rather than defaults. Cluster deployment
numbers — instances sharing a prefix store, hit rates across instances —
are production-specific and not generally published in detail. The
detailed treatment of tiered offload, including the four-tier pyramid
(HBM → DRAM → SSD → remote) and the transport landscape, lives in
[see §30/02-kv-tiered-offload](../30-kv-cache/02-kv-tiered-offload.md).

## 10. Forward reference: GRPO and prefix reuse in RL post-training

A class of workload that profits unusually from prefix caching is
reinforcement-learning post-training under GRPO (Group Relative Policy
Optimization) and its descendants. In a GRPO step, the rollout engine
samples $G$ responses (typically $G = 4 \dots 64$) to the **same
prompt group**. All $G$ responses share an identical prefix — the
prompt — and only the generated suffix varies. For a prompt of length
$L_P$, naive sampling prefills it $G$ times at $G \cdot O(L_P^2)$;
with prefix caching, the prompt is prefilled once and reused $G$
times, paying $O(L_P^2) + (G-1) \cdot O(L_P)$ — the savings ratio is
essentially $G$ on the prefill. This is the central reason rollout
engines (vLLM, SGLang) ship into RL frameworks (veRL, OpenRLHF, AReaL,
slime, SkyRL, NeMo-Aligner) with prefix caching enabled by default.
Detailed treatment lives in
[see §60/06-rl-post-training-infrastructure](../60-adjacent-workloads/06-rl-post-training-infrastructure.md).

## 11. Failure modes and sharp edges

A few practical issues that recur in production:

- **Cache poisoning across tenants.** If `cache_salt` (vLLM) or
  `extra_key` (SGLang) is misconfigured, two tenants whose prompts
  tokenize identically could collide. Both engines moved to `sha256`
  defaults and explicit per-tenant salts in late 2024 / 2025.
- **MLA + chunked prefill.** Some 2025 vLLM releases required chunked
  prefill to be enabled for MLA prefix caching to function correctly —
  a release-tracker item rather than a permanent property.
- **First-chunk-only hit under chunked prefill.** vLLM's V1 chunked
  prefill admits a partial cache hit only on the first chunk of a
  request; subsequent chunks proceed as fresh prefill.
- **Block size choice.** vLLM defaults to 16; SGLang's `page_size`
  defaults to 1 with 16 and 64 also supported. Smaller blocks give
  finer alignment at the cost of more bookkeeping.
- **Prefix-cache hit rate as a metric.** vLLM exposes
  `vllm:prefix_cache_hit_rate` as a Prometheus metric; SGLang does
  similarly. Hit rates are heavily workload-dependent — chat traffic
  typically falls in a 30–70% range; RAG and reasoning workloads with
  heavy context reuse can reach above 90%. Reporting hit rates without
  specifying workload is meaningless.

## Current production state

As of mid-2026, prefix caching is default-on across every major engine
this textbook covers. vLLM V1 ships hash-chained block-level prefix
caching with `cache_salt` namespacing and the connector framework that
lets LMCache, NIXL-Mooncake, llm-d's filesystem connector, and a
CPU-offload tier sit behind the same scheduler interface. SGLang ships
RadixAttention with multiple eviction strategies, the cache-aware `LPM`
and `DFS_WEIGHT` scheduling policies, the HiCache hierarchical
extension (GA September 2025), and a Rust gateway with an approximate
radix tree for cluster-level cache-aware routing. TensorRT-LLM ships
block reuse with `enableBlockReuse=true` plus priority-based offload to
host memory. NVIDIA Dynamo's Smart Router maintains a radix tree of KV
blocks across the GPU fleet and routes by an overlap score against
that tree; AIBrix ships a structurally similar prefix index with
multi-engine support; llm-d inherits vLLM's prefix cache and adds a
POSIX-agnostic filesystem connector. The Anthropic API exposes the
productized version with cache breakpoints and ephemeral TTLs.

The kernel-side complement to caching — Hydragen-style shared-prefix
attention — is in production as cascade attention in vLLM's
`GPUModelRunner` and as `flashinfer.cascade.merge_state` in SGLang.
The chunk-level reuse path — CacheBlend — is in production inside
LMCache and in specialized RAG deployments that route through it; it is
not the default in any general-purpose engine. Cross-instance KV reuse
via LMCache and Mooncake is an enterprise deployment pattern with
substantial reported speedups for shared-prefix-heavy workloads, but the
absolute numbers are workload-conditional and cluster-deployment numbers
are largely not public.

The center of gravity in 2026 is moving toward two things. The first
is **unifying the radix-tree family**: SGLang's UnifiedRadixCache work
and vLLM V1's hybrid KV cache manager reflect a recognition that
caches for SWA, Mamba, NSA, and full attention need a single tree with
per-component data rather than parallel trees. The second is
**cluster-level cache-aware routing as a first-class scheduling
dimension**, with the Inference Gateway API standardizing the Endpoint
Picker Protocol for routing decisions that depend on per-worker
prefix-cache state
[see §50/01-router-gateway](../50-cluster-systems/01-router-gateway.md).
Prefix caching is no longer just an engine optimization; it is a
cluster-scheduling primitive.
