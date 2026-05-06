# Tiered KV Offload

**After reading this chapter, the reader will be able to:**

- Reason about the four-tier KV memory hierarchy — HBM, host DRAM, NVMe SSD, remote/cluster — in terms of bandwidth, capacity, and access latency, and decide which blocks belong in which tier for a given workload.
- Distinguish the transport layer (NIXL, Mooncake TransferEngine, Perplexity TransferEngine, UCX, GPUDirect Storage) from the KV-store layer (LMCache, Mooncake KVStore, AIBrix KVCache, SGLang HiCache, Dynamo KVBM) and explain how engines compose them.
- Read the engine-store matrix below as a system of design choices, and predict how a deployment's prompt-length, prefix-reuse, and TTFT-budget profile maps onto a particular column.

Tiered KV is the cluster-level answer to a problem that PagedAttention solved only within a single GPU. PagedAttention [see §10/02-paged-kv-memory](../10-engine-core/02-paged-kv-memory.md) gives an engine the right abstraction for KV memory — fixed-size blocks, a logical block table, copy-on-write sharing — but it does not solve capacity. A single GB200 has 192 GB of HBM; long-context reasoning, multi-tenant chat with deep prefix reuse, and RAG ingest pipelines all overflow that budget by orders of magnitude. The engine then has two choices: evict the block and lose the prefix-cache hit, or move it to a tier with more capacity. Tiered offload industrializes the second choice. It treats KV cache as a cluster-resident *storage tier* rather than a per-engine ephemeral, and asks what every storage engineer asks: which blocks belong on which medium, who pays for the transfer, and what is the consistency model.

## 1. The four-tier hierarchy

The 2026 hierarchy has four named tiers. Each tier is roughly an order of magnitude slower than the one above it and roughly an order of magnitude larger.

```
                                 KV memory tiers (per-node figures)
            ┌──────────────────────────────────────────────────────────────────────┐
   HBM      │ on-package GPU memory                                                │
   T0       │   H100 SXM5  : 80 GB     @ 3.35 TB/s    (~1–5 µs to read a block)    │
            │   H200 SXM5  : 141 GB    @ 4.8  TB/s                                 │
            │   B200 SXM   : 192 GB    @ ~8   TB/s                                 │
            │   GB300      : 288 GB    @ 8    TB/s                                 │
            └──────────────────────────────────────────────────────────────────────┘
                       ↕ PCIe Gen5 (~64 GB/s) or NVLink-C2C (Grace-Hopper)
            ┌──────────────────────────────────────────────────────────────────────┐
   DRAM     │ host-side pinned memory                                              │
   T1       │   per node   : 0.5–2 TB  @ 50–100 GB/s aggregate                     │
            │   block read : ~50–500 µs depending on PCIe contention               │
            └──────────────────────────────────────────────────────────────────────┘
                       ↕ PCIe Gen5 NVMe (~7 GB/s/drive seq)
            ┌──────────────────────────────────────────────────────────────────────┐
   SSD      │ NVMe (local or NVMe-oF)                                              │
   T2       │   per node   : 4–32 TB   @ ~7 GB/s seq read per drive (~30 GB/s with │
            │                          GDS aggregating 4 drives)                  │
            │   block read : ~100–500 µs (GDS); 1–5 ms (CPU-mediated)              │
            └──────────────────────────────────────────────────────────────────────┘
                       ↕ IB NDR / Spectrum-X (per-link ~100 GB/s, multi-link ~400 GB/s)
            ┌──────────────────────────────────────────────────────────────────────┐
   Remote   │ cluster-attached KV / object store / shared filesystem               │
   T3       │   capacity   : petabyte-scale, shared across rack/cluster            │
            │   bandwidth  : ~400 GB/s sustained (4-link NDR); ~50 GB/s TCP        │
            │   block read : 200 µs – 5 ms; cold-tier (S3) at 50–500 ms            │
            └──────────────────────────────────────────────────────────────────────┘
```

The HBM and DRAM rows match standard datasheets. The SSD row reflects measured GPUDirect Storage (GDS) performance with a few NVMe drives behind a single GPU; CPU-mediated access is two to ten times slower. The Remote row is variable: a co-located rack with 4×NDR-400 InfiniBand NICs per host saturates near 400 GB/s aggregate, while TCP-fallback or cross-DC paths can be 10× slower.

The latency-vs-throughput shape that motivates the hierarchy is

$$t_{\text{read}}(\text{block}) \approx t_{\text{setup}} + \frac{B \cdot 2 L H_{kv} d_h b}{W_{\text{tier}}}$$

with block size $B$ tokens, layer count $L$, KV heads $H_{kv}$, head dim $d_h$, bytes-per-element $b$, and tier bandwidth $W_{\text{tier}}$. For a single 16-token Llama-3-70B (GQA-8) block of 5 MB, the formula gives roughly $1.5\,\mu\text{s}$ in HBM, $50\text{–}100\,\mu\text{s}$ over PCIe to DRAM, $700\,\mu\text{s}$ from a single NVMe drive, and $12\,\mu\text{s}$ from a 400 GB/s RDMA fabric — *if* the setup cost is hidden. Setup is what makes this a queueing problem rather than a bandwidth problem: a NIXL `XferReq` registration, a Mooncake `PutStart` round-trip, or an RDMA `READ` work-request post each add tens of microseconds that batching amortizes but a single small block does not.

Two practical implications follow. **Tier selection is workload-dependent**: a long-prefix RAG workload with multi-megabyte prefixes hides setup cost easily and runs near the bandwidth ceiling, while a chat workload with kilobyte block reads is dominated by setup latency. **The boundary between T2 and T3 is moving**: cluster-attached NVMe (CMX, hf3fs, WEKA) plus RDMA collapses what used to be two tiers into a single petabyte-scale persistent KV layer.

## 2. What goes where

Within the four-tier pyramid the placement question reduces to three ordered policies that every production KV store implements with minor variations.

**Active KV stays in HBM.** Any block in the attention compute path of an in-flight request is pinned in HBM. FlashAttention, FA-3, FA-4, FlashInfer's BSR-paged kernels, FlashMLA, and AMD AITER all assume HBM-resident KV [see §10/01-attention-kernels](../10-engine-core/01-attention-kernels.md). A tiered KV system never relaxes this assumption; it stages blocks into HBM before the kernel runs.

**Recently-evicted blocks go to T1 (DRAM).** When the engine's block manager evicts under memory pressure, the offload path copies to host pinned memory. T1 is the staging buffer for everything below it: vLLM's KV connector, LMCache's `LocalCPUBackend`, SGLang HiCache's L2, AIBrix's L1 DRAM, Dynamo KVBM's G2, and llm-d's tiered prefix cache all use this shape. Pinned DRAM is the only memory the NIC can DMA from for RDMA writes without an extra copy, which is why every architecture treats T1 as the *mandatory* staging layer.

**Prefix-cache history goes to T2/T3.** Blocks not touched recently but still useful as prefix content move to NVMe (T2) or remote KV (T3). Three policies dominate: write-through to all tiers with TTL-driven expiration (Dynamo KVBM, AIBrix L2); write-through-selective, where a hash-distance heuristic gates the push (SGLang HiCache); write-back, where blocks spill only on capacity pressure (LMCache default). Mooncake KVStore is unusual in that the master service decides centrally rather than each engine making local choices.

**Cross-instance sharing happens at T3.** Two engines on different nodes share KV through T3 alone. The transport layer moves the bytes; the store layer keeps the directory of which blocks live where. KV-aware routing — covered in [see §50/01-router-gateway](../50-cluster-systems/01-router-gateway.md) — is the cluster-level mechanism that makes T3 sharing pay off, because if requests are not steered to the engine whose T1+T2 already holds their prefix, the savings vanish into transfer cost.

The block lifecycle traces a clean path: allocate in T0, become active, evict to T1 under pressure, demote to T2 or T3 when cold, and promote back through the tiers on a prefix hit. Mooncake, LMCache, AIBrix, Dynamo, and SGLang HiCache all walk this lifecycle; they differ in eviction policy at each tier, in how they detect a prefix hit on T3, and in whether the promotion path bypasses CPU staging.

## 3. The transport layer

KV transfer between any two tiers is mediated by a transport library. Five matter in 2026.

**NIXL — NVIDIA Inference Xfer Library.** NIXL is NVIDIA's unified abstraction for KV transfer between any tier pair. The repo is `ai-dynamo/nixl`; it ships as the Rust `nixl-sys` crate (used by Dynamo's KVBM), the `nixl-cu12` Python wheel (used by vLLM's `NixlConnector` and SGLang's disagg path), and a C ABI for non-Python engines. NIXL exposes a `register / transfer / poll` triple over a `NixlAgent` and an `XferRequest` pool; under the hood it brokers between UCX (RDMA over IB and RoCE), GPUDirect for HBM↔HBM, GDS for HBM↔NVMe, POSIX for filesystem, and cloud-blob endpoints for S3 and Azure. Production usage runs vLLM↔vLLM PD over UCX-IB, KVBM offload to disk via POSIX or GDS, and KVBM offload to remote via the cloud-blob backend. NIXL is consumed by vLLM KV connector V1, Dynamo KVBM, SGLang's disagg path, LMCache's `nixl_storage_backend.py`, and the llm-d routing sidecar's `connector_nixlv2.go`.

**Mooncake TransferEngine.** Mooncake's peer-to-peer RDMA transport sits at a similar level but is older and has a broader transport set. Its C++ API supports IB, RoCE, eRDMA, TCP, NVLink, intranode-NVLink (`cudaIpc*`), EFA, NVMe-oF, CXL, Ascend, HIP, Kunpeng, and a multi-protocol `mp_*` family that registers one segment under several transports and selects per batch. The implementation runs one `RdmaContext` per detected NIC, stripes large transfers across NICs (vendor numbers report 87 GB/s on 4×200 Gbps RoCE and 190 GB/s on 8×400 Gbps), and reroutes around failed NICs. TransferEngine is layered over UCX in the original FAST paper but reimplements its own RDMA queue-pair management in current code. NIXL added Mooncake TransferEngine as a backend plugin in May 2025, so a stack speaking NIXL can reach Mooncake without engine changes. (Bandwidth numbers are vendor-supplied.)

**Perplexity TransferEngine.** Perplexity AI open-sourced their RDMA-native KV transfer library in 2025 alongside disclosures of their disaggregated-serving stack. RDMA-native (no TCP fallback), built for very-large-scale-out PD serving. As of mid-2026 it is not as widely adopted externally as NIXL or Mooncake TransferEngine but provides an independent reference. The same library is also used for Perplexity's RL post-training weight-broadcast path [see §60/06-rl-post-training-infrastructure](../60-adjacent-workloads/06-rl-post-training-infrastructure.md).

**UCX — Unified Communication X.** UCX is the underlying RDMA abstraction that both Mooncake TransferEngine and NIXL build on. UCX exposes tagged messages, active messages, RDMA `READ`/`WRITE`, atomic operations, and a transport-selection layer that picks IB Verbs, RoCE, CMA, or TCP. Most engines do not call UCX directly — they call NIXL or TransferEngine, which bind UCX as one transport among several.

**GPUDirect Storage (GDS).** GDS is NVIDIA's DMA path from GPU HBM directly to NVMe, bypassing CPU DRAM. It uses the `cuFile` API and an NVMe driver that supports GPU-initiated I/O. The benefit is two-fold: it removes the bounce-buffer copy through pinned DRAM, and it enables fast HBM↔SSD KV spill at NVMe-sequential rates. LMCache's `GdsBackend` and Dynamo KVBM's G3 disk pool both use GDS. For KV blocks of tens of kilobytes the setup cost dominates; for batched megabyte transfers GDS reaches roughly 7 GB/s per drive and aggregates linearly with drive count. CMX is built around GDS.

The relationship between these layers is a stack: engines call either a store layer or a transport directly; the store layer in turn calls the transport; both bottom out at UCX, IB Verbs, RoCE, or POSIX/cuFile. Store and transport cross-pollinate freely: LMCache wraps Mooncake Store as a remote backend, Mooncake registers as a NIXL backend, Dynamo KVBM uses NIXL with a POSIX backend.

## 4. The engine-store matrix

The KV-store layer is where production engines actually disagree. The matrix below collapses into one row per engine, naming the canonical store implementation, the tiers it spans, and the transport it speaks.

| Engine | KV store | Tiers | Transport |
|---|---|---|---|
| vLLM V1 | KV connector (NIXL, Mooncake, LMCache, MoriIO, p2p, hf3fs) | HBM→DRAM→remote | pluggable |
| SGLang | HiCache (hierarchical radix cache) | HBM→DRAM→SSD | NVLink, IB |
| TRT-LLM | external KV cache (Mooncake, Dynamo path) | HBM→remote | Dynamo / Mooncake |
| Dynamo | KVBM (G1–G4 tiers) | HBM→DRAM→SSD→remote | NIXL |
| AIBrix | KVCache (Redis-backed) | HBM→remote | gRPC / RDMA |
| LMCache | 8-tier storage backends | HBM→DRAM→SSD→remote | pluggable |
| Mooncake KVStore | master + worker | HBM→SSD→remote | Mooncake TE |

Three observations earn their place over the table itself. **vLLM and LMCache are both "pluggable" by design but at different granularities.** vLLM's KV connector V1 (`vllm.distributed.kv_transfer.kv_connector.v1.base.KVConnectorBase_V1`) is an interface to a single connector at a time; the engine sees one logical KV destination, and the connector hides the tiers behind it. LMCache is a multi-backend tiered library that implements that interface and exposes its own internal stack. The dominant deployment shape is therefore vLLM ↔ LMCache ↔ {DRAM, NVMe, GDS, P2P, NIXL, Redis, S3, Mooncake Store} rather than vLLM directly to a remote store. **SGLang's HiCache is the cleanest "engine owns the tiers" story.** HiRadixTree extends RadixAttention into L1 (GPU), L2 (host CPU), and L3 (Mooncake Store, hf3fs, NIXL, AIBrix KVCache, or local file). Three write policies (write-through, write-through-selective, write-back) and three KV layouts (`layer first`, `page first`, `page first direct`) are exposed as configuration; the radix structure means hits at any tier reuse the same prefix-tree path. **Dynamo's KVBM is the most NIXL-coupled design.** It splits across `lib/kvbm-physical/`, `lib/kvbm-logical/`, `lib/kvbm-engine/`, and `lib/kvbm-config/` (Rust) with G1 (device), G2 (host pinned), G3 (NVMe, optionally GDS), and G4 (remote: S3, Azure, NFS, Lustre, GPFS, Dell PowerScale, WEKA). Block identity is by `sequence_hash`, fungible across pools.

Four cross-cutting design choices span all rows:

- **Hash function.** LMCache uses vLLM's `init_none_hash` for cross-stack agreement. SGLang uses RadixAttention's token-level hash. Dynamo uses block-level `sequence_hash`. Cross-engine KV reuse only works when two engines compute the same hash, which makes the hash function the single most important interop boundary.
- **Block size.** vLLM defaults to 16 tokens, LMCache chunks at 256 (default `chunk_size`), SGLang HiCache batches at ≤128 pages of 16 tokens, Mooncake Store slices objects at ~64 MB. Block size sets transfer granularity and therefore where the setup-cost knee falls.
- **Eviction policy.** LMCache exposes a per-tier registry (LRU default, with LFU/FIFO/MRU); AIBrix L1 uses LRU/FIFO/S3FIFO; Mooncake's master uses LRU or FIFO with `with_soft_pin`/`with_hard_pin` per object; Dynamo KVBM is reference-counted with sequence-hash dedup. The Mooncake hard-pin is the most production-realistic for protecting hot system prompts.
- **Consistency.** Mooncake Store enforces strong consistency for completed objects (immutable until `Remove`); LMCache and AIBrix are best-effort per-tier. Strong consistency is what lets Mooncake's master sit out of the data path and return raw descriptors that clients pull with no further coordination — a property that matters more for cross-engine sharing than for single-engine offload.

## 5. LMCache

LMCache is a Python-first KV middleware that embeds in vLLM, SGLang, and TRT-LLM via connectors. It treats KV reuse as a *Knowledge Delivery Network*: chunks of KV, keyed by running prefix hash, are pushed and pulled across DRAM, NVMe, peer LMCache instances, and remote stores. Numbers in this section come from project documentation and the LMCache technical report (arXiv:2510.09665) and should be treated as vendor-supplied.

LMCache's eight storage backends are wired in a fixed order by the `CreateStorageBackends` factory: `PDBackend` (PD sidecar over ZMQ), `LocalCPUBackend` (always created, owns the pinned-CPU allocator and stages everything below), `P2PBackend` (peer-to-peer mesh over ZMQ + NIXL or sockets), `NixlStorageBackend` (first-class NIXL path), `LocalDiskBackend` (NVMe via async file I/O), `GdsBackend` (GPU-to-NVMe via cuFile), `MaruBackend` (vendor), and `RemoteBackend` (with connectors for Redis, Valkey, S3, Mooncake Store, InfiniStore, EIC, shared filesystem, HF Bucket, SageMaker HyperPod, and LMCache's own server protocol). The `StorageManager` runs an asyncio loop on a dedicated thread and multiplexes operations across backends in declaration order, with a `WeightedSemaphore` for chunk-budget back-pressure.

The vLLM integration (`lmcache/integration/vllm/lmcache_connector_v1.py`) is the reference pattern. `LMCacheConnectorV1Dynamic` subclasses vLLM's `KVConnectorBase_V1`. The single most important call is `LMCacheEngine.lookup`, fired synchronously inside `get_num_new_matched_tokens`, which asks how many cached prefix tokens the external store owns; the integer returned becomes vLLM's `num_external_tokens` and is used to skip prefill blocks.

Two LMCache features differentiate it on the matrix.

**CacheGen** (Liu et al., SIGCOMM'24) is a serde plugin for the remote and disk paths that compresses KV blocks for network transfer and storage. The codec applies per-layer-sensitive quantization (32 bins for early layers, 16 for later, per a `from_model_name` heuristic) — the production LMCache build pairs this with KIVI-style asymmetric per-channel quantization plus LZ4 compression for a CPU-friendly path; the original SIGCOMM paper used arithmetic coding on the residuals. The original paper reports 3.5–4.3× cache-size reduction with bandwidth-adaptive compression. CacheGen wins when the network is the bottleneck on tiers T2 and T3; on the active T0↔T1 path FP8 KV [see §30/01-kv-compression](01-kv-compression.md) wins because it has no encode/decode overhead.

**CacheBlend** reconstructs attention output from a *mix* of fresh and cached KV blocks. The setup is RAG-shaped: a request prefix is composed of cached document chunks plus a fresh user query that needs cross-attention against them. CacheBlend keeps the cached K and V blocks, prepends the fresh tokens' K and V, and selectively recomputes a small subset of cross-attention positions to repair layers where the recombination materially changes the attention output. Vendor numbers report 2.2–3.3× TTFT and 2.8–5× throughput on RAG. The engineering point is that *partial cache hits at chunk granularity become possible*: a traditional prefix cache is a binary primitive — either the full prefix is cached or none of it is — and CacheBlend converts the binary into a continuous knob.

**Multi-node KV mesh.** LMCache's `P2PBackend` plus `LMCacheClusterExecutor` form a peer mesh: a hit on a peer's CPU cache is one ZMQ round-trip plus one NIXL or socket transfer, with no Redis or S3 in between. The cache controller's wire protocol is a tagged-msgspec message family (`LookupMsg`, `MoveMsg`, `CompressMsg`, `PinMsg`, `HeartbeatMsg`) over ZMQ. LMCache's role on the matrix is as middleware: it sits between the engine's KV connector and the storage tier, and most production deployments place LMCache in front of Mooncake Store, AIBrix KVCache, or Redis rather than choosing one over the other.

## 6. Mooncake KVStore

Mooncake is the serving substrate for Moonshot AI's Kimi. The FAST'25 paper [Mooncake] frames it as a *KVCache-centric disaggregated architecture*. (FAST'25 venue is confirmed; a "Best Paper" qualifier circulated in early reporting but is not confirmed by the official program record, so this chapter omits it.) The OSS repo (`kvcache-ai/Mooncake`, latest tag `v0.3.10.post2`) ships four pieces: TransferEngine (data plane, §3 above), Mooncake Store (distributed KV object store on top of TransferEngine), the older P2P Store (master-less BitTorrent-shaped checkpoint distribution), and more recently Mooncake-EP (MoE all-to-all) and Mooncake-PG (PyTorch process-group backend on TransferEngine).

Mooncake Store is a master-out-of-data-path object store. The **master service** (`master_service.h`) tracks which clients have mounted memory segments, what capacity each has, and for each `key` the list of `Replica::Descriptor`s identifying which client holds which slice. The master returns descriptors; the client issues TransferEngine transfers directly. The **Real client** (`real_client.h`) is a `PyClient` subclass that owns segment registration, runs heartbeats, and drives SSD offload (`file_storage.h` with three on-disk layouts: `Bucket`, `FilePerKey`, `Offset`). The **Dummy client** is a per-rank stub that forwards to the per-instance Real client over zero-copy shared memory; with TP=8 a typical deployment runs one Real and eight Dummy clients per inference instance. **etcd-based HA**: multi-master mode coordinates through etcd; master instances elect a leader and only the leader serves writes; a hot standby keeps a follower warm. **SSD offload via GDS** is a per-Real-client subsystem driven by the master via heartbeat: the master tells each client to evict specified keys to disk, the client runs `BatchOffload`, and `Get` on an evicted object falls back to disk transparently.

The two-phase write protocol — `PutStart(key, slice_length, config)` returns target descriptors with memory pre-allocated by the master, then the client issues TransferEngine `WRITE`s, then `PutEnd` (or `PutRevoke`) — keeps the master out of the data path while enforcing strong consistency. Once `Put` is acknowledged, the object is immutable until `Remove`.

There are two integration patterns into vLLM. The **direct KVConnector V1 over TransferEngine** path (`mooncake-wheel/mooncake/mooncake_connector_v1.py`) has `MooncakeConnector` subclass vLLM's `KVConnectorBase_V1` and uses TransferEngine in pure prefill→decode KV-transfer mode, no Mooncake Store needed; the scheduler tracks `_reqs_need_recv` and `_reqs_need_send` keyed by `kv_role` (`kv_producer`, `kv_consumer`, `kv_both`), and a `vllm_v1_proxy_server.py` round-robins requests across prefiller and decoder hosts. This is the path Mooncake's prefill-decode disaggregated scheduler uses. The **Mooncake Store as L3** path is used by vLLM PD via Mooncake Store, or by SGLang HiCache with Mooncake as the L3 backend; the engine plays the role of a Mooncake Store client and the cache pool spans the cluster. SGLang HiCache's "page-first" layouts are designed exactly so that one HiRadixTree page is a contiguous object Mooncake can transfer zero-copy.

## 7. NVMe-resident KV: CMX

CMX (Inference Context Memory Storage Platform) is NVIDIA's productized 4-tier KV pyramid with NVMe at the lowest tier, announced January 2026 alongside BlueField-4. It treats KV as its own persistent storage tier with a consistency model and SLO rather than as a swap target, pairing GPUDirect Storage with RDMA-accelerated NVMe-oF behind a BlueField-4 DPU. (CMX is 2025–2026 announcement-stage with limited independent production evidence; this discussion is based on NVIDIA's own materials.)

The case is straightforward: long-context reasoning and large RAG workloads need cache capacity beyond HBM+DRAM but cannot tolerate object-store latency. Petabyte-scale local NVMe at GDS speeds — roughly 100–500 µs per block versus 1–5 µs from DRAM and 50–500 ms from cold S3 — fills the gap that the T2/T3 boundary used to span. For a 100k-token prefix at 16 tokens per block (6,250 blocks at 200 µs each), sequential read is 1.25 seconds; striped across four drives this drops below 500 ms — a clear win against a 5–10 second cold prefill.

The trajectory CMX represents — *KV is its own storage tier* rather than *swap to host* — matters more than the specific NVIDIA productization. AIBrix's L2 distributed cache, llm-d's filesystem connector with CephFS / IBM Storage Scale, and Mooncake Store's hf3fs integration all point at the same trajectory from different vendors.

## 8. KV-aware routing — forward reference

Tiered KV is only useful if requests are routed to the engine instance whose KV tier already holds their prefix. Without affinity, every cache hit becomes a transfer cost. The cluster-level mechanism that pairs with tiered storage is KV-aware routing, which scores backend engines on prefix-cache affinity, KV utilization, queue depth, and LoRA-adapter affinity. The scoring formula in production usage is

$$\text{cost}(w_i) = \alpha \cdot \text{prefill\_blocks}(w_i) + \text{decode\_blocks}(w_i) - \beta \cdot \text{overlap}(w_i, \text{request prefix})$$

where the overlap term measures how many of the request's prefix blocks are cached in worker $w_i$'s tier hierarchy. Dynamo's KV router calls $\alpha$ the `--router-kv-overlap-score-weight`; llm-d implements the same idea as `precise-prefix-cache-scorer` in its EPP plugin chain; AIBrix's `prefix_cache.go` and `prefix_cache_preble.go` carry the same structure with different defaults. The scorer reads from a global indexer fed by KV events that engines publish over ZMQ or NATS as blocks are stored or evicted: `BlockStored`, `BlockRemoved`, `AllBlocksCleared`. The Inference Gateway API (GIE) has standardized the Endpoint Picker Protocol over Envoy ext-proc as the conformance contract for this scoring step.

The full mechanics — router architecture, the GIE `InferencePool` CRD, sticky and consistent-hashing variants, and the power-of-two-choices family — are covered in [see §50/01-router-gateway](../50-cluster-systems/01-router-gateway.md). The takeaway here is that tiered KV without KV-aware routing is a partial system; the two are designed to be deployed together.

## Current production state

Tiered KV is ubiquitous at frontier-scale serving as of May 2026. Every major engine ships a tiered-offload solution: vLLM via the KV connector V1 with LMCache as the dominant middleware (Mooncake, NIXL, MoriIO, p2p, hf3fs as alternatives); SGLang via HiCache (GA September 2025); TRT-LLM via priority-based block reuse plus host-memory offload plus Dynamo or Mooncake paths to remote; Dynamo via KVBM with G1–G4 tiers and NIXL transport; AIBrix via the L1+L2 KV cache framework; llm-d via the filesystem connector to CephFS, IBM Storage Scale, AWS EFS, or local NVMe. The abstraction stack — engine ↔ KV store middleware ↔ transport ↔ fabric — is settled, even if specific stores and transports differ between deployments.

The transport layer has converged on three first-class options: NIXL (NVIDIA-led, pluggable backend list, strong vLLM/Dynamo/SGLang integration), Mooncake TransferEngine (Moonshot-led, multi-protocol-native, broadest hardware coverage including AMD HIP and Huawei Ascend), and Perplexity TransferEngine (Perplexity-led, RDMA-native). UCX sits below all three; GDS is the standard mechanism for HBM↔NVMe. The fragmentation between NIXL and Mooncake TransferEngine has been partially resolved by NIXL accepting Mooncake TE as a backend plugin in May 2025.

The KV-store layer is more diverse. LMCache is the dominant middleware in OSS deployments because it is upstream-integrated into vLLM and SGLang, ships eight backends out of the box, and adds CacheGen and CacheBlend as differentiating codecs. Mooncake Store is the dominant cluster KV at frontier-MoE production scale (Kimi K2's 128×H200 deployment, deeply-integrated SGLang stacks). AIBrix's L1+L2 KVCache is the dominant choice in ByteDance-shaped deployments. Dynamo KVBM is the choice when the stack is NVIDIA-default and operators want NIXL fully integrated. The numbers cited throughout this chapter — LMCache's "up to 15× throughput in shared-prefix workloads," Mooncake's "87 GB/s on 4×200 Gbps RoCE," AIBrix's "50% throughput / 70% latency improvements," Dynamo's "750× over pre-Dynamo baselines" — are largely vendor-supplied; published independent reproductions remain limited. NVMe-resident KV (CMX, hf3fs, WEKA, AWS EFS-backed llm-d tiers) is the live frontier; the operational consistency model across cluster restarts, weight rotations, and fine-tune deltas remains an open engineering question. KV-aware routing has matured into the standard pairing for tiered storage, with the GIE Endpoint Picker Protocol providing a stable contract; the routing-and-tiering pair is now essentially a design *unit* rather than two separable choices.
