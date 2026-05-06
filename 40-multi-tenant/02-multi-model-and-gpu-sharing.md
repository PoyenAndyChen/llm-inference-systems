# Multi-Model Serving and GPU Sharing

**After reading this chapter, the reader will be able to:**

- Reason about *statistical multiplexing* of multiple models on a shared GPU pool — what AlpaServe established at placement time and how Llumnix-style live request migration extends it to runtime rebalancing.
- Compare cold-start mitigation strategies — ServerlessLLM's NVMe-to-GPU loading, Aegaeon's token-granularity model swapping, Fluid's overlap-of-compute-with-weight-transfer pipeline — and pick the right one for a given workload.
- Place the four mainstream **GPU-sharing primitives** (MIG, MPS, time-slicing, fractional-GPU schedulers including KAI Scheduler) on a 4-axis trade-off grid (isolation, latency interference, HBM sharing, suitability for LLM inference).

The previous chapter [see §40/01-lora-serving](01-lora-serving.md) treated multiplexing of one base model across many adapters. This chapter treats the orthogonal problem: many *distinct* models — different architectures, sizes, vendors — sharing the same GPU pool. Adapter switching is a kilobyte-scale reshuffle that finishes inside one iteration; *model* switching is a gigabyte-scale weight transfer that, naïvely, dominates request latency. The chapter has two halves: a software lineage (AlpaServe → Llumnix → ServerlessLLM → Aegaeon → Fluid) that extracts more capacity from a fixed pool, and a primitive lineage (MIG, MPS, time-slicing, KAI-class fractional schedulers) that *enforces* the sharing the software layer wants.

## 1. The multi-model regime

A frontier inference provider's catalog looks nothing like a single workload. The Aegaeon paper [Aegaeon] reports a model marketplace serving more than a thousand distinct models at Alibaba Cloud in mid-2025; hyperscaler-hosted inference is in the same range, with steeply heavy-tailed traffic. The naïve response — one dedicated pool per model — is wasteful for two reasons. First, **idle headroom**: a model with bursty traffic spends most wall-clock time below saturation, and a pool sized for peak wastes GPU-hours during the troughs; two models each averaging 30% utilization on dedicated pools cost roughly $2\times$ a shared pool sized for combined peak — provided the peaks do not align. Second, **cold-start asymmetry**: a model receiving fewer than one request per minute has an honest economic case for no resident copy at all, only an on-demand load when traffic arrives — and the scale-from-zero latency is the binding constraint.

A taxonomy clarifies which technique fits where:

```
Traffic pattern        | Catalog size | Right tool
-----------------------+--------------+-------------------------
Steady, single peak    | 1            | Dedicated pool, no sharing
Bursty, uncorrelated   | 5–50         | AlpaServe + Llumnix
Spiky, heavy-tailed    | 50–10,000    | Aegaeon swap; ServerlessLLM
Seasonal / scale-to-0  | any          | Fluid + activator
```

Cross-references: routing [§50/01-router-gateway](../50-cluster-systems/01-router-gateway.md), autoscaling [§50/02-autoscaling-cost-and-sustainability](../50-cluster-systems/02-autoscaling-cost-and-sustainability.md), fairness [§40/03-fairness-slo-routing](03-fairness-slo-routing.md), trust boundaries [§40/04-trust-boundaries-and-isolation](04-trust-boundaries-and-isolation.md).

## 2. AlpaServe — statistical multiplexing

[AlpaServe] (Li et al., OSDI'23) is the canonical statistical-multiplexing system. The setting is a cluster with $N$ models and $M$ GPUs, possibly $N > M$. Each model has a bursty, uncorrelated request stream. The question is: how should models be placed on GPUs, with what parallelism strategy, to maximize sustained request rate at SLO?

The classical pre-AlpaServe answer was *one model per GPU group, sized with headroom for the peak.* AlpaServe overturns the headroom assumption: under bursty, uncorrelated arrivals, *sharing* GPUs across multiple models strictly increases sustainable throughput, even when each model fits on a single GPU.

### 2.1 The multiplexing arithmetic

Let $\lambda_i$ be model $i$'s arrival rate, $\mu_i$ its single-GPU service rate, $\rho_i = \lambda_i / \mu_i$. Under M/M/1, expected queue length is $\rho / (1-\rho)$, so per-request wait grows superlinearly as $\rho \to 1$. For $K$ models sharing one GPU FCFS, aggregate utilization is $\rho_{\text{shared}} = \sum_i \rho_i$; as long as $\rho_{\text{shared}} < 1$ the queue is stable, and the variance of the aggregate (relative to mean) is lower than that of any single stream. The shared tail is strictly smaller than the sum of dedicated tails. AlpaServe quantifies the gain: for a representative bursty workload, **10× higher request rate or 6× more burst tolerance** at SLO than dedicated placement.

### 2.2 The placement problem

Multiplexing is a strict win *if weights are free*. Real weights have memory cost $M_i$, and HBM is finite. AlpaServe formulates placement as a joint optimization over which models share which GPU group, the parallelism strategy per model, and the routing policy; the placer is a mixed-integer program with greedy and ILP solvers. The non-obvious result: **model parallelism is sometimes justified even when a single model fits on one GPU**, because splitting model A across two GPUs (TP=2) lets each GPU also host parts of model B, raising per-GPU multiplexing diversity. The trade-off is concrete — TP costs an all-reduce per layer ($\sim 5\%$ NVLink-bound overhead on Hopper); the multiplexing gain on bursty workloads is order-of-magnitude.

AlpaServe is a *placement-time* system: it decides at deploy time which models live where. It does not migrate live requests, swap models at runtime, or address cold-start. It also assumes models are *resident* once placed; the regime where catalogs are too large to keep resident is Aegaeon / ServerlessLLM territory in §4.

## 3. Llumnix — live request migration

AlpaServe's static placement is brittle to load shifts. If model A's traffic spikes and its GPUs overload, queued requests suffer head-of-line blocking *even if* a sister GPU group hosting model B is idle, because the routing decision has already been made and the KV state lives on the overloaded worker. Drop-and-reroute is unacceptable for long-context generation where the prefill cost is sunk.

[Llumnix] (Sun et al., OSDI'24, [Llumnix-repo]) introduces **live request migration**: an in-flight request, including its KV cache, is moved from an overloaded worker to a less-loaded one mid-decode without dropping the request. This is the LLM-serving analog of OS process migration; it lifts the runtime scheduler from a placement-time decision to a per-iteration rebalancing primitive.

The dominant cost is moving the KV cache. For a request at decode position $t$, $L$ layers, $H_{\text{kv}}$ KV-attention heads (post-GQA), head-dimension $d_h$, FP16:

$$
M_{\text{KV}}(t) = 2 \cdot L \cdot H_{\text{kv}} \cdot d_h \cdot t \cdot 2 \text{ bytes}
$$

For Llama-3-70B ($H_{\text{kv}}=8$, $d_h=128$, $L=80$) at $t = 8192$: $M_{\text{KV}} \approx 2.7$ GB. Over NVLink at $\sim$900 GB/s (H100 NVLink 4.0): $\sim$3 ms in-rack. Over 400 Gb RoCE between racks: $\sim$54 ms. This is the migration budget — it must finish faster than the request's *remaining* generation time, or the operation is a net loss.

Llumnix uses a copy-and-prefetch hybrid: the source snapshots its current KV state and stops emitting new tokens for the migrated request; KV blocks transfer destination-pulled over NVLink in-rack or RDMA across racks (often via NIXL or UCX [see §30/02-kv-tiered-offload](../30-kv-cache/02-kv-tiered-offload.md)); the destination resumes decoding at the exact next position. Net overhead on a well-provisioned NVLink interconnect is single-digit percent of total request time, and the order-of-magnitude tail-latency improvement Llumnix reports is the *net* of higher migration cost vs. avoided HOL blocking.

Migration is a primitive, not an end. With it in the toolbox a fragmented worker can drain its requests and shut down for reclamation, a request predicted to miss SLO can move to a faster worker, and live model scale-out can drain in-flight generation across new replicas without dropping. In 2026 production, **migration is a core primitive of multi-model and disaggregated-prefill-decode serving**; the open questions are *when* to migrate, not *whether* the primitive should exist.

## 4. Cold-start: ServerlessLLM, Aegaeon, Fluid

AlpaServe and Llumnix assume the involved models are GPU-resident. The opposite regime — models that come and go on demand — has its own pressures. The dominant cost is the time to get a model from somewhere (object store, NVMe, CPU memory) onto the GPU and ready to serve. Three lineage entries define the state of the art.

### 4.1 ServerlessLLM — fast checkpoint loading

[ServerlessLLM] (Fu et al., OSDI'24) attacks cold-start head-on for serverless deployments — "no model resident; load on first request; possibly evict after idle." Naïve cold-start of a 70B model from cloud object storage at $\sim$1 GB/s effective ingress takes 2–3 minutes — uncompetitive for any interactive workload.

ServerlessLLM reduces cold-start from minutes to seconds via three changes. **NVMe-to-GPU direct loading** uses GPUDirect Storage (GDS) so checkpoint bytes flow NVMe → HBM via PCIe DMA without staging through CPU memory, eliminating the host-memcpy step and roughly halving wall-clock load time on Gen5 NVMe + Hopper. **Checkpoint locality** caches recently-served checkpoints on local NVMe so a request for a recently-evicted model hits NVMe ($\sim$10 GB/s) rather than the object store ($\sim$1 GB/s) — an order-of-magnitude jump. **Predictive prefetching** pre-warms checkpoints onto NVMe before traffic arrives, converting first-of-the-day cold starts into NVMe-resident warm starts. Composed, the techniques get a 70B model resident in seconds, not minutes. ServerlessLLM does not handle multi-model packing — that is Aegaeon's territory.

### 4.2 Aegaeon — token-granularity model swapping

[Aegaeon] (Xiang et al., SOSP'25) takes the cold-start logic to the extreme: instead of swapping models *between* requests, swap them *between iterations* of continuous batching. A single GPU hosts model A for one iteration, swaps in model B for one iteration of B's queue, swaps in model C, and so on, with swaps at iteration boundaries.

The arithmetic that makes this work: a decode iteration on an 8B-class model on H100 at modest batch is $\sim$10–50 ms; moving 16 GB (an 8B FP16 model's weights) over NVLink at 900 GB/s is $\sim$18 ms; over PCIe Gen5, $\sim$320 ms; over Grace-Hopper's NVLink-C2C coherent fabric, sub-10 ms. The swap-to-compute ratio is unfavorable on PCIe but favorable on NVLink and explicitly *good* on Grace-Hopper. The Aegaeon paper reports the production deployment **reduced 1192 GPUs to 213 in Alibaba's model marketplace and supports up to 7 models concurrently per GPU** at sub-second swap granularity.

The mechanism resembles continuous batching extended across models: per-model request queues, a scheduler that picks the most-eligible queue (FIFO mixed with fairness and SLO urgency), and swaps interleaved at iteration boundaries. KV state for in-flight requests of a swapped-out model lives on host memory or NVMe and streams back when its model is next swapped in. The take-away: *with sufficient interconnect bandwidth, a single GPU is a multi-model server at millisecond timescale.* Whether Aegaeon's primitives generalize beyond Alibaba's heavy-tailed marketplace mix is open empirically — the paper is recent (SOSP'25) and replication evidence outside Alibaba is still accumulating.

### 4.3 Fluid — overlap compute with weight transfer

Rather than swap models, **overlap** the loading of a not-yet-resident model with compute on a currently-resident one. *Fluid* names the pattern: asynchronous loading pipelines that stream weights from CPU/NVMe to GPU in the background, with serving beginning on freshly-loaded layers as they arrive layer-by-layer. Total bytes transferred are unchanged; what shrinks is *first-token-after-cold-start* latency — from full-model-load-time to first-layer-load-time plus first-layer-prefill. With careful pipelining a model emits tokens before its later layers have fully arrived; the sampler waits on the last layer, but TTFT is dominated by queue plus first-layer-prefill, not load time.

This is the cross-reference point with autoscaling [see §50/02-autoscaling-cost-and-sustainability](../50-cluster-systems/02-autoscaling-cost-and-sustainability.md). Without Fluid-style overlap, scale-to-zero is too slow for interactive workloads above toy scale; with it, cold-start latency is competitive with always-on min-replicas-of-1, and zero idle GPUs is realized in production. The llm-d 0.5 release [llm-d-v0.5] (February 2026) ships activator-based scale-from-zero atop a Fluid-style pipeline.

The three are complementary, not substitutive: a 2026 serverless multi-model platform typically uses Fluid-style overlap during the load, ServerlessLLM-style NVMe + GDS for the bytes path, and Aegaeon-style iteration-level swapping for the resident-set policy.

```
Technique        | Granularity       | Best when
-----------------+-------------------+--------------------------------
ServerlessLLM    | Whole-model load  | Cold catalog, infrequent reuse
Aegaeon          | Per iteration     | Heavy-tailed catalog, fast NVLink
Fluid            | Per layer overlap | Scale-to-zero; first token
                 |                   |   before full load completes
```

## 5. GPU sharing primitives

The systems above assume the GPU can be partitioned, multiplexed, or co-scheduled by *some* lower mechanism. This section names the four mainstream mechanisms and places them on a trade-off grid.

### 5.1 MIG — Multi-Instance GPU

**MIG** is NVIDIA's hardware-level GPU partitioning. An A100, H100, or Blackwell GPU is physically partitioned into up to seven isolated instances, each with a dedicated slice of streaming multiprocessors (SMs), HBM, and crossbar to NVLink. The partition is enforced at the silicon level: one instance cannot read another's HBM, and SMs cannot be borrowed across instances. To CUDA, each instance appears as a separate GPU device.

Properties: **isolation is hard** (memory and compute physically partitioned; a crash in one instance does not affect others; performance interference is eliminated by construction). **HBM sharing is none** — each instance owns its slice and cannot lend to peers. **Reconfiguration is heavyweight**: MIG profiles are set via `nvidia-smi mig` and require draining all instances before re-partitioning. **Availability** is confirmed across A100, H100/H200, and most B100/B200 SKUs as of mid-2026, though the 36-GPU coherent-fabric NVL72 racks have a more nuanced per-SKU story that operators should check against the latest support matrix.

MIG suits **multi-model serving where strong isolation is required** — confidential workloads with tenant separation, regulated industries, or workloads whose interference would violate latency SLOs. It does not suit highly bursty workloads, because the static partition prevents one instance absorbing a peer's burst. A 7B-class model fits a 1g.10gb profile; a 70B-class model spans the full GPU and MIG offers no benefit.

### 5.2 MPS — Multi-Process Service

**MPS** is NVIDIA's *software* mechanism for concurrent kernel execution from multiple processes. Without MPS, kernels from different processes are serialized at the CUDA driver level; with MPS they run *concurrently*, sharing SMs at the warp-scheduler level.

Properties: **isolation is weak** — processes share SMs, shared memory, L2, HBM bandwidth; a misbehaving process can starve peers, and a crash can in some failure modes affect other clients. **HBM** is shared physically with per-process contexts and is subject to cross-process fragmentation. **Performance interference is real**: two processes on one SM contend for warp slots, so aggregate throughput exceeds time-slicing but per-process latency exceeds dedicated GPU. **Reconfiguration is lightweight** (`cudaSetDevice` plus connect to the MPS daemon, no drain).

MPS suits **packing many small inference workers** that individually under-utilize the GPU — small-model embedding fleets, many-small-base scenarios, low-load multi-tenant services. It is not suitable when isolation matters.

### 5.3 Time-slicing — the default

**Time-slicing** is the GPU's default multi-process model. Without MIG or MPS, processes are serialized at the CUDA-context-switch level: one process's kernels run to completion (or yield) before another's begin. There is no true parallelism; only one CUDA context is live at a time.

Properties: **isolation is temporal-only** — no SM-level interference because only one process is on the SMs, but context-switch overhead is hundreds of microseconds at kernel-launch granularity. **HBM** is per-process, contending only via fragmentation. **Compatibility is maximum** (every CUDA-capable GPU; no driver flags or daemons). **Latency interference is high**: a long-running batched-attention kernel from process A (hundreds of microseconds to milliseconds) blocks process B, so B's TTFT spikes unpredictably.

Time-slicing is the appropriate fallback when MIG and MPS are unavailable (consumer GPUs, restricted environments) or when sharing is incidental. It is rarely the right primary substrate for latency-SLO LLM workloads.

### 5.4 KAI Scheduler and fractional-GPU schedulers

The primitives above operate at the CUDA driver level. A different layer — the **Kubernetes scheduler** — addresses the cluster-wide allocation problem: which pod gets what fraction of which GPU? The vanilla Kubernetes scheduler treats GPUs as integer-count resources (`nvidia.com/gpu: 1`); a pod requests whole GPUs. This is incompatible with workloads wanting a fraction (an embedding service that needs 0.2 of an A10).

**KAI Scheduler** (Run.AI / NVIDIA) is a Kubernetes-native GPU scheduler with three capabilities. **Fractional GPU scheduling**: a pod requests `gpu: 0.5` and the scheduler packs two such pods onto one GPU using MPS (or, if requested, an MIG instance of matching profile). **Bin-packing across nodes** raises realized utilization without violating per-pod requests. **Preemption with checkpointing** lets lower-priority workloads yield to higher-priority ones and resume rather than restart.

[Hedge: NVIDIA's acquisition of Run.AI was announced in late 2024 and closed by year-end per public press; the open-source release of KAI Scheduler followed in 2025, though precise feature parity between the OSS release and the Run.AI Atlas commercial offering is in flux as of mid-2026.] The OSS release covers core scheduling logic; surrounding control-plane features (workspace UX, dashboards, quota policy) are not all OSS. [Hedge: adoption is concentrated in NVIDIA-aligned customers — cloud providers running NVIDIA AI Enterprise, sovereign-AI deployments, dedicated enterprise GPU clusters — and broader OSS adoption beyond that base is still developing.]

### 5.5 Fragmentation-aware fractional schedulers

A subtler problem the basic fractional schedulers do not solve: **HBM fragmentation across co-tenants**. Two pods each requesting 0.4 of an 80 GB H100 nominally get 32 GB each, but if pod A's first kernel allocates 30 GB and pod B's first kernel allocates 30 GB, the remaining 20 GB is fragmented across two address spaces and cannot be coalesced; a third pod requesting 0.2 may not fit even though 20 GB is nominally free.

The 2025–2026 wave of fragmentation-aware schedulers addresses this. [Prism] (2025) coordinates cross-model memory via on-demand virtual-page mapping, letting two co-resident LLMs share an HBM pool with dynamic page-level transfer rather than statically partitioned allocations; >2× cost reduction or 3.5× more requests at SLO under multi-LLM workloads. A concurrent NSDI'26-targeted line extends Kubernetes with per-GPU memory fragmentation as a first-class scheduling signal, not just compute. These extend the KAI baseline by treating HBM as packable at finer granularity; they are not a replacement for MIG / MPS but a smarter Kubernetes layer on top.

### 5.6 Comparison

```
Axis              | MIG               | MPS              | Time-slice    | KAI / Frac. sched.
------------------+-------------------+------------------+---------------+----------------------
Isolation         | Hard (silicon)    | Soft (driver)    | Temporal only | Inherits MIG / MPS;
                  |                   |                  |               |   plus pod boundary
Latency           | Zero between      | Real (SM-level)  | High (ctx-    | Inherits from
  interference    |   instances       |                  |   switch)     |   underlying primitive
HBM sharing       | None (static)     | Shared physical, | Per-process   | Logical fractional;
                  |                   |   per-process    |   contention  |   physical inherited
                  |                   |   contexts       |               |
Reconfiguration   | Heavyweight       | Lightweight      | None needed   | Pod-level replan
LLM inference fit | Strong-isolation  | Many small       | Single tenant | Cluster-wide
                  |   workloads;      |   co-tenants     |   only;       |   fractional
                  |   regulatory      |                  |   default     |   allocation;
                  |                   |                  |   fallback    |   composes with above
Availability      | A100, H100/H200,  | All NVIDIA       | All NVIDIA    | KAI: NVIDIA / Run.AI
                  |   most B100/B200  |   GPUs           |   GPUs        |   ecosystem; Prism:
                  |                   |                  |               |   research
```

A practical decision tree: **strict isolation → MIG; many small co-tenants without isolation → MPS; legacy or fallback → time-slice; cluster-wide fractional packing → KAI on top of MIG / MPS.** Aegaeon-style iteration-level swapping is orthogonal — it runs as a single process that internally swaps models, sees the GPU as undivided, and needs none of these primitives.

## Current production state

As of mid-2026, multi-model GPU sharing is established but heterogeneous. AlpaServe's statistical-multiplexing argument is now baseline: any deployment with more than a handful of bursty models on dedicated pools is leaving order-of-magnitude utilization on the floor. Llumnix-style live request migration has crossed from research into production schedulers — the Llumnix open-source release [Llumnix-repo] is integrated by Alibaba and has inspired migration primitives across the 2025–2026 cluster control planes (llm-d's scheduler architecture, NVIDIA Dynamo's planner, AIBrix's autoscaler-coordinated migration), without those systems necessarily wrapping the OSS Llumnix codebase verbatim.

Cold-start has bifurcated. ServerlessLLM is the canonical reference for fast checkpoint loading; its three techniques (NVMe-to-GPU via GDS, checkpoint locality, predictive prefetch) are now standard machinery that every serverless-LLM offering implements in some form. Fluid-style overlap-of-compute-with-load is the enabling technology for credible scale-to-zero, and the llm-d 0.5 release [llm-d-v0.5] is the reference for production scale-to-zero at non-trivial model size. Aegaeon-style iteration-granularity swapping is the most aggressive design point and is in production at Alibaba's marketplace; whether it generalizes elsewhere depends on workload heaviness-of-tail and interconnect speed, and the answer is workload-specific.

On primitives the picture is settling rather than converging. **MIG** is the strong-isolation default on Hopper and Blackwell where the workload mix demands it, confirmed across H100, H200, and most B100/B200 SKUs (the GB200/GB300 NVL72 picture is more nuanced per SKU; operators should check the support matrix rather than assume uniform Blackwell coverage). **MPS** remains the default for packing many small inference workers without isolation. **Time-slicing** is the always-available fallback, the wrong primary choice for latency-SLO workloads. **KAI Scheduler** is the OSS reference for Kubernetes-native fractional GPU scheduling; NVIDIA's Run.AI acquisition and the subsequent open-source release have made KAI the most-cited fractional scheduler in the cloud-native AI conversation, though adoption beyond NVIDIA-aligned customers remains in the developing-rather-than-dominant phase. Fragmentation-aware schedulers ([Prism] and the NSDI'26 wave) are the research frontier and will likely fold into the KAI-class baseline over the next year.

The take-away: **multi-model serving in 2026 is a layered stack, not one technique** — placement (AlpaServe) at deploy time, migration (Llumnix) at runtime, cold-start machinery (ServerlessLLM / Fluid) for the long tail, hardware partitioning (MIG / MPS) for the substrate, Kubernetes fractional scheduling (KAI-class) for cluster-wide allocation. Each layer has a canonical reference and a production implementation; the operator composes the layers that fit the workload rather than picking one master abstraction. The next chapter [see §40/03-fairness-slo-routing](03-fairness-slo-routing.md) treats fairness and SLO arbitration between tenants once the sharing substrate is in place.
