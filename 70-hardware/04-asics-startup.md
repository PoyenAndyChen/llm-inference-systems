# Startup ASICs: Groq, Cerebras, SambaNova, Tenstorrent, Furiosa, Etched

**After reading this chapter, the reader will be able to:**

- Place each major non-hyperscaler inference ASIC — Groq's TSP / NVIDIA Groq 3 LPX, Cerebras WSE-3, SambaNova SN40L, Tenstorrent Wormhole / Blackhole, Furiosa RNGD, Etched Sohu — onto a single architectural map organized by memory tier (SRAM-only, wafer-scale SRAM, three-tier DDR-augmented, HBM + on-chip SRAM) and explain why the niche each occupies cannot be served by a general-purpose GPU at the same operating point.
- Read the Groq → NVIDIA Groq 3 LPX productization story as a structural change in the ASIC landscape: a $20B IP licensing deal (not an acquisition) that turns a SRAM-only deterministic-dataflow chip into the *decode FFN/MoE co-processor* of NVIDIA's Vera-Rubin heterogeneous rack — and recognize that the LPX is announced but not shipped as of May 2026.
- Distinguish vendor-reported headline numbers from independently reproduced production figures across this corner of the market, and identify which products are shipping silicon to external customers at scale (Cerebras CS-3, SambaNova SN40L, Tenstorrent Blackhole, Furiosa RNGD) versus which are essentially press releases and venture rounds (Etched Sohu).

The startup ASIC market for LLM inference is not one market. The companies in this chapter compete on different axes than NVIDIA — single-user latency, MoE serving cost per token, developer customizability, Korean-market sovereignty, transformer-only specialization — and each has chosen a different memory hierarchy to win the axis it picked. The hyperscaler in-house silicon discussed in [§70/03-asics-hyperscaler](03-asics-hyperscaler.md) is structurally different again: TPU, Trainium, MTIA, and Maia exist to serve a single owner's workload at $1T-scale capex, not to win benchmarks against H100. Reading the two ASIC chapters together: hyperscaler silicon optimizes the cost curve of its parent's own training and serving fleet; startup silicon optimizes a niche an external customer can be persuaded to pay for. The structural fact for both is that GPU peak FLOPs and HBM bandwidth scale faster than any single startup's product cycle, so the niche must be defended on a dimension other than raw spec-sheet competition with Blackwell.

## 1. The ASIC landscape: niches, not a market

Five architectural choices recur and explain almost everything else about the products in this chapter.

**SRAM-only, no HBM, no DRAM (Groq TSP, NVIDIA Groq 3 LPX).** All weights and activations live on chip in distributed SRAM. The win is bandwidth: on-chip SRAM at 80 TB/s (Groq TSP) or 40 PB/s aggregate (LPX rack) is one to two orders of magnitude above HBM3e on a Blackwell die, which buys best-in-class single-user decode latency on whatever fits. The cost is capacity: a single chip holds tens to hundreds of megabytes, so any non-trivial model demands hundreds to thousands of chips wired together. This pushes the design point toward deterministic dataflow, plesiosynchronous chip-to-chip clocking, and ahead-of-time compilation.

**Wafer-scale SRAM (Cerebras WSE-3).** A single 46,225 mm² wafer-die holds 900,000 cores and 44 GB of on-chip SRAM at 21 PB/s aggregate. The wafer *is* the chip; chip-to-chip bandwidth becomes inter-tile bandwidth on the same piece of silicon. Like the SRAM-only path, weights and KV must fit in fast memory for the architecture to pay; unlike Groq, Cerebras spreads tens of gigabytes across one die rather than hundreds across racks.

**Three-tier memory (SambaNova SN40L).** SRAM + co-packaged HBM + DDR5 DIMMs in one socket. The DDR tier — up to 1.5 TiB per socket — is too slow for hot-path matmul, but large enough to hold the cold experts of a Mixture-of-Experts model without offloading to host. SambaNova's *Composition of Experts* serving sits exactly on this point: a router selects an expert subset per request, the subset moves DDR→HBM→SRAM through the dataflow, and the cost-per-token is set by the slowest tier the path actually uses.

**HBM + on-chip SRAM (Tenstorrent Blackhole, Furiosa RNGD).** The conventional GPU-style hierarchy with smaller HBM and larger on-chip distributed SRAM than a Blackwell die. Tenstorrent and Furiosa both occupy this corner; their differentiation is at the programming model and packaging level (Tenstorrent's open RISC-V "baby cores" and developer workstation, Furiosa's tensor-contraction-first compiler and Korean enterprise positioning) rather than in raw memory hierarchy.

**Single-architecture transformer ASIC (Etched Sohu, claimed).** A 4nm part with 144 GB HBM3E and a programming model that admits only the transformer forward pass — no Mamba, no SSM, no CNN. The design bet is that locking architecture choice into silicon buys 5–10× the perf-per-watt of a general-purpose accelerator on transformer workloads. As of May 2026 there is no shipping silicon and no third-party benchmarks; the entry exists in this chapter as a footnote on what the niche-defense playbook looks like when execution lags announcement.

The rest of the chapter walks each of these points in turn. The framing throughout is that production share is set not by who has the highest peak FLOP/s but by who has converted niche advantage into a deployable serving stack and a paying customer base.

A useful comparison anchor is the GPU memory hierarchy table of [§00/03-gpu-hardware-primer](../00-foundations/03-gpu-hardware-primer.md): registers (zero latency) → shared/L1 (~30 cycles, ~33 TB/s aggregate) → L2 (~150 cycles, ~12 TB/s) → HBM (~600 cycles, 3.35 TB/s on H100) → NVLink (~1 μs, 900 GB/s) → PCIe (~5 μs, 128 GB/s). Each architecture in this chapter relocates the structural boundary in this hierarchy:

| Product | What it changes vs. GPU hierarchy | Result |
|---|---|---|
| Groq TSP / LPX | SRAM scaled to weight-resident capacity by spreading across many chips | Bandwidth ceiling moves up ~10×; deterministic dataflow replaces dynamic scheduling |
| Cerebras WSE-3 | Wafer-scale: chip-and-NVLink boundary collapses into one die | 44 GB SRAM at 21 PB/s; no off-die boundary on hot path |
| SambaNova SN40L | DDR tier added below HBM, dataflow routes around it | Cold-expert capacity at DDR cost; CoE-style serving |
| Tenstorrent Blackhole | More on-chip SRAM (~210 MB) than typical GPU; GDDR6 instead of HBM | Mid-tier compute, Ethernet-native fabric, open programming model |
| Furiosa RNGD | More on-chip SRAM (256 MB) than typical GPU; HBM3 retained | TCP-style compilation; mid-scale efficiency |
| Etched Sohu (claimed) | Hierarchy unchanged; compute path specialized to transformer | Architectural lock-in for perf-per-watt (unverified) |

None of these wins on every workload; each wins on the workload whose memory access pattern matches the relocated boundary. The implications echo through the rest of the engine stack — quantization choices ([§10/04-quantization](../10-engine-core/04-quantization.md)) interact with each architecture differently, KV-compression strategies ([§30/01-kv-compression](../30-kv-cache/01-kv-compression.md)) pay different multipliers depending on which tier the KV lives in, and parallelism strategies ([§20/01-parallelism-strategies](../20-distributed-inference/01-parallelism-strategies.md)) face different scale-up bandwidth budgets on each fabric.

## 2. Groq LPU: deterministic dataflow on SRAM only

The original Groq Tensor Streaming Processor (TSP), shipped at scale through GroqCloud since 2024, is the cleanest expression of the SRAM-only inference idea. Each TSP holds 230 MB of on-chip SRAM and no HBM or external DRAM; the chip's architecture exposes a sea of arithmetic units (vector, matrix, switch) connected by a static, ahead-of-time-scheduled crossbar. The compiler decides every cycle of every unit; runtime decisions are essentially absent. *Plesiosynchronous* chip-to-chip clocking — chips run on independent free-running clocks with bounded relative drift — lets a multi-chip system behave as a single deterministic dataflow without the cost of a global clock tree [Groq-LPU-Page, Groq-LPU-Inside].

The architecture's strengths follow directly from the constraints. SRAM at ~80 TB/s aggregate per chip is roughly 25× the HBM3 bandwidth of an H100 and ~10× the HBM3e bandwidth of B200. Decoding a 70B model from this kind of bandwidth at small batch produces dramatic single-user latency — on Llama-2-70B Chat, Groq reported approximately 241 tokens/second/user in 2024, roughly an order of magnitude over GPU baselines at the time [SemiA-Groq]. The cost is equally direct: serving Mixtral 8×7B required 576 chips arranged as 8 racks × 9 servers × 8 chips, because the model's parameters and activations together do not fit in any smaller SRAM envelope. The economics work only if utilization stays high across the rack-scale chip count, which is why the primary product has always been GroqCloud (a SaaS endpoint) rather than on-prem boxes — multi-tenant traffic amortizes the per-rack cost across many concurrent users.

The roofline lens of [§00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md) applies cleanly here. Decode is HBM-bound on a GPU because the per-token weight read pulls the full parameter tensor from HBM at ~3.35 TB/s. On Groq TSP the weight read pulls from on-chip SRAM at ~80 TB/s on the chip that owns each shard, with chip-to-chip transfer over deterministic interconnect for cross-shard fragments. The bandwidth ceiling moves up by an order of magnitude; the operating point at fixed batch lands far above the GPU-decode ceiling. What does not move is peak compute per chip — Groq TSP at 14nm GlobalFoundries is well below H100 in raw FLOPs — and the architecture is not chosen for FLOP-density. It is chosen because the bandwidth-to-FLOP ratio puts decode comfortably on the bandwidth-favorable side of the roofline.

A second consequence of the deterministic-dataflow model is the relationship between batch size and latency. On a GPU, increasing batch size at fixed model size amortizes the HBM read of weights across more tokens, raising arithmetic intensity and pushing the operating point toward the compute ceiling — single-user latency degrades with batch but throughput rises. On Groq TSP the compiler statically schedules the entire computation graph at compile time; batch size is part of the graph signature and a different batch size compiles to a different program. The architecture trades dynamic flexibility (continuous batching, chunked prefill, request preemption — the machinery of [§10/03-batching-scheduling](../10-engine-core/03-batching-scheduling.md)) for the latency floor that static scheduling buys. GroqCloud's published positioning treats this as a feature: the SaaS layer aggregates requests upstream into batches the underlying compiled graphs accept, and the deterministic substrate guarantees that whatever decode latency the graph delivers is delivered consistently rather than jittered by scheduling decisions.

The compiler-driven nature of the TSP also shapes which models the platform supports in production. Each model architecture must be compiled for a specific deployment topology (number of chips, sharding strategy, batch-size set), and adding a new model is a non-trivial engineering effort relative to dropping a checkpoint into a vLLM container. GroqCloud's published model catalog reflects this: a curated set of frontier-lab open-weight models (Llama, Mixtral / DeepSeek-V3 family at various points), updated on a release cadence rather than per-checkpoint. The implicit cost trade — fewer supported models in exchange for the latency floor on the supported ones — is the structural feature that distinguishes Groq's market position from, say, Together AI or Fireworks running the same models on GPUs.

## 3. NVIDIA Groq 3 LPX: the licensing deal and the decode co-processor

The deal that reshaped the chapter's framing came twenty months after the original TSP's headline benchmarks.

In December 2025 NVIDIA announced a $20 billion *non-exclusive IP licensing* agreement with Groq — explicitly *not* an acquisition. Groq remains an independent company; NVIDIA gained the right to build product on top of Groq's deterministic-dataflow architecture, scaled to NVIDIA's manufacturing partners and integrated into the Vera-Rubin platform [Toms-NV-Groq, NP-NV-Groq]. The result was unveiled at GTC March 2026 as **NVIDIA Groq 3 LPX** [NV-Groq3LPX, Decoder-LPX]. As of May 2026, the LPX is *announced but not shipped*; production silicon is targeted for the Vera-Rubin ramp window and early racks are reported in lab integration but not at customer.

The LPX architecture, per NVIDIA's published specifications, is a substantially scaled productization of the original Groq TSP idea on a modern process node:

| Parameter | Groq TSP (2024) | NVIDIA Groq 3 LPX (announced 2026) |
|---|---|---|
| Process | GlobalFoundries 14 nm | Samsung 4 nm |
| On-chip SRAM | 230 MB / chip | 128 GB / 256-LPU rack (~500 MB / chip) |
| HBM / DRAM | none | none |
| Peak FP8 compute | not headlined | 315 PFLOPS / rack |
| On-chip SRAM BW | 80 TB/s / chip | 40 PB/s aggregate / rack |
| Scale-up BW | plesiosynchronous chip-to-chip | 640 TB/s scale-up fabric |
| Primary deployment | GroqCloud SaaS | Vera-Rubin NVL72 + LPX heterogeneous rack |

The structurally important fact is the *co-processor* deployment model. NVIDIA's published positioning of LPX is not "alternative to a Rubin GPU" but "complement to a Rubin GPU." A Vera-Rubin NVL72 + LPX rack assigns prefill and the attention layer to the GPUs (which have HBM3e at ~8+ TB/s, well-suited to streaming the KV cache) and assigns the FFN and MoE expert layers of decode to the LPX chips (which have SRAM at 40 PB/s, well-suited to streaming weights). The split exploits the structural asymmetry that attention is KV-bandwidth-bound while FFN/MoE is weight-bandwidth-bound — a refinement of the prefill–decode disaggregation logic of [§20/02-prefill-decode-disagg](../20-distributed-inference/02-prefill-decode-disagg.md), but at a finer grain (within decode) and across heterogeneous silicon rather than within a homogeneous GPU pool.

A worked example makes the asymmetry concrete. For a 70B-parameter dense model at FP8 with 8K-token context, the per-token decode cost decomposes roughly as: ~70 GB of weight reads (FFN-dominated, since FFN is roughly 2/3 of dense parameters), plus ~1-2 GB of KV reads from the active context (attention-dominated, growing with sequence length). On a single H200, both reads come from the same 4.8 TB/s HBM3e pool and contend for the same bandwidth. On a Vera-Rubin + LPX rack as NVIDIA describes it, the FFN's 70 GB is satisfied by SRAM at rack-aggregate 40 PB/s — roughly 8000× over a single H200's HBM, even after deflating for the fact that only a fraction of LPX SRAM is on the read path per token — while the KV's 1-2 GB stays on the GPU's HBM. The arithmetic suggests the LPX side completes its work in roughly the time the GPU spends on attention; whether the actual latency overlap matches the arithmetic depends on the inter-silicon transfer cost and the scheduler's ability to keep both sides busy, both of which are open variables until the rack ships.

The split has a second-order implication for the engine layer. A homogeneous-GPU disaggregation deployment routes prefill and decode to physically separate GPU pools; the unit of dispatch is the request, and KV state moves between pools over NVLink or InfiniBand once per request. A heterogeneous Vera-Rubin + LPX rack pushes the boundary deeper: within decode, attention layers stay on the GPU and FFN/MoE layers move to LPX, with activations crossing the heterogeneous interconnect once per layer. The cross-silicon traffic is therefore O(layers × hidden) per token, and the rack-internal scale-up fabric (the 640 TB/s figure NVIDIA cites) must be sized to absorb it without becoming the bottleneck. Whether this layer-grain routing reaches production maturity depends on the scheduler being able to keep both silicon pools loaded — a non-trivial extension of the prefill-decode separation logic that the engine implementations of [§80](../80-oss-deep-dives/00-overview-comparison.md) (vLLM, SGLang, TRT-LLM, Dynamo) currently support — and is one of the structural unresolved questions for the LPX rollout.

NVIDIA's own framing draws a parallel to the 2019 Mellanox acquisition: an adjacent-silicon company integrated to expand the rack's capability surface rather than to compete on the same die. With caveats: the parallel is plausible architecturally, but it depends on LPX shipping, on the heterogeneous scheduler reaching production maturity, and on independent benchmarks confirming the claimed FFN/MoE decode throughput [NP-NV-Groq]. None of these is proven as of May 2026; the LPX entry in this chapter is a structural prediction rather than a deployed reality.

## 4. Cerebras WSE-3 and CS-3: wafer-scale, latency-first

Cerebras takes the SRAM-resident philosophy in the opposite direction: instead of hundreds of small chips wired together, one wafer-scale die holds the model. The WSE-3 (announced 2024) integrates 4 trillion transistors and 900,000 AI cores onto a 46,225 mm² die — the entire 300 mm wafer minus edge yield, fabricated as one logical chip. On-chip SRAM totals 44 GB at 21 PB/s aggregate bandwidth, and the same fabric that connects cores within an SM-equivalent unit also connects the unit's neighbors, with no off-die bandwidth tier intervening [Cerebras-CS3, Cerebras-HC24].

The CS-3 system packages one WSE-3 with optional **MemoryX** (off-wafer parameter store for training) and **SwarmX** (cluster fabric for multi-CS-3 scale-out). At the inference operating point relevant here, the configuration that matters is a single CS-3 box delivering ~125 PFLOPS of FP-inference compute at ~25 kW [NBF-CS3]. Fleet scale: Cerebras has reported scaling its inference capacity toward >40M tokens/second across eight data centers by Q4 2025 (vendor-supplied; production-scale corroboration is partial) [Cerebras-Compare]. Up to 2,048 CS-3 systems can be aggregated into a logical training cluster delivering 256 EFLOPS FP16 with ~24 trillion-parameter capacity through MemoryX-staged parameters; the training-scale numbers are mentioned here only because they color the narrative around inference cluster claims.

Cerebras's distinctive performance result is single-user decode token rate. The vendor-reported figures, partly reproduced by independent benchmark services [Cerebras-Compare, Cerebras-press]:

- Llama-3.1-70B at approximately 2,100 tokens/second/user — roughly 8× a single H200's decode rate at the same operating point.
- Llama-4 Maverick at approximately 2,500 tokens/second/user — roughly 2× a DGX B200.

These are vendor-supplied benchmarks; ArtificialAnalysis and SemiAnalysis have independently sampled some of them, but the production-scale reproducibility under arbitrary serving traffic is not at the level of NVIDIA's MLPerf baseline. The hedged characterization: Cerebras occupies the *highest-tok/s/user* corner of the inference market, a corner that matters for latency-critical applications (real-time customer service, voice agents, code-completion at human-perceptible speeds) and that no GPU configuration matches at any batch size — but the corner is narrower than the headline numbers suggest, weakening on long context (where KV growth strains the SRAM-resident model) and bound by a proprietary software stack that ships with the box.

Cerebras was pre-IPO as of mid-2026 per public industry reporting; the production deployment list is heavily oriented toward vertical-AI partners and government customers rather than the hyperscaler accounts that drive GPU revenue.

Two architectural facts behind the latency advantage are worth separating. First, weight-resident SRAM means decode does not pay HBM bandwidth at all — the per-token weight read is satisfied entirely from the wafer's distributed SRAM at the 21 PB/s aggregate rate, which is roughly 7× the HBM3e bandwidth of a single B200 GPU and roughly 150× the per-GPU NVLink 5 bandwidth that a multi-GPU TP shard would use to assemble the weight tensor. The second fact is that *cross-tile* bandwidth on the wafer is also SRAM-class: a tensor-parallel-style activation transfer between cores does not cross a chip boundary because there is no chip boundary. The combination eliminates both the HBM-bound and the NVLink-bound terms of the GPU decode latency budget, leaving only compute-time and intra-tile data-movement on the critical path. Below the line where the model fits in 44 GB of SRAM, this is genuinely a different operating regime from any GPU configuration. Above that line — long-context KV that grows with sequence, or models too large to fit even with WSE-3-grade compression — the architecture is forced into MemoryX-style off-wafer staging that gives back much of the bandwidth advantage.

## 5. SambaNova SN40L: three-tier memory and Composition of Experts

SambaNova's SN40L (2024 Hot Chips) takes the orthogonal approach. Where Groq removes HBM and DRAM and Cerebras absorbs the chip into a wafer, SambaNova adds a third memory tier and uses it to serve large MoE models at lower cost than fitting them in HBM-only silicon would permit [SambaNova-SN40L].

The SN40L socket is a 2.5D CoWoS chiplet built on TSMC 5nm: two **Reconfigurable Dataflow Dies (RDDs)** plus co-packaged HBM in one package. The 1,040 Pattern Compute Units (PCUs) across the two RDDs deliver 638 BF16 TFLOPS. The memory hierarchy:

| Tier | Capacity / socket | Latency tier |
|---|---|---|
| On-chip PMU SRAM | 520 MiB | hot, ns-scale |
| Co-packaged HBM | 64 GiB | warm, hundreds of ns |
| DDR5 via DIMMs | up to 1.5 TiB | cold, μs-scale |

Standard product is the **SambaRack SN40L-16**, a 16-socket rack with 24 TiB of DDR5 across the rack — enough to hold the cold experts of a multi-trillion-parameter MoE model in low-cost commodity memory while keeping the hot experts in HBM and the active expert's weights and activations streaming through PMU SRAM [SambaN-RDU, SambaN-Rack].

The bandwidth ratio between tiers is the architecturally important ratio. PMU SRAM at sub-microsecond latency, HBM at ~hundreds of nanoseconds and ~TB/s aggregate, DDR5 at ~tens of GB/s aggregate — roughly two orders of magnitude between SRAM and HBM and another order of magnitude between HBM and DDR. A naive "swap experts in and out of HBM from DDR per request" would be hopeless: at DDR's bandwidth, loading a single 8 GB expert takes hundreds of milliseconds, dominating any serving SLA. The architecture pays only when the *expert hit rate* in HBM is high enough that DDR transfers happen out-of-band relative to the inference request — pre-warmed, speculatively staged, or batched across many requests with locality.

**Composition of Experts (CoE)** is the serving pattern this hardware was built for. A CoE deployment hosts a large library of expert sub-models — possibly hundreds of billions of parameters across all experts — in DDR; an upstream router activates the relevant subset per request; the activated subset streams through the memory tiers as the dataflow compiler routes it. The cost-per-token is dominated by the warmest tier the path actually uses, so CoE economics depend on hit-rate locality (similar inputs activating similar expert subsets, allowing the warm cache to amortize across requests). When the locality holds, CoE on SN40L undercuts the same model on HBM-only silicon by the ratio of DDR-$/GiB to HBM-$/GiB — substantial enough to anchor SambaNova's enterprise positioning. The MoE-inference machinery this depends on is developed at the algorithmic level in [§20/03-moe-inference](../20-distributed-inference/03-moe-inference.md); SN40L is one of the few production silicon points that maps cleanly onto the per-expert dataflow that EP-large serving requires.

The dataflow nature of the SN40L is a second axis of differentiation. A GPU executes a model as a sequence of kernel launches, each materializing the input activation in HBM, computing, and writing the output back; the per-kernel HBM round-trip is amortized only if the engine can fuse adjacent operations into a single kernel (the rationale behind FlashAttention, FlashInfer, and the megakernel direction discussed in [§10/01-attention-kernels](../10-engine-core/01-attention-kernels.md)). The SN40L's RDDs reconfigure the on-chip dataflow per layer, streaming activations directly between PCUs without round-tripping through HBM at every operator boundary. On layered workloads where the GPU's kernel-launch overhead is non-trivial (small batch, short sequence, MoE per-expert dispatch), this is the architectural source of SambaNova's claimed efficiency — though the same property makes the compiler more critical: an SN40L deployment depends heavily on SambaNova's proprietary compiler stack to route the dataflow well, with substantially less engine-level flexibility than a CUDA-graph-captured GPU pipeline ([§10/08-cuda-graphs-compilation](../10-engine-core/08-cuda-graphs-compilation.md)).

The roadmap successor **SN50** is publicly announced as a successor product line; specifications and shipping schedule are not in the public record at the level of independent confirmation as of May 2026, and production claims should be hedged accordingly.

## 6. Tenstorrent Wormhole and Blackhole: open RISC-V, developer-first

Tenstorrent occupies the developer-platform corner of the ASIC landscape — its differentiator is not raw spec-sheet competition with Blackwell but the openness and customizability of its programming model, plus a price point and packaging that encourage on-prem development rather than cloud-only access [TT-WH-Specs, TT-BH-Specs].

The architecture is built around the **Tensix core**: a tile of compute units (matrix engine, vector engine, packer/unpacker units) coordinated by *open* RISC-V "baby cores" that handle control flow, decompression, and tile choreography in software. The baby cores are the differentiator. On a GPU, equivalent control logic is hidden behind a vendor-proprietary microcode layer; on Tensix, it runs on standard RISC-V cores whose source-level programmability is exposed to the kernel developer.

The current production lineup:

| Product | Tensix cores | On-chip SRAM | Off-chip memory | Process | Off-board interconnect |
|---|---|---|---|---|---|
| Wormhole n150 | 80 | ~120 MB | 12 GB GDDR6 | GF 12 nm | 16× 200 G QSFP-DD (board-to-board mesh) |
| Wormhole n300 | 128 (2 ASICs) | ~192 MB | 24 GB GDDR6 | GF 12 nm | 16× 200 G QSFP-DD |
| Blackhole p100a / p150a / p150b | 140 | ~210 MB distributed | 32 GB GDDR6 (24 controllers) | TSMC 6 nm | 4× 800 G QSFP-DD |

Vendor-reported headline: Blackhole runs roughly 2–3× a Wormhole on FP8 workloads, partly reproduced via the ASPLOS '25 microbenchmarking paper [TT-BH-Bench], which characterizes the Blackhole compute and memory subsystem at the kernel level. The same paper's headline finding is consistent with Tenstorrent's positioning: Blackhole is competitive on per-kernel throughput at developer-grade workloads, with software stack maturity (the **TT-Metal** runtime and the tt-NN higher-level library) lagging the hardware's nominal capability.

The **TT-QuietBox** workstation (announced November 2025) is a four-Blackhole desktop product aimed at solo developers and small ML teams — a niche neither NVIDIA nor AMD targets at this price point [Reg-TT]. Tenstorrent's published positioning is explicit: Blackhole is *not* a hyperscale alternative to a Blackwell rack; it is a customizable platform for ML workloads where the user wants to write their own kernels in the open stack and deploy on hardware the team can buy outright.

The 4× 800 G QSFP-DD ports on Blackhole are a structurally important product choice. They expose Ethernet directly off the board with no proprietary scale-up fabric — a Blackhole cluster wires up as a flat Ethernet mesh rather than as a Tenstorrent-branded equivalent of NVLink. The trade-off is that scale-up bandwidth is bounded by Ethernet rather than NVLink-class pricing/performance, but the build-it-yourself fabric maps onto the same Ethernet infrastructure the rest of the data center uses. This is consistent with the open-hardware positioning: a customer who buys Blackhole boards gets a deployment story that does not depend on the vendor's proprietary networking stack. The ASPLOS '25 microbenchmark paper is explicit that the developer-facing performance of the on-chip kernels is competitive with mid-tier GPU silicon at FP8; the gap that remains is in software-stack maturity (matrix-engine kernel libraries, attention kernel coverage, tensor-parallel collectives over the Ethernet fabric) rather than in the hardware itself [TT-BH-Bench].

## 7. Furiosa RNGD: Tensor Contraction Processor and Korean enterprise serving

Furiosa is the Korean entrant in the inference ASIC market, and its product positioning maps cleanly onto a national-scale customer base rather than a global benchmark race. The RNGD card and its NXT RNGD Server productization are built around Furiosa's **Tensor Contraction Processor (TCP)** architecture, in which the compiler reasons about computation as tensor contraction primitives rather than the matmul-plus-activation tile that drives GPU codegen [Furiosa-RNGD].

The RNGD card specifications:

- 48 GB HBM3 at 1.5 TB/s
- 256 MB on-chip SRAM
- Dual-slot PCIe Gen5 ×16
- 180 W TDP

The **NXT RNGD Server** (general availability September 25, 2025) packages eight RNGD cards into a single 3 kW chassis: 4 PFLOPS FP8 per server, 384 GB HBM3 aggregate at 12 TB/s, 1 TB DDR5 host memory [Furiosa-Server]. Vendor-reported throughput: approximately 3,000 tokens/second average per card at 180 W on small-model workloads; production-scale reproduction is limited and the figure should be read as illustrative rather than benchmark-grade.

The most-cited production deployment is **LG AI Research running EXAONE 3.5 32B inference on RNGD hardware** [Reg-Furiosa] — a strategic anchor that also illustrates the broader pattern. Korean enterprise customers (chaebol conglomerates with sovereign-AI ambitions, government-funded research labs, telecoms with on-prem inference needs) are prioritized customers for non-NVIDIA silicon as a hedge against US export controls and supply concentration. Furiosa's pre-IPO funding round in 2025 was reported to lean heavily on this domestic anchor base, with international expansion as a roadmap aspiration rather than a current production reality. The architecture's longer-term differentiator — TCP-style compilation as a route to better mid-scale efficiency than a generic GPU — is technically interesting but production-confirmed only at the LG-EXAONE deployment scale and similar anchor accounts.

The TCP architecture is worth one paragraph of unpacking. A conventional GPU compiler lowers a model to a sequence of matmul + elementwise tiles; the matmul shape, the activation function, and the operator boundaries are the units the compiler reasons about. A tensor-contraction-first compiler instead represents the model as a graph of arbitrary-rank tensor contractions — a generalization of matmul to rank-N tensors with arbitrary contraction indices — and decides at compile time how to factor each contraction into hardware-native primitives. The claimed advantage is that contractions which a matmul-first compiler would need to materialize as a sequence of (reshape, matmul, reshape) round-trips can be fused into a single hardware operation when the compiler reasons about them as a single contraction. Whether this advantage compounds into measurable per-token efficiency at scale is a vendor-claimed result; independent reproduction is limited to the deployment partners.

The Furiosa story also illustrates a broader pattern in mid-scale ASIC economics. The 180 W TDP per RNGD card is roughly 1/3 of an H100 SXM5's TDP, and the dual-slot PCIe Gen5 form factor lets RNGD fit into commodity server chassis without specialized cooling or power infrastructure. For deployments where commodity-server compatibility matters more than rack-density optimization — the typical enterprise data center, as opposed to a hyperscaler's purpose-built AI hall — this is a meaningful packaging win. Whether that win translates into market share at scale depends on the same factors as Tenstorrent's: software-stack maturity, model-coverage breadth, and the customer's willingness to operate two parallel inference stacks (one on RNGD, one on whatever GPUs the existing fleet runs).

## 8. Etched Sohu: the transformer-only ASIC that has not shipped

Etched announced the Sohu ASIC in June 2024 as a *single-architecture* inference chip: TSMC 4nm, 144 GB HBM3E, programming model that admits only the transformer forward pass and cannot run non-transformer architectures (no Mamba, no SSM, no CNN, no LSTM) [Etched-Tweet]. The marketing claim was 500,000 tokens/second on Llama-3 70B in an 8-Sohu server. The pitch reads as an extreme expression of architectural specialization: by burning the transformer arithmetic into silicon, the company claimed perf-per-watt advantages no general-purpose accelerator could match.

The status as of May 2026, approximately twenty-two months after announcement [Etched-Status, DCD-Etched]:

- **No shipping silicon to external customers.**
- **No third-party benchmarks.**
- **No published inference-provider production numbers.**
- A $500M raise at a $5B valuation occurred in 2024 [DCD-Etched].
- All performance claims remain unverified vendor numbers.
- Architecturally, the chip permanently cannot run non-transformer models — no Mamba / SSM, no CNN, no LSTM, no future hybrid architecture that diverges from the transformer forward pass.

The footnote-level treatment in this chapter is deliberate. Sohu illustrates an end-state of the niche-defense playbook, where the niche chosen — transformer-architecture lock-in — is so specialized that any execution slip jeopardizes the entire premise: by the time the chip ships, the transformer architecture itself may have evolved (new attention variants, new positional schemes, hybrid Mamba-transformers per [§30/03-attention-variants](../30-kv-cache/03-attention-variants.md)) in ways the silicon cannot support, and the perf-per-watt advantage erodes in proportion. The architectural restriction is permanent — the chip cannot be repurposed — which makes the time-to-market window extremely narrow. Whether Sohu eventually ships to external customers or remains a venture-funded press release is unresolved as of May 2026; the entry exists here to flag the structural risk that single-architecture ASICs face when their software ecosystem moves faster than their tape-out cycles.

The contrast with the SambaNova approach is informative. SambaNova also bets on architectural specialization, but its specialization is at the *dataflow-graph* level — the RDDs reconfigure per layer, so a model that compiles to a different graph runs on the same silicon with a recompile. Etched's specialization is at the *operator* level: the transformer attention and FFN structures are baked into the hardware datapath itself, with no mechanism to support a graph that does not match. The two approaches share the rhetoric of "specialized inference silicon" but occupy opposite ends of the flexibility spectrum, with the rhetoric obscuring the difference more than the engineering does.

## 9. Comparative summary

The headline rule: each product wins on the dimension it picked, and loses on the others.

| Product | Niche | Memory model | Peak compute | Production scale (May 2026) | Differentiator |
|---|---|---|---|---|---|
| Groq TSP (GroqCloud) | low-latency decode SaaS | SRAM-only, 230 MB / chip | not the headline | shipping at scale via GroqCloud; multi-rack racks for frontier models | deterministic dataflow, plesiosynchronous |
| NVIDIA Groq 3 LPX | decode FFN/MoE co-processor | SRAM-only, 128 GB / 256-LPU rack | 315 PFLOPS FP8 / rack | **announced, not shipped** | integrated into Vera-Rubin NVL72 + LPX heterogeneous racks |
| Cerebras CS-3 | highest tok/s/user, latency-critical | wafer-scale 44 GB SRAM | ~125 PFLOPS / box | shipping; >40M tok/s aggregate fleet (Q4 2025, vendor-reported) | wafer-scale die, ~25 kW / box |
| SambaNova SN40L | enterprise large-MoE serving | 520 MiB SRAM + 64 GiB HBM + 1.5 TiB DDR5 | 638 BF16 TFLOPS / socket | shipping (SambaRack SN40L-16); SN50 announced | three-tier memory, Composition of Experts |
| Tenstorrent Blackhole | developer / customizable platform | 210 MB SRAM + 32 GB GDDR6 | mid-tier (vendor) | shipping (boards, TT-QuietBox workstation Nov 2025) | open RISC-V baby cores, on-prem developer pricing |
| Furiosa RNGD | Korean enterprise / mid-scale | 256 MB SRAM + 48 GB HBM3 | 4 PFLOPS FP8 / 8-card server | shipping (NXT RNGD Server GA Sept 2025); LG EXAONE deployment | TCP architecture, sovereign-AI anchor |
| Etched Sohu | transformer-only acceleration | 144 GB HBM3E (claimed) | 500K tok/s Llama-3 70B / 8-server (claimed) | **no shipping silicon, no third-party benchmarks** | architecturally locked to transformer forward pass |

The market-structure read on the table: only Cerebras, SambaNova, Tenstorrent, and Furiosa are running customer-facing inference on their own silicon at meaningful scale today. Groq's TSP is shipping production traffic but only via the GroqCloud SaaS layer; the LPX productization extends Groq's architecture into NVIDIA's rack-scale platform but is unshipped. Etched is a public reminder that announcements and shipping silicon are not the same thing.

A second cut, organized by *what the customer is actually paying for*:

| Customer pays for | Product(s) |
|---|---|
| Tokens, via API | GroqCloud (TSP), Cerebras inference cloud |
| Rack-scale inference appliance | SambaRack SN40L-16; (planned) Vera-Rubin + LPX rack |
| Boards, build your own cluster | Tenstorrent Wormhole / Blackhole, Furiosa RNGD card |
| On-prem developer workstation | Tenstorrent TT-QuietBox |
| Enterprise on-prem server | Furiosa NXT RNGD Server (8-card) |
| (Nothing yet) | Etched Sohu |

The product-shape table tracks the niche-defense table closely but not perfectly: Cerebras and Groq both sell tokens, but Groq's underlying chip is shipping at scale to Groq's own data centers while Cerebras is more vertically integrated still. Tenstorrent and Furiosa both sell hardware, but to different customer segments — Tenstorrent's open-RISC-V positioning attracts research labs and ML systems engineers, while Furiosa's TCP-and-server positioning attracts enterprise IT buyers who want a deployment-ready box. The shape of the product, not just its peak FLOPs, is part of the niche.

## 10. The production-trajectory question

These products are at meaningfully different points in the production-readiness cycle:

```
shipping at scale ──── shipping to anchor ──── announced/ramping ──── press release
                       customers
        │                       │                        │                      │
   GroqCloud (TSP)        SambaRack SN40L-16     NVIDIA Groq 3 LPX        Etched Sohu
   Cerebras CS-3          Furiosa NXT RNGD       SambaNova SN50           
   inference cloud        Server (Sept 2025)     Cerebras inference 
                          Tenstorrent            scale-out (Q4 2025
                          Blackhole boards       buildout)
                          TT-QuietBox
                          (Nov 2025)
```

Five forces shape whether any of the products in this chapter consolidate into long-term market positions or remain niche occupants:

1. **The pace of GPU peak-spec improvement.** Hopper → Blackwell delivered ~2× peak FLOP/s and ~2× HBM bandwidth in roughly two years; Blackwell → Rubin is targeted to deliver another ~2× by 2027 ([§70/01-nvidia-roadmap](01-nvidia-roadmap.md)). A startup ASIC that wins on a 5-10× advantage on a specific workload at announcement-time can lose half that margin on each NVIDIA generation, and the niche must either deepen (acquire more workload-specific advantage per cycle) or consolidate (lock in customer commitments before the GPU catches up).

2. **The cost of software-stack maintenance.** vLLM, SGLang, and TRT-LLM together carry the inference-engine load on GPU ([§80](../80-oss-deep-dives/00-overview-comparison.md)); each ASIC vendor maintains some equivalent stack of its own. The maintenance burden grows as model architectures proliferate (dense, MoE, hybrid Mamba-transformer, multimodal, reasoning models with long thinking budgets per [§60/01-test-time-compute](../60-adjacent-workloads/01-test-time-compute.md)). Vendors with smaller customer bases struggle to keep pace; the production gap between hardware capability and software-stack coverage is a reliable predictor of which products lose share.

3. **Power and rack-density constraints in production data centers.** Cerebras's ~25 kW per CS-3 box is a workable density today; gigawatt-scale clusters are reorganizing around higher per-rack power than that ([§70/01-nvidia-roadmap](01-nvidia-roadmap.md) on Stargate, xAI Colossus 2, etc.). Each ASIC competes for rack space against a density-and-power-optimized GB200 / GB300 / Rubin rack; at >$1B per data center the colocation calculus rewards consolidation.

4. **Sovereignty and supply-diversification demand.** Korea, the EU, and several Middle Eastern states have publicly stated AI-infrastructure-sovereignty goals; this is part of Furiosa's anchor demand and a source of latent demand for SambaNova, Tenstorrent, and Cerebras outside the US-hyperscaler buyer base. The size of this demand pool is hard to forecast but bounded — sovereignty buyers are willing to pay a premium for non-US-controlled silicon up to some ceiling, and the ceiling sets a floor for non-NVIDIA market share.

5. **The integration question that LPX raises.** If the NVIDIA Groq 3 LPX rollout succeeds, NVIDIA will have established a template for absorbing adjacent-architecture silicon as decode co-processors in heterogeneous racks. That changes the competitive frame for the rest of the startup ASIC market: Cerebras, SambaNova, Tenstorrent, and Furiosa would face the question of whether they want to remain standalone alternatives or seek a similar integration path with NVIDIA / AMD / a hyperscaler. As of May 2026 none of them has signaled such a path publicly, but the option exists and the LPX deal makes the option more visible than it was a year ago.

These forces do not all push the same direction. Sovereignty demand and customer-niche specialization push toward more independent ASICs; GPU-peak improvement and software-stack-maintenance cost push toward fewer. The May 2026 read is that the market is differentiating, not consolidating: each of the products covered in §§2-7 occupies a meaningfully different corner of the design space, and each has a customer base that justifies continued investment, but the long tail (Etched and similar single-architecture or single-claim entrants) is thinning.

## Current production state

As of May 2026 the startup ASIC market is dominated, in production-traffic terms, by GroqCloud (the SaaS layer over the original Groq TSP) and Cerebras inference cloud. Both serve specific niches that no GPU configuration matches at any operating point: Groq for SRAM-bandwidth-bounded decode at moderate model scale, Cerebras for highest-tok/s/user latency-critical applications. The aggregate token volume of these two clouds, while non-trivial, remains a small percentage of total industry inference traffic — the median production token still flows over Hopper GPUs in 2026, and the Blackwell ramp at hyperscalers makes that a moving target the startup cloud has to keep pace with. SambaNova ships SN40L hardware to enterprise on-prem CoE deployments — narrower in publicly-cited customer base but architecturally distinct — and Furiosa serves Korean enterprise and government customers at the low-volume but strategically anchored end of the market. Tenstorrent is the developer-platform entrant; its production deployments are smaller in aggregate token volume but its on-prem workstation product (TT-QuietBox) addresses a buyer segment NVIDIA does not target at this price point, and the open RISC-V architecture has accumulated genuine community engagement.

The most consequential structural change in this corner of the market between 2024 and 2026 is the NVIDIA-Groq IP licensing arrangement and the resulting NVIDIA Groq 3 LPX productization. The deal is not an acquisition — Groq remains an independent company — but it transforms the SRAM-only deterministic-dataflow architecture from a standalone GPU competitor into a *complement* to NVIDIA's Vera-Rubin platform: an FFN/MoE decode co-processor in heterogeneous racks where GPUs handle prefill and attention and LPX chips handle the weight-bandwidth-bound parts of decode. The architecture is announced but not shipped as of May 2026; whether the heterogeneous co-processor pattern reaches production maturity, and whether the rest of the startup ASIC market follows it as a template (the NVIDIA-LPX pattern as a 2026 update of the 2019 NVIDIA-Mellanox pattern), is the open structural question for the next product cycle.

The careful read on the rest of the field. Etched Sohu has not shipped silicon to external customers in roughly twenty-two months since announcement, and the architecturally permanent transformer-only restriction means its time-to-market window narrows as transformer variants and hybrid architectures proliferate ([§30/03-attention-variants](../30-kv-cache/03-attention-variants.md)). The May 2026 status as captured in [Etched-Status] reflects no change in this trajectory: a $5B paper valuation, no shipping silicon, no third-party benchmarks, and a permanent architectural restriction that grows more constraining as the field's frontier moves. The Cerebras inference numbers, while genuinely category-leading on tok/s/user, are weaker on long-context workloads and rest on a proprietary software stack that has not converged with vLLM / SGLang / TRT-LLM ([§80](../80-oss-deep-dives/00-overview-comparison.md)). SambaNova's CoE economics depend on locality assumptions that not every MoE workload satisfies, and the SN50 successor's specifications are public-roadmap rather than confirmed shipping silicon. Furiosa's strategic anchor (Korean sovereign-AI demand) is durable but bounded in scale. Tenstorrent's open RISC-V differentiator is real but principally pays off in hand-tuned workloads where developer customizability matters more than per-rack token cost. Read together, the bottom line is that *startup ASIC* in 2026 means *defending a niche* — not competing with the hyperscaler in-house silicon of [§70/03-asics-hyperscaler](03-asics-hyperscaler.md) at $1T-scale, and not competing with NVIDIA's Hopper / Blackwell / Rubin trajectory of [§70/01-nvidia-roadmap](01-nvidia-roadmap.md) on raw spec sheets, but offering a corner of the inference market a serving experience that the GPU-default cannot match at the chosen operating point.
