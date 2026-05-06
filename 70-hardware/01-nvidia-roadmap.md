# The NVIDIA Roadmap: Hopper, Blackwell, Rubin

**After reading this chapter, the reader will be able to:**

- Place the three NVIDIA cohorts that matter for 2026 inference — Hopper (H100/H200, the installed base), Blackwell (B200/B300 and GB200/GB300 NVL72, the active ramp), and Rubin (R100, Rubin CPX, Rubin Ultra NVL576, near-production) — on a single timeline of die, package, rack, and fabric, and explain what each generation actually buys for serving.
- Trace the Tensor Core generations (4th-gen on Hopper, 5th-gen on Blackwell with tcgen05/TMEM) and the precision-format ladder (FP8 → NVFP4 → MXFP4) that doubles peak throughput per node every roughly eighteen months, and connect each format to the kernel software that exploits it.
- Derive the compute roofline, memory-bandwidth roofline, ridge point, NVLink-vs-compute balance, and effective batch for compute-bound decode for each generation, and use those numbers to read the structural shift from "GPU as the unit of capacity" (Hopper) to "NVL72 rack as the unit of capacity" (Blackwell) to "disaggregated prefill/decode silicon" (Rubin CPX + Groq 3 LPX).

The previous chapter ([§00/03-gpu-hardware-primer](../00-foundations/03-gpu-hardware-primer.md)) introduced the GPU execution model and named the per-generation primitives — TMA, WGMMA, tcgen05, TMEM, NVLink 4 / 5 — at the level needed to read kernel descriptions. This chapter is the systems-level deep-dive: what a Hopper SM actually contains, why GB200 NVL72 is treated as one logical accelerator instead of 72 GPUs, what NVFP4 is and how it differs from MXFP4, and what changes structurally when Rubin's Vera-Rubin NVL144 ships with dedicated Rubin CPX prefill silicon and Groq 3 LPX decode coprocessors. The chapter closes with the hardware feature that already shapes 2026 production but rarely makes the marketing slides: the gap between Hopper's installed base (where most production tokens still flow) and Blackwell's ramp (where the headline benchmarks are run).

---

## 1. The three-cohort transition (May 2026)

At any moment a fleet operator's inventory spans three cohorts. The 2026 mapping:

| Cohort | Status | Generations | Token share (est.) |
|---|---|---|---|
| Installed base | shipping volume since 2022–24 | A100, H100, H200, GH200 | majority of tokens served |
| Active ramp | shipping volume from 2024-Q4 | B200, GB200 NVL72, GB300 NVL72 | most frontier-lab production |
| Near-production | sampling / first racks | R100, Vera-Rubin NVL144, Rubin CPX, Groq 3 LPX | preview, MLPerf demos |

Token-share figures are rough industry estimates, not survey data; the structural claim — that the *median* production token in mid-2026 still flows through a Hopper SM — is widely reported in cloud-GPU price telemetry (vendor-side estimates from CoreWeave, Vast, Lambda) and is consistent with NVIDIA's own FY26-Q3 disclosure that Hopper revenue had not yet been displaced by Blackwell at the unit level [see [NV-FY26Q3](../papers.md#nv-fy26q3)]. The Blackwell ramp accelerated through 2025 and is the default at frontier-lab production buildouts; Rubin moves to volume in H2 2026.

The structural takeaway for the rest of this book: software targets all three cohorts simultaneously. FA-3 ([FA-3](../papers.md#fa-3)) is the dominant attention kernel by token because its installed base is Hopper; FA-4 ([FlashAttn-4](../papers.md#flashattn-4), shipped March 2026) is Blackwell-only; FlashInfer's BSR substrate ([FlashInfer](../papers.md#flashinfer)) is what lets one engine serve both. The same multi-cohort fact is why the parallelism strategy chapter ([§20/01-parallelism-strategies](../20-distributed-inference/01-parallelism-strategies.md)) develops separate operating points for "8-way Hopper HGX" and "72-way GB200 NVL72."

---

## 2. Hopper deep-dive (H100 / H200 / GH200)

The Hopper architecture (Compute Capability 9.0) launched on the GH100 die in 2022 and is the dominant production GPU through 2026. The deep-dive matters because the Hopper SM is the model the kernel writer carries in their head when reading any 2024–2026 attention kernel.

### 2.1 The GH100 die and the H100 SXM5

Hopper's GH100 die is fabricated on TSMC 4N (a custom 5 nm process), 814 mm², 80 billion transistors. The H100 SXM5 SKU enables 132 of the die's 144 Streaming Multiprocessors (the others disabled for yield). Per-SM resources:

- 128 FP32 CUDA cores, 64 FP64 cores, 64 INT32 cores
- 4 fourth-generation Tensor Core units (one per warp scheduler quadrant)
- 4 warp schedulers (one warp issued per cycle each)
- 256 KB combined L1/shared, 228 KB usable as shared memory
- 65,536 × 32-bit registers, 64 resident warps maximum
- 1 Tensor Memory Accelerator (TMA) unit

Off-die: 50 MB L2 (shared across all 132 SMs), 80 GB HBM3 at 3.35 TB/s, and 18 NVLink 4 lanes aggregating to 900 GB/s bidirectional per GPU. Peak throughput: 989 TFLOP/s BF16 dense, 1,978 TFLOP/s FP8 dense. TDP 700 W. The full-die GH100 (used in H100 NVL and some HPC SKUs) enables more SMs but the SXM5 SKU is the production default [see [NV-H100-WP](../papers.md#nv-h100-wp), [NV-Hopper-Blog](../papers.md#nv-hopper-blog)].

### 2.2 What 4th-gen Tensor Cores brought

Hopper's headline contributions are four hardware features that together set the kernel-engineering pattern that FA-3 codified:

**TMA (Tensor Memory Accelerator).** A per-SM hardware engine for asynchronous bulk copies between HBM and shared memory. Before TMA, every byte transfer cost per-warp instruction-issue bandwidth. With TMA, one instruction issues a tile copy and the warp can immediately schedule Tensor Core work on a *different* tile. This is the structural prerequisite for FA-3's producer/consumer warp specialization.

**WGMMA (warp-group MMA).** A single asynchronous instruction issued by a warp group of 4 warps (128 threads) drives the Tensor Core through a $64 \times M_n \times 16$ matmul-accumulate, where $M_n$ varies with operand precision. Asynchrony lets multiple WGMMAs pipeline; FA-3's "ping-pong" pattern is two warp groups alternating WGMMA issue.

**FP8 (E4M3 / E5M2).** First Tensor Core support for 8-bit float, paired with the original Transformer Engine library which manages per-tensor scale factors automatically. The decode-bandwidth win comes from halving weight bytes rather than doubling FLOP/s; for memory-bound decode, the operating point at FP8 sees ~2× tokens/s at fixed batch versus FP16.

**DSMEM and thread-block clusters.** A *cluster* of 2–16 thread blocks the scheduler co-locates onto one Graphics Processing Cluster (GPC), and which can address each other's shared memory as Distributed Shared Memory (DSMEM). This adds a new tier between per-block shared and global L2: cross-block shared at ~1 TB/s. FA-3 uses DSMEM only sparingly; the feature is more aggressively exploited by future kernels.

### 2.3 H100 vs H200 vs GH200

The H100 → H200 transition (late 2024) reuses the GH100 die and upgrades only the memory: from 80 GB HBM3 at 3.35 TB/s to 141 GB HBM3e at 4.8 TB/s. Peak compute is unchanged. The +43% HBM bandwidth translates near-linearly into decode tokens-per-second at fixed batch on bandwidth-bound workloads — the dominant decode regime — and is the pure structural reason H200 displaced H100 as the default for new decode-heavy clusters from Q1 2025 onward. The increased capacity (+76%) also relaxes KV pressure: a 70B-class model at FP8 with extended GQA cache fits at higher batch on a single H200 than on an H100.

The GH200 Grace Hopper Superchip integrates one Grace CPU (72 Arm Neoverse V2 cores) with one H100 or H200 over NVLink-C2C at 900 GB/s — the first production CPU-GPU coherent-memory link, and the architectural precursor to GB200's Grace + 2× B200. GH200 is the platform of choice for workloads that need large host memory addressable from the GPU at near-NVLink bandwidth — long-context KV offload pioneered some of these patterns [see [§30/02-kv-tiered-offload](../30-kv-cache/02-kv-tiered-offload.md)].

### 2.4 What FP8 actually buys on Hopper

The 4th-gen Tensor Cores' headline is FP8 at $2 \times$ FP16 throughput (1,978 vs 989 TFLOP/s on H100). For inference, the FLOP-doubling rarely shows up directly because production decode is memory-bound, not compute-bound. The win that *does* show up is the halving of weight-byte traffic: at FP8 the weights are 1 byte per parameter instead of 2, so a memory-bound decode reads half as much per token and runs at $\sim 2\times$ tokens-per-second at fixed batch. This is the structural reason FP8 has been near-universal for Hopper production decode since 2024 and the reason H100/H200 throughput tables are typically quoted at FP8 rather than FP16.

The Transformer Engine v1 library handles per-tensor scale management: a calibration pass picks per-tensor scales, and during inference the engine applies the scale on each tensor's load and the inverse scale on each store. Quantization-aware training has made post-training FP8 inference accuracy near-lossless on most production models; the residual cases (small models, very long context) usually retain FP16 for a single layer or two. The full quantization toolchain story is in [§10/04-quantization](../10-engine-core/04-quantization.md).

---

## 3. Blackwell deep-dive (B200 / GB200 NVL72 / GB300)

Blackwell (Compute Capability 10.0) is the active ramp. The architecture's structural innovation is that it is *not* a single die: B200 is a dual-die reticle-stitched package, and the rack — GB200 NVL72 — is treated by software as one logical accelerator. The die-level deep-dive matters because everything from Tensor Core programming model to NVLink topology changes.

### 3.1 The B200 package

B200 is two GB100 dies on a single package, joined by a 10 TB/s die-to-die link (NV-HBI, the Nvidia High-Bandwidth Interface), each die paired with four HBM3e stacks. Total package: 192 GB HBM3e, 8 TB/s aggregate bandwidth, ~2.25 PFLOP/s BF16 dense, ~4.5 PFLOP/s FP8, ~9 PFLOP/s NVFP4 dense (vendor-reported peak; sustained NVFP4 utilization in microbenchmarks lands around 70% of headline [see [ASPLOS-Blackwell](../papers.md#asplos-blackwell)]). 18 NVLink 5 lanes per package, 1.8 TB/s bidirectional per GPU. TDP scales by SKU: B200 SXM at 1 kW, GB200 (one Grace + two B200 on one board) at 2.7 kW total, GB300 at 1.4 kW per GPU [see [NV-Blackwell-Arch](../papers.md#nv-blackwell-arch), [NV-Blackwell-PR](../papers.md#nv-blackwell-pr)].

The dual-die packaging is the asymmetric-scaling fact that drove FA-4: Blackwell's compute scales faster than its on-chip transcendental units, so attention's softmax `exp` becomes a structural bottleneck unless replaced. FA-4's polynomial `exp2` on FMA pipelines is the workaround [see [FlashAttn-4](../papers.md#flashattn-4) and [§10/01-attention-kernels](../10-engine-core/01-attention-kernels.md)].

### 3.2 5th-generation Tensor Cores: tcgen05 and TMEM

The 5th-gen Tensor Core programming model is materially different from Hopper's WGMMA. Two new primitives:

**tcgen05.** The replacement for WGMMA. Where WGMMA accumulated into per-thread registers, tcgen05 accumulates into Tensor Memory (TMEM), a new on-chip buffer separately addressed by the Tensor Cores. The kernel writer sees a deeper async pipeline:

1. HBM → shared (TMA, as on Hopper).
2. Shared → TMEM (input staging, Blackwell-new).
3. TMEM → Tensor Core compute (tcgen05).
4. Tensor Core → TMEM (accumulator residence).
5. TMEM → shared/registers (drain to where the kernel needs it).

Each stage is async; the kernel writer schedules across them explicitly via barrier-and-arrival-token primitives. This is why FA-4 is built in CuTe-DSL ([CuTe-DSL](../papers.md#cute-dsl)) rather than the previous CUTLASS-template style — the explicit async pipeline does not fit the template-instantiation model.

**TMEM (Tensor Memory).** A new on-chip buffer per SM, sized at 256 KB on B200, used as the residence for matmul accumulators. Where WGMMA's accumulator registers were per-thread (and therefore competed with the rest of the kernel for register-file pressure), TMEM is a separate addressable region that does not consume registers. This decouples accumulator size from per-thread register budget — the structural enabler of larger attention tiles than Hopper allowed.

### 3.3 NVFP4 and the precision-format ladder

NVFP4 is the flagship inference format on Blackwell. Mechanically: weights and activations are stored as 4-bit floats (E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit) in *micro-blocks of 16 elements*, with each block carrying a per-block scale stored at FP8 E4M3. A second-level scale (one FP32 per tensor) recovers global dynamic range. Per-element storage cost is therefore $4 + 8/16 = 4.5$ bits, plus the negligible per-tensor FP32. Compared to MXFP4 — the OCP-standardized format with 32-element blocks and FP8 E8M0 scales — NVFP4's smaller blocks and finer-grained scale tame quantization noise, recovering accuracy that MXFP4 leaves on the table. NVIDIA's own pretraining experiments report NVFP4 matching FP8 baselines at 12B / 10T tokens with the OAS+MBS recipe, closing the MXFP4-vs-NVFP4 accuracy gap from ~10% to <1% [see [NVFP4-Pretraining](../papers.md#nvfp4-pretraining), [NVFP4-Inference](../papers.md#nvfp4-inference)].

The Tensor Core hardware path: 5th-gen Tensor Cores execute NVFP4 matmuls natively, with the per-block FP8 scale folded into the accumulator update. MXFP4 is also supported — Blackwell's Tensor Cores run both — but the per-block-of-32 scale and the E8M0 (exponent-only, no mantissa) format leave more error per block than NVFP4. The OpenAI gpt-oss release ([GPT-OSS-MXFP4](../papers.md#gpt-oss-mxfp4)) ships native MXFP4; most NVIDIA-partner deployments choose NVFP4 with the Four-Over-Six adaptive-block-scaling extension when the marginal accuracy matters [see [Four-Over-Six](../papers.md#four-over-six)].

The structural consequence: NVFP4 doubles peak FLOP/s vs FP8 at fixed silicon and halves weight-byte traffic vs FP8, so a memory-bound decode at FP8 sees roughly another ~2× tokens/s at fixed batch when transitioned to NVFP4. For Blackwell-default deployments, decode TPS at fixed batch on the same model typically lands 3–4× that of Hopper FP8. The full quantization story — calibration, the four-over-six adaptive-block extension, QAT and QAD — is in [§10/04-quantization](../10-engine-core/04-quantization.md).

### 3.4 GB200 / GB300: the package, not the chip

GB200 is *not* a chip; it is a board with one Grace CPU and two B200 packages, joined by NVLink-C2C at 900 GB/s. The Grace + 2× B200 trio shares a single coherent address space — *Extended GPU Memory* (EGM) — through which the GPU's CUDA kernels can address the Grace's LPDDR memory at C2C bandwidth. This is the production form-factor of the GB200 NVL72 building block: 18 GB200 boards = 18 Grace + 36 B200, plus another 18 Grace + 36 B200 in the second half of the rack = 36 Grace + 72 B200 per NVL72.

GB300 (Blackwell Ultra) reuses the GB200 NVL72 chassis pin-compatibly. The chip differs: 288 GB HBM3e per GPU (12-high HBM3e stacks), 8 TB/s, ~15 PFLOP/s NVFP4 dense (1.5× over B200), ~2× attention throughput per chip via increased on-chip transcendental capacity, 1.4 kW per GPU. Rack-level: ~1.1 EFLOP/s NVFP4 (vs ~1.4 EFLOP/s for GB200 NVL72 — the increase per chip is offset by the same chip count but different precision mix) and integrated ConnectX-8 SuperNIC giving 800 Gb/s scale-out per GPU [see [NV-Blackwell-Ultra](../papers.md#nv-blackwell-ultra), [NV-GB300-Page](../papers.md#nv-gb300-page), [Introl-B300](../papers.md#introl-b300)]. CoreWeave's first GB300 instances opened on August 19, 2025; mainstream ramp through H2 2025–early 2026.

### 3.5 Non-data-center Blackwell SKUs

Two Blackwell-class non-data-center products complete the lineup. **DGX Spark / DGX Workstation** (March 2026) ships the GB10 Superchip: one Vera-class CPU plus a single Blackwell GPU, 128 GB unified memory, 1 PFLOP/s NVFP4, 240 W desk-side. **Jetson AGX Thor** (October 2025) is the Blackwell-class edge SKU: 2,070 TFLOP/s NVFP4 with sparsity, 128 GB, 40–130 W [see [NV-DGX-Spark](../papers.md#nv-dgx-spark), [NV-Jetson-Thor](../papers.md#nv-jetson-thor)]. These cohabit the Tensor Core programming model with the data-center SKUs but ship without NVLink-domain serving and so do not enter the rack-scale story below.

---

## 4. NVL72 architecture: the rack as one logical GPU

The structural shift from Hopper to Blackwell is not the chip; it is the rack. GB200 NVL72 (and pin-compatible GB300 NVL72) is the unit of capacity planning at frontier-lab production: 72 B200 GPUs and 36 Grace CPUs in one liquid-cooled chassis, all connected by 5th-gen NVSwitch into a single 72-way NVLink fabric. Software treats the rack as one logical accelerator.

### 4.1 The rack envelope

A GB200 NVL72 rack contains:

- **18 compute trays**: each with two Grace + four B200, total 36 Grace + 72 B200.
- **9 switch trays**: each with two NVSwitch ASICs, total 18 NVSwitch ASICs forming a fat-tree NVLink fabric with full bisection bandwidth.
- **30 TB of fast memory**: 72 × 192 GB HBM3e = 13.8 TB GPU-side, plus 36 × ~480 GB Grace LPDDR ≈ 17 TB CPU-side, all in one coherent address space (EGM).
- **1.4 EFLOP/s NVFP4** (vendor headline, dense; ~70% sustained on attention-shaped microbenchmarks).
- **130 TB/s aggregate intra-rack NVLink bandwidth** (72 GPUs × 1.8 TB/s per GPU).
- **~120 kW** total rack power, liquid-cooled.

The simplified topology view:

```
GB200 NVL72 rack
├── 18 compute trays (top + bottom halves)
│   └── each tray: 2 Grace CPUs + 4 B200 GPUs
│       Grace ──NVLink-C2C 900 GB/s── B200 (×2)
│       Grace ──NVLink-C2C 900 GB/s── B200 (×2)
├── 9 NVSwitch trays (rack midplane)
│   └── 18 NVSwitch ASICs, full bisection fabric
└── all 72 B200 reach all other 71 B200 at 1.8 TB/s
    (one or two NVSwitch hops; uniform bandwidth)
```

The first customer GB200 NVL72 shipped to CoreWeave in December 2024; HPE shipped its first GB200 NVL72 in February 2025; mass production through Q2–Q3 2025; broadly available across CoreWeave, Azure, OCI, and GCP by early 2026 [see [NV-GB200-Page](../papers.md#nv-gb200-page), [SGLang-GB200-1](../papers.md#sglang-gb200-1), [SGLang-GB200-2](../papers.md#sglang-gb200-2), [vLLM-DSR1-GB200](../papers.md#vllm-dsr1-gb200)].

### 4.2 The "one logical GPU" claim

Three properties make NVL72 a single accelerator from software's perspective:

1. **Uniform NVLink bandwidth.** Any pair of GPUs in the rack reaches each other at full per-GPU bandwidth (1.8 TB/s) through one or two NVSwitch hops. There is no NUMA-style locality penalty within the rack.
2. **No PCIe between compute units.** Every GPU↔GPU and every GPU↔Grace path is NVLink (5 or C2C); PCIe enters only for storage and out-of-band management.
3. **Single coherent EGM address space.** A GPU kernel can dereference a host-side LPDDR pointer through the Grace partition without explicit copy; the C2C runtime handles paging.

The consequence for parallelism: TP can span the full 72-way (vs 8-way on HGX H100); EP all-to-all uses the rack fabric directly with no IB hop; the attention kernel can place KV pages on any GPU's HBM and reach them at NVLink bandwidth. The SGLang and vLLM GB200 deployment posts ([SGLang-GB200-1](../papers.md#sglang-gb200-1), [vLLM-DSR1-GB200](../papers.md#vllm-dsr1-gb200)) document the parallelism layouts in practice — large-EP MoE serves DeepSeek-V3 / Kimi-K2 on a single rack with EP=64+ over the intra-rack fabric.

The structural shift this enables is developed in [§20/01-parallelism-strategies](../20-distributed-inference/01-parallelism-strategies.md) and [§20/03-moe-inference](../20-distributed-inference/03-moe-inference.md). The headline: for frontier MoE serving, "the rack is the unit" replaces "the GPU is the unit."

---

## 5. The Rubin roadmap

Rubin (Compute Capability 11.x) is the next architecture, in volume production at the chip level as of CES 2026 and shipping in racks in H2 2026. Three SKU families: R100 (data-center GPU), Rubin CPX (prefill-dedicated), and Rubin Ultra GR300 NVL576 (H2 2027). The complete platform is named *Vera-Rubin* after its companion CPU.

### 5.1 R100 and Vera-Rubin NVL144

R100 is the standard Rubin SKU: 288 GB HBM4 at >13 TB/s (vendor figure, exact per-package bandwidth not yet disclosed), ~50 PFLOP/s NVFP4, 336 billion transistors, dual-die package on a custom TSMC node. Vera is the companion CPU: 88 custom Arm "Olympus" cores, up to 1.5 TB LPDDR5X at 1.2 TB/s, 1.8 TB/s NVLink-C2C to each paired R100. Vera-Rubin NVL72 (72 R100 + 36 Vera) and Vera-Rubin NVL144 (144 R100 + 36 Vera) form the two rack-scale SKUs; the NVL144 form-factor doubles GPU count per rack via denser packaging. Aggregate intra-rack bandwidth: 260 TB/s scale-up. Vendor-claimed inference-per-rack improvement vs Blackwell: ~5× inference, ~3.5× training; these figures are headline NVIDIA numbers and have not been independently benchmarked at production scale as of May 2026 [see [NV-Vera-Rubin-Blog](../papers.md#nv-vera-rubin-blog), [NP-VeraRubin](../papers.md#np-verarubin), [STH-Rubin](../papers.md#sth-rubin)].

### 5.2 Rubin CPX: the first prefill-dedicated accelerator

Rubin CPX is the most architecturally distinctive Rubin SKU. A single-die GPU at ~30 PFLOP/s NVFP4, 128 GB GDDR7 (not HBM), and a hardware path that triples attention throughput per chip vs GB300. The format is critical: GDDR7 is cheaper-per-byte than HBM3e but lower bandwidth; CPX's hardware is sized for the prefill regime where compute dominates and cache pressure is moderate (one pass over weights, one pass over the prompt).

The marketing SKU is *Vera-Rubin NVL144 CPX*: a disaggregated rack with R100s for decode and CPX dies for prefill on the same NVLink fabric. This is the first NVIDIA accelerator explicitly designed around the prefill/decode split, the architecture pattern this book has been developing since [§00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md) and [§20/02-prefill-decode-disagg](../20-distributed-inference/02-prefill-decode-disagg.md). The structural argument it embodies: prefill is compute-bound and tolerates GDDR; decode is memory-bound and demands HBM. Hardware-level disaggregation lets the rack serve both regimes at lower TCO than uniform-HBM compute [see [NV-RubinCPX-Blog](../papers.md#nv-rubincpx-blog), [Reg-RubinCPX](../papers.md#reg-rubincpx)].

### 5.3 Rubin Ultra GR300 NVL576

Rubin Ultra (H2 2027) is the long-horizon SKU: a 4-chiplet GPU at ~100 PFLOP/s NVFP4 each (per-chip aggregate), 1 TB HBM4 per GPU package, 144 GPUs per rack with 576 dies total (4 chiplets × 144 packages). Rack power: ~600 kW, requiring direct-to-chip cold-plate liquid plus rear-door heat exchangers. Vendor-disclosed H2 2027.

### 5.4 NVIDIA Groq 3 LPX: heterogeneous decode silicon

GTC 2026 introduced a class of NVIDIA-stewarded silicon that does not look like a GPU. **Groq 3 LPX** (Language Processing Unit) is a 256-LPU package at 315 PFLOP/s FP8 with 128 GB SRAM (and *no* HBM), 40 PB/s on-chip bandwidth, and 640 TB/s scale-up bandwidth. It is sold as a decode coprocessor for the Vera-Rubin platform — the first heterogeneous-inference SKU NVIDIA has fielded. Fabrication on Samsung 4 nm; a $20 B IP licensing deal with Groq closed in December 2025 [see [NV-Groq3LPX](../papers.md#nv-groq3lpx), [Decoder-LPX](../papers.md#decoder-lpx)].

The architectural argument: a 70B decode at batch 1 reads ~140 GB of weights per token at FP16 (or ~17 GB at NVFP4); a Groq LPX rack with all weights resident in SRAM at 40 PB/s reads them in microseconds, achieving decode-TPS at very small batch that GPU-HBM cannot reach without much higher batch. The price is silicon-cost-per-byte; the cost-effective regime is single-tenant low-latency decode where high batch is unavailable. Whether this replaces GPU decode in 2026 production at scale is unsettled — the chips are sampling, not shipping in volume — but the *architectural direction* (memory-bandwidth specialization for decode, compute specialization for prefill) is the headline. The full ASIC story is in [§70/02-asic-ecosystem](02-asic-ecosystem.md); the heterogeneous-serving software story is [§20/05-heterogeneous-inference](../20-distributed-inference/05-heterogeneous-inference.md).

---

## 6. NVLink generations and NVLink Fusion

The interconnect roadmap matters because every TP all-reduce, every EP all-to-all, and every cross-stage KV transfer goes over NVLink within a node. The bandwidth ladder is the structural fact that makes "TP within a node, PP across nodes" the production-default layout in 2026 [see [NV-NVLinkScaling](../papers.md#nv-nvlinkscaling)].

| Generation | First product | Per-GPU BW (bidir) | Aggregate node BW | NVSwitch |
|---|---|---|---|---|
| NVLink 3 | A100 | 600 GB/s | 4.8 TB/s (8-GPU) | NVSwitch 2 |
| NVLink 4 | H100 / H200 | 900 GB/s | 7.2 TB/s (8-GPU HGX) | NVSwitch 3 |
| NVLink 5 | B200 / GB200 | 1,800 GB/s | 14.4 TB/s (HGX) / 130 TB/s (NVL72) | NVSwitch 4 |
| NVLink-next | Rubin (H2 2026) | ~3,600 GB/s expected | 260 TB/s (NVL144 quoted) | NVSwitch 5 |

Rubin's NVLink-next per-GPU figure is vendor-published headline; independent characterization will appear after first racks ship in H2 2026.

**NVLink Fusion** (announced May 2025, [NV-NVLinkFusion-PR](../papers.md#nv-nvlinkfusion-pr)) opens the scale-up fabric: NVIDIA licenses NVLink IP to partner silicon (MediaTek, Marvell, Alchip, Fujitsu, Qualcomm) so non-NVIDIA CPUs and ASICs can join an NVLink domain alongside NVIDIA GPUs. As of May 2026 no partner silicon has shipped into a production NVLink Fusion rack, but the licensing model is the structural opening of the fabric — the prerequisite for heterogeneous racks where, for example, a partner-CPU host shares EGM with NVIDIA GPUs. The strategic interpretation: NVIDIA prefers to keep the GPU monopolized but is willing to share the *fabric* to forestall a competitor scale-up standard [see [NV-NVLinkFusion-Page](../papers.md#nv-nvlinkfusion-page)].

---

## 7. Tensor Core generations

Each Tensor Core generation has roughly doubled effective throughput by halving operand precision and adding async / on-chip-buffer machinery. The summary table:

| Gen | Architecture | Key formats | New machinery | Notes |
|---|---|---|---|---|
| 1st | Volta (V100) | FP16 | mma instruction | First tensor-core matmul |
| 2nd | Turing | FP16, INT8 | — | Inference focus |
| 3rd | Ampere (A100) | TF32, BF16, INT8, FP16 | TF32 (transparent) | BF16 mainstreamed |
| 4th | Hopper (H100/H200) | FP8 (E4M3/E5M2), BF16, FP16 | TMA, WGMMA, DSMEM, async | Transformer Engine v1 |
| 5th | Blackwell (B200/GB300) | NVFP4, MXFP4/FP6, FP8, BF16 | tcgen05, TMEM, deeper async | Transformer Engine v2 |
| 6th | Rubin (R100) | NVFP4, lower-precision TBD | (Rubin-specific) | TE v3 expected |

The interpretive axis: each generation's headline FLOP/s number is roughly $2 \times$ the prior generation's at the same operand precision, *and* the new generation introduces a precision tier below the prior generation's lowest, doubling again in the new tier. So Hopper at FP8 ≈ 2× Ampere at FP8-equivalent (INT8); Blackwell at NVFP4 ≈ 2× Hopper at FP8. The bandwidth ladder grows ~2× per generation as well (3.35 → 4.8 → 8.0 TB/s per GPU package). The compute-to-bandwidth ratio therefore grows: ridge points move right, and the operating regime where decode is memory-bound expands.

---

## 8. Math: roofline, ridge point, effective batch, NVLink balance

The arithmetic that anchors every chapter. Quantities defined as in [§00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md): $P$ active parameters, $B$ batch, $S$ sequence, $b$ bytes per parameter element.

### 8.1 Compute roofline

Per-token decode FLOPs (the 2N rule): $F_{\text{decode}} \approx 2 P B$ FLOPs per iteration. Prefill FLOPs ignoring the attention quadratic: $F_{\text{prefill}} \approx 2 P B S$. The peak-FLOP rooflines:

| GPU | Peak NVFP4 | Peak FP8 | Peak BF16 |
|---|---|---|---|
| H100 SXM5 | — | 1,978 TFLOP/s | 989 TFLOP/s |
| H200 SXM5 | — | 1,978 TFLOP/s | 989 TFLOP/s |
| B200 (package) | ~9,000 TFLOP/s | ~4,500 TFLOP/s | ~2,250 TFLOP/s |
| GB300 (package) | ~15,000 TFLOP/s | — | — |
| R100 (vendor) | ~50,000 TFLOP/s | — | — |

### 8.2 Memory-bandwidth roofline

Decode at batch 1 reads $P \cdot b$ bytes of weights per token (every weight is touched once per token in the dense case). Decode time is bounded below by

$$T_{\text{decode}}^{\text{BW}} \;=\; \frac{P \cdot b}{\text{HBM BW}}.$$

For Llama-3.1-70B at FP8 ($P b = 70$ GB) on H100 (3.35 TB/s), $T \ge 21$ ms — peak decode rate ~48 tok/s/GPU at batch 1. On H200 (4.8 TB/s), the same model decodes at ~69 tok/s/GPU. On B200 (8 TB/s) at NVFP4 ($P b = 35$ GB), $T \ge 4.4$ ms, peak ~230 tok/s. The 4.8× B200/H100 ratio (versus the 1.4× H200/H100 ratio) is the structural benchmark gap.

### 8.3 Ridge point

Arithmetic intensity is FLOPs per byte of HBM traffic. The ridge point — the intensity at which compute and bandwidth bound balance — is

$$I_{\text{ridge}} \;=\; \frac{\text{peak FLOP/s}}{\text{peak HBM BW}}.$$

For B200 at NVFP4: $I_{\text{ridge}} \approx 9{,}000 \text{ TFLOP/s} / 8 \text{ TB/s} = 1{,}125$ FLOP/byte. For H100 at FP8: $I_{\text{ridge}} \approx 1{,}978 / 3.35 \approx 590$ FLOP/byte. Decode at batch 1 sits at intensity $\approx 2$ FLOP/byte (two FLOPs per parameter byte) — strictly memory-bound on every GPU. The kernel must batch up to bring intensity above ridge to be compute-bound.

### 8.4 Effective batch for compute-bound decode

Increasing batch raises FLOPs ($\propto B$) faster than bytes (weight bytes are amortized across the batch, KV bytes scale with $B$). For weight-dominated decode the operating point moves up with $B$ until ridge:

$$B^{\star} \;=\; \frac{\text{peak FLOP/s} \cdot b}{2 \cdot \text{HBM BW}}.$$

For H100 FP8: $B^{\star} \approx 1{,}978 \cdot 1 / (2 \cdot 3.35) \approx 295$. For B200 NVFP4: $B^{\star} \approx 9{,}000 \cdot 0.5 / (2 \cdot 8) \approx 281$. The required batch to drive decode compute-bound is roughly the same order of magnitude across generations because peak FLOPs and bandwidth scale together — and is much larger than typical production decode batch (32–128). The structural conclusion: production decode is memory-bound on every NVIDIA generation, which is why the bandwidth roofline (and KV math, and quantization) carries the chapters that follow.

### 8.5 NVLink BW vs compute (TP all-reduce balance)

Ring all-reduce of an activation tensor of size $A$ bytes across $T$ GPUs costs $\approx 2 (T-1) A / T$ bytes per GPU on the ring. The communication time at NVLink bidirectional bandwidth $W$ is $T_{\text{comm}} \approx 2(T-1) A / (T W)$. Per-layer activations for a transformer of hidden $d$ at batch $B$ and sequence $S$ are $A = B S d \cdot b$ bytes (FP16/FP8/etc.). Per-layer compute (the 2N rule applied to a single layer's $\sim 12 d^2$ params) at peak is $T_{\text{compute}} \approx 2 \cdot 12 d^2 \cdot B S / \text{peak FLOP/s}$.

For TP=8 on HGX H100 (NVLink 4 at 900 GB/s per GPU, 7.2 TB/s aggregate ring) and a 70B-class layer ($d = 8192$, FP8, $B=64$, $S=1$ for decode): $A = 64 \cdot 8192 \cdot 1 = 0.5$ MB. $T_{\text{comm}} \approx 1.0$ µs. Per-layer compute at FP8 peak: $T_{\text{compute}} \approx 2 \cdot 8 \cdot 10^8 \cdot 64 / 1.978 \times 10^{15} \approx 52$ µs. Communication is well below compute — TP=8 on H100 is structurally fine for decode at moderate batch, and the headline-rate decode losses to all-reduce are typically under 5%. On NVL72 with TP=72 on a 70B-class model, $T = 72$ inflates the all-reduce factor 2(T-1)/T → ~2.0 (vs 1.75 at T=8); per-GPU NVLink rises to 1.8 TB/s, so per-layer all-reduce stays in the ~1 µs regime — the structural reason rack-scale TP works without compute starvation.

For prefill at long context the same arithmetic flips: $A$ scales with $S$, so all-reduce time scales linearly with prompt length while compute scales as $S$ (linear) plus $S^2$ (attention quadratic). For long prompts the compute side wins back margin; for short prompts the all-reduce can become the bottleneck, motivating sequence parallelism [see [§20/04-long-context-inference](../20-distributed-inference/04-long-context-inference.md)].

### 8.6 NVFP4 vs FP8 throughput

Native Tensor Core throughput: NVFP4 is 2× FP8 dense at the same silicon, by direct hardware lane count. Operand bytes halve as well, so a memory-bound decode sees ~2× tokens-per-second at fixed batch when transitioning FP8 → NVFP4. The compound effect from Hopper FP8 to Blackwell NVFP4 is roughly $2 \times \text{(BW)} \times 2 \times \text{(precision)} \approx 4\times$ at fixed batch on bandwidth-bound decode, which matches widely reported B200-vs-H100 ratios within ~70% utilization factor.

### 8.7 Power-per-token

GB200 NVL72 dissipates ~120 kW at ~1.4 EFLOP/s NVFP4 (sustained at headline utilization). The compute-per-watt figure: $1.4 \times 10^{18} / 1.2 \times 10^5 \approx 12$ TFLOP/W. For a 70B-class model at NVFP4 decoding at batch 32, per-token energy at peak utilization: $E \approx 2 \cdot 70 \times 10^9 \cdot 32 / 12 \times 10^9 \approx 0.37$ J/token. At fleet utilization (~30–40% of peak in practice), the effective number is 1–1.5 J/token, and the rack at full duty produces on the order of $0.4 \times 10^6$ tok/s aggregate. These numbers anchor the rough $/M-token cost model: 120 kW × $0.05/kWh = $6/h power cost; at 0.4M tok/s that is 1,440 M tok/h, giving approximately **$0.004/M-token** in power cost alone — three orders of magnitude below the API list price of frontier models, confirming that capex amortization and margin dominate the total, not energy.

---

## Sidebar: GW-scale cluster economics

The production buildouts of 2025–2026 are no longer measured in racks; they are measured in *gigawatts of facility power*. The flagship sites:

- **xAI Colossus 2** (Memphis): announced 2 GW at full buildout, 555,000 GPUs as of January 2026, capex on the order of $18 B. World's largest single-site cluster on these reports [see [xAI-Colossus2](../papers.md#xai-colossus2), [SemiA-Colossus2](../papers.md#semia-colossus2)].
- **OpenAI Stargate / Abilene**: Phase 1 came online September 2025; the full program targets 8 sites and ~7 GW of facility power, with $400 B+ in committed financing across the partner consortium (OpenAI, Oracle, SoftBank, MGX). Cited capex ~$29 B per GW of facility power, blending GPU silicon, networking, building, cooling, and grid interconnect.
- **Meta Prometheus** (New Albany, Ohio): 1 GW, online 2026.
- **Meta Hyperion** (Louisiana): 5 GW planned, multi-year buildout.
- **Anthropic-AWS partnership**: announced 5 GW commit ([Anthropic-AWS](../papers.md#anthropic-aws)).

The structural shift these sites represent: training and inference were historically run on different fleets at different scales; GW-scale buildouts amortize across both, with inference taking the larger fraction of the steady-state load. The economic question — whether per-token serving margin clears the capex of $29 B/GW — is the implicit driver of every quantization, disaggregation, and KV-management technique this book catalogs. The cluster-scale story is in [§50/02-multi-region-and-cross-cluster](../50-cluster-systems/02-multi-region-and-cross-cluster.md) and [§50/03-cluster-economics](../50-cluster-systems/03-cluster-economics.md); the figures here are publicly reported and may be revised as buildouts complete.

## Sidebar: FA-3 still dominates the installed base

FlashAttention-4 ([FlashAttn-4](../papers.md#flashattn-4)) shipped March 2026 and is the state-of-the-art Blackwell attention kernel, achieving 1,605 TFLOP/s BF16 on B200 (~71% headline utilization), built in CuTe-DSL, using a polynomial `exp2` on FMA pipelines to dodge Blackwell's MUFU bottleneck. By every published benchmark FA-4 is the fastest attention kernel in production. And yet most production tokens in mid-2026 still flow through FA-3 (or its FlashInfer / FA-Hopper derivatives), not FA-4 — because the installed base is Hopper, not Blackwell.

The gap between *the SOTA kernel* and *the kernel that serves the median token* is a recurring fact in 2026 inference infrastructure. Two practical implications. First, engine maintainers (vLLM, SGLang, TRT-LLM, MLC) ship both backends and select per-architecture; FlashInfer's BSR substrate is the cross-architecture abstraction layer. Second, the per-paper headline ("X% faster on B200") has to be discounted against the Hopper-share of the fleet to land at fleet-average performance. The pattern repeats per generation: the last generation's kernel runs the median token until the next generation's installed base catches up.

---

## Current production state

As of May 2026, the production NVIDIA hardware inventory for LLM inference resolves into the three-cohort split: Hopper for the median token, Blackwell for frontier-lab production and the largest hyperscaler racks, and Rubin for the next-twelve-months ramp. The H200 has roughly displaced the H100 as the default for new decode-heavy clusters because its +43% HBM bandwidth converts near-linearly into decode TPS at fixed batch. GB200 NVL72 is the production unit at frontier labs (OpenAI, Anthropic, Google DeepMind, xAI, Meta) and at the largest hyperscaler buildouts (CoreWeave from December 2024; Azure, OCI, GCP through 2025); GB300 NVL72 began shipping in volume from August 2025 and is the default for new frontier-lab racks built H2 2025 and after. The Vera-Rubin platform (R100, NVL72/NVL144, Rubin CPX) is sampling at chip level as of CES 2026 and is expected in volume through H2 2026; Rubin Ultra GR300 NVL576 is on the H2 2027 roadmap.

The structural shift the past eighteen months have ratified is the move from "the GPU as the unit of capacity" to "the NVL72 rack as the unit of capacity." Frontier-scale serving — DeepSeek-V3 / V3.1 / V3.2, Kimi-K2, GPT-OSS-large, and Llama-4 frontier MoE — runs on multi-rack NVL72-class topologies with TP and EP intra-rack and PP and DP inter-rack. Capacity planning, cost modeling, and the parallelism-strategy decisions in [§20/01-parallelism-strategies](../20-distributed-inference/01-parallelism-strategies.md) are framed in those rack-scale units, not in per-GPU units. The next shift, visible but not yet ratified, is hardware-level prefill/decode disaggregation: Rubin CPX (GDDR7 prefill silicon) and Groq 3 LPX (SRAM-only decode silicon) are the first NVIDIA SKUs to embody the architectural argument that prefill and decode want different memory hierarchies. Whether software can exploit those splits at scale in 2026–2027 is the open question that [§20/02-prefill-decode-disagg](../20-distributed-inference/02-prefill-decode-disagg.md) and [§20/05-heterogeneous-inference](../20-distributed-inference/05-heterogeneous-inference.md) develop.

The kernel-software population mirrors the hardware split: FA-3 ([FA-3](../papers.md#fa-3)) carries the median Hopper token; FA-4 ([FlashAttn-4](../papers.md#flashattn-4)) is Blackwell-only and tracks Blackwell ramp; FlashInfer ([FlashInfer](../papers.md#flashinfer)) is the cross-engine abstraction layer that shields the engine from per-architecture kernel divergence; ThunderKittens, CuTe-DSL, TileLang, and Triton compete as the DSLs in which the next generation of kernels is written. NVFP4 is the dominant inference format on Blackwell-default deployments and is in active production ramp at frontier labs; FP8 remains the dominant Hopper inference format. The accuracy-recovery toolchain ([NVFP4-Inference](../papers.md#nvfp4-inference), [NVFP4-Pretraining](../papers.md#nvfp4-pretraining), [NVFP4-QAD](../papers.md#nvfp4-qad), [Four-Over-Six](../papers.md#four-over-six), [Quartet-II](../papers.md#quartet-ii)) is mature enough that NVFP4 inference at near-FP8 accuracy is routine, removing the residual reason to stay at FP8 on Blackwell.
