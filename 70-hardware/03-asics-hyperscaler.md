# Hyperscaler ASICs: TPU, Trainium, MTIA, Maia

**After reading this chapter, the reader will be able to:**

- Explain why each of the four largest cloud-and-model operators — Google, AWS, Meta, Microsoft — has built its own inference and training silicon, and place each accelerator family on a roofline against H100/B200 in [§00/03-gpu-hardware-primer](../00-foundations/03-gpu-hardware-primer.md).
- Read the architecture of a TPU v7 "Ironwood" superpod (chip → 4×4×4 cube → OCS-glued 9,216-chip pod) and an AWS Trn2 UltraServer (16-chip Trn2 instance → 64-chip NeuronLink domain), and connect each topology to the parallelism strategies of [§20/01-parallelism-strategies](../20-distributed-inference/01-parallelism-strategies.md).
- Identify which production workloads target which silicon (Gemini and Anthropic Claude on TPU; Anthropic on Trainium; Meta ranking and gen-AI on MTIA; OpenAI/Azure on Maia), and articulate the software-stack lock-in (XLA/Pathways, Neuron SDK, proprietary stacks) that determines whether a given model can move between them.

This chapter does not relitigate the GPU as the inference baseline; that role is filled by [§70/01-nvidia-roadmap](01-nvidia-roadmap.md) and, on the AMD side, by [§70/02-amd-and-non-nvidia-gpu](02-amd-and-non-nvidia-gpu.md). The job here is to characterize the *parallel* silicon track: chips designed and operated by hyperscalers for their own workloads, where the economic logic differs from the merchant-GPU economy in ways that have, over the last 18 months, produced four credible alternatives to NVIDIA at meaningful scale. As of May 2026, three of the four — Google TPU v7, AWS Trainium2, Microsoft Maia 200 — serve frontier-lab workloads in production; the fourth (Meta MTIA) serves Meta-internal ranking and is ramping toward gen-AI. The chapter also flags what each program is *not*: none of them runs vLLM, SGLang, or TRT-LLM unmodified, and the software-portability story is the structural risk.

## 1. The "in-house silicon" pattern

At trillion-dollar-scale workloads, the economics of merchant accelerators diverge from the economics of workload-optimized silicon. A hyperscaler running 1M+ accelerators for inference pays for every wasted FLOP and every excess byte of HBM. The merchant-GPU calculus — "buy the part with the broadest software ecosystem and the highest per-chip resale value" — is replaced by a build-vs-buy calculation in which custom silicon wins if the per-token cost reduction over a 4–6 year amortization window exceeds the design-and-fab NRE plus the cost of staffing a software-stack alternative to CUDA.

The pattern is recognizable across all four hyperscalers. Each starts with a narrow first-target workload, ships a minimum-viable accelerator for that workload, and broadens scope generation by generation as operational confidence accrues:

- **Google.** TPU v1 (2015) targeted internal inference for ranking and translation; TPU v2/v3 added training; v4 introduced optical reconfiguration ([Jouppi-TPUv4](../papers.md#jouppi-tpuv4)); v5e/v5p split into inference- and training-optimized SKUs; v6e Trillium and v7 Ironwood serve frontier-LLM training and serving.
- **AWS.** Inferentia (2019) targeted CPU-displacing inference; Trainium (2022) added training; Inferentia2 (2022) extended to LLM-class inference; Trainium2 (Dec 2024 GA) is the first AWS chip with a credible LLM-training claim; Trainium3 (announced Dec 2025) targets the Trn2 successor and the Anthropic-scale buildout.
- **Meta.** MTIA v1 (2023) targeted ranking and recommendation; v2 (2024) extended to inference and light training; the 2025 four-chip roadmap (300/400/450/500) extends MTIA into generative-AI workloads.
- **Microsoft.** Maia 100 (2023) targeted Azure-internal OpenAI inference; Maia 200 (announced Jan 2026) extends to FP4-class inference at frontier model size.

Common threads as of May 2026: all four have HBM3e on their most-recent generation (Ironwood, Trainium3, MTIA 400/450, Maia 200). All four have FP8 as a baseline tensor-core path. Only Maia 200 and Trainium3 match NVIDIA Blackwell on FP4-class formats; TPU v7 deliberately omits FP4. All four use a proprietary software stack that requires porting — not recompilation — for a model targeting CUDA. The risk is uniform: a workload built against vLLM/SGLang/TRT-LLM ([§80-oss-deep-dives/](../80-oss-deep-dives/)) does not move to TPU/Trainium/MTIA/Maia without engine-level work.

A second structural theme: the unit of capacity at hyperscaler scale is no longer "the accelerator" but "the scale-up domain." TPU's OCS-glued superpod, Trainium's NeuronLink UltraServer, and Maia's rack-scale fabric all answer the same question NVL72 answers — how to make a many-chip group behave as one logical accelerator with a single high-bandwidth memory pool. The math on bisection bandwidth and pod scaling is in §3.

## 2. Google TPU: v5e/v5p → Trillium → Ironwood

Google's TPU program is the only hyperscaler ASIC with a continuous public lineage (v1 in 2015 through v7 in 2025) and the only one with an end-to-end software stack — JAX/XLA/Pathways — used by an external frontier lab (Anthropic) at production scale. The lineage is the load-bearing example of the in-house pattern.

### 2.1 The lineage

TPU v5e (2023) and v5p (2023) were the first TPU generation to bifurcate into inference- and training-optimized SKUs. v5e targets per-chip cost for single-host inference and small-pod multi-host serving via Sax: 197 TFLOPS BF16, 16 GB HBM, 256-chip pods. v5p targets training: 459 TFLOPS BF16, 95 GB HBM, 8,960-chip pods. v5p is the chip on which Google trained Gemini 1.0 / 1.5; v5e remains the workhorse for medium-cost inference [see [GCloud-v5e](../papers.md#gcloud-v5e), [GCloud-v5p](../papers.md#gcloud-v5p)].

TPU v6e "Trillium" (preview 2024 → GA 2025) is the inference-and-training generalist. ~4.7× peak compute per chip vs v5e, 32 GB HBM at ~1,600 GB/s, 256-chip pods, 256×256 MXU. Google claims ~1.8× perf/$ vs v5e and ~2× vs v5p; vendor numbers, not independently audited [see [GCloud-Trillium](../papers.md#gcloud-trillium)].

TPU v7 "Ironwood" (announced 2025, commercialized Nov 2025) is the first TPU explicitly framed as a serving chip — the marketing line is "the first Google TPU for the age of inference." Per chip: 4,614 FP8 TFLOPS, 192 GB HBM3e at ~7.37 TB/s, two TensorCores plus four SparseCores, 9.6 Tb/s ICI per chip. Per pod: 9,216 chips, 42.5 EFLOPS FP8, 1.77 PB aggregate HBM, 2× perf/W vs Trillium [see [Google-Iron-Blog](../papers.md#google-iron-blog), [GCloud-TPUv7](../papers.md#gcloud-tpuv7), [STH-Iron](../papers.md#sth-iron)].

The Ironwood-specific design choices that matter for [§10-engine-core](../10-engine-core/) and [§20-distributed-inference](../20-distributed-inference/):

- **FP8/INT8 only — no FP4.** Ironwood's tensor cores stop at FP8/INT8. This is a deliberate divergence from Blackwell and Trainium3 ([§70/01-nvidia-roadmap](01-nvidia-roadmap.md), §4 below), and the TPU team's bet is that for serving workloads where activation outliers and KV-cache numerics dominate end-quality, the FP4 ceiling is not yet worth the integration cost. The decision should be read alongside [§10/04-quantization](../10-engine-core/04-quantization.md): FP4 deployments on Blackwell rely on NVFP4 block scaling at block size 16, and the per-block scale machinery would need a TPU-side analog. The trade-off as the TPU team appears to read it: at 192 GB HBM3e per chip, the capacity headroom for FP8 weights at 70B–500B model class is sufficient, and the marginal byte-traffic reduction from going to FP4 (a further 2× on weight bytes) does not justify the silicon cost of a new datatype path plus its software ecosystem. Google has not publicly committed to a v8 timeline, so whether v8 reverses this is unconfirmed.
- **SparseCores.** Two TensorCores plus four SparseCores per chip. SparseCores accelerate sparse-attention and sparse-activation patterns — the workloads relevant to long-context serving (sparse-attention variants in [§30/03-attention-variants](../30-kv-cache/03-attention-variants.md)), MoE expert dispatch (sparse activation across an expert population), and embedding-lookup-heavy ranking. The architectural disclosures are partial ([Google-Iron-Stack](../papers.md#google-iron-stack)); the relevant detail for engine designers is that sparse paths are exposed through XLA primitives rather than as a user-facing instruction set, so the workload's sparsity is a compiler-time property.
- **Two-tier memory.** 192 GB HBM3e per chip, 1.77 PB pod-aggregate, ~7.37 TB/s per chip. Per-chip HBM bandwidth is in the same range as B200 (~8 TB/s); per-pod HBM aggregate exceeds NVL72's 13.8 TB by ~130×, but the comparison is misleading because the addressing domain is different (TPU pods address through ICI + OCS, NVL72 through NVSwitch).
- **ICI bandwidth.** 9.6 Tb/s per chip across six face-direction ports — comparable in aggregate to NVLink 5's 1.8 TB/s per GPU, though split across more directions. The 3D-torus topology trades flat all-to-all bandwidth (NVL72's strength) for higher-radix nearest-neighbor bandwidth, which favors workloads whose collective patterns are nearest-neighbor (data-parallel all-reduce on 3D-torus-aware reductions, pipeline-parallel across cube columns).

A second design point worth flagging: TPU v7 keeps the systolic-array MXU as its primary matrix engine. The systolic-array architecture (a 256×256 MXU on Ironwood) trades flexibility for arithmetic-density-per-watt — a systolic array runs a fixed dataflow at very high utilization but is poorly suited to control-flow-heavy or irregular kernels. The XLA compiler's job is to lower JAX programs into MXU-friendly shapes; this constrains what can be efficiently expressed and is the reason TPU-targeted attention kernels look different from FlashAttention-3 on Hopper. The Pallas kernel-DSL ([§10/01-attention-kernels](../10-engine-core/01-attention-kernels.md)) is the user-facing escape hatch when XLA's auto-lowering does not produce the desired schedule.

### 2.2 Anchor users and the JAX/XLA/Pathways stack

Ironwood's anchor users are (a) Google internal — Gemini training and serving, search-side inference — and (b) Anthropic, which announced in October 2025 a commitment of up to ~1M TPU chips ([Anthropic-TPU](../papers.md#anthropic-tpu)). Anthropic's commitment is the load-bearing external evidence that the JAX/XLA/Pathways software stack is production-credible at frontier scale; Anthropic also runs Claude inference on AWS Trainium (§4), making it the only frontier lab simultaneously serving production traffic on two non-NVIDIA stacks.

The software stack: model code is JAX, the compiler is XLA, the multi-host runtime is Pathways. Pathways is the layer that handles cross-pod scheduling, fault tolerance, and elastic resharding; it is not open-source and is not portable to non-Google hardware. The structural implication is that a model targeting Ironwood is also targeting Pathways, and the engine-level engineering work in [§80-oss-deep-dives](../80-oss-deep-dives/) (vLLM, SGLang) does not transfer.

## 3. Pod scaling math: 3D torus, OCS, and bisection bandwidth

The TPU v4 paper ([Jouppi-TPUv4](../papers.md#jouppi-tpuv4)) introduced optical circuit switching (OCS) as the inter-cube reconfiguration layer. Ironwood inherits and scales the design.

### 3.1 The 3D torus

The basic TPU pod topology is a 3D torus: each chip is connected to its six neighbors (±x, ±y, ±z) via copper Inter-Chip Interconnect (ICI). A $4\times4\times4$ cube contains 64 chips with diameter 6 hops in the wraparound torus (4/2 + 4/2 + 4/2 = 6, since the torus halves each dimension's diameter). The cube is the unit of locality: collective communication within a cube is bandwidth-uniform and avoids the OCS layer.

For an Ironwood pod: 9,216 chips = 144 cubes of 64 chips each. The cubes are connected by 48 OCS units exposing 13,824 optical ports total ([FM-TPUOCS](../papers.md#fm-tpuocs)). Within a cube, traffic is copper ICI; between cubes, traffic traverses the OCS layer.

### 3.2 Optical circuit switching

OCS is *not* a packet switch. It is a MEMS-based 2D mirror array that physically connects an input fiber to an output fiber — once configured, a path is a continuous photonic circuit until the mirrors reconfigure. The reconfiguration latency is on the order of milliseconds; once set, the path adds essentially no latency beyond fiber length, and the photonic path's throughput is set by the fiber and the transceivers, not by an electrical switching ASIC. The economic argument the TPU v4 paper made and Ironwood inherits: at <5% of system cost and <3% of system power, OCS replaces the equivalent electrical switching layer with a layer whose throughput scales with optics, not silicon switching.

The numbers as Google publishes them for Ironwood ([FM-TPUOCS](../papers.md#fm-tpuocs)): 9,216 chips = 144 cubes × 64 chips/cube; 48 OCS units; 13,824 optical ports total — 96 ports per OCS unit, 96 cubes' worth of inter-cube edges. Each cube exposes its six face-direction inter-cube links via fiber to one or more OCS units; a logical pod topology is realized by mirror configurations connecting fiber-to-fiber pairs across the OCS layer.

Three production properties follow:

- **Topology flexibility.** With OCS, the cube-to-cube wiring is a software decision. The stack can present a logical "twisted torus" of arbitrary aspect ratio (e.g., 16×16×36 for the 9,216-chip pod), or carve out a contiguous slice of $N$ cubes for a specific job, or wrap a slice into a sub-pod that has full bisection bandwidth despite spanning the larger pod. The twisted-torus layouts are the standard TPU-side answer to the problem NVL72 solves with a flat NVSwitch fabric: getting collective-heavy workloads onto a topology where the all-reduce or all-to-all path length is bounded.
- **Failure isolation.** A failed cube's OCS connections can be re-routed without taking down neighbors. The stack treats cube failure as a routine event, and Pathways' fault-tolerance layer handles the resharding without user-code changes — a structural requirement at 9,216-chip scale where MTBF arithmetic forces reliability into the runtime, not the silicon.
- **Scheduling.** XLA and Pathways consume OCS reconfigurations; user code does not. The user sees a contiguous device mesh, and the slice manager decides which cubes back the slice. Multi-tenant capacity (XLA jobs co-resident on a pod) is a slice-level allocation problem, not a chip-level one.

### 3.3 Bisection bandwidth comparison

Bisection bandwidth is the standard metric for collective-heavy workloads: the minimum bandwidth crossing a partition of the network into two equal halves. For a 3D-torus cube of side $n$ at per-link bidirectional bandwidth $B$, bisection BW is $\approx 2 n^2 B$ (two of the three planes' worth of links cross the bisection in each direction). For NVL72 with 72 GPUs at 1.8 TB/s per GPU and full NVSwitch bisection, the aggregate bisection is $\approx 36 \cdot 1.8 = 64.8$ TB/s.

For an Ironwood cube ($n=4$, ICI per chip 9.6 Tb/s = 1.2 TB/s bidirectional, but split across six ports so ~200 GB/s per direction per port): bisection per cube $\approx 2 \cdot 16 \cdot 0.2 = 6.4$ TB/s. The full Ironwood pod's effective bisection through OCS depends on the chosen logical topology; the headline number Google reports is the pod-aggregate ICI bandwidth, not the strict bisection. The structural takeaway is not that one number beats another — the per-chip and per-domain figures are within an order of magnitude of NVL72 — but that *the unit of allocation is the slice, not the chip*, and the slice's bisection is what determines whether tensor-parallel all-reduce (matmul activations) or expert-parallel all-to-all (MoE dispatch) fits inside the iteration budget.

For [§20/03-moe-inference](../20-distributed-inference/03-moe-inference.md): a DeepEP-style MoE all-to-all on 64 cubes (4,096 chips) has a different cost model on TPU than on NVL72 because the inter-cube hops traverse OCS, not NVSwitch. Production MoE serving on TPU is less publicly documented; the Gemini-class MoE models are served on Ironwood but the dispatch internals are not disclosed.

### 3.4 Pod scaling: 4×4×4 cube to 9,216-chip pod

Working out the cube-and-OCS arithmetic explicitly. Each Ironwood cube is $4 \times 4 \times 4 = 64$ chips. The cube has six face-direction surfaces, each $4 \times 4 = 16$ chips wide, and the inter-cube fiber count is one fiber per face-chip per direction (or some small multiple). With 9,216 chips arranged as 144 cubes, the natural logical layout is $6 \times 6 \times 4$ cubes — a $24 \times 24 \times 16$ chip mesh. The torus diameter at this layout is $24/2 + 24/2 + 16/2 = 32$ hops worst case, traversed at copper ICI within cubes and OCS-fiber across cube boundaries.

Slice allocation works as follows: a job requests $C$ contiguous cubes. The OCS reconfigures to make those cubes adjacent in the logical topology, regardless of which physical cubes are healthy. The slice is presented to the user as a $C \cdot 64$-chip mesh with a chosen $a \times b \times c$ shape; the JAX/XLA compiler maps the device mesh onto that shape. Two slices on the same pod can run independently because their OCS paths don't share fibers; this is how XLA serves multiple tenants on one pod.

The bandwidth cost of going off-cube: each chip has six ICI ports. For the four ports facing in-cube neighbors, the connection is copper; for the two facing out-of-cube neighbors (at the cube's boundary), the connection is fiber to an OCS unit. From the chip's perspective, the bandwidth is identical — the OCS layer adds zero throughput cost — but the latency for a packet crossing the OCS layer is bounded by fiber length, typically O(100 ns) per cube-boundary hop in a rack-scale pod and several hundred nanoseconds for cross-pod fiber runs. The latency-per-hop is comparable to NVSwitch's, so for collective workloads the difference is the topology, not the per-hop latency.

### 3.5 Power-per-token: a rough roofline

For a decode-bandwidth-bound workload — the regime that dominates production token traffic — the per-token energy at fixed batch size is approximately the per-token bytes-read times the J/byte at the HBM interface, plus the chip-static power amortized over the iteration's tokens. A rough estimate using public power figures:

- B200 SXM at 1,200 W TDP, 8 TB/s HBM. At decode batch 64 reading 8 GB of weights per layer-pass for a 70B-class model in NVFP4 (~4 GB) and KV cache (~1 GB), each iteration takes ~0.6 ms (5 GB / 8 TB/s). At 1.2 kW draw, that is 0.72 J per iteration, divided by 64 tokens ≈ 11 mJ/token at the chip level (excludes fabric and host).
- Ironwood at vendor-claimed 2× perf/W vs Trillium and ~7.37 TB/s HBM, with 192 GB capacity supporting larger residency: similar order of magnitude per token at like-for-like batch, with the same caveat that the HBM-bandwidth-bound math holds.
- Trainium2 at 2.9 TB/s HBM and ~1 kW per chip (vendor-reported envelope is higher than B200/Ironwood for like-for-like): a per-token energy ratio noticeably worse than Ironwood/B200 on bandwidth-bound decode at small batch, which is the structural reason Trainium3's 4.9 TB/s HBM3e generation is the one that closes the gap.

These figures are arithmetic on vendor-published peaks and should be read as order-of-magnitude. Real production J/token at the data-center boundary includes fabric, host CPU, cooling, and PUE, which typically add a 1.5–2× multiplier. The serving-economics numbers in [§50/02-autoscaling-cost-and-sustainability](../50-cluster-systems/02-autoscaling-cost-and-sustainability.md) work the math at the cluster level.

## 4. AWS Trainium: Trainium2 / UltraServer → Trainium3

AWS's accelerator program has a discontinuous lineage — Inferentia for inference, Trainium for training, with Inferentia2 and Trainium2 effectively merging the use cases at LLM-class scale. The structural difference from TPU is that AWS sells the chips through EC2 as instances, so the unit of operational visibility is the instance type, not the chip.

### 4.1 Trainium2 and the UltraServer

Trainium2 (GA December 2024) is the chip that made AWS a credible LLM-training contender. Per chip: 8 NeuronCores, 96 GiB HBM, 2.9 TB/s HBM bandwidth, 1.3 PFLOPS dense FP8, 5.2 PFLOPS sparse FP8 ([AWS-Trn2-Page](../papers.md#aws-trn2-page), [AWS-Trn2-Blog](../papers.md#aws-trn2-blog)). The Trn2 instance contains 16 chips (one server) at 20.8 PFLOPS FP8 dense.

The NeuronCore is structurally closer to a TPU MXU than a GPU SM: a dataflow-oriented matrix engine with limited control-flow freedom but high arithmetic density per area. This shapes what the Neuron compiler can reasonably target — supported operators run at high efficiency, but expressing FlashAttention-class kernels requires Neuron-specific lowering that the compiler has been progressively adding. The 4× ratio between dense and sparse FP8 (1.3 vs 5.2 PFLOPS) is a 2:4 structured-sparsity acceleration analogous to A100/H100's sparse Tensor Cores; in production LLM serving the dense path dominates because dense weights are the norm.

The headline product is the **Trn2 UltraServer** (GA 2025): 64 chips connected via NeuronLink in a single scale-up domain — 83.2 PFLOPS FP8, 6 TB aggregate HBM, 185 TB/s aggregate HBM bandwidth, 12.8 Tbps EFAv3 networking out of the domain. NeuronLink's role is structurally analogous to NVLink: a high-bandwidth chip-to-chip fabric that lets the 64-chip group behave as one logical accelerator with one logical memory pool. AWS has not published per-link NeuronLink bandwidth or topology details at the level NVIDIA publishes for NVSwitch ([SemiA-Trn2](../papers.md#semia-trn2) is the most detailed external analysis).

A rough scale-up comparison, useful for the parallelism-strategy choices in [§20/01](../20-distributed-inference/01-parallelism-strategies.md):

| Domain | Chips | Aggregate HBM | Aggregate HBM BW | Aggregate FP8 PFLOPS | Out-of-domain link |
|---|---|---|---|---|---|
| HGX H100 (8 GPU) | 8 | 640 GB | 26.8 TB/s | ~16 | InfiniBand NDR (50 GB/s) |
| GB200 NVL72 | 72 | 13.8 TB | ~576 TB/s | ~324 | InfiniBand XDR / Spectrum-X |
| Ironwood cube (4×4×4) | 64 | 12.3 TB | ~472 TB/s | ~295 | OCS to other cubes |
| Trn2 UltraServer | 64 | 6 TB | 185 TB/s | 83.2 | EFAv3 12.8 Tbps |
| Trn3 UltraServer | 64 | 20.7 TB | 706 TB/s | 362 (MXFP8) | EFA next gen |

Trn2's HBM aggregate is structurally the tightest of the four frontier-class scale-up domains; Trn3 closes that gap. The ratio of HBM bandwidth to compute (TB/s per PFLOP) is the indicator that maps to bandwidth-bound decode throughput: Trn2 at 185/83 ≈ 2.2 TB/s per PFLOP, Ironwood cube at ~472/295 ≈ 1.6, NVL72 at ~576/324 ≈ 1.8. Higher is better for decode at small batch; lower is fine for prefill and training where the workload is compute-bound. The Trn2 ratio is a hint that the chip is comparatively decode-friendly at fixed HBM, even though the absolute HBM BW per chip is below Blackwell.

Anchor user: Anthropic. The Anthropic-AWS announcement of late 2025 ([Anthropic-AWS](../papers.md#anthropic-aws)) commits to ~1M+ Trainium chips and an additional 5 GW of capacity through 2027, putting Anthropic on Trainium for production Claude inference and on TPU (§2) for training and partial serving. Anthropic's dual-stack (TPU + Trainium) is the only externally observable production case of a frontier lab serving a single model family on two non-NVIDIA software stacks simultaneously, and the engineering investment that requires is the load-bearing detail in any "did the hyperscaler ASIC bet pay off" assessment.

### 4.2 Trainium3

Trainium3 (announced re:Invent 2025, [Introl-Trn3](../papers.md#introl-trn3)) targets TSMC 3nm and ~2× per-chip performance over Trainium2: 2.52 PFLOPS FP8 per chip, 144 GB HBM3e, 4.9 TB/s HBM bandwidth. The new precision additions are **MXFP8** and **MXFP4** — the OCP Microscaling Formats ([§10/04-quantization](../10-engine-core/04-quantization.md)). The Trn3 UltraServer scales to a claimed 20.7 TB HBM, 706 TB/s aggregate HBM bandwidth, and 362 PFLOPS MXFP8 — vendor-claimed 4.4× the Trn2 UltraServer at like-for-like FP8.

GA timing for Trainium3 is not confirmed as of May 2026; the announcement is for ramp through 2026.

### 4.3 The Neuron SDK and engine support

The software stack is the **Neuron SDK** — a Trainium/Inferentia compiler and runtime that ingests PyTorch and (for some workloads) JAX, lowers through a Neuron-specific IR, and emits NeuronCore binaries. Engine integration is ongoing:

- **vLLM Trainium2 backend.** vLLM has a Neuron-SDK backend that supports Trainium2 inference; performance characteristics differ from CUDA backends and the kernel coverage is narrower [see §80/01-vllm](../80-oss-deep-dives/01-vllm.md).
- **SGLang on Trainium.** Reported in development; production stability is less established than vLLM's path.
- **Custom kernels.** The [NeuronMM](../papers.md#neuronmm) paper (Oct 2025) is the first peer-reviewed work on Trainium2 matmul performance and is the canonical reference for what the chip can sustain on an LLM matmul.

The portability story: a model that runs on vLLM on H100 may run on vLLM on Trn2 with engine-side configuration changes, but kernel-level optimizations (FA-3, FA-4, FlashInfer paged attention) are CUDA-specific and have Neuron analogs that are still maturing.

### 4.4 A worked Trainium2 decode example

To make the engine-porting issue concrete: consider a 70B-class FP8 dense model on a Trn2 instance (16 chips, 96 GiB HBM each, 1.536 TiB aggregate, 46.4 TB/s aggregate bandwidth). Weights at FP8 occupy ~70 GB, fitting comfortably with KV-cache headroom. At decode time, the relevant arithmetic is bandwidth-bound: each decode iteration reads the full 70 GB weight set once, plus per-token KV reads scaling with batch size and context length. At 46.4 TB/s aggregate read bandwidth and 70 GB per iteration, the compute-only iteration time is ~1.5 ms; with KV reads at decode batch 64 and 8K context, the HBM read budget approximately doubles to ~140 GB per iteration, giving ~3 ms per iteration and ~21K tokens/s aggregate throughput at the chip-bandwidth bound. Realistic delivery is some fraction of that — typically 50–70% under production scheduling overhead, attention-kernel inefficiency, and KV-paging waste.

The same workload on H100 SXM at 8 GPUs (640 GB HBM, 26.8 TB/s aggregate): ~140 GB / 26.8 TB/s ≈ 5.2 ms per iteration ideal, ~12.5K tokens/s aggregate. The Trn2 instance is structurally faster on this workload, in line with the HBM aggregate. The catch is that vLLM-on-Trn2 currently delivers a smaller fraction of peak than vLLM-on-H100 due to the kernel-portability gap; production benchmarks (when available, [NeuronMM](../papers.md#neuronmm)) typically report 60–80% of theoretical decode throughput rather than the ~85% vLLM-on-H100 achieves. Closing that gap is what the Neuron-SDK kernel team has been shipping over 2025–2026.

## 5. Meta MTIA: from ranking to gen-AI

MTIA's lineage is the most workload-specific of the four. MTIA v1 (2023) targeted Meta's ranking and recommendation models — not LLMs ([Meta-MTIA-v1](../papers.md#meta-mtia-v1)). The chip carried 64 GB DDR memory and 128 PEs at 800 MHz; the workload was DLRM-class embeddings and shallow MLPs.

MTIA v2 (2024) doubled SRAM, added 128 GB capacity, and broadened the workload to inference and light training ([Meta-MTIA-v2](../papers.md#meta-mtia-v2), with architectural details in the [Meta-ISCA25](../papers.md#meta-isca25) paper). The v2 design choices — 256 MB shared on-chip SRAM, large activation residency on chip — are recognizable as a ranking-first, attention-second optimization.

The 2025 roadmap, "Four MTIA Chips in Two Years" ([Meta-MTIA-2025](../papers.md#meta-mtia-2025), [STH-MTIA](../papers.md#sth-mtia)), is the explicit pivot to generative AI:

| Generation | Status (May 2026) | Focus | Notable parameters |
|---|---|---|---|
| MTIA 300 | In production | Ranking/recommendation training | Successor to v2; HBM details partial |
| MTIA 400 | Currently shipping | Generative AI capable | 5× compute, 1.5× HBM BW vs 300 |
| MTIA 450 | Mass deployment early 2027 | Gen-AI scaling | HBM BW 18.4 TB/s/accel (vs 9.2 TB/s on 400) |
| MTIA 500 | Mass deployment late 2027 | Frontier-scale | 1.5× HBM BW, 1.8× HBM capacity, 1.43× MX4 FLOPS over 450; 2×2 chiplet |

The HBM-bandwidth progression — 9.2 → 18.4 TB/s/accel from 400 to 450, a 2× generation — is the structural signal that Meta is taking decode bandwidth seriously: at fixed model size, decode TPS at small batch is bandwidth-bound ([§00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md)), and the 2× HBM jump is read as a Llama-4-class serving target.

The MTIA design lineage is also the clearest example in the four programs of *narrow workload first, broaden later*. MTIA v1 was a ranking accelerator — DLRM models with embedding tables in the tens to hundreds of GB and shallow MLPs running on the activations. The architectural distinguishing features — DDR rather than HBM, SRAM-heavy on-chip memory, large embedding-lookup throughput — are not LLM-shaped. v2 added HBM and broadened to inference; the 2025 roadmap (300/400/450/500) is the bet that the same architectural team can pivot a ranking-optimized lineage into a gen-AI-capable one without a clean-sheet redesign. Whether the bet works will be visible in 2027 with the 450/500 generations; as of May 2026, the gen-AI capable 400 is shipping but external benchmarks against frontier-LLM workloads are scarce.

MTIA is co-developed with Broadcom and is part of Meta's >$600B planned US infrastructure capex through 2030. The software stack is internal; there is no externally documented engine analogous to vLLM/Neuron-SDK. Meta has not committed publicly to a portability story for MTIA outside Meta workloads.

The MTIA roadmap also illustrates the chiplet inflection: MTIA 500's 2×2 chiplet construction is the same architectural pattern Blackwell reaches with its dual-die reticle-stitched B200, and that AMD reaches with MI300X's chiplet IO die. Reticle-limited monolithic dice are the binding constraint for all four hyperscaler programs at the leading edge, and MTIA 500 commits to a chiplet-based response in the same generation as Trainium3 (TSMC 3nm) and Maia 200 (TSMC 3nm, 140B transistors). The trend that the chapter on [§70/01-nvidia-roadmap](01-nvidia-roadmap.md) traces for NVIDIA — chiplets, advanced packaging, HBM3e/HBM4 — is being recapitulated in the hyperscaler track on a 1–2 year lag.

A note on the MTIA-vs-NVIDIA-vs-merchant split inside Meta's own infrastructure: Meta has not committed to displacing NVIDIA across the board. Meta's frontier training (Llama-4-class) and a substantial fraction of inference continues to run on NVIDIA H100/H200/B200 hardware, with MTIA covering ranking and the early gen-AI workloads where its architectural fit and cost structure are favorable. The published Meta capex through 2030 implies a heterogeneous fleet — MTIA where MTIA wins on per-token cost, NVIDIA where the engine ecosystem matters or where MTIA is not yet shipping at the required scale.

## 6. Microsoft Maia: Maia 100 → Maia 200

Maia is the youngest of the four programs. Maia 100 (introduced 2023, Azure preview 2024, [MS-Maia100](../papers.md#ms-maia100)): TSMC N5, 820 mm² reticle die, 64 GB HBM2E, 1.8 TB/s, 500 W. Maia 100 was deployed for OpenAI internal Azure workloads and is the first hyperscaler ASIC explicitly co-designed with the largest external frontier-lab tenant.

Maia 200 (announced January 2026, deployed Azure US Central, [MS-Maia200-Blog](../papers.md#ms-maia200-blog), [MS-Maia200-Deep](../papers.md#ms-maia200-deep)) is the inference-targeted successor:

- TSMC 3nm, 140B transistors.
- 216 GB HBM3e at 7 TB/s — capacity comparable to Ironwood (192 GB), bandwidth slightly below.
- 272 MB on-chip SRAM (larger than B200's TMEM by an order of magnitude, structurally aimed at attention staging).
- >10 PFLOPS FP4, >5 PFLOPS FP8, native FP8 and FP4 tensor cores.
- 750 W SoC TDP.
- 30% perf/$ improvement over Maia 100 (Microsoft-reported).

Microsoft's headline benchmark claims — "3× FP4 vs Trainium3 and FP8 above TPUv7" — are vendor-reported and have not been independently verified in MLPerf or peer-reviewed work as of May 2026. Capacity is being expanded from US Central to US West 3 (Phoenix); the deployment scale is not publicly disclosed at the chip count.

The structural significance of Maia 200: it is the only hyperscaler ASIC with native FP4 tensor cores in production (Trainium3 has them announced; TPU v7 deliberately omits them). For OpenAI workloads on Azure, Maia 200 is the bet that FP4 inference economics will track or exceed NVFP4 on Blackwell; the workload-specific co-design with OpenAI is the load-bearing detail. The 272 MB on-chip SRAM is itself a workload-specific signal: at ~4× B200's TMEM aggregate, it appears sized to keep larger attention tiles and a meaningful fraction of KV cache resident on-chip during decode, an optimization that pays off proportionally to the model's sequence length and matters most for long-context reasoning models.

The Maia 200 power envelope (750 W) is in the same range as B200 SXM (1,000–1,200 W) and Ironwood (Google does not publish per-chip TDP cleanly). For datacenter integration, Microsoft's published deployment uses liquid cooling and a custom rack design ([MS-Maia200-Deep](../papers.md#ms-maia200-deep)); like NVL72 and Ironwood pods, the unit of physical deployment is the rack, not the server.

The software stack is proprietary. Microsoft has not announced a vLLM-on-Maia path; the inference engine is internal to Azure and the OpenAI deployment. The integration with OpenAI's serving stack is the analog of Google-Anthropic on TPU: a single-tenant frontier-lab workload as the primary customer, with general-availability serving on Azure as a future expansion. As of May 2026 there is no public Maia 200 SKU available to general Azure customers in the way Trainium2 is available as Trn2 instances on EC2.

## 7. Software ecosystem comparison

The four hyperscaler stacks differ on more than instruction sets — they differ on which engine code one writes:

| Stack | Frontend | Compiler | Runtime | Engine integration |
|---|---|---|---|---|
| TPU (Google) | JAX (primary), PyTorch/XLA (secondary) | XLA | Pathways | MaxText, Sax; vLLM-on-TPU exists but is less mature |
| Trainium (AWS) | PyTorch (primary), JAX (Neuron-JAX) | Neuron compiler | Neuron runtime | vLLM Trainium2 backend; SGLang in development |
| MTIA (Meta) | PyTorch (Meta-internal variants) | Internal | Internal | Internal Meta engines; no public engine support |
| Maia (Microsoft) | PyTorch / ONNX | Internal MAIA SDK | Internal | Internal Azure engine; no public vLLM/SGLang path |
| CUDA reference | PyTorch / JAX | nvcc + Triton + CuTe-DSL | CUDA + NCCL | vLLM, SGLang, TRT-LLM, MLC ([§80](../80-oss-deep-dives/)) |

The portability gradient: TPU has the deepest non-NVIDIA OSS ecosystem (JAX, MaxText, public Pathways docs); Trainium has the broadest engine integration (vLLM); Meta and Microsoft have effectively no externally usable engine path. This matters for the chapters in [§80-oss-deep-dives](../80-oss-deep-dives/): a workload that targets vLLM/SGLang/TRT-LLM is, in May 2026, a CUDA-or-Trainium workload at production scale, with TPU as a JAX-side alternative.

A second observation that recurs in [§40-multi-tenant](../40-multi-tenant/) and [§50-cluster-systems](../50-cluster-systems/): the operational tooling (Kubernetes operators, autoscalers, observability hooks) for hyperscaler ASICs is hyperscaler-specific. A model deployed on TPU runs under GKE with Pathways; on Trainium, under EKS with Neuron device plugins; on Maia, only under Azure-internal orchestration. Cross-cloud portability is a function of the engine, not the silicon.

A worked example from a serving-engineer's perspective: a Llama-3.1-70B FP8 deployment on H100 with vLLM uses FlashAttention-3, FlashInfer paged-KV BSR layouts, and NVIDIA-side LoRA adapters via Punica/Triton. Porting that stack to Trainium2 requires (a) a Neuron-SDK kernel substitute for FA-3, (b) a Neuron-side KV pager (vLLM's Trainium backend has one but the BSR coverage is narrower), and (c) re-validation of LoRA correctness against Neuron numerics. Porting to TPU requires the same three steps in JAX/XLA — Pallas kernels for attention, the JetStream serving runtime's KV manager, and re-validation in JAX. Porting to Maia 200 is, as of May 2026, not generally possible outside Azure-internal teams. The portability cost is not a fixed multiplier; it is a per-feature reimplementation cost, and it scales with the model's reliance on engine-specific optimizations like prefix caching ([§10/07-prompt-prefix-caching](../10-engine-core/07-prompt-prefix-caching.md)) and speculative decoding ([§10/05-speculative-decoding](../10-engine-core/05-speculative-decoding.md)).

## 7a. Economics: when does in-house silicon make sense?

The build-vs-buy threshold for hyperscaler silicon, as a back-of-envelope:

- **Design and tape-out NRE.** A leading-edge ASIC at TSMC 3nm reportedly costs $500M–$1B in NRE through first silicon, including IP licensing, mask sets, and pre-silicon validation. Hyperscalers offset some of this by reusing IP across generations (Broadcom's collaboration with Google and Meta, Marvell's with AWS).
- **Per-chip cost at volume.** At 3nm with high-bandwidth HBM3e and advanced packaging, per-die cost is on the order of $5–15K depending on yield and binning; assembled per-accelerator cost (with HBM, packaging, board, power delivery) is several times that. This is comparable to NVIDIA's AIB cost basis but lower than NVIDIA's wholesale pricing, which carries a substantial margin.
- **Software-stack cost.** A team capable of maintaining a non-CUDA stack at production-LLM-engine quality (compiler, kernels, runtime, engine integration, multi-tenant orchestration, observability) is in the low-hundreds of FTEs minimum. For Google with JAX/XLA already in place, and for AWS with Annapurna's pre-existing Neuron program, the marginal cost is lower; for Microsoft and Meta, the team-build is part of the program cost.
- **Workload concentration.** The bet pays off when the captive workload is large enough that per-token cost reductions amortize the program. A useful threshold: at 1M+ accelerators in service for a single first-party workload (Google's Gemini, AWS's Claude-via-Anthropic-commitment, Meta's ranking, Microsoft's OpenAI), the math works. Below that, the merchant-GPU economy wins.
- **Time-to-deployment.** A clean-sheet ASIC is a 3–5 year program from architecture freeze to volume production. Hyperscalers compress this with IP reuse and partner co-design (Broadcom for Google and Meta, Marvell and Alchip for various AWS components, AMD/TSMC packaging IP), but the lead-time still exceeds NVIDIA's 1.5–2 year roadmap cadence. The result is that hyperscaler ASICs are typically one architectural generation behind NVIDIA on FLOP/s and HBM bandwidth at announce time, and the gap is narrowed only by NVIDIA's pricing power on its current generation.

The pattern visible across the four programs: each hyperscaler ASIC is effectively underwritten by a single or near-single anchor tenant whose workload is large enough to amortize the program. TPU is underwritten by Gemini + Anthropic; Trainium by Anthropic; MTIA by Meta-internal; Maia by OpenAI-on-Azure. A fifth hyperscaler with a less concentrated workload would have a harder economic case, which is part of why Oracle's OCI and Google Cloud's smaller external customers buy NVIDIA-class hardware for the long tail rather than building their own silicon.

A second economic axis is power. Frontier serving in 2026 is being capacity-planned at the gigawatt level (xAI Colossus 2 at 2 GW, Stargate at multi-GW, Anthropic's announced 5 GW of AWS capacity through 2027). At GW-scale buildouts, every percentage-point improvement in J/token translates into rack-scale capex avoidance and grid-interconnect headroom. The 2× perf/W claim Ironwood makes over Trillium, the 30% perf/$ Maia 200 makes over Maia 100, and the structural HBM-BW progression on MTIA (9.2 → 18.4 TB/s/accel from 400 to 450) are all framed as serving-cost claims more than peak-FLOP claims. The chapter's roofline analysis (§3.4, §8 below) supports the framing: bandwidth-bound decode is the dominant workload, and bandwidth scaling — not FLOP scaling — is the differentiator.

## 8. Production deployment, by tier

The May 2026 picture, as best public sources support:

| Tier | Silicon | Anchor workloads | Public scale signal |
|---|---|---|---|
| Frontier serving | TPU v7 Ironwood | Gemini (Google), Claude (Anthropic) | Anthropic ~1M chip commitment ([Anthropic-TPU](../papers.md#anthropic-tpu)) |
| Frontier serving | NVL72 (B200/B300) | OpenAI, xAI, Meta-Hyperion, Anthropic-partial | Stargate, Colossus 2 buildouts ([§70/01](01-nvidia-roadmap.md)) |
| Frontier serving | Maia 200 | OpenAI on Azure | Azure US Central; capacity not disclosed |
| Frontier serving | Trainium2 / Trn2 UltraServer | Anthropic Claude | Anthropic-AWS 5 GW commitment through 2027 ([Anthropic-AWS](../papers.md#anthropic-aws)) |
| Hyperscaler-internal | MTIA 300/400 | Meta ranking + early gen-AI | Internal; chip count not public |
| Mid-tier inference | TPU v6e Trillium, v5e | Google Cloud customers, Anthropic | GA 2025 |
| Cost-optimized | Inferentia2 | AWS managed inference | Production since 2022 |

For the workloads of [§60-adjacent-workloads](../60-adjacent-workloads/) (RL post-training, multimodal serving, safety models): the hyperscaler-ASIC track is, as of May 2026, primarily a *frontier-LLM-serving* track. RLHF rollout serving and reasoning-trace generation lean on the same anchor workloads listed above; multimodal serving and small-model embedding workloads remain mostly on GPUs because the engine support (vLLM-multimodal, structured outputs, small-model autoscaling) is CUDA-mature.

The single most informative signal of which silicon is winning a given workload is *which engine ships first-class support*. vLLM ships TPU support (via Pallas) and Trainium support (via Neuron); both are second-tier compared to CUDA in feature coverage. SGLang ships TPU support and is adding Trainium. TRT-LLM is NVIDIA-only by design. MAX (Modular's engine) targets cross-vendor portability but production deployments are minimal. Until or unless one of vLLM/SGLang/TRT-LLM achieves feature parity on a non-CUDA backend, the structural answer is that the merchant-GPU economy continues to dominate for workloads outside the four hyperscaler captive ones.

A roofline-comparison footnote, applying the analysis of [§00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md). At decode-bandwidth-bound regimes, the relevant figure is HBM bandwidth × FP8 weight density. Per-chip: B200 ~8 TB/s; Ironwood ~7.37 TB/s; Maia 200 ~7 TB/s; Trainium2 ~2.9 TB/s; Trainium3 ~4.9 TB/s; MTIA 450 ~18.4 TB/s (announced). Per-pod aggregate diverges much more: NVL72 130 TB/s NVLink fabric; Ironwood 9,216-chip pod aggregate is in the multi-PB/s range. The comparison hedges: per-chip is comparable across the frontier-class chips, per-pod is governed by the scale-up domain's fabric, and effective serving throughput is dominated by software-stack maturity and parallelism strategy in [§20-distributed-inference](../20-distributed-inference/) more than by raw peak.

The roofline ridge points, computed as peak FP8 / HBM BW following [§00/02](../00-foundations/02-transformer-arithmetic-roofline.md):

- Ironwood: 4,614 / 7.37 ≈ 626 FLOP/byte at FP8.
- B200: ~4,500 / 8.0 ≈ 562 FLOP/byte at FP8 (~281 at FP16).
- Trainium2: 1,300 / 2.9 ≈ 448 FLOP/byte.
- Trainium3: 2,520 / 4.9 ≈ 514 FLOP/byte at FP8.
- Maia 200: 5,000 / 7.0 ≈ 714 FLOP/byte at FP8 (FP4 ridge ≈ 1,428 if compute is ~10 PFLOPS).

The ridge points cluster between 450 and 700 FLOP/byte for the frontier-class FP8 chips. The takeaway is that bandwidth-bound decode at small batch is similar across all five — the ridge is several hundred FLOP/byte and decode arithmetic intensity is a few tens — and the differentiator at production scale is software, scale-up-domain bisection, and SKU-level economics, not peak FLOP/s.

## Current production state

As of May 2026, the production picture is that NVIDIA dominates the merchant-GPU population and a small set of hyperscaler ASICs serve specific frontier workloads at meaningful but bounded scale. Google TPU v7 Ironwood is the only non-NVIDIA accelerator that simultaneously (a) serves frontier-LLM training and serving for the company that built it (Gemini), (b) serves an external frontier lab's production inference (Anthropic), and (c) has an end-to-end open software stack (JAX/XLA, with Pathways closed but documented). AWS Trainium2 and the Trn2 UltraServer have moved from "promising training chip" in 2024 to "Anthropic Claude serving substrate" in 2026, and Trainium3's GA in 2026 is what determines whether AWS holds parity with Blackwell and Ironwood through 2027. Microsoft Maia 200 is the fresh entrant with the strongest FP4 claims, but external benchmarks are scarce and the deployment is Azure-internal. Meta MTIA is the workload-specific accelerator most clearly *not* aimed at external customers; the four-chip 2025 roadmap shifts MTIA from ranking to gen-AI with mass deployment of the gen-AI-class 450/500 starting in 2027.

The economic signal underlying all four programs is that hyperscaler-scale serving has crossed the threshold where workload-optimized silicon pays back its NRE inside a single product generation. The signal is most visible in the Anthropic dual-stack: a frontier lab is willing to underwrite serving on two non-CUDA stacks simultaneously, which is only rational if the per-token cost differential against NVIDIA is meaningful and durable. The second signal is the sheer capacity announcements: Anthropic's 1M+ TPU chips and 5 GW of AWS capacity, OpenAI's Stargate and Maia deployment, Meta's 600B+ infrastructure capex through 2030, and Google's continuous TPU buildout add up to a multi-hundred-billion-dollar non-NVIDIA accelerator market through 2027. NVIDIA still owns the median production token, but the marginal frontier-serving token is increasingly served on hyperscaler silicon.

The structural takeaway for engine and parallelism design: the hyperscaler-ASIC track is real, the per-chip numbers are within striking distance of B200, and the scale-up domain analogs of NVL72 (Ironwood pods, Trn2 UltraServers) genuinely behave as one logical accelerator at trillion-parameter scale. But the software-portability story is a wall, not a ramp. A model targeting vLLM/SGLang/TRT-LLM on CUDA does not move to Ironwood, MTIA, or Maia without engine-side porting work; only Trainium has a meaningful vLLM integration. The hyperscaler bet is that the captive workload scale justifies the captive software stack, and the Anthropic dual-stack (TPU + Trainium) is the load-bearing external evidence as of May 2026 that the bet is paying off — for at least one frontier lab.

For a reader returning to this chapter from [§80-oss-deep-dives](../80-oss-deep-dives/) or [§90-synthesis](../90-synthesis/01-production-stack-recipes.md), the practical question to carry forward is which engine-and-silicon pair the workload at hand should target. The answer is workload-dependent: frontier-LLM serving at NVL72-or-larger scale will, through 2027, run on a mix of Blackwell, Ironwood, and Trainium3 with workload-anchor selection (Anthropic on TPU+Trainium, OpenAI on Maia+Blackwell, Google on Ironwood, xAI/Meta-Hyperion on Blackwell). Mid-tier inference and small-model serving will continue to run on H100/H200/B200 because the engine ecosystem is mature there. Embedding, multimodal, RL-rollout, and ranking workloads will run on whichever silicon has the relevant kernel coverage — increasingly mixed but still CUDA-default for the long tail. The production-stack recipes chapter ([§90/01](../90-synthesis/01-production-stack-recipes.md)) works through three concrete deployments using these axes.

The forward question, addressed in [§90-synthesis/01-production-stack-recipes](../90-synthesis/01-production-stack-recipes.md) and revisited in [§70/05-networking-fabric](05-networking-fabric.md), is whether Maia 200 and Trainium3 close the FP4 gap with Blackwell on independent benchmarks, whether MTIA 450/500 ship on schedule for Meta gen-AI, and whether the JAX/XLA ecosystem expands beyond Anthropic to a second frontier lab. The base-rate forecast in May 2026 is that the four programs continue to differentiate by anchor workload rather than converge on a single architecture, and that engine porting (vLLM-on-Trainium, vLLM-on-TPU) becomes the practical bottleneck rather than peak-FLOP parity.

A second forward question, less often addressed publicly: whether the *startup ASIC* track of [§70/04-asics-startup](04-asics-startup.md) — Groq, Cerebras, SambaNova, Tenstorrent, Etched — competes with or complements the hyperscaler-ASIC track. The hyperscaler chips share the structural advantage of a captive anchor workload; the startup chips compete on per-token economics for non-captive customers and on architectural specialization for narrower workloads. The two tracks face the same software-portability wall and arrive at it from opposite directions: hyperscalers from "we own the workload, we'll build the stack to match"; startups from "the workload is the model architecture, we'll build the chip to match." The shared lesson, as of May 2026, is that the engine ecosystem (vLLM, SGLang, TRT-LLM) is the gravitational well around which both tracks orient.
