# AMD and Non-NVIDIA GPUs

**After reading this chapter, the reader will be able to:**

- Trace the AMD GPU lineage relevant to LLM inference — MI300X (CDNA 3) through MI325X, MI350X / MI355X (CDNA 4 with native FP4/FP6), MI400-series and the Helios rack — and place each generation against its NVIDIA contemporary on the axes that matter for serving: HBM capacity, HBM bandwidth, supported low-precision formats, and scale-up fabric.
- Read a ROCm + vLLM gap analysis and identify the three structural deficits that prevent a working CUDA kernel from porting one-to-one to MI355X — FlashAttention-3 / FA-4 parity, custom MoE kernels (DeepEP and expert all-to-all), and full graph capture for agentic decode loops — and explain the mitigation path through AITER, IREE, and ROCm-Triton.
- Reason about when an AMD or Intel Gaudi deployment is the right production choice (HBM-bound large-model serving where MI355X's 288 GB and 8 TB/s win on per-GPU capacity, or Gaudi-anchored enterprise estates) versus where the CUDA software moat still dominates (long-tail kernels, frontier MoE, agentic graphs).

The non-NVIDIA GPU story for LLM inference in 2026 is short to summarize and long to qualify. AMD has caught up on hardware: MI300X reached HBM parity with H100 in 2023, MI325X exceeded H200 in 2024, and MI355X in 2025 ships ahead of B200 on per-GPU memory capacity (288 GB vs. 192 GB) and at parity on bandwidth (8 TB/s) and on the FP4 compute path. Intel Gaudi 3 ships 96 GB HBM2e at 3.7 TB/s and is positioned as a TCO play rather than a peak-throughput challenger. The qualifier is software: the CUDA / cuDNN / NCCL / Triton / TensorRT-LLM stack has a roughly five-year head start, and the gap is narrowing on the most-trafficked operators (dense GEMM, GQA attention, FP8 GEMM) faster than on the long tail (FA-3 with warp specialization, DeepEP all-to-all, full CUDA-graph capture for tool-calling decode). This chapter treats hardware and software as separate fronts and tracks where each is closing.

The full NVIDIA hardware deep-dive — Hopper, Blackwell, GB200 NVL72, Rubin — is in [§70/01-nvidia-roadmap](01-nvidia-roadmap.md). Numerics from that chapter (H100, H200, B200, B300) are reused here for direct comparison. The compiler-and-IR layer that both vendors share is sketched in the sidebar at the end of this chapter and treated in full in [§10/08-cuda-graphs-compilation](../10-engine-core/08-cuda-graphs-compilation.md).

## 1. The AMD MI300X lineage

The Instinct MI300 line is AMD's response to A100/H100 for LLM training and inference. Three generations are relevant to current serving and one is on the near roadmap.

### 1.1 MI300X (2023, CDNA 3)

MI300X is a chiplet-based design: eight XCD compute chiplets (each carrying 38 active CUs out of 40 physical) on four base IODs, integrated via 3D hybrid bonding and a 256 MB Infinity Cache. It ships 192 GB of HBM3 across eight stacks at 5.3 TB/s aggregate bandwidth. Peak dense throughput is roughly 1.3 PFLOPS at FP16/BF16 and ~2.6 PFLOPS at FP8 with sparsity disabled. MI300X had no FP4 path.

The headline number in 2023 was the 192 GB capacity. H100 SXM5 shipped 80 GB; on per-GPU capacity, MI300X was 2.4× ahead, which translated into the ability to fit a 70B-class model at FP16 on a single GPU (rather than across two H100s with tensor parallelism) and a 405B-class model on a single 8-way node. Bandwidth (5.3 TB/s vs H100's 3.35 TB/s) was a roughly 1.6× lead. On compute, MI300X and H100 were near-parity at FP16 and FP8; on the software stack to extract that compute, H100 was decisively ahead.

### 1.2 MI325X (2024, CDNA 3 refresh)

MI325X is a memory upgrade of MI300X: same CDNA 3 compute, 256 GB of HBM3e at ~6 TB/s. It targeted the H200 launch window (H200 ships 141 GB HBM3e at 4.8 TB/s) and again won on raw capacity. MI325X did not change the format support story — still no FP4, still on the CDNA 3 matrix-core ISA — and is mostly transitional. Production deployments treat MI325X and MI300X as a single tuning target; the ROCm software stack does not branch significantly between them.

### 1.3 MI350X / MI355X (2025, CDNA 4)

CDNA 4 is the inflection point. The MI350-series moves to TSMC N3P, refactors the matrix-core data path, and adds **hardware support for the MX low-precision formats** — MXFP4 (E2M1 with a per-block 8-bit scale), MXFP6 (E3M2), MXFP8 — over the same compute lanes that handle FP8. The reported peak numbers:

| Format | MI355X peak (dense) | Multiplier vs. MI355X FP16 |
|---|---|---|
| FP16 / BF16 | ~2.5 PFLOPS | 1× |
| FP8 | ~5 PFLOPS | 2× |
| FP6 (MXFP6) | ~10 PFLOPS | 4× |
| FP4 (MXFP4) | ~10 PFLOPS | 4× |

The MX-family numerics (block size 32, 8-bit shared scale) are the same as the OCP MX standard NVIDIA adopted for Blackwell, so the *format* is portable across the two vendors; the *kernel* is not, because the matrix-core ISAs differ.

Memory: 288 GB HBM3e at 8 TB/s, eight stacks of 12-Hi. MI355X is the higher-power liquid-cooled SKU; MI350X is the air-cooled variant. On capacity, MI355X is 1.5× ahead of B200 (192 GB HBM3e at 8 TB/s); on bandwidth, the two are at parity. On FP4 peak, MI355X (~10 PFLOPS) sits below B200 (~20 PFLOPS NVFP4 dense, vendor-quoted) but in the same order of magnitude. The hedge: vendor PFLOP numbers are notoriously aspirational across both companies, with sustained kernel throughput in production typically 50–70% of marketing peak.

The deployment implication is that a single MI355X fits Llama-3.3 70B at FP16 with KV-cache headroom for substantial concurrency, and an 8× MI355X node fits Llama-3.1 405B at FP8 without tensor parallelism crossing nodes. For long-context serving where KV cache dominates, the 1.5× capacity edge over B200 is the structural reason a fraction of frontier-lab inference workloads landed on MI355X in 2025–2026.

### 1.4 MI400 / Helios (H2 2026)

The MI400 series, announced at CES 2026 and shipping in the second half of 2026, splits into three SKUs: MI430X (HPC + AI mixed precision, FP64 emphasis), MI440X (8-way OAM box, AI-focused), and MI455X (rack-scale, the part that goes into Helios). The process node moves to TSMC N2; the architecture is described as CDNA-Next, with claimed but unconfirmed FP4 throughput in the 30–40 PFLOPS range per GPU.

The structural shift is **Helios**, AMD's answer to GB200 NVL72. A Helios rack integrates 72× MI455X with Zen-6 EPYC "Venice" head nodes, advertising 31 TB of aggregate HBM4 and 1.4 PB/s of aggregate bandwidth. The scale-up fabric is **UALink 1.0** — the open Ultra Accelerator Link standard AMD has been driving since 2024 — physically tunneled over Ethernet via Broadcom's Tomahawk 6 switch silicon. UALink targets NVLink-like semantics (load/store, atomics, GPU-to-GPU memory access without a CPU round-trip) on commodity Ethernet PHYs. Whether the realized latency and bandwidth match NVLink 5 / NVSwitch will be settled empirically in 2026; AMD's claims are competitive on paper and unconfirmed in the field as of May 2026.

A speculative MI500 / CDNA 6 generation is projected for 2027 with HBM4E and a marketing claim of "1000× MI300X cumulative performance," which should be read as forward-looking and not yet load-bearing.

## 2. CDNA 4 and the MX format path

The most consequential CDNA 4 microarchitectural change for inference is that FP4, FP6, and FP8 share **one matrix-core data path**. The same instruction issues to the same lanes; format selection is a mode bit on the operand. This matters because it means:

- A model quantized to MXFP4 weights with MXFP8 activations runs on the same kernel scaffolding as one in pure FP8. The scheduling, tiling, and pipeline structure are unchanged; only the operand width and the dequant path change.
- Per-block scales (the "X" in MX) are loaded alongside operand tiles in the matrix-core pipeline, so MXFP4 and NVFP4 (NVIDIA's 16-element FP4 with FP8 scale, see [§10/04-quantization](../10-engine-core/04-quantization.md)) have similar arithmetic-intensity profiles.
- FP6 is a real production format on AMD in a way it has never been on NVIDIA. NVIDIA Blackwell supports FP6 nominally but ships kernels primarily for FP4 and FP8; AMD's MXFP6 path is instrumented in AITER and benchmarked alongside MXFP4. For models where MXFP4 loses too much accuracy (some long-context, some reasoning workloads), MXFP6 on MI355X is a legitimate middle point; the equivalent middle on Blackwell is FP8.

Activation outliers and rotation preprocessing (QuaRot, SpinQuant — see [§10/04-quantization](../10-engine-core/04-quantization.md)) are required for AMD MXFP4 just as for NVIDIA NVFP4. The calibration recipe is portable; the kernel is not.

## 3. Hardware comparison: AMD vs. NVIDIA on the inference axes

The four axes that determine LLM serving outcomes — HBM capacity, HBM bandwidth, low-precision compute peak, and scale-up fabric — line up as follows for the parts in current production or about to ship:

| GPU | Memory | BW | Low-prec formats | Scale-up | Year |
|---|---|---|---|---|---|
| H100 SXM5 | 80 GB HBM3 | 3.35 TB/s | FP8 | NVLink 4 (900 GB/s) | 2022 |
| MI300X | 192 GB HBM3 | 5.3 TB/s | FP8 | Infinity Fabric | 2023 |
| H200 SXM5 | 141 GB HBM3e | 4.8 TB/s | FP8 | NVLink 4 (900 GB/s) | 2024 |
| MI325X | 256 GB HBM3e | 6 TB/s | FP8 | Infinity Fabric | 2024 |
| B200 SXM | 192 GB HBM3e | 8 TB/s | NVFP4 / FP8 | NVLink 5 (1.8 TB/s) | 2025 |
| MI350X / MI355X | 288 GB HBM3e | 8 TB/s | MXFP4 / FP6 / FP8 | Infinity Fabric | 2025 |
| GB200 NVL72 | 13.5 TB aggregate | — | NVFP4 / FP8 | NVLink 5 / NVSwitch | 2025 |
| Helios (72× MI455X) | ~31 TB aggregate | ~1.4 PB/s | (CDNA-Next) | UALink 1.0 over Ethernet | 2026 H2 |

Two readings of the table. First, on per-GPU memory and bandwidth, AMD has been ahead on capacity since MI300X and reached parity on bandwidth at MI355X. Second, on rack-scale fabric, NVIDIA shipped first (GB200 NVL72 in 2025) and AMD's Helios arrives roughly a year later with an open-standard fabric (UALink) of unproven equivalence to NVLink 5.

The **roofline implication**, reusing the ridge derivation from [§00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md): for a bandwidth-bound decode workload at FP8, MI355X's 8 TB/s is at parity with B200 and 1.7× ahead of H200. For a compute-bound prefill at FP4, B200's quoted peak is ~2× MI355X. The crossover batch size — the operating point at which the workload moves from the bandwidth-bound region to the compute-bound region — therefore sits at a smaller batch on MI355X than on B200 for the same model, meaning MI355X retains its bandwidth advantage to higher batch sizes. Whether the realized kernel achieves these peaks is a software question.

*Note on B200 FP4 figures.* B200 NVFP4 peak depends on sparsity: ~9 PFLOPS dense, ~18–20 PFLOPS with structured sparsity (2:4 pruning). The [§70/01-nvidia-roadmap](01-nvidia-roadmap.md) numbers use the dense figure; the table above uses the headline vendor number. For capacity planning, use the dense figure unless your model supports 2:4 sparsity.

## 4. ROCm software stack: production status

ROCm is the CUDA-equivalent stack: HIP (the CUDA-source-portable C++ language), rocBLAS / hipBLASLt (BLAS), MIOpen (DNN primitives), RCCL (the NCCL-equivalent collectives library), MIGraphX (the inference-time graph compiler), and ROCm-Triton (a HIP backend for the Triton DSL). The 7.x line is the production cadence; 7.2.3 is current as of late 2025 and 7.12.0-preview is the development branch.

The vLLM-on-ROCm path is the most mature. The official AMD-maintained `rocm/vllm` Docker image supports MI300X, MI325X, MI350X, and MI355X with first-class status in vLLM V1's worker layer; configuration switches like `VLLM_USE_TRITON_FLASH_ATTN`, `NCCL_*` → `RCCL_*` env-var translations, and AITER-backed kernels are documented and benchmarked. SGLang has working ROCm support but tunes against a smaller test matrix. TensorRT-LLM does not run on ROCm (and will not, given its NVIDIA-vendor nature); the CUDA-only frontier kernels — DeepEP, certain flash-decoding kernels, NVIDIA-supplied MoE all-to-all — have AITER counterparts of varying maturity.

**AITER** (AI Tensor Engine for ROCm) is AMD's analog to a curated kernel zoo: hand-tuned HIP kernels for the operators that dominate LLM inference (GQA attention, mixed-precision GEMM, fused QKV, fused activation+norm, MoE token dispatch). AITER ships as a Python package and as kernels integrated into vLLM's ROCm path. It is the largest single source of LLM speedups on MI300X / MI355X and is where AMD's optimization effort is most visible. The pattern resembles CUTLASS + cuBLAS + custom-kernel zoo on the NVIDIA side, with a smaller team and a tighter set of supported configurations.

## 5. The ROCm gap analysis

Three structural deficits remain in the ROCm + vLLM software path relative to CUDA + vLLM. Listed in approximately decreasing order of business impact for frontier serving:

### 5.1 FlashAttention-3 / FA-4 parity

FA-3 on Hopper exploits warp-specialized producer/consumer pipelines and TMA-based asynchronous tile movement (see [§10/01-attention-kernels](../10-engine-core/01-attention-kernels.md)). FA-4 on Blackwell extends this with tcgen05 and tensor-memory operands. The CDNA 4 matrix-core ISA does not have one-to-one analogs of TMA or tcgen05; AITER's attention kernels use a different pipeline structure (tile-based with HIP async copies) and reach a fraction — empirically 60–80% on long-context, more on short — of the FA-3 throughput on equivalent hardware. The gap is narrowing as AITER's attention library matures and as ROCm-Triton's HIP backend gains the scheduling primitives needed to compile FA-style kernels at quality.

### 5.2 Custom MoE kernels

DeepEP and the surrounding ecosystem of MoE all-to-all kernels (see [§20/03-moe-inference](../20-distributed-inference/03-moe-inference.md)) are built on NVIDIA-specific primitives: NVSHMEM, IBGDA-driven RDMA over NVLink/IB, and specific NVLink topology assumptions. RCCL has all-to-all but not the same kernel-fused dispatch/combine that DeepEP provides. AITER ships an MoE dispatch kernel; it does not yet match DeepEP's overlap of compute and communication. For dense models this gap is invisible; for large-EP MoE serving (DeepSeek-V3-class, Kimi K2-class, frontier MoE in general), it is decisive.

### 5.3 Graph capture for agentic / structured-output decode

Agentic decode loops (tool calling, structured-output grammars, speculative decoding with verification — see [§60/02-structured-output-and-tools](../60-adjacent-workloads/02-structured-output-and-tools.md) and [§10/05-speculative-decoding](../10-engine-core/05-speculative-decoding.md)) hit the CPU launch overhead floor at small batch sizes. CUDA Graphs and `torch.compile`'s piecewise capture (see [§10/08-cuda-graphs-compilation](../10-engine-core/08-cuda-graphs-compilation.md)) eliminate per-step launch latency on NVIDIA. ROCm has HIP graphs, but the integration into vLLM's compile path and the coverage of dynamic-shape decode patterns (variable batch, variable sequence length, branch on grammar state) lag the CUDA path. As of mid-2026 the practical recommendation is that AMD deployments tolerate slightly higher TPOT on agentic workloads or eat the launch overhead; full parity is on the public roadmap and not yet shipping.

A fourth gap, less structural and more about ecosystem velocity: when a new SOTA technique lands (a new attention variant, a new spec-dec drafter, a new quantization kernel), the CUDA implementation typically ships first, often by a quarter or two. By the time the ROCm port lands, a second-generation CUDA kernel is in development. The gap is closing as AMD invests in AITER and in upstream contributions to vLLM, but the velocity asymmetry is a structural fact of a smaller ecosystem.

## 6. MLPerf Inference v6.0 (2026): where AMD lands

MLPerf Inference v6.0 (the closed-division 2026 round) gives the cleanest cross-vendor numbers. AMD's MI355X submissions:

- **Llama-2 70B (offline / server)**: competitive with B200 within ~10–20% on offline throughput; closer on server-mode latency-bounded throughput at the relevant SLOs.
- **Mixtral 8x7B**: similarly competitive — the model is small enough that the MoE all-to-all gap doesn't dominate.
- **Llama-3.1 405B**: not yet competitive end-to-end. The combination of a larger model (more weight bandwidth pressure where MI355X's capacity helps) and a more complex serving regime (where the FA-3 and graph-capture gaps compound) leaves a meaningful per-token throughput delta.
- **SDXL (image diffusion)**: AMD's submission used the IREE-compiled path (see sidebar) and was competitive on per-image latency.

The aggregate reading, with the caveat that MLPerf submissions are heavily tuned and may not reflect median production behavior, is that AMD has reached competitive parity on standard dense-model serving up to roughly 70B class, and remains 1.5–2× behind on frontier-scale dense and on MoE-heavy serving. The gap on standard workloads is small enough that capacity advantages (MI355X's 288 GB) and TCO can flip the decision in AMD's favor for HBM-bound deployments.

## 7. Sidebar B5 — Cross-vendor compiler stack

The full treatment of the LLM compiler stack is in [§10/08-cuda-graphs-compilation](../10-engine-core/08-cuda-graphs-compilation.md); this sidebar sketches the cross-vendor IR / kernel-DSL story.

- **MLIR as common ground.** Both vendors' modern compiler efforts are MLIR-based. MLIR is a multi-level intermediate representation framework originally from Google; it provides reusable IR dialects (linalg, vector, scf, gpu) and a pass infrastructure that can target multiple backends. NVIDIA's modern compiler internals (parts of cuDNN's graph backend, parts of the Triton compiler) are MLIR-based. AMD's investments are heavier on MLIR by necessity — it provides the abstraction layer that makes a single kernel description retargetable to HIP.

- **IREE.** AMD-backed (with broader open-source contribution) MLIR-based compiler that takes a high-level model graph (StableHLO, Torch-MLIR) and lowers to vendor-specific binaries. AMD's MLPerf SDXL submission used IREE to compile the diffusion graph to HIP; the same toolchain targets CUDA, CPU, and Vulkan backends. IREE is the cleanest example of a portable, MLIR-based, end-to-end inference compiler; its production footprint for LLM serving is still smaller than vLLM-on-ROCm.

- **Triton.** Originally CUDA-only; the HIP backend (ROCm-Triton) compiles the same Triton DSL to MI300X / MI355X. Quality lags CUDA Triton by one generation, especially for kernels that exploit Hopper-specific features (warp specialization, TMA). Triton is the path by which many community and research kernels (FlashAttention variants, fused MoE primitives, custom GEMM) reach AMD hardware without a hand-written HIP rewrite.

- **AITER / Wave.** AMD's curated kernel zoo; "Wave" is the in-development DSL for authoring AITER kernels at a level above HIP. Conceptually parallel to CUTLASS's role on the NVIDIA side: a higher-level abstraction over the matrix-core ISA, intended as the building block for an AMD-native FA-3 equivalent.

- **Toolchain naming map.** For readers translating between stacks: cuBLAS ↔ rocBLAS / hipBLASLt; cuDNN ↔ MIOpen; NCCL ↔ RCCL; CUTLASS ↔ AITER / Wave; cuGraph (CUDA Graphs) ↔ HIP Graphs; TensorRT / TensorRT-LLM ↔ MIGraphX (with the caveat that MIGraphX has narrower LLM-specific coverage).

The portability gradient runs roughly: model graph (StableHLO / ONNX) is fully portable; Triton-DSL kernels are portable with quality loss; CUTLASS-level kernels require a HIP rewrite (as AITER); and PTX-level or Hopper/Blackwell-specific intrinsics are not portable at all.

## 8. Intel Gaudi (light coverage)

Intel Gaudi 3 (2024, the current generation) ships 96 GB HBM2e at 3.7 TB/s with FP8 and BF16 matrix-engine support and 24× 200 GbE Ethernet ports integrated on-package for scale-out. The Ethernet integration is the architecturally distinctive choice: Gaudi treats Ethernet RoCE as the primary collective fabric rather than as a fallback to a vendor scale-up bus.

Production status. vLLM and SGLang have experimental Gaudi support via Intel-maintained backends; the matrix of supported models, quantization formats, and serving features is narrower than ROCm and substantially narrower than CUDA. Intel Gaudi is rarely the default choice for greenfield deployments; it shows up most often in Intel-anchored enterprise estates where the deployment is part of a larger Xeon + Gaudi + OneAPI commitment, or in workloads where the per-port-per-rack TCO calculation favors Gaudi's integrated Ethernet over separately purchased InfiniBand fabric. On peak per-GPU throughput Gaudi 3 is not directly competitive with MI355X or B200; the positioning is on TCO for specific deployment shapes.

**Roofline placement.** At 96 GB HBM2e and 3.7 TB/s, the Gaudi 3 bandwidth-compute ratio places its ridge at approximately 3,700 / 895 ≈ 4.1 FLOP/byte at BF16 (using ~895 TFLOP/s BF16 dense per Intel's datasheet; hedge: Intel-supplied). This is modestly lower than the H100 ridge (~293 FLOP/byte at BF16), reflecting HBM2e's lower bandwidth-to-compute ratio than HBM3. For bandwidth-bound decode at batch 1, Gaudi 3's token throughput at BF16 is bounded by 3.7 TB/s / (2 × model_bytes) — for a 7B BF16 model (~14 GB), that is roughly 3.7 TB/s / 14 GB ≈ 264 tok/s/card theoretical peak. Independent third-party benchmarks from ArtificialAnalysis (as of mid-2026) place Gaudi 3 at roughly 20–50% behind MI300X on the models tested (Llama-3-8B and 70B at INT8); Intel-supplied numbers position Gaudi 3 closer to MI300X parity on BF16. The gap to MI355X is larger — MI355X has 2.2 TB/s more HBM bandwidth and native FP4/FP6 paths Gaudi 3 lacks.

**When Gaudi 3 is the right choice.** The honest decision frame has three scenarios where Gaudi 3 wins or ties:

1. *Intel-anchored enterprise estates.* Organizations running Xeon + Gaudi racks under an Intel OneAPI commitment get Gaudi 3 as the AI accelerator in a consolidated vendor relationship. TCO math for the full stack (compute + networking + vendor support) can favor Gaudi in this context because the Ethernet-only fabric (no separate InfiniBand fabric purchase) reduces capital and operational complexity for deployments below a few hundred cards.

2. *BF16 workloads without quantization requirements.* Gaudi 3's FP8 and BF16 support covers the common case; it lacks native FP4 (Blackwell NVFP4 and AMD MXFP4). For workloads where FP8 is the minimum acceptable precision and FP4 is not needed, Gaudi 3 is competitive.

3. *Compliance-sensitive on-prem deployments.* Gaudi 3's PCIe form factor and commodity-Ethernet fabric simplify on-prem deployment in regulated environments where GPU-direct RDMA over InfiniBand is considered an additional attack surface or requires network-security exceptions.

**Where the CUDA moat dominates.** For any of the following, Gaudi 3 is not the right default: frontier MoE serving (DeepEP and EP-scale all-to-all have no Gaudi equivalent), long-context inference requiring FA-3/FA-4 kernels, agentic decode loops requiring full CUDA-graph capture, or deployments where the production model zoo includes models relying on Triton-compiled custom kernels. The vLLM and SGLang Gaudi backends are maintained by Intel with narrower feature coverage than the CUDA and ROCm backends; the model matrix as of mid-2026 covers major dense models (Llama-3, Mistral, Qwen3) but lags on MoE (DeepSeek-V3 is partial) and multimodal (no parity with CUDA on VLMs).

**Software direction.** Intel's Gaudi software path runs through the Habana SynapseAI SDK (C++ graph compiler, not Triton), with the Intel Extension for PyTorch (IPEX) providing the eager-mode bridge and vLLM/SGLang for serving. The Intel Gaudi Model References repo ships optimized kernels for the model families above. The MLIR/IREE portability story (§7) is relevant here: as IREE adds Gaudi backends, more models gain Gaudi support through the portable-kernel pathway rather than requiring Habana-native kernel rewrites.

A successor (Gaudi 4 / Falcon Shores) was previously on the public roadmap and has been repositioned as a converged GPU/AI part on the Intel Data Center GPU side; the LLM-inference relevance is still emerging and not load-bearing for current production.

## Current production state

As of mid-2026, the practical AMD-vs-NVIDIA decision for LLM serving is sharper than it was in 2024 but still rarely a coin-flip. MI355X is the right choice when capacity dominates: long-context serving where KV cache pressure exhausts B200's 192 GB before MI355X's 288 GB, single-node 405B-class deployments where the 1.5× capacity edge avoids cross-node tensor parallelism, or HBM-bound 70B serving where the 8 TB/s matches B200 at lower TCO. NVIDIA remains the default for frontier MoE serving (large-EP DeepSeek-V3-class with DeepEP), agentic and structured-output workloads where graph capture latency matters, and any deployment whose roadmap depends on shipping the latest research kernel within a quarter of its public release. The gap is narrower on standard workloads than the 2023 baseline suggested and wider on frontier than the 2025 marketing implied.

The Helios rack and UALink 1.0 are the open question of late 2026. If realized UALink performance approaches NVLink 5 — a hedge, since the only public numbers are AMD's projections — then AMD has a credible NVL72 competitor with an open-standard fabric story that resonates in hyperscaler procurement. If UALink lands meaningfully behind on latency or aggregate bandwidth, MI455X reverts to a per-node-competitive part without the rack-scale answer. Production signals to watch through 2027: the first non-AMD UALink switch silicon (Astera Labs and Marvell have announced parts), the first independent benchmark of all-reduce on a fully-populated Helios rack, and whether a frontier-lab inference workload publicly lands on Helios at scale.

Intel Gaudi remains a niche play. Gaudi 3 deployments are stable where they exist but are not expanding the way MI300X / MI355X deployments expanded across 2024–2026. The cross-vendor compiler stack — MLIR, IREE, ROCm-Triton — is the structural mechanism by which the long-tail kernel gap closes; production investment from AMD in AITER and from the broader community in IREE has accelerated meaningfully through 2025–2026, and the 2027-and-beyond trajectory depends on whether portable kernel languages (Triton, MLIR-linalg, Wave) reach quality parity with hand-tuned vendor zoos before the next hardware generation invalidates the current optimization targets.
