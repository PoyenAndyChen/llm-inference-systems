# GPU Hardware Primer

**After reading this chapter, the reader will be able to:**

- Describe the GPU execution model (SIMT, warps, thread blocks, occupancy) at the level needed to read a kernel description and understand what FlashAttention's tiling decisions are optimizing.
- Place every level of the GPU memory hierarchy — registers, shared memory / L1, L2, HBM, NVLink, PCIe — on a single mental table of capacity, bandwidth, and latency, and explain which level holds which inference state (weights, KV cache, attention tiles, activations, gradients-of-the-moment).
- Read the names of the modern Tensor Core instructions (WGMMA on Hopper, tcgen05 on Blackwell), the data movement instructions (TMA, TMEM), and the intra-node interconnect (NVLink 4 / 5, NVSwitch) and connect each one to the workload — attention, GEMM, tensor-parallel all-reduce, expert-parallel all-to-all — that motivates them.

This chapter does not retread the FLOP and bandwidth arithmetic of [§00/02-transformer-arithmetic-roofline](02-transformer-arithmetic-roofline.md). The job here is the *why* behind the hardware: why a GPU is organized as a sea of warps and SMs, why the memory hierarchy spans two orders of magnitude across five levels, why Tensor Core generations have doubled effective throughput by halving precision, and why NVLink rather than PCIe is the structural fact that makes intra-node tensor parallelism and large-EP MoE serving practical. The full hardware deep-dive — Hopper SM internals, Blackwell's asymmetric scaling, GB200 NVL72, the Rubin roadmap — is deferred to [§70/01-nvidia-roadmap](../70-hardware/01-nvidia-roadmap.md). Numbers anchor on H100 SXM5 because the median production token in 2026 still flows over Hopper; H200 and B200 figures recur later, with vendor-supplied numbers flagged as such.

## 1. The GPU execution model

A GPU is not a CPU with more cores. It is a throughput machine whose organizational unit is the *warp*: 32 threads that execute the same instruction at the same time on different data. This is single-instruction multiple-thread (SIMT), the model NVIDIA introduced with the original CUDA architecture and which every successor — Volta, Ampere, Hopper, Blackwell — has refined without abandoning.

### 1.1 Warps, thread blocks, and the SIMT contract

A CUDA kernel launches a *grid* of *thread blocks* (the CUDA Programming Guide also calls these cooperative thread arrays, CTAs). Each thread block is assigned to exactly one Streaming Multiprocessor (SM) and runs to completion there; thread blocks do not migrate. Inside a thread block, threads are partitioned into warps of 32 consecutive threads. The hardware scheduling unit is the warp, not the individual thread.

The SIMT contract: at every clock an SM's warp scheduler picks one resident warp and issues one instruction; all 32 threads in that warp execute that instruction in lockstep. When threads in a warp branch — `if (threadIdx.x < 16) { ... } else { ... }` — the hardware *masks* inactive lanes and executes both sides serially. This is *warp divergence*, the single most expensive thing a kernel can do per cycle. Production attention kernels are written to keep warps coherent: every thread in a warp follows the same control flow on the same shape of input.

A thread block is the unit that shares fast on-chip memory. Threads within a block synchronize via `__syncthreads()` and communicate through shared memory (SRAM allocated per-block out of the SM's L1/shared partition). Threads in different blocks cannot synchronize cheaply; they coordinate only through HBM or the much slower kernel-launch boundary. This is the structural reason the *tile* is the unit of work in FlashAttention and CUTLASS: a tile is precisely "as much work as one thread block can hold in shared memory and complete without leaving the SM."

Hopper extended this hierarchy with *thread block clusters*: 2–16 thread blocks the scheduler co-locates onto one GPC (Graphics Processing Cluster) and which can address each other's shared memory through *Distributed Shared Memory* (DSMEM). FA-3's producer/consumer warp specialization exploits this; the mechanics are in [§10/01-attention-kernels](../10-engine-core/01-attention-kernels.md).

### 1.2 Occupancy

An SM holds many warps simultaneously — up to 64 on H100, scheduled by four warp schedulers each capable of issuing one instruction per cycle. Most resident warps are *not* executing at any given moment; they are stalled waiting on memory, dependent operands, or a Tensor Core retirement. The fraction of issue slots the scheduler can keep filled is *occupancy*, and occupancy is the GPU's mechanism for *latency hiding*: an HBM load at ~600 cycles does not stall the SM as long as some other warp has 600 cycles of work to run. Pool depth is bounded by per-SM resources — H100 has 65,536 32-bit registers, 228 KB usable shared memory, and a hard cap of 2,048 resident threads (64 warps). A kernel using 64 registers per thread caps at $65536/(64 \cdot 32) = 32$ resident warps; a kernel using 96 KB of shared per block fits at most two blocks per SM.

A kernel hits "100% occupancy" only if no resource binds. In practice, FlashAttention and other tile-based kernels deliberately *trade* occupancy for arithmetic intensity — using more shared memory per block to keep larger tiles on chip, accepting fewer blocks per SM in exchange for more useful work per HBM byte fetched. The optimization is not "maximize occupancy"; it is "find the occupancy that maximizes effective FLOP/s under the kernel's data-reuse pattern." This logic underlies every FA-1 → FA-2 → FA-3 → FA-4 redesign [see §10/01-attention-kernels](../10-engine-core/01-attention-kernels.md).

### 1.3 The execution hierarchy on H100

The full hierarchy on H100 SXM5, from the chip down to the thread:

```
GH100 die
├── 132 SMs (8 GPCs × 16–18 SMs; some disabled for yield)
│   └── Each SM
│       ├── 128 FP32 CUDA cores
│       ├── 64 INT32 CUDA cores
│       ├── 64 FP64 CUDA cores
│       ├── 4 × 4th-gen Tensor Core units
│       ├── 4 warp schedulers (each issues 1 instr/cycle)
│       ├── 65,536 × 32-bit registers
│       ├── 256 KB combined L1 / shared memory (228 KB usable for shared)
│       ├── 1 Tensor Memory Accelerator (TMA) unit
│       └── up to 64 resident warps (2,048 threads)
└── 50 MB L2 cache (shared across all 132 SMs)
└── 80 GB HBM3 at 3.35 TB/s
└── 18 NVLink 4 lanes (900 GB/s bidirectional aggregate)
└── PCIe Gen5 x16 (128 GB/s bidirectional)
```

The four warp schedulers per SM are the reason H100 can issue a TMA load, an HBM read, a Tensor Core matmul, and an integer-pointer update *in the same cycle from four different warps*. Software pipelining — keeping the Tensor Cores fed while async copies remain in flight — is the entire point of FA-3's producer/consumer split. Blackwell's B200 follows the same structural template with different counts: dual-die reticle-stitched (two GB100-class chips presented as one logical GPU), more SMs total, 5th-generation Tensor Cores, 192 GB HBM3e at ~8 TB/s, and the new tcgen05 / TMEM machinery covered in §6 below. The full comparison lives in [§70/01-nvidia-roadmap](../70-hardware/01-nvidia-roadmap.md).

## 2. The memory hierarchy

The GPU memory hierarchy is brutally non-uniform: bandwidth and latency change by two orders of magnitude between adjacent levels. Where weights and KV cache live — and how they move — determines whether a kernel runs at peak or at 5% of peak.

### 2.1 The hierarchy table

The reference table the rest of the book cites:

| Level | Capacity (H100 SXM5) | Bandwidth | Latency | Scope |
|---|---|---|---|---|
| Registers | 65,536 × 32-bit per SM | n/a (issue-rate) | 0 cycles | Per thread |
| Shared memory / L1 | 228 KB per SM (256 KB combined L1+shared) | ~33 TB/s aggregate across SMs | ~30 cycles | Per thread block (cluster-shared on Hopper) |
| L2 cache | 50 MB | ~12 TB/s | ~150–200 cycles | Per GPU |
| HBM3 | 80 GB | 3.35 TB/s | ~400–700 cycles | Per GPU |
| NVLink 4 | — | 900 GB/s bidirectional per GPU | ~1–2 μs | Intra-node (8-GPU node, NVSwitch all-to-all) |
| PCIe Gen5 ×16 | — | 128 GB/s bidirectional | ~1–5 μs | CPU ↔ GPU, host staging |

Numbers for shared-memory aggregate bandwidth and L2 bandwidth are derived from architectural disclosures and microbenchmark studies; expect ±15% in production conditions. HBM bandwidth and NVLink/PCIe figures are vendor-specified peaks; effective bandwidth is typically 70–85% of peak under realistic kernel pressure. H200 increases HBM to 141 GB and 4.8 TB/s on the same GH100 die; B200 lifts HBM to 192 GB at ~8 TB/s on a dual-die reticle-stitched part. The B200 figures are NVIDIA-reported; independent microbenchmarks ([ASPLOS-Blackwell](../papers.md#asplos-blackwell)) corroborate the order of magnitude but report effective sustained bandwidth ~5–10% below the peak headline.

### 2.2 What lives where, and why it matters

**Registers** hold a thread's working state. Register pressure most often binds occupancy: FA-2 keeps running softmax statistics in register tiles, which keeps the inner loop fast but caps occupancy. Registers are the only memory level with zero access latency.

**Shared memory** is where the per-block tile lives. In FlashAttention, blocks of Q, K, V are loaded from HBM into shared memory once per outer-loop iteration; the inner loop computes $Q K^\top$, online softmax, and the $\text{softmax} \cdot V$ accumulation entirely against shared memory. The tiled-attention idea collapses to one sentence: shared memory is ~100× faster than HBM per byte, so reducing HBM reads of Q/K/V by a factor of (tile-size / seq-length) is a 100×-bounded speedup; FA-1 is the argument that the bound is approached. Shared memory is partitioned into 32 banks of 4 bytes each — if two threads in a warp target the same bank at different addresses, the access serializes. Avoiding bank conflicts is the reason production attention kernels lay out tiles with seemingly arbitrary strides and swizzle patterns.

**L2 cache** sits between SMs and HBM. At 50 MB on H100 it is large enough that a small model's full layer weights can be resident if access patterns are right. Persistent kernels and the `cudaAccessPolicyWindow` residency hint signal to hardware that certain regions should preferentially stay in L2; for long-context attention with reused KV blocks, L2 is the level where prefix-cache hits are cheapest.

**HBM** holds the weights, the KV cache, activations between layers, and everything that does not fit in L2. The single most important formula in LLM serving — that decode reads the full weight tensor from HBM once per token at small batch — is a direct consequence of HBM's place in the hierarchy. Chapter 2's roofline analysis compresses to "decode is HBM-bound"; every quantization, KV-compression, and attention-variant trick in this book is in some sense an attempt to move bytes out of HBM and into something faster or smaller [see §10/04-quantization](../10-engine-core/04-quantization.md), [see §30/01-kv-compression](../30-kv-cache/01-kv-compression.md). HBM access latency at ~600 cycles is the latency that occupancy must hide; the async-copy machinery (TMA on Hopper, tcgen05's pipelined stages on Blackwell) is the hardware response.

**NVLink and PCIe** connect GPU to GPU and GPU to CPU respectively. Within an 8-GPU HGX node or 72-GPU NVL72 rack, NVLink carries tensor-parallel all-reduce, expert-parallel all-to-all, and KV transport between prefill and decode workers. PCIe carries weight loading, KV offload to CPU DRAM, and (on consumer hardware) inter-GPU traffic. PCIe Gen5 ×16 at 128 GB/s is ~7× slower per GPU than NVLink 4 and ~14× slower than NVLink 5; this asymmetry is the structural reason every modern engine keeps collective traffic on NVLink and pushes it to InfiniBand only when a node boundary is unavoidable [see §70/05-networking-fabric](../70-hardware/05-networking-fabric.md).

### 2.3 An ASCII view of the hierarchy

```
              ┌─────────────────────────────────────────────────────┐
              │  Host DRAM (CPU)              HBM-class capacity     │
              │      │                                               │
              │   PCIe Gen5 x16 (128 GB/s)                           │
              │      │                                               │
              ▼      ▼                                               ▼
    ┌─────────────────────┐    NVLink 4 / 5 (900 GB/s | 1.8 TB/s)
    │ GPU 0               │ ◄──────────────────────────────────────► GPU 1..7
    │                     │
    │  HBM3 (80 GB,       │
    │   3.35 TB/s)        │  ◄── weights, KV cache, activations
    │      │              │
    │  L2 (50 MB,         │  ◄── staging, persistent kernel residency
    │   ~12 TB/s)         │
    │      │              │
    │  ┌───┴───────┐      │
    │  │ SM 0      │ ...  │  132 SMs share L2 + HBM
    │  │           │      │
    │  │ Shared/L1 │      │  ◄── tile storage, ~33 TB/s aggregate
    │  │ (228 KB)  │      │
    │  │   │       │      │
    │  │ Registers │      │  ◄── per-thread state, 0-cycle access
    │  │ (65,536   │      │
    │  │  × 32-bit)│      │
    │  └───────────┘      │
    └─────────────────────┘
```

A bandwidth heuristic the rest of the book uses: each step *up* the hierarchy is roughly 10× faster and 100× smaller. The kernel's job is to get a byte to the highest level possible the smallest number of times.

## 3. Tensor Cores and mixed-precision math

A CUDA core is a scalar arithmetic unit; a Tensor Core is a matrix-matrix unit, multiplying two small matrices and accumulating into a third in a single instruction. Tensor Cores are the differentiator between "GPU as a vector machine" and "GPU as deep-learning hardware." Every generation since Volta has roughly doubled effective throughput by halving precision.

### 3.1 The lineage of Tensor Core generations

**1st-generation (Volta V100, 2017).** $4\times4\times4$ FP16-input tiles, FP32 accumulation; the first dedicated deep-learning matrix unit on a GPU.

**2nd-generation (Turing, 2018).** Added INT8 and INT4 paths — the first commercial signal that low-precision was where inference economics would be won.

**3rd-generation (Ampere A100, 2020).** Added BF16, TF32 (FP32 range, FP16 mantissa), and sparse Tensor Cores for 2:4 structured sparsity. A100's 312 TFLOP/s BF16 set the baseline subsequent generations are measured against.

**4th-generation (Hopper H100, 2022).** Added FP8 (E4M3 and E5M2), with the Transformer Engine v1 providing per-tensor automatic scaling. The marquee instruction is *WGMMA* (warp-group matrix-multiply-accumulate): a single asynchronous instruction on $64\times16\times16$ tiles driven by all four warps of a warp group. WGMMA is what made FA-3 possible — its asynchrony lets Tensor Core compute and TMA-based memory transfer overlap, turning the kernel into a pipeline rather than a fetch-then-compute loop. FP8 doubles peak throughput vs. FP16 (1,978 vs. 989 TFLOP/s on H100 SXM5).

**5th-generation (Blackwell B100/B200, 2024–2025).** Added NVFP4 (E2M1 with per-block E4M3 scale at block size 16) and MXFP4/MXFP8 (the open block-scaled formats from the OCP Microscaling Formats spec). Introduced *tcgen05*, a Tensor Core instruction family that operates not on shared memory but on *Tensor Memory* (TMEM), a new on-chip buffer separately addressed by the Tensor Cores. tcgen05 instructions are issued, run asynchronously, drain into TMEM, and free the warp scheduler to issue the next instruction immediately. NVFP4 doubles peak vs. FP8.

Each precision halving has come with software support to keep accuracy intact: per-block scale factors (NVFP4's 16-element block, MXFP's 32-element block), automatic mixed-precision via the Transformer Engine, and increasingly quantization-aware training where pre-training already accounts for the format. The full taxonomy is in [§10/04-quantization](../10-engine-core/04-quantization.md); the headline as of mid-2026 is that FP8 is the dominant low-precision format on Hopper production deployments, and NVFP4 is displacing it on Blackwell-default frontier-lab serving ([SageAttn3](../papers.md#sageattn3), [FlashAttn-4](../papers.md#flashattn-4), [TK-2.0](../papers.md#tk-2-0)).

### 3.2 Peak throughput comparison

The reference table the rest of the book cites for accelerator peak throughput. All NVIDIA dense numbers are without sparsity; sparse-Tensor-Core figures are 2× the dense numbers for kernels that can exploit 2:4 structured sparsity, which most LLM inference cannot.

| Accelerator | FP16 / BF16 | FP8 | FP4 / NVFP4 | HBM bandwidth |
|---|---|---|---|---|
| A100 SXM4 80 GB | 312 TFLOP/s | — | — | 2.04 TB/s |
| H100 SXM5 80 GB | 989 TFLOP/s | 1,978 TFLOP/s | — | 3.35 TB/s |
| H200 SXM5 141 GB | 989 TFLOP/s | 1,978 TFLOP/s | — | 4.8 TB/s |
| B200 SXM (dual-die) | ~2,250 TFLOP/s | ~4,500 TFLOP/s | ~9,000 TFLOP/s | ~8.0 TB/s |
| GB300 (B200 Ultra) | — | — | ~15 PFLOP/s | 8 TB/s |
| MI300X (CDNA 3) | 1,307 TFLOP/s | 2,614 TFLOP/s | — | 5.3 TB/s |
| MI355X (CDNA 4) | — | ~5 PFLOP/s | ~10 PFLOP/s | 8 TB/s |

B200 and GB300 figures are NVIDIA-reported headline peaks. Independent microbenchmarks ([ASPLOS-Blackwell](../papers.md#asplos-blackwell)) report FP4 sustained throughput ~70% of headline at typical attention shapes; the 71% utilization FA-4 reports on B200 BF16 ([FlashAttn-4](../papers.md#flashattn-4)) is consistent. AMD MI355X figures are vendor-reported and rest on the CDNA 4 native FP4/FP6 path with one shared datapath; production-comparable independent benchmarks are still accumulating in MLPerf v6.0 ([AMD-MLPerf6](../papers.md#amd-mlperf6)). MI300X numbers are well-established from ROCm production.

The interpretive lens: peak FP throughput has near-doubled per generation ($312 \to 989 \to \sim 2{,}250$ FP16 across A100→H100→B200), while HBM bandwidth has scaled only ~2× over the same span ($2.04 \to 3.35 \to 8.0$ TB/s). The compute-to-bandwidth ratio has grown, the ridge point has moved right, and every chapter that touches operating-point analysis takes this fact for granted.

## 4. Memory bandwidth vs. compute: the roofline, revisited

Chapter 2 derived the roofline; this section ties the hardware numbers above back to it. The ridge point at FP16 on H100 is $I_{\text{ridge}} = 989 / 3.35 \approx 295$ FLOP/byte; on B200 it is $\approx 281$. At FP8 the ridge doubles (~590 on H100, ~562 on B200) because peak doubles while bandwidth holds. At NVFP4 on B200 it doubles again to ~1,125. A kernel below ridge is bandwidth-bound, a kernel above ridge is compute-bound.

The implication for low-precision formats is subtle: FP8 vs. FP16 *raises the ceiling* but does not move the ridge proportionally to byte traffic — bytes per parameter halve too, so a memory-bound decode at FP16 stays memory-bound at FP8, with halved per-token traffic and ~2× decode throughput at fixed batch. This is why FP8 has been near-universal for Hopper production decode since 2024: the structural decode win is a *bandwidth* win, not a *FLOP* win. NVFP4 on Blackwell repeats this story one tier down (4× peak FLOP/s, 4× weight compression vs. FP16), which is the structural reason Blackwell-default deployments expect 3–4× decode TPS vs. Hopper at fixed batch on the same model. Worked numbers are in [§00/02-transformer-arithmetic-roofline](02-transformer-arithmetic-roofline.md); the chapters that exploit this most heavily are [§10/01-attention-kernels](../10-engine-core/01-attention-kernels.md), [§10/04-quantization](../10-engine-core/04-quantization.md), and [§20/01-parallelism-strategies](../20-distributed-inference/01-parallelism-strategies.md).

## 5. NVLink and multi-GPU within a node

A single H100 has 80 GB of HBM. Llama-3.1-70B at FP16 occupies 140 GB of weights — twice the capacity of one H100. DeepSeek-V3 at FP8 occupies ~336 GB for total weights, fitting on no single GPU in 2026. Frontier-scale serving requires multi-GPU parallelism, and intra-node parallelism requires that GPUs talk to each other faster than PCIe allows. NVLink is the answer.

### 5.1 What NVLink is

NVLink is a point-to-point GPU-to-GPU interconnect. NVSwitch is a switch ASIC that turns the point-to-point links into an all-to-all fabric within a node.

| Generation | First product | Per-GPU bidirectional BW | Aggregate node BW |
|---|---|---|---|
| NVLink 3 | A100 | 600 GB/s | 4.8 TB/s in 8-GPU NVSwitch node |
| NVLink 4 | H100 | 900 GB/s | 7.2 TB/s in 8-GPU HGX H100 |
| NVLink 5 | B100/B200 | 1.8 TB/s | 14.4 TB/s in 8-GPU HGX B200; 130 TB/s aggregate in GB200 NVL72 |
| NVLink Fusion | announced 2025-05 | licensable | scale-up fabric for partner CPUs / ASICs |

H100's 900 GB/s bidirectional per GPU is built from 18 NVLink 4 lanes at 50 GB/s each. NVSwitch (3rd-gen on Hopper) provides full bisection so every GPU can talk to every other GPU at full per-GPU bandwidth simultaneously; the standard 8-GPU HGX H100 baseboard uses 4 NVSwitch chips. GB200 NVL72 takes the same idea to rack scale: 72 B200 GPUs and 36 Grace CPUs in a single liquid-cooled chassis with 5th-gen NVSwitches forming one 72-way NVLink fabric. The rack is treated as one logical GPU domain — tensor parallelism can span the full 72-way, expert-parallel all-to-all uses the NVL72 fabric directly, and there is no PCIe between any two compute units in the rack. The deep-dive is in [§70/01-nvidia-roadmap](../70-hardware/01-nvidia-roadmap.md).

### 5.2 What NVLink enables for LLM inference

Three workload patterns make NVLink the structural enabler of multi-GPU inference.

**Tensor parallelism (TP).** Sharding a matmul across $T$ GPUs requires an all-reduce of the activation tensor at each layer's output. With NVLink 4's 900 GB/s per GPU and ring-allreduce's $\approx 2(N-1)/N$ scaling factor, a single per-token activation all-reduce takes a few microseconds at TP=8 — short enough to overlap with the next layer's compute under software pipelining. TP at 8-way on a single HGX node is the default for 70B-class dense models in 2026 production [see §20/01-parallelism-strategies](../20-distributed-inference/01-parallelism-strategies.md).

**Expert parallelism (EP) all-to-all.** Large-EP MoE serving — DeepSeek-V3 at EP=64+, Kimi-K2 large-EP — moves tokens between experts during dispatch and combines them after expert computation. DeepEP (released by DeepSeek in February 2025) issues these all-to-alls over NVLink intra-node and IB inter-node; the intra-node NVLink phase is what makes large-EP serving practical at all. Without NVLink-class intra-node bandwidth, the all-to-all dominates iteration time [see §20/03-moe-inference](../20-distributed-inference/03-moe-inference.md).

**KV transfer for prefill-decode disaggregation.** When prefill and decode run on different GPUs (Splitwise → DistServe → Mooncake → Dynamo lineage; [see §20/02-prefill-decode-disagg](../20-distributed-inference/02-prefill-decode-disagg.md)), the KV cache produced by prefill must move to the decode worker before decode begins. Within a node this goes over NVLink at hundreds of GB/s, fast enough to hide behind the first decode iteration. Across nodes the same transfer goes over InfiniBand at ~50–100 GB/s (NDR/XDR) and becomes a scheduling concern, motivating the family of KV-transport libraries (NIXL, Mooncake TransferEngine, Perplexity TransferEngine).

The asymmetry between intra-node (NVLink, 900 GB/s Hopper / 1.8 TB/s Blackwell) and inter-node (IB NDR 400 Gb/s = 50 GB/s; XDR 800 Gb/s = 100 GB/s) is the structural fact that makes "TP within a node, PP/EP across nodes" the production-default parallelism layout in 2026. The full networking story is in [§70/05-networking-fabric](../70-hardware/05-networking-fabric.md). One footnote: NVIDIA's May 2025 *NVLink Fusion* announcement licenses NVLink IP to partner silicon (MediaTek, Marvell, Alchip, Fujitsu, Qualcomm). As of May 2026 no partner silicon has shipped in an NVLink Fusion configuration; treated as a sidebar in [§70/01-nvidia-roadmap](../70-hardware/01-nvidia-roadmap.md).

## 6. What the kernel writer sees

Modern attention kernels — FA-3 on Hopper, FA-4 on Blackwell, FlashMLA, FlashInfer's BSR family — are dense with vocabulary this section unpacks at the level needed to read the kernel chapter.

**Grid, block, warp.** A kernel launches a *grid* of *thread blocks*; each block contains warps of 32 threads. The grid shape maps onto the output tensor — typically one block per (batch, head, query-tile) triple in attention.

**Memory coalescing.** When 32 threads in a warp issue loads, the hardware satisfies them with a single 128-byte cache line transaction *if* the addresses are contiguous and aligned. Q/K/V tensors are laid out so the natural per-thread access pattern is coalesced — typically with the head dimension innermost, since 32 threads cover 32 elements (128 bytes at FP32, 64 at FP16, 32 at FP8).

**Bank conflicts.** Shared memory has 32 banks of 4 bytes; if two threads target the same bank at different addresses, accesses serialize. Attention kernels avoid this through *swizzling* — the storage offset is XORed with a bit pattern that breaks alignment between thread index and bank.

**TMA (Hopper).** The Tensor Memory Accelerator is a hardware engine for bulk async copies between HBM and shared memory. A single TMA instruction issues a tile copy and returns; the warp can immediately issue Tensor Core work on a *previous* tile. Without TMA, every byte transfer costs per-warp instruction-issue bandwidth. FA-3's producer/consumer warp specialization assigns one warp group to issue TMAs and another to consume tiles in WGMMA — and is the structural reason FA-3 cannot run on Ampere.

**WGMMA (Hopper).** Warp-group matrix-multiply-accumulate: a single asynchronous instruction issued by a warp group (4 warps, 128 threads) drives the Tensor Core through a $64 \times 16 \times 16$ matmul-accumulate. Asynchrony lets the kernel keep multiple WGMMAs in flight, the foundation of FA-3's ping-pong pattern.

**tcgen05 and TMEM (Blackwell).** The 5th-generation Tensor Core instruction family. Where WGMMA accumulates into registers, tcgen05 accumulates into *Tensor Memory* (TMEM), a new on-chip buffer separately addressed by the Tensor Cores. The pipeline deepens: shared → TMEM (input staging), TMEM → Tensor Core compute, Tensor Core → TMEM (accumulator residence), TMEM → shared/registers (drain). Each stage is async and the kernel writer schedules across them explicitly. FA-4 ([FlashAttn-4](../papers.md#flashattn-4)) is built on these primitives in CuTe-DSL ([CuTe-DSL](../papers.md#cute-dsl)); the kernel chapter develops this in detail.

The kernel-writer DSL landscape — Triton ([Triton-Anatomy](../papers.md#triton-anatomy)), CuTe-DSL, ThunderKittens ([ThunderKittens](../papers.md#thunderkittens), [TK-2.0](../papers.md#tk-2-0)), TileLang ([TileLang](../papers.md#tilelang)), and the AMD-side AITER / Wave stack ([HipKittens](../papers.md#hipkittens)) — is the subject of [§10/01-attention-kernels](../10-engine-core/01-attention-kernels.md). The vocabulary above is the minimum to follow that discussion.

## Current production state

As of May 2026, the production GPU population for LLM inference is Hopper-dominant by token volume. The median production token flows over an H100 SXM5 (80 GB HBM3, 3.35 TB/s) or H200 SXM5 (141 GB HBM3e, 4.8 TB/s); the H200 has roughly displaced the H100 for new decode-heavy clusters because its +43% HBM bandwidth translates near-linearly into decode TPS at fixed batch size on bandwidth-bound workloads. A non-trivial fraction of cost-optimized clusters still runs A100 80 GB (HBM2e, 2.04 TB/s); L40S and A10G are common for embeddings and small-model serving. Blackwell — B200 SXM, GB200 NVL72, GB300 / Blackwell Ultra NVL72 — is in active ramp at every major hyperscaler (CoreWeave from Dec 2024, Aug 2025 for GB300; Azure, OCI, GCP through 2025–2026). It is the default at frontier-lab production (the OpenAI Stargate buildout, xAI Colossus 2 at 2 GW, Meta Hyperion) and at the largest hyperscaler GB200/GB300 racks, but it is *not* the default at the median enterprise.

The kernel-software population mirrors the hardware split. FA-3 (2024) is the dominant production attention kernel because its installed base is Hopper. FA-4 ([FlashAttn-4](../papers.md#flashattn-4), March 2026) is Blackwell-only — built in CuTe-DSL on tcgen05 and TMEM, requiring the polynomial `exp2` workaround for Blackwell's MUFU bottleneck — and adoption tracks Blackwell ramp. FlashInfer ([FlashInfer](../papers.md#flashinfer)) is the cross-engine NVIDIA-side kernel collection used by vLLM, SGLang, MLC, and increasingly TRT-LLM; it provides the Block-Sparse-Row substrate that unifies paged, ragged, radix, and tree KV layouts. ROCm has a parallel AITER / Wave lineage; CuTe-DSL, ThunderKittens, TileLang, and Triton compete as DSLs above. NVIDIA stewardship of FlashInfer is widely reported but no formal acquisition has been confirmed.

NVLink-domain serving is the production unit for frontier-scale models: DeepSeek-V3 / V3.1 / V3.2, Kimi-K2, GPT-OSS large variants, and Llama-4 frontier MoE all serve over multi-rack NVL72-class topologies with intra-rack TP/EP and inter-rack PP/DP. The structural shift between 2024 and 2026 is that the unit of capacity planning has moved from "the GPU" to "the NVLink domain": a GB200 NVL72 rack is treated as one logical accelerator with 13.8 TB of HBM (72 × 192 GB) and 130 TB/s of intra-domain NVLink bandwidth, and the engine's parallelism strategy is chosen to fit that envelope. This is the frame [§20/01-parallelism-strategies](../20-distributed-inference/01-parallelism-strategies.md), [§20/02-prefill-decode-disagg](../20-distributed-inference/02-prefill-decode-disagg.md), and [§20/03-moe-inference](../20-distributed-inference/03-moe-inference.md) develop in detail.
