# OSS Landscape: A Comparative Survey

**After reading this chapter, the reader will be able to:**

- Place any inference engine on the reach–throughput–flexibility spectrum and explain which axis its design optimizes for.
- Choose a starting engine for a given deployment scenario — multi-tenant cloud fleet, maximum NVIDIA throughput, disaggregated P/D, Kubernetes-native autoscaling, on-device edge, or huge sparse MoE on a limited GPU budget — and justify why the alternatives are worse fits.
- Read the detailed chapters (01–10) in the correct order and know in advance which concepts from earlier sections of the textbook each chapter builds on.
- Identify the cross-cutting convergence trends — FlashInfer, NIXL, token-budget scheduling, radix caching, EPLB, CNCF-native orchestration — that make individual engines look increasingly similar at the API surface while diverging at the implementation level.

---

## Taxonomy of covered engines

The OSS LLM inference landscape is not a single market with interchangeable products; it is four overlapping stacks, each solving a different bottleneck. The chapters in this section follow that grouping.

### Cluster inference engines: vLLM, SGLang, TRT-LLM

These are the workhorses that actually run the model forward pass at production scale. Each is responsible for batching requests, managing a paged KV cache, executing attention kernels, and streaming tokens back to callers. All three now support disaggregated prefill/decode, multi-GPU tensor and pipeline parallelism, NVIDIA FP8 and NVFP4, speculative decoding, and an OpenAI-compatible HTTP API. The differences are in scheduler design philosophy, performance on specific hardware, degree of NVIDIA coupling, and operational surface area.

vLLM's defining property is breadth: it supports the widest hardware matrix (NVIDIA, AMD, Intel Gaudi, TPU, CPU), the most quantization schemes, and the most model architectures. Its V1 engine — separated into a front-end process, one EngineCore process per DP rank, and worker processes communicating over ZMQ + msgpack — makes the token-budget scheduler and hash-chained prefix cache the default for the industry. The token-budget scheduler collapses the historic prefill/decode distinction into a single `num_tokens_with_spec − num_computed_tokens` counter per request, with chunked prefill and speculative decode consuming the same budget path. Because the front-end process never holds the GIL during model execution, DP scaling is achieved by spawning additional EngineCore processes without touching front-end code — a clean horizontal scaling story that the legacy v0 engine lacked.

SGLang's defining property is forward-pass efficiency on NVIDIA hardware. Its `event_loop_overlap` scheduler prepares batch N+1 on CPU while batch N is running on the GPU, hiding Python and dispatch overhead behind the previous forward pass. RadixAttention is the entire KV cache management strategy: a global radix tree where every node represents a contiguous KV segment, supporting precise split-on-divergence, lock-reference counting, and multiple eviction strategies (LRU, LFU, FIFO, MRU, FILO, Priority, SLRU). The tree is exposed directly to the scheduler for cache-aware request admission — the Longest Prefix Match policy sorts the waiting queue by prefix depth before deciding batch composition — which is why SGLang achieves higher prefix-cache hit rates than vLLM on workloads with structured prompt reuse. The models it targets natively — DeepSeek, Kimi-K2, Qwen3-MoE, GPT-OSS — are the ones that matter most at the frontier in 2026.

TRT-LLM's defining property is ceiling performance on NVIDIA hardware. The build-time AOT compilation step (for the TRT engine path) or piecewise `torch.compile` (for the now-default PyTorch path) extracts the last FLOP through proprietary CUTLASS-DSL ("TRT-LLM-Gen") kernels, NVLink one-sided all-to-all for EP all-to-all without NVSHMEM serialization, Helix parallelism (same N GPUs reconfigured between attention and FFN phases per layer), NVLS multicast for TP all-reduce, and DWDP expert prefetch that eliminates the inter-rank synchronization in expert weight all-gather. None of these mechanisms are available to the portability-oriented engines. The TRT engine path additionally captures shape profiles at build time and uses TensorRT's tactic search to select the best GEMM kernel per layer; the result is that two identical H100s running different TRT plans for the same model may produce different throughput because of tactic selection differences — behavior that vLLM and SGLang, which rely on runtime autotuning or generic torch.compile, do not exhibit.

### KV cache infrastructure: LMCache and Mooncake

These are not inference engines. They are data-plane components that provide a shared, hierarchical KV cache *underneath* an inference engine. They sit at the boundary between the engine's local GPU KV pool and the outside world — CPU DRAM, NVMe, remote RDMA memory, and object stores.

LMCache is a Python-first library whose integration surface is the `KVConnectorBase_V1` interface in vLLM (and analogous hooks in SGLang and TRT-LLM). It treats KV reuse as a Knowledge Delivery Network: chunks of KV, keyed by a running prefix hash matching vLLM's `init_none_hash` format, are pushed and pulled across a configurable backend hierarchy — LocalCPUBackend (always-on staging tier) → LocalDiskBackend (NVMe async I/O) → GdsBackend (NVIDIA GPUDirect Storage) → P2PBackend (peer mesh via NIXL or sockets) → RemoteBackend (Redis/Valkey, S3, Mooncake Store, InfiniStore, EIC, SageMaker HyperPod, LMCache server protocol). The CacheGen arithmetic-coding compression scheme reduces the wire size of remote KV chunks by encoding per-layer K/V into 32-bin or 16-bin histograms; KIVI 2-bit quantization is an alternative serde plugin. The `StorageManager` runs an asyncio event loop in a dedicated thread with layerwise async saves that overlap with the next-layer model forward, keeping KV copy-out off the critical latency path.

Mooncake is a C++-core RDMA transfer engine and distributed object store that grew out of Moonshot AI's Kimi production stack. Its Transfer Engine abstracts a *segment* of remote memory — GPU VRAM, CPU DRAM, NVMe — behind a batch `TransferRequest` API that stripes large payloads across multiple NICs, saturating the aggregate fabric. A `(source, target)` pair is routed to the NIC closest in NUMA topology; on NIC failure the engine reroutes to an alternative without surfacing the error to the caller. The Mooncake Store layers an object abstraction on top: `MasterService` handles allocation and replica placement metadata out of the data path; Real Clients transfer data directly via TE; a two-phase commit (`PutStart` / `PutEnd`) ensures immutability once an object is acknowledged. The FAST '25 best-paper result demonstrated 190 GB/s aggregate fabric throughput on 8×400 Gbps RoCE. The two systems are complementary: LMCache is the cache controller, serde, and multi-backend connector; Mooncake is the data plane and distributed object store beneath it.

### Cluster orchestration: NVIDIA Dynamo, llm-d, AIBrix

These are Kubernetes-native control planes that sit *above* the inference engines and coordinate disaggregated prefill/decode routing, KV-aware load balancing, autoscaling, and multi-tenant request admission. None of them runs a model forward pass.

NVIDIA Dynamo is the highest-integration option: a Rust-core request plane, a dedicated KV router with a global radix index (`ConcurrentRadixTree` with per-worker sticky thread shards), a multi-tier KV Block Manager (KVBM, G1 GPU / G2 host / G3 SSD / G4 remote), and a Planner that uses SLA-aware performance models (Constant / ARIMA / Kalman / Prophet load predictors plus a performance interpolator for decode-side ITL prediction) to set replica counts. The routing cost function `cost = w · prefill_blocks + decode_blocks` is a single-knob trade-off between TTFT optimization and ITL load smoothing; the same formula underpins the llm-d prefix scorer but Dynamo expresses it as a closed-form per-worker decision rather than a scorer in a pipeline. Dynamo's defining liability is its depth of NVIDIA coupling: NIXL as the transfer fabric, AIConfigurator as the profiler, Grove for topology-aware gang scheduling, and the DGDR zero-config CRD that automates profiling sweeps — all of these are useful precisely because they integrate tightly with NVIDIA's broader software ecosystem, and none of them have portable alternatives.

llm-d is the CNCF Sandbox reference implementation (Red Hat, Google Cloud, IBM Research, CoreWeave, NVIDIA as founders) of the Kubernetes Gateway API Inference Extension (GAIE). Its architecture is deliberately un-monolithic: intelligence lives in the Endpoint Picker (EPP) that proxies consult via `ext-proc`; the data plane is delegated entirely to industry-standard Envoy/kgateway/agentgateway; all engine-side changes are upstreamed into vLLM rather than forked. The EPP runs a composable Filter→Score→Pick pipeline whose plugins include `precise-prefix-cache-scorer` (backed by a Go KV cache indexer consuming vLLM/SGLang ZMQ events), `latency-scorer` (XGBoost predictor sidecar), `slo-headroom-tier-filter`, and a `disagg-profile-handler` that runs two scheduling sub-pipelines per request. This composition is the technical differentiator: llm-d can add a new routing dimension (say, LoRA affinity or VLM encoder cache awareness) by writing a scorer plugin without touching the proxy, the engine, or the P/D sidecar.

AIBrix originated at ByteDance and is the most CRD-heavy of the three, with eight orchestration CRDs covering almost every aspect of serving: KVCache pool, ModelAdapter (LoRA registry), PodAutoscaler, PodSet, RoleSet, StormService (P/D disaggregated rollouts), RayClusterFleet, and RayClusterReplicaSet. The native APA autoscaler adds a fluctuation-tolerance band to the standard utilization-based formula to suppress scale oscillation. The Virtual Token Counter scoring implements per-user fairness under multi-tenant load. The GPU Optimizer runs an offline profiler to recommend GPU count and type on heterogeneous pools. The L1+L2 KV offload framework is a Python library the engine imports directly, distinct from Dynamo's KVBM (fully Rust, managed by the orchestrator) and llm-d's tiered prefix cache (delegated to upstream vLLM tiering) — it is the most general KV offload abstraction of the three and the one that predated NIXL's broad availability.

### Edge and single-machine engines: llama.cpp, MLX, mistral.rs

These engines sacrifice cluster-class throughput for reach, simplicity, and hardware portability. They are the right choice when the GPU budget is a consumer card, a laptop, or a mobile SoC, not a data center rack.

llama.cpp is the canonical portable engine: a single C/C++ codebase with a dlopen backend registry that runs GGUF weights on every accelerator from an Apple M4 laptop to an IBM zSystems mainframe. Its defining properties are mmap'd weights (the model's data region is `mmap`'d from disk; cold start is bounded by storage bandwidth and the kernel page cache, not by heap allocations), the GGUF self-describing format (tokenizer, chat template, RoPE configuration, quantization metadata, and weights in one file), and the broadest backend matrix in OSS: CUDA/ROCm/MUSA, Metal, Vulkan, SYCL, CANN (Ascend), OpenCL, Hexagon HVX/HMX (Snapdragon NPU), OpenVINO, zDNN (IBM Z), WebGPU, and ggml-rpc for networked tensor serving. The deliberate absence of paged attention and tensor parallelism is a design choice for portability and simplicity, not a gap to fill; `llama-server` implements continuous batching, prefix caching, speculative decoding, and multimodal inputs on top of the contiguous per-sequence KV layout.

MLX is Apple's answer: a lazy, unified-memory array framework for Apple Silicon that turns the M-series unified LPDDR5X into a peer inference substrate. An `mlx::array` lives in Metal-shared memory; CPU and GPU ops read the same buffer with no copy, only a Metal fence. The lazy execution model — all ops return deferred `Primitive` nodes, computation fires on `mx.eval()` — enables the scheduler to fuse and reorder kernels before dispatch. The JACCL backend (macOS 26.3+) provides Thunderbolt RDMA between Mac Studios, making multi-Mac distributed inference a viable high-end researcher configuration. The CUDA backend (Linux, late 2025) means MLX models can be trained or served on NVIDIA GPUs without rewriting in PyTorch.

mistral.rs is the Rust-native all-in-one: format-agnostic loading across HF safetensors, GGUF, and its own UQFF (Universal Quantized File Format), In-Situ Quantization (ISQ) at load time that produces a UQFF file for reproducibility, per-layer topology assignment (different quants and devices per layer from a TOML config), and a built-in agentic tool loop with MCP client and web search. It rides on Hugging Face's `candle` Rust tensor library for base ops, adding its own paged-attention CUDA/Metal kernels, FlashInfer decode integration, and MLA decode path. The `mistralrs tune` / `doctor` / `from-config` UX is unique among the engines in this survey: no other engine tries to autodetect the best quant for the host hardware and emit a reproducible config.

### Others: TGI, lmdeploy, MAX, ktransformers

**TGI** (Hugging Face) is archived as of March 2026; its final release is v3.3.7. Hugging Face directs users to vLLM and SGLang. It is included here for historical context: TGI's Rust-router-plus-Python-workers pattern influenced mistral.rs and the `sgl-model-gateway` design, and several features it pioneered — continuous batching, paged KV, structured outputs — are now universal. TGI's `router` crate, which handled request batching, SSE streaming, and health checking in Rust before routing to Python model worker processes over gRPC, is the architectural precedent for the Rust-core-plus-Python-worker split seen in both mistral.rs and SGLang's `sgl-model-gateway`. Operators maintaining TGI-based deployments should treat v3.3.7 as the final stable point and plan a migration to vLLM or SGLang; the OpenAI-compatible API surface is identical and most TGI configuration flags have direct equivalents.

**lmdeploy** (Shanghai AI Lab) is the engine of record for the InternLM and Qwen model families. Its TurboMind engine is a hand-written CUDA engine in the FasterTransformer lineage with day-0 model support, FlashMLA, DeepGEMM-based GEMMs, FP8 MoE, and AWQ as the canonical quantization. For operators deploying InternLM or Qwen at scale, TurboMind's claimed 1.8× vLLM throughput advantage on its target models is the deciding factor. lmdeploy also ships a PyTorch engine path for broader model coverage (Falcon, Baichuan, InternVL, Phi series) where TurboMind kernels are not yet implemented; the two paths share the same HTTP API layer (`lmdeploy serve api_server`) and the same AWQ quantization toolchain (`lmdeploy lite auto_awq`). The `lmdeploy chat`, `lmdeploy serve`, and `lmdeploy convert` commands make it one of the simpler deployment paths for the InternLM ecosystem.

**Modular MAX** is an inference engine built on the Mojo language and the Modular MLIR compiler. Its claim is first-class kernel DSL access and NVIDIA H100/B200 plus AMD MI300X/MI355X support; the Apache 2.0 release of the MAX Python API happened in late 2025. As of May 2026 it is an active contender on Blackwell but has not yet reached vLLM parity in model coverage or production deployment evidence. MAX's architectural claim is that its `MAX Graph` compiler — which takes a Mojo-level kernel description and emits ISA-specific code for H100, B200, MI300X, MI355X, and CPU without a separate CUDA kernel rewrite — will reduce the maintenance overhead of supporting multiple hardware generations; whether this claim survives contact with production-scale tuning requirements remains to be seen.

**ktransformers** is the canonical CPU+GPU heterogeneous MoE serving engine: dense components on GPU, cold routed experts on CPU using Intel AMX/AVX-512, with a CUDA-stream overlap that runs the CPU MoE expert pass concurrently with the next GPU layer. It is not a fleet-grade serving system; from v0.6 it is shipped as a kernel library that SGLang calls for the expert pass, with SGLang handling request scheduling and KV cache management. It is the right choice when the target model — DeepSeek-V3, Kimi-K2, MiniMax-M2, GLM-5, Qwen3-MoE — does not fit in the available VRAM. The heterogeneous architecture assumes a specific memory topology: GPU VRAM for attention KV, the non-expert MoE components (embedding, norms, MLA latent projection), and the currently hot experts; DRAM for cold experts (up to 560 GB on high-end workstations); NVMe SSD as a third tier. The CUDA stream overlap is the key performance insight: the GPU issues the next FFN/attention layer while the CPU AMX path computes the current MoE expert pass asynchronously, hiding the AMX latency behind GPU memory-bound operations.

---

## Scheduler and KV cache architecture: a closer look at the cluster engines

The three cluster engines have converged on the same set of features but implemented them with meaningfully different internal architectures. Understanding those differences matters when diagnosing production issues, choosing tuning knobs, or evaluating whether an engine fits a new workload.

**Scheduler process model.**
vLLM V1 runs the scheduler in a separate `EngineCore` process that communicates with the front-end over
ZMQ + msgpack. The front-end never blocks on model execution; the EngineCore can be scaled horizontally
by spawning one per DP rank, with a lightweight `Coordinator` ensuring cross-rank `has_unfinished`
consensus for dummy-batch alignment.
Each DP/PP shard has a dedicated Scheduler process that dispatches work to TpModelWorker processes via ZMQ PUSH/PULL,
achieving CPU/GPU overlap via the `event_loop_overlap` design:
the scheduler dispatches GPU work asynchronously on a dedicated `schedule_stream` and prepares the next
batch while the current one executes.
TRT-LLM's `PyExecutor` runs an overlap scheduler (`docs/source/features/overlap-scheduler.md`) that
launches GPU work for step n+1 while CPU finalizes step n's responses — the same idea as SGLang's
overlap loop, described as inspired by the NanoFlow paper.

**KV cache eviction and prefix sharing.**
vLLM's `BlockPool` evicts by LRU over the entire pool: free blocks that are also prefix-cache residents
sit at the tail of a doubly-linked list and are evicted only when no free non-cached blocks remain.
Each block carries a `BlockHash = hash(parent_hash, block_tokens, extra_keys)` — a chained SHA-256
by default, with `xxhash`, `sha256_cbor`, and `xxhash_cbor` alternatives; the `cbor` variants are
deterministic across Python versions, required for cross-process cache sharing (LMCache).
SGLang's `RadixCache` operates at the node level: a node is evicted when its `lock_ref` is zero and no
live-device children exist; the `evictable_leaves` set is maintained incrementally. Eviction of a leaf
walks up the tree to re-expose the parent as a candidate, making the amortized cost of eviction
$O(\text{depth})$ rather than $O(\text{pool-size})$.
TRT-LLM's `KVCacheBlock` tree evicts with a pluggable `EvictionPolicy` that includes a
`KvCacheRetentionConfig` for per-request retention hints — an operator-visible knob absent from the
other two engines.

**Speculative decoding integration.**
All three engines now treat speculative decode as a variant of the standard forward pass rather than a
separate pipeline.
In vLLM V1, the scheduler's token budget includes `num_lookahead_tokens` as an explicit allocation;
draft tokens occupy the same KV cache slots as verified tokens until the rejection sampler decides which
to keep. The rejection sampler is part of the sampling step, not a separate inference pass; EAGLE,
Medusa, MTP, and draft-model methods share the same loop.
SGLang's speculative workers (`eagle_worker_v2.py`, `dflash_worker.py`, `standalone_worker_v2.py`) each
implement a `draft` + `verify` pair; the "v2" variants support `event_loop_overlap` and can pipeline
draft generation behind the previous target forward.
TRT-LLM has the richest speculative decoding zoo: two-engine mode (draft and target are separate engines
with CUDA-graph-compatible KV swap), one-engine mode (MTP/EAGLE3 baked into the target via
`modeling_speculative.py` and a `MTPHiddenStatesManager`), suffix automaton enhancement, dynamic EAGLE3
tree, and the MTP "relaxed acceptance" extension that accepts a draft token if it appears in the
target's top-N rather than requiring exact match.

**Multi-GPU parallelism axes.**
vLLM and SGLang share a similar TP/PP/DP/EP abstraction but implement context parallelism differently:
vLLM calls the axes PCP (Prefill Context Parallelism) and DCP (Decode Context Parallelism), exposed as
separate config knobs; SGLang uses `attn_cp_size` and integrates CP with the `DP-attention + EP-MoE`
pattern that is the canonical DeepSeek/Kimi serving topology.
TRT-LLM adds Helix parallelism (`CpType.HELIX`) which reconfigures the same $N$ GPUs between the
attention phase ($\text{KVP} \times \text{TP}_A$, sequence-sharding the KV cache with log-sum-exp
rescaling across ranks for exact online-softmax reduction) and the FFN phase ($\text{TP}_F \times
\text{EP}$). This is architecturally distinct from ring-attention or Ulysses and achieves higher
utilization on GB200 NVL72 by exploiting the NVLink bandwidth asymmetry between intra-rack and
inter-rack hops.

**Quantization integration depth.** FP8 and NVFP4 are table stakes for all three engines on Blackwell. The interesting differences are at lower bit widths and in the MoE path. vLLM's `compressed-tensors` integration with NeuralMagic's LLM Compressor is the deepest off-the-shelf pipeline for W4A4, W4A8, and sparse quantization: the same checkpoint format works across vLLM, SGLang (via `compressed-tensors` config), and TRT-LLM (via ModelOpt export). TRT-LLM's `NVFP4_AWQ` and mixed-precision MoE quantization types (`W4A8_NVFP4_FP8`, `W4A8_MXFP4_FP8`, `W4A8_MXFP4_MXFP8`) are specific to the Blackwell architecture's `fp4GemmPlugin` and have no equivalent in vLLM or SGLang at present. SGLang's `QoQ` (Quattuor-Octo-Quant, W4A8 with FP8 activations) and its `quark_int4fp8_moe` path cover AMD ROCm quant targets that TRT-LLM does not address.

---

## Model architecture and hardware coverage

Not all engines support all model classes. Engine selection requires verifying that the target model architecture — particularly its attention variant and its MoE structure — is in the engine's supported path.

### Attention variant coverage

The dominant attention variants in 2025–2026 frontier models are Multi-Head Latent Attention (MLA, DeepSeek V2/V3/Kimi-K2), Grouped Query Attention (GQA, Llama-4/Qwen3/Gemma 3), Sliding Window Attention (SWA, Mistral/Gemma hybrid), and standard MHA. MLA is the most operationally significant variant because its KV representation is a latent vector of rank $d_c \ll n_h \cdot d_h$ that must be projected back to per-head KV at decode time; caching the latent versus caching the projected KV is a throughput–memory trade-off that each engine exposes differently.

vLLM supports MLA via FlashMLA (NVIDIA Hopper/Blackwell) and CutlassMLA backends, with `MLA_KV_LAYOUT` controlling whether the latent ($c^{KV}$) or the projected ($k^C$, $v^C$) tensors are stored in the paged KV pool; the `absorb_kv_lora` option fuses the up-projection into the attention kernel. SGLang has equivalent FlashMLA and CutlassMLA paths plus a TRT-LLM-MLA backend; its RadixAttention operates on the latent representation natively, reducing KV block size by $\approx 2\times$ for DeepSeek-V3 relative to storing projected heads. TRT-LLM caches the latent vector in a separate `mla_latent` pool and projects on the fly via its XQA kernel; the MicroBatchScheduler's per-window-size pool sizing must account for this. llama.cpp uses a handwritten MLA attention path in its Metal and CUDA fattn kernels; its KV layout is contiguous per-sequence and always stores the projected form, which is correct but larger. MLX and mistral.rs implement MLA decode paths; ktransformers delegates entirely to SGLang for the attention pass when used as a kernel library.

SWA (Sliding Window Attention) requires the engine to handle per-layer window sizes and the KV cache eviction boundary at $w$ tokens per head. vLLM's hybrid attention support (mixing SWA layers and full-attention layers in the same model) is the most tested path for Gemma 3 and Mistral 3. TRT-LLM has per-window-size KV pool allocation as a first-class concern in its paged KV V2 design. SGLang implements SWA via the `window_size` RadixCache variant and exposes it in its chunked-prefill scheduler. llama.cpp supports SWA via its `RotatingKVCache` design (ring-buffer semantics with a configurable window); MLX's `RotatingKVCache` follows the same pattern.

### MoE architecture coverage

Expert Parallel (EP) and the associated EPLB are the dominant multi-GPU scaling axes for MoE models in 2025–2026. The relevant model variants are fine-grained MoE (DeepSeek-V3: 256 routed experts, top-8 selection; Kimi-K2: 384 experts, top-8; MiniMax-M2: similar scale), medium-scale MoE (Qwen3-MoE: 128 experts, top-8; Mixtral: 8 experts, top-2), and hybrid MoE-dense (Gemma 3 MoE, Llama-4-Scout/Maverick).

vLLM uses DeepEP's `Buffer` class for fine-grained MoE all-to-all (low-latency mode with 20 SM reservation or high-throughput mode), with `DeepEPDispatcher` as the scheduler-side abstraction. SGLang uses the same DeepEP path plus a `Triton` fallback for AMD and its own `deep_ep_norm_norm` fused dispatch kernel; its `DeepGEMM` GEMM path (JIT-compiled grouped GEMM) is distinct from TRT-LLM's DeepGEMM integration. TRT-LLM's `ConfigurableMoE` backend selects between `DeepEPPlugin` (for fine-grained MoE with NVSHMEM NVLink one-sided), `CutlassMoEPlugin` (medium-scale), and `DWDPPlugin` (expert weight prefetch for Dense-While-Dispatching-Prefetch); this three-way switch is TRT-LLM-specific and has no equivalent in vLLM or SGLang. ktransformers' AMX/AVX-512 CPU path is relevant only for fine-grained MoE on CPU DRAM and is the only path that treats CPU DRAM as primary rather than overflow storage.

### Hardware compatibility matrix

The engines span a wide hardware support surface, but the depth of support varies significantly. "Supported" in most engine documentation means "can run the model forward pass"; it does not mean "achieves optimal throughput" or "has CUDA-graph-compatible kernels." The most important hardware-specific caveats are:

- **NVIDIA Hopper (H100/H200)**: vLLM, SGLang, and TRT-LLM all reach full performance on Hopper.
  FP8 (per-tensor and rowwise) and FlashInfer BSR attention are Hopper-compatible.
  NVFP4 and FP8 rowwise block-scale quantization are Blackwell-only and are not available on H100.
- **NVIDIA Blackwell (B200/GB200 NVL72)**: all three cluster engines support Blackwell, but only TRT-LLM
  exposes Helix parallelism, NVLS multicast, and DWDP — features that require GB200 NVL72's NVLink
  topology. TRT-LLM-Gen (CUTLASS-DSL) kernels are Blackwell-optimized; FlashInfer FP4 paged decode
  is the Blackwell path in vLLM and SGLang.
- **AMD ROCm (MI300X/MI350X)**: vLLM is the primary tested path, with AITER as its ROCm attention
  kernel library and `rocm_flash_attn` as the FA backend. SGLang has growing ROCm support via AITER
  and Wave backends. TRT-LLM lacks ROCm support entirely.
- **Intel Gaudi (Gaudi 3)**: vLLM has a Gaudi HPU backend (`vllm/platforms/hpu.py`) maintained by
  Intel; SGLang has an Intel XPU path. Neither TRT-LLM nor lmdeploy support Gaudi.
- **Google TPU (v5e/v5p)**: vLLM has a PyTorch/XLA TPU backend; SGLang has a JAX TPU path with
  disaggregated P/D via TpuCommunicator; llm-d supports TPU disagg routing as of v0.5.
- **Apple Silicon (M-series)**: MLX is the primary path with Metal-native kernels and unified memory.
  llama.cpp's Metal backend (fattn MSL shaders) is the fallback for GGUF models. mistral.rs has
  Metal paged-attention kernels. TRT-LLM, vLLM, and SGLang do not support Apple Silicon.

---

## The reach–throughput–flexibility axes

Three axes capture the most important design trade-offs in the landscape.

**Throughput ceiling** measures whether the engine can reach cluster-class tokens-per-second or is bounded by a single machine's memory bandwidth. **Hardware reach** measures whether the engine runs on arbitrary accelerators or requires an NVIDIA GPU. **Deployment model** describes where the component sits in the serving stack.

```
Engine            Throughput ceiling   Hardware reach          Deployment model
─────────────────────────────────────────────────────────────────────────────
vLLM V1           Cluster-class        Multi-vendor (NVIDIA    Single-node or fleet
                                        primary, AMD/Gaudi/     engine process
                                        TPU/CPU)
SGLang            Cluster-class        NVIDIA primary, AMD     Single-node or fleet
                                        growing, JAX TPU        engine process
TRT-LLM           Cluster-class        NVIDIA-only             Single-node or fleet
                                                                engine process
LMCache           Cluster-class        Vendor-agnostic         Library / KV connector
                  (as shared cache)    (runs on host CPU)
Mooncake          Cluster-class        NVIDIA primary          Library + daemon
                  (transfer engine)    (Transport Engine       (RDMA agent + Store
                                        is CPU-side)            master)
NVIDIA Dynamo     Orchestration-tier   NVIDIA-aligned (NIXL)   Kubernetes orchestrator
llm-d             Orchestration-tier   Engine-agnostic         Kubernetes orchestrator
                                                                (EPP + sidecar)
AIBrix            Orchestration-tier   Engine-agnostic         Kubernetes orchestrator
                                                                (ext-proc + controllers)
llama.cpp         Single-machine       Truly portable          Single-binary /
                                        (15+ backends)          library
MLX               Single-machine       Apple Silicon primary,  Python library /
                                        CUDA secondary          single-process server
mistral.rs        Single-machine       NVIDIA + Metal + CPU    Single binary /
                                        (candle-based)          PyO3 library
ktransformers     Single-machine       Intel AMX + NVIDIA +    Kernel library (inside
                  (high token/s on     AMD + Ascend            SGLang or standalone
                   huge sparse MoE)                             Python server)
lmdeploy          Cluster-class        NVIDIA primary, ROCm    Single-node or fleet
                  (InternLM/Qwen)      + Ascend + others        engine process
MAX               Cluster-class        NVIDIA + AMD            Single-node or fleet
                                                                engine process
```

The axes are not fully independent. llama.cpp's portability is inseparable from its single-machine ceiling — the absence of paged attention and tensor parallelism is a design choice for simplicity, not an oversight. TRT-LLM's throughput ceiling is inseparable from its NVIDIA-only hardware requirement — the CUTLASS-DSL, NVLink one-sided all-to-all, and NVLS multicast are NVIDIA intellectual property with no portable equivalent. vLLM's breadth of hardware reach comes with genuine architectural overhead: the plugin-based attention backend system, the platform abstraction layer, and the vendor-specific code paths that must all be maintained.

---

## Major feature comparison

<table>
<thead>
<tr>
<th>Engine</th>
<th>License</th>
<th>Primary language</th>
<th>Scheduler</th>
<th>KV cache</th>
<th>Attention backends</th>
<th>Quantization formats</th>
<th>LoRA</th>
<th>Spec decode</th>
<th>Disagg P/D</th>
<th>Multi-GPU parallelism</th>
<th>K8s operator</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>vLLM V1</strong></td>
<td>Apache 2.0</td>
<td>Python + C++/CUDA</td>
<td>Token-budget, chunked prefill, FCFS/Priority</td>
<td>Paged blocks + hash-chained prefix (LRU), CPU offload tier, hybrid multi-type</td>
<td>FA-2/3/4, FlashInfer, FlashMLA, CutlassMLA, Triton, FlexAttn, ROCm AITER, Mamba</td>
<td>FP8, MXFP4, NVFP4, AWQ/Marlin, GPTQ/Marlin, compressed-tensors, ModelOpt, GGUF, bitsandbytes, TorchAO, Quark, online quant</td>
<td>Yes (Punica/S-LoRA BGMV/SGMV, MoE-LoRA)</td>
<td>EAGLE/2/3, MTP, Medusa, ngram, draft-model, DFlash, suffix</td>
<td>Yes (NIXL, Mooncake, LMCache, p2p-NCCL, CPU offload; MultiConnector)</td>
<td>TP, PP, DP, EP+EPLB, PCP/DCP, Elastic EP</td>
<td>Via Dynamo / llm-d / AIBrix operators</td>
</tr>
<tr>
<td><strong>SGLang</strong></td>
<td>Apache 2.0</td>
<td>Python + C++/CUDA + Rust (gateway)</td>
<td>Overlap scheduler (zero-overhead), LPM/DFS-weight cache-aware, chunked prefill, retraction</td>
<td>RadixAttention global tree + HiCache (GPU/CPU/L3), UnifiedRadixTree (2026), SWA/Mamba variants</td>
<td>FlashInfer (non-MLA/MLA), FA3, FA4, FlashMLA, CutlassMLA, TRT-LLM MHA/MLA, NSA, Triton, AITER/Wave (AMD), Ascend, Intel AMX/XPU, DualChunk</td>
<td>FP8, MXFP4, NVFP4, AWQ/Marlin, GPTQ/Marlin, ModelOpt, compressed-tensors, QoQ, W4AFP8, Quark, GGUF</td>
<td>Yes (S-LoRA/Punica chunked SGMV, MoE-LoRA, LoRA drainer/overlap loader)</td>
<td>EAGLE/2/3, multi-layer EAGLE, DFlash, NGRAM, standalone draft-model</td>
<td>Yes (Mooncake, NIXL, Mori, Ascend HIXL; PD-mux SM-group)</td>
<td>TP, PP, DP+EP, MoE-EP (DeepEP), CP (NSA), TBO/SBO, EPLB, Elastic EP</td>
<td>Via sgl-model-gateway Rust router; Dynamo / llm-d / AIBrix backends</td>
</tr>
<tr>
<td><strong>TRT-LLM</strong></td>
<td>Apache 2.0</td>
<td>C++ (runtime) + Python (frontend)</td>
<td>IFB: CapacityScheduler + MicroBatchScheduler, MAX_UTILIZATION / GUARANTEED_NO_EVICT, chunked prefill with reuse-skip</td>
<td>Two-tier paged (GPU+CPU), radix prefix cache, per-window-size pools (SWA/hybrid), KV events for routers, MLA latent cache; KVCacheV2 for Mamba/hybrid</td>
<td>TRT-LLM FMHA/MMHA/XQA (internal cubins), FlashMLA, TRT-LLM-Gen (Blackwell CUTLASS-DSL), FlashInfer, sparse (DSA, RocketKV, BLASST)</td>
<td>FP8 (per-tensor, rowwise, block-scale), NVFP4, MXFP4, AWQ, GPTQ, SmoothQuant, QServe (W4A8), NVFP4-KV, FP8-KV, weight-only INT4/INT8</td>
<td>Yes (LoRA plugin + PeftCacheManager, DoRA, NeMo/HF formats, multi-LoRA per request)</td>
<td>MTP (vanilla + Eagle + relaxed-acceptance), EAGLE-3 (dynamic tree), Medusa, ngram, draft-model, Lookahead, Suffix Automaton, DFlash, PARD</td>
<td>Yes (NIXL default, UCX, Mooncake, MPI; per-layer overlap via contextProgress; heterogeneous TP layout conversion)</td>
<td>TP, PP, DP (Attention-DP), CP (Ring/Helix/Ulysses), EP+EPLB, Wide-EP (DeepEP/NVSHMEM), DWDP</td>
<td>Dynamo integration; Triton Inference Server backend (`tensorrtllm_backend`); Ray executor</td>
</tr>
<tr>
<td><strong>LMCache</strong></td>
<td>Apache 2.0</td>
<td>Python + C++/CUDA (csrc) + Rust (slivers)</td>
<td>N/A (library; integrates into engine scheduler via KVConnector V1)</td>
<td>Multi-tier: DRAM → NVMe → GDS → P2P mesh → remote (Redis/S3/Mooncake/InfiniStore/EIC/S3/SageMaker); LRU/LFU/FIFO/MRU eviction per tier</td>
<td>N/A (consumes engine backends)</td>
<td>CacheGen arithmetic coding, KIVI 2-bit, naive pass-through</td>
<td>N/A</td>
<td>N/A</td>
<td>Yes (PDBackend + transfer_channel; vLLM KVConnector V1 scheduler/worker)</td>
<td>MLA-aware (save_only_first_rank); layerwise async saves</td>
<td>N/A (library)</td>
</tr>
<tr>
<td><strong>Mooncake</strong></td>
<td>Apache 2.0</td>
<td>C++ core + pybind11 + Rust + Go</td>
<td>N/A (transfer engine + object store)</td>
<td>Distributed object store (DRAM/VRAM contributed by clients) + SSD offload + 3FS/etcd HA; LRU/FIFO eviction at master; soft/hard pin</td>
<td>N/A</td>
<td>N/A (opaque bytes; compression is tenant concern)</td>
<td>N/A</td>
<td>N/A</td>
<td>Yes (MooncakeConnector V1 for vLLM; SGLang HiCache L3; TRT-LLM TE integration)</td>
<td>Multi-NIC RDMA striping; Mooncake-EP for MoE all-to-all; Mooncake-PG PyTorch process group</td>
<td>N/A (library + daemons)</td>
</tr>
<tr>
<td><strong>NVIDIA Dynamo</strong></td>
<td>Apache 2.0</td>
<td>Rust + Python + Go (operator)</td>
<td>KV-aware cost routing: `w·prefill_blocks + decode_blocks`; Planner (SLA throughput-based + load-based)</td>
<td>KVBM: G1 GPU / G2 host / G3 SSD / G4 remote (S3/Azure/WEKA/Dell); sequence-hash dedup across tiers; NIXL transport</td>
<td>Delegates to engine</td>
<td>Delegates to engine</td>
<td>Yes (via engine)</td>
<td>Yes (via engine)</td>
<td>Yes (PrefillRouter; NIXL P→D transfer; heterogeneous TP layout)</td>
<td>Orchestrates TP/PP/EP across vLLM/SGLang/TRT-LLM; Grove topology gang scheduling</td>
<td>Yes (DGD/DCD/DGDR CRDs; GAIE EPP plugin; Kubernetes-native discovery)</td>
</tr>
<tr>
<td><strong>llm-d</strong></td>
<td>Apache 2.0</td>
<td>Go + Helm/Kustomize</td>
<td>EPP: Filter→Score→Pick plugin chain; `precise-prefix-cache-scorer` (event-driven) + heuristic; disagg-profile-handler</td>
<td>Tiered prefix cache via vLLM CPU offload + LMCache/Mooncake/KVBM connectors; llmd_fs_backend; KV indexer (Go library)</td>
<td>Delegates to engine</td>
<td>Delegates to engine</td>
<td>Yes (cache-aware LoRA routing since v0.5)</td>
<td>Yes (via engine)</td>
<td>Yes (EPP disagg-profile-handler + PD routing sidecar; NIXL via vLLM connector; SGLang bootstrap; TPU disagg)</td>
<td>Wide-EP via LeaderWorkerSet; Intel XPU; Google TPU</td>
<td>Yes (reference GIE implementation; InferencePool CRD; WVA; KEDA scale-to-zero)</td>
</tr>
<tr>
<td><strong>AIBrix</strong></td>
<td>Apache 2.0</td>
<td>Go + Python (KV offload)</td>
<td>Routing algorithms: random, least-* family, prefix-cache, prefix-cache-Preble, VTC fairness, PD disaggregation, SLO</td>
<td>L1 DRAM (LRU/FIFO/S3FIFO) + L2 distributed (InfiniStore/HPKV/Vineyard/RocksDB/shfs); CUDA kernels for G↔C; meta service for cross-instance placement</td>
<td>Delegates to engine</td>
<td>Delegates to engine</td>
<td>Yes (ModelAdapter CRD + controller; JIT load/unload per pod)</td>
<td>Yes (via engine)</td>
<td>Yes (pd_disaggregation.go; StormService RoleSet labels; vLLM/SGLang/TRT-LLM e2e tests)</td>
<td>RayClusterFleet/ReplicaSet for multi-node</td>
<td>Yes (8 CRDs; Envoy ext-proc; APA autoscaler; GPU Optimizer; console UI)</td>
</tr>
<tr>
<td><strong>llama.cpp</strong></td>
<td>MIT</td>
<td>C/C++ (ggml)</td>
<td>Continuous batching in llama-server; FCFS; no token-budget prefill splitting</td>
<td>Contiguous per-sequence KV (no paging); hash-based prefix cache in llama-server</td>
<td>Handwritten FlashAttention variants (CUDA fattn, Metal MSL, Vulkan); MMA-based per ISA</td>
<td>Q2..Q8_K K-quants, IQ1..IQ4 (imatrix), TQ1/2 (BitNet 1.58-bit), MXFP4, NVFP4, Q1_0, BF16/F16</td>
<td>No</td>
<td>Draft model, n-gram cache, n-gram map; mixed modes in llama-server</td>
<td>No</td>
<td>Layer-pipeline multi-GPU (--split-mode layer/row); ggml-rpc networked backend; no TP collectives</td>
<td>No</td>
</tr>
<tr>
<td><strong>MLX / mlx-lm</strong></td>
<td>MIT</td>
<td>C++ + Metal MSL + Python</td>
<td>Single-process BatchGenerator; no fleet scheduler</td>
<td>KVCache, RotatingKVCache, QuantizedKVCache (4/8-bit), LRUPromptCache; Mamba/hybrid variants; unified memory (no H↔D copy)</td>
<td>Metal Steel SDPA (full / vector / 2-pass variants); M5 NAX kernels; CUDA cuBLAS/CUTLASS (Linux)</td>
<td>MLX affine block-quant (2/3/4/6/8-bit, gs 32–128), AWQ, GPTQ, DWQ, MXFP8, NVFP4, QQMM</td>
<td>Yes (mlx-lm LoRA/QLoRA fine-tuning)</td>
<td>Draft model (separate model)</td>
<td>No</td>
<td>mlx.distributed: JACCL (Thunderbolt RDMA, macOS 26.3+), ring TCP, NCCL, MPI; sharded load for multi-Mac TP</td>
<td>No</td>
</tr>
<tr>
<td><strong>mistral.rs</strong></td>
<td>MIT</td>
<td>Rust + C++/CUDA kernels</td>
<td>default_scheduler; continuous batching; no fleet scheduler</td>
<td>Paged attention (CUDA + Metal); prefix caching (block-level); rotating / hybrid KV cache</td>
<td>FA2/FA3 (candle-flash-attn), FlashInfer decode, custom paged-attn CUDA V1/V2, MLA decode, Metal paged-attn</td>
<td>GGUF (Q*_K, IQ*), GPTQ, AWQ, HQQ, AFQ, FP8 (scalar/blockwise/per-tensor/per-vector), MXFP4, BNB, F8Q8; ISQ + UQFF</td>
<td>Yes (per-layer adapter via candle)</td>
<td>Draft model</td>
<td>No</td>
<td>NCCL (CUDA); TCP ring (cross-machine TP without NCCL)</td>
<td>No</td>
</tr>
<tr>
<td><strong>ktransformers</strong></td>
<td>Apache 2.0</td>
<td>C++ + Python + CUDA kernels</td>
<td>Delegates to SGLang (v0.6+) or minimal standalone</td>
<td>3-tier (GPU/CPU/disk) prefix cache (June 2025)</td>
<td>Delegates to SGLang</td>
<td>AMXINT4/INT8, BF16, FP8, FP8-per-channel, MXFP4, GPTQ-INT4 (Marlin), MOE_INT4/INT8, llamafile GGUF, K2-MoE block-FP8</td>
<td>Via SGLang</td>
<td>Via SGLang</td>
<td>No</td>
<td>CPU+GPU heterogeneous expert split; moe-tp / mla-tp for multi-socket/GPU</td>
<td>No</td>
</tr>
<tr>
<td><strong>lmdeploy</strong></td>
<td>Apache 2.0</td>
<td>C++ (TurboMind) + Python</td>
<td>Dynamic split-and-fuse (chunked prefill); blocked KV; continuous batching</td>
<td>Blocked KV cache + KV-cache quant; prefix cache</td>
<td>FlashMLA, DeepGEMM GEMM, FP8 MoE kernels (TurboMind); HF transformers (pytorch engine)</td>
<td>AWQ (canonical), FP8, MXFP4, KV-cache quant</td>
<td>Yes</td>
<td>Draft model (TurboMind)</td>
<td>No (native); KV reuse via prefix cache</td>
<td>TP + PP</td>
<td>No</td>
</tr>
<tr>
<td><strong>MAX</strong></td>
<td>Apache 2.0 (Modular Community License for brand)</td>
<td>Mojo + Python API</td>
<td>OpenAI-compatible server; batching via MAX graph compiler</td>
<td>Managed by MAX engine</td>
<td>Mojo/MLIR-compiled kernels; B200/H200/H100 and MI355X/MI300X targets</td>
<td>FP8, INT4, INT8 (kernel-by-kernel)</td>
<td>Via engine</td>
<td>Via engine</td>
<td>Via `max serve`</td>
<td>NVIDIA and AMD (data center); CPU secondary</td>
<td>No</td>
</tr>
</tbody>
</table>

---

## Decision guide

The following heuristics apply to new projects; brownfield deployments must weigh migration cost against the benefit of switching.

**Multi-tenant cloud serving on an NVIDIA GPU fleet with diverse model and workload mix.** Start with vLLM. Its V1 token-budget scheduler, hash-chained prefix cache, EPLB for MoE, and the full connector zoo for disaggregated P/D give the widest coverage with the least bespoke work. Operators hitting vLLM throughput limits on specific frontier models — DeepSeek V4, Kimi K2, Qwen3-MoE — should benchmark SGLang's overlap scheduler on the same hardware before concluding vLLM is insufficient; the gap is often 20–40% on TTFT-sensitive workloads and larger on high-batch-size decode.

Why not TRT-LLM? Because the AOT build pipeline and NVIDIA-only requirement break the "diverse model mix" constraint; every new model architecture requires a rebuild, and any non-NVIDIA hardware in the fleet is unservable. Why not SGLang first? SGLang's deep focus on frontier-model throughput is an advantage on a homogeneous fleet of DeepSeek/Kimi traffic, but vLLM's wider model-architecture coverage (encoder-decoder, Mamba-hybrid, Vision-Language, cross-attention models) and larger ecosystem of quantization formats give it the lower integration risk on a mixed workload.

**Maximum throughput on a fixed NVIDIA hardware configuration with full control of the software stack.** Start with TRT-LLM. The AOT TRT engine plan (or `torch.compile` piecewise path) plus proprietary CUTLASS-DSL kernels, NVLS multicast, Helix parallelism, and the DWDP expert prefetch achieve throughput ceilings that vLLM and SGLang cannot match on the same hardware. The cost is NVIDIA-only support, a two-headed Python+TRT codebase, and ModelOpt as a calibration dependency.

Why not vLLM or SGLang? Both rely on runtime autotuning (`torch.compile` + Inductor, or FlashInfer JIT) for kernel selection; they cannot perform the tactic search over hundreds of GEMM implementations that TRT's offline plan selection does. The throughput advantage of TRT-LLM on GB200 NVL72 — primarily from Helix parallelism and DWDP — has no counterpart in either general-purpose engine. The relevant question is whether the throughput delta (empirically 15–35% on dense models, 40–60% on large MoE) justifies the operational overhead of AOT rebuilds, which typically take 30–90 minutes per model per hardware SKU per software version.

**Disaggregated prefill/decode architecture.** Three paths are production-ready. (a) vLLM with its NIXL connector gives the most turnkey experience for pure P/D disagg with RDMA KV transfer; add LMCache in front for multi-tier hierarchical reuse across instances. (b) SGLang with Mooncake as the disagg transport is the highest-performance path on NVIDIA for the frontier models SGLang targets natively. (c) Dynamo provides disaggregated P/D *plus* cluster-level autoscaling, KV block tiering to remote storage, and a GAIE-conformant Kubernetes operator — at the cost of full Dynamo adoption.

The choice between paths (a), (b), and (c) reduces to how much cluster-wide coordination is required. Path (a) is engine-first: the engine manages its own local KV pool and LMCache sits between instances; cluster state is eventually consistent through cache evictions and misses. Path (b) is transfer-engine-first: Mooncake's `MasterService` holds authoritative placement metadata out-of-band; the engine pulls blocks explicitly. Path (c) is orchestrator-first: Dynamo's global `ConcurrentRadixTree` and KVBM make placement decisions centrally, which enables the tightest SLA guarantees but also creates the tightest operational coupling to the Dynamo software release cycle.

**Kubernetes-native request routing and autoscaling without vendor lock-in.** llm-d fits deployments that are vLLM-primary and want to stay as close to upstream Kubernetes primitives as possible; it is the CNCF Sandbox reference implementation of the GAIE EPP protocol and deliberately avoids reinventing the data plane. AIBrix fits deployments needing a fuller operator surface — dedicated LoRA adapter lifecycle management, the APA autoscaler, VTC multi-tenant fairness, and heterogeneous GPU pool optimization — or those coming from a ByteDance-adjacent stack.

Why not Dynamo for this scenario? Dynamo fits deployments requiring deep NVIDIA integration (NIXL, AIConfigurator, Grove gang scheduling). Vendor-neutral Kubernetes-native operation is harder under Dynamo's Rust-core KV router and NIXL transport assumption, which complicate running on AMD or Intel Gaudi fleets. llm-d's principle of zero data-plane reinvention — the data plane is Envoy, the engine is unmodified vLLM/SGLang — enables replacing either component without changing the other; this is structurally impossible in Dynamo's tighter integration model.

**On-device, edge, or consumer hardware where Python is not available.** llama.cpp is the default: single static binary, GGUF weights, mmap'd loading, continuous batching in `llama-server`, and the widest accelerator matrix in OSS. For Apple Silicon researchers or teams deploying multi-Mac inference clusters, MLX's unified-memory model and JACCL Thunderbolt RDMA make it the better choice. For teams wanting a Rust binary with format-agnostic loading, In-Situ Quantization, and built-in agentic tool calling, mistral.rs fills the gap.

The key differentiation between llama.cpp, MLX, and mistral.rs for edge is the hardware target and the format story. llama.cpp's GGUF format is the interchange standard — every model lab releases GGUF weights; quantization tooling (llama.cpp's `llama-quantize`, `imatrix`) is the most mature for K-quants and IQ-quants on arbitrary hardware. MLX's format is Apple-proprietary (though it can load GGUF and safetensors); its strength is unified memory, not portability. mistral.rs's ISQ + UQFF pipeline is the most developer-friendly for teams that want a reproducible quantization artifact (`mistralrs tune` generates a UQFF file tied to the detected hardware) but currently supports fewer hardware backends than llama.cpp's dlopen registry.

**Huge sparse MoE on a limited GPU budget (one or two consumer GPUs plus fat CPU DRAM).** ktransformers with SGLang as the front end. This is the only production-verified path to serving DeepSeek-V3, Kimi-K2, or MiniMax-M2 on a single workstation. The CPU AMX expert pass is the bottleneck, not the GPU, so the GPU budget can be as small as 24 GB.

Why not llama.cpp for this workload? llama.cpp can serve MoE models via its layer-split multi-GPU mode, but it does not implement the CPU AMX/AVX-512 kernel specialization for expert passes that ktransformers provides, and its throughput on 256-expert configurations is materially lower. The practical threshold is approximately 8× the dense token throughput of llama.cpp at the same hardware configuration, demonstrated on DeepSeek-V3 at ktransformers v0.6. The tradeoff is ecosystem depth: ktransformers has narrower model coverage than llama.cpp and requires the SGLang process boundary from v0.6 onward.

**InternLM or Qwen family models where maximum throughput on NVIDIA hardware is the primary metric.** lmdeploy TurboMind. Its hand-written CUDA kernels (FlashMLA, DeepGEMM, FP8 MoE) are tuned specifically for the model families it targets, and its AWQ-canonical quantization pipeline is the most tested on these architectures.

Why not vLLM or SGLang? Both engines support InternLM and Qwen and have FlashMLA backends, but TurboMind's kernel implementations are authored by the same team that wrote the model architectures, giving day-0 support for new model variants (InternLM3, Qwen3-MoE dense blocks) without waiting for upstream kernel review. On benchmark traces, TurboMind's claimed 1.8× vLLM throughput advantage is real for the models in its optimized path, though the gap narrows significantly on models outside that path. For teams outside the InternLM/Qwen target set, the ecosystem coverage argument reverses: vLLM and SGLang have more active contributor coverage for other architectures.

---

## Cross-cutting themes

The most important observation about the 2025–2026 OSS inference landscape is convergence at the algorithm and protocol level concurrent with divergence at the implementation level. Six themes stand out.

**FlashInfer as the common attention kernel abstraction.** The BSR unification [§10/01](../10-engine-core/01-attention-kernels.md) underpins all three cluster engines on NVIDIA. vLLM has FlashInfer as its default on Blackwell; SGLang uses it for non-MLA prefill/decode, MLA decode, and the FlashInfer-backed TRT-LLM-Gen FP4 MoE path; TRT-LLM uses it in its PyTorch-path `flashinfer.py` attention backend. The JIT-compiled inspector-executor design makes FlashInfer CUDA-graph-compatible in a way that per-engine custom kernels are not: per-batch metadata (block tables, query offsets, KV lengths) is a tensor argument, not a launch-shape variable, so CUDA graph replay does not require a re-capture when the batch changes. The practical effect is that attention kernel improvements — FP4 paged decode, variable-length ragged prefill, tree-mask speculative decode, FP8 MLA — land in one project and propagate to all three engines on the same release cycle. The flip side is a shared failure mode: FlashInfer regressions affect all three engines simultaneously, which is why each engine also maintains a fallback backend chain (Triton unified attention in vLLM, FA3/FA4 in SGLang, TRT-LLM-Gen in TRT-LLM).

**NIXL and Mooncake as the common disaggregated transfer fabric.** Splitwise (ISCA 2024) and DistServe (OSDI 2024) established the goodput argument for P/D disaggregation; NIXL and Mooncake are the production implementations of the data plane those arguments require [§04, §08](../80-oss-deep-dives/). vLLM, SGLang, TRT-LLM, Dynamo, and llm-d all speak NIXL. Mooncake's Transfer Engine was registered as a NIXL transport backend in May 2025, so a deployment can mix NIXL-native connectors (intra-cluster NVLink) with Mooncake RDMA (inter-cluster IB/RoCE) without a protocol translation layer. The result is a de-facto standard for KV block transfer that removes the last significant reason for engines to implement their own custom RDMA stacks — the converging idiom being: engine emits a `KVConnectorBase_V1` (or equivalent), the connector mediates with NIXL or Mooncake TE, and the orchestrator (Dynamo KVBM or LMCache StorageManager) handles tiering and eviction above the wire. The remaining diversity is in what sits *above* the wire: KVBM's sequence-hash dedup across four tiers, LMCache's CacheGen compression, and Mooncake Store's immutable-object semantics are not competing standards but complementary functions that stack vertically.

**Token-budget plus chunked-prefill scheduling becoming universal.** vLLM V1's token-budget scheduler — where prefill and decode are not separate phases, only `num_computed_tokens` catching up to `num_tokens_with_spec` — is the clearest formulation of the Sarathi-Serve insight [§10/03](../10-engine-core/03-batching-scheduling.md). SGLang's `chunked_prefill_size` (with optional dynamic chunking that predicts the next chunk size) and TRT-LLM's `MicroBatchScheduler` with chunked context (and the "skip-and-share" scheduler optimization that defers a new request by one iteration so it can share the first new block with an in-flight request carrying the same prefix) implement the same idea. The consequence is that the historic separation between "prefill-heavy" and "decode-heavy" tuning is collapsing: all three engines now mix prefill chunks and decode steps in a single forward pass, and the scheduler knobs (max scheduled tokens, long-prefill token threshold, per-step token budget cap) are the primary throughput–latency levers. The "long_prefill_token_threshold" knob in vLLM, `chunked_prefill_size` in SGLang, and `max_num_tokens` per context phase in TRT-LLM are different names for the same trade-off: capping a single request's contribution per step bounds TTFT variance for all other requests at the cost of throughput for the capped request.

**Radix/prefix-aware caching in every production engine.** PagedAttention provided the block-level memory management [§30/01](../30-kv-cache/); radix-tree or hash-chained prefix caching provides the content-aware reuse layer on top. SGLang's `RadixCache` (a global radix tree with reference-counted nodes, multiple eviction strategies, and split-on-divergence semantics), vLLM's hash-chained `BlockPool` (SHA-256 block hashes chained from parent to child, with a multimap for collision handling), TRT-LLM's `radixBlockTree` (per-BlockKey hash tree with copy-on-write split), and Dynamo's global radix indexer in `ConcurrentRadixTree` (per-worker sticky thread shards for concurrent updates) are four independent implementations of the same logical structure. The convergence is not accidental: system-prompt sharing, multi-turn conversation reuse, and document-grounded RAG are the three highest-value prefix-cache scenarios, and they collectively dominate the traffic at every major inference provider. The interesting differentiator is no longer whether a cache exists but how it handles multimodal inputs (vLLM includes mm-placeholder hashes in the block hash chain; SGLang uses `extra_key` namespacing on `RadixKey` for per-image subtrees), LoRA isolation (vLLM uses a cache salt per LoRA ID; SGLang namespaces by `extra_key`), and cross-instance sharing (LMCache and Mooncake are the layer that extends the cache beyond a single node by externalizing the block store while the engine's local tree remains the L1 hot tier).

**EPLB plus Elastic EP for MoE becoming standard.** DeepSeek V3's Expert Parallel Load Balancing (EPLB) approach — logical experts with redundant physical copies, periodic rebalance via async weight migration, a cost function that measures per-expert traffic — is now implemented in vLLM (`vllm/distributed/eplb/` with `DefaultEplbPolicy` and async weight migration), SGLang (`eplb/` with `eplb_algorithms/{deepseek,deepseek_vec,elasticity_aware}` pluggable placement algorithms and `EPLBManager.rebalance` chunked across layers), and TRT-LLM (`moe_load_balancer.py` with online and offline variants, copy-engine `cudaMemcpyAsync` for non-blocking weight migration). All three also carry Elastic EP for runtime rank recovery: surviving ranks pick up the missing expert weights from periodic GPU P2P backups without restarting the serving process. The pattern reflects the scale at which MoE models are now deployed: at 256-expert configurations on 96–128-GPU clusters, static expert assignment guarantees load imbalance because the top-k routing distribution is neither uniform across experts nor stable across request batches. Chapter [§20/04](../20-distributed-inference/) covers the algorithmic and communication details; the implementation details in each engine are covered in chapters 01–03 of this section.

**CNCF and Kubernetes-native orchestration solidifying.** llm-d's admission to the CNCF Sandbox in March 2026 and the ratification of the Kubernetes Gateway API Inference Extension (GAIE) as the conformance surface for EPP plugins are the most significant governance events in the inference infrastructure space in the 2024–2026 inference-infrastructure era. Both Dynamo and llm-d build GAIE-conformant EPP plugins: Dynamo's CGO-linked Rust router provides a token-aware KV scorer to the gateway; llm-d's EPP is the reference implementation of the `InferencePool` CRD and `EndpointPickerProtocol` specification. AIBrix's Envoy ext-proc interface predates GAIE and is compatible at the protocol level but not formally conformant. The practical consequence is that teams deploying on any Kubernetes-compatible cloud can now expect an open EPP interface between their L7 proxy (Envoy, kgateway, agentgateway, GKE managed Gateway) and their inference routing logic, with vLLM's ZMQ KV event stream as the standard observability channel feeding the EPP's cache indexer. The vLLM `BlockStored` / `BlockRemoved` / `AllBlocksCleared` event format — originally defined for llm-d and Dynamo interoperability — has become the de-facto wire format for cache-aware routing intelligence across the ecosystem.

---

## How to read this section

Chapters **01–03** cover the cluster inference engines in order of increasing NVIDIA coupling — vLLM (01), SGLang (02), TRT-LLM (03). They are intended to be read in order; each chapter assumes the reader has seen the V1 token-budget scheduler and paged KV designs from [§10/02–03](../10-engine-core/), and chapters 02 and 03 frequently contrast against the vLLM baseline. Chapters **04 and 08** cover LMCache (04) and Mooncake (08) respectively; they can be read immediately after the KV cache chapter [§30](../30-kv-cache/) without reading 01–03 first, though the connector interface discussion assumes familiarity with vLLM's `KVConnectorBase_V1`. Chapters **05–07** cover the cluster orchestration systems — Dynamo (05), llm-d (06), AIBrix (07) — and are most valuable after the distributed inference section [§20](../20-distributed-inference/), which establishes the TP/PP/EP/DP framework these orchestrators coordinate. Chapters **09–10** cover the edge and other engines (llama.cpp, MLX, mistral.rs in 09; ktransformers, lmdeploy, and MAX in 10) and can be read independently at any point.

---

## Current production state

As of mid-2026, the production inference tier at every major hyperscaler and frontier model lab runs on one of three engines: vLLM, SGLang, or TRT-LLM, in roughly that order of deployment breadth. vLLM leads because it is the de-facto standard for the broadest class of workloads — multi-tenant serving of diverse model families, rapid new-model onboarding, and multi-vendor hardware support. SGLang leads in raw TTFT and throughput on the specific models (DeepSeek, Kimi, Qwen, GPT-OSS) that dominate frontier deployment, and its two-batch-overlap scheduler for MoE is the highest-performing open implementation of a class of tricks originally developed inside closed-source engines. TRT-LLM is the ceiling-performance option for NVIDIA-locked deployments where a few percentage points of additional throughput justify the engineering cost of the AOT build pipeline and proprietary kernel dependencies.

The disaggregated P/D tier is maturing but not yet universal. NIXL as the KV transfer fabric is production-proven at scale (Dynamo's claimed 750× throughput gain on GB300 NVL72 in MLPerf 2026 was achieved with disaggregation); LMCache and Mooncake are in production at Kimi and at several US cloud providers. The remaining friction is operational: xPyD ratio tuning, EPLB rebalance cadence, and the interaction between the disagg transceiver and request-level priority — none of which have stable, engine-agnostic abstractions yet. The KVBM/LMCache/aibrix_kvcache three-way split in tiered KV management reflects the same immaturity; the layer directly above the GPU KV pool is the most actively contested surface in the stack.

The Kubernetes orchestration tier is converging on the GAIE EPP protocol but has not yet converged on a single implementation. llm-d's CNCF Sandbox status, Dynamo's production-scale deployment evidence, and AIBrix's feature depth are complementary rather than competitive claims — different organizations will adopt different points on the make-versus-buy spectrum. The most important unresolved question for operators is whether the GAIE conformance surface will prove stable enough to allow EPP implementations to be swapped without breaking production routing logic, or whether the rich plugin ecosystems of Dynamo and AIBrix will make portability aspirational rather than practical.

The edge tier has consolidated around llama.cpp's GGUF format as the interchange standard and Apple Silicon as the premium single-machine target. MLX's JACCL multi-Mac inference is the most interesting emerging capability in this tier: a cluster of Mac Studios sharing an NVLink-speed fabric over Thunderbolt is not a production data-center option, but it is a viable high-end researcher configuration that no other engine currently supports.
