# Glossary

## How to use this glossary

This glossary collects the load-bearing technical terms used throughout the
textbook. Each entry is a short reference-grade definition (1–4 sentences),
optionally followed by a "See also" line that points to related entries and to
chapter sections (chapter paths use the form `10/02` for
`10-engine-core/02-paged-kv-memory.md`). Where a term has a canonical paper or
release artifact, a citation key in square brackets — for example
`[PagedAttention]` — points into the master bibliography at `papers.md`. Acronyms
are expanded on first use within an entry. Entries are grouped by theme so a
reader can also browse by topic; an alphabetical index at the top links into
each thematic section.

## Alphabetical index

A · [AFD](#moe) · [AIBrix](#cluster-systems) · [AlpaServe](#multi-tenant-and-fairness) · [ALiBi](#attention-and-kernels) · [Andes](#multi-tenant-and-fairness) · [AnTKV](#kv-cache) · [AReaL](#rl-post-training) · [Atom](#quantization-formats) · [AutoRound](#quantization-formats) · [AWQ](#quantization-formats)

B · [BASS](#speculative-decoding-and-mtp) · [BGMV](#lora-and-multi-tenant) · [BitNet b1.58](#quantization-formats) · [Blackwell](#hardware) · [Block manager](#kv-cache) · [Block-sparse-row (BSR)](#attention-and-kernels)

C · [CacheBlend](#kv-cache) · [CacheGen](#kv-cache) · [Capacity factor](#moe) · [Cerebras WSE](#hardware) · [Chunked prefill](#batching-and-scheduling) · [ColBERT / ColBERTv2 / PLAID](#workload-primitives) · [ColPali / ColQwen](#workload-primitives) · [Constrained decoding](#workload-primitives) · [Context parallelism (CP)](#distributed-inference) · [Continuous batching](#batching-and-scheduling) · [CPO (Co-Packaged Optics)](#hardware) · [CuTe-DSL](#attention-and-kernels) · [CUDA Graphs](#attention-and-kernels)

D · [DAPO](#rl-post-training) · [DeepEP](#moe) · [DeepGEMM](#moe) · [DeepSeek-R1](#workload-primitives) · [DeepSeek-V3 / V3.2](#moe) · [DéjàVu](#disaggregation-patterns) · [DistServe](#disaggregation-patterns) · [DLPM / D²LPM](#multi-tenant-and-fairness) · [dLoRA](#lora-and-multi-tenant) · [DoRA](#lora-and-multi-tenant) · [DSA (DeepSeek Sparse Attention)](#attention-and-kernels) · [DualPipe](#moe) · [DuoAttention](#kv-cache) · [DualMap](#cluster-systems) · [Dynamo (NVIDIA)](#cluster-systems) · [Dynamic SplitFuse](#batching-and-scheduling)

E · [EAGLE-1 / 2 / 3](#speculative-decoding-and-mtp) · [Eigen Attention](#kv-cache) · [Encode-Prefill-Decode (EPD)](#disaggregation-patterns) · [Endpoint Picker (EPP)](#cluster-systems) · [Envoy AI Gateway](#cluster-systems) · [EPLB](#moe) · [Equinox](#multi-tenant-and-fairness) · [Expert parallelism (EP)](#distributed-inference)

F · [FastVLM](#workload-primitives) · [Fairness routing](#multi-tenant-and-fairness) · [FlashAttention](#attention-and-kernels) · [FlashAttention-2](#attention-and-kernels) · [FlashAttention-3](#attention-and-kernels) · [FlashAttention-4](#attention-and-kernels) · [FlashDecoding](#attention-and-kernels) · [FlashDMoE](#moe) · [FlashInfer](#attention-and-kernels) · [FlashMLA](#attention-and-kernels) · [FlatQuant](#quantization-formats) · [FlexAttention](#attention-and-kernels) · [FlexGen](#heterogeneous-inference) · [FP4 / E2M1](#quantization-formats) · [FP8 / E4M3 / E5M2](#quantization-formats) · [Furiosa RNGD](#hardware)

G · [GDDR](#hardware) · [GDS (GPUDirect Storage)](#hardware) · [GEAR](#kv-cache) · [GGUF / GGML](#quantization-formats) · [GIE (Inference Gateway API)](#cluster-systems) · [GLA](#attention-variants) · [Goodput](#metrics-units-and-workload-taxonomy) · [GPTQ](#quantization-formats) · [Granite 4](#workload-primitives) · [Groq LPU](#hardware) · [Group-limited routing](#moe) · [GRPO](#rl-post-training) · [GTA](#attention-variants)

H · [H2O](#kv-cache) · [HBM](#hardware) · [Helix](#heterogeneous-inference) · [HeteroScale](#cluster-systems) · [HexGen / HexGen-2](#heterogeneous-inference) · [HiCache](#kv-cache) · [Hopper](#hardware) · [HQQ](#quantization-formats) · [HybridEngine](#rl-post-training) · [Hydragen](#kv-cache)

I · [In-flight batching](#batching-and-scheduling) · [InferencePool](#cluster-systems) · [InferenceObjective](#cluster-systems) · [InfiniGen](#kv-cache) · [InfiniBand (NDR / XDR)](#hardware) · [Iteration-level scheduling](#batching-and-scheduling) · [ITL (Inter-Token Latency)](#metrics-units-and-workload-taxonomy)

J · [Jamba](#workload-primitives) · [Janus](#moe) · [JITServe](#multi-tenant-and-fairness)

K · [KIVI](#quantization-formats) · [KServe](#cluster-systems) · [KTransformers](#heterogeneous-inference) · [KV cache](#kv-cache) · [KV-aware routing](#cluster-systems) · [KVBM (KV Block Manager)](#cluster-systems) · [KV transfer (NIXL / Mooncake TE / UCX)](#disaggregation-patterns) · [KVQuant](#quantization-formats)

L · [Late interaction](#workload-primitives) · [LiteLLM](#cluster-systems) · [llm-d](#cluster-systems) · [Llumnix](#multi-tenant-and-fairness) · [LMCache](#kv-cache) · [Lookahead Decoding](#speculative-decoding-and-mtp) · [Lookahead Reasoning](#speculative-decoding-and-mtp) · [LongRoPE / LongRoPE2](#long-context) · [LoRA](#lora-and-multi-tenant) · [LoRAX](#lora-and-multi-tenant) · [LoraHub](#lora-and-multi-tenant) · [LTR (Learning-to-Rank scheduling)](#multi-tenant-and-fairness)

M · [MagicDec](#speculative-decoding-and-mtp) · [Maia (Microsoft)](#hardware) · [Mamba / Mamba-2](#workload-primitives) · [MagicDec](#speculative-decoding-and-mtp) · [Marlin / Machete](#quantization-formats) · [Medusa](#speculative-decoding-and-mtp) · [Mélange](#heterogeneous-inference) · [MegaBlocks](#moe) · [MegaScale-Infer](#moe) · [Microsoft Maia](#hardware) · [MiniCache](#kv-cache) · [MInference](#long-context) · [Mirror Speculative Decoding](#speculative-decoding-and-mtp) · [Mixtral](#moe) · [MLA (Multi-head Latent Attention)](#attention-variants) · [MLX](#workload-primitives) · [Modal](#cluster-systems) · [MoE (Mixture of Experts)](#moe) · [Mooncake](#disaggregation-patterns) · [Mooncake Transfer Engine](#disaggregation-patterns) · [MTIA (Meta)](#hardware) · [MTP (Multi-Token Prediction)](#speculative-decoding-and-mtp) · [MQA / GQA / MLA](#attention-variants) · [MXFP4 / MXFP6 / MXFP8](#quantization-formats)

N · [NaN-aware routing](#moe) · [Native Sparse Attention (NSA)](#attention-and-kernels) · [NCCL-EP](#moe) · [Nexus](#disaggregation-patterns) · [NIXL](#disaggregation-patterns) · [NSA](#attention-and-kernels) · [NVFP4](#quantization-formats) · [NVLink / NVSwitch / NVLink Fusion](#hardware) · [NVMe-resident KV (CMX)](#kv-cache)

O · [OmniQuant](#quantization-formats) · [Online softmax](#attention-and-kernels) · [OpenRLHF](#rl-post-training) · [OpenTelemetry GenAI](#cluster-systems) · [ORCA](#batching-and-scheduling) · [Outlines](#workload-primitives)

P · [P-EAGLE](#speculative-decoding-and-mtp) · [PARD](#speculative-decoding-and-mtp) · [PagedAttention](#kv-cache) · [Parallax](#heterogeneous-inference) · [Petals](#heterogeneous-inference) · [Pipeline parallelism (PP)](#distributed-inference) · [PLD (Prompt Lookup Decoding)](#speculative-decoding-and-mtp) · [POD-Attention](#batching-and-scheduling) · [PowerInfer](#heterogeneous-inference) · [Prefill–Decode disaggregation](#disaggregation-patterns) · [Prefix cache](#kv-cache) · [PRM / ORM](#rl-post-training) · [Punica](#lora-and-multi-tenant) · [PyramidKV](#kv-cache)

Q · [QLoRA](#lora-and-multi-tenant) · [QServe](#quantization-formats) · [QuaRot](#quantization-formats) · [Quartet](#quantization-formats) · [Quest](#kv-cache)

R · [RadixAttention](#kv-cache) · [RAG](#workload-primitives) · [Ray Serve LLM](#cluster-systems) · [REST](#speculative-decoding-and-mtp) · [Reranker](#workload-primitives) · [Reasoning model](#workload-primitives) · [Ring Attention](#distributed-inference) · [RLAIF](#rl-post-training) · [RLHF](#rl-post-training) · [RLVR](#rl-post-training) · [Rollout engine](#rl-post-training) · [RoPE](#long-context) · [Roofline](#metrics-units-and-workload-taxonomy) · [Rubin / Rubin CPX / Rubin Ultra](#hardware) · [RWKV-7](#workload-primitives)

S · [S-LoRA](#lora-and-multi-tenant) · [SageAttention](#attention-and-kernels) · [Sarathi-Serve](#batching-and-scheduling) · [SambaNova SN40L](#hardware) · [Selective batching](#batching-and-scheduling) · [Sequence parallelism (SP)](#distributed-inference) · [SGLang](#cluster-systems) · [SGMV](#lora-and-multi-tenant) · [Sliding-window attention (SWA)](#attention-and-kernels) · [SLO / SLO-aware scheduling](#metrics-units-and-workload-taxonomy) · [slime](#rl-post-training) · [SmoothQuant](#quantization-formats) · [SnapKV](#kv-cache) · [SOLA](#multi-tenant-and-fairness) · [Spec-Bench](#speculative-decoding-and-mtp) · [Speculative decoding](#speculative-decoding-and-mtp) · [Speculators](#speculative-decoding-and-mtp) · [SpecForge](#speculative-decoding-and-mtp) · [Splitwise](#disaggregation-patterns) · [Striped Attention](#distributed-inference) · [StreamingLLM](#attention-and-kernels) · [SuffixDecoding](#speculative-decoding-and-mtp)

T · [TaiChi](#disaggregation-patterns) · [TBT (Token-Between-Tokens)](#metrics-units-and-workload-taxonomy) · [tcgen05 / TMEM](#hardware) · [Tensor parallelism (TP)](#distributed-inference) · [TensorRT-LLM](#cluster-systems) · [TEI (Text Embeddings Inference)](#workload-primitives) · [Tessera](#heterogeneous-inference) · [TGI](#cluster-systems) · [ThunderKittens](#attention-and-kernels) · [TileLang](#attention-and-kernels) · [TMA (Tensor Memory Accelerator)](#hardware) · [TPU (v5/v6/v7)](#hardware) · [Trainium2 / Trainium3](#hardware) · [TransferEngine (Perplexity)](#disaggregation-patterns) · [Tree of Thought (ToT)](#workload-primitives) · [TRT-LLM](#cluster-systems) · [TurboQuant](#quantization-formats) · [TurboSpec / SmartSpec](#speculative-decoding-and-mtp)

U · [UALink](#hardware) · [UCCL](#cluster-systems) · [UCX](#disaggregation-patterns) · [Ulysses (DeepSpeed)](#distributed-inference) · [Ultra Ethernet (UEC)](#hardware) · [USP](#distributed-inference)

V · [vAttention](#kv-cache) · [veRL](#rl-post-training) · [vLLM](#cluster-systems) · [VTC (Virtual Token Counter)](#multi-tenant-and-fairness)

W · [WGMMA](#hardware) · [Wide-EP](#moe)

X · [XGrammar](#workload-primitives) · [XQA](#attention-and-kernels) · [xFLoRA / zFLoRA](#lora-and-multi-tenant)

Y · [YaRN](#long-context) · [YOCO](#attention-variants)

---

## Metrics, units, and workload taxonomy

**Arithmetic intensity.** Ratio of compute work (FLOPs) to memory traffic
(bytes) for a kernel or phase, in FLOP/byte. Compared against a hardware ridge
point (peak FLOPs / peak HBM bandwidth) to determine whether a workload is
compute-bound or memory-bound on a given accelerator.
- See also: Roofline. See `00/02-transformer-arithmetic-roofline`.

**Decode.** Autoregressive token-generation phase: one model forward pass per
output token, reading all weights and the request's KV cache to produce a single
new token. Memory-bandwidth bound at small batch on modern accelerators; reaches
compute-bound regime only at large concurrent batch (the saturation batch).
- See also: Prefill, Roofline, Continuous batching. See `00/02`, `10/03`.

**Goodput.** Throughput restricted to requests that meet their SLOs — for LLM
serving usually $\mathrm{TTFT} \le \tau_{\mathrm{TTFT}}$ and
$\mathrm{p95}(\mathrm{ITL}) \le \tau_{\mathrm{TPOT}}$. Introduced as the canonical
LLM-serving objective by [DistServe] and now standard across cluster-level
schedulers.
- See also: SLO, TTFT, ITL. See `50/01`, `20/02`.

**ITL (Inter-Token Latency).** Mean per-token gap during a single request's
decode stream, $T_{\mathrm{out}}^{(i+1)} - T_{\mathrm{out}}^{(i)}$. Reported as
p50/p95/p99. Often used interchangeably with TPOT in LLM-serving literature; in
strict accounting, ITL is the user-perceived inter-token gap including any
client-side smoothing, while TPOT is the engine's intrinsic per-output-token
latency. See also: TPOT, TBT, Goodput.

**Prefill.** Initial single forward pass that ingests the full prompt,
materializes its KV cache, and emits the first output token. Compute-bound at
typical prompt lengths; sets TTFT.
- See also: Decode, TTFT, Chunked prefill. See `00/02`, `10/03`.

**Roofline.** Performance model that bounds achievable throughput as
$\min(\text{peak FLOPs},\, \text{AI} \cdot \text{peak BW})$ and locates the
ridge point at $\text{AI} = \text{peak FLOPs} / \text{peak BW}$. Anchors prefill
vs. decode analysis and informs batching, parallelism, and quantization choices.
- See also: Arithmetic intensity, Saturation batch. See `00/02`.

**Saturation batch.** Batch size $B^*$ at which the per-rank decode arithmetic
intensity crosses the hardware ridge and the workload transitions from
memory-bound to compute-bound. Sets the natural batch ceiling for throughput
optimization in continuous batching.
- See also: Roofline, Continuous batching. See `00/02`, `10/03`.

**SLO (Service Level Objective).** Per-request latency / throughput target —
typically a TTFT bound and a TPOT bound, sometimes with end-to-end latency or
TBT additions for streaming UIs. Schedulers like SOLA, SLOs-Serve, and JITServe
optimize for SLO attainment rather than raw throughput.
- See also: Goodput, TTFT, TPOT. See `50/01`, `40/03`.

**TBT (Time-Between-Tokens).** Streaming-UX-oriented variant of ITL that
measures the gap between *visible* tokens reaching the client; surfaces
client-side buffering and tokenizer-coalescence effects that ITL hides.
- See also: ITL, TPOT, Andes.

**TPOT (Time-Per-Output-Token).** Engine-side average decode latency per output
token, often reported as the inverse of decode TPS. Equal to ITL when no
client-side smoothing is involved.
- See also: ITL, TBT.

**TTFT (Time-To-First-Token).** Latency from request enqueue to first visible
output token, decomposable as $T_{\mathrm{queue}} + T_{\mathrm{prefill}} +
T_{\mathrm{network}}$. The dominant SLO for interactive chat and the variable
that motivates chunked prefill and PD disaggregation.
- See also: Goodput, Prefill, Chunked prefill.

---

## Attention and kernels

**ALiBi (Attention with Linear Biases).** Position-encoding scheme that adds a
linear distance penalty to attention logits instead of using rotary
embeddings. Supported in modern engines as a `score_mod` in FlexAttention; less
common in 2025–2026 frontier models than RoPE.
- See also: RoPE, FlexAttention.

**Block-Sparse-Row (BSR).** Sparse-matrix layout that FlashInfer generalizes to
unify paged, ragged, radix-tree, and tree-mask KV layouts under a single family
of customizable attention kernels.
- See also: FlashInfer, PagedAttention. See `10/01`, `10/02`.

**Cascade attention.** Two-pass attention used in vLLM where a shared prefix is
attended to in one matmul-shaped pass and unique suffixes are attended to in a
second pass; recovers tensor-core utilization for prefix-heavy serving. Same
idea as Hydragen, integrated as a backend in vLLM.
- See also: Hydragen, RadixAttention, Prefix cache. See `10/07`, `80/01`.

**CuTe-DSL.** Python-embedded tile programming DSL from NVIDIA that targets
CUTLASS performance with 20–30× faster compile than C++ templates. The base
language for FlashAttention-4 and a growing set of Blackwell kernels.
- See also: FlashAttention-4, ThunderKittens, TileLang. See `10/01`. `[CuTe-DSL]`

**CUDA Graphs.** Kernel-launch capture/replay mechanism that replaces per-step
Python and dispatcher overhead with a single replay call. Engines mix
*piecewise* (attention eager, rest captured) and *full* graph modes; choice
trades dynamic-shape flexibility for replay speed.
- See also: torch.compile, Megakernel. See `10/08`.

**DSA (DeepSeek Sparse Attention).** Native sparse attention shipped with
DeepSeek-V3.2 (Sept 2025): a lightning-indexer FP8 head produces a top-k
selection of KV tokens per query, reducing complexity from $O(L^2)$ to $O(Lk)$
and enabling 128K–1M context at ~50% the cost of dense attention. Day-0
kernel support arrived in vLLM and SGLang via FlashMLA.
- See also: NSA, MLA, FlashMLA. See `10/01`, `20/04`. `[DeepSeek-V3.2]`

**FlashAttention.** Tiled, IO-aware exact attention that avoids materializing
the full $N \times N$ score matrix in HBM, using online softmax and a tiled
recurrence. Original CUDA C++ implementation by Tri Dao et al. (2022); the
foundation for every successor kernel.
- See also: Online softmax, FlashAttention-2/3/4. See `10/01`. `[FlashAttn-1]`

**FlashAttention-2.** Re-partitioning of FlashAttention along the sequence axis
(rather than only batch×head), reducing non-matmul FLOPs and reaching ~50–73% of
A100 peak. The default attention algorithm on Ampere.
- See also: FlashAttention, FlashDecoding. See `10/01`. `[FlashAttn-2]`

**FlashAttention-3.** Hopper-specific FlashAttention using TMA + WGMMA + warp
specialization with a producer/consumer pipeline (ping-pong scheduling between
two warpgroups) and FP8 with block-quantization plus incoherent processing.
~740 TFLOP/s FP16 and ~1.2 PFLOP/s FP8 on H100.
- See also: TMA, WGMMA, Warp specialization. See `10/01`, `70/01`. `[FlashAttn-3]`

**FlashAttention-4.** First FlashAttention written in CuTe-DSL, co-designed for
Blackwell's asymmetric scaling. Uses a software-emulated `exp2` polynomial on
FMA units to bypass the SFU bottleneck and ping-pongs over two query tiles per
CTA. ~1605 TFLOP/s BF16 on B200, 1.1–1.3× cuDNN, 2.1–2.7× Triton; the BWD pass
adds 2-CTA MMA mode.
- See also: CuTe-DSL, tcgen05, Blackwell. See `10/01`, `70/01`. `[FlashAttn-4]`

**FlashDecoding.** Split-K decode kernel that partitions along the KV axis to
recover SM utilization for batch-1 long-context decode. The basis for every
modern decode-only attention kernel.
- See also: FlashAttention-2, FlashMLA, XQA. See `10/01`. `[FlashDecode]`

**FlashInfer.** Customizable attention engine (MLSys 2025 best paper) that
unifies paged, ragged, radix, and tree KV layouts under a Block-Sparse-Row
family of JIT-compiled templates. Used by vLLM, SGLang, and MLC; NVIDIA-stewarded
since the OctoAI acquisition.
- See also: Block-Sparse-Row, PagedAttention. See `10/01`, `80/00`. `[FlashInfer]`

**FlashMLA.** DeepSeek's BF16 paged-KV (block size 64) MLA decoder for Hopper,
reaching ~3000 GB/s memory-bound and 580–660 TFLOP/s compute-bound on H800.
The canonical MLA decode kernel; extended in 2025 with DSA sparse paths.
- See also: MLA, DSA, ThunderMLA. See `10/01`, `30/03`. `[FlashMLA]`

**FlexAttention.** PyTorch programming model that lowers user-supplied
`score_mod` and `mask_mod` Python functions through TorchDynamo to a fused
Triton FlashAttention-grade kernel with block-sparse pruning. Subsumes ALiBi,
sliding-window, document masks, and attention sinks under one API.
- See also: Triton, ALiBi, Sliding-window. See `10/01`. `[FlexAttn]`

**GLA (Grouped Latent Attention).** Parallel-friendly variant of MLA that splits
the latent KV across head groups so multiple GPUs can decode without resharding
the cache; ≤2× faster than FlashMLA in spec-decode workloads.
- See also: MLA, GTA, Attention variants. See `30/03`. `[GLA-GTA]`

**GTA (Grouped-Tied Attention).** KV-shape variant that ties keys and values
within a head group, halving GQA's KV bytes at parity quality.
- See also: GQA, GLA. See `30/03`. `[GLA-GTA]`

**Megakernel.** One persistent CUDA kernel per request (or per layer block)
that subsumes attention, MLP, and communication, eliminating launch overhead
entirely. Hazy Research's experimental direction with ThunderKittens 2.0; not
yet shipped as a default in any production engine.
- See also: ThunderKittens, CUDA Graphs.

**Native Sparse Attention (NSA).** Hardware-aligned, natively trainable sparse
attention with three branches — token-compression, token-selection,
sliding-window — that gives up to 11.6× decode speedup at 64K context with no
quality regression. The precursor to DSA in DeepSeek-V3.2.
- See also: DSA, Sliding-window attention. See `20/04`, `10/01`. `[NSA]`

**Online softmax.** Streaming-stable softmax recurrence that keeps a running
max $m$ and partial sum $\ell$ and rescales prior partial outputs by
$\exp(m_{\mathrm{old}} - m_{\mathrm{new}})$ when a new max appears. The algebraic
heart of every FlashAttention variant.
- See also: FlashAttention. See `10/01`. `[FlashAttn-1]`

**SageAttention.** Quantized attention kernel family. SageAttention-1 uses
FP8; SageAttention-2 uses INT4 for $QK$ and FP8 for $PV$ at thread granularity
(~3× FA2 on H100); SageAttention-3 uses NVFP4/MXFP4 microscaling for ~5× FA on
RTX 5090 and is the first plug-and-play FP4 attention kernel.
- See also: NVFP4, MXFP4, FlashAttention-3. See `10/01`, `10/04`. `[SageAttn3]`

**Sliding-window attention (SWA).** Attention restricted to a fixed-radius
window of recent tokens, often combined with a few global tokens or an
attention sink. Used in Mistral, Gemma 3 (5:1 SWA:full), GPT-OSS (1:1 SWA:full
with a learned sink bias), and Phi-3.
- See also: StreamingLLM, Hybrid KV cache, Attention sink. See `20/04`,
  `10/01`. `[StreamingLLM]`

**StreamingLLM.** Discovery that keeping the first few "sink" tokens
permanently along with a sliding window preserves quality at very long contexts
where naive sliding-window attention collapses. Underpins the learned-sink-bias
trick in GPT-OSS / Gemma-3 kernels.
- See also: Sliding-window attention, Attention sink. See `10/01`,
  `30/01`. `[StreamingLLM]`

**ThunderKittens.** Tile-primitive C++ DSL from Hazy Research (Stanford) that
hits cuBLAS / FA-grade speeds in a fraction of the code. Basis for ThunderMLA,
HipKittens (AMD MI300), and ParallelKittens (multi-GPU); v2.0 (2026) supports
NVFP4/MXFP8 GEMMs on Blackwell.
- See also: CuTe-DSL, TileLang. See `10/01`, `10/09`. `[ThunderKittens]`

**TileLang.** Composable tiled programming DSL (PKU/Microsoft, 2025) that
reaches ~98% of FlashMLA performance on H100 in ~70 lines of Python; basis of
QwenLM/FlashQLA and `deepseek-ai/TileKernels`.
- See also: CuTe-DSL, ThunderKittens. See `10/01`. `[TileLang]`

**Triton.** Python-embedded GPU kernel DSL from OpenAI; the universal portable
backend for attention kernels in vLLM (default on AMD), the FlexAttention
lowering target, and the basis of the Liger / Unsloth fused-kernel libraries.
A pure-Triton attention kernel reaches 100.7% of FA3 on H100 (Ringlein et al.,
2025).
- See also: FlexAttention, Liger. See `10/01`. `[Triton-Anatomy]`

**vAttention.** Alternative to PagedAttention (ASPLOS 2025) that keeps each
request's KV contiguous in *virtual* memory and defers paging to CUDA Virtual
Memory Management. Removes the kernel-rewrite tax of PagedAttention; reports
up to 1.97× over vLLM in benchmark settings; not yet adopted as a default in
mainline engines.
- See also: PagedAttention, Block manager. See `10/02`. `[vAttention]`

**Warp specialization.** Producer/consumer pipeline pattern where some warps
issue TMA loads while others execute WGMMA, hiding memory latency behind
compute. Introduced in FA-2-Hopper / FA-3 and increasingly automated by
compilers (Tawa, 2025).
- See also: TMA, WGMMA, FlashAttention-3. See `10/01`, `70/01`.

**XQA.** TensorRT-LLM's MQA/GQA-specialized decode kernel, reporting 2.4×
Llama-70B throughput at iso-latency vs. prior decode kernels. A Hopper QGMMA
path was added in 2025.
- See also: GQA, FlashDecoding. See `10/01`, `80/03`. `[XQA]`

---

## Quantization formats

**Atom.** W4A4 PTQ recipe with mixed-precision (outliers in INT8) plus
group-wise quantization and custom CUDA kernels; up to 7.73× over FP16 on
A100. Predecessor of QServe.
- See also: QServe, QuaRot. See `10/04`. `[Atom]`

**AutoRound.** Block-wise signed-gradient-descent learning of rounding offsets
$v$ and clip parameters $\alpha,\beta$ on top of GPTQ; beats GPTQ/AWQ at 2-bit.
Integrated into LLM Compressor (Dec 2025) and Intel's quantization stack.
- See also: GPTQ, LLM Compressor. See `10/04`. `[AutoRound]`

**AWQ (Activation-aware Weight Quantization).** Identifies the ~1% of weight
channels that align with high-magnitude activations and protects them via
per-channel scaling; the canonical W4A16 recipe alongside GPTQ. MLSys 2024 best
paper.
- See also: GPTQ, SmoothQuant, Marlin. See `10/04`. `[AWQ]`

**BitNet b1.58.** Native ternary {−1, 0, +1} weights via absmean quantization
(~1.58 bits/weight), matching FP perplexity from 3B parameters upward.
BitNet b1.58 2B-4T (Microsoft, April 2025) is the first open production-scale
ternary LLM. The de-facto inference runtime is `bitnet.cpp`.
- See also: BitNet a4.8, GGUF. See `10/04`. `[BitNet b1.58]`

**E4M3 / E5M2.** The two FP8 mantissa/exponent splits standardized by
Micikevicius et al. (2022). E4M3 has range ±448 and is preferred for
forward/weight/activation use; E5M2 has range ±57344 and is preferred for
gradients in training. Both are native on Hopper and later.
- See also: FP8, FP8-LM, NVFP4. See `10/04`, `70/01`. `[FP8 Formats for DL]`

**FlatQuant.** Per-layer Kronecker-factored learnable affine transforms fused
into a single runtime kernel; W4A4 LLaMA-3-70B with <1% accuracy drop and 2.3×
prefill / 1.7× decode speedup vs FP16. Closes the rotation-based-W4A4 quality
gap further than QuaRot/SpinQuant/DuQuant.
- See also: QuaRot, SpinQuant, Rotation lineage. See `10/04`. `[FlatQuant]`

**FP4 (E2M1).** 4-bit floating-point representation with one sign, two exponent,
one mantissa bit (max value 6). Native on Blackwell and AMD CDNA 4 tensor
cores; the carrier format inside MXFP4 and NVFP4 microscaling.
- See also: NVFP4, MXFP4, Blackwell. See `10/04`, `70/01`.

**FP8.** 8-bit floating-point used for both weights/activations and KV cache;
native on Hopper, Blackwell, MI300/MI350, TPU v6+. Production default for most
Hopper inference deployments since 2024.
- See also: E4M3, E5M2, FP8-LM. See `10/04`. `[FP8 Formats for DL]`

**GGUF / GGML.** Quantized model format (GGUF) and tensor-library kernels
(GGML) used by llama.cpp. Supports k-quants (Q4_K_M, etc.) and i-quants
(IQ4_XS, IQ3_S, etc., codebook-based with importance matrix); MXFP4 and NVFP4
ggml types added in 2025 to support GPT-OSS.
- See also: HQQ, AWQ, llama.cpp. See `80/06`, `10/04`.

**GPTQ.** Approximate-second-order (Hessian-based) layer-wise weight
quantization (ICLR 2023). The reference 4-bit / 3-bit weight-only recipe from
which most production W4A16 paths descend; surviving kernel form is Marlin /
Machete.
- See also: AWQ, OmniQuant, Marlin. See `10/04`. `[GPTQ]`

**HQQ (Half-Quadratic Quantization).** Calibration-free closed-form
half-quadratic optimization that minimizes weight error directly; ~50× faster
to quantize than GPTQ at competitive 4-bit quality.
- See also: GPTQ, AWQ, AutoRound. See `10/04`. `[HQQ]`

**KIVI.** Tuning-free 2-bit KV-cache quantization with per-channel scaling on
keys (which exhibit channel-aligned outliers) and per-token scaling on values.
2.6× peak KV memory reduction and 2.35–3.47× throughput.
- See also: KVQuant, FP8 KV. See `30/01`, `10/04`. `[KIVI]`

**KVQuant.** Pre-RoPE per-channel key quantization, non-uniform NUQ codebooks,
and per-vector dense-and-sparse outlier isolation; 3-bit KV with <0.1
perplexity hit and 10M-token context demonstrations.
- See also: KIVI, TurboQuant. See `30/01`, `10/04`. `[KVQuant]`

**LLM Compressor.** Open-source compression toolchain stewarded by Red Hat (ex
Neural Magic) for vLLM-targeted deployments; ships W4A16, FP8, NVFP4, MXFP4,
attention quantization, and distributed GPTQ paths. The de-facto open
production toolkit for vLLM compression.
- See also: Marlin, Machete, ModelOpt. See `10/04`, `80/01`.

**Marlin / Machete.** vLLM's W4A16 GEMM kernels for INT4/FP4 weights with FP16
activations. Marlin runs on Ampere/Hopper; Machete is the Hopper+ revision
optimized for B200's tcgen05.
- See also: GPTQ, AWQ. See `10/04`, `80/01`.

**MX (Microscaling).** OCP-standardized block-floating-point family in which a
block of $k$ elements (typically 32) shares an E8M0 scale with $w$ bits; MXFP4
(E2M1 element), MXFP6 (E2M3 / E3M2), MXFP8 (E4M3 / E5M2), and MXINT8 are the
defined formats.
- See also: NVFP4, FP4, OCP MX spec. See `10/04`, `70/01`. `[OCP MX v1.0 Spec]`

**MXFP4.** 32-element MX block with E2M1 elements and an E8M0 shared scale
(136 bits/block, 17 bytes). The native weight format of OpenAI's GPT-OSS
release (Aug 2025) and AMD's CDNA 4 path.
- See also: NVFP4, GPT-OSS. See `10/04`, `70/01`.

**MXFP6 / MXFP8.** 6-bit and 8-bit MX block-floating-point variants. MXFP8 is
production-grade on Blackwell tensor cores; MXFP6 sits between FP4 and FP8 on
the accuracy/throughput frontier.
- See also: MX, NVFP4. See `10/04`.

**NVFP4.** NVIDIA-defined FP4 with two-level scaling: per-block (size 16) E4M3
scale plus a per-tensor FP32 scale. Native on Blackwell tensor cores; provides
3.5× memory vs FP16 and 1.8× vs FP8 with <1% accuracy drop. Pre-training
recipe demonstrated at 12B scale (NVIDIA, Sep 2025).
- See also: FP4, MXFP4, Quartet. See `10/04`, `70/01`. `[NVFP4 Inference]`

**OmniQuant.** GPTQ-derived PTQ with two learnable components — Learnable
Weight Clipping (LWC) and Learnable Equivalent Transformation (LET); strong at
W2A16 / W3A16 / W4A4 but heavier than GPTQ/AWQ.
- See also: GPTQ, SmoothQuant. See `10/04`. `[OmniQuant]`

**QServe (W4A8KV4).** MIT Han Lab co-design (MLSys 2025) with progressive
dequantization (W4 → W8) and SmoothAttention to protect KV4. 1.2–1.4× over
TRT-LLM on A100, 2.4–3.5× on L40S for 72B-class models.
- See also: SmoothQuant, KIVI. See `10/04`. `[QServe / QoQ]`

**QuaRot.** End-to-end W4A4KV4 by inserting four orthogonal rotations
(R1/R2 fused offline into weights, R3/R4 online via Hadamard) so weights and
activations become outlier-free in the rotated basis. Foundation of the
rotation lineage.
- See also: SpinQuant, FlatQuant, Rotation lineage. See `10/04`. `[QuaRot]`

**Quartet / Quartet II.** End-to-end FP4 / NVFP4 *training* recipes (IST-DASLab,
2025–2026) using random Hadamard rotations, 2D quantization, and stochastic
rounding (Quartet) or MS-EDEN unbiased micro-scale rounding (Quartet II) for
unbiased FP4 gradient estimation. Establishes that FP4 pre-training is
competitive with FP8.
- See also: NVFP4, Rotation lineage. See `10/04`. `[Quartet]`

**Rotation lineage.** Family of preprocessing transforms — QuaRot →
SpinQuant → DuQuant → FlatQuant → BASE-Q → Quartet — that fuse an orthogonal
rotation into RMSNorm-bordered linear layers to make weights and activations
isotropic before quantizing. The default preprocessing step before any
4-bit-activation scheme.
- See also: QuaRot, SpinQuant, DuQuant, FlatQuant. See `10/04`.

**SmoothQuant.** Offline activation smoothing $X' = X \cdot \mathrm{diag}(s)^{-1}$,
$W' = \mathrm{diag}(s) \cdot W$ that migrates outlier difficulty from activations
into weights, enabling W8A8 PTQ. The reference INT8 recipe; the smoothing
reformulation reappears in SpinQuant, FlatQuant, and OmniQuant's LET.
- See also: AWQ, Rotation lineage. See `10/04`. `[SmoothQuant]`

**SpinQuant.** Replaces QuaRot's random Hadamard rotation with
Cayley-optimized *learned* rotations that preserve full-precision invariance;
on LLaMA-3-8B closes the gap to FP by +45.1% relative to QuaRot.
- See also: QuaRot, FlatQuant. See `10/04`. `[SpinQuant]`

**TurboQuant.** Online vector quantization (Google, ICLR 2026) that applies a
random rotation, per-coordinate scalar quantization, and a 1-bit JL residual
to obtain unbiased inner-product estimates near the information-theoretic
distortion lower bound. KV cache parity at 3.5 bits/channel and marginal
degradation at 2.5 bits/channel, training-free.
- See also: KIVI, KVQuant, Rotation lineage. See `30/01`, `10/04`.
  `[TurboQuant]`

---

## KV cache

**AnTKV.** Vector-quantization KV scheme with anchor-token-aware selective FP16
retention of ~1% high-sensitivity tokens; 1-bit Mistral-7B perplexity 6.32 vs
KVQuant's 15.36 and 840K-token context on a single A100.
- See also: KVQuant, TurboQuant. See `30/01`. `[AnTKV]`

**Block manager.** Engine-side allocator that maps logical KV blocks (e.g.
16-token slots) to physical HBM frames, maintains a per-request block table,
handles copy-on-write sharing for prefix cache hits, and tracks free-list /
LRU eviction. The data-structure backbone of PagedAttention.
- See also: PagedAttention, Prefix cache. See `10/02`, `80/01`.

**CacheBlend.** Selective recompute scheme (EuroSys 2025) that reuses
precomputed KVs for non-prefix RAG-style chunks and repairs cross-attention by
recomputing a small subset of tokens; 2.2–3.3× TTFT and 2.8–5× throughput on
RAG workloads. Lives inside LMCache.
- See also: LMCache, CacheGen, RAG. See `10/07`, `30/02`. `[CacheBlend]`

**CacheGen.** Cross-tier KV codec (SIGCOMM 2024) using per-layer-sensitive
quantization plus arithmetic coding to compress KV for network transit;
3.5–4.3× cache-size reduction with bandwidth-adaptive compression. The default
encode/decode path for cross-node KV in LMCache.
- See also: CacheBlend, LMCache. See `30/02`. `[CacheGen]`

**DroidSpeak.** Cross-LLM KV-cache sharing scheme: across same-architecture
LLMs, recompute only a few layers and reuse the rest; 4× throughput, 3.1× TTFT
in compound-AI / agent settings. Not in any production engine yet.
- See also: CacheBlend, Multi-model. See `30/01`.

**DuoAttention.** Heads partitioned by an output-deviation training procedure
into "retrieval" (full KV) and "streaming" (sink + window) types; 2.55× MHA /
1.67× GQA memory reduction at long context. Closer to deployable than purely
heuristic per-head eviction.
- See also: SnapKV, Quest, Sliding-window. See `30/01`,
  `20/04`. `[DuoAttention]`

**Eigen Attention.** Joint low-rank approximation of $Q$, $K$, $V$ in a shared
subspace for KV compression; up to 40% KV reduction and 60%
attention-latency reduction without quantization.
- See also: MLA, Loki, MiniCache. See `30/01`. `[Eigen-Attn]`

**FastGen.** Per-head adaptive policy mix (special-token / local / heavy / full)
chosen via a one-shot prefill profile. The first head-specific KV-compression
recipe and a precursor to DuoAttention.
- See also: H2O, DuoAttention. See `30/01`. `[FastGen]`

**FP8 KV cache.** Storing the KV cache in FP8 (E4M3 typically) with per-tensor
or per-head scales; halves cache vs BF16 and is the recommended starting point
on Hopper/Blackwell for long-context decode in vLLM and TRT-LLM.
- See also: KIVI, NVFP4 KV. See `30/01`, `10/04`.

**GEAR.** Hybrid KV compression: low-bit base $Q$ + low-rank residual $L$ +
sparse outliers $S$. The most explicit hybrid-compression paper; influences
GEAR-style stacks in 2025 work.
- See also: KIVI, MiniCache. See `30/01`. `[GEAR]`

**H2O (Heavy-Hitter Oracle).** Eviction policy that keeps recent tokens plus
"heavy hitter" tokens by cumulative attention scores, with a dynamic
submodular formulation. The lineage starting point for attention-score-based
KV eviction.
- See also: SnapKV, Scissorhands, StreamingLLM. See `30/01`. `[H2O]`

**HiCache (SGLang).** Hierarchical KV cache (GA Sept 2025) that extends
RadixAttention across L1 GPU / L2 host / L3 distributed backends (Mooncake,
3FS, NIXL, AIBrix, local) with write-through, write-through-selective, and
write-back policies plus GPU-assisted I/O.
- See also: RadixAttention, LMCache, KVBM. See `30/02`,
  `80/02`. `[SGLang-HiCache]`

**Hydragen.** Decomposes attention into shared-prefix (matmul-shaped) and
unique-suffix passes; up to 32× over baselines for long shared prefixes.
Conceptual basis for vLLM's Cascade attention and SGLang's prefix-aware
kernels.
- See also: Cascade attention, ChunkAttention. See `10/07`. `[Hydragen]`

**InfiniGen.** Speculatively prefetches only essential KV pages from CPU using
a partial-rehearsal predictor on the next layer's $Q$; up to 3× speedup over
offload baselines.
- See also: Quest, RetrievalAttention. See `30/01`. `[InfiniGen]`

**KV cache.** Sequence of cached key and value projections at every transformer
layer for an in-flight request; per-token bytes are
$2 \cdot L \cdot H_{\mathrm{kv}} \cdot d_h \cdot b$ and dominate decode-time HBM
traffic. The largest piece of state in autoregressive serving.
- See also: PagedAttention, Block manager, MLA. See `10/02`, `30/03`.

**LMCache.** Open-source pluggable KV-cache layer used by vLLM Production
Stack and llm-d; supports DRAM, NVMe, Redis, Mooncake Store, and NIXL
backends, and embeds CacheGen + CacheBlend codecs. Reports up to ~15× throughput
in shared-prefix workloads.
- See also: Mooncake, NIXL, KVBM. See `30/02`, `80/04`. `[LMCache]`

**MiniCache.** Cross-layer KV merging via magnitude+direction decomposition;
1.53× compression alone, ~5× stacked with 4-bit. The leading "depth-axis"
KV-compression method.
- See also: Eigen Attention, GEAR. See `30/01`. `[MiniCache]`

**MLA prefix-cache.** Variant of prefix-cache addressing that stores the
shared latent $c^{KV}$ and per-head decoupled-RoPE key $k^R$ rather than
per-head K/V. Engines (vLLM, SGLang) had to add MLA-aware paths because the
default 16-token-block prefix-cache assumed per-head KV layout.
- See also: MLA, Prefix cache. See `10/07`, `30/03`.

**Mooncake KVStore.** Distributed KV cache pool over CPU DRAM / SSD / NIC of
a GPU cluster, accessed via Mooncake Transfer Engine. Production system behind
Kimi; integrated as a backend by SGLang HiCache, LMCache, and AIBrix.
- See also: Mooncake Transfer Engine, LMCache. See `30/02`, `80/04`. `[Mooncake]`

**NVMe-resident KV (CMX).** Inference Context Memory Storage Platform (NVIDIA
BlueField-4, CES 2026): petabyte-scale RDMA-accelerated NVMe tier holding
persistent KV across runs. The fourth tier in the modern KV pyramid.
- See also: KVBM, NIXL. See `30/02`, `70/05`. `[CMX]`

**PagedAttention.** Memory-management abstraction (SOSP 2023) that allocates
KV in fixed-size logical blocks and maps them through a per-request block
table to physical HBM frames; enables copy-on-write prefix sharing and
near-zero external fragmentation. The dominant KV memory model in vLLM,
SGLang, TRT-LLM, and TGI.
- See also: vAttention, Block manager, KV cache. See `10/02`,
  `80/01`. `[PagedAttention]`

**Prefix cache.** Cross-request reuse of KV blocks for shared prompt prefixes
(system prompts, few-shot exemplars, repeated documents). Implemented as
hash-chained blocks (vLLM V1) or radix tree (SGLang); default-on in modern
engines.
- See also: RadixAttention, vLLM V1 prefix cache, Anthropic prompt
  caching. See `10/07`, `30/02`.

**PyramidKV.** Layer-tapered KV-cache budget — more cache in lower layers,
less in higher — reflecting observed pyramidal information funneling. Matches
full-cache LongBench at 12% retention.
- See also: SnapKV, Ada-KV. See `30/01`. `[PyramidKV]`

**Quest.** Per-page min/max key bounds plus per-query criticality estimate;
loads only the top-K critical pages per query at decode; 2.23× attention
speedup, 7.03× latency. Query-aware sparsity, distinct from history-based
eviction.
- See also: MInference, RetrievalAttention. See `30/01`,
  `20/04`. `[Quest]`

**RadixAttention.** Token-level radix-tree shared-prefix KV cache (SGLang,
NeurIPS 2024) with LRU eviction and cache-aware request scheduling; the
conceptual basis for prefix-cache-aware routing in production routers and the
dominant abstraction in SGLang.
- See also: Prefix cache, KV-aware routing, HiCache. See `10/07`,
  `80/02`. `[RadixAttention-SGLang]`

**Scissorhands.** Persistence-of-importance hypothesis: pivotal tokens stay
pivotal. 5× cache reduction; 20× when stacked with 4-bit. Co-canonical with
H2O and FastGen.
- See also: H2O, FastGen. See `30/01`. `[Scissorhands]`

**SnapKV.** Uses the last $N$ "observation" prompt tokens to detect head-specific
attention patterns and keeps clustered important keys; the dominant non-MLA
eviction baseline through 2024–2025.
- See also: H2O, PyramidKV, DuoAttention. See `30/01`,
  `20/04`. `[SnapKV]`

**vLLM V1 prefix cache.** Hash-chained block prefix cache: each block's hash
includes the prior block's hash (Merkle-style), `cache_salt` provides tenant
isolation, and LoRA-ID is mixed into the hash. Default-on in V1 with reported
<1% overhead at 0% hit rate.
- See also: Prefix cache, RadixAttention. See `10/07`, `80/01`.

**YOCO (You Only Cache Once).** Decoder-decoder architecture in which a
self-decoder produces a single global K/V that the cross-decoder layers attend
to; ~80× memory reduction at 65B and 1M-context prefill in seconds. Proof of
concept that the per-layer cache is not necessary.
- See also: MLA, Hybrid models. See `30/03`. `[YOCO]`

---

## Distributed inference

**Context parallelism (CP).** Sharding a single sequence's attention
computation across devices along the context axis at inference time; the
inference-side analog of training-side sequence parallelism. Used in TRT-LLM
and SGLang for ultra-long context prefill.
- See also: Sequence parallelism, Ring Attention, Ulysses. See `20/01`,
  `20/04`.

**Data parallelism (DP).** Independent replicas serving disjoint requests;
trivial scale-out for stateless serving but requires routing/affinity for
prefix-cache reuse. In MoE serving DP is often combined with EP so each rank
holds a subset of experts plus a replica of attention.
- See also: TP, PP, EP. See `20/01`.

**DualPipe.** Bidirectional pipeline-parallel schedule from DeepSeek that
splits each chunk into attention / dispatch / MLP / combine and runs forward
and backward passes in opposing directions to fully overlap compute with
cross-node EP all-to-all. Training-side primitive whose ideas leak into
inference micro-batch overlap (TBO/SBO).
- See also: DeepEP, Two-Batch Overlap. See `20/01`,
  `20/03`. `[DualPipe]`

**Expert parallelism (EP).** Sharding MoE experts across devices so each rank
hosts a subset; tokens are routed via all-to-all dispatch and combine. EP
degree at frontier scale ranges from EP-32 (DeepSeek-V3 prefill) to EP-256+
(GB200 NVL72 Wide-EP); enables MoE serving at frontier sparsity ratios.
- See also: DeepEP, EPLB, Wide-EP, MoE. See `20/01`, `20/03`.

**Pipeline parallelism (PP).** Splitting the model along its layer axis across
devices, with micro-batches flowing through stages. Communication is small
(activations only at stage boundaries) but introduces pipeline bubbles
proportional to depth × stage-time variance; relevant at inference for
extreme-context prefill (Mnemosyne, SGLang chunked PP).
- See also: TP, EP, Mnemosyne. See `20/01`, `20/04`.

**Ring Attention.** Distributes KV blocks around a P2P ring with overlapped
communication and computation; linearly scales context length with device
count. The canonical long-sequence prefill kernel; basis of
Striped/USP/Long-Ctx-Attn.
- See also: Striped Attention, Ulysses, USP. See `20/01`,
  `20/04`. `[Ring-Attn]`

**Sequence parallelism (SP).** Sharding along the sequence axis to remove
duplicated activations across tensor-parallel ranks (Megatron-SP) or to
distribute attention work across devices for very long contexts. SP variants
include Megatron-SP (head-parallel companion to TP), Ulysses (all-to-all on
heads), and Ring/Striped (P2P ring on KV blocks).
- See also: TP, Ring Attention, USP. See `20/01`, `20/04`. `[Megatron-SP]`

**Striped Attention.** Permutes the token-to-device mapping in Ring Attention
to even out the causal-mask workload imbalance; up to 1.65× over Ring at 786K
context on TPUv4.
- See also: Ring Attention. See `20/04`. `[Striped-Attn]`

**Tensor parallelism (TP).** Sharding individual layers (heads in attention,
columns/rows in MLP) across devices; communication is an all-reduce per layer.
The default scale-up parallelism on NVLink-connected GPUs; bandwidth-hungry,
which is why TP usually lives within an NVLink domain.
- See also: PP, EP, NVLink. See `20/01`.

**Two-Batch Overlap (TBO).** SGLang scheduling primitive that runs two
micro-batches with offset stages so MoE all-to-all dispatch/combine of one
micro-batch overlaps the compute of the other; +27–35% throughput on prefill
in DeepSeek-style EP. Single-Batch Overlap (SBO) is the latency-sensitive
variant.
- See also: DualPipe, Dual Batch Overlap, DeepEP. See `20/03`, `80/02`.

**Ulysses (DeepSpeed Ulysses).** Sequence-parallel scheme that uses an
all-to-all on heads so communication volume stays constant as devices scale
with sequence length; bounded by head count, compositional with Ring.
- See also: Sequence parallelism, USP. See `20/04`. `[DS-Ulysses]`

**USP (Unified Sequence Parallelism).** 2D hybrid (Ulysses outer, Ring
inner) sequence-parallel recipe that became the de-facto OSS long-context
attention library (`feifeibear/long-context-attention`).
- See also: Ulysses, Ring Attention, Striped Attention. See `20/04`. `[USP / Long-Ctx-Attn]`

**Wide-EP.** Production label for very-large expert-parallelism deployments
(EP-64, EP-72, EP-128, EP-256+) on NVL72-class fabrics; vLLM Wide-EP and
TRT-LLM Wide-EP refer to the engine support for this regime.
- See also: EP, EPLB, NVL72. See `20/03`, `80/01`.

---

## Speculative decoding and MTP

**Acceptance length / acceptance rate.** $\alpha$ is the per-token acceptance
probability; for $\gamma$ proposed tokens, expected accepted-prefix length
$E[L] = (1 - \alpha^{\gamma+1})/(1 - \alpha)$ (Leviathan's geometric form).
- See also: Speculative decoding, Modified rejection sampling. See `10/05`. `[SpecDec-Leviathan]`

**BASS.** Batched Attention-optimized Speculative Sampling (AWS, ACL Findings
2024); first systematic study of batched SD under continuous batching, 2.15×
over optimized regular decoding at batch 8 on A100.
- See also: TurboSpec, BatchSpec. See `10/05`. `[BASS]`

**EAGLE-1 / EAGLE-2 / EAGLE-3.** Self-speculative drafter family that drafts
at the *feature* level. EAGLE-1 introduced one-token-shifted feature
prediction; EAGLE-2 added confidence-based dynamic tree expansion; EAGLE-3
replaced feature-prediction with direct token prediction and added training-time
test fusion of early/middle/late layer features. EAGLE-3 is the de-facto
post-hoc drafter for non-MTP models in vLLM, SGLang, and TRT-LLM.
- See also: Speculators, SpecForge, MTP. See `10/05`. `[EAGLE-1]`, `[EAGLE-2]`, `[EAGLE-3]`

**Goodput (SD).** SmartSpec's framing: dynamically modulate $\gamma$ per request
to maximize accepted-tokens-per-second under continuous batching, rather than
optimizing per-request speedup in isolation.
- See also: Goodput, TurboSpec, MagicDec. See `10/05`. `[SmartSpec / TurboSpec]`

**Lookahead Decoding.** Training-free speculative decoder that uses
Jacobi-iteration-style parallel n-gram extraction on the model's own past
outputs; 1.8× MT-bench, 4× code, no draft model needed.
- See also: PLD, REST, Token Recycling. See `10/05`. `[Lookahead-Decoding]`

**Lookahead Reasoning.** Step-level (not token-level) speculative decoding for
reasoning models: a lightweight drafter proposes future *reasoning steps* and
the verifier checks semantic correctness rather than exact tokens; takes SD
speedup on R1-class models from 1.4× to 2.1× on GSM8K/AIME.
- See also: Speculative decoding, Reasoning model. See `10/05`,
  `60/01`. `[Lookahead-Reasoning]`

**MagicDec.** Long-context KV-bound regime: a fixed-context-window draft using
even the full target model can speed up because KV-load dominates; 2.51× on
Llama-3.1-8B at batch 32–256.
- See also: Speculative decoding, Long context. See `10/05`. `[MagicDec]`

**Medusa.** Self-speculative drafter that adds *parallel*, sequentially
independent prediction heads to draft $k$ tokens per step plus tree-attention
verification. Medusa-2 reaches 2.3–3.6× walltime; superseded as the default
post-hoc drafter by EAGLE-3 in 2025.
- See also: EAGLE-1, Hydra. See `10/05`. `[Medusa]`

**Mirror Speculative Decoding.** Apple, 2025–2026: drafter and target speculate
*for each other* with cross-device (GPU + NPU) pipelining; +30% over EAGLE-3 on
SpecBench. Not yet in OSS engines.
- See also: EAGLE-3, P-EAGLE. See `10/05`. `[Mirror-SD]`

**Modified rejection sampling.** Verification rule that accepts draft token
$\tilde x$ with probability $\min(1, p(\tilde x)/q(\tilde x))$ and otherwise
samples from the corrected distribution; preserves the target distribution
exactly. The algorithmic backbone of all SD variants.
- See also: Acceptance length, Speculative decoding. See `10/05`. `[SpecDec-Leviathan]`

**MTP (Multi-Token Prediction).** Pre-training auxiliary loss with shared trunk
and depth-$D$ heads (Gloeckle et al., 2024) that improves base-model quality
*and* yields drafter heads usable at inference. DeepSeek-V3's MTP module (D=1,
~14B params on top of 671B) gives MTP1 acceptance >80% and ~1.8× generation
TPS. Qwen3-Next, Qwen3.6, and Gemma 4 also ship native MTP heads; complementary
to SD verification rather than a replacement.
- See also: Speculative decoding, EAGLE-3. See `10/06`. `[Better-Faster-MTP]`,
  `[DeepSeek-V3]`

**P-EAGLE.** AWS's parallel EAGLE variant in vLLM ≥0.16.0 that pipelines
drafter and verifier forward passes; 1.69× over vanilla EAGLE-3 on B200.
- See also: EAGLE-3, Mirror-SD. See `10/05`. `[P-EAGLE]`

**PARD.** AMD's target-independent parallel drafter with conditional drop-token
training; 3.67× on Llama-3.1-8B and 1.15× over EAGLE-3.
- See also: EAGLE-3. See `10/05`. `[PARD]`

**PLD (Prompt Lookup Decoding).** Replaces a drafter with n-gram match against
the prompt history; ~2.4× on summarization/QA, training-free.
- See also: REST, SuffixDecoding, Lookahead. See `10/05`. `[PLD]`

**REST.** Retrieval-Based Speculative Decoding: external datastore retrieval
as drafter; 1.62–2.36× on 7B/13B; plug-and-play.
- See also: PLD, SuffixDecoding. See `10/05`. `[REST]`

**SmartSpec / TurboSpec.** Online controller that adjusts proposed/verified
token counts per iteration based on profiled goodput; up to 3.2× over non-spec
baselines under continuous batching. Renamed TurboSpec in v2.
- See also: Goodput, BASS. See `10/05`,
  `10/03`. `[SmartSpec / TurboSpec]`

**Spec-Bench.** Standard 6-task benchmark for speculative decoding (Heming Xia
et al.); the basis for almost all per-task SD numbers cited in the chapter.
- See also: SPEED-Bench. See `10/05`. `[Spec-Bench]`

**SPEED-Bench.** NVIDIA-led 2026 benchmark unifying domain coverage and
serving-regime diversity for SD; reports per-batch-size and per-workload
acceptance/throughput tables.
- See also: Spec-Bench. See `10/05`. `[SPEED-Bench]`

**SpecForge.** LMSYS's open-source EAGLE-3 training framework (July 2025);
trains in-house drafters for Llama-4 Scout / Maverick at 2.0× / 2.18× MT-Bench.
- See also: EAGLE-3, Speculators. See `10/05`. `[SpecForge]`

**Speculative decoding (SD).** Lossless latency-reduction technique in which a
cheaper *drafter* proposes $\gamma$ tokens, the *target* verifies them in one
forward pass, and modified rejection sampling preserves the target
distribution exactly. The canonical algorithm by Leviathan et al. (2022) and
Chen et al. (2023).
- See also: Modified rejection sampling, EAGLE, MTP, MagicDec.
  See `10/05`. `[SpecDec-Leviathan]`, `[SpecSamp-Chen]`

**Speculators.** Red Hat / vLLM's standardized HF format for spec-dec drafters
plus a one-button training/serving pipeline; v0.3.0 (Dec 2025) ships in vLLM.
- See also: EAGLE-3, SpecForge. See `10/05`. `[Speculators]`

**SuffixDecoding.** Suffix-tree-over-history training-free drafter; up to 5.3×
on agentic / repetitive workloads (SWE-bench, AgenticSQL). Production-deployed
in vLLM via Arctic Inference.
- See also: PLD, REST. See `10/05`. `[SuffixDecoding]`

**Tree attention / token-tree verification.** Verifier-side mechanism (SpecInfer,
Sequoia) that uses a tree-causal attention mask to verify multiple candidate
continuations in one forward pass; enables EAGLE-2-style dynamic draft trees
and Sequoia's hardware-aware DP tree shapes.
- See also: SpecInfer, Sequoia, EAGLE-2. See `10/05`. `[SpecInfer]`,
  `[Sequoia]`

---

## Disaggregation patterns

**AFD (Attention-FFN Disaggregation).** Placing attention and FFN/expert pools
on separate GPUs because their parallelisms, memory pressure, and tensor cores
are different. Demonstrated by Step-3 (Aug 2025), MegaScale-Infer (SIGCOMM
2025), and Janus (Dec 2025); a sub-section of the MoE chapter, not a standalone
chapter.
- See also: PD disaggregation, MoE. See `20/03`. `[Step-3]`, `[MegaScale-Infer]`,
  `[Janus]`

**DéjàVu.** Three-in-one library (ICML 2024) for prompt-token disaggregation,
microbatch swapping, and per-stage KV replication to a logical-ring neighbor for
fault tolerance; 1.24× recovery overhead vs 1.91× without FT.
- See also: PD disaggregation, fault tolerance. See `20/02`. `[DejaVu]`

**DistServe.** OSDI 2024 paper that formalized PD disaggregation: defined
goodput, derived independent per-phase replica counts, and showed 7.4× more
requests / 12.6× tighter SLOs vs co-located baselines.
- See also: Splitwise, Mooncake, PD disaggregation, Goodput. See `20/02`. `[DistServe]`

**EPD (Encode–Prefill–Decode).** Three-way disaggregation for multimodal
serving that adds an encode stage (vision/audio/video) on top of PD. HydraInfer,
ModServe, vLLM-Omni, and SpaceServe instantiate the pattern; SGLang's #8223
roadmap and Mooncake's transfer protocols support it.
- See also: PD disaggregation, Multimodal. See `60/03`,
  `20/02`. `[ModServe-2025]`

**Mooncake.** KVCache-centric disaggregated architecture (FAST 2025 Best Paper)
behind Kimi: distributed KV pool over CPU/DRAM/SSD/NIC of the GPU cluster, a
Conductor global scheduler, and prediction-based early rejection under
overload. Open-sourced as Mooncake Transfer Engine + Mooncake Store.
- See also: Mooncake Transfer Engine, NIXL, LMCache. See `20/02`,
  `30/02`. `[Mooncake]`

**Mooncake Transfer Engine.** RDMA / RoCE / NVMe-oF / NVLink / CXL transport
under a unified zero-copy API; multi-NIC GPUDirect RDMA pooling drives line-rate
KV transfers. Used as a transport in SGLang, LMCache, vLLM disagg, and AIBrix.
- See also: NIXL, UCX, Mooncake. See `20/02`,
  `30/02`. `[Mooncake-Store]`

**Nexus.** Intra-GPU disaggregation (arXiv 2507.06608, 2025): dynamic SM
allocation between prefill and decode on a single GPU. Up to 2× throughput vs
SGLang and TTFT 2–20× better via SPF + dynamic SM reallocation.
- See also: POD-Attention, PD disaggregation. See `20/02`. `[Nexus]`

**NIXL (NVIDIA Inference Xfer Library).** Vendor-agnostic point-to-point KV
transfer with pluggable backends (UCX, GPUDirect RDMA, GDS, S3); single
register/transfer/poll API across NVLink, IB, RoCE, and Ethernet. The default
KV transport in NVIDIA Dynamo and one of the connectors in vLLM and LMCache.
- See also: Mooncake Transfer Engine, UCX. See `20/02`, `30/02`,
  `80/05`. `[Dynamo-NIXL]`

**PD disaggregation (Prefill–Decode disaggregation).** Architectural pattern
of running the prompt-prefill and autoregressive-decode stages on physically
distinct GPU pools, transferring KV across the boundary. Production default
for high-traffic large-model deployments since 2025.
- See also: DistServe, Splitwise, Mooncake, NIXL. See `20/02`.

**Splitwise.** ISCA 2024 "phase splitting" study: first systems treatment of
PD as a deployment problem; introduced layer-wise overlapped KV transfer and
heterogeneous-cluster (H100-prefill / A100-decode) results.
- See also: DistServe, GreenLLM. See `20/02`. `[Splitwise]`

**TaiChi.** Unifying architecture (arXiv 2508.01989, Aug 2025) with a slider
between aggregated (chunked-prefill) and disaggregated regimes plus P-heavy /
D-heavy instances; goodput +9–47% vs aggregation, +29–77% vs disaggregation,
on Qwen2.5. Refutes the "always disaggregate" narrative.
- See also: PD disaggregation, BeyondBuzz. See `20/02`. `[TaiChi]`

**TetriInfer.** Three-pillar PD design (Jan 2024): chunked prompt to keep
prefill compute-saturated, PD disaggregation, and a two-level scheduler with
predicted resource usage to avoid decode hotspots; 38% fewer resources, −97%
TTFT, −47% JCT vs co-located baseline.
- See also: DistServe, Mooncake. See `20/02`. `[TetriInfer]`

**TransferEngine (Perplexity).** RDMA point-to-point library (open-sourced Nov
2025) that abstracts NIC differences (NVIDIA ConnectX-7, AWS EFA). Used in
disaggregated PD, RL weight transfer, and MoE dispatch/combine.
- See also: NIXL, Mooncake Transfer Engine. See `20/02`,
  `60/06`. `[Perplexity-WeightTransfer]`

**UCX.** Unified Communication X — vendor-agnostic point-to-point comm library
historically used for KV transport in TRT-LLM disagg. Increasingly displaced by
NIXL and Mooncake TE in 2025–2026 deployments.
- See also: NIXL, RDMA. See `20/02`.

---

## MoE

**Aux-loss-free balancing.** DeepSeek-V3 / "Loss-Free Balancing" technique that
adds a per-expert *bias* only to the top-k decision (gating values still use
raw affinity); replaces the auxiliary load-balance loss as the primary
balancer.
- See also: EPLB, Capacity factor. See `20/03`. `[Loss-Free-Balancing]`

**Capacity factor (C).** Multiplier on the per-expert token capacity
$C \cdot \lceil L K / E \rceil$; tokens routed past capacity are dropped.
Typical training values $C \in [1.25, 2.0]$; expert-choice routing achieves
$C = 1$ by construction.
- See also: Top-k routing, Expert-choice routing. See `20/03`. `[GShard-2020]`

**DeepEP.** DeepSeek's expert-parallel communication library (Open-Source Week
Day 2, Feb 2025): high-throughput (HT) kernels for training/prefill and
low-latency (LL) kernels for decode; intranode NVLink + internode RDMA;
SM-light, FP8 dispatch + BF16 combine. Purpose-built for group-limited routing.
- See also: NCCL-EP, DualPipe, EPLB. See `20/03`. `[DeepEP]`

**DeepGEMM.** DeepSeek's clean-and-efficient FP8 GEMM kernel library with
fine-grained tile/block scaling and grouped GEMMs over the M-axis (shared
N, K across experts); ~1550 TFLOP/s on H800. "Mega MoE" overlapped-comm fused
MoE GEMM was added in 2026.
- See also: DeepEP, FlashDMoE. See `20/03`. `[DeepGEMM]`

**DeepSeek-V3 / V3.2.** 671B total / 37B active MoE with 256 routed + 1 shared
expert per layer, top-8 + group-limited routing, FP8 mixed-precision training,
aux-loss-free balancing, MTP heads, and an 18-node EP32 prefill / 40-node EP320
decode reference deployment. V3.2 (Dec 2025) layers DeepSeek Sparse Attention
on top of the same MoE backbone.
- See also: MLA, MTP, EPLB, DSA. See `20/03`. `[DeepSeek-V3]`,
  `[DeepSeek-V3.2]`

**DualPipe (MoE).** See entry in *Distributed inference*. The DualPipe schedule
is the canonical comp-comm overlap recipe for cross-node EP all-to-all in
DeepSeek-style serving.

**EPLB (Expert Parallelism Load Balancer).** DeepSeek's redundant-experts
balancer: replicate hot experts and pack to GPUs; hierarchical (group-aware,
intra-node replication, for prefill) and global (regardless of groups, for
decode at large EP) policies. Integrated in vLLM Wide-EP and SGLang.
- See also: LPLB, Aux-loss-free balancing, Wide-EP. See `20/03`. `[EPLB]`

**Expert-choice routing.** Routing variant in which each expert picks its top-k
tokens (instead of each token picking top-k experts), eliminating dropped
tokens and giving perfect load balance by construction. Limited adoption in
autoregressive decoding.
- See also: Top-k routing, Capacity factor. See `20/03`. `[ExpertChoice]`

**FlashDMoE.** Single-kernel fused MoE that combines dispatch + GEMM + combine
to tighten compute/comm overlap and reduce memory traffic.
- See also: DeepGEMM, DeepEP. See `20/03`. `[FlashDMoE]`

**Group-limited routing.** DeepSeek-V3 routing constraint that limits a token's
top-K experts to at most $M$ groups (groups = nodes); halves inter-node fan-out
vs unconstrained routing. The structural reason DeepEP can use a hierarchical
NVLink/RDMA pattern.
- See also: Top-k routing, DeepEP. See `20/03`. `[DeepSeek-V3]`

**Janus.** Disaggregated attention/expert serving (Dec 2025) with adaptive
two-phase comm exploiting the NVLink/RDMA hierarchy and a microsecond
activation scheduler; up to 4.7× per-GPU throughput vs SOTA.
- See also: AFD, MegaScale-Infer, Step-3. See `20/03`. `[Janus]`

**LPLB.** Linear-programming-based EP load balancer (DeepSeek, 2025) that
replaces EPLB heuristic packing with an LP relaxation of the placement
problem.
- See also: EPLB. See `20/03`. `[LPLB]`

**MegaScale-Infer.** ByteDance + PKU (SIGCOMM 2025) AFD system: disaggregates
attention from FFN onto separate GPU pools, ping-pong micro-batch pipeline,
M2N comm library bypassing GPU↔CPU copies; 2.56× / 1.28× per-GPU decode vs
vLLM / TRT-LLM in homogeneous setups.
- See also: AFD, Janus, Step-3. See `20/03`. `[MegaScale-Infer]`

**Mixtral.** Mistral AI's open MoE family: 8x7B (47B / 13B active, top-2) and
8x22B (141B / 39B active). The first mainstream open-weight transformer MoE;
reset community baselines for inference.
- See also: MoE, Mixtral-Offload. See `20/03`. `[Mixtral-8x7B]`, `[Mixtral-8x22B]`

**MoE (Mixture of Experts).** Layer in which a router gates each token to a
small subset (top-k) of expert FFNs out of $E$ total. Decode is
memory-bandwidth bound on expert weights, motivating large-scale EP and
expert offloading. Frontier deployments (DeepSeek-V3, Kimi-K2, Qwen3-Next,
GPT-OSS) operate at <5% activation ratios.
- See also: EP, Top-k routing, AFD. See `20/03`. `[Shazeer-2017]`,
  `[GShard-2020]`

**NCCL-EP.** Unified expert-parallel communication API built on the NCCL
Device API: `ncclEpDispatch` / `ncclEpCombine` primitives with LL mode for
decode and HT mode for prefill/training. The convergence point for MoE
collectives in the standard collective library.
- See also: DeepEP, NCCL. See `20/03`. `[NCCL-EP]`

**Shared expert.** Always-active expert (DeepSeekMoE) that captures common
knowledge; routed experts then specialize. Used by DeepSeek-V2/V3 and
Llama-4-Maverick; Qwen3 dropped shared experts in favor of more routed
experts. The choice is contested empirically.
- See also: DeepSeekMoE, MoE. See `20/03`. `[DeepSeekMoE]`

**Step-3.** StepFun's 321B VLM (Aug 2025) with Multi-Matrix Factorization
Attention (MFA) and Attention-FFN Disaggregation; reports 4039 tok/s/GPU
decode under 50 ms TPOT SLA on Hopper.
- See also: AFD, MegaScale-Infer, Janus. See `20/03`. `[Step-3]`

**Top-k routing.** Each token sends activations to its top-$K$ experts by
gating affinity (typical $K=2$, $K=4$, $K=8$); the dispatched activations are
combined back via the combine all-to-all weighted by gate values.
- See also: MoE, Capacity factor, Group-limited routing. See `20/03`.

**Tutel.** Adaptive MoE library (Microsoft, MLSys 2023) with adaptive
parallelism switching and pipelining; still maintained in 2026 for
GPT-OSS / DeepSeek / Kimi-K2 / Qwen3 with FP8/NVFP4/MXFP4 paths.
- See also: MegaBlocks, DeepEP. See `20/03`. `[Tutel]`

---

## LoRA and multi-tenant

**AdaFuse.** Custom CUDA kernel (March 2026) that fuses adapter switching by
merging selected adapters' parameters into the backbone in one pass; reduces
adapter overhead from 250–950% to ~29%.
- See also: zFLoRA, BGMV. See `40/01`.

**BGMV (Batched Gather Matrix-Vector).** Punica's per-request gather kernel:
each request in the batch indexes a different adapter from a stacked tensor
$\mathcal{A}$, fused into one decode-shaped GEMM. Optimized for decode (one
token per request, large batch, many distinct adapters).
- See also: SGMV, Punica, Triton shrink/expand. See `40/01`. `[Punica]`

**dLoRA.** OSDI 2024: per-batch credit-based decision of whether to merge or
unmerge an adapter, plus migration of requests + adapters across replicas;
up to 1.8× lower latency vs S-LoRA.
- See also: S-LoRA, Llumnix. See `40/01`. `[dLoRA]`

**DoRA.** Weight-Decomposed Low-Rank Adaptation: decomposes $W$ into magnitude
+ direction and applies LoRA on the direction. Mergeable, no inference
overhead; serving target alongside vanilla LoRA.
- See also: LoRA, IA³, VeRA. See `40/01`. `[DoRA]`

**FastLibra / ELORA.** HPCA 2026 system that treats LoRA cache and KV cache
as a single dependency-aware unified pool with one swap cost model; −63% TTFT,
−40% TPOT vs SOTA.
- See also: S-LoRA, P-LoRA. See `40/01`.

**LoRA.** Low-Rank Adaptation: $y = xW + s \cdot xAB$ with
$A \in \mathbb{R}^{H_1 \times r}$, $B \in \mathbb{R}^{r \times H_2}$, scaling
$s = \alpha/r$. Mergeable single-tenant; multi-tenant serving keeps adapters
unmerged and uses BGMV/SGMV-style kernels to batch heterogeneous adapters.
- See also: DoRA, QLoRA, Punica. See `40/01`. `[LoRA]`

**LoRAX (LoRA eXchange).** Predibase's open-source production multi-LoRA
server (Nov 2023): heterogeneous continuous batching plus async adapter
exchange between GPU and CPU.
- See also: S-LoRA, Punica. See `40/01`,
  `80/06`. `[LoRAX]`

**LoraHub.** Coefficient-weighted composition of multiple trained LoRAs for
unseen tasks (COLM 2024); reference for the "multi-LoRA composition" serving
pattern. Subsumed in serving by LoGo and K-Merge.
- See also: K-Merge, LoGo. See `40/01`. `[LoraHub]`

**P-LoRA.** Predictive-LoRA (Dec 2025): LSTM traffic predictor + page-based
adapter memory manager; 1.52× throughput vs S-LoRA, 35% TTFT reduction.
Targets the cold-start tail by *predicting* hot adapters.
- See also: ServerlessLoRA, Toppings. See `40/01`.

**Punica.** UW/Duke serving system (MLSys 2024) that introduced BGMV and
SGMV — the first kernels to batch heterogeneous-adapter requests in one
launch; 12× throughput vs SOTA at the time.
- See also: BGMV, SGMV, S-LoRA. See `40/01`. `[Punica]`

**QLoRA.** 4-bit NormalFloat (NF4) + double-quant + paged optimizers; defines
the "4-bit base + bf16 adapter" recipe inherited by quantized-base multi-tenant
LoRA serving.
- See also: LoRA, NF4. See `40/01`,
  `10/04`. `[QLoRA]`

**ServerlessLoRA.** Pre-loads LoRA artifacts and shares the backbone LLM across
isolated functions to remove ~99% redundant weight duplication; up to 86% TTFT
and 89% cost reduction in serverless settings.
- See also: P-LoRA, AlpaServe. See `40/02`.

**SGMV (Segmented Gather Matrix-Vector).** Punica kernel that groups requests
sharing an adapter into segments, turning each segment into a regular GEMM
with intra-segment Tensor-Core utilization. Optimized for prefill (where many
tokens share an adapter); SGLang's `csgmv` is a chunked variant with
`max_lora_chunk_size`.
- See also: BGMV, Punica, S-LoRA. See `40/01`. `[Punica]`

**S-LoRA.** Berkeley/Stanford/SJTU (MLSys 2024): Unified Paging memory pool
(adapters + KV in one allocator), heterogeneous-rank kernels, tensor-parallel
adapter strategy. Up to 4× over vLLM-naive at orders of magnitude more
concurrent adapters; established the dominant multi-tenant LoRA architecture.
- See also: Punica, Unified Paging. See `40/01`. `[S-LoRA]`

**Triton shrink/expand (vLLM V1).** vLLM V1's replacement for SGMV/BGMV: tuned
Triton kernels that compute the LoRA shrink ($H_1 \to r$) and expand
($r \to H_2$) GEMMs with adapter gather fused in. Retired Punica BGMV/SGMV
in PRs #14685 / #13096 / #13110.
- See also: BGMV, SGMV. See `40/01`, `80/01`.

**zFLoRA / Fused-FLoRA.** Adapter forms (Samsung EMNLP 2025; arXiv 2511.00050)
that compile to ~zero added latency on H100 and on-device NPUs by fusing the
adapter projections into adjacent base GEMMs. Limited to single-adapter /
homogeneous-task batches; complementary to BGMV/SGMV multi-tenant kernels.
- See also: AdaFuse. See `40/01`.

---

## Multi-tenant and fairness

**AlpaServe.** OSDI 2023 paper establishing that model parallelism is
justified for *multiplexing* multiple models even when each fits on one GPU;
up to 10× higher request rate or 6× more burst tolerance. Backbone reasoning
for multi-model serving and LLM-suite gateways.
- See also: Llumnix, Aegaeon. See `40/02`. `[AlpaServe]`

**Aegaeon.** SOSP 2025: token-granularity model auto-scaling for multi-tenant
GPU pools; reduces 1192 GPUs to 213 in a real model marketplace and supports
up to 7 models concurrently per GPU.
- See also: AlpaServe, AIBrix autoscaler. See `40/02`. `[Aegaeon-SOSP25]`

**Andes.** Token-granularity preemptive scheduler (2024) driven by a
per-request QoE function (TTFT + token cadence vs reading speed) plus a
client-side smoothing buffer.
- See also: VTC, FastServe. See `40/03`. `[Andes]`

**DLPM / D²LPM.** Deficit Longest Prefix Match (2025): extends VTC-style
fairness with prefix-cache locality; up to 2.87× throughput vs VTC and 7.18×
lower latency vs SOTA distributed schedulers, fairness preserved.
- See also: VTC, Equinox. See `40/03`. `[DLPM]`

**Equinox.** Holistic fair scheduler (2025) with a dual-counter framework
(User Fairness + Resource Fairness) and a Mixture-of-Prediction-Experts
predictor; 1.3× throughput, 60% lower TTFT, 13% higher fairness vs VTC.
- See also: VTC, DLPM. See `40/03`. `[Equinox]`

**FastServe.** Skip-join multi-level feedback queue for token-level preemption
(2023); the first serious attack on head-of-line blocking in LLM serving.
- See also: VTC, Llumnix. See `10/03`. `[FastServe]`

**JITServe.** SLO-aware scheduler (NSDI 2026) that handles imprecise
length/dependency information and gradually relaxes conservatism as generation
reveals truth; 1.4–6.3× goodput.
- See also: SOLA, SLOs-Serve. See `40/03`,
  `10/03`. `[JITServe]`

**Llumnix.** OSDI 2024: live KV-cache migration of running requests across
instances pipelined with decode compute; the canonical "OS-style context
switch" of LLM serving and the foundation for migration-based fairness/SLO
mechanisms.
- See also: Aegaeon, AlpaServe. See `40/02`,
  `10/03`. `[Llumnix]`

**LTR (Learning-to-Rank scheduling).** NeurIPS 2024: predict pairwise *ranks*
rather than output lengths to approximate SJF; 2.8× chatbot latency reduction,
6.5× synthetic-data-gen throughput.
- See also: VTC, Andes. See `10/03`. `[LTR]`

**SLOs-Serve.** MLSys 2025 multi-stage application-level SLO scheduler with
admission control + DP allocation across stages; 2.2× per-GPU capacity over
prior art.
- See also: SOLA, JITServe. See `40/03`. `[SLOs-Serve]`

**SOLA.** State-aware MLSys 2025 scheduler that rebalances TTFT vs TPOT
distribution per request and across the system; SLO attainment
45.5% → 99.4% on identical hardware.
- See also: SLOs-Serve, JITServe. See `40/03`. `[SOLA]`

**VTC (Virtual Token Counter).** OSDI 2024 fair LLM scheduler with a
token-cost virtual counter per client and a proven 2× upper bound on
service-difference under work-conservation; a new client inherits the minimum
counter to avoid starvation. Foundation for DLPM, Equinox, and AIBrix's
fairness routing.
- See also: DLPM, Equinox. See `40/03`. `[VTC]`

---

## Cluster systems

**AIBrix.** ByteDance's Kubernetes-native vLLM control plane (paper Feb 2025):
distributed KV cache pool, LoRA-aware routing, KV-aware autoscaler,
StormService CRD for P/D lifecycle, KV event subscription via messaging
middleware. Joined the vLLM project as a sibling.
- See also: vLLM, llm-d, Dynamo. See `50/01`,
  `80/05`. `[AIBrix-paper]`

**DualMap.** Two-hash power-of-two-choices routing scheme that argues PoT
generalizes naturally to cache-affinity routing; cited by vLLM router's PoT
policy.
- See also: KV-aware routing, EPP. See `50/01`. `[DualMap]`

**Dynamo (NVIDIA).** Distributed inference framework (GTC 2025): Rust Smart
Router, Planner, KVBM (multi-tier KV block manager), and NIXL transport.
Tightly integrates TRT-LLM, vLLM, and SGLang as engines; up to 30× throughput
on DeepSeek-R1 / GB200 NVL72 vs co-located baselines.
- See also: KVBM, NIXL, llm-d. See `50/01`,
  `80/05`. `[Dynamo-launch]`

**Endpoint Picker (EPP).** Sidecar invoked via Envoy `ext-proc` to make
per-request scheduling decisions in the Inference Gateway API model. The
reference EPP implements Discover → Filter → Score → Select with prefix
affinity, KV utilization, queue depth, and LoRA affinity scores.
- See also: GIE, KV-aware routing. See `50/01`. `[GIE-blog]`

**Envoy AI Gateway.** First CNCF-backed AI gateway (Tetrate + Bloomberg, Feb
2025), built on Envoy Gateway. Provides token-level flow control, unified
provider access, and upstream auth.
- See also: LiteLLM, Portkey, GIE. See `50/01`. `[EnvoyAI-launch]`

**GIE (Gateway API Inference Extension).** Kubernetes SIG-Network /
WG-Serving project that defines `InferencePool`, `InferenceObjective` /
`InferenceModel`, and the Endpoint Picker Protocol; conformance shipped in
Envoy AI Gateway, Istio 1.27+, kgateway, NGINX Gateway Fabric, GKE Gateway,
and ACK Gateway.
- See also: EPP, InferencePool, InferenceObjective. See `50/01`. `[GIE-blog]`

**HeteroScale.** ByteDance autoscaling study (arXiv 2508.19559, Aug 2025) at
tens-of-thousands-of-GPU scale: establishes decode TPS as the most robust
autoscaling signal and explicitly debunks decode-GPU-utilization. +26.6 pp
average GPU utilization in production.
- See also: AIBrix autoscaler, KV-aware autoscaling. See `20/02`,
  `50/02`. `[HeteroScale]`

**InferenceObjective / InferenceModel.** Model-owner contract CRD in the
Inference Gateway API model that declares the SLO, traffic split, and
priority for a model served by an `InferencePool`.
- See also: GIE, InferencePool. See `50/01`.

**InferencePool.** Platform-owned pod-pool CRD in the Inference Gateway API
model. Reached **v1** in 2025; selected by the gateway and dispatched to via
EPP.
- See also: GIE, EPP. See `50/01`.

**KServe.** CNCF Incubating model-serving CRD framework (joined CNCF Nov
2025); v0.16+ adds `LLMInferenceService` for OpenAI-compatible LLM workloads
and composes with llm-d EPP/disagg sidecars.
- See also: GIE, llm-d, Open Inference Protocol. See `50/01`. `[KServe-OIP]`

**KV-aware routing.** Routing policy that scores candidate pods by
prefix-cache match length, KV utilization, queue depth, and LoRA-adapter
affinity; the canonical EPP scoring formula. Implemented in llm-d, vLLM
router, AIBrix, Dynamo Smart Router, and SGLang sgl-router.
- See also: EPP, RadixAttention, GIE. See `50/01`.

**KVBM (KV Block Manager).** NVIDIA Dynamo's hierarchical KV manager: G1
device, G2 CPU local/remote, G3 SSD, G4 remote storage. Sits between engine
runtime connectors (TRT-LLM/vLLM/SGLang) and NIXL transport.
- See also: Dynamo, NIXL, KV cache. See `50/01`,
  `30/02`. `[Dynamo-KVBM]`

**LiteLLM.** Open-source Python proxy fronting 100+ providers; virtual keys,
spend tracking, guardrails, load balancing. The de facto OSS client-facing
gateway choice.
- See also: Portkey, Bifrost. See `50/01`. `[LiteLLM]`

**llm-d.** Kubernetes-native distributed inference stack (Red Hat / Google /
IBM / CoreWeave / NVIDIA, May 2025; CNCF Sandbox). vLLM as default engine,
EPP for routing, optional disagg sidecars, LMCache or NIXL for hierarchical KV;
v0.5 adds scale-to-zero, UCCL transport, and active-active HA.
- See also: vLLM, AIBrix, Dynamo. See `50/01`,
  `80/05`. `[llm-d-launch]`

**Modal.** Container-on-demand serverless GPU platform with fast cold start
(custom snapshot/sandbox runtime); auto-scaling with minimum-replicas-zero by
default. Often the comparison point for "managed" tier in benchmarks.
- See also: BentoML, Replicate, Cloudflare AI Gateway. See `50/01`,
  `80/06`. `[Modal]`

**OpenTelemetry GenAI.** Semantic conventions for generative-AI spans, metrics,
and events: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`,
etc., plus provider conventions for Anthropic, AWS Bedrock, Azure AI Inference,
OpenAI, MCP. In Development as of v1.37; opt-in via
`OTEL_SEMCONV_STABILITY_OPT_IN`.
- See also: vLLM metrics, KServe. See `50/03`. `[OTel-GenAI]`

**Portkey.** Managed AI gateway with metadata-tagged per-tenant cost tracking,
observability, guardrails, and MCP routing.
- See also: LiteLLM, Bifrost, Cloudflare AI Gateway. See `50/01`.

**Ray Serve LLM.** Anyscale's LLM-serving framework: LLMServer in three modes
(isolated / coordinated within deployment / coordinated across deployments),
OpenAiIngress, and engine-agnostic LLMEngine protocol (vLLM, SGLang). Built-in
prefix-aware routing, custom autoscaling, and async inference paths.
- See also: vLLM, SGLang, BentoML. See `50/01`. `[RayServe-LLM-arch]`

**SGLang.** Frontend DSL + backend runtime that originated RadixAttention and
the cache-aware load balancer; default attention backends are FA3 on Hopper and
TRTLLM-MHA / FlashInfer on Blackwell. Powers DeepSeek-V3/V3.2 and Kimi-K2
production serving via Mooncake.
- See also: vLLM, RadixAttention, HiCache. See `80/02`.

**TensorRT-LLM (TRT-LLM).** NVIDIA's inference engine with dual frontends
(legacy TensorRT plan + PyTorch `_torch`), shared C++ Batch Manager / KV-cache /
Executor. Ships ConfigurableMoE (Backend × Communication × EPLB), `trtllm-gen`
FMHA cubins, XQA, MTP/EAGLE-3/Medusa/ReDrafter spec-dec, and three disagg
paths (`trtllm-serve`, Dynamo, Triton BLS).
- See also: XQA, FMHA, MMHA. See `80/03`.

**TGI (Text Generation Inference).** Hugging Face's serving framework; archived
as read-only March 2026, with HF officially recommending vLLM/SGLang.
- See also: vLLM, SGLang. See `80/06`.

**UCCL.** Host-resident software transport stack used in llm-d 0.5 inside NIXL
to replace UCX in some paths; reports 2.4× more congestion resilience vs UCX
baseline.
- See also: NIXL, UCX. See `50/01`. `[UCCL-llm-d]`

**vLLM.** Open-source serving engine (UC Berkeley → community / Red Hat
stewardship) introduced with PagedAttention; V1 (Jan 2025) re-architects the
EngineCore / Scheduler / KVCacheManager / Worker / ModelRunner / Executor
boundaries, defaults FA4 on SM100+ / FA3 on SM90 / FA2 on SM80, and ships
piecewise CUDA graphs, Triton shrink/expand LoRA, and KV connector V1.
- See also: PagedAttention, V1 prefix cache. See `80/01`. `[vLLM-V1-Blog]`

**vLLM Production Stack.** Reference Kubernetes deployment of vLLM with
prefix-aware router, LMCache for KV offload, autoscaler, and observability.
- See also: vLLM, LMCache, AIBrix. See `50/01`,
  `80/05`. `[vLLM-prod-stack]`

**vLLM router.** Rust-based vLLM-native router (Dec 2025): PoT, consistent
hashing, round-robin, random; native P/D-disagg routing; circuit breakers,
retries, Prometheus metrics. +25% throughput vs llm-d on Llama 3.1 8B in
vendor-reported numbers.
- See also: sgl-router, EPP. See `50/01`. `[vLLM-router]`

---

## Heterogeneous inference

**FlexGen.** ICML 2023: GPU+CPU+disk offload via LP-derived schedule for
tensor placement; first to demonstrate 1 token/s OPT-175B on a 16 GB GPU. The
foundational ancestor of LMCache, AIBrix, ALISA, and KTransformers.
- See also: ZeRO-Inference, PowerInfer, KTransformers. See `20/05`,
  `30/02`. `[FlexGen]`

**GreenLLM.** Sustainability-oriented heterogeneous serving (Dec 2024):
PD-disagg with old GPUs handling decode; spec-decode with old GPU drafting +
new GPU verifying. Up to 40.6% emissions reduction at >90% SLO compliance.
- See also: PD disaggregation, Speculative decoding. See `20/05`. `[GreenLLM]`

**Helix.** ASPLOS 2025 max-flow + MILP scheduler that jointly assigns model
layers and routes requests across heterogeneous GPUs and links; up to 3.3×
throughput / −66% prefill / −24% decode latency on 24–42 node mixed clusters.
The canonical heterogeneous-inference paper.
- See also: HexGen, Mélange. See `20/05`. `[Helix]`

**HexGen / HexGen-2.** Asymmetric TP+PP partitioning over heterogeneous GPUs
and links; HexGen-2 (ICLR 2025) adds disaggregated PD on top with
graph-partitioning + max-flow placement.
- See also: Helix, Tessera. See `20/05`. `[HexGen]`,
  `[HexGen-2]`

**KTransformers.** SOSP 2025: AMX-aware kernels and arithmetic-intensity-aware
kernel selection for CPU/GPU hybrid MoE inference; runs DeepSeek-R1 671B on a
single 24 GB 4090D + DRAM at ~14 tok/s decode. Pivoted to a kernel library
called by SGLang in Oct 2025.
- See also: PowerInfer, MoE-Infinity, Fiddler. See `20/05`,
  `80/06`. `[KTransformers]`

**Mélange.** UC Berkeley + Microsoft cost-allocation framework (April 2024)
that picks the cheapest GPU type per (request size, request rate, SLO);
9–77% savings in chat workloads.
- See also: Cauchy, Demystify-Cost-Efficiency. See `20/05`. `[Mélange]`

**Parallax.** Modern Petals descendant (Sept 2025): two-phase scheduler with
DP+water-filling for layer placement and per-request DAG path selection;
3.1× E2E latency / 5.3× ITL improvement.
- See also: Petals, Decentralized inference. See `20/05`. `[Parallax]`

**Petals.** BitTorrent-style swarm inference (2022): cross-Internet pipeline
parallel; conceptual ancestor of Parallax.
- See also: Parallax. See `20/05`. `[Petals]`

**PowerInfer.** SOSP 2024: hot/cold neuron split via power-law activation;
hot on GPU, cold on CPU. 11.69× over llama.cpp on RTX 4090; reaches 82% of
A100 throughput on OPT-30B.
- See also: KTransformers, DejaVu. See `20/05`. `[PowerInfer]`

**Tessera.** Heterogeneous-GPU serving (April 2026) with PTX-kernel-granularity
disaggregation; offline MILP for batch + online policy for serving. A
heterogeneous pair (e.g., A100+L40S) beats two homogeneous high-end GPUs in
some regimes.
- See also: Helix, Hetis. See `20/05`. `[Tessera]`

---

## Hardware

**Blackwell.** NVIDIA's 2024–2026 GPU generation: B100, B200 (192 GB HBM3e,
8 TB/s, 20 PFLOPS FP4 / 10 PFLOPS FP8), GB200 NVL72 (rack-scale, 72 B200 + 36
Grace, 130 TB/s aggregate NVLink), GB300 / Blackwell Ultra (288 GB HBM3e,
15 PFLOPS FP4, 1.4 kW). Native NVFP4 / MXFP4 / MXFP6 / MXFP8 tensor cores.
- See also: NVFP4, NVL72, tcgen05, TMEM. See `70/01`.

**Cerebras WSE-3 / CS-3.** Wafer-scale 4T-transistor die with 900,000 AI
cores, 44 GB on-chip SRAM, 21 PB/s on-chip BW. CS-3 system reaches 8× H200
on Llama-3.1 70B in vendor benchmarks; up to 256 EFLOPS FP16 in 2,048-CS-3
clusters.
- See also: WaferLLM, SambaNova SN40L. See `70/04`.

**CPO (Co-Packaged Optics).** Optical engines integrated on the switch package
to reduce power and signal-integrity loss vs pluggable optics. NVIDIA Quantum-X
Photonics IB and Spectrum-X Photonics Ethernet were announced at GTC 2025;
Google has used MEMS-based Optical Circuit Switching at TPU pod boundaries
since v4.
- See also: NVLink, Spectrum-X, OCS. See `70/05`.

**Furiosa RNGD.** Korean ASIC: Tensor Contraction Processor (TCP) architecture,
48 GB HBM3 at 1.5 TB/s, 256 MB on-chip SRAM, 180 W. Anchors LG / SK enterprise
deployments.
- See also: ASICs, Etched Sohu. See `70/04`.

**GDDR.** Graphics-DDR memory used in Rubin CPX (128 GB GDDR7) for the
prefill-specialized GPU. Rationale: prefill is compute-bound, so peak HBM
bandwidth is over-provisioned vs decode.
- See also: HBM, Rubin CPX. See `70/01`.

**GDS (GPUDirect Storage).** Storage subsystem ↔ GPU memory direct path,
bypassing system memory. Required for high-throughput KV-cache offload to NVMe
or parallel filesystems (VAST, WEKA).
- See also: GPUDirect RDMA, NVMe-resident KV. See `70/05`,
  `30/02`.

**GPUDirect RDMA.** NIC ↔ GPU memory direct path, bypassing host CPU/RAM;
~10× latency / 2–8× bandwidth vs CPU-mediated. Cooperation between NVIDIA
driver and Mellanox/ConnectX OFED.
- See also: GDS, NVSHMEM. See `70/05`.

**Groq LPU.** SRAM-only deterministic-dataflow inference accelerator (TSP);
plesiosynchronous chip-to-chip clocking. NVIDIA licensed Groq's IP in Dec 2025
and productized it as **NVIDIA Groq 3 LPX** (GTC 2026): 256 LPUs per rack, 315
PFLOPS FP8, 128 GB SRAM, 40 PB/s on-chip BW, 640 TB/s scale-up — pairs with
Rubin GPUs as a decode co-processor.
- See also: Heterogeneous inference, Rubin CPX. See `70/04`.

**HBM (HBM3 / HBM3e / HBM4).** High-Bandwidth Memory used by datacenter AI
GPUs and ASICs. HBM3 up to 819 GB/s/stack and 24 GB; HBM3e up to 1.2 TB/s and
36 GB (H200, B200, MI350, TPU v7); HBM4 up to 2.0–3.3 TB/s and 64 GB
(Rubin R100, MI400).
- See also: GDDR, Memory bandwidth. See `70/01`.

**Hopper.** NVIDIA's H100 / H200 GPU generation: 4th-gen Tensor Cores, native
FP8 (E4M3, E5M2), Transformer Engine v1, TMA, distributed-shared-memory
clusters. H200 keeps the same die but upgrades to 141 GB HBM3e at 4.8 TB/s.
- See also: TMA, WGMMA, FP8. See `70/01`.

**InfiniBand (NDR / XDR).** NVIDIA Quantum scale-out fabric. NDR is 400 Gb/s
(Quantum-2, ConnectX-7); XDR is 800 Gb/s (Quantum-X800 Q3400-RA, ConnectX-8) and
ships in production with HGX B300 / GB300 NVL72.
- See also: Spectrum-X, Ultra Ethernet, NVLink. See `70/05`.

**Maia (Microsoft).** In-house AI accelerator. Maia 100 (TSMC N5, 64 GB HBM2E,
1.8 TB/s, deployed for OpenAI workloads). Maia 200 (Jan 2026, deployed in
Azure US Central): TSMC 3 nm, 216 GB HBM3e at 7 TB/s, native FP8/FP4, 750 W.
- See also: ASICs, MTIA. See `70/03`.

**MTIA (Meta Training and Inference Accelerator).** Meta's in-house ASIC
co-developed with Broadcom. MTIA 300 / 400 in production for ranking and
gen-AI; MTIA 450 / 500 on the 2027 roadmap with HBM4-class bandwidth.
- See also: Maia, Trainium, ASICs. See `70/03`.

**NVLink / NVSwitch.** NVIDIA's GPU-domain scale-up fabric. NVLink 4 = 600/900
GB/s (A100/H100), NVLink 5 = 1.8 TB/s/GPU (Blackwell, NVSwitch 4); NVL72
aggregates to 130 TB/s and treats 72 GPUs as one NVLink domain. NVLink-C2C =
Grace ↔ Hopper/Blackwell coherent at 900 GB/s.
- See also: NVLink Fusion, NVL72. See `70/01`, `70/05`.

**NVLink Fusion.** Licensable NVLink IP (announced May 2025) for partner CPUs
and ASICs (MediaTek, Marvell, Alchip, Astera Labs, Synopsys, Cadence; later
Fujitsu, Qualcomm). NVIDIA's response to the custom-silicon trend — be the
fabric if not the compute.
- See also: NVLink, UALink. See `70/01`,
  `70/05`.

**NVL72.** GB200 / GB300 rack-scale system with 72 Blackwell GPUs + 36 Grace
CPUs treated as one NVLink domain (130 TB/s aggregate). Makes EP-72 an
"intra-node" operation and is the canonical Wide-EP target.
- See also: NVLink, GB200, GB300. See `70/01`.

**NVSHMEM.** PGAS-style one-sided GPU-to-GPU communication over NVLink + IB.
Used in DeepEP and other MoE all-to-all kernels.
- See also: NCCL, GPUDirect RDMA. See `70/05`.

**Rubin / Rubin CPX / Rubin Ultra.** NVIDIA's post-Blackwell roadmap. Rubin
R100: 288 GB HBM4, ~50 PFLOPS FP4. Vera CPU: 88 Olympus cores, 1.5 TB LPDDR5X.
Vera-Rubin NVL72/144: 5× inference / 3.5× training vs Blackwell. **Rubin CPX**:
single-die 30 PFLOPS NVFP4 with 128 GB GDDR7 instead of HBM, purpose-built for
the prefill / long-context phase. **Rubin Ultra GR300 NVL576**: 4-chiplet GPU,
1 TB HBM4 per GPU, 576 dies / rack, 600 kW per rack, H2 2027.
- See also: Blackwell, NVL72, GDDR. See `70/01`.

**SambaNova SN40L.** Reconfigurable-Dataflow Architecture chiplet with three-tier
memory (PMU SRAM + HBM + DDR) for "Composition of Experts" MoE-like serving.
- See also: ASICs. See `70/04`.

**tcgen05 / TMEM.** Blackwell-specific tensor-core ISA (`tcgen05.mma`) and
Tensor Memory (TMEM) backing storage; the substrate for FA4 and Blackwell MoE
GEMMs. SM120/SM121 (consumer Blackwell) lacks these and falls back to
`mma.sync.aligned.block_scale`.
- See also: Blackwell, FlashAttention-4, WGMMA. See `70/01`.

**TMA (Tensor Memory Accelerator).** Hopper-introduced bulk-async global ↔
shared movement engine; the basis for FlashAttention-2-Hopper / FA-3
producer/consumer pipelines.
- See also: WGMMA, Warp specialization, FlashAttention-3. See `70/01`,
  `10/01`.

**TPU (v5p / v6e Trillium / v7 Ironwood).** Google's tensor processing unit
generations. v5p was training-focused; v6e Trillium added inference perf
(4.7× v5e); v7 Ironwood (192 GB HBM3e/chip, 4,614 FP8 TFLOPS, 9,216-chip
superpod, 9.6 Tb/s ICI) is the first TPU branded for inference and is the
substrate for Anthropic's Google deployment.
- See also: OCS, JAX, Pathways. See `70/03`.

**Trainium2 / Trainium3.** AWS's in-house AI accelerator generations. Trainium2
(GA Dec 2024): 8 NeuronCores, 96 GiB HBM, 2.9 TB/s, 1.3 PFLOPS dense FP8.
Trainium2 UltraServer = 64 chips via NeuronLink. Trainium3 (re:Invent 2025
announce): 2.52 PFLOPS FP8/chip, 144 GB HBM3e, MXFP8/MXFP4 native.
- See also: NeuronLink, MTIA, Maia. See `70/03`.

**UALink.** Open scale-up alternative to NVLink (AMD-led; debuting in AMD
Helios H2 2026). UALink 1.0 tunnels over Ethernet using Broadcom Tomahawk 6 in
initial implementations.
- See also: NVLink Fusion, Ultra Ethernet. See `70/05`.

**Ultra Ethernet (UEC) 1.0.** Open Ethernet-native AI/HPC stack (released
June 2025). NICs, switches, optics, cables; packet delivery sub-layer (PDS)
plus future trimming + semantic sub-layer (SES) and congestion mgmt.
Counter-narrative to proprietary IB and NVLink.
- See also: Spectrum-X, UALink. See `70/05`.

**WGMMA (Warpgroup MMA).** Hopper-introduced asynchronous warpgroup
matrix-multiply-accumulate instruction; the matmul half of the FA-3
producer/consumer pipeline. Replaced on Blackwell by `tcgen05.mma`.
- See also: TMA, tcgen05, FlashAttention-3. See `70/01`,
  `10/01`.

---

## Long context

**Hybrid (Mamba-Transformer) models.** Architectures alternating attention
layers and SSM (Mamba-2) layers — Granite 4 (9:1 SSM:attn), Falcon-H1 (parallel
attn+SSM), Jamba 1.5 (1:7 attn:SSM), Hymba (parallel hybrid heads). Cuts
long-context KV by an order of magnitude; optimal ratio is unsettled.
- See also: Mamba-2, Jamba, RWKV-7. See `20/04`.

**LongRoPE / LongRoPE2.** Microsoft's RoPE-extension recipes. LongRoPE
(ICML 2024): evolutionary search over per-dim RoPE rescaling to reach 2M.
LongRoPE2 (Feb 2025): hypothesizes that high RoPE dims are under-trained and
uses mixed-context training; reaches 128K with 10B tokens (vs Meta's 800B).
- See also: PI, NTK-aware, YaRN. See `20/04`. `[LongRoPE]`,
  `[LongRoPE2]`

**MInference.** Microsoft NeurIPS 2024 spotlight: three head-pattern templates
(A-shape, Vertical-Slash, Block-Sparse) with offline classification + online
sparse index; 10× prefill speedup at 1M on A100.
- See also: Quest, NSA, DSA. See `20/04`. `[MInference 1.0]`

**NTK-aware (RoPE).** Community-derived RoPE extension that spreads
interpolation pressure non-uniformly across frequency bands; formalized in
YaRN.
- See also: PI, YaRN. See `20/04`.

**PI (Position Interpolation).** Linear down-scaling of position indices for
RoPE extension (Meta, 2023); the first practical "extend cheaply" recipe.
- See also: NTK-aware, YaRN, LongRoPE. See `20/04`. `[PI]`

**RoPE (Rotary Position Embedding).** Position encoding by per-dimension
rotation: $q'_m = R_{\theta,m} q_m$, $k'_n = R_{\theta,n} k_n$, with relative
position recoverable in the inner product. Used by virtually every modern
open-weights LLM.
- See also: PI, YaRN, LongRoPE2. See `20/04`. `[RoFormer]`

**SCBench.** ICLR 2025 KV-cache-centric long-context benchmark: evaluates
compression / retrieval / loading across the full KV lifecycle (multi-turn,
shared-prefix). Argues sub-O(n) memory methods break in multi-turn; dynamic
sparsity beats static.
- See also: RULER, LongBench v2. See `20/04`. `[SCBench]`

**YaRN.** Adds attention-temperature scaling and "NTK-by-parts" to PI/NTK-aware
(ICLR 2024); SOTA RoPE-extension recipe through 2024 and still a baseline in
2025–2026.
- See also: NTK-aware, LongRoPE2. See `20/04`. `[YaRN]`

---

## Attention variants

**GQA (Grouped-Query Attention).** Several query heads share one KV head;
reduces KV bytes by the group factor. Used by Llama-3, Mistral, Qwen2.5, and
most modern dense models.
- See also: MQA, MLA. See `30/03`.

**Hybrid KV cache.** KV-cache allocator that handles models with mixed full /
sliding-window / SSM / no-KV layers (Gemma 3, GPT-OSS, Granite 4) by
maintaining different per-layer slot sizes and life cycles.
- See also: Sliding-window, MLA, Hybrid models. See `30/03`,
  `20/04`.

**MLA (Multi-head Latent Attention).** DeepSeek-V2/V3 attention variant that
down-projects K and V to a shared latent $c^{KV}$ of dim $d_c$ (~512) plus a
per-head decoupled-RoPE key $k^R$ of dim $d^R_h$ (~64); the latent vector *is*
the cache. ~93% KV reduction vs equivalent MHA. Requires absorbed Q-projection
at decode.
- See also: GQA, MQA, FlashMLA. See `30/03`,
  `10/01`. `[MLA-V2]`

**MQA (Multi-Query Attention).** All query heads share one KV head; the
extreme form of GQA. Used in early efficient designs and as the MQA-mode of MLA.
- See also: GQA, MLA. See `30/03`.

---

## Workload primitives

**ColBERT / ColBERTv2 / PLAID.** Late-interaction passage retrieval lineage:
ColBERT introduced per-token contextualized vectors; v2 compressed them ~10×;
PLAID (CIKM 2022) is the canonical late-interaction serving engine (up to 7×
GPU / 45× CPU latency reduction).
- See also: ColPali, Late interaction, RAG. See `60/04`,
  `60/05`. `[ColBERT-2020]`, `[ColBERTv2-2022]`, `[PLAID-2022]`

**ColPali / ColQwen.** Visual-document late-interaction retrieval (Illuin,
ICLR 2025): ViT patches as multi-vector embeddings, MaxSim aggregation;
ColQwen2/2.5 backs the same idea with Qwen2-VL. Drives multi-vector / "tensor"
index types in Vespa, Milvus, and Qdrant.
- See also: ColBERTv2, RAG. See `60/04`. `[ColPali-2024]`,
  `[ColQwen-2025]`

**Constrained decoding.** Restricting the next-token distribution to tokens
valid under a typed grammar (JSON / regex / CFG) using a precomputed bitmask.
~99% of mask entries are context-independent and looked up; ~1% require runtime
stack inspection. The default tool-calling and structured-output mechanism.
- See also: XGrammar, Outlines, llguidance. See `60/02`.

**FastVLM.** Apple CVPR 2025: hybrid FastViTHD encoder designed to emit fewer
tokens at high resolution; 85× faster TTFT than LLaVA-OneVision-0.5B and 7.9×
faster TTFT for a 7B variant. The "encoder→token-count" function is the
master TTFT knob in VLM serving.
- See also: VLM, EPD. See `60/03`. `[FastVLM-CVPR2025]`

**Granite 4.** IBM hybrid Mamba-2 + attention model family (Oct 2025) at 9:1
SSM:attn ratio; 3B / 7B-1B-active / 32B-9B-active. Cuts production RAM >70% on
long-context multi-session workloads.
- See also: Hybrid models, Mamba-2. See `20/04`.

**Hydragen.** See entry under *KV cache*. Cited from RAG / reasoning chapters
because shared-prefix kernels matter for sibling completions and shared
documents.

**Jamba.** AI21 hybrid Mamba/attention MoE; Jamba-1.5 has 1:7 attention:Mamba
interleave and 256K context with ~4 GB KV.
- See also: Hybrid models, Falcon-H1. See `20/04`. `[Jamba]`

**Late interaction.** Retrieval pattern that emits per-token (or per-patch)
embeddings and computes MaxSim aggregation at query time, capturing
fine-grained matches that a single-vector embedding loses.
- See also: ColBERT, ColPali, RAG. See `60/04`,
  `60/05`.

**llama.cpp.** GGML/GGUF-based CPU-first LLM inference framework with broad
hardware coverage (CUDA, Metal, Vulkan, ROCm, Hexagon HMX). Adds MXFP4 / NVFP4
ggml types and GGUF support for almost all open MoEs.
- See also: GGUF, MLX. See `80/06`.

**LMFE (lm-format-enforcer).** Token-mask-per-step constrained-decoder
integration that informed early vLLM/Mistral integrations; outpaced by
XGrammar / llguidance in 2024–2025.
- See also: Outlines, XGrammar, llguidance. See `60/02`. `[LMFE-2023]`

**Mamba-2.** Generalized Mamba SSM (ICML 2024) with structured state-space
duality; the SSM block in modern hybrid models.
- See also: Mamba, Mamba-3, RWKV-7, Hybrid models. See `20/04`. `[Mamba-2]`

**Mamba-3.** ICLR 2026 oral (arXiv:2603.15569). The latest canonical SSM
reference, generalizing Mamba-2's structured-state-space duality with
per-token data-dependent state transitions. Relevant for long-context and
attention-variants chapters as the SSM cell that newer hybrid stacks may
adopt.
- See also: Mamba-2, Hybrid models. See `20/04`, `30/03`. `[Mamba-3]`

**MLX / MLX-LM.** Apple's array framework with unified-memory CPU/GPU
execution; M5's Neural Accelerators add tensor-core-class kernels. mlx-lm is
the LLM-serving wrapper.
- See also: llama.cpp. See `80/06`.

**Outlines.** First widely-deployed FSM-based constrained decoder (.txt, July
2023); drove industry adoption of "regex / JSON-schema as a serving primitive"
and contributed jump-forward / coalescence.
- See also: XGrammar, llguidance, LMFE. See `60/02`. `[Outlines-2023]`

**RAG (Retrieval-Augmented Generation).** Pipeline of ingest → embed → index
→ retrieve (sparse + dense, hybrid) → rerank → generate. Production budget is
typically retrieval ≤50 ms, rerank ≤200 ms, generate ≤2 s for short answers.
- See also: HNSW, ColBERT, Reranker. See `60/04`.

**Reasoning model.** Class of LLM (o1/o3, DeepSeek-R1, Claude with extended
thinking, Gemini "thinking") that emits thousands to tens of thousands of
"thinking" tokens before the user-visible answer. Shifts the workload toward
long-decode and motivates branching, prefix-cache reuse for siblings, and
budget-control APIs.
- See also: GRPO, RLVR, Lookahead Reasoning. See `60/01`.

**Reranker.** Cross-encoder model that re-scores top-K candidates from the
first-stage retriever; adds 100–600 ms but boosts precision 10–30%. Cohere
Rerank, Jina Rerank v2/v3, and BGE-Reranker are the production references.
- See also: Late interaction, RAG. See `60/04`,
  `60/05`.

**RWKV-7 "Goose".** Generalized delta rule with vector-valued gating
(COLM 2025); constant memory + time per token; 3B SOTA in linear-time/
constant-memory regime.
- See also: Mamba-2, Hybrid models. See `20/04`. `[RWKV-7 "Goose"]`

**TEI (Text Embeddings Inference).** Hugging Face's reference open serving
stack for embeddings; batches by total token count rather than request count
to smooth GPU utilization across heterogeneous request lengths.
- See also: Infinity, Embeddings. See `60/05`. `[TEI-Repo]`

**Tree of Thought (ToT) / Graph of Thought (GoT) / AGoT.** Branching
reasoning structures that fan out a single prompt to many sibling
completions. ToT is tree-shaped; GoT generalizes to DAGs; AGoT (Feb 2025)
selectively expands DAG nodes. Drive prefix-cache reuse as the dominant
optimization in reasoning serving.
- See also: Reasoning model, RadixAttention. See `60/01`. `[ToT-2023]`,
  `[GoT-2023]`, `[AGoT-2025]`

**XGrammar.** MLSys 2025 constrained-generation engine; default backend in
vLLM, SGLang, TRT-LLM, and MLC-LLM. Up to 14× JSON-schema and 80–100× CFG
speedup over prior; ~40 µs CPU/token at 128K vocab.
- See also: llguidance, Outlines. See `60/02`. `[XGrammar-2024]`

---

## RL post-training

**AReaL.** Tsinghua + Ant Group asynchronous RL system (May 2025). Rollout
workers never block on training; modified PPO (decoupled-/staleness-enhanced)
handles stale samples. Reports 2.77× speedup over synchronous baselines at
matched accuracy.
- See also: veRL, slime, GRPO. See `60/06`. `[AReaL]`

**Awex.** Ant Group's ultra-fast weight-sync framework (open-sourced 2025):
cross-engine layout translation; 1T params synced in 6 s (RDMA) / 20 s
(NCCL) on thousand-GPU clusters.
- See also: Checkpoint Engine, NIXL, RL post-training. See `60/06`.

**Bitwise-Consistent RL.** Configuration that ensures bit-exact match between
vLLM rollouts and TorchTitan training, sidestepping importance-sampling
correction entirely (vLLM blog, Nov 2025).
- See also: Train-inference mismatch, FP16-RL, GRPO. See `60/06`.

**Checkpoint Engine.** Mooncake (Sept 2025) middleware to update vLLM/SGLang
weights from a remote training process; second-level updates of trillion-param
models. Now a transport in veRL 0.7.
- See also: Awex, NIXL, RL weight sync. See `60/06`.

**DAPO.** ByteDance Seed's GRPO extension (March 2025): Clip-Higher,
Dynamic Sampling, Token-Level Policy Gradient Loss, Overlong Reward Shaping;
50 on AIME-2024 with Qwen2.5-32B in half the steps of R1-Zero. Built on veRL.
- See also: GRPO, Dr.GRPO, veRL. See `60/06`. `[DAPO]`

**Dr.GRPO.** Sea AI Lab GRPO variant (March 2025) that removes the length and
std normalization terms; addresses a length-bias in GRPO; 43.3% AIME-2024
with 7B in 27 GPU-hours.
- See also: GRPO, DAPO. See `60/06`. `[Dr.GRPO]`

**FP16-RL.** Switching BF16→FP16 during RL fine-tuning largely eliminates
rollout/training divergence; 2025 stability finding.
- See also: Bitwise-Consistent RL, Train-inference mismatch. See `60/06`.

**GRPO (Group Relative Policy Optimization).** DeepSeekMath's PPO variant: $N$
completions per prompt; advantage estimated from group-relative rewards; no
critic. Made GRPO the default RL algorithm overnight after DeepSeek-R1; turns
rollout into a many-completions-per-prompt batched-generation problem with
heavy prefix-cache reuse.
- See also: DAPO, Dr.GRPO, RLVR. See `60/06`. `[DeepSeekMath / GRPO]`

**HybridEngine.** Single-process colocation of trainer and rollout (DeepSpeed-
Chat, veRL/HybridFlow) where weight transfer is "free" via in-process IPC. The
early architectural pattern; in 2026 increasingly displaced by fully
disaggregated pipeline-async designs (PipelineRL, AReaL boba², ROLL Flash,
veRL 0.7 fully-async).
- See also: veRL, DeepSpeed-Chat. See `60/06`. `[HybridFlow / veRL]`

**NeMo-RL.** NVIDIA's Ray-based RL framework that replaced NeMo-Aligner; v0.3
adds a Megatron-Core backend (6D parallelism) and DAPO; v0.6 (May 2026) ships
speculative decoding inside the RL training loop with 1.8× rollout speedup at
8B.
- See also: veRL, AReaL, slime. See `60/06`.

**OpenRLHF.** Independent RL framework (EMNLP-Demos 2025) with Ray + vLLM +
DeepSpeed ZeRO; established NCCL broadcast and CUDA-IPC handles as the
canonical weight-sync mechanisms; ~8.5k LoC simplest-to-adopt option.
- See also: veRL, NeMo-RL. See `60/06`. `[OpenRLHF]`

**PipelineRL.** ServiceNow Research framework distinguished by per-forward-pass
weight swaps (vs per-request or per-step in everything else); Redis-backed
rollout stream.
- See also: TorchForge, AReaL. See `60/06`.

**PRM / ORM (Process Reward Model / Outcome Reward Model).** Reward-model
shapes for reasoning RL. PRMs grade intermediate reasoning steps; ORMs grade
only the final outcome. PRIME (PRIME-RL group) trains an implicit PRM as an
ORM and uses it as a PRM at inference.
- See also: RLVR, LLM-as-judge. See `60/06`.

**RLAIF (RL from AI Feedback).** Constitutional-AI-style RL with synthetic
preference data (Anthropic, 2022). Conceptual ancestor of LLM-as-judge as a
serving workload.
- See also: RLHF, LLM-as-judge. See `60/06`. `[CAI]`

**RLHF.** Reinforcement Learning from Human Feedback. The InstructGPT
SFT → RM → PPO three-stage recipe (Ouyang et al., 2022) is the structural
template every RL framework still uses.
- See also: RLAIF, GRPO. See `60/06`. `[InstructGPT]`

**RLVR (RL with Verifiable Rewards).** Rule-based / verifier-pool rewards (no
learned reward model), popularized by DeepSeek-R1 for math/code reasoning;
displaced learned reward models for verifiable tasks. Reward-model serving
survives in agentic / rubric-graded domains.
- See also: GRPO, DeepSeek-R1, PRM/ORM. See `60/06`.

**Rollout engine.** Inference engine specialized for RL training: long
generations, frequent weight churn, partial-rollout pause/resume, and
NCCL/RDMA channels for sync from a separate trainer process. vLLM and SGLang
are the two dominant rollout engines; TRT-LLM is being adopted in
NeMo-RL / NVIDIA stacks.
- See also: HybridEngine, Weight sync, AReaL, slime. See `60/06`.

**SkyRL.** NovaSky-AI (Berkeley) + Anyscale modular RL library; multi-turn
agentic-first; supports sync or async, colocated or disaggregated, integrated
or external inference engine. SkyRL-tx (Nov 2025) provides a Tinker-compatible
local backend.
- See also: veRL, AReaL, Tinker. See `60/06`.

**slime.** SGLang-native post-training framework for RL scaling (THUDM +
LMSYS, July 2025). Three-component design (rollout via SGLang+sgl-router,
trainer via Megatron, data buffer between them); powers the GLM-4.5/4.6/4.7
family.
- See also: veRL, AReaL, NeMo-RL. See `60/06`.

**SPEC-RL.** Speculative-rollout RL acceleration (Sep 2025): reuses
prior-epoch trajectory prefixes via draft-and-verify; 2–3× rollout speedup,
drop-in for PPO/GRPO/DAPO.
- See also: GRPO, Speculative decoding. See `60/06`.

**Tinker.** Thinking Machines Lab's hosted RL/fine-tuning API (Mira Murati,
GA Dec 2025); SkyRL-tx provides an open Tinker-compatible local backend.
- See also: SkyRL. See `60/06`.

**Train-inference mismatch.** Numerical divergence between vLLM-class rollout
engines and Megatron-class trainers at the same checkpoint; the active
2025–2026 stability problem in async RL. Fixes include FP16-RL, bitwise-
consistent vLLM+TorchTitan setups, and importance-sampling corrections (TIS,
TOPR, CISPO).
- See also: FP16-RL, Bitwise-Consistent RL. See `60/06`.

**veRL (HybridFlow).** ByteDance Seed + HKU framework (arXiv:2409.19256). Hybrid
single-controller (Ray, MPMD orchestration) + multi-controller (SPMD trainer/
rollout workers). 3D-HybridEngine reshards actor weights between training-shape
(3D) and inference-shape (TP-only) without going through host memory; 1.53×–
20.57× claimed throughput. Reference framework for DAPO.
- See also: HybridEngine, Checkpoint Engine, slime. See `60/06`. `[HybridFlow / veRL]`

**Weight sync.** Channel that propagates trainer weight deltas to the rollout
engine: NCCL broadcast, CUDA-IPC handles for colocated trainers, NIXL / Awex /
Mooncake Checkpoint Engine for cross-node, or per-forward-pass swaps
(PipelineRL). Determines staleness and downtime per RL step.
- See also: Awex, Checkpoint Engine, NIXL. See `60/06`.

---

*End of glossary.*
