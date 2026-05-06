# Bibliography

## How to use this bibliography

This bibliography supports a textbook on LLM inference infrastructure. Entries are grouped topically to mirror the chapter structure (foundations, attention kernels, quantization, KV cache, batching, disaggregation, MoE, long-context, heterogeneous serving, LoRA, cluster systems, hardware, adjacent workloads, RL post-training, and OSS engines). Within each topic, entries are ordered chronologically (oldest first) so the lineage is visible.

Each entry carries two tags:

- **Tier** — `[CANONICAL]` (lineage-defining), `[SOTA]` (current best-in-class), `[EMERGING]` (preprint or exploratory), `[REFERENCE]` (vendor docs, blog posts, standards). A fourth tag, `[UNVERIFIED]`, is added when the source briefs flagged the metadata as not directly confirmed.
- **Recency** — `[2026-NEW]` for items dated 2025-05 to 2026-05 inclusive (the "past 12 months" frontier). Older entries carry no recency tag.

Chapters cite into this file by citation key (e.g., `[FlashAttn-3]`). Where a paper appears in multiple briefs under different keys, a single canonical key is used here and an "also cited as" line lists synonyms. The "Past 12 months: at a glance" section at the end re-lists every `[2026-NEW]` entry in flat chronological order so the reader can scan recent developments in one view.

Date format is YYYY-MM. Venues are listed where known; "preprint" indicates arXiv-only as of the cutoff (May 2026). URLs are direct links to papers, blog posts, or repositories.

## Topical index

### Foundations and roofline

- `[Roofline-Survey]` *LLM Inference Unveiled: Survey and Roofline Model Insights*. (multi-author). 2024-02 (rev v5 2025). preprint. https://arxiv.org/html/2402.16363v5 [REFERENCE]
  - Roofline analysis of LLM inference across major GPUs; baseline reference for arithmetic-intensity arguments.
- `[Scale-Book-Roofline]` *All About Rooflines (How To Scale Your Model)*. JAX scaling book. 2024–25. blog. https://jax-ml.github.io/scaling-book/roofline/ [REFERENCE]
  - Pedagogical roofline reference used widely in serving discussions.
- `[Scale-Book-TPU]` *How to Think About TPUs*. JAX scaling book. 2024–25. blog. https://jax-ml.github.io/scaling-book/tpus/ [REFERENCE]
  - TPU-side roofline plus pod-topology discussion.

### Attention kernels

#### Lineage

- `[FlashAttn-1]` *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. Dao, Fu, Ermon, Rudra, Ré (Stanford). 2022-05. NeurIPS 2022, arXiv:2205.14135. https://arxiv.org/abs/2205.14135 [CANONICAL]
  - Introduced IO-aware tiled attention and the online-softmax recurrence.
- `[FlashAttn-2]` *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. Dao (Princeton/Together). 2023-07. ICLR 2024, arXiv:2307.08691. https://arxiv.org/abs/2307.08691 [CANONICAL]
  - Sequence-axis parallelism, reduced non-matmul FLOPs, 50–73% A100 peak.
- `[FlashDecode]` *Flash-Decoding for long-context inference*. Dao, Haziza, Massa, Sizov (Stanford/Meta/Together). 2023-10. blog. https://crfm.stanford.edu/2023/10/12/flashdecoding.html [CANONICAL]
  - KV-axis split for batch-1 long-context decode; basis of every modern decode kernel.
- `[FlashDecode++]` *FlashDecoding++: Faster Large Language Model Inference on GPUs*. Hong et al. (Tsinghua/Infinigence). 2023-11. MLSys 2024, arXiv:2311.01282. https://arxiv.org/abs/2311.01282 [CANONICAL]
  - Asynchronized softmax with unified max; flat-GEMM for query-length-1 decode.
- `[StreamingLLM]` *Efficient Streaming Language Models with Attention Sinks*. Xiao, Tian, Chen, Han, Lewis (MIT/Meta/CMU/NVIDIA). 2023-09. ICLR 2024, arXiv:2309.17453. https://arxiv.org/abs/2309.17453 [CANONICAL]
  - Discovered the attention-sink phenomenon; foundation for sink-aware kernels and sliding-window serving.
- `[FA-2-Hopper]` *A Case Study in CUDA Kernel Fusion: FlashAttention-2 on Hopper using CUTLASS*. Spector, Shah, Cazenavette, Thakkar (Colfax). 2023-12. arXiv:2312.11918. https://arxiv.org/abs/2312.11918 [CANONICAL]
  - Blueprint for TMA + WGMMA + CuTe attention; precursor to FA3.
- `[FlashAttn-3]` *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*. Shah, Bikshandi, Zhang, Thakkar, Ramani, Ré, Dao (Colfax/Together/Meta/NVIDIA/Princeton). 2024-07. NeurIPS 2024, arXiv:2407.08608. https://arxiv.org/abs/2407.08608 [CANONICAL]
  - Producer/consumer warp specialization, ping-pong scheduling, FP8 with block-quant.
- `[FlexAttn]` *FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention*. Dong, He, Reizenstein, et al. (PyTorch/Meta). 2024-08. blog → arXiv:2412.05496. https://pytorch.org/blog/flexattention/ [CANONICAL]
  - score_mod / mask_mod Python functions lowered to fused Triton FA kernels.
- `[ThunderKittens]` *ThunderKittens: Simple, Fast, and Adorable AI Kernels*. Spector, Arora, Singhal, Fu, Ré (Hazy Research, Stanford). 2024-10. arXiv:2410.20399. https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk [CANONICAL]
  - Tile-primitive C++ DSL; basis for ThunderMLA, HipKittens, ParallelKittens.
- `[Liger]` *Liger Kernel: Efficient Triton Kernels for LLM Training*. Hsu et al. (LinkedIn). 2024-10. arXiv:2410.10989. https://arxiv.org/abs/2410.10989 [CANONICAL]
  - Production Triton fused kernels (RMSNorm, RoPE, FLCE); now standard in HF/TRL/Axolotl.
- `[FlashInfer]` *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving*. Ye, Chen, Lai, Lin et al. (UW/CMU/NVIDIA/OctoAI). 2025-01. MLSys 2025 best paper, arXiv:2501.01005. https://arxiv.org/abs/2501.01005 [CANONICAL]
  - Unifies paged/ragged/radix/tree KV under a Block-Sparse-Row family; integrated into vLLM/SGLang/MLC.

#### Past 12 months SOTA

- `[GLA-GTA]` *Hardware-Efficient Attention for Fast Decoding*. Zadouri et al. (Princeton). 2025-05. arXiv:2505.21487. https://arxiv.org/abs/2505.21487 [SOTA] [2026-NEW]
  - GTA halves KV vs GQA; GLA kernel ≤2× FlashMLA in spec-decode.
- `[SageAttn3]` *SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training*. Zhang, Wei et al. (THU/Shengshu). 2025-05. arXiv:2505.11594. https://arxiv.org/abs/2505.11594 [SOTA] [2026-NEW]
  - First plug-and-play NVFP4/MXFP4 attention; 1038 TOPS on RTX 5090.
- `[FlashMLA]` *FlashMLA: Efficient Multi-head Latent Attention Kernels*. DeepSeek. 2025-02. open-source week day 1, https://github.com/deepseek-ai/FlashMLA [SOTA] [2026-NEW]
  - Canonical MLA decode kernel for Hopper; 3000 GB/s memory-bound on H800.
- `[ThunderMLA]` *ThunderMLA: FlashMLA, Faster and Fused-er!*. Hazy Research. 2025-03. blog. https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla [SOTA] [2026-NEW]
  - ThunderKittens re-implementation of FlashMLA with on-device scheduler.
- `[TK-Blackwell]` *ThunderKittens Now on Blackwells!*. Hazy Research. 2025-03. blog. https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell [SOTA] [2026-NEW]
  - BF16 + FP8 GEMM and attention kernels for B200 written in TK; tcgen05 reference.
- `[TileLang]` *TileLang: A Composable Tiled Programming Model for AI Systems*. Wang et al. (PKU/Microsoft). 2025-04. arXiv:2504.17577. https://arxiv.org/abs/2504.17577 [SOTA] [2026-NEW]
  - ~98% FlashMLA performance on H100 in ~70 lines of Python; basis of FlashQLA.
- `[CuTe-DSL]` *Achieve CUTLASS C++ Performance with Python APIs Using CuTe DSL*. NVIDIA. 2025. blog. https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/ [REFERENCE] [2026-NEW]
  - The Python DSL FA4 is built on; FMHA examples for SM100 and SM80.
- `[FlashInfer-NVIDIA]` *Run High-Performance LLM Inference Kernels from NVIDIA Using FlashInfer*. NVIDIA Technical Blog. 2025. https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/ [REFERENCE] [2026-NEW]
  - NVIDIA shipping TRT-LLM kernels through FlashInfer.
- `[XQA]` *New XQA-kernel*. NVIDIA TRT-LLM blog. 2025. https://nvidia.github.io/TensorRT-LLM/blogs/XQA-kernel.html [REFERENCE] [2026-NEW]
  - TRT-LLM MQA/GQA-aware decode kernel; 2.4× Llama-70B at fixed latency.
- `[FA2-cuDNN]` *Accelerating Transformers with NVIDIA cuDNN 9*. NVIDIA. 2024–2025. https://developer.nvidia.com/blog/accelerating-transformers-with-nvidia-cudnn-9/ [REFERENCE]
  - Fused SDPA in cuDNN 9.13.1+; up to 1.2 PFLOPS FP8 on H200; baseline FA4 measures against.
- `[vLLM-torchcompile]` *Introduction to torch.compile and How It Works with vLLM*. vLLM blog. 2025-08. https://blog.vllm.ai/2025/08/20/torch-compile.html [REFERENCE] [2026-NEW]
  - Piecewise-compile + piecewise-CUDA-graph; attention stays eager.
- `[DSA-V32]` *DeepSeek Sparse Attention (DSA) in FlashMLA*. DeepSeek. 2025-09. https://api-docs.deepseek.com/news/news250929 [SOTA] [2026-NEW]
  - Lightning-indexer + top-k drives O(L²)→O(Lk); 640 TFLOPs prefill, 410 TFLOPs decode; day-0 in vLLM and SGLang.
- `[Triton-Anatomy]` *The Anatomy of a Triton Attention Kernel*. Ringlein et al. (IBM Research). 2025-11. arXiv:2511.11581. https://arxiv.org/abs/2511.11581 [SOTA] [2026-NEW]
  - 800-line Triton kernel reaches 105.9% of FA3 on H100 after auto-tuning; basis of vLLM Triton backend.
- `[HipKittens]` *HipKittens: Fast and Furious AMD Kernels*. Hazy Research. 2025-11. blog. https://hazyresearch.stanford.edu/blog/2025-11-09-hk [EMERGING] [2026-NEW]
  - AMD-MI300 port of TK; AMD answer to FA's Hopper specificity.
- `[ParallelKittens]` *Systematic and Practical Simplification of Multi-GPU AI Kernels*. Hazy Research. 2025-11. blog. [EMERGING] [2026-NEW]
  - Collective-aware tile primitives; multi-GPU attention experiments.
- `[Megakernels]` *Loads and Loads of Fluffy Kittens*. Hazy Research. 2025-11. https://hazyresearch.stanford.edu/blog/2025-11-17-fluffy-kittens [EMERGING] [2026-NEW]
  - Megakernel idea (one persistent kernel per request); potential successor to piecewise compilation.
- `[TK-2.0]` *ThunderKittens 2.0: Even Faster Kernels for Your GPUs*. Hazy Research. 2026-02. https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2 [SOTA] [2026-NEW]
  - Full Blackwell support; NVFP4/MXFP8 GEMMs at-or-above cuBLAS on B200.
- `[GPT-OSS-vLLM]` *GPT-OSS Performance Optimizations on NVIDIA Blackwell*. vLLM blog. 2026-02. https://blog.vllm.ai/2026/02/01/gpt-oss-optimizations.html [REFERENCE] [2026-NEW]
  - Attention-sink + 128-token sliding-window kernel work; hybrid KV allocator.
- `[FlashAttn-4]` *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*. Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao (Princeton/Together/Meta/xAI/NVIDIA). 2026-03. arXiv:2603.05451. https://tridao.me/blog/2026/flash4/ [SOTA] [2026-NEW]
  - First FA in CuTe-DSL; software-emulated exp2 polynomial; 1605 TFLOPs/s BF16 on B200.
- `[FA4-RE]` *We reverse-engineered Flash Attention 4*. Modal team. 2026. blog. https://modal.com/blog/reverse-engineer-flash-attention-4 [REFERENCE] [2026-NEW]
  - Engineering walkthrough of the FA4 polynomial-exp trick and Cody-Waite range reduction.
- `[FA4-Together]` *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design*. Together AI blog. 2026-03. https://www.together.ai/blog/flashattention-4 [REFERENCE] [2026-NEW]
  - Production-engine view of FA4 deployment.
- `[FA4-Colfax]` *FlashAttention-4 Algorithm and Kernel Pipelining Co-Design*. Colfax Research. 2026. https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/ [REFERENCE] [2026-NEW]
  - Kernel-author commentary on TMEM, tcgen05, and pipeline scheduling.
- `[FA4-FlexAttn]` *FlexAttention + FlashAttention-4: Fast and Flexible*. PyTorch blog. 2026. https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/ [REFERENCE] [2026-NEW]
  - ALiBi / sliding-window / score_mod reaching FA4-class speeds via DSL lowering.
- `[vLLM-Triton]` *vLLM Triton Attention Backend Deep Dive*. vLLM blog. 2026-03. https://vllm.ai/blog/vllm-triton-backend-deep-dive [REFERENCE] [2026-NEW]
  - Persistent kernels, 3D parallel-tiled-softmax decode; default for AMD.
- `[ICLR-Evo]` *The Evolution of FlashAttention*. ICLR 2026 blogposts track. 2026. https://iclr-blogposts.github.io/2026/blog/2026/the-evolution-of-flashattention/ [REFERENCE] [2026-NEW]
  - Peer-reviewed synthesis of FA1→4.

#### Emerging

- `[DeFT]` *DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference*. ICLR 2025. https://arxiv.org/abs/2404.00242 [EMERGING] [2026-NEW]
  - Speculative-decoding-friendly flash-attention for tree-shaped Q with shared KV.
- `[Kvax]` *Kvax: Fast and easy-to-use Flash Attention implementation for JAX*. Nebius. 2025. [EMERGING] [2026-NEW]
  - FA for JAX/TPU/GPU.
- `[BitDecoding]` *BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache*. 2025-03. arXiv:2503.18773. https://arxiv.org/abs/2503.18773 [EMERGING] [2026-NEW]
  - Tensor-core-native low-bit KV decode kernel; complements SageAttn3.
- `[FLASH-D]` *FLASH-D: FlashAttention with Hidden Softmax Division*. 2025-05. arXiv:2505.14201. https://arxiv.org/abs/2505.14201 [EMERGING] [2026-NEW]
  - Removes the explicit normalization step.
- `[Tawa]` *Tawa: Automatic Warp Specialization for Modern GPUs*. 2025-10. arXiv:2510.14719. https://arxiv.org/abs/2510.14719 [EMERGING] [2026-NEW]
  - Compiler-driven warp-spec generation; would automate FA3-style producer/consumer split.
- `[FlashInfer-Bench]` *MLSys 2026 NVIDIA Track: FlashInfer Kernel Generation Contest*. 2026. https://mlsys26.flashinfer.ai/ [REFERENCE] [2026-NEW]
  - Institutionalizes AI-agent-written GPU kernels (CuTe-DSL/Triton/TileLang) as a benchmark.
- `[SageAttn2]` *SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization*. Zhang et al. (THU). 2024-11. ICML 2025, arXiv:2411.10958. https://arxiv.org/abs/2411.10958 [SOTA]
  - INT4(QK) + FP8(PV) thread-granularity quant; ~3× FA2 throughput on H100.


### Quantization

#### Lineage — weight-only PTQ

- `[GPTQ]` *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. Frantar, Ashkboos, Hoefler, Alistarh (IST Austria/ETH). 2022-10. ICLR 2023, arXiv:2210.17323. https://arxiv.org/abs/2210.17323 [CANONICAL]
  - Approximate-second-order layer-wise weight quantization; ancestor of every W4A16 path.
- `[AWQ]` *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang, Gan, Han (MIT-Han Lab). 2023-06. MLSys 2024 (Best Paper), arXiv:2306.00978. https://arxiv.org/abs/2306.00978 [CANONICAL]
  - Salient channels by activation magnitude; ~1% protected via per-channel scaling.
- `[QLoRA]` *QLoRA: Efficient Finetuning of Quantized LLMs*. Dettmers, Pagnoni, Holtzman, Zettlemoyer. 2023-05. NeurIPS 2023, arXiv:2305.14314. https://arxiv.org/abs/2305.14314 [CANONICAL]
  - 4-bit NormalFloat (NF4); foundation of bitsandbytes 4-bit and 4-bit-base + bf16-adapter pattern.
- `[SqueezeLLM]` *SqueezeLLM: Dense-and-Sparse Quantization*. Kim et al. (UC Berkeley). 2023-06. ICML 2024, arXiv:2306.07629. https://arxiv.org/abs/2306.07629 [CANONICAL]
  - Sensitivity-based non-uniform quantization plus dense-and-sparse decomposition.
- `[OmniQuant]` *OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models*. OpenGVLab/Shanghai AI Lab. 2023-08. ICLR 2024, arXiv:2308.13137. https://arxiv.org/abs/2308.13137 [CANONICAL]
  - LWC + LET on top of GPTQ; superior at extreme bit-widths.
- `[AutoRound]` *AutoRound: Optimize Weight Rounding via Signed Gradient Descent*. Cheng et al. (Intel). 2023-09. arXiv:2309.05516. https://arxiv.org/abs/2309.05516 [CANONICAL]
  - Block-wise signed-gradient descent; integrated into LLM Compressor in 2025.
- `[HQQ]` *Half-Quadratic Quantization*. Badri (Mobius Labs). 2023-11. blog. https://mobiusml.github.io/hqq_blog/ [REFERENCE]
  - Calibration-free closed-form weight quantization; 50× faster than GPTQ.
- `[GGUF]` *GGUF k-/i-quants*. llama.cpp / GGML. ongoing. https://github.com/ggml-org/llama.cpp [REFERENCE]
  - k-quants and importance-matrix-aware i-quants; default for CPU/Apple/consumer GPU edge.

#### Lineage — weight + activation PTQ

- `[LLM.int8]` *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. Dettmers, Lewis, Belkada, Zettlemoyer. 2022-08. NeurIPS 2022, arXiv:2208.07339. https://arxiv.org/abs/2208.07339 [CANONICAL]
  - Mixed-precision decomposition; foundational characterization of emergent outlier features.
- `[FP8-Formats]` *FP8 Formats for Deep Learning*. Micikevicius et al. (NVIDIA, ARM, Intel). 2022-09. arXiv:2209.05433. https://arxiv.org/abs/2209.05433 [CANONICAL]
  - Defines E4M3 and E5M2 and the standard hybrid recipe.
- `[SmoothQuant]` *SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs*. Xiao, Lin, Seznec, Wu, Demouth, Han (MIT-Han Lab/NVIDIA). 2022-11. ICML 2023, arXiv:2211.10438. https://arxiv.org/abs/2211.10438 [CANONICAL]
  - W8A8 PTQ via offline smoothing; recipe reappears in SpinQuant/FlatQuant.
- `[FP8-LM]` *FP8-LM: Training FP8 Large Language Models*. Microsoft. 2023-10. arXiv:2310.18313. https://arxiv.org/abs/2310.18313 [CANONICAL]
  - Earliest end-to-end FP8 LLM training framework.
- `[Atom]` *Atom: Low-bit Quantization for Efficient and Accurate LLM Serving*. Zhao et al. 2023-10. MLSys 2024, arXiv:2310.19102. https://arxiv.org/abs/2310.19102 [CANONICAL]
  - W4A4 with mixed-precision outlier handling; predecessor of QServe.
- `[QServe]` *QServe: W4A8KV4 Quantization and System Co-design*. Lin, Tang, Yang, Han (MIT-Han Lab). 2024-05. MLSys 2025, arXiv:2405.04532. https://arxiv.org/abs/2405.04532 [CANONICAL]
  - W4A8KV4 with progressive dequantization; 1.2–3.5× over TRT-LLM.
- `[DeepSeek-V3-FP8]` *DeepSeek-V3 Technical Report*. DeepSeek-AI. 2024-12. arXiv:2412.19437. https://arxiv.org/abs/2412.19437 [CANONICAL] [2026-NEW]
  - First very-large-scale (671B MoE) model trained natively in FP8 with fine-grained tile/block-wise scaling.
  - Also cited as: `[MLA-V3]`, `[DeepSeek-V3]`.

#### Lineage — KV cache quantization

- `[KIVI]` *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*. Liu, Yuan, Yu et al. 2024-02. ICML 2024, arXiv:2402.02750. https://arxiv.org/abs/2402.02750 [CANONICAL]
  - Per-channel K, per-token V at 2-bit; 2.6× memory, 2.35–3.47× throughput.
- `[KVQuant]` *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization*. Hooper, Kim, Mohtashami et al. (UC Berkeley). 2024-01. NeurIPS 2024, arXiv:2401.18079. https://github.com/SqueezeAILab/KVQuant [CANONICAL]
  - Pre-RoPE K quant, NUQ codebooks, 3-bit with <0.1 PPL hit; 10M-token-context demos.
- `[GEAR]` *GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Inference*. Kang et al. (Georgia Tech). 2024-03. COLM 2024, arXiv:2403.05527. https://arxiv.org/abs/2403.05527 [CANONICAL]
  - Q (low-bit base) + L (low-rank residual) + S (sparse outliers); explicit hybrid.
- `[KCache]` *Efficient LLM Inference with KCache*. He et al. 2024-04. arXiv:2404.18057. https://arxiv.org/abs/2404.18057 [REFERENCE]
  - Argues some V cache can be dropped entirely; ~40% throughput at minimal quality loss.

#### Lineage — rotation / Hadamard family

- `[QuaRot]` *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*. Ashkboos, Mohtashami, Croci et al. (ETH/IST Austria/MSR). 2024-04. NeurIPS 2024, arXiv:2404.00456. https://arxiv.org/abs/2404.00456 [CANONICAL]
  - 4 inserted rotations (R1/R2 offline, R3/R4 online); foundation of rotation lineage.
- `[SpinQuant]` *SpinQuant: LLM Quantization with Learned Rotations*. Liu et al. (Meta). 2024-05. ICLR 2025, arXiv:2405.16406. https://arxiv.org/abs/2405.16406 [CANONICAL]
  - Cayley-optimized learned rotations; +45.1% relative to QuaRot on Llama-3-8B.
- `[DuQuant]` *DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs*. Lin, Xu et al. 2024-06. NeurIPS 2024 Oral, arXiv:2406.01721. https://arxiv.org/abs/2406.01721 [CANONICAL]
  - Block-wise rotation + zigzag permutation; SOTA W4A4 at the time.
- `[FlatQuant]` *FlatQuant: Flatness Matters for LLM Quantization*. Liu et al. 2024-10. ICML 2025, arXiv:2410.09426. https://arxiv.org/abs/2410.09426 [CANONICAL]
  - Per-layer Kronecker-factored learnable affine transforms; W4A4 LLaMA-3-70B with <1% drop.

#### Lineage — formats

- `[OCP-MX]` *OCP Microscaling Formats (MX) Specification v1.0*. Open Compute Project. 2023-09. https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf [REFERENCE]
  - Standardizes block-floating-point: MXFP8, MXFP6, MXFP4, MXINT8.
- `[MX-DataFormats]` *Microscaling Data Formats for Deep Learning*. Rouhani et al. (Microsoft, AMD, Arm, Intel, Meta, NVIDIA, Qualcomm). 2023-10. arXiv:2310.10537. https://arxiv.org/abs/2310.10537 [CANONICAL]
  - Empirical paper underlying the OCP spec; sub-8-bit accuracy parity.
- `[NVFP4-Inference]` *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference*. NVIDIA Technical Blog. 2025. https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ [REFERENCE] [2026-NEW]
  - NVFP4 = 16-element micro-blocks with two-level scaling; 3.5× memory vs FP16.

#### Lineage — QAT and 1-bit

- `[LSQ]` *Learned Step Size Quantization*. Esser et al. (IBM). 2019. ICLR 2020, arXiv:1902.08153. https://arxiv.org/abs/1902.08153 [CANONICAL]
  - Pre-LLM but foundational; learnable step size with STE.
- `[LLM-QAT]` *LLM-QAT: Data-Free Quantization Aware Training for LLMs*. Liu et al. (Meta). 2023-05. ACL Findings 2024, arXiv:2305.17888. https://arxiv.org/abs/2305.17888 [CANONICAL]
  - QAT using model self-generations as distillation data.
- `[BitNet]` *BitNet: Scaling 1-bit Transformers for Large Language Models*. Wang, Ma, Dong et al. (Microsoft). 2023-10. arXiv:2310.11453. https://arxiv.org/abs/2310.11453 [CANONICAL]
  - Replaces nn.Linear with BitLinear; native binary training.
- `[BitNet-1.58]` *The Era of 1-bit LLMs: All LLMs are in 1.58 Bits*. Ma, Wang, Ma et al. (Microsoft). 2024-02. arXiv:2402.17764. https://arxiv.org/abs/2402.17764 [CANONICAL]
  - Ternary weights via absmean; matches FP at 3B+ on perplexity.
- `[EfficientQAT]` *EfficientQAT: Efficient Quantization-Aware Training for LLMs*. 2024-07. ACL 2025, arXiv:2407.11062. https://arxiv.org/abs/2407.11062 [CANONICAL]
  - Two-phase block-wise then end-to-end QAT.
- `[BitNet-a4.8]` *BitNet a4.8: 4-bit Activations for 1-bit LLMs*. Microsoft Research. 2024-11. https://www.microsoft.com/en-us/research/publication/bitnet-a4-8-4-bit-activations-for-1-bit-llms/ [REFERENCE]
  - Hybrid 4-bit-act / 8-bit-elsewhere with intermediate-state sparsification.
- `[bitnet.cpp]` *BitNet (inference framework)*. Microsoft. 2024–25. https://github.com/microsoft/BitNet [REFERENCE]
  - Official 1-bit inference framework; CPU-first kernels.

#### Past 12 months SOTA

- `[BitNet-2B-4T]` *BitNet b1.58 2B4T Technical Report*. Ma, Wang, Huang et al. (Microsoft). 2025-04. arXiv:2504.12285. https://arxiv.org/abs/2504.12285 [SOTA] [2026-NEW]
  - First open-source native 1-bit LLM at 2B scale on 4T tokens.
- `[TurboQuant]` *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. Zandieh, Daliri, Hadian, Mirrokni (Google Research/NYU/DeepMind). 2025-04. ICLR 2026, arXiv:2504.19874. https://arxiv.org/abs/2504.19874 [SOTA] [2026-NEW]
  - Random rotation + 1-bit residual VQ; 3.5-bit KV at parity, 2.5-bit marginal degradation.
- `[GBS-DimExpansion]` *Gradual Binary Search and Dimension Expansion: A general method for activation quantization in LLMs*. Maisonnave et al. (Inria/CEA). 2025-04. arXiv:2504.13989. https://arxiv.org/abs/2504.13989 [EMERGING] [2026-NEW]
  - Hadamard + Paley-extension to non-power-of-2 dims; pushes to 3-bit W/A/KV.
- `[MILLION]` *MILLION: Mastering Long-Context LLM Inference Via Outlier-Immunized KV Product Quantization*. 2025-04. arXiv:2504.03661. https://arxiv.org/abs/2504.03661 [EMERGING] [2026-NEW]
  - KV product quantization at 4-bit with 2.09× e2e gain at 32K context.
- `[Quartet]` *Quartet: Native FP4 Training Can Be Optimal for Large Language Models*. Castro, Panferov, Tabesh, Sieberling, Chen, Nikdan, Ashkboos, Alistarh (IST Austria/Red Hat AI/ETH). 2025-05. NeurIPS 2025, arXiv:2505.14669. https://arxiv.org/abs/2505.14669 [SOTA] [2026-NEW]
  - End-to-end FP4 training using Random Hadamard + 2D quantization + stochastic rounding.
- `[Oaken]` *Oaken: Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization*. 2025. ISCA 2025. https://dl.acm.org/doi/10.1145/3695053.3731019 [SOTA] [2026-NEW]
  - Algo/HW co-design with offline outlier thresholds + online scale.
- `[BASE-Q]` *BASE-Q: Bias and Asymmetric Scaling Enhanced Rotational Quantization*. 2025-06. arXiv:2506.15689. https://arxiv.org/abs/2506.15689 [EMERGING] [2026-NEW]
  - Bias correction and asymmetric scaling on top of rotation.
- `[AnTKV]` *AnTKV: Anchor Token-Aware Sub-Bit Vector Quantization for KV Cache*. 2025-06. arXiv:2506.19505. https://arxiv.org/abs/2506.19505 [EMERGING] [2026-NEW]
  - 1-bit Mistral-7B PPL 6.32 vs KVQuant 15.36; 840K-token context on one A100.
- `[OSP]` *Outlier-Safe Pre-Training*. 2025-06. arXiv:2506.19697. https://arxiv.org/abs/2506.19697 [EMERGING] [2026-NEW]
  - Muon + Single-Scale RMSNorm trains 1.4B / 1T tokens without emergent outliers.
- `[MXFP8-Recipes]` *Recipes for Pre-training LLMs with MXFP8*. 2025-06. arXiv:2506.08027. https://arxiv.org/abs/2506.08027 [SOTA] [2026-NEW]
  - Round-to-infinity scale-factor rounding required for stable MXFP8 pretraining.
- `[MicroMix]` *MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats*. 2025-08. arXiv:2508.02343. https://arxiv.org/abs/2508.02343 [EMERGING] [2026-NEW]
  - Mixes MXFP4/6/8 per channel; demonstrated on RTX 5090 / 5070Ti.
- `[GPT-OSS-MXFP4]` *Introducing gpt-oss*. OpenAI. 2025-08. https://openai.com/index/introducing-gpt-oss/ [REFERENCE] [2026-NEW]
  - First major open-weight model shipped natively in MXFP4; gpt-oss-120B in 80GB.
- `[NVFP4-Pretraining]` *Pretraining Large Language Models with NVFP4*. NVIDIA. 2025-09. arXiv:2509.25149. https://arxiv.org/abs/2509.25149 [SOTA] [2026-NEW]
  - 12B / 10T tokens; NVFP4 matches FP8 baseline.
- `[Bridging-MXFP4]` *Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization*. 2025-09. arXiv:2509.23202. https://arxiv.org/abs/2509.23202 [EMERGING] [2026-NEW]
  - OAS + MBS closes MXFP4-vs-NVFP4 gap from ~10% to <1%.
- `[MX+]` *MX+: Pushing the Limits of Microscaling Formats for Efficient LLM Serving*. MICRO 2025. arXiv:2510.14557. https://arxiv.org/abs/2510.14557 [SOTA] [2026-NEW]
  - Repurposes outlier exponent field as extended mantissa; +42.15% accuracy over MXFP4.
- `[MX-Plus-emulation]` Same as MX+, software emulation via CUTLASS/Triton.
- `[Four-Over-Six]` *Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling*. 2025-12. arXiv:2512.02010. https://arxiv.org/abs/2512.02010 [EMERGING] [2026-NEW]
  - Adaptive (4 vs 6 bits) per-block scale-bit-width.
- `[Quartet-II]` *Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation*. IST-DASLab. 2026-01. arXiv:2601.22813 [SOTA] [2026-NEW]
  - MS-EDEN unbiased micro-scale rounding (>2× lower quant error).
- `[NVFP4-QAD]` *Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery*. NVIDIA. 2026-03. https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf [REFERENCE] [2026-NEW]
  - Distillation recipe for NVFP4 inference checkpoints.
- `[GPT-OSS-QAT]` *Fine-Tuning gpt-oss for Accuracy and Performance with QAT*. NVIDIA Tech Blog. 2025. https://developer.nvidia.com/blog/fine-tuning-gpt-oss-for-accuracy-and-performance-with-quantization-aware-training/ [REFERENCE] [2026-NEW]
  - QAT recipe for OpenAI GPT-OSS on top of native MXFP4.
- `[Marlin]` *Marlin: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models*. 2024-08. arXiv:2408.11743. https://arxiv.org/pdf/2408.11743 [CANONICAL]
  - The Marlin FP16/INT4 mixed-precision GEMM kernel.
- `[Marlin-RH]` *How Marlin Pushes the Boundaries of Mixed-Precision LLM Inference*. Red Hat blog. 2024. https://developers.redhat.com/articles/2024/04/17/how-marlin-pushes-boundaries-mixed-precision-llm-inference [REFERENCE]
  - Production-context blog on Marlin in vLLM.
- `[LLM-Compressor]` *LLM Compressor*. Red Hat / vLLM. 2024-2026. https://github.com/vllm-project/llm-compressor [REFERENCE] [2026-NEW]
  - De-facto open compression toolchain for vLLM-targeted deployments.
- `[LLM-Compressor-09]` *LLM Compressor 0.9: Attention Quantization, MXFP4 Support, and More*. Red Hat. 2026-01. https://developers.redhat.com/articles/2026/01/16/llm-compressor-090-attention-quantization-mxfp4-support-and-more [REFERENCE] [2026-NEW]
- `[LLM-Compressor-010]` *LLM Compressor 0.10: Faster Compression, Distributed GPTQ*. Red Hat. 2026-03. https://developers.redhat.com/articles/2026/03/18/llm-compressor-010-faster-compression-distributed-gptq [REFERENCE] [2026-NEW]
- `[AutoRound-LLMC]` *Advancing Low-Bit Quantization for LLMs: AutoRound × LLM Compressor*. Red Hat. 2025-12. https://developers.redhat.com/articles/2025/12/09/advancing-low-bit-quantization-llms-autoround-x-llm-compressor [REFERENCE] [2026-NEW]
- `[GPTQModel]` *GPTQModel*. modelcloud. ongoing. https://github.com/modelcloud/gptqmodel [REFERENCE]
  - Successor maintenance of GPTQ; integrates with vLLM/TRT-LLM.
- `[NVIDIA-ModelOpt]` *NVIDIA Model Optimizer*. NVIDIA. ongoing. https://github.com/NVIDIA/Model-Optimizer [REFERENCE]
  - NVFP4/FP8 quantization toolchain; integrates with TRT-LLM/SGLang/vLLM.
- `[SGLang-ModelOpt]` *SGLang × NVIDIA ModelOpt Integration*. LMSYS. 2025-12. https://www.lmsys.org/blog/2025-12-02-modelopt-quantization/ [REFERENCE] [2026-NEW]
- `[SGLang-GPT-OSS-MXFP4]` *SGLang GPT-OSS MXFP4 / QAT*. LMSYS. 2025-08. https://www.lmsys.org/blog/2025-08-28-gpt-oss-qat/ [REFERENCE] [2026-NEW]
- `[vLLM-FP8KV]` *The State of FP8 KV-Cache and Attention Quantization in vLLM*. vLLM project. 2024–25. https://vllm.ai/blog/fp8-kvcache [REFERENCE]
- `[TRT-LLM-KVQ]` *TensorRT-LLM KV Cache Quantization*. NVIDIA. 2024–25. https://nvidia.github.io/TensorRT-LLM/features/kvcache.html [REFERENCE]
- `[AMD-MXFP4]` *AMD MXFP4/MXFP6 Quantization on ROCm*. AMD. 2025. https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html [REFERENCE] [2026-NEW]
- `[AMD-MI325-MLPerf]` *MI325X Accelerates MLPerf Inference*. AMD. 2025. https://rocm.blogs.amd.com/artificial-intelligence/mi325x-accelerates-mlperf-inference/README.html [REFERENCE] [2026-NEW]
- `[vLLM-DSR1-GB200]` *vLLM × GB200 / DeepSeek-R1*. vLLM. 2026-02. https://blog.vllm.ai/2026/02/03/dsr1-gb200-part1.html [REFERENCE] [2026-NEW]


### Speculative decoding and multi-token prediction

#### Lineage

- `[SpecDec-Leviathan]` *Fast Inference from Transformers via Speculative Decoding*. Leviathan, Kalman, Matias (Google). 2022-11. ICML 2023, arXiv:2211.17192. https://arxiv.org/abs/2211.17192 [CANONICAL]
  - Original modified-rejection-sampling speculative decoding; 2–3× T5-XXL.
- `[SpecSamp-Chen]` *Accelerating Large Language Model Decoding with Speculative Sampling*. Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper (DeepMind). 2023-02. arXiv:2302.01318. https://arxiv.org/abs/2302.01318 [CANONICAL]
  - Independently re-derived; 2–2.5× distributed on Chinchilla-70B.
- `[SpecInfer]` *SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification*. Miao, Oliaro et al. (CMU/FlexFlow). 2023-05. ASPLOS 2024, arXiv:2305.09781. https://arxiv.org/abs/2305.09781 [CANONICAL]
  - Token-tree verification with tree attention; 1.5–3.5× speedup.
- `[OSD]` *Online Speculative Decoding*. Liu et al. (UC Berkeley). 2023-10. ICML 2024, arXiv:2310.07177. https://arxiv.org/abs/2310.07177 [CANONICAL]
  - Continually retrains drafter on observed traffic; +0.1–0.65 acceptance.
- `[REST]` *REST: Retrieval-Based Speculative Decoding*. He, Zhong, Cai, Lee, He. 2023-11. NAACL 2024, arXiv:2311.08252. https://arxiv.org/abs/2311.08252 [CANONICAL]
  - External datastore retrieval as drafter; 1.62–2.36×; plug-and-play.
- `[PLD]` *Prompt Lookup Decoding*. Saxena. 2023. https://github.com/apoorvumang/prompt-lookup-decoding [REFERENCE]
  - N-gram match against prompt history; ~2.4× on summarization/QA.
- `[Medusa]` *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*. Cai et al. (Princeton/Together/UIUC). 2024-01. ICML 2024, arXiv:2401.10774. https://arxiv.org/abs/2401.10774 [CANONICAL]
  - Parallel sequentially-independent draft heads; 2.2–3.6× with tree attention.
- `[EAGLE-1]` *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*. Li, Wei, Zhang, Zhang. 2024-01. ICML 2024, arXiv:2401.15077. https://arxiv.org/abs/2401.15077 [CANONICAL]
  - Feature-level draft with one-token shift; 2.7–3.5× Llama-2-Chat-70B.
- `[Lookahead-Decoding]` *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding*. Fu, Bailis, Stoica, Zhang (Hao AI Lab). 2024-02. ICML 2024, arXiv:2402.02057. https://arxiv.org/abs/2402.02057 [CANONICAL]
  - Jacobi-iteration parallel n-gram extraction without a draft model.
- `[GliDe-CaPE]` *GliDe with a CaPE: A Low-Hassle Method to Accelerate Speculative Decoding*. Du et al. 2024-02. ICML 2024, arXiv:2402.02082. https://arxiv.org/abs/2402.02082 [CANONICAL]
  - Draft model reuses target's KV cache; 2.17–2.61× Vicuna.
- `[Hydra]` *Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding*. Ankner, Parthasarathy, Nrusimha, Rinard, Ragan-Kelley, Brandon (MIT). 2024-02. COLM 2024, arXiv:2402.05109. https://arxiv.org/abs/2402.05109 [CANONICAL]
  - 1.31× over Medusa, 2.7× autoregressive.
- `[SpecStream-Apple]` *Speculative Streaming: Fast LLM Inference Without Auxiliary Models*. Apple. 2024-02. NeurIPS 2024 ENLSP, arXiv:2402.11131. https://arxiv.org/abs/2402.11131 [CANONICAL]
  - Multi-stream attention; 1.8–3.1× with ~10000× fewer extra params than Medusa.
- `[Sequoia]` *Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding*. Chen, May, Svirschevski, Huang, Ryabinin, Jia, Chen (CMU/Together). 2024-02. arXiv:2402.12374. https://arxiv.org/abs/2402.12374 [CANONICAL]
  - DP-based optimal tree-shape selection; 4× on Llama-2-7B.
- `[ReDrafter]` *Recurrent Drafter for Fast Speculative Decoding in Large Language Models*. Cheng, Zhang et al. (Apple). 2024-03. arXiv:2403.09919. https://arxiv.org/abs/2403.09919 [CANONICAL]
  - RNN drafter + dynamic tree attention + KD; 2.8× H100; in TensorRT-LLM.
- `[Spec-Bench]` *Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding*. Xia et al. 2024-01. ACL Findings 2024, arXiv:2401.07851. https://arxiv.org/abs/2401.07851 [REFERENCE]
  - Standard 6-task benchmark; basis for per-task numbers.
- `[BASS]` *BASS: Batched Attention-optimized Speculative Sampling*. Qian et al. (AWS). 2024-04. ACL Findings 2024, arXiv:2404.15778. https://arxiv.org/abs/2404.15778 [CANONICAL]
  - First systematic batched-SD; 2.15× at batch 8 on A100.
- `[Better-Faster-MTP]` *Better & Faster Large Language Models via Multi-token Prediction*. Gloeckle, Idrissi, Rozière, Lopez-Paz, Synnaeve (Meta FAIR). 2024-04. ICML 2024, arXiv:2404.19737. https://arxiv.org/abs/2404.19737 [CANONICAL]
  - Pre-training auxiliary loss with n parallel heads; intellectual antecedent of DeepSeek MTP.
- `[Kangaroo]` *Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting*. Liu, Tang et al. (Huawei). 2024-04. NeurIPS 2024, arXiv:2404.18911. https://arxiv.org/abs/2404.18911 [CANONICAL]
  - Shallow target sub-network as drafter; 2.04× walltime.
- `[SmartSpec]` *Optimizing Speculative Decoding for Serving LLMs Using Goodput*. Liu et al. (UC Berkeley/UCSD/Anyscale). 2024-06. arXiv:2406.14066. https://arxiv.org/abs/2406.14066 [CANONICAL]
  - Goodput metric; dynamically modulates γ per request; renamed TurboSpec.
  - Also cited as: `[TurboSpec]`.
- `[EAGLE-2]` *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees*. Same authors as EAGLE-1. 2024-06. EMNLP 2024, arXiv:2406.16858. https://arxiv.org/abs/2406.16858 [CANONICAL]
  - Confidence-based dynamic tree expansion.
- `[Token-Recycling]` *Turning Trash into Treasure: Accelerating Inference of LLMs with Token Recycling*. Luo et al. 2024-08. ACL 2025, arXiv:2408.08696. https://arxiv.org/abs/2408.08696 [CANONICAL]
  - BFS over adjacency matrix of prior top-k; ~2× training-free, <2 MB extra memory.
- `[HASS]` *Learning Harmonized Representations for Speculative Sampling*. 2024-08. ICLR 2025, arXiv:2408.15766. https://arxiv.org/abs/2408.15766 [CANONICAL]
  - Distillation + context-aligned training; +8–16% acceptance over EAGLE-2.
- `[MagicDec]` *MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding*. Chen et al. (CMU/Princeton). 2024-08. ICLR 2025, arXiv:2408.11049. https://arxiv.org/abs/2408.11049 [CANONICAL]
  - Long-context KV-bound regime; 2.51× Llama-3.1-8B at batch 32–256.
- `[Suffix-Decoding]` *SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications*. Snowflake/CMU. 2024-11. NeurIPS 2025 Spotlight, arXiv:2411.04975. https://arxiv.org/abs/2411.04975 [CANONICAL]
  - Suffix tree over generated history; 5.3× for agentic workloads; in vLLM/Arctic Inference.
- `[Theory-NeurIPS24]` *A Theoretical Perspective for Speculative Decoding Algorithm*. Yin et al. NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/e7349e785900b93d8b4971a3f2c1cefe-Paper-Conference.pdf [REFERENCE]
  - Formal expected-acceptance bounds.
- `[SpecKD]` *Speculative Knowledge Distillation*. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/a2747a3844ca1e4667fbff3f558eb39b-Paper-Conference.pdf [SOTA] [2026-NEW]
  - Distillation algorithm tailored to the SD setting.
- `[Survey-2024]` *A Comprehensive Survey of Speculative Decoding*. Xia et al. ACL Findings 2024. https://aclanthology.org/2024.findings-acl.456.pdf [REFERENCE]
- `[Tutorial-2025]` *Tutorial Proposal: Speculative Decoding for Efficient LLM Inference*. 2025-03. arXiv:2503.00491. https://arxiv.org/abs/2503.00491 [REFERENCE] [2026-NEW]

#### Past 12 months SOTA

- `[DeepSeek-V3-MTP]` See `[DeepSeek-V3-FP8]` above. MTP1 acceptance >80%; ~1.8× generation TPS.
- `[Jakiro]` *Jakiro: Boosting Speculative Decoding with Decoupled Multi-Head via MoE*. 2025-02. arXiv:2502.06282. https://arxiv.org/abs/2502.06282 [EMERGING] [2026-NEW]
  - Drafter itself is MoE-decoupled.
- `[Survey-2025]` *Speculative Decoding and Beyond: An In-Depth Survey of Techniques*. Hu et al. 2025-02. arXiv:2502.19732. https://arxiv.org/abs/2502.19732 [REFERENCE] [2026-NEW]
  - Most current taxonomy.
- `[EAGLE-3]` *EAGLE-3: Scaling Up Inference Acceleration of LLMs via Training-Time Test*. Li, Wei, Zhang, Zhang. 2025-03. NeurIPS 2025, arXiv:2503.01840. https://arxiv.org/abs/2503.01840 [SOTA] [2026-NEW]
  - Up to 6.5× batch-1; 1.4× over EAGLE-2; default in vLLM/SGLang/TRT-LLM.
- `[PARD]` *PARD: Accelerating LLM Inference with Low-Cost PARallel Draft Model Adaptation*. An et al. (AMD). 2025-04. arXiv:2504.18583. https://arxiv.org/abs/2504.18583 [SOTA] [2026-NEW]
  - Target-independent parallel drafters; 3.67× Llama-3.1-8B; 1.15× over EAGLE-3.
- `[Theory-Scaling]` *Scaling Laws for Speculative Decoding*. 2025-05. arXiv:2505.07858. https://arxiv.org/abs/2505.07858 [SOTA] [2026-NEW]
  - Log-linear acceptance scaling with drafter capacity, pretraining tokens, batch size.
- `[Lookahead-Reasoning]` *Scaling Speculative Decoding with Lookahead Reasoning*. Fu, Ge et al. (UCSD). 2025-06. NeurIPS 2025, arXiv:2506.19830. https://arxiv.org/abs/2506.19830 [SOTA] [2026-NEW]
  - Step-level speculation for reasoning models; SD speedup 1.4×→2.1× on R1-style.
- `[MoE-Cascade]` *Utility-Driven Speculative Decoding for Mixture-of-Experts*. 2025-06. arXiv:2506.20675. https://arxiv.org/abs/2506.20675 [EMERGING] [2026-NEW]
  - Selectively enables/tunes γ to bound MoE slowdown to <5%.
- `[SpecForge]` *SpecForge: Accelerating Speculative Decoding Training for SGLang*. LMSYS. 2025-07. https://www.lmsys.org/blog/2025-07-25-spec-forge/ [SOTA] [2026-NEW]
  - Open-source EAGLE-3 training framework; 2.0–2.18× MT-Bench on Llama-4 Scout/Maverick.
- `[Llama-Spec-Meta]` *Efficient Speculative Decoding for Llama at Scale: Challenges and Solutions*. Meta. 2025-08. arXiv:2508.08192. https://arxiv.org/abs/2508.08192 [SOTA] [2026-NEW]
  - Production EAGLE for Llama-3.3-70B and Llama-4 Maverick; 4 ms/token batch-1.
- `[SpecMoEOff]` *Accelerating MoE Inference by Hiding Offloading Latency with Speculative Decoding*. 2025-08. arXiv:2508.21706. https://arxiv.org/abs/2508.21706 [EMERGING] [2026-NEW]
  - SD enlarges per-expert workload while CPU↔GPU streams.
- `[SpecVerify]` *Speculative Verification: Exploiting Information Gain for Speculative Decoding*. 2025-09. arXiv:2509.24328. https://arxiv.org/html/2509.24328v2 [EMERGING] [2026-NEW]
  - Information-gain-based verification scheduling.
- `[Mirror-SD]` *Mirror Speculative Decoding: Breaking the Serial Barrier in LLM Inference*. Apple. 2025-10 (rev 2026-01). arXiv:2510.13161. https://arxiv.org/abs/2510.13161 [SOTA] [2026-NEW]
  - Bidirectional drafter↔target speculation; 2.8–5.8× SpecBench.
- `[DVI]` *Draft, Verify, & Improve: Toward Training-Aware Speculative Decoding*. 2025-10. arXiv:2510.05421. https://arxiv.org/abs/2510.05421 [EMERGING] [2026-NEW]
  - On-the-fly drafter LoRA from accept/reject feedback.
- `[BatchSpec-Right]` *Batch Speculative Decoding Done Right*. 2025-10. arXiv:2510.22876. https://arxiv.org/abs/2510.22876 [SOTA] [2026-NEW]
  - Re-examines correctness pitfalls of naive batched SD.
- `[MoE-SpeQ]` *MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching*. 2025-11. arXiv:2511.14102. https://arxiv.org/abs/2511.14102 [EMERGING] [2026-NEW]
  - On-device draft predicts future expert sequence to drive prefetch.
- `[Speculators]` *Speculators v0.3.0*. Red Hat / vLLM. 2025-11 → 2025-12. https://blog.vllm.ai/2025/12/13/speculators-v030.html [SOTA] [2026-NEW]
  - Standardised HF format for spec-dec drafters; one-button training/serving.
- `[Speculators-RH]` *Speculators: Standardized Production-Ready Speculative Decoding*. Red Hat. 2025-11. https://developers.redhat.com/articles/2025/11/19/speculators-standardized-production-ready-speculative-decoding [REFERENCE] [2026-NEW]
- ~~`[VSD]`~~ **DROPPED 2026-05-05** — arXiv:2602.05774 verified to resolve to a martingale-theory paper, not a speculative-decoding paper. The "Variational Speculative Decoding" citation does not exist as listed. Do not cite.
- `[SPEED-Bench]` *SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding*. NVIDIA. 2026-02. arXiv:2604.09557. https://arxiv.org/abs/2604.09557 [REFERENCE] [2026-NEW]
- `[Theory-AcceptDyn]` *Acceptance Dynamics Across Cognitive Domains in Speculative Decoding*. 2026. arXiv:2604.14682. https://arxiv.org/abs/2604.14682 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[Google-SD-Retro]` *Looking Back at Speculative Decoding*. Google Research blog. 2024. https://research.google/blog/looking-back-at-speculative-decoding/ [REFERENCE]
- `[P-EAGLE]` *P-EAGLE: Faster LLM Inference with Parallel Speculative Decoding in vLLM*. AWS. 2025–2026. https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/ [SOTA] [2026-NEW]
  - Pipelines drafter and verifier; 1.69× over EAGLE-3 on B200; in vLLM ≥0.16.
- `[SuffixDec-Arctic]` *SuffixDecoding at Production Scale with Arctic Inference and vLLM*. Snowflake. 2025. https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/ [REFERENCE] [2026-NEW]
- `[MoE-Spec-Cohere]` *Why MoE Models Get More From Speculative Decoding*. Cohere blog. 2025. https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding [REFERENCE] [2026-NEW]
  - Bandwidth-bound sweet-spot argument; idle compute amortizes weight movement.
- `[Speculative-Diffusion]` *Speculative Diffusion Decoding*. NAACL 2025. https://aclanthology.org/2025.naacl-long.601/ [EMERGING] [2026-NEW]
- `[SpecExtend]` *SpecExtend: Drop-in Enhancement for Long-Sequence SD*. 2025. arXiv:2505.20776. https://arxiv.org/html/2505.20776 [EMERGING] [2026-NEW]
- `[Qwen3-Next]` *Qwen3-Next-80B-A3B*. Alibaba Qwen. 2025. https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct [REFERENCE] [2026-NEW] [UNVERIFIED]
  - Hybrid Gated DeltaNet + sparse MoE; ships native MTP head as drafter.
- `[Qwen3.6-MTP]` *Qwen3.6-27B*. Alibaba Qwen. 2026-04. https://huggingface.co/ [REFERENCE] [2026-NEW] [UNVERIFIED]
  - Open-weight dense model with built-in MTP head.
- `[Kimi-K2.5-EAGLE]` *How We Built the Fastest Kimi K2.5 on Artificial Analysis*. Baseten. 2025. https://www.baseten.co/blog/how-we-built-the-fastest-kimi-k2-5-on-artificial-analysis/ [REFERENCE] [2026-NEW]
  - Kimi K2.5 with custom-trained ~1B EAGLE-3 speculator + INT4→NVFP4; 340+ tok/s.
- `[Gemma4-MTP]` *Accelerating Gemma 4: Faster Inference with Multi-Token Prediction Drafters*. Google. 2026. https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/ [REFERENCE] [2026-NEW] [UNVERIFIED]


### KV cache (memory mgmt, compression, tiered offload)

#### Lineage — memory management

- `[PagedAttention]` *Efficient Memory Management for Large Language Model Serving with PagedAttention*. Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica (UC Berkeley/Stanford/UCSD). 2023-09. SOSP 2023 (best paper), arXiv:2309.06180. https://arxiv.org/abs/2309.06180 [CANONICAL]
  - Block-structured KV; the OS-paging analogy that became vLLM.
  - Also cited as: `[vLLM-SOSP23]`.
- `[vAttention]` *vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention*. Prabhu et al. (MSR India). 2024-05. ASPLOS 2025, arXiv:2405.04437. https://arxiv.org/abs/2405.04437 [CANONICAL]
  - Contiguous virtual address space + CUDA VMM; up to 1.97× over vLLM.
- `[vLLM-V1-prefix]` *Automatic Prefix Caching design (V1)*. vLLM project. 2024–25. https://docs.vllm.ai/en/stable/design/prefix_caching/ [REFERENCE]
  - Hash-chained block prefix cache; default-on in V1.

#### Lineage — prefix / prompt caching

- `[Prompt-Cache]` *Prompt Cache: Modular Attention Reuse for Low-Latency Inference*. Gim et al. (Yale/Google). 2023-11. MLSys 2024, arXiv:2311.04934. https://arxiv.org/abs/2311.04934 [CANONICAL]
  - Modular reuse of non-prefix schema fragments; 8–60× TTFT.
- `[RadixAttention]` *SGLang: Efficient Execution of Structured Language Model Programs*. Zheng et al. (LMSYS/UCB/Stanford). 2023-12. NeurIPS 2024, arXiv:2312.07104. https://arxiv.org/abs/2312.07104 [CANONICAL]
  - Token-level radix tree of KV with LRU eviction; default in SGLang.
- `[Hydragen]` *Hydragen: High-Throughput LLM Inference with Shared Prefixes*. Juravsky et al. (Stanford). 2024-02. ICML 2024, arXiv:2402.05099. https://arxiv.org/abs/2402.05099 [CANONICAL]
  - Decompose into shared-prefix and unique-suffix; 32× over baselines for long shared prefix.
- `[ChunkAttention]` *ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache*. Ye et al. (Microsoft). 2024-02. ACL 2024, arXiv:2402.15220. https://arxiv.org/abs/2402.15220 [CANONICAL]
  - Prefix-tree of KV chunks + two-phase partitioned kernel.
- `[CachedAttention]` *Cost-Efficient LLM Serving for Multi-turn Conversations*. Gao et al. (Bytedance). 2024-03. USENIX ATC 2024. https://www.usenix.org/conference/atc24/presentation/gao-bin-cost [CANONICAL]
  - Save inactive KV to cheaper tiers; pre-cursor to LMCache/Mooncake session caches.
- `[CacheBlend]` *CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion*. Yao et al. (U. Chicago). 2024-05. EuroSys 2025, arXiv:2405.16444. https://arxiv.org/abs/2405.16444 [CANONICAL]
  - Reuse precomputed KVs for non-prefix RAG chunks; 2.2–3.3× TTFT; in LMCache.
- `[CacheGen]` *CacheGen: KV Cache Compression and Streaming for Fast LLM Serving*. Liu et al. (U. Chicago). 2023-10. SIGCOMM 2024, arXiv:2310.07240. https://arxiv.org/abs/2310.07240 [CANONICAL]
  - Per-layer-sensitive KV quant + arithmetic coding; 3.5–4.3× cache-size reduction.
- `[Anthropic-PC]` *Prompt caching (API documentation)*. Anthropic. 2024-08+. https://docs.claude.com/en/docs/build-with-claude/prompt-caching [REFERENCE]
  - Production prompt-caching API contract: 5-min default, 1-hour extended TTL.

#### Lineage — eviction / sparsity

- `[H2O]` *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs*. Zhang et al. 2023-06. NeurIPS 2023, arXiv:2306.14048. https://arxiv.org/abs/2306.14048 [CANONICAL]
  - Recent + cumulative-attention heavy hitter eviction; eviction lineage starts here.
- `[Scissorhands]` *Scissorhands: Exploiting the Persistence of Importance Hypothesis*. Liu et al. (Stevens/UMD). 2023-05. NeurIPS 2023, arXiv:2305.17118. https://arxiv.org/abs/2305.17118 [CANONICAL]
  - 5× cache reduction; pivotal-token persistence.
- `[FastGen]` *Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs*. Ge et al. (UIUC/Microsoft). 2023-10. ICLR 2024 oral, arXiv:2310.01801. https://arxiv.org/abs/2310.01801 [CANONICAL]
  - Per-head adaptive policy mix; precursor to DuoAttention.
- `[DejaVu]` *Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time*. Liu et al. (Rice/Stanford). 2023-10. ICML 2023, arXiv:2310.17157. https://arxiv.org/abs/2310.17157 [CANONICAL]
  - Predict per-input head/MLP sparsity; basis for PowerInfer-style hot/cold.
- `[ALISA]` *ALISA: Accelerating LLM Inference via Sparsity-Aware KV Caching*. Zhao et al. (NCSU). 2024-03. ISCA 2024, arXiv:2403.17312. https://arxiv.org/abs/2403.17312 [CANONICAL]
  - Sparse Window Attention + 3-phase scheduler; 3× over FlexGen.
- `[SnapKV]` *SnapKV: LLM Knows What You Are Looking for Before Generation*. Li et al. (UIUC/Cohere). 2024-04. NeurIPS 2024, arXiv:2404.14469. https://arxiv.org/abs/2404.14469 [CANONICAL]
  - Last-N observation tokens detect head-specific patterns; 380× compression at 380K.
- `[PyramidInfer]` *PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inference*. Yang et al. (SJTU). 2024-05. ACL Findings 2024, arXiv:2405.12532. https://arxiv.org/abs/2405.12532 [CANONICAL]
- `[PyramidKV]` *PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling*. Cai et al. (Pekin/Microsoft). 2024-06. ICLR 2025, arXiv:2406.02069. https://arxiv.org/abs/2406.02069 [CANONICAL]
  - Layer-tapered KV budget; matches full-cache LongBench at 12% retention.
- `[Quest]` *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference*. Tang et al. (MIT-HAN-Lab). 2024-06. ICML 2024, arXiv:2406.10774. https://arxiv.org/abs/2406.10774 [CANONICAL]
  - Per-page min/max + per-query criticality top-K; 2.23× attention speedup.
- `[Loki]` *Loki: Low-rank Keys for Efficient Sparse Attention*. Singhania et al. (UMD). 2024-06. NeurIPS 2024, arXiv:2406.02542. https://arxiv.org/abs/2406.02542 [CANONICAL]
  - Top-K key selection in PCA-projected key space.
- `[InfiniGen]` *InfiniGen: Efficient Generative Inference of LLMs with Dynamic KV Cache Management*. Lee et al. (SNU). 2024-06. OSDI 2024, arXiv:2406.19707. https://arxiv.org/abs/2406.19707 [CANONICAL]
  - Speculatively prefetch essential KV pages from CPU; 3× over offload.
- `[Ada-KV]` *Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation*. Feng et al. (USTC). 2024-07. NeurIPS 2025, arXiv:2407.11550. https://arxiv.org/abs/2407.11550 [CANONICAL]
  - Per-head adaptive budget; plug-in upgrade to SnapKV/PyramidKV.
- `[KVMerger]` *Adaptive KV Cache Merging for LLMs on Long-Context Tasks*. Wang et al. 2024-07. arXiv:2407.08454. https://arxiv.org/abs/2407.08454 [CANONICAL]
  - Token-merging instead of dropping; pivotal-token Gaussian-weighted merge sets.
- `[KV-Bench]` *KV Cache Compression, But What Must We Give in Return?*. EMNLP 2024. arXiv:2407.01527. https://arxiv.org/abs/2407.01527 [REFERENCE]
  - Honest benchmark across 10+ methods, 7 task families.
- `[NACL]` *NACL: A General and Effective KV Cache Eviction Framework*. Chen et al. (Baidu). 2024-08. ACL 2024. https://aclanthology.org/2024.acl-long.428/ [CANONICAL]
  - Proxy-token attention statistics + diversified random eviction; 5× cache reduction.
- `[RetrievalAttention]` *RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval*. Liu et al. (Microsoft). 2024-09. arXiv:2409.10516. https://arxiv.org/abs/2409.10516 [CANONICAL]
  - CPU ANN over offloaded KV; 1–3% of cache accessed.
- `[DuoAttention]` *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads*. Xiao et al. (MIT-HAN-Lab). 2024-10. ICLR 2025, arXiv:2410.10819. https://arxiv.org/abs/2410.10819 [CANONICAL]
  - Heads split into retrieval (full) vs streaming (sink+window); 2.55× MHA reduction.
- `[KV-Survey-2024]` *A Survey on LLM Acceleration based on KV Cache Management*. 2024-12. arXiv:2412.19442. https://arxiv.org/abs/2412.19442 [REFERENCE]
- `[KV-Survey-2025]` *Key, Value, Compress: A Systematic Exploration of KV Cache Compression Techniques*. 2025-03. arXiv:2503.11816. https://arxiv.org/abs/2503.11816 [REFERENCE] [2026-NEW]

#### Past 12 months SOTA — KV cache

- `[DroidSpeak]` *DroidSpeak: KV Cache Sharing for Cross-LLM Communication*. Liu et al. (UChicago/Microsoft). 2024-11. arXiv:2411.02820. https://arxiv.org/abs/2411.02820 [EMERGING]
  - 4× throughput, 3.1× TTFT in compound-AI/agent settings.
- `[SCBench]` *SCBench: A KV Cache-Centric Analysis of Long-Context Methods*. Microsoft/HKUST. 2024-12. ICLR 2025, arXiv:2412.10319. https://arxiv.org/abs/2412.10319 [SOTA]
  - Evaluates compression/retrieval/loading across full KV lifecycle.
- `[Strata]` *Strata: Hierarchical Context Caching for Long Context LM Serving*. 2025-08. arXiv:2508.18572. https://arxiv.org/abs/2508.18572 [EMERGING] [2026-NEW]
  - GPU-assisted I/O against KV fragmentation; cache-aware request scheduling.
- `[LMCache]` *LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference*. LMCache team / U. Chicago. 2025-10. arXiv:2510.09665. https://arxiv.org/abs/2510.09665 [SOTA] [2026-NEW]
  - Reference open-source KV layer; 8+ backends; 15× throughput in shared-prefix workloads.
- `[Pitfalls-KV]` *The Pitfalls of KV Cache Compression*. 2025-10. arXiv:2510.00231. https://arxiv.org/abs/2510.00231 [SOTA] [2026-NEW]
  - Critical evaluation: instructions silently dropped under compression; system-prompt leakage.
- `[Rethinking-KV]` *Rethinking Key-Value Cache Compression Techniques for LLM Serving*. MLSys 2025. arXiv:2503.24000. https://arxiv.org/abs/2503.24000 [SOTA] [2026-NEW]
  - Eviction methods don't compose with FA/PagedAttention as deployed.
- `[LookaheadKV]` *LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future*. Samsung Research. 2025. arXiv:2603.10899. https://arxiv.org/abs/2603.10899 [UNVERIFIED] [EMERGING] [2026-NEW]
  - Trained "lookahead" tokens + LoRA predict importance scores via KL.
- `[ForesightKV]` *ForesightKV: Optimizing KV Cache Eviction for Reasoning Models*. 2025. arXiv:2602.03203. https://arxiv.org/abs/2602.03203 [UNVERIFIED] [EMERGING] [2026-NEW]
  - Lightweight scoring model trained to predict long-term contribution.
- `[LRAgent]` *LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents*. 2025. arXiv:2602.01053. https://arxiv.org/abs/2602.01053 [UNVERIFIED] [EMERGING] [2026-NEW]
  - Tree-based KV-cache + LoRA-adapter co-management.
- `[CMX]` *NVIDIA BlueField-4 Powered CMX (Inference Context Memory Storage Platform)*. NVIDIA. 2026-01. https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/ [REFERENCE] [2026-NEW]
  - 4-tier KV pyramid down to NVMe-resident persistent KV.

#### Lineage — low-rank / latent / merging (KV-related)

- `[MLA-V2]` *DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model*. DeepSeek-AI. 2024-05. arXiv:2405.04434. https://arxiv.org/abs/2405.04434 [CANONICAL]
  - Multi-head Latent Attention; ~93.3% KV reduction vs MHA; 5.76× generation throughput.
- `[YOCO]` *You Only Cache Once: Decoder-Decoder Architectures for Language Models*. Sun et al. (Microsoft). 2024-05. NeurIPS 2024 oral, arXiv:2405.05254. https://arxiv.org/abs/2405.05254 [CANONICAL]
  - Cache one global KV; ~L× memory reduction; 1M-context prefill in seconds.
- `[MiniCache]` *MiniCache: KV Cache Compression in Depth Dimension for LLMs*. Liu et al. (Monash). 2024-05. NeurIPS 2024, arXiv:2405.14366. https://arxiv.org/abs/2405.14366 [CANONICAL]
  - Cross-layer KV merging via magnitude+direction; 1.53× compression alone.
- `[Eigen-Attn]` *Eigen Attention: Attention in Low-Rank Space for KV Cache Compression*. Saxena et al. 2024-08. EMNLP Findings 2024, arXiv:2408.05646. https://arxiv.org/abs/2408.05646 [CANONICAL]
  - Joint low-rank approx of Q,K,V; 40% KV reduction.
- `[XQuant]` *XQuant: Achieving Ultra-Low Bit KV Cache Quantization with Cross-Layer Compression*. 2025. [UNVERIFIED] [EMERGING] [2026-NEW]
  - Cross-layer KV compression / sharing.

#### Past 12 months SOTA — tiered KV / disagg KV

- `[Mooncake]` *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*. Qin et al. (Moonshot AI / Tsinghua). 2024-06. FAST 2025, arXiv:2407.00079. https://arxiv.org/abs/2407.00079 [CANONICAL] [2026-NEW]
  - Distributed KVCache pool over CPU/DRAM/SSD/NIC; Conductor scheduler; Transfer Engine.
- `[Mooncake-Store]` *Mooncake Store / Transfer Engine*. open-sourced 2024-11 / 2025-03. https://github.com/kvcache-ai/Mooncake [REFERENCE] [2026-NEW]
- `[Dynamo]` *NVIDIA Dynamo: Distributed Inference Framework*. NVIDIA. 2025-03 (GTC). https://docs.nvidia.com/dynamo/ [REFERENCE] [2026-NEW]
  - KVBM + NIXL transfer + KV-aware Smart Router (radix-tree); disaggregated by default.
- `[Dynamo-launch]` *Introducing NVIDIA Dynamo*. NVIDIA. GTC 2025-03. https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/ [REFERENCE] [2026-NEW]
- `[NIXL]` *NVIDIA Inference Transfer Library*. NVIDIA / ai-dynamo. 2025-03. https://github.com/ai-dynamo/nixl [REFERENCE] [2026-NEW]
  - Transport for async KV transfers across NVLink/IB/RoCE/GDS/S3.
- `[Dynamo-NIXL]` *Enhancing Distributed Inference Performance with NIXL*. NVIDIA Tech Blog. https://developer.nvidia.com/blog/enhancing-distributed-inference-performance-with-the-nvidia-inference-transfer-library/ [REFERENCE] [2026-NEW]
- `[Dynamo-KVBM]` *KVBM Architecture*. NVIDIA Dynamo docs. https://docs.nvidia.com/dynamo/latest/kvbm/kvbm_architecture.html [REFERENCE] [2026-NEW]
- `[AIBrix]` *AIBrix: Towards Scalable, Cost-Effective LLM Inference Infrastructure*. Bytedance. 2025-04. arXiv:2504.03648. https://arxiv.org/abs/2504.03648 [CANONICAL] [2026-NEW]
  - Distributed KV cache, multi-tier manager, pluggable eviction; 50% throughput improvement.
- `[SGLang-HiCache]` *SGLang HiCache: Fast Hierarchical KV Caching*. SGLang/LMSYS. 2025-09. https://www.lmsys.org/blog/2025-09-10-sglang-hicache/ [REFERENCE] [2026-NEW]
  - HiRadixTree as page table over GPU/CPU/storage; write-through/back policies.
- `[llm-d-FS]` *Native KV Cache Offloading to Any Filesystem with llm-d*. Red Hat/IBM. 2025. https://llm-d.ai/blog/native-kv-cache-offloading-to-any-file-system-with-llm-d [REFERENCE] [2026-NEW]
  - Filesystem-agnostic POSIX connector with async I/O and GPU DMA.
- `[LMCache-GKE]` *LMCache on Google Kubernetes Engine*. 2025-10. https://blog.lmcache.ai/en/2025/10/07/lmcache-on-google-kubernetes-engine-boosting-llm-inference-performance-with-kv-cache-on-tiered-storage/ [REFERENCE] [2026-NEW]
- `[MLA-V3]` See `[DeepSeek-V3-FP8]` above. MLA at scale (671B MoE, 37B activated); ~32× compression ratio.
- `[ExpertChoice]` See MoE section. Cross-references KV via low-rank principles.


### Batching and scheduling

#### Lineage

- `[ORCA]` *Orca: A Distributed Serving System for Transformer-Based Generative Models*. Yu, Jeong, Kim, Kim, Chun (Seoul Nat'l U/FriendliAI). 2022-07. OSDI 2022. https://www.usenix.org/conference/osdi22/presentation/yu [CANONICAL]
  - Iteration-level scheduling and selective batching; canonical origin of continuous batching.
- `[FlexGen]` *FlexGen: High-Throughput Generative Inference of LLMs with a Single GPU*. Sheng et al. (Stanford/UCB). 2023-03. ICML 2023, arXiv:2303.06865. https://arxiv.org/abs/2303.06865 [CANONICAL]
  - LP-derived schedule for tensor placement; foundational for offload lineage.
- `[FastServe]` *Fast Distributed Inference Serving for Large Language Models*. Wu, Bai, Liu, Zhang, Yi, Jin (PKU). 2023-05. arXiv:2305.05920. https://arxiv.org/abs/2305.05920 [CANONICAL]
  - Skip-join multi-level feedback queue for token-level preemption.
- `[Sarathi]` *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*. Agrawal, Panwar, Mohan, Kwatra, Gulavani, Ramjee (MSR India/GaTech). 2023-08. arXiv:2308.16369. https://arxiv.org/abs/2308.16369 [CANONICAL]
  - Chunked-prefill primitive; precursor to Sarathi-Serve.
- `[DS-FastGen]` *DeepSpeed-FastGen: High-throughput Text Generation for LLMs*. Holmes et al. (Microsoft). 2024-01. arXiv:2401.08671. https://arxiv.org/abs/2401.08671 [CANONICAL]
  - Dynamic SplitFuse; constant-forward-size composer.
- `[DistServe]` *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving*. Zhong, Liu, Chen, Hu, Zhu, Liu, Jin, Zhang (PKU/UCSD/StepFun). 2024-01. OSDI 2024, arXiv:2401.09670. https://arxiv.org/abs/2401.09670 [CANONICAL]
  - Goodput formalism; disaggregated scheduling with co-optimized parallelism.
- `[VTC]` *Fairness in Serving Large Language Models*. Sheng, Cao, Li, Zhu, Zhuo, Gonzalez, Stoica. 2023-12. OSDI 2024, arXiv:2401.00588. https://arxiv.org/abs/2401.00588 [CANONICAL]
  - Virtual Token Counter; first formally fair LLM scheduler with 2× bound.
- `[Llumnix]` *Llumnix: Dynamic Scheduling for Large Language Model Serving*. Sun et al. (Alibaba). 2024-07. OSDI 2024, arXiv:2406.03243. https://arxiv.org/abs/2406.03243 [CANONICAL]
  - KV-cache live migration across replicas; OS-style context switch of LLM serving.
- `[Sarathi-Serve]` *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*. Agrawal et al. (Microsoft/GaTech). 2024-03. OSDI 2024, arXiv:2403.02310. https://arxiv.org/abs/2403.02310 [CANONICAL]
  - Production-ready chunked prefill + stall-free schedule; defines TTFT/ITL knobs.
- `[Andes]` *Andes: Defining and Enhancing Quality-of-Experience in LLM-Based Text Streaming Services*. Liu, Wu, et al. 2024-04. arXiv:2404.16283. https://arxiv.org/abs/2404.16283 [CANONICAL]
  - Token-granularity preemptive scheduler with QoE function.
- `[LTR]` *Efficient LLM Scheduling by Learning to Rank*. Fu, Zhu, Su, Sheng, Yang, Jiao, Zhang. 2024-08. NeurIPS 2024, arXiv:2408.15792. https://arxiv.org/abs/2408.15792 [CANONICAL]
  - Predict pairwise ranks (not lengths); 2.8× chatbot latency reduction.
- `[POD-Attn]` *POD-Attention: Unlocking Full Prefill-Decode Overlap*. Kamath, Mohan, Panwar, Ramjee, et al. 2024-10. ASPLOS 2025, arXiv:2410.18038. https://arxiv.org/abs/2410.18038 [CANONICAL]
  - Single GPU kernel runs prefill-attention and decode-attention on same SM; 59% faster.
- `[Mnemosyne]` *Mnemosyne: Parallelization Strategies for Multi-Million Context Length LLM Inference*. Microsoft. 2024-09. arXiv:2409.17264. https://arxiv.org/abs/2409.17264 [CANONICAL]
  - Adaptive chunking + Sequence Pipeline Parallelism + KV Cache Parallelism.

#### Past 12 months SOTA — batching/scheduling

- `[vLLM-V1]` *vLLM V1 Alpha: A major upgrade to vLLM's core architecture*. vLLM team. 2025-01. https://blog.vllm.ai/2025/01/27/v1-alpha-release.html [SOTA] [2026-NEW]
  - Unified scheduler; prefill/decode phase distinction abolished; chunked prefill default.
- `[vLLM-V1-RH]` *vLLM V1: A Major Upgrade*. Red Hat developer write-up. 2025-01. https://developers.redhat.com/articles/2025/01/28/vllm-v1-a-major-upgrade-vllms-core-architecture [REFERENCE] [2026-NEW]
- `[SGLang-v0.4]` *SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs*. LMSYS. 2024-12. https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/ [REFERENCE]
  - CPU-side scheduling overlapped one batch ahead; Rust router with approximate radix tree.
- `[DLPM]` *Locality-aware Fair Scheduling in LLM Serving*. Cao, Wang, Mao, Hsu et al. 2025-01. arXiv:2501.14312. https://arxiv.org/abs/2501.14312 [SOTA] [2026-NEW]
  - DLPM/D²LPM: fairness ⊥ prefix-locality tension; up to 2.87× over VTC.
- `[OnlineKV]` *Online Scheduling for LLM Inference with KV Cache Constraints*. Jaillet et al. 2025-02. arXiv:2502.07115. http://www.mit.edu/~jaillet/general/2502.07115v5.pdf [SOTA] [2026-NEW]
  - Competitive-ratio analysis with KV memory as explicit resource.
- `[QPredLLM]` *Queueing, Predictions, and Large Language Models*. arXiv:2503.07545. 2025-03. https://arxiv.org/abs/2503.07545 [REFERENCE] [2026-NEW]
- `[ThroughputOpt]` *Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents*. 2025-04. arXiv:2504.07347. https://arxiv.org/abs/2504.07347 [SOTA] [2026-NEW]
  - SLAI scheduler for TBT-deadline awareness.
- `[SLOs-Serve]` *SLOs-Serve: Optimized Serving of Multi-SLO LLMs*. 2025-04. MLSys 2025, arXiv:2504.08784. https://arxiv.org/abs/2504.08784 [SOTA] [2026-NEW]
  - 2.2× per-GPU capacity over prior art.
- `[JITServe]` *JITServe: SLO-aware LLM Serving with Imprecise Request Information*. Zhang et al. 2025-04. NSDI 2026, arXiv:2504.20068. https://arxiv.org/abs/2504.20068 [SOTA] [2026-NEW]
  - 1.4–6.3× goodput; grouped margin goodput allocation.
- `[FlowKV]` *FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer*. 2025-04. arXiv:2504.03775. https://arxiv.org/abs/2504.03775 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[Arrow]` *Arrow: Adaptive Scheduling Mechanisms for Disaggregated LLM Inference*. 2025-05. arXiv:2505.11916. https://arxiv.org/abs/2505.11916 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[SOLA]` *SOLA: Optimizing SLO Attainment for LLM Serving with State-Aware Scheduling*. Tsinghua/Infinigence/SJTU/PKU. 2025-05. MLSys 2025. https://mlsys.org/virtual/2025/poster/3231 [SOTA] [2026-NEW]
  - SLO attainment 45.5% → 99.4% on identical hardware.
- `[Helix]` *Helix: Serving LLMs over Heterogeneous GPUs and Network via Max-Flow*. Mei, Tang, Vinayak, Zaharia, Zhang. 2024-06. ASPLOS 2025, arXiv:2406.01566. https://arxiv.org/abs/2406.01566 [SOTA]
  - Max-flow MILP joint placement + scheduling; 3.3× throughput on heterogeneous clusters.
- `[KVFlow]` *KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows*. 2025-07. arXiv:2507.07400. https://arxiv.org/abs/2507.07400 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[BucketServe]` *Bucket-based dynamic batching*. 2025-07. arXiv:2507.17120. https://arxiv.org/abs/2507.17120 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[Optimal-LLM-Sched]` *Optimal Scheduling Algorithms for LLM Inference: Theory and Practice*. 2025-08. arXiv:2508.01002. https://arxiv.org/abs/2508.01002 [SOTA] [2026-NEW]
  - Throughput-optimality results; RAD scheduler.
- `[HotPrefix]` *HotPrefix: Hotness-Aware KV Cache Scheduling*. 2025. https://dl.acm.org/doi/10.1145/3749168 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[Multi-bin]` *Multi-Bin Batching for Increasing LLM Inference Throughput*. 2024-12. arXiv:2412.04504. https://arxiv.org/abs/2412.04504 [SOTA]
- `[Don't-Stop-Me-Now]` *Embedding-based scheduling for LLMs*. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/9eb8b5ccb0de594a16548f7c058fdadf-Paper-Conference.pdf [UNVERIFIED] [EMERGING] [2026-NEW]
- `[Semi-Clairvoyant]` *Semi-Clairvoyant Scheduling of Speculative Decoding Requests*. IJCAI 2025. https://www.ijcai.org/proceedings/2025/0951.pdf [UNVERIFIED] [EMERGING] [2026-NEW]
- `[FairBatching]` *FairBatching: Fairness-Aware Batch Formation*. 2025-10. arXiv:2510.14392. https://arxiv.org/abs/2510.14392 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[Hummingbird]` *Hummingbird: SLO-Oriented GPU Preemption at Microsecond-scale*. 2026-01. arXiv:2601.04071. https://arxiv.org/abs/2601.04071 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[NV-Dynamo-04]` *Dynamo 0.4: 4× Faster Performance, SLO-based Autoscaling, Real-time Observability*. NVIDIA Tech Blog. 2025. https://developer.nvidia.com/blog/dynamo-0-4-delivers-4x-faster-performance-slo-based-autoscaling-and-real-time-observability/ [REFERENCE] [2026-NEW]

### Prefill–decode disaggregation

#### Lineage

- `[Splitwise]` *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*. Patel, Choukse, Zhang, Shah, Goiri, Maleki, Bianchini (Microsoft). 2023-11. ISCA 2024 (best paper), arXiv:2311.18677. https://arxiv.org/abs/2311.18677 [CANONICAL]
  - First systems paper articulating PD as deployment problem; 1.4× throughput at -20% cost.
- `[TetriInfer]` *Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads*. Hu et al. (PKU/ByteDance). 2024-01. arXiv:2401.11181. https://arxiv.org/abs/2401.11181 [CANONICAL]
  - Three-pillar: chunked prompt + PD disagg + two-level scheduler.
- `[DejaVu-FT]` *DéjàVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving*. Strati, McAllister, Phanishayee, Tarnawski, Klimovic (ETH/MSR). 2024-03. ICML 2024, arXiv:2403.01876. https://arxiv.org/abs/2403.01876 [CANONICAL]
  - Prompt-token disaggregation, microbatch swap, per-stage KV replication; FT reference.
- `[P/D-Serve]` *P/D-Serve: Serving Disaggregated Large Language Model at Scale*. Huawei Cloud. 2024-08. arXiv:2408.08147. https://arxiv.org/html/2408.08147v1 [UNVERIFIED] [REFERENCE]

#### Past 12 months SOTA — disaggregation

- `[BeyondBuzz]` *Beyond the Buzz: A Pragmatic Take on Inference Disaggregation*. Mitra, Borkar, et al. (NVIDIA). 2025-06. arXiv:2506.05508. https://arxiv.org/abs/2506.05508 [SOTA] [2026-NEW]
  - First systematic NVIDIA-internal study; disaggregation pays off for prefill-heavy + larger models.
- `[Nexus]` *Nexus: Proactive Intra-GPU Disaggregation of Prefill and Decode in LLM Serving*. 2025-07. arXiv:2507.06608. https://arxiv.org/abs/2507.06608 [SOTA] [2026-NEW]
  - Intra-GPU disagg via dynamic SM allocation; up to 2× over SGLang.
- `[TaiChi]` *Prefill-Decode Aggregation or Disaggregation? Unifying Both for Goodput-Optimized LLM Serving*. Wang et al. 2025-08. arXiv:2508.01989. https://arxiv.org/abs/2508.01989 [SOTA] [2026-NEW]
  - SLO-regime dependence; aggregation under tight TTFT, disagg under tight TPOT.
- `[HeteroScale]` *Taming the Chaos: Coordinated Autoscaling for Heterogeneous and Disaggregated LLM Inference*. ByteDance. 2025-08. arXiv:2508.19559. https://arxiv.org/abs/2508.19559 [SOTA] [2026-NEW]
  - Decode TPS as best autoscaling signal; +26.6 pp utilization on tens of thousands of GPUs.
- `[Hetero-PD]` *Disaggregated Prefill and Decoding Inference System for Heterogeneous GPUs*. 2025-09. arXiv:2509.17542. https://arxiv.org/abs/2509.17542 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[PDTrim]` *PDTrim (originally cited as PD-Pruning): Targeted Pruning for Prefill-Decode Disaggregation*. 2025-09. arXiv:2509.04467. https://arxiv.org/abs/2509.04467 [EMERGING] [2026-NEW]
  - Verified system name is PDTrim.
- `[DOPD]` *DOPD: A Dynamic PD-Disaggregation Architecture for Maximizing Goodput*. 2025-11. arXiv:2511.20982. https://arxiv.org/abs/2511.20982 [UNVERIFIED] [SOTA] [2026-NEW]
  - Goodput +1.5×; P90 TTFT -67.5% vs vLLM/DistServe.
- `[DuetServe]` *DuetServe: Harmonizing Prefill and Decode for LLM Serving*. 2025-11. arXiv:2511.04791. https://arxiv.org/abs/2511.04791 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[TokenScale]` *TokenScale: Timely and Accurate Autoscaling for Disaggregated LLM Serving with Token Velocity*. 2025-12. arXiv:2512.03416. https://arxiv.org/abs/2512.03416 [UNVERIFIED] [SOTA] [2026-NEW]
  - SLO attainment 50–88% → 80–96%; cost -4–14% vs DistServe/BlitzScale/AIBrix.
- `[TraCT]` *TraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale*. 2025-12. arXiv:2512.18194. https://arxiv.org/html/2512.18194v1 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[TogetherCPD]` *Cache-aware Prefill-Decode Disaggregation*. Together AI engineering blog. 2025. https://www.together.ai/blog/cache-aware-disaggregated-inference [REFERENCE] [2026-NEW]
  - Three-tier (pre-prefill / prefill / decode); +40% throughput.
- `[PrefillShare]` *PrefillShare: A Shared Prefill Module for KV Reuse in Multi-LLM Disaggregated Serving*. 2026-02. arXiv:2602.12029. https://arxiv.org/abs/2602.12029 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[PrefillAaS]` *Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter*. 2026-04. arXiv:2604.15039. https://arxiv.org/html/2604.15039v1 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[SGLang-LargeEP]` *Deploying DeepSeek with PD Disaggregation and Large-Scale EP on 96 H100 GPUs*. LMSYS. 2025-05. https://www.lmsys.org/blog/2025-05-05-large-scale-ep/ [REFERENCE] [2026-NEW]
- `[SGLang-K2-128]` *Deploying Kimi K2 with PD Disaggregation and Large-Scale EP on 128 H200 GPUs*. LMSYS. 2025-07. https://www.lmsys.org/blog/2025-07-20-k2-large-scale-ep/ [REFERENCE] [2026-NEW]
- `[SGLang-GB200-1]` *Deploying DeepSeek on GB200 NVL72 with PD and Large-Scale EP (Part I)*. LMSYS. 2025-06. https://www.lmsys.org/blog/2025-06-16-gb200-part-1/ [REFERENCE] [2026-NEW]
- `[SGLang-GB200-2]` *Deploying DeepSeek on GB200 NVL72 (Part II)*. LMSYS. 2025-09. https://www.lmsys.org/blog/2025-09-25-gb200-part-2/ [REFERENCE] [2026-NEW]
- `[Perplexity-TE]` *Disaggregated Prefill and Decode at Perplexity*. https://research.perplexity.ai/articles/disaggregated-prefill-and-decode [REFERENCE] [2026-NEW]
- `[Perplexity-RDMA]` *RDMA Point-to-point Communication for LLM Systems*. Perplexity. https://research.perplexity.ai/articles/rdma-point-to-point-communication-for-llm-systems [REFERENCE] [2026-NEW]
- `[DeepSeek-V3-Insights]` *Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures*. DeepSeek-AI. 2025-05. ISCA 2025, arXiv:2505.09343. https://arxiv.org/abs/2505.09343 [REFERENCE] [2026-NEW]
- `[DS-Inference-Overview]` *DeepSeek-V3/R1 Inference System Overview*. DeepSeek (Open Source Week Day 6). 2025-02. https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md [REFERENCE] [2026-NEW]
- `[TRT-LLM-Disagg]` *Disaggregated Serving in TensorRT-LLM*. NVIDIA. https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html [REFERENCE] [2026-NEW]
- `[vLLM-Disagg-Doc]` *Disaggregated Prefill in vLLM*. vLLM docs. https://docs.vllm.ai/en/latest/features/disagg_prefill/ [REFERENCE] [2026-NEW]
- `[PyTorch-DisaggInfer]` *Disaggregated Inference at Scale with PyTorch + vLLM*. PyTorch blog. https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/ [REFERENCE] [2026-NEW]


### MoE inference

#### Lineage — model

- `[Shazeer-2017]` *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. Shazeer et al. (Google Brain). 2017-01. ICLR 2017, arXiv:1701.06538. https://arxiv.org/abs/1701.06538 [CANONICAL]
  - The MoE-as-a-layer paper; top-k gating, load-balance loss, noisy gating.
- `[GShard]` *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*. Lepikhin et al. (Google). 2020-06. ICLR 2021, arXiv:2006.16668. https://arxiv.org/abs/2006.16668 [CANONICAL]
  - First production transformer MoE; established dispatch/combine pattern.
- `[Switch]` *Switch Transformers: Scaling to Trillion Parameter Models*. Fedus, Zoph, Shazeer (Google). 2021-01. JMLR 2022, arXiv:2101.03961. https://arxiv.org/abs/2101.03961 [CANONICAL]
  - Top-1 routing simplification; canonical "MoE = capacity for free" reference.
- `[GLaM]` *GLaM: Efficient Scaling of Language Models with Mixture-of-Experts*. Du et al. (Google). 2021-12. ICML 2022, arXiv:2112.06905. https://arxiv.org/abs/2112.06905 [CANONICAL]
  - 1.2T-param 64-expert decoder; ½ inference FLOPs of GPT-3 at better quality.
- `[ST-MoE]` *ST-MoE: Designing Stable and Transferable Sparse Expert Models*. Zoph et al. 2022-02. arXiv:2202.08906. https://arxiv.org/abs/2202.08906 [CANONICAL]
  - 269B sparse with router-z-loss; first to top SuperGLUE/ARC.
- `[ExpertChoice]` *Mixture-of-Experts with Expert Choice Routing*. Zhou et al. (Google). 2022-02. NeurIPS 2022, arXiv:2202.09368. https://arxiv.org/abs/2202.09368 [CANONICAL]
  - Inverts routing: experts pick top-k tokens; eliminates dropping.
- `[Mixtral-8x7B]` *Mixtral of Experts*. Jiang et al. (Mistral AI). 2024-01. arXiv:2401.04088. https://arxiv.org/abs/2401.04088 [CANONICAL]
  - 47B/13B-active 8-expert; first mainstream open-weight transformer MoE.
- `[Mixtral-8x22B]` *Cheaper, Better, Faster, Stronger*. Mistral AI. 2024-04. https://mistral.ai/news/mixtral-8x22b [REFERENCE]
- `[Grok-1]` *Open Release of Grok-1*. xAI. 2024-03. https://x.ai/news/grok-os [REFERENCE]
- `[DBRX]` *Introducing DBRX*. Mosaic/Databricks. 2024-03. https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm [REFERENCE]
- `[Snowflake-Arctic]` *Snowflake Arctic: Open Efficient Foundation Models*. Snowflake. 2024-04. https://www.snowflake.com/en/blog/arctic-open-efficient-foundation-language-models-snowflake/ [REFERENCE]
- `[DeepSeekMoE]` *DeepSeekMoE: Towards Ultimate Expert Specialization in MoE Language Models*. Dai et al. (DeepSeek). 2024-01. ACL 2024, arXiv:2401.06066. https://arxiv.org/abs/2401.06066 [CANONICAL]
  - Fine-grained segmentation + shared-expert isolation; foundation of V2/V3/R1.
- `[Loss-Free-Balancing]` *Auxiliary-Loss-Free Load Balancing Strategy for MoE*. Wang et al. (DeepSeek). 2024-08. arXiv:2408.15664. https://arxiv.org/abs/2408.15664 [CANONICAL]
- `[OLMoE]` *OLMoE: Open Mixture-of-Experts Language Models*. Muennighoff et al. (AI2/Contextual). 2024-09. arXiv:2409.02060. https://arxiv.org/abs/2409.02060 [CANONICAL]
  - Fully open (data, code, ckpts); canonical reproducible MoE baseline.
- `[Phi-3.5-MoE]` *Phi-3.5-MoE-instruct*. Microsoft. 2024-08. https://huggingface.co/microsoft/Phi-3.5-MoE-instruct [REFERENCE]
- `[Granite-3-MoE]` *IBM Granite 3.0*. IBM. 2024-10. https://www.ibm.com/new/announcements/ibm-granite-3-0-open-state-of-the-art-enterprise-models [REFERENCE]
- `[DeepSeek-R1]` *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL*. DeepSeek-AI. 2025-01. arXiv:2501.12948. https://arxiv.org/abs/2501.12948 [CANONICAL] [2026-NEW]
  - V3 backbone (671B/37B, 256+1) trained for reasoning.
- `[Llama-4]` *The Llama 4 Herd*. Meta. 2025-04. https://ai.meta.com/blog/llama-4-multimodal-intelligence/ [REFERENCE] [2026-NEW]
  - Scout (109B/17B) and Maverick (400B/17B); Meta's first MoE.
- `[Qwen3]` *Qwen3 Technical Report*. Qwen Team (Alibaba). 2025-05. arXiv:2505.09388. https://arxiv.org/abs/2505.09388 [REFERENCE] [2026-NEW]
- `[Step-3]` *Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding*. StepFun. 2025-07. arXiv:2507.19427. https://arxiv.org/abs/2507.19427 [SOTA] [2026-NEW]
  - 321B VLM with MFA + Attention-FFN Disaggregation; 4039 tok/s/GPU decode.
- `[Kimi-K2]` *Kimi K2: Open Agentic Intelligence*. Moonshot AI. 2025-07. arXiv:2507.20534. https://arxiv.org/abs/2507.20534 [SOTA] [2026-NEW]
  - 1.04T/32B-active, 384 routed experts, MLA, MuonClip optimizer.
- `[GPT-OSS-Deploy]` *GPT-OSS Deployment Analysis*. 2025-08. arXiv:2508.16700. https://arxiv.org/abs/2508.16700 [REFERENCE] [2026-NEW]
- `[Qwen3-Next-NVIDIA]` *Qwen3-Next 80B-A3B*. Qwen + NVIDIA. 2025-09. https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform/ [REFERENCE] [2026-NEW]
- `[DeepSeek-V3.2]` *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*. DeepSeek-AI. 2025-12. arXiv:2512.02556. https://arxiv.org/abs/2512.02556 [SOTA] [2026-NEW]
  - Adds DSA (lightning indexer + top-k); reduces attention to O(L·k) at 128K.

#### Lineage — systems

- `[FastMoE]` *FastMoE: A Fast Mixture-of-Expert Training System*. He, Qiu, Zeng, Yang, Zhai, Tang. 2021-03. arXiv:2103.13262. https://arxiv.org/abs/2103.13262 [CANONICAL]
- `[DeepSpeed-MoE]` *DeepSpeed-MoE: Advancing MoE Inference and Training to Next-Generation AI Scale*. Rajbhandari et al. (Microsoft). 2022-01. ICML 2022, arXiv:2201.05596. https://arxiv.org/abs/2201.05596 [CANONICAL]
  - 7.3× lower MoE inference latency.
- `[Tutel]` *Tutel: Adaptive Mixture-of-Experts at Scale*. Hwang et al. (Microsoft). 2022-06. MLSys 2023, arXiv:2206.03382. https://arxiv.org/abs/2206.03382 [CANONICAL]
- `[HetuMoE]` *HetuMoE: An Efficient Trillion-scale MoE Distributed Training System*. Nie, Zhao et al. (PKU). 2022-03. arXiv:2203.14685. https://arxiv.org/abs/2203.14685 [CANONICAL]
- `[Lina]` *Accelerating Distributed MoE Training and Inference with Lina*. Li et al. 2022-10. USENIX ATC 2023, arXiv:2210.17223. https://arxiv.org/abs/2210.17223 [CANONICAL]
  - All-to-all prioritization over allreduce; 1.73× / 1.63× speedups.
- `[MegaBlocks]` *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts*. Gale, Narayanan, Young, Zaharia. 2022-11. MLSys 2023, arXiv:2211.15841. https://arxiv.org/abs/2211.15841 [CANONICAL]
  - MoE as block-sparse GEMMs (BCSR); basis of DBRX.
- `[Brainstorm]` *Optimizing Dynamic Neural Networks with Brainstorm*. Cui et al. (MSRA). OSDI 2023. https://www.usenix.org/system/files/osdi23-cui.pdf [CANONICAL]
- `[Pre-gated-MoE]` *Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable MoE Inference*. 2023-08. ISCA 2024, arXiv:2308.12066. https://arxiv.org/abs/2308.12066 [CANONICAL]
- `[Mixtral-Offload]` *Fast Inference of Mixture-of-Experts Language Models with Offloading*. Eliseev, Mazur. 2023-12. arXiv:2312.17238. https://arxiv.org/abs/2312.17238 [CANONICAL]
- `[MoE-Infinity]` *MoE-Infinity: Activation-Aware Expert Offloading*. Xue et al. (Edinburgh). 2024-01. arXiv:2401.14361. https://arxiv.org/abs/2401.14361 [CANONICAL]
- `[ExFlow]` *Exploiting Inter-Layer Expert Affinity for Accelerating MoE Inference*. 2024-01. arXiv:2401.08383. https://arxiv.org/abs/2401.08383 [CANONICAL]
- `[Fiddler]` *Fiddler: CPU-GPU Orchestration for Fast Inference of MoE Models*. Kamahori, Tang, Gu, Zhu, Kasikci (UW). 2024-02. ICLR 2025, arXiv:2402.07033. https://arxiv.org/abs/2402.07033 [CANONICAL]
- `[ExpertFlow-2024]` *ExpertFlow: Predictive Expert Caching and Token Scheduling*. 2024-10. arXiv:2410.17954. https://arxiv.org/abs/2410.17954 [CANONICAL]
- `[ProMoE]` *ProMoE: Fast MoE-based LLM Serving using Proactive Caching*. 2024-10. arXiv:2410.22134. https://arxiv.org/abs/2410.22134 [CANONICAL]
- `[HOBBIT]` *HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference*. 2024-11. arXiv:2411.01433. https://arxiv.org/abs/2411.01433 [CANONICAL]

#### Past 12 months SOTA — MoE systems

- `[DeepEP]` *DeepEP: Efficient Expert-Parallel Communication Library*. DeepSeek. 2025-02. https://github.com/deepseek-ai/DeepEP [SOTA] [2026-NEW]
  - Two-mode kernels (HT/LL); intranode NVLink + internode RDMA; native FP8 dispatch.
- `[DualPipe]` *DualPipe: Bidirectional Pipeline Parallelism for V3/R1 Training*. DeepSeek. 2025-02. https://github.com/deepseek-ai/DualPipe [SOTA] [2026-NEW]
  - Forward and backward in opposing directions; minimizes pipeline bubbles.
- `[EPLB]` *Expert Parallelism Load Balancer*. DeepSeek. 2025-02. https://github.com/deepseek-ai/EPLB [SOTA] [2026-NEW]
  - Redundant-experts strategy; hierarchical (prefill) and global (decode) policies.
- `[LPLB]` *LPLB: Linear-Programming-based Expert-Parallel Load Balancer*. DeepSeek. 2025. https://github.com/deepseek-ai/LPLB [EMERGING] [2026-NEW]
- `[DeepGEMM]` *DeepGEMM: Clean and Efficient FP8 GEMM Kernels with Fine-Grained Scaling*. DeepSeek. 2025-02. https://github.com/deepseek-ai/DeepGEMM [SOTA] [2026-NEW]
  - Grouped FP8 GEMMs; ~1550 TFLOPS on H800.
- `[DS-Profile]` *DeepSeek profile-data*. https://github.com/deepseek-ai/profile-data [REFERENCE] [2026-NEW]
- `[MegaScale-MoE]` *MegaScale-MoE: Large-Scale Communication-Efficient Training of MoE in Production*. ByteDance. 2025-05. arXiv:2505.11432. https://arxiv.org/abs/2505.11432 [SOTA] [2026-NEW]
  - 1.88× MFU vs Megatron-LM on 352B MoE / 1,440 GPUs.
- `[MegaScale-Infer]` *MegaScale-Infer: Serving MoE at Scale with Disaggregated Expert Parallelism*. ByteDance + PKU. 2025-04. SIGCOMM 2025, arXiv:2504.02263. https://arxiv.org/abs/2504.02263 [SOTA] [2026-NEW]
  - Disaggregates attention from FFN; 2.56× / 1.28× per-GPU vs vLLM/TRT-LLM.
- `[FlashDMoE]` *FlashDMoE: Fast Distributed MoE in a Single Kernel*. 2025-06. arXiv:2506.04667. https://arxiv.org/abs/2506.04667 [SOTA] [2026-NEW]
- `[FlashCommV2]` *FlashCommunication V2: Bit Splitting and Spike Reserving for Any Bit Communication*. 2025-08. arXiv:2508.03760. https://arxiv.org/abs/2508.03760 [EMERGING] [2026-NEW]
- `[FineMoE]` *FineMoE: Taming Latency-Memory Trade-Off via Fine-Grained Expert Offloading*. 2025-02. arXiv:2502.05370. https://arxiv.org/abs/2502.05370 [SOTA] [2026-NEW]
  - Verified system name is FineMoE (was incorrectly cited as fMoE).
- `[HybriMoE]` *HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management*. 2025-04. arXiv:2504.05897. https://arxiv.org/abs/2504.05897 [UNVERIFIED] [SOTA] [2026-NEW]
- `[PreScope]` *PreScope: Unleashing the Power of Prefetching for Resource-Constrained MoE Inference*. 2025-09. arXiv:2509.23638. https://arxiv.org/abs/2509.23638 [SOTA] [2026-NEW]
- `[ExpertFlow-2025]` *ExpertFlow: Adaptive Expert Scheduling and Memory Coordination*. 2025-10. arXiv:2510.26730. https://arxiv.org/abs/2510.26730 [SOTA] [2026-NEW]
- `[Janus]` *Janus: Disaggregating Attention and Experts for Scalable MoE Inference*. 2025-12. arXiv:2512.13525. https://arxiv.org/abs/2512.13525 [SOTA] [2026-NEW]
  - Up to 4.7× per-GPU vs SOTA; 25% GPU-cost reduction.
- `[NCCL-EP]` *NCCL EP: Towards a Unified Expert Parallel Communication API*. 2026-03. arXiv:2603.13606. https://arxiv.org/html/2603.13606 [UNVERIFIED] [SOTA] [2026-NEW]
- `[vLLM-WideEP]` *vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP*. vLLM. 2025-12. https://blog.vllm.ai/2025/12/17/large-scale-serving.html [REFERENCE] [2026-NEW]
- `[TRT-LLM-WideEP-Pt2]` *Scaling Expert Parallelism in TensorRT-LLM (Part 2)*. NVIDIA. 2025. https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog8_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.html [REFERENCE] [2026-NEW]
- `[TRT-LLM-NVL72]` *Scaling Large MoE Models with Wide Expert Parallelism on NVL72*. NVIDIA. 2025. https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/ [REFERENCE] [2026-NEW]
- `[NVIDIA-Dynamo-MoE]` *How NVIDIA GB200 NVL72 and Dynamo Boost MoE Inference*. NVIDIA. https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models/ [REFERENCE] [2026-NEW]
- `[MoE-Inf-Bench]` *MoE-Inference-Bench: Performance Evaluation of MoE LLMs and VLMs*. SC 2025 Workshops. arXiv:2508.17467. https://arxiv.org/abs/2508.17467 [REFERENCE] [2026-NEW]
- `[MoE-Survey-2024]` *A Survey on Inference Optimization Techniques for MoE Models*. 2024-12. ACM CSur 2025, arXiv:2412.14219. https://arxiv.org/abs/2412.14219 [REFERENCE]
- `[Elastic-EP-SGLang]` *Elastic EP in SGLang: Achieving Partial Failure Tolerance for DeepSeek MoE Deployments*. LMSYS. 2026-03. https://lmsys.org/blog/2026-03-25-eep-partial-failure-tolerance/ [REFERENCE] [2026-NEW]


### Long-context inference

#### Lineage — position encodings

- `[RoFormer]` *RoFormer: Enhanced Transformer with Rotary Position Embedding*. Su, Lu, Pan, Murtadha, Wen, Liu (Zhuiyi). 2021-04. Neurocomputing 2024, arXiv:2104.09864. https://arxiv.org/abs/2104.09864 [CANONICAL]
  - Origin of RoPE.
- `[PI]` *Extending Context Window of LLMs via Positional Interpolation*. Chen, Wong, Chen, Tian (Meta). 2023-06. arXiv:2306.15595. https://arxiv.org/abs/2306.15595 [CANONICAL]
  - First practical RoPE extension via linear down-scaling.
- `[YaRN]` *YaRN: Efficient Context Window Extension of Large Language Models*. Peng, Quesnelle, Fan, Shippole. 2023-08. ICLR 2024, arXiv:2309.00071. https://arxiv.org/abs/2309.00071 [CANONICAL]
  - SOTA RoPE extension recipe; NTK-by-parts + temperature scaling.
- `[NTK-aware]` *NTK-aware RoPE scaling*. EleutherAI / community blog. 2023. https://blog.eleuther.ai/yarn/ [REFERENCE]
- `[LongRoPE]` *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens*. Ding et al. (Microsoft). 2024-02. ICML 2024, arXiv:2402.13753. https://arxiv.org/abs/2402.13753 [CANONICAL]
- `[LongRoPE2]` *LongRoPE2: Near-Lossless LLM Context Window Scaling*. Microsoft. 2025-02. arXiv:2502.20082. https://arxiv.org/abs/2502.20082 [SOTA] [2026-NEW]
  - 128K with 10B tokens (vs Meta's 800B); high RoPE dims under-trained hypothesis.

#### Lineage — sequence/context parallelism

- `[Megatron-SP]` *Reducing Activation Recomputation in Large Transformer Models*. Korthikanti et al. (NVIDIA). 2022-05. arXiv:2205.05198. https://arxiv.org/abs/2205.05198 [CANONICAL]
- `[Ring-Attn]` *Ring Attention with Blockwise Transformers for Near-Infinite Context*. Liu, Zaharia, Abbeel. 2023-10. ICLR 2024, arXiv:2310.01889. https://arxiv.org/abs/2310.01889 [CANONICAL]
  - KV blocks ring with overlapped P2P; canonical long-sequence attention algorithm.
- `[Striped-Attn]` *Striped Attention: Faster Ring Attention for Causal Transformers*. Brandon et al. (MIT). 2023-11. arXiv:2311.09431. https://arxiv.org/abs/2311.09431 [CANONICAL]
  - Permutes token-to-device mapping to balance causal-mask workload.
- `[DS-Ulysses]` *DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models*. Microsoft. 2023-09. arXiv:2309.14509. https://arxiv.org/abs/2309.14509 [CANONICAL]
  - All-to-all on heads; constant comm volume with seq-length scaling.
- `[USP]` *USP: A Unified Sequence Parallelism Approach for Long Context Generative AI*. Fang, Zhao. 2024-05. arXiv:2405.07719. https://arxiv.org/abs/2405.07719 [CANONICAL]
  - 2D hybrid (Ulysses outer, Ring inner).

#### Lineage — sparse / sliding-window attention

- `[Longformer]` *Longformer: The Long-Document Transformer*. Beltagy, Peters, Cohan. 2020-04. arXiv:2004.05150. https://arxiv.org/abs/2004.05150 [CANONICAL]
- `[BigBird]` *Big Bird: Transformers for Longer Sequences*. Zaheer et al. (Google). 2020-07. arXiv:2007.14062. https://arxiv.org/abs/2007.14062 [CANONICAL]
- `[LongLoRA]` *LongLoRA: Efficient Fine-tuning of Long-Context LLMs*. Chen et al. (CUHK/MIT). 2023-09. ICLR 2024, arXiv:2309.12307. https://arxiv.org/abs/2309.12307 [CANONICAL]

#### Past 12 months SOTA — long-context

- `[Qwen2.5-1M]` *Qwen2.5-1M Technical Report*. Qwen Team. 2025-01. arXiv:2501.15383. https://arxiv.org/abs/2501.15383 [SOTA] [2026-NEW]
  - 7B/14B-Instruct-1M; sparse attention + chunked prefill; 3–7× faster prefill.
- `[NSA]` *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*. DeepSeek-AI. 2025-02. ACL 2025, arXiv:2502.11089. https://arxiv.org/abs/2502.11089 [SOTA] [2026-NEW]
  - Three branches (compressed/selected/sliding); up to 11.6× decode speedup at 64K.
- `[MMInference]` *MMInference: Modality-Aware Permutation Sparse Attention*. Microsoft. 2025-04. ICML 2025. [UNVERIFIED] [SOTA] [2026-NEW]
- `[MInference]` *Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention*. Microsoft. 2024-07. NeurIPS 2024 Spotlight, arXiv:2407.02490. https://arxiv.org/abs/2407.02490 [CANONICAL]
  - Three head-pattern templates; 10× prefill speedup at 1M on A100.
- `[RWKV-7]` *RWKV-7 "Goose"*. Peng et al. 2025-03. COLM 2025, arXiv:2503.14456. https://arxiv.org/abs/2503.14456 [SOTA] [2026-NEW]
  - Generalized delta rule with vector-valued gating.
- `[Mamba]` *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. Gu, Dao. 2023-12. arXiv:2312.00752. https://arxiv.org/abs/2312.00752 [CANONICAL]
- `[Mamba-2]` *Transformers are SSMs: Generalized Models and Efficient Algorithms*. Dao, Gu. 2024-05. ICML 2024, arXiv:2405.21060. https://arxiv.org/abs/2405.21060 [CANONICAL]
- `[Jamba]` *Jamba: A Hybrid Transformer-Mamba Language Model*. AI21. 2024-03. arXiv:2403.19887. https://arxiv.org/abs/2403.19887 [CANONICAL]
- `[Jamba-1.5]` *Jamba-1.5*. AI21. 2024-08. arXiv:2408.12570. https://arxiv.org/abs/2408.12570 [CANONICAL]
- `[Hymba]` *Hymba: A Hybrid-Head Architecture for Small Language Models*. Dong et al. (NVIDIA). 2024-11. ICLR 2025, arXiv:2411.13676. https://arxiv.org/abs/2411.13676 [CANONICAL]
- `[Zamba]` *Zamba: A Compact 7B SSM Hybrid Model*. Glorioso, Anthony. 2024-05. arXiv:2405.16712. https://arxiv.org/abs/2405.16712 [CANONICAL]
- `[Zamba2]` *Zamba2-7B*. Zyphra. 2024-08. https://www.zyphra.com/post/zamba2-7b [REFERENCE]
- `[Falcon-H1]` *Falcon-H1: A Family of Hybrid-Head Language Models*. TII. 2025-07. arXiv:2507.22448. https://arxiv.org/abs/2507.22448 [SOTA] [2026-NEW]
  - 0.5B–34B; 256K context; parallel attn+SSM concatenated.
- `[Granite-4]` *IBM Granite 4.0: Hyper-Efficient Hybrid Models*. IBM. 2025-10. https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models [REFERENCE] [2026-NEW]
  - 9:1 Mamba-2 : attention; >70% RAM reduction at long ctx.
- `[Llama 3.1]` *Llama 3.1 (multi-stage RoPE-base)*. Meta. 2024-07. https://huggingface.co/blog/llama31 [REFERENCE]
- `[Llama 3.2]` *Llama 3.2*. Meta. 2024-09. https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/ [REFERENCE]
- `[Gemini-1.5]` *Unlocking Multimodal Understanding Across Millions of Tokens of Context*. Google DeepMind. 2024-03. arXiv:2403.05530. https://arxiv.org/abs/2403.05530 [REFERENCE]
- `[Claude-1M]` *Claude Opus 4.6 / Sonnet 4.6 (1M GA)*. Anthropic. 2025. https://www.anthropic.com/news/claude-opus-4-6 [REFERENCE] [2026-NEW]
- `[NIAH]` *Needle-in-a-Haystack benchmark*. Kamradt. 2023. https://github.com/gkamradt/LLMTest_NeedleInAHaystack [REFERENCE]
- `[RULER]` *RULER: What's the Real Context Size of Your Long-Context Language Models?*. NVIDIA. 2024-04. COLM 2024, arXiv:2404.06654. https://arxiv.org/abs/2404.06654 [CANONICAL]
- `[InfiniteBench]` *InfiniteBench: 200K+ token tasks*. Zhang et al. 2024. [UNVERIFIED] [REFERENCE]
- `[LongBench-v2]` *LongBench v2*. 2024. https://aclanthology.org/2025.findings-acl.903.pdf [REFERENCE] [2026-NEW]
- `[MRCR]` *Multi-Round Co-reference Reasoning*. (Anthropic/OpenAI variants). [UNVERIFIED] [REFERENCE]
- `[Twilight]` *Twilight: Hierarchical Top-p Adaptive Sparsity*. NeurIPS 2025 Spotlight. https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/twilight.neurips25.pdf [SOTA] [2026-NEW]
- `[Mamba-3]` *Mamba-3: A Universal Architecture for State-Space Models*. (lineage promoted 2026-05-05 from emerging to canonical). 2026-03. ICLR 2026 oral, arXiv:2603.15569. https://arxiv.org/abs/2603.15569 [CANONICAL] [2026-NEW]
  - The latest canonical SSM reference; relevant to long-context (`20/04`) and attention-variants (`30/03`) chapters.
- `[NOSA]` *NOSA: Native and Offloadable Sparse Attention*. 2025-10. arXiv:2510.13602. https://arxiv.org/pdf/2510.13602 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[RocketKV]` *RocketKV: Two-stage KV compression at long ctx*. ICML 2025. arXiv:2502.14051. https://arxiv.org/abs/2502.14051 [SOTA] [2026-NEW]
- `[Unsloth-Flex]` *Long Context gpt-oss Training*. Unsloth. 2025. https://docs.unsloth.ai/models/gpt-oss-how-to-run-and-fine-tune/long-context-gpt-oss-training [REFERENCE] [2026-NEW]
- `[SCBench-Long]` See `[SCBench]` above.
- `[HiSparse]` *HiSparse: SGLang CPU-offloaded sparse-attn KV*. LMSYS. 2026. [UNVERIFIED] [SOTA] [2026-NEW]
- `[SGLang-Pipeline-PP]` *Chunked Pipeline Parallelism for Ultra-Long Context*. SGLang. 2026-01. [REFERENCE] [2026-NEW]

### Heterogeneous inference

#### Lineage

- `[ZeRO-Inference]` *ZeRO-Inference (DeepSpeed blog)*. Aminabadi et al. (Microsoft). 2022-09. https://www.deepspeed.ai/2022/09/09/zero-inference.html [UNVERIFIED] [REFERENCE]
- `[Petals]` *Petals: Collaborative Inference and Fine-tuning of Large Models*. Borzunov et al. 2022-09. ACL 2023 demo, arXiv:2209.01188. https://arxiv.org/abs/2209.01188 [CANONICAL]
- `[PowerInfer]` *PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU*. Song, Xie, Zhang et al. (SJTU-IPADS). 2023-12. SOSP 2024, arXiv:2312.12456. https://arxiv.org/abs/2312.12456 [CANONICAL]
  - Hot/cold neuron split via power-law activation; 11.69× over llama.cpp on RTX 4090.
- `[PowerInfer-2]` *PowerInfer-2: Fast LLM Inference on a Smartphone*. Xue, Song, Chen et al. (SJTU-IPADS). 2024-06. arXiv:2406.06282. https://arxiv.org/abs/2406.06282 [CANONICAL]
- `[KTransformers]` *KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models*. Chen et al. (Tsinghua/kvcache-ai). 2025-10. SOSP 2025. https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/ [SOTA] [2026-NEW]
  - DeepSeek-R1 671B on single 4090D + DRAM at ~14 tok/s decode.
- `[HexGen]` *HexGen: Generative Inference of LLM over Heterogeneous Environment*. Jiang, Yan, Yuan. 2023-11. ICML 2024, arXiv:2311.11514. https://arxiv.org/abs/2311.11514 [CANONICAL]
- `[LLM-PQ]` *LLM-PQ: Phase-Aware Partition and Adaptive Quantization*. Zhao et al. 2024-03. PPoPP 2024, arXiv:2403.01136. https://arxiv.org/abs/2403.01136 [CANONICAL]
- `[Mélange]` *Mélange: Cost Efficient Large Language Model Serving by Exploiting GPU Heterogeneity*. Griggs, Liu, Ren, Patel, Stoica (UCB/Microsoft). 2024-04. arXiv:2404.14527. https://arxiv.org/abs/2404.14527 [CANONICAL]
- `[AlpaServe]` *AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving*. Li, Zheng, Zhuang, Yu, Stoica et al. OSDI 2023. https://www.usenix.org/conference/osdi23/presentation/li-zhuohan [CANONICAL]
- `[SpotServe]` *SpotServe: Serving Generative Large Language Models on Preemptible Instances*. Miao et al. ASPLOS 2024. arXiv:2311.15566. https://arxiv.org/abs/2311.15566 [CANONICAL]
- `[ServerlessLLM]` *ServerlessLLM: Low-Latency Serverless Inference for LLMs*. Fu, Xue, Huang et al. (Edinburgh). OSDI 2024. https://www.usenix.org/conference/osdi24/presentation/fu [CANONICAL]
- `[GreenLLM]` *GreenLLM: Disaggregating LLM Serving on Heterogeneous GPUs for Lower Carbon Emissions*. Shi, Wu, Liu, Ding. 2024-12. arXiv:2412.20322. https://arxiv.org/abs/2412.20322 [SOTA]

#### Past 12 months SOTA

- `[WaferLLM]` *WaferLLM: Large Language Model Inference at Wafer Scale*. He et al. 2025-02. OSDI 2025, arXiv:2502.04563. https://arxiv.org/abs/2502.04563 [SOTA] [2026-NEW]
- `[HexGen-2]` *HexGen-2: Disaggregated Generative Inference of LLMs in Heterogeneous Environment*. Jiang, Yan, Yuan. 2025-02. ICLR 2025, arXiv:2502.07903. https://arxiv.org/abs/2502.07903 [SOTA] [2026-NEW]
  - 2.0× throughput / 1.5× latency over SOTA at same price.
- `[Demystify-CostEff]` *Demystifying Cost-Efficiency in LLM Serving over Heterogeneous GPUs*. Jiang, Fu, Yao, He, Miao, Klimovic, Cui, Yuan, Yoneki. 2025-02. ICML 2025, arXiv:2502.00722. https://arxiv.org/abs/2502.00722 [SOTA] [2026-NEW]
- `[SageServe]` *Serving Models, Fast and Slow: Optimizing Heterogeneous LLM Inferencing Workloads at Scale*. Microsoft + UIUC + IISc. 2025-02. arXiv:2502.14617. https://arxiv.org/abs/2502.14617 [SOTA] [2026-NEW]
  - 25% GPU-hour savings; ~$2M/month at provider scale.
- `[Prism]` *Prism: Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving*. 2025-05. arXiv:2505.04021. https://arxiv.org/abs/2505.04021 [SOTA] [2026-NEW]
- `[HEXGEN-FLOW]` *Optimizing LLM Inference Request Scheduling for Agentic Text-to-SQL*. Relaxed System Lab. 2025-05. arXiv:2505.05286. https://arxiv.org/abs/2505.05286 [SOTA] [2026-NEW]
- `[Hetis]` *Hetis: Serving LLMs in Heterogeneous GPU Clusters with Fine-grained and Dynamic Parallelism*. Mo, Liao, Xu, Zhou, Xu (UMacau). 2025-09. SC 2025, arXiv:2509.08309. https://arxiv.org/abs/2509.08309 [SOTA] [2026-NEW]
- `[Parallax]` *Parallax: Efficient LLM Inference Service over Decentralized Environment*. GradientHQ. 2025-09. arXiv:2509.26182. https://arxiv.org/abs/2509.26182 [SOTA] [2026-NEW]
- `[Cauchy]` *Cauchy: Cost-Efficient LLM Serving through Adaptive Heterogeneous Deployment*. SoCC 2025. https://dl.acm.org/doi/10.1145/3772052.3772264 [SOTA] [2026-NEW]
- `[NeuronMM]` *NeuronMM: High-Performance Matrix Multiplication for LLM Inference on AWS Trainium*. 2025-10. arXiv:2510.25977. https://arxiv.org/abs/2510.25977 [EMERGING] [2026-NEW]
- `[FlexLLM-hetero]` *FlexLLM: Flexible and Cost-Efficient LLM Serving with Heterogeneous GPUs*. Kim et al. MASCOTS 2025. https://discos.sogang.ac.kr/file/2025/intl_conf/MASCOTS_2025_K_Kim.pdf [SOTA] [2026-NEW]
- `[Hetero-MM]` *Cost-Efficient Multimodal LLM Inference via Cross-Tier GPU Heterogeneity*. Yu. 2026-03. arXiv:2603.12707. https://arxiv.org/abs/2603.12707 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[Tessera]` *Tessera: Unlocking Heterogeneous GPUs through Kernel-Granularity Disaggregation*. 2026-04. arXiv:2604.10180. https://arxiv.org/abs/2604.10180 [UNVERIFIED] [SOTA] [2026-NEW]
- `[FastHetero]` *Fast Heterogeneous Serving: Scalable Mixed-Scale LLM Allocation*. 2026-04. arXiv:2604.07472. https://arxiv.org/abs/2604.07472 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[Edge-Cloud-Survey]` *Collaborative Inference and Learning between Edge SLMs and Cloud LLMs: A Survey*. 2025-07. arXiv:2507.16731. https://arxiv.org/abs/2507.16731 [REFERENCE] [2026-NEW]
- `[SambaNova-SN40L]` *SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts*. Prabhakar et al. 2024-05. arXiv:2405.07518. https://arxiv.org/html/2405.07518v1 [REFERENCE]
- `[MLX]` *Apple MLX framework*. Apple. https://machinelearning.apple.com/research/exploring-llms-mlx-m5 [REFERENCE] [2026-NEW]


### LoRA and multi-tenant serving

#### Lineage

- `[LoRA]` *LoRA: Low-Rank Adaptation of Large Language Models*. Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen (Microsoft). 2021-06. ICLR 2022, arXiv:2106.09685. https://arxiv.org/abs/2106.09685 [CANONICAL]
  - Defines W + (α/r)BA adapter; premise of the entire serving stack.
- `[IA3]` *Few-Shot PEFT is Better and Cheaper than In-Context Learning (T-Few/IA³)*. Liu et al. 2022-05. NeurIPS 2022, arXiv:2205.05638. https://arxiv.org/abs/2205.05638 [CANONICAL]
- `[LoraHub]` *LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition*. Huang et al. 2023-07. COLM 2024, arXiv:2307.13269. https://arxiv.org/abs/2307.13269 [CANONICAL]
- `[Punica]` *Punica: Multi-Tenant LoRA Serving*. Chen, Ye, Jiang, Cao, Yang, Krishnamurthy (UW/Duke). 2023-10. MLSys 2024, arXiv:2310.18547. https://arxiv.org/abs/2310.18547 [CANONICAL]
  - BGMV / SGMV kernels; first to batch heterogeneous-adapter requests in one launch.
- `[VeRA]` *VeRA: Vector-based Random Matrix Adaptation*. Kopiczko, Blankevoort, Asano. 2023-10. ICLR 2024, arXiv:2310.11454. https://arxiv.org/abs/2310.11454 [CANONICAL]
- `[S-LoRA]` *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*. Sheng, Cao, Li, Hooper, Ho, Zhuo, Kasikci, Stoica et al. 2023-11. MLSys 2024, arXiv:2311.03285. https://arxiv.org/abs/2311.03285 [CANONICAL]
  - Unified Paging memory pool; up to 4× over vLLM-naive.
- `[LoRAX]` *LoRA eXchange: Serve 100s of fine-tuned LLMs*. Predibase. 2023-11. https://github.com/predibase/lorax [REFERENCE]
  - First production-grade open-source multi-LoRA server.
- `[mLoRA]` *mLoRA: Fine-Tuning LoRA Adapters via Highly-Efficient Pipeline Parallelism*. Tang et al. (TUDB-Labs/Ant). 2023-12. VLDB 2025, arXiv:2312.02515. https://arxiv.org/abs/2312.02515 [CANONICAL]
- `[FLoRA-Batched]` *Batched Low-Rank Adaptation of Foundation Models*. Wen, Chaudhuri. 2023-12. ICLR 2024, arXiv:2312.05677. https://arxiv.org/abs/2312.05677 [CANONICAL]
- `[CaraServe]` *CaraServe: CPU-Assisted and Rank-Aware LoRA Serving*. Li, Lu, Weng, Wu et al. 2024-01. arXiv:2401.11240. https://arxiv.org/abs/2401.11240 [CANONICAL]
- `[DoRA]` *DoRA: Weight-Decomposed Low-Rank Adaptation*. Liu, Wang, Yin, Molchanov, Wang, Cheng, Chen (NVIDIA). 2024-02. ICML 2024 Oral, arXiv:2402.09353. https://arxiv.org/abs/2402.09353 [CANONICAL]
- `[dLoRA]` *dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving*. Wu, Zhu, Zhang, Sun, Liu, Jin (PKU/Shanghai AI Lab). 2024-07. OSDI 2024. https://www.usenix.org/conference/osdi24/presentation/wu-bingyang [CANONICAL]
  - Credit-based merge/unmerge batching; up to 1.8× over S-LoRA.

#### Past 12 months SOTA

- `[ELORA]` *Improving the Serving Performance of Multi-LoRA LLMs via Efficient LoRA and KV Cache Management*. Zhang, Shi et al. (HKUST). 2025-05. HPCA 2026, arXiv:2505.03756. https://arxiv.org/abs/2505.03756 [SOTA] [2026-NEW]
  - Treats LoRA cache and KV cache as one dependency-aware pool; -63% TTFT.
- `[ServerlessLoRA]` *ServerlessLoRA: Minimizing Latency and Cost in Serverless Inference for LoRA-Based LLMs*. 2025-05. arXiv:2505.14468. https://arxiv.org/abs/2505.14468 [SOTA] [2026-NEW]
  - 86% TTFT reduction, 89% cost reduction in serverless.
- `[EdgeLoRA]` *EdgeLoRA: An Efficient Multi-Tenant LLM Serving System on Edge Devices*. 2025-07. MobiSys 2025, arXiv:2507.01438. https://arxiv.org/abs/2507.01438 [SOTA] [2026-NEW]
  - Adaptive adapter selection; 4× throughput vs llama.cpp.
- `[Toppings]` *Toppings: CPU-Assisted, Rank-Aware Adapter Serving for LLM Inference*. Li, Lu, Weng et al. 2025-07. USENIX ATC 2025. https://www.usenix.org/conference/atc25/presentation/li-suyi-toppings [SOTA] [2026-NEW]
- `[Equinox]` *Equinox: Holistic Fair Scheduling in Serving Large Language Models*. Wei et al. 2025-08. arXiv:2508.16646. https://arxiv.org/abs/2508.16646 [SOTA] [2026-NEW]
  - Dual-counter (User + Resource) fairness; 1.3× throughput, 60% lower TTFT vs VTC.
- `[K-Merge]` *K-Merge: Online Continual Merging of Adapters for On-device LLMs*. 2025-10. arXiv:2510.13537. https://arxiv.org/abs/2510.13537 [EMERGING] [2026-NEW]
- `[zFLoRA]` *zFLoRA: Zero-Latency Fused Low-Rank Adapters*. Gowda, Song et al. (Samsung). 2025-10. EMNLP 2025, arXiv:2510.25784. https://arxiv.org/abs/2510.25784 [SOTA] [2026-NEW]
- `[L-MoE]` *L-MoE: Gating Network Composes Adapter Parameters per Token*. 2025-10. arXiv:2510.17898. https://arxiv.org/abs/2510.17898 [EMERGING] [2026-NEW]
- `[Fused-FLoRA]` *FLoRA: Fused forward-backward adapters for parameter efficient fine-tuning*. 2025-11. arXiv:2511.00050. https://arxiv.org/abs/2511.00050 [EMERGING] [2026-NEW]
- `[LoGo]` *LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging*. 2025-11. arXiv:2511.07129. https://arxiv.org/abs/2511.07129 [EMERGING] [2026-NEW]
- `[LoRAServe]` *Serving Heterogeneous LoRA Adapters in Distributed LLM Inference Systems*. Jaiswal, Arun, Parayil, Mallick, Mastorakis et al. 2025-11. arXiv:2511.22880. https://arxiv.org/abs/2511.22880 [SOTA] [2026-NEW]
  - Rank-induced performance skew + GPUDirect RDMA; 2× throughput, 9× lower TTFT.
- `[P-LoRA]` *Predictive-LoRA: A Proactive and Fragmentation-Aware Serverless Inference System*. 2025-12. arXiv:2512.20210. https://arxiv.org/abs/2512.20210 [SOTA] [2026-NEW]
  - LSTM traffic predictor + page-based memory mgr; 1.52× throughput vs S-LoRA.
- `[AdaFuse]` *AdaFuse: Accelerating Dynamic Adapter Inference via Token-Level Pre-Gating*. 2026-03. arXiv:2603.11873. https://arxiv.org/abs/2603.11873 [SOTA] [2026-NEW]
  - Adapter overhead reduced from 250–950% to ~29%.
- `[Turbo-LoRA]` *Turbo-LoRA: Joint LoRA + Draft-Head Training*. Predibase. 2024. https://predibase.com/blog/turbo-lora [REFERENCE]
- `[TT-LoRA-MoE]` *TT-LoRA MoE: Sparse-MoE Router Selects One Adapter Per Input*. SC 2025. https://dl.acm.org/doi/10.1145/3712285.3759888 [REFERENCE] [2026-NEW]
- `[X-LoRA]` *X-LoRA*. https://github.com/EricLBuehler/xlora [REFERENCE]
- `[NIM-LoRA]` *Seamlessly Deploying a Swarm of LoRA Adapters with NVIDIA NIM*. NVIDIA. https://developer.nvidia.com/blog/seamlessly-deploying-a-swarm-of-lora-adapters-with-nvidia-nim/ [REFERENCE]
- `[CF-Workers-LoRA]` *Fine-tuned inference with LoRAs on Cloudflare Workers*. https://blog.cloudflare.com/fine-tuned-inference-with-loras/ [REFERENCE]
- `[SageMaker-LoRA]` *Efficient and cost-effective multi-tenant LoRA serving with Amazon SageMaker*. https://aws.amazon.com/blogs/machine-learning/efficient-and-cost-effective-multi-tenant-lora-serving-with-amazon-sagemaker/ [REFERENCE]
- `[Together-Multi-LoRA]` *Together AI Serverless Multi-LoRA*. https://www.together.ai/blog/serverless-multi-lora-fine-tune-and-deploy-hundreds-of-adapters-for-model-customization-at-scale [REFERENCE]
- `[Friendli-LoRA]` *Multi-LoRA on Friendli Container*. https://friendli.ai/docs/guides/container/serving_multi_lora_models [REFERENCE]
- `[HF-PEFT]` *HuggingFace PEFT*. https://github.com/huggingface/peft [REFERENCE]

### Cluster-level systems (router, gateway, autoscaling, observability)

#### Lineage

- `[GIE-blog]` *Introducing Gateway API Inference Extension*. Kubernetes blog. 2025-06. https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/ [REFERENCE] [2026-NEW]
- `[GIE-repo]` *kubernetes-sigs/gateway-api-inference-extension*. https://github.com/kubernetes-sigs/gateway-api-inference-extension [REFERENCE] [2026-NEW]
  - InferencePool v1; Endpoint Picker Protocol.
- `[GIE-docs]` *Kubernetes GIE docs*. https://gateway-api-inference-extension.sigs.k8s.io/ [REFERENCE] [2026-NEW]
- `[Istio-GIE]` *Istio Gateway API Inference Extension Support*. Istio blog. 2025. https://istio.io/latest/blog/2025/inference-extension-support/ [REFERENCE] [2026-NEW]

#### Past 12 months SOTA — control planes

- `[llm-d-launch]` *llm-d: Kubernetes-native distributed inferencing*. Red Hat. 2025-05. https://developers.redhat.com/articles/2025/05/20/llm-d-kubernetes-native-distributed-inferencing [REFERENCE] [2026-NEW]
- `[llm-d-CNCF]` *Donating llm-d to CNCF*. IBM Research. 2025. https://research.ibm.com/blog/donating-llm-d-to-the-cloud-native-computing-foundation [REFERENCE] [2026-NEW]
- `[llm-d-v0.5]` *llm-d 0.5: Sustaining Performance at Scale*. 2026-02. https://llm-d.ai/blog/llm-d-v0.5-sustaining-performance-at-scale [REFERENCE] [2026-NEW]
  - Scale-to-zero with activator; UCCL transport (2.4× more congestion resilience).
- `[llm-d-EPP-arch]` *llm-d Inference Scheduler Architecture*. https://github.com/llm-d/llm-d-inference-scheduler/blob/main/docs/architecture.md [REFERENCE] [2026-NEW]
- `[llm-d-prefix-precise]` *Precise Prefix Cache Aware Routing*. https://github.com/llm-d/llm-d/blob/main/guides/precise-prefix-cache-aware/README.md [REFERENCE] [2026-NEW]
- `[llm-d-pred-latency]` *Predicted-Latency Based Scheduling for LLMs*. https://llm-d.ai/blog/predicted-latency-based-scheduling-for-llms [REFERENCE] [2026-NEW]
- `[Wide-EP-llmd]` *Scaling DeepSeek-style MoEs with vLLM and llm-d using Wide EP*. Red Hat. 2025-09. https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep [REFERENCE] [2026-NEW]
- `[AIBrix-launch]` *Introducing AIBrix*. vLLM blog. 2025-02. https://blog.vllm.ai/2025/02/21/aibrix-release.html [REFERENCE] [2026-NEW]
- `[AIBrix-v0.3]` *AIBrix v0.3.0: KVCache Offloading, Prefix Cache, Fairness Routing*. 2025-05. https://aibrix.github.io/posts/2025-05-21-v0.3.0-release/ [REFERENCE] [2026-NEW]
- `[AIBrix-v0.4]` *AIBrix v0.4.0: P/D Disaggregation, EP, KVCache v1, KV Event Sync*. 2025-08. https://aibrix.github.io/posts/2025-08-04-v0.4.0-release/ [REFERENCE] [2026-NEW]
- `[AIBrix-arch]` *AIBrix Architecture*. https://aibrix.readthedocs.io/latest/designs/architecture.html [REFERENCE] [2026-NEW]
- `[KServe-OIP]` *Open Inference Protocol (V2)*. KServe. https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol [REFERENCE]
- `[KServe-CNCF]` *KServe joins CNCF as Incubating project*. 2025-11. https://kserve.github.io/website/ [REFERENCE] [2026-NEW]
- `[KServe-llmd]` *Combining KServe and llm-d for optimized generative AI inference*. Red Hat. 2026-04. https://developers.redhat.com/articles/2026/04/21/kserve-llm-d-optimized-gen-ai-inference [REFERENCE] [2026-NEW]
- `[Aegaeon]` *Aegaeon: Effective GPU Pooling for Concurrent LLM Serving on the Market*. Xiang et al. (Alibaba). SOSP 2025. https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf [SOTA] [2026-NEW]
  - Token-granularity model auto-scaling; 1192 → 213 GPUs in real model marketplace.
- `[AdaServe]` *AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding*. CMU/Princeton/Purdue. EuroSys 2026, arXiv:2501.12162. https://arxiv.org/abs/2501.12162 [SOTA] [2026-NEW]
  - 4.3× SLO-violation reduction.
- `[PolyServe]` *PolyServe: Efficient Multi-SLO Serving at Scale*. arXiv:2507.17769. https://arxiv.org/html/2507.17769 [SOTA] [2026-NEW]
- `[BrownoutServe]` *BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads*. arXiv:2507.17133. https://arxiv.org/pdf/2507.17133 [EMERGING] [2026-NEW]
- `[DualMap]` *DualMap: Enabling Both Cache Affinity and Load Balancing*. arXiv:2602.06502. https://arxiv.org/abs/2602.06502 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[RouterWise]` *RouterWise: Joint Resource Allocation and Routing for Latency-Aware Multi-Model LLM Serving*. arXiv:2604.10907. https://arxiv.org/abs/2604.10907 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[KV-event-sync-AIBrix]` AIBrix KV Event Subscription System (v0.4.0). [REFERENCE] [2026-NEW]
- `[UCCL-llm-d]` UCCL host-resident transport in llm-d 0.5. [REFERENCE] [2026-NEW]

#### Engine-level routers / gateways

- `[vLLM-router]` *vLLM Router: A High-Performance and Prefill/Decode Aware Load Balancer*. vLLM blog. 2025-12. https://blog.vllm.ai/2025/12/13/vllm-router-release.html [REFERENCE] [2026-NEW]
- `[vLLM-prod-stack]` *vLLM Production Stack*. https://github.com/vllm-project/production-stack [REFERENCE] [2026-NEW]
- `[sgl-router]` *SGLang sgl-router*. https://github.com/sgl-project/sglang [REFERENCE]
- `[EnvoyAI-launch]` *Tetrate and Bloomberg Release Open Source Envoy AI Gateway*. Tetrate. 2025-02. https://tetrate.io/press/tetrate-and-bloomberg-release-open-source-envoy-ai-gateway-built-on-cncfs-envoy-gateway-project [REFERENCE] [2026-NEW]
- `[EnvoyAI-site]` *Envoy AI Gateway*. https://aigateway.envoyproxy.io/ [REFERENCE] [2026-NEW]
- `[CNCF-genai-platform]` *Building a scalable, flexible, cloud-native GenAI platform with open source*. CNCF. 2025-08. https://www.cncf.io/blog/2025/08/28/building-a-scalable-flexible-cloud-native-genai-platform-with-open-source-solutions/ [REFERENCE] [2026-NEW]
- `[LiteLLM]` *BerriAI/litellm*. https://github.com/BerriAI/litellm [REFERENCE]
- `[Portkey]` *Portkey AI Gateway*. https://portkey.ai/buyers-guide/ai-gateway-solutions [REFERENCE]
- `[Bifrost]` *Maxim AI Bifrost*. https://github.com/maximhq/bifrost [UNVERIFIED] [REFERENCE]
- `[Kong-AIGW]` *Kong AI Gateway*. https://developer.konghq.com/ai-gateway/ [REFERENCE]

#### Observability

- `[OTel-GenAI]` *Semantic conventions for generative AI systems*. OpenTelemetry. https://opentelemetry.io/docs/specs/semconv/gen-ai/ [REFERENCE] [2026-NEW]
- `[OTel-GenAI-spans]` *Semantic conventions for generative client AI spans*. https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/ [REFERENCE] [2026-NEW]
- `[OTel-agent]` *GenAI agent and framework spans*. https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/ [REFERENCE] [2026-NEW]
- `[vLLM-metrics]` *vLLM Metrics*. https://docs.vllm.ai/en/stable/design/metrics/ [REFERENCE]
- `[Datadog-OTel-GenAI]` *Datadog LLM Observability OTel GenAI Semantic Conventions*. https://www.datadoghq.com/blog/llm-otel-semantic-convention/ [REFERENCE]
- `[llm-d-observability]` *Monitoring and Observability with llm-d*. https://llm-d.ai/docs/usage/monitoring [REFERENCE] [2026-NEW]

#### Managed runtimes

- `[RayServe-LLM]` *Ray Serve LLM architecture*. https://docs.ray.io/en/latest/serve/llm/architecture/overview.html [REFERENCE]
- `[RayServe-async]` *Ray Serve: Async Inference, Custom Routing, Custom Autoscaling*. https://www.anyscale.com/blog/ray-serve-autoscaling-async-inference-custom-routing [REFERENCE]
- `[RayServe-prefix]` *Prefix-aware routing in Ray Serve*. https://docs.ray.io/en/latest/serve/llm/user-guides/prefix-aware-routing.html [REFERENCE]
- `[Anyscale-LLM]` *Anyscale LLM Suite*. https://www.anyscale.com/product/platform/llm-suite [REFERENCE]
- `[BentoML-OpenLLM]` *bentoml/OpenLLM*. https://github.com/bentoml/OpenLLM [REFERENCE]
- `[Modal]` *Modal*. https://modal.com/ [REFERENCE]
- `[Modal-Almanac]` *LLM Engineer's Almanac*. https://modal.com/llm-almanac/summary [REFERENCE]
- `[Bedrock-CRIS]` *Cross-Region Inference for Anthropic Claude on Amazon Bedrock*. AWS. https://aws.amazon.com/blogs/machine-learning/unlock-global-ai-inference-scalability-using-new-global-cross-region-inference-on-amazon-bedrock-with-anthropics-claude-sonnet-4-5/ [REFERENCE] [2026-NEW]
- `[Bedrock-Opus47]` *Introducing Anthropic's Claude Opus 4.7 model in Amazon Bedrock*. https://aws.amazon.com/blogs/aws/introducing-anthropics-claude-opus-4-7-model-in-amazon-bedrock/ [REFERENCE] [2026-NEW]
- `[Helicone-Survey]` *Top 11 LLM API Providers in 2025*. Helicone. https://www.helicone.ai/blog/llm-api-providers [UNVERIFIED] [REFERENCE]
- `[Anyscale-vLLM-WideEP]` *Ray Serve LLM, Anyscale APIs, Wide-EP, Disaggregated Serving*. https://www.anyscale.com/blog/ray-serve-llm-anyscale-apis-wide-ep-disaggregated-serving-vllm [REFERENCE] [2026-NEW]


### Hardware (NVIDIA, AMD, ASIC, networking)

#### NVIDIA roadmap

- `[NV-H100-WP]` *NVIDIA H100 Tensor Core GPU Architecture (whitepaper)*. NVIDIA. 2022-03. https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c [REFERENCE]
- `[NV-Hopper-Blog]` *NVIDIA Hopper Architecture In-Depth*. 2022-03. https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ [REFERENCE]
- `[NV-H200-Page]` *NVIDIA H200 Tensor Core GPU*. 2024. https://www.nvidia.com/en-us/data-center/h200/ [REFERENCE]
- `[NV-Blackwell-Arch]` *NVIDIA Blackwell Architecture*. NVIDIA. 2024–25. https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/ [REFERENCE]
- `[NV-Blackwell-PR]` *NVIDIA Blackwell Platform Arrives to Power a New Era*. 2024-03. https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing [REFERENCE]
- `[NV-GB200-Page]` *NVIDIA GB200 NVL72*. https://www.nvidia.com/en-us/data-center/gb200-nvl72/ [REFERENCE]
- `[NV-GB300-Page]` *NVIDIA GB300 NVL72*. https://www.nvidia.com/en-us/data-center/gb300-nvl72/ [REFERENCE] [2026-NEW]
- `[NV-Blackwell-Ultra]` *Inside NVIDIA Blackwell Ultra*. 2025-03. https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/ [REFERENCE] [2026-NEW]
- `[Introl-B300]` *NVIDIA Blackwell Ultra and B300 Infrastructure Requirements*. 2025. https://introl.com/blog/nvidia-blackwell-ultra-b300-infrastructure-requirements-2025 [REFERENCE] [2026-NEW]
- `[CW-GB300]` *CoreWeave Launches GB300 NVL72-Powered Cloud Instances*. 2025-08. https://docs.coreweave.com/docs/changelog/release-notes/gb300-nvl72 [REFERENCE] [2026-NEW]
- `[NV-FY26Q3]` *NVIDIA Q3 FY 2026 Financial Results*. 2025-11. https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-third-quarter-fiscal-2026 [REFERENCE] [2026-NEW]
- `[NV-Vera-Rubin-Blog]` *Inside the NVIDIA Vera Rubin Platform: Six New Chips*. 2026. https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/ [REFERENCE] [2026-NEW]
- `[NV-Rubin-Wiki]` *Rubin (microarchitecture)*. https://en.wikipedia.org/wiki/Rubin_(microarchitecture) [REFERENCE] [2026-NEW]
- `[STH-Rubin]` *NVIDIA Launches Next-Generation Rubin AI Compute Platform at CES 2026*. 2026-01. https://www.servethehome.com/nvidia-launches-next-generation-rubin-ai-compute-platform-at-ces-2026/ [REFERENCE] [2026-NEW]
- `[DCD-RubinUltra]` *Nvidia's Rubin Ultra NVL576 rack expected to be 600kW*. 2025. https://www.datacenterdynamics.com/en/news/nvidias-rubin-ultra-nvl576-rack-expected-to-be-600kw-coming-second-half-of-2027/ [REFERENCE] [2026-NEW]
- `[NV-RubinCPX-Blog]` *NVIDIA Rubin CPX Accelerates 1M+ Token Context Workloads*. 2025-09. https://developer.nvidia.com/blog/nvidia-rubin-cpx-accelerates-inference-performance-and-efficiency-for-1m-token-context-workloads/ [REFERENCE] [2026-NEW]
- `[Reg-RubinCPX]` *Nvidia's context-optimized Rubin CPX GPUs were inevitable*. 2025-09. https://www.theregister.com/2025/09/10/nvidia_rubin_cpx/ [REFERENCE] [2026-NEW]
- `[NV-Groq3LPX]` *Inside NVIDIA Groq 3 LPX*. 2026-03. https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/ [REFERENCE] [2026-NEW]
- `[Decoder-LPX]` *GTC 2026: With Groq 3 LPX, Nvidia adds dedicated inference hardware*. 2026-03. https://the-decoder.com/gtc-2026-with-groq-3-lpx-nvidia-adds-dedicated-inference-hardware-to-its-platform-for-the-first-time/ [REFERENCE] [2026-NEW]
- `[Toms-NV-Groq]` *How Nvidia's $20 billion Groq 3 LPU deal reshapes Vera Rubin*. 2026. https://www.tomshardware.com/tech-industry/semiconductors/nvidias-20-billion-groq-deal-produces-its-first-chip [REFERENCE] [2026-NEW]
- `[NV-DGX-Spark]` *NVIDIA DGX Spark*. https://www.nvidia.com/en-us/products/workstations/dgx-spark/ [REFERENCE] [2026-NEW]
- `[NV-Jetson-Thor]` *NVIDIA Jetson AGX Thor benchmarks*. 2025-10. https://jetsonhacks.com/2025/10/31/nvidia-jetson-agx-thor-vs-dgx-spark-benchmarks/ [REFERENCE] [2026-NEW]
- `[NP-VeraRubin]` *Nvidia's Vera-Rubin Platform Obsoletes Current AI Iron*. 2026-01. https://www.nextplatform.com/ai/2026/01/06/nvidias-vera-rubin-platform-obsoletes-current-ai-iron-six-months-ahead-of-launch/4092179 [REFERENCE] [2026-NEW]
- `[NP-NV-Groq]` *Nvidia Finally Admits Why It Shelled Out $20B For Groq*. 2026-03. https://www.nextplatform.com/ai/2026/03/17/nvidia-finally-admits-why-it-shelled-out-20-billion-for-groq/5209495 [REFERENCE] [2026-NEW]
- `[Hopper-MS]` *Comparing Blackwell vs Hopper*. https://www.exxactcorp.com/blog/hpc/comparing-nvidia-tensor-core-gpus [REFERENCE]
- `[ASPLOS-Blackwell]` *Microbenchmarking NVIDIA's Blackwell Architecture*. 2025. arXiv:2512.02189. https://arxiv.org/html/2512.02189v3 [REFERENCE] [2026-NEW]
- `[NV-MLPerf5.0]` *NVIDIA Blackwell in MLPerf Inference v5.0*. 2025. https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/ [REFERENCE]
- `[NV-MLPerf5.1]` *NVIDIA Blackwell Ultra Sets New Inference Records in MLPerf Debut*. 2025. https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/ [REFERENCE] [2026-NEW]
- `[HPCwire-MLPerf5.1]` *MLPerf Inference v5.1 Results Land*. 2025-09. https://www.hpcwire.com/2025/09/10/mlperf-inference-v5-1-results-land-with-new-benchmarks-and-record-participation/ [REFERENCE] [2026-NEW]

#### AMD

- `[AMD-CDNA4-WP]` *Introducing AMD CDNA 4 Architecture (whitepaper)*. AMD. 2025. https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf [REFERENCE] [2026-NEW]
- `[AMD-MI350-Blog]` *AMD Instinct MI350 Series and Beyond*. 2025. https://www.amd.com/en/blogs/2025/amd-instinct-mi350-series-and-beyond-accelerating-the-future-of-ai-and-hpc.html [REFERENCE] [2026-NEW]
- `[AMD-MI350-Game]` *MI350 Series GPUs: A Game Changer*. 2025. https://www.amd.com/en/blogs/2025/amd-instinct-mi350-series-game-changer.html [REFERENCE] [2026-NEW]
- `[STH-CDNA4]` *AMD Dives Deep on CDNA 4 at Hot Chips 2025*. 2025-08. https://www.servethehome.com/amd-dives-deep-on-cdna-4-architecture-and-mi350-accelerator-at-hot-chips-2025/ [REFERENCE] [2026-NEW]
- `[Toms-MI400]` *AMD touts Instinct MI430X / MI440X / MI455X and Helios*. 2026-01. https://www.tomshardware.com/tech-industry/artificial-intelligence/amd-touts-instinct-mi430x-mi440x-and-mi455x-ai-accelerators-and-helios-rack-scale-ai-architecture-at-ces-full-mi400-series-family-fulfills-a-broad-range-of-infrastructure-and-customer-requirements [REFERENCE] [2026-NEW]
- `[SemiA-MI400]` *AMD Advancing AI: MI350X and MI400 UALoE72, MI500 UAL256*. 2025-06. https://semianalysis.com/2025/06/13/amd-advancing-ai-mi350x-and-mi400-ualoe72-mi500-ual256/ [REFERENCE] [2026-NEW]
- `[AMD-MLPerf6]` *AMD Delivers Breakthrough MLPerf Inference 6.0 Results*. 2026. https://www.amd.com/en/blogs/2026/amd-delivers-breakthrough-mlperf-inference-6-0-results.html [REFERENCE] [2026-NEW]
- `[ROCm-vLLM-First]` *ROCm Becomes a First-Class Platform in the vLLM Ecosystem*. 2025. https://rocm.blogs.amd.com/software-tools-optimization/vllm-omni/README.html [REFERENCE] [2026-NEW]
- `[ROCm-vLLM-Doc]` *vLLM inference - ROCm Documentation*. https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/vllm.html [REFERENCE] [2026-NEW]
- `[ROCm-vLLM-Opt]` *vLLM V1 performance optimization on ROCm*. https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html [REFERENCE] [2026-NEW]

#### ASIC ecosystem

- `[GCloud-TPUv7]` *TPU7x (Ironwood)*. 2025. https://docs.cloud.google.com/tpu/docs/tpu7x [REFERENCE] [2026-NEW]
- `[Google-Iron-Blog]` *Ironwood: The first Google TPU for the age of inference*. 2025. https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/ [REFERENCE] [2026-NEW]
- `[Google-Iron-Stack]` *Inside the Ironwood TPU codesigned AI stack*. 2025-11. https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack [REFERENCE] [2026-NEW]
- `[GCloud-Trillium]` *Trillium TPU v6e in preview*. 2024. https://cloud.google.com/blog/products/compute/trillium-sixth-generation-tpu-is-in-preview [REFERENCE]
- `[GCloud-v5e]` *TPU v5e*. https://docs.cloud.google.com/tpu/docs/v5e [REFERENCE]
- `[GCloud-v5p]` *TPU v5p*. https://docs.cloud.google.com/tpu/docs/v5p [REFERENCE]
- `[Jouppi-TPUv4]` *TPU v4: An Optically Reconfigurable Supercomputer*. 2023-04. arXiv:2304.01433. https://arxiv.org/abs/2304.01433 [CANONICAL]
- `[STH-Iron]` *This is the Google TPU v7 Ironwood Chip*. 2025. https://www.servethehome.com/this-is-the-google-tpu-v7-ironwood-chip/ [REFERENCE] [2026-NEW]
- `[FM-TPUOCS]` *Unveiling Google's TPU Architecture: OCS*. 2025. https://www.fibermall.com/blog/unveiling-google-tpu-architecture.htm [REFERENCE] [2026-NEW]
- `[Anthropic-TPU]` *Anthropic to Expand Use of Google Cloud TPUs*. 2025-10. https://www.anthropic.com/news/expanding-our-use-of-google-cloud-tpus-and-services [REFERENCE] [2026-NEW]
- `[Anthropic-AWS]` *Anthropic and Amazon expand collaboration for up to 5 GW*. 2025. https://www.anthropic.com/news/anthropic-amazon-compute [REFERENCE] [2026-NEW]
- `[AWS-Trn2-Page]` *Amazon EC2 Trn2 Instances*. https://aws.amazon.com/ec2/instance-types/trn2/ [REFERENCE]
- `[AWS-Trn2-Blog]` *Trn2 Instances and Trn2 UltraServers GA*. 2024-12. https://aws.amazon.com/blogs/aws/amazon-ec2-trn2-instances-and-trn2-ultraservers-for-aiml-training-and-inference-is-now-available/ [REFERENCE]
- `[SemiA-Trn2]` *Trainium2 Architecture & Networking*. 2025. https://newsletter.semianalysis.com/p/amazons-ai-self-sufficiency-trainium2-architecture-networking [REFERENCE] [2026-NEW]
- `[Introl-Trn3]` *Amazon's Trainium3 Throws Down the Gauntlet*. 2025-12. https://introl.com/blog/amazon-trainium3-aws-nvidia-ai-chip-competition-2025 [REFERENCE] [2026-NEW]
- `[Groq-LPU-Page]` *Groq LPU Architecture*. https://groq.com/lpu-architecture [REFERENCE]
- `[Groq-LPU-Inside]` *Inside the LPU: Deconstructing Groq's Speed*. 2024. https://groq.com/blog/inside-the-lpu-deconstructing-groq-speed [REFERENCE]
- `[SemiA-Groq]` *Groq Inference Tokenomics*. 2024. https://newsletter.semianalysis.com/p/groq-inference-tokenomics-speed-but [REFERENCE]
- `[Cerebras-CS3]` *Cerebras CS-3*. 2024. https://www.cerebras.ai/blog/cerebras-cs3 [REFERENCE]
- `[Cerebras-HC24]` *Cerebras Wafer-Scale AI (Hot Chips 2024)*. https://hc2024.hotchips.org/assets/program/conference/day2/72_HC2024.Cerebras.Sean.v03.final.pdf [REFERENCE]
- `[Cerebras-Compare]` *Cerebras WSE Comparison with NVIDIA GPU-based Systems*. 2025. arXiv:2503.11698. https://arxiv.org/html/2503.11698v1 [REFERENCE] [2026-NEW]
- `[NBF-CS3]` *Cerebras CS-3 wafer-scale 25 kW*. 2025-11. https://armdevices.net/2025/11/27/cerebras-cs-3-wafer-scale-million-core-ai-chip-25kw-wse-3-125-pflops-inference-engine-tsunami-hpc/ [REFERENCE] [2026-NEW]
- `[Cerebras-press]` *Cerebras inference release*. https://www.cerebras.ai/press-release/cerebras-launches-the-worlds-fastest-ai-inference [UNVERIFIED] [REFERENCE]
- `[SambaN-RDU]` *SN40L RDU product page*. https://sambanova.ai/products/rdu-ai-chips [REFERENCE]
- `[SambaN-Rack]` *SambaRack SN40L-16 datasheet*. 2025. https://sambanova.ai/hubfs/SambaRack%20data%20sheet%20template%2007%2009%2025.pdf [REFERENCE]
- `[SambaN-Intel]` *Heterogeneous Inference Blueprint: GPUs/RDUs/CPUs*. 2026-04. https://www.businesswire.com/news/home/20260408117878/ [UNVERIFIED] [REFERENCE] [2026-NEW]
- `[Etched-Tweet]` *Etched Sohu announcement*. 2024-06. https://x.com/Etched/status/1805625693113663834 [UNVERIFIED] [REFERENCE]
- `[Etched-Status]` *Etched Sohu — status review*. 2026. https://awesomeagents.ai/hardware/etched-sohu/ [UNVERIFIED] [REFERENCE] [2026-NEW]
- `[DCD-Etched]` *Etched.ai raises $500m for $5bn valuation*. https://www.datacenterdynamics.com/en/news/etchedai-raises-500m-for-a-5bn-valuation-report/ [REFERENCE]
- `[Meta-MTIA-v1]` *MTIA v1*. 2023. https://ai.meta.com/blog/meta-training-inference-accelerator-AI-MTIA/ [REFERENCE]
- `[Meta-MTIA-v2]` *Our next generation MTIA*. 2024. https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/ [REFERENCE]
- `[Meta-MTIA-2025]` *Four MTIA Chips in Two Years*. 2025. https://ai.meta.com/blog/meta-mtia-scale-ai-chips-for-billions/ [REFERENCE] [2026-NEW]
- `[STH-MTIA]` *Meta Outlines New MTIA Accelerator Roadmap*. 2025. https://www.servethehome.com/meta-outlines-new-mtia-accelerator-roadmap-for-its-next-gen-ai-compute-mix/ [REFERENCE] [2026-NEW]
- `[Meta-ISCA25]` *Meta's Second Generation AI Chip: Model-Chip Co-Design (ISCA '25)*. 2025. https://dl.acm.org/doi/10.1145/3695053.3731409 [REFERENCE] [2026-NEW]
- `[MS-Maia200-Blog]` *Maia 200: The AI accelerator built for inference*. 2026-01. https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/ [REFERENCE] [2026-NEW]
- `[MS-Maia200-Deep]` *Deep dive into the Maia 200 architecture*. 2026. https://techcommunity.microsoft.com/blog/azureinfrastructureblog/deep-dive-into-the-maia-200-architecture/4489312 [REFERENCE] [2026-NEW]
- `[MS-Maia100]` *Inside Maia 100*. 2024. https://techcommunity.microsoft.com/blog/azureinfrastructureblog/inside-maia-100-revolutionizing-ai-workloads-with-microsofts-custom-ai-accelerat/4229118 [REFERENCE]
- `[TT-BH-Specs]` *Tenstorrent Blackhole Specifications*. https://docs.tenstorrent.com/aibs/blackhole/specifications.html [REFERENCE]
- `[TT-WH-Specs]` *Tenstorrent Wormhole Specifications*. https://docs.tenstorrent.com/aibs/wormhole/specifications.html [REFERENCE]
- `[TT-BH-Bench]` *Dissecting the Tenstorrent Blackhole Architecture via Microbenchmarking*. 2025-09. https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf [REFERENCE] [2026-NEW]
- `[Reg-TT]` *Blackhole QuietBox review*. 2025-11. https://www.theregister.com/2025/11/27/tenstorrent_quietbox_review/ [REFERENCE] [2026-NEW]
- `[Furiosa-RNGD]` *RNGD Specifications*. https://furiosa.ai/renegade-spec [REFERENCE]
- `[Furiosa-Server]` *Furiosa NXT RNGD Server*. 2025-09. https://furiosa.ai/blog/introducing-rngd-server-efficient-ai-inference-at-data-center-scale [REFERENCE] [2026-NEW]
- `[Reg-Furiosa]` *How AI chip upstart FuriosaAI won over LG*. 2025-07. https://www.theregister.com/2025/07/22/sk_furiosa_ai_lg/ [REFERENCE] [2026-NEW]

#### Networking and interconnect

- `[NV-NVLinkFusion-PR]` *NVIDIA Unveils NVLink Fusion*. 2025-05. https://investor.nvidia.com/news/press-release-details/2025/NVIDIA-Unveils-NVLink-Fusion-for-Industry-to-Build-Semi-Custom-AI-Infrastructure-With-NVIDIA-Partner-Ecosystem/default.aspx [REFERENCE] [2026-NEW]
- `[NV-NVLinkFusion-Page]` *NVIDIA NVLink Fusion product page*. https://www.nvidia.com/en-us/data-center/nvlink-fusion/ [REFERENCE] [2026-NEW]
- `[NV-NVLinkScaling]` *Scaling AI Inference Performance with NVLink and NVLink Fusion*. https://developer.nvidia.com/blog/scaling-ai-inference-performance-and-flexibility-with-nvidia-nvlink-and-nvlink-fusion/ [REFERENCE] [2026-NEW]
- `[NV-CX8-Blog]` *NVIDIA ConnectX-8 SuperNICs*. https://developer.nvidia.com/blog/nvidia-connectx-8-supernics-advance-ai-platform-architecture-with-pcie-gen6-connectivity/ [REFERENCE] [2026-NEW]
- `[STH-CX8]` *NVIDIA ConnectX-8 SuperNIC PCIe Gen6 800G NIC Detailed*. https://www.servethehome.com/nvidia-connectx-8-supernic-pcie-gen6-800g-nic-detailed/ [REFERENCE]
- `[IBTA-XDR]` *IBTA Launches XDR (800G) InfiniBand Specification*. 2024. https://www.fibermall.com/news/ibta-launches-xdr-800g-infiniband-spec.htm [REFERENCE]
- `[NV-QX-Photonics]` *NVIDIA Spectrum-X / Quantum-X Photonics CPO*. 2025-03. https://investor.nvidia.com/news/press-release-details/2025/NVIDIA-Announces-Spectrum-X-Photonics-Co-Packaged-Optics-Networking-Switches-to-Scale-AI-Factories-to-Millions-of-GPUs/default.aspx [REFERENCE] [2026-NEW]
- `[NV-CPO-Blog]` *A New Era in Data Center Networking with NVIDIA Silicon Photonics*. 2025. https://developer.nvidia.com/blog/a-new-era-in-data-center-networking-with-nvidia-silicon-photonics-based-network-switching/ [REFERENCE] [2026-NEW]
- `[NV-XGS-PR]` *NVIDIA Introduces Spectrum-XGS Ethernet*. 2025-08. https://nvidianews.nvidia.com/news/nvidia-introduces-spectrum-xgs-ethernet-to-connect-distributed-data-centers-into-giga-scale-ai-super-factories [REFERENCE] [2026-NEW]
- `[NV-XGS-Blog]` *Connecting Distributed Data Centers Into Large AI Factories*. https://developer.nvidia.com/blog/how-to-connect-distributed-data-centers-into-large-ai-factories-with-scale-across-networking/ [REFERENCE] [2026-NEW]
- `[UEC-PR]` *Ultra Ethernet Consortium Launches Specification 1.0*. 2025-06. https://ultraethernet.org/ultra-ethernet-consortium-uec-launches-specification-1-0-transforming-ethernet-for-ai-and-hpc-at-scale/ [REFERENCE] [2026-NEW]
- `[UEC-Spec]` *UE Specification 1.0*. https://ultraethernet.org/wp-content/uploads/sites/20/2025/06/UE-Specification-6.11.25.pdf [REFERENCE] [2026-NEW]
- `[UEC-Arch-Paper]` *Ultra Ethernet's Design Principles and Architectural Innovations*. 2025-08. arXiv:2508.08906. https://arxiv.org/html/2508.08906v1 [REFERENCE] [2026-NEW]
- `[Phoronix-UEC]` *UEC 1.0 Specification*. 2025-06. https://www.phoronix.com/news/Ultra-Ethernet-1.0-UEC [REFERENCE] [2026-NEW]
- `[NV-GPUDirect]` *GPUDirect (developer page)*. https://developer.nvidia.com/gpudirect [REFERENCE]
- `[NV-GPUDirect-Op]` *GPUDirect RDMA and GPUDirect Storage — GPU Operator*. https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-rdma.html [REFERENCE]
- `[NV-Quantum2]` *Quantum-2 InfiniBand Platform*. https://www.nvidia.com/en-us/networking/quantum2/ [REFERENCE]
- `[NV-Q800-Switch]` *Quantum-X800 XDR InfiniBand Switch*. https://www.naddod.com/products/nvidia-networking/102612 [REFERENCE] [2026-NEW]
- `[Tweak-RubinCPO]` *NVIDIA GTC 2025: GB300, Rubin, CPO details*. 2025-03. https://www.tweaktown.com/news/103938/nvidia-gtc-2025-gb300-ai-gpu-with-1-4kw-power-new-details-on-rubin-cpo-tech-and-more/index.html [REFERENCE] [2026-NEW]
- `[xAI-Colossus2]` *xAI Colossus Hits 2 GW: 555,000 GPUs*. 2026-01. https://introl.com/blog/xai-colossus-2-gigawatt-expansion-555k-gpus-january-2026 [REFERENCE] [2026-NEW]
- `[SemiA-Colossus2]` *xAI's Colossus 2*. https://newsletter.semianalysis.com/p/xais-colossus-2-first-gigawatt-datacenter [REFERENCE] [2026-NEW]
- `[NV-NVLink-Intuit]` *NVIDIA NVLink Explained*. https://intuitionlabs.ai/articles/nvidia-nvlink-gpu-interconnect [REFERENCE] [2026-NEW]
- `[Hopper-Wiki]` *Hopper (microarchitecture)*. https://en.wikipedia.org/wiki/Hopper_(microarchitecture) [REFERENCE]

#### Memory

- `[HBM-Wiki]` *High Bandwidth Memory*. https://en.wikipedia.org/wiki/High_Bandwidth_Memory [REFERENCE]
- `[Introl-HBM]` *HBM evolution: HBM3 → HBM3E → HBM4*. 2025. https://introl.com/blog/hbm-evolution-hbm3-hbm3e-hbm4-memory-ai-gpu-2025 [REFERENCE] [2026-NEW]
- `[Kynix-HBM]` *HBM3e vs HBM4: 2026 Specs*. https://www.kynix.com/Blog/hbm3e-vs-hbm4-2026-specs-performance--supply-guide.html [REFERENCE] [2026-NEW]


### Test-time compute and reasoning serving

#### Lineage

- `[Self-Consistency-2022]` *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. Wang et al. (Google). 2022-03. ICLR 2023, arXiv:2203.11171. https://arxiv.org/abs/2203.11171 [CANONICAL]
  - Establishes majority-vote-over-N as the dominant test-time aggregation.
- `[ToT-2023]` *Tree of Thoughts: Deliberate Problem Solving with LLMs*. Yao et al. (Princeton). 2023-05. NeurIPS 2023. https://openreview.net/forum?id=5Xc1ecxO1h [CANONICAL]
  - Canonical tree-search reasoning structure.
- `[GoT-2023]` *Graph of Thoughts: Solving Elaborate Problems with LLMs*. Besta et al. 2023-08. arXiv:2308.09687. https://arxiv.org/abs/2308.09687 [CANONICAL]
- `[OpenAI-o1-2024]` *Learning to Reason with LLMs*. OpenAI. 2024-09. https://openai.com/index/learning-to-reason-with-llms/ [REFERENCE]
- `[Snell-Compute-Optimal-2024]` *Scaling LLM Test-Time Compute Optimally*. Snell et al. (DeepMind). 2024-08. arXiv:2408.03314. https://openreview.net/forum?id=4FWAwZtd2n [CANONICAL]
  - Compute-budget framework: shortest-CoT for low budget, beam search medium, voting high.
- `[Tanay-ITS-2024]` *OpenAI's o-1 and Inference-Time Scaling Laws*. T. Janakiraman. 2024. https://www.tanayj.com/p/openais-o-1-and-inference-time-scaling [REFERENCE]
- `[SemiA-O1-Pro-2024]` *Scaling Laws – O1 Pro Architecture*. SemiAnalysis. 2024-12. https://semianalysis.com/2024/12/11/scaling-laws-o1-pro-architecture-reasoning-training-infrastructure-orion-and-claude-3-5-opus-failures/ [REFERENCE]

#### Past 12 months SOTA — reasoning serving

- `[OpenAI-o3-2025]` *Introducing OpenAI o3 and o4-mini*. OpenAI. 2025-04. https://openai.com/index/introducing-o3-and-o4-mini/ [REFERENCE] [2026-NEW]
- `[Claude-Adaptive-Thinking-2026]` *Adaptive thinking — Claude API*. Anthropic. 2026. https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking [REFERENCE] [2026-NEW]
- `[LRM-Survey-2025]` *Efficient Inference for Large Reasoning Models: A Survey*. Liu et al. 2025-03. arXiv:2503.23077. https://arxiv.org/abs/2503.23077 [REFERENCE] [2026-NEW]
- `[Stop-Overthinking-2025]` *Stop Overthinking: A Survey on Efficient Reasoning for LLMs*. 2025-03. arXiv:2503.16419. https://arxiv.org/pdf/2503.16419 [REFERENCE] [2026-NEW]
- `[AGoT-2025]` *Adaptive Graph of Thoughts: Test-Time Adaptive Reasoning Unifying Chain, Tree, and Graph Structures*. 2025-02. arXiv:2502.05078. https://arxiv.org/abs/2502.05078 [SOTA] [2026-NEW]
- `[Sample-Scrutinize-Scale-2025]` *Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification*. Google. 2025-02. ICML 2025, arXiv:2502.01839. https://arxiv.org/abs/2502.01839 [SOTA] [2026-NEW]
- `[Forest-of-Thought-2025]` *Forest-of-Thought: Scaling Test-Time Compute*. ICLR 2025. https://openreview.net/forum?id=BMJ3pyYxu2 [UNVERIFIED] [SOTA] [2026-NEW]
- `[Hermes-2025]` *Understanding and Optimizing Multi-Stage AI Inference Pipelines*. Krishna et al. (MIT CSAIL). 2025. https://people.csail.mit.edu/suvinay/pubs/2025.hermes.arxiv.pdf [SOTA] [2026-NEW]
- `[Reasoning-Serving-Empirical-2025]` *Reasoning Language Model Inference Serving Unveiled: An Empirical Study*. 2025-10. arXiv:2510.18672. https://arxiv.org/pdf/2510.18672 [SOTA] [2026-NEW]
- `[SparseSpec-2025]` *Accelerating Large-Scale Reasoning Model Inference with Sparse Self-Speculative Decoding*. 2025-12. arXiv:2512.01278. https://arxiv.org/abs/2512.01278 [SOTA] [2026-NEW]
  - Up to 2.13× throughput; co-designs scheduler, delayed verification, KV mgmt for self-speculation on long CoTs.

### Structured output and tool calling

#### Lineage

- `[Outlines-2023]` *Efficient Guided Generation for Large Language Models*. Willard, Louf (.txt). 2023-07. arXiv:2307.09702. https://arxiv.org/abs/2307.09702 [CANONICAL]
- `[Guidance-2023]` *guidance (library)*. Microsoft. 2023. https://github.com/guidance-ai/llguidance [REFERENCE]
- `[LMFE-2023]` *lm-format-enforcer*. N. Gat. 2023. https://github.com/noamgat/lm-format-enforcer [REFERENCE]
- `[Llama.cpp-Grammars]` *GBNF grammar support in llama.cpp*. ggerganov. 2023. https://github.com/ggerganov/llama.cpp [REFERENCE]
- `[DOMINO-2024]` *Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation*. Beurer-Kellner et al. (ETH SRI). 2024-03. arXiv:2403.06988. https://arxiv.org/html/2403.06988v1 [CANONICAL]

#### Past 12 months SOTA — structured output

- `[XGrammar-2024]` *XGrammar: Flexible and Efficient Structured Generation Engine for LLMs*. Dong, Ruan, Cai, Lai, Xu, Zhao, Chen (CMU/MLC, NVIDIA, SJTU, Berkeley). 2024-11. MLSys 2025, arXiv:2411.15100. https://arxiv.org/pdf/2411.15100 [SOTA] [2026-NEW]
  - Default backend in vLLM/SGLang/TRT-LLM/MLC; up to 14× JSON-schema, 80–100× CFG.
- `[llguidance-2024]` *llguidance: Super-fast Structured Outputs*. Microsoft / guidance-ai. 2024–25. https://github.com/guidance-ai/llguidance [SOTA] [2026-NEW]
  - ~50 µs CPU/token at 128K vocab.
- `[Outlines-Coalescence]` *Coalescence: making LLM inference 5x faster*. dottxt. 2024. https://blog.dottxt.ai/coalescence.html [REFERENCE]
- `[GenOutputs-Bench-2025]` *Generating Structured Outputs from Language Models: Benchmark and Studies*. 2025-01. arXiv:2501.10868. https://arxiv.org/html/2501.10868v1 [SOTA] [2026-NEW]
- `[FlexGCD-ICML2025]` *Flexible and Efficient Grammar-Constrained Decoding*. ICML 2025, arXiv:2502.05111. https://icml.cc/virtual/2025/poster/45613 [SOTA] [2026-NEW]
- `[IterGen-2025]` *IterGen: Forward-Backward Grammar Generation*. ICLR 2025. [UNVERIFIED] [EMERGING] [2026-NEW]
- `[GuidedDecoding-RAG-2025]` *Guided Decoding and Its Critical Role in RAG*. 2025-09. arXiv:2509.06631. https://arxiv.org/html/2509.06631v1 [SOTA] [2026-NEW]
- `[SqueezeBits-Bench-2025]` *Guided Decoding Performance on vLLM and SGLang*. SqueezeBits. 2025. https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang [REFERENCE] [2026-NEW]
- `[Anthropic-StructuredOutputs-2025]` *Structured Outputs (Claude 4.5/Opus 4.1)*. Anthropic. 2025. https://platform.claude.com/docs/en/build-with-claude/structured-outputs [REFERENCE] [2026-NEW]
- `[Anthropic-Advanced-Tool-Use-2025]` *Introducing advanced tool use on the Claude Developer Platform*. Anthropic. 2025. https://www.anthropic.com/engineering/advanced-tool-use [REFERENCE] [2026-NEW]
- `[Brenndoerfer-2025-survey]` *Constrained Decoding: Grammar-Guided Generation for Structured LLM Output*. 2025. https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output [REFERENCE] [2026-NEW]
- `[vLLM-StructDec-Blog]` *Structured Decoding in vLLM: A Gentle Introduction*. vLLM/BentoML. 2025-01. https://blog.vllm.ai/2025/01/14/struct-decode-intro.html [REFERENCE] [2026-NEW]

### Multimodal serving

#### Lineage

- `[LLaVA-2023]` *Visual Instruction Tuning*. Liu et al. 2023-04. NeurIPS 2023, arXiv:2304.08485. https://arxiv.org/abs/2304.08485 [CANONICAL]
- `[InternVL-2024]` *InternVL: Scaling up Vision Foundation Models*. OpenGVLab. 2024. CVPR 2024 Oral. https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_InternVL_Scaling_up_Vision_Foundation_Models_and_Aligning_for_Generic_CVPR_2024_paper.pdf [CANONICAL]
- `[Qwen2-VL-2024]` *Qwen2-VL: Enhancing VLM's Perception of the World at Any Resolution*. Qwen Team. 2024. [REFERENCE]
- `[Whisper-2022]` *Robust Speech Recognition via Large-Scale Weak Supervision*. Radford et al. (OpenAI). 2022-12. arXiv:2212.04356. https://arxiv.org/abs/2212.04356 [CANONICAL]
- `[GPT-4o-2024]` *Hello GPT-4o*. OpenAI. 2024-05. https://openai.com/index/hello-gpt-4o/ [REFERENCE]

#### Past 12 months SOTA — multimodal

- `[vLLM-V1-Multimodal]` *vLLM V1 (multimodal cache + chunked prefill for VLMs)*. vLLM Team. 2025-01. https://blog.vllm.ai/2025/01/27/v1-alpha-release.html [REFERENCE] [2026-NEW]
- `[HydraInfer-2025]` *HydraInfer: Hybrid Disaggregated Scheduling for Multimodal LLM Inference*. 2025-05. arXiv:2505.12658. https://arxiv.org/pdf/2505.12658 [SOTA] [2026-NEW]
- `[ModServe-2025]` *ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving*. Qiu et al. SoCC 2025. https://dl.acm.org/doi/pdf/10.1145/3772052.3772254 [SOTA] [2026-NEW]
- `[Nova-2025]` *Nova: Real-Time Agentic Vision-Language Model Serving with Adaptive Cross-Stage Parallelization*. 2025-09. arXiv:2509.21301. https://arxiv.org/abs/2509.21301 [SOTA] [2026-NEW]
- `[SpaceServe-2025]` *SpaceServe: Spatial Multiplexing of Complementary Resources*. OpenReview 2025. https://openreview.net/pdf?id=...3b4c3201 [UNVERIFIED] [SOTA] [2026-NEW]
- `[LMM-Characterization-2025]` *Towards Efficient Large Multimodal Model Serving*. Qiu et al. 2025-02. arXiv:2502.00937. https://arxiv.org/html/2502.00937v1 [SOTA] [2026-NEW]
- `[FastVLM-CVPR2025]` *FastVLM: Efficient Vision Encoding for Vision Language Models*. Apple ML. 2025-06. CVPR 2025. https://machinelearning.apple.com/research/fast-vision-language-models [SOTA] [2026-NEW]
  - 85× TTFT vs LLaVA-OneVision-0.5B; 7.9× at 7B.
- `[HF-VLMs-2025]` *Vision Language Models (Better, faster, stronger)*. Hugging Face. 2025. https://huggingface.co/blog/vlms-2025 [REFERENCE] [2026-NEW]
- `[Qwen2.5-Omni-2025]` *Qwen2.5-Omni Technical Report*. Alibaba Qwen. 2025-03. https://github.com/QwenLM/Qwen2.5-Omni [REFERENCE] [2026-NEW]
- `[Qwen3-Omni-2025]` *Qwen3-Omni*. Alibaba Qwen. 2025. https://github.com/QwenLM/Qwen3-Omni [REFERENCE] [2026-NEW]
- `[F16-VideoLLM-2025]` *Improving LLM Video Understanding with 16 Frames Per Second*. ICML 2025. https://icml.cc/virtual/2025/poster/46540 [REFERENCE] [2026-NEW]
- `[StreamingVideoLLM-2025]` *Streaming VideoLLMs for Real-time Procedural Video Understanding*. ICCV 2025. https://www.openaccess.thecvf.com/content/ICCV2025/papers/Chatterjee_Streaming_VideoLLMs_for_Real-Time_Procedural_Video_Understanding_ICCV_2025_paper.pdf [SOTA] [2026-NEW]
- `[LMCache-MM-2025]` *LMCache Extends Its Turbo-Boost to Multimodal Models in vLLM V1*. 2025-07. https://blog.lmcache.ai/en/2025/07/03/lmcache-extends-its-turbo-boost-to-multimodal-models-in-vllm-v1/ [REFERENCE] [2026-NEW]
- `[MLPerf-Whisper-2025]` *Whisper: An MLPerf Inference Benchmark for ASR*. MLCommons. 2025-09. https://mlcommons.org/2025/09/whisper-inferencev5-1/ [REFERENCE] [2026-NEW]
- `[FasterWhisper]` *SYSTRAN/faster-whisper*. SYSTRAN. https://github.com/SYSTRAN/faster-whisper [REFERENCE]
- `[Batched-Whisper-2024]` *Speeding up Whisper (ASR)*. Mobius ML. 2024. https://mobiusml.github.io/batched_whisper_blog/ [REFERENCE]
- `[vLLM-Omni-2026]` *vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models*. 2026. arXiv:2602.02204. https://arxiv.org/abs/2602.02204 [SOTA] [2026-NEW]

### RAG infrastructure

#### Lineage

- `[FAISS-2017]` *Billion-scale similarity search with GPUs*. Johnson, Douze, Jégou (FAIR). 2017. arXiv:1702.08734. https://arxiv.org/abs/1702.08734 [CANONICAL]
- `[HNSW-2018]` *Efficient and robust ANN search using Hierarchical Navigable Small World graphs*. Malkov, Yashunin. 2018. arXiv:1603.09320. https://arxiv.org/abs/1603.09320 [CANONICAL]
- `[ScaNN-2020]` *Accelerating Large-Scale Inference with Anisotropic Vector Quantization*. Guo et al. (Google). 2020. ICML 2020, arXiv:1908.10396. https://arxiv.org/abs/1908.10396 [CANONICAL]
- `[DiskANN-2019]` *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search*. Subramanya et al. (Microsoft). 2019. NeurIPS 2019. https://suhasjs.github.io/files/diskann_neurips19.pdf [CANONICAL]
- `[SPANN-2021]` *SPANN: Highly-efficient Billion-scale ANN*. Chen et al. (Microsoft). 2021. NeurIPS 2021, arXiv:2111.08566. https://arxiv.org/abs/2111.08566 [CANONICAL]
- `[ColBERT-2020]` *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*. Khattab, Zaharia. 2020-04. SIGIR 2020, arXiv:2004.12832. https://arxiv.org/abs/2004.12832 [CANONICAL]
- `[ColBERTv2-2022]` *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction*. Santhanam et al. 2022. NAACL 2022, arXiv:2112.01488. https://arxiv.org/abs/2112.01488 [CANONICAL]
- `[PLAID-2022]` *PLAID: An Efficient Engine for Late Interaction Retrieval*. Santhanam et al. 2022-05. CIKM 2022, arXiv:2205.09707. https://arxiv.org/abs/2205.09707 [CANONICAL]
- `[SPLADE-2021]` *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking*. Formal, Lassance, Piwowarski, Clinchant. 2021. SIGIR 2021/2022, arXiv:2107.05720. https://arxiv.org/abs/2107.05720 [CANONICAL]
- `[BGE-M3-2024]` *BGE-M3: Multi-Functionality, Multi-Linguistic, Multi-Granular Embeddings*. BAAI. 2024. arXiv:2402.03216. https://arxiv.org/abs/2402.03216 [CANONICAL]

#### Past 12 months SOTA — RAG

- `[ColPali-2024]` *ColPali: Efficient Document Retrieval with Vision Language Models*. Faysse et al. (Illuin). 2024-07. ICLR 2025, arXiv:2407.01449. https://arxiv.org/abs/2407.01449 [SOTA]
- `[ColQwen-2025]` *ColQwen2 / ColQwen2.5*. Illuin / community. 2025. https://github.com/illuin-tech/colpali [SOTA] [2026-NEW]
- `[Cache-Craft-2025]` *Cache-Craft: Managing Chunk-Caches for Efficient RAG*. Kejriwal et al. (Adobe Research). 2025-02. SIGMOD 2025, arXiv:2502.15734. https://arxiv.org/abs/2502.15734 [SOTA] [2026-NEW]
  - -51% redundancy vs prefix caching; 1.6× lower production latency.
- `[AgenticRAG-Survey-2025]` *Agentic Retrieval-Augmented Generation: A Survey*. Singh et al. 2025-01. arXiv:2501.09136. https://arxiv.org/abs/2501.09136 [REFERENCE] [2026-NEW]
- `[AgenticRAG-KG-2025]` *Agentic RAG with Knowledge Graphs for Complex Multi-Hop Reasoning*. 2025-07. arXiv:2507.16507. https://arxiv.org/abs/2507.16507 [SOTA] [2026-NEW]
- `[DistributedANN-2025]` *DistributedANN: Efficient Scaling of a Single DiskANN Graph Across Thousands of Computers*. Microsoft. 2025-09. arXiv:2509.06046. https://arxiv.org/abs/2509.06046 [SOTA] [2026-NEW]
  - 26 ms median, >100K QPS over 50B-vector index.
- `[SPFresh-2024]` *SPFresh: Incremental In-Place Update for Billion-Scale Vector Search*. 2024-10. arXiv:2410.14452. https://arxiv.org/pdf/2410.14452 [SOTA]
- `[Rerankers-Bench-2025]` *Top 8 Rerankers: Quality vs Cost benchmark*. 2025-09. https://aimultiple.com/rerankers [REFERENCE] [2026-NEW]
- `[VectorDBBench]` *VectorDBBench*. Zilliz. https://github.com/zilliztech/VectorDBBench [REFERENCE]
- `[Vespa-Billion-2024]` *Building Billion-Scale Vector Search (Vespa)*. https://medium.com/vespa/building-billion-scale-vector-search-part-two-94f0101d15dd [REFERENCE]
- `[pgvector-08]` *pgvector 0.8 on Aurora*. AWS. 2024–25. https://aws.amazon.com/blogs/database/supercharging-vector-search-performance-and-relevance-with-pgvector-0-8-0-on-amazon-aurora-postgresql/ [REFERENCE]
- `[pgvectorscale-2024]` *pgvectorscale (Timescale)*. https://github.com/timescale/pgvectorscale [REFERENCE]
- `[MCP-Donation-2025]` *Anthropic donates MCP to Linux Foundation Agentic AI Foundation*. 2025-12. [UNVERIFIED] [REFERENCE] [2026-NEW]

### Embedding and reranker serving

#### Lineage

- `[Sentence-BERT-2019]` *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. Reimers, Gurevych. 2019-08. EMNLP 2019, arXiv:1908.10084. https://arxiv.org/abs/1908.10084 [CANONICAL]
- `[E5-2022]` *Text Embeddings by Weakly-Supervised Contrastive Pre-training*. Wang et al. (Microsoft). 2022. arXiv:2212.03533. https://arxiv.org/abs/2212.03533 [CANONICAL]
- `[BGE-2023]` *C-Pack / BGE: Packaged Resources To Advance General Chinese Embeddings*. Xiao et al. (BAAI). 2023. arXiv:2309.07597. https://arxiv.org/abs/2309.07597 [CANONICAL]
- `[Matryoshka-2022]` *Matryoshka Representation Learning*. Kusupati et al. 2022. NeurIPS 2022, arXiv:2205.13147. https://arxiv.org/abs/2205.13147 [CANONICAL]
- `[MTEB-2022]` *MTEB: Massive Text Embedding Benchmark*. Muennighoff et al. 2022. EACL 2023, arXiv:2210.07316. https://arxiv.org/abs/2210.07316 [CANONICAL]

#### Past 12 months SOTA — embeddings/rerankers

- `[MMTEB-2025]` *MMTEB: Massive Multilingual Text Embedding Benchmark*. Enevoldsen et al. 2025-02. ICLR 2025, arXiv:2502.13595. https://arxiv.org/abs/2502.13595 [SOTA] [2026-NEW]
  - 250+ languages, 500+ tasks; new leaderboard default.
- `[MTEB-Maintenance-2025]` *Maintaining MTEB*. 2025-06. arXiv:2506.21182. https://arxiv.org/html/2506.21182v1 [REFERENCE] [2026-NEW]
- `[Qwen3-Embedding-2025]` *Qwen3 Embedding: Advancing Text Embedding and Reranking through Foundation Models*. Qwen Team. 2025-06. arXiv:2506.05176. https://arxiv.org/pdf/2506.05176 [SOTA] [2026-NEW]
  - Qwen3-Embedding-8B leads MMTEB multilingual at 70.58.
- `[NV-Embed-v2]` *NV-Embed-v2*. NVIDIA. 2024-25. https://huggingface.co/nvidia/NV-Embed-v2 [REFERENCE]
- `[Voyage-3.5]` *Voyage 3.5 / 3.5-Lite*. Voyage AI / MongoDB. 2025. https://docs.voyageai.com/docs/flexible-dimensions-and-quantization [REFERENCE] [2026-NEW]
- `[HF-Embedding-Quant-2024]` *Binary and Scalar Embedding Quantization*. HF / Sentence-Transformers. 2024. https://huggingface.co/blog/embedding-quantization [REFERENCE]
- `[Vespa-Matryoshka-Binary]` *Matryoshka 🤝 Binary vectors*. Vespa. 2024. https://blog.vespa.ai/combining-matryoshka-with-binary-quantization-using-embedder/ [REFERENCE]
- `[TEI-Repo]` *huggingface/text-embeddings-inference*. https://github.com/huggingface/text-embeddings-inference [REFERENCE]
- `[Infinity-Repo]` *michaelfeil/infinity*. https://github.com/michaelfeil/infinity [REFERENCE]
- `[Cohere-Rerank-Docs]` *Cohere Rerank*. https://docs.cohere.com/docs/rerank [REFERENCE]
- `[Jina-Rerank]` *Jina Reranker v2/v3*. Jina AI. 2024–25. https://jina.ai [REFERENCE]
- `[BGE-Reranker]` *BGE Reranker family*. BAAI. https://huggingface.co/BAAI [REFERENCE]
- `[ModernBERT-2024]` *ModernBERT: A Modern Bidirectional Encoder*. Warner et al. (Answer.AI / LightOn). 2024-12. arXiv:2412.13663. https://arxiv.org/abs/2412.13663 [CANONICAL]
- `[4bit-Quant-RAG-2025]` *4bit-Quantization in Vector-Embedding for RAG*. 2025-01. arXiv:2501.10534. https://arxiv.org/html/2501.10534v1 [EMERGING] [2026-NEW]
- `[Qwen3-Embedding-Repo]` *QwenLM/Qwen3-Embedding*. Alibaba Qwen. 2025-06. https://github.com/QwenLM/Qwen3-Embedding [REFERENCE] [2026-NEW]
- `[AER-Labs-2025-blog]` *Optimizing Embedding Model Inference*. AER Labs. 2025. https://aerlabs.tech/blogs/optimizing-embedding-model-inference [REFERENCE] [2026-NEW]
- `[Filip-Substack-2024]` *Comparing Embedding Inference Solutions: TEI, Infinity, FastEmbed*. F. Makraduli. https://open.substack.com/pub/filipmakraduli/p/comparing-embedding-inference-solutions [REFERENCE]
- `[HF-MTEB-Leaderboard]` *HF MTEB Leaderboard*. https://huggingface.co/spaces/mteb/leaderboard [REFERENCE]


### RL post-training infrastructure

#### Lineage

- `[InstructGPT]` *Training language models to follow instructions with human feedback*. Ouyang et al. (OpenAI). 2022. arXiv:2203.02155. https://arxiv.org/abs/2203.02155 [CANONICAL]
  - SFT → RM → PPO three-stage RLHF recipe template.
- `[CAI]` *Constitutional AI: Harmlessness from AI Feedback*. Bai et al. (Anthropic). 2022-12. arXiv:2212.08073. https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback [CANONICAL]
- `[DeepSpeed-Chat]` *DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training*. Yao et al. (Microsoft). 2023-08. arXiv:2308.01320. https://arxiv.org/abs/2308.01320 [CANONICAL]
  - Hybrid Engine; 10× speedup over HF/Colossal-AI baselines.
- `[OpenRLHF]` *OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework*. Hu et al. 2024-05. EMNLP-Demos 2025, arXiv:2405.11143. https://arxiv.org/abs/2405.11143 [CANONICAL]
  - Ray + vLLM + DeepSpeed ZeRO; canonical NCCL broadcast and CUDA-IPC weight-sync.
- `[DeepSeekMath-GRPO]` *DeepSeekMath: Pushing the Limits of Mathematical Reasoning*. Shao et al. (DeepSeek). 2024-02. arXiv:2402.03300. https://arxiv.org/abs/2402.03300 [CANONICAL]
  - Origin of GRPO.
- `[HybridFlow-veRL]` *HybridFlow: A Flexible and Efficient RLHF Framework*. Sheng et al. (ByteDance Seed + HKU). 2024-09. arXiv:2409.19256. https://arxiv.org/abs/2409.19256 [CANONICAL]
  - Hybrid single+multi-controller; 3D-HybridEngine resharding; 1.53–20.57× over baselines.
- `[NeMo-Aligner]` *NeMo-Aligner: Scalable Toolkit for Efficient Model Alignment*. Shen et al. (NVIDIA). 2024-05. arXiv:2405.01481. https://arxiv.org/abs/2405.01481 [CANONICAL]
- `[Async-RLHF]` *Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models*. Noukhovitch et al. (MILA). 2024-10. ICLR 2025, arXiv:2410.18252. https://arxiv.org/abs/2410.18252 [CANONICAL]
  - ~40–70% speedup; off-policy tolerance increases with policy-model scale.

#### Past 12 months SOTA — RL frameworks

- `[veRL-0.7]` *verl 0.7 release blog*. verl-project / ByteDance. 2026-01. https://verl.readthedocs.io/en/latest/blog/v0.7.html [SOTA] [2026-NEW]
  - Native server-mode rollout; Checkpoint Engine (NCCL + NIXL); on-policy / one-step-off / fully-async.
- `[AReaL]` *AReaL: A Large-Scale Asynchronous RL System for Language Reasoning*. Fu et al. (Tsinghua/Ant). 2025-05. arXiv:2505.24298. https://arxiv.org/abs/2505.24298 [SOTA] [2026-NEW]
  - Fully asynchronous; modified PPO; 2.77× speedup over sync.
- `[slime]` *slime: An SGLang-Native Post-Training Framework for RL Scaling*. THUDM (Tsinghua) + LMSYS. 2025-07. https://www.lmsys.org/blog/2025-07-09-slime/ [SOTA] [2026-NEW]
  - Powers GLM-4.5/4.6/4.7.
- `[SkyRL]` *SkyRL: A Modular Full-stack RL Library for LLMs*. NovaSky-AI (Berkeley) + Anyscale. 2025. https://github.com/NovaSky-AI/SkyRL [SOTA] [2026-NEW]
- `[NeMo-RL]` *NeMo-RL*. NVIDIA. 2025. https://github.com/NVIDIA-NeMo/RL [SOTA] [2026-NEW]
  - Megatron-Core backend; supports DAPO; v0.6 ships speculative decoding inside RL loop.
- `[ROLL-Flash]` *ROLL Flash – Accelerating RLVR and Agentic Training with Asynchrony*. Alibaba. 2025-10. arXiv:2510.11345. https://arxiv.org/abs/2510.11345 [SOTA] [2026-NEW]
  - 2.24× RLVR / 2.72× agentic; near-linear scaling at 100+ GPUs.
- `[RollMux]` *RollMux: Phase-level Multiplexing*. 2025. arXiv:2512.11306. https://arxiv.org/abs/2512.11306 [EMERGING] [2026-NEW]
- `[RollArt]` *RollArt: Agentic Disaggregated Infra*. 2025. arXiv:2512.22560. https://arxiv.org/abs/2512.22560 [EMERGING] [2026-NEW]
- `[OpenRLHF-2025]` *OpenRLHF*. Hu et al. 2025. EMNLP-Demos 2025, https://github.com/OpenRLHF/OpenRLHF [SOTA] [2026-NEW]
  - Adds DAPO, REINFORCE++, async agentic RL, partial-rollout, TIS.
- `[Awex]` *Awex: An Ultra-Fast Weight Sync Framework for Trillion-Scale RL*. Ant Group. 2025. https://github.com/inclusionAI/asystem-awex [SOTA] [2026-NEW]
  - 1T params synced in 6s (RDMA) / 20s (NCCL) on thousand-GPU clusters.
- `[checkpoint-engine]` *Mooncake checkpoint-engine*. Moonshot AI. 2025-09. https://github.com/MoonshotAI/checkpoint-engine [SOTA] [2026-NEW]
  - Second-level updates of trillion-param models for vLLM/SGLang.
- `[Tinker]` *Tinker*. Thinking Machines Lab. 2025-10. https://thinkingmachines.ai/tinker/ [REFERENCE] [2026-NEW]
- `[PipelineRL]` *PipelineRL*. ServiceNow Research. 2025. [SOTA] [2026-NEW]
  - Per-forward-pass weight swaps; Redis-backed rollout stream.
- `[TorchForge]` *TorchForge*. Meta. 2025. [EMERGING] [2026-NEW]

#### Past 12 months SOTA — RL algorithms / objectives

- `[DAPO]` *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. ByteDance Seed. 2025-03. arXiv:2503.14476. https://arxiv.org/abs/2503.14476 [SOTA] [2026-NEW]
  - Clip-Higher, Dynamic Sampling, Token-Level PG Loss, Overlong Reward Shaping.
- `[Dr.GRPO]` *Understanding R1-Zero-Like Training: A Critical Perspective*. Liu et al. (Sea AI Lab). 2025-03. arXiv:2503.20783. https://arxiv.org/abs/2503.20783 [SOTA] [2026-NEW]
- `[Open-Reasoner-Zero]` *Open-Reasoner-Zero*. StepFun. 2025-03. arXiv:2503.24290. https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero [SOTA] [2026-NEW]
- `[J1]` *J1: Incentivizing Thinking in LLM-as-a-Judge via RL*. 2025-05. arXiv:2505.10320. https://arxiv.org/abs/2505.10320 [SOTA] [2026-NEW]
- `[Truncated-PPO]` *Truncated Proximal Policy Optimization*. arXiv:2506.15050. https://arxiv.org/abs/2506.15050 [SOTA] [2026-NEW]
- `[Infinite-Sampling]` *Infinite Sampling: Efficient and Stable Grouped RL Training*. arXiv:2506.22950. https://arxiv.org/abs/2506.22950 [SOTA] [2026-NEW]
- `[SPEC-RL]` *SPEC-RL: Accelerating On-Policy RL via Speculative Rollouts*. 2025-09. arXiv:2509.23232. https://arxiv.org/abs/2509.23232 [SOTA] [2026-NEW]
- `[Asys-Surveys]` *A Survey of Reinforcement Learning for Large Reasoning Models*. Tsinghua C3I. 2025-09. arXiv:2509.08827. https://arxiv.org/abs/2509.08827 [REFERENCE] [2026-NEW]
- `[FP16-RL]` *Defeating the Training-Inference Mismatch via FP16*. arXiv:2510.26788. https://arxiv.org/abs/2510.26788 [SOTA] [2026-NEW]
- `[Bitwise-Consistent-RL]` *No More Train-Inference Mismatch: Bitwise Consistent On-Policy RL with vLLM and TorchTitan*. vLLM Blog. 2025-11. https://blog.vllm.ai/2025/11/10/bitwise-consistent-rl.html [SOTA] [2026-NEW]
- `[Laminar]` *Laminar: A Scalable Asynchronous RL Post-Training Framework*. arXiv:2510.12633. https://arxiv.org/abs/2510.12633 [EMERGING] [2026-NEW]
- `[RollPacker]` *RollPacker: Mitigating Long-Tail Rollouts*. arXiv:2509.21009. https://arxiv.org/abs/2509.21009 [EMERGING] [2026-NEW]
- `[Periodic-Async]` *Periodic Asynchrony: An Effective Method for Accelerating On-Policy RL*. arXiv:2511.18871. https://arxiv.org/abs/2511.18871 [EMERGING] [2026-NEW]
- `[A-3PO]` *A-3PO: Accelerating Asynchronous LLM Training with Staleness-aware Proximal Policy Approximation*. arXiv:2512.06547. https://arxiv.org/abs/2512.06547 [EMERGING] [2026-NEW]
- `[TensorHub]` *TensorHub: Scalable and Elastic Weight Transfer for LLM RL Training*. arXiv:2604.09107. https://arxiv.org/abs/2604.09107 [UNVERIFIED] [EMERGING] [2026-NEW]
- `[NeMo-RL-SpecDec]` *Speculative Decoding in NeMo-RL*. NVIDIA. 2026-05. [SOTA] [2026-NEW]
  - 1.8× rollout speedup at 8B; projected 2.5× E2E at 235B.
- `[HF-AsyncRL]` *Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries*. Hugging Face. 2026-03. https://huggingface.co/blog/async-rl-training-landscape [REFERENCE] [2026-NEW]
- `[Anyscale-Comparison]` *Open Source RL Libraries for LLMs*. Anyscale. 2025. [REFERENCE] [2026-NEW]
- `[LangCopilot-Comparison]` *OpenRLHF vs veRL: Ray Framework Deep Dive*. 2025-11. https://langcopilot.com/posts/2025-11-06-openrlhf-vs-verl-ray-framework-deep [REFERENCE] [2026-NEW]
- `[Absolute-Zero]` *Absolute Zero: Reinforced Self-play Reasoning with Zero Data*. arXiv:2505.03335. https://arxiv.org/abs/2505.03335 [EMERGING] [2026-NEW]
- `[Llama-4-RL]` *Llama 4 Multimodal Intelligence — RL pipeline*. Meta. 2025-04. https://ai.meta.com/blog/llama-4-multimodal-intelligence/ [REFERENCE] [2026-NEW]
- `[ProRL-Agent]` *ProRL Agent: Rollout-as-a-Service for RL Training of Multi-Turn LLM Agents*. NVIDIA. 2026-03. arXiv:2603.18815. https://arxiv.org/abs/2603.18815 [SOTA] [2026-NEW]
- `[VerlTool]` *VerlTool: Holistic Agentic RL with Tool Use*. arXiv:2509.01055. https://arxiv.org/abs/2509.01055 [SOTA] [2026-NEW]
- `[OpenSandbox]` *Alibaba's open-source sandbox runtime for agentic RL*. https://github.com/alibaba/OpenSandbox [REFERENCE]
- `[Outcome-Process-Harmonization]` *Beyond Correctness: Harmonizing Process and Outcome Rewards*. arXiv:2509.03403. https://arxiv.org/abs/2509.03403 [EMERGING] [2026-NEW]
- `[awesome-RLVR]` *Awesome-RLVR (curated list)*. https://github.com/opendilab/awesome-RLVR [REFERENCE]
- `[vLLM-SleepMode]` *Zero-Reload Model Switching with vLLM Sleep Mode*. vLLM Blog. 2025-10. https://blog.vllm.ai/2025/10/26/vllm-sleep-mode.html [REFERENCE] [2026-NEW]
- `[vLLM-RFC-31848]` *Native Weight Syncing APIs (vLLM)*. https://github.com/vllm-project/vllm/issues/31848 [REFERENCE] [2026-NEW]
- `[SGLang-RL-Doc]` *SGLang for RL Systems*. https://sgl-project.github.io/advanced_features/sglang_for_rl.html [REFERENCE] [2026-NEW]
- `[Perplexity-WeightTransfer]` *Weight Transfer for RL Post-Training in under 2 seconds*. Perplexity Research. https://research.perplexity.ai/articles/rdma-point-to-point-communication-for-llm-systems [REFERENCE] [2026-NEW]
- `[Kimi-K2-RL]` See `[Kimi-K2]` above. RLVR + self-critique rubric reward; K8s sandbox >10k concurrent instances.

### Production engines and orchestrators (OSS deep-dive references)

- `[vLLM]` *vLLM*. vLLM Project (Berkeley/UC). https://github.com/vllm-project/vllm [REFERENCE]
  - Reference open-source LLM serving engine; PagedAttention, V1 architecture, ecosystem hub.
- `[vLLM-V1-blog]` *vLLM V1: A Major Upgrade to vLLM's Core Architecture*. vLLM Team. 2025-01. https://blog.vllm.ai/2025/01/27/v1-alpha-release.html [REFERENCE] [2026-NEW]
- `[SGLang]` *SGLang*. LMSYS / SGLang. https://github.com/sgl-project/sglang [REFERENCE]
  - RadixAttention, prefix-aware scheduler, native MoE/EP/disagg.
- `[TensorRT-LLM]` *TensorRT-LLM*. NVIDIA. https://github.com/NVIDIA/TensorRT-LLM [REFERENCE]
  - NVIDIA's official LLM serving stack; in-flight batching, FP8/NVFP4, Wide-EP.
- `[TGI]` *Text Generation Inference*. Hugging Face. https://github.com/huggingface/text-generation-inference [REFERENCE]
- `[llama.cpp]` *llama.cpp*. ggerganov. https://github.com/ggerganov/llama.cpp [REFERENCE]
  - GGUF quants, CPU/Apple/edge baseline.
- `[mistral.rs]` *mistral.rs*. EricLBuehler. https://github.com/EricLBuehler/mistral.rs [REFERENCE]
- `[mlx]` *Apple MLX*. https://github.com/ml-explore/mlx [REFERENCE]
- `[mlx-lm]` *Apple MLX-LM*. https://github.com/ml-explore/mlx-lm [REFERENCE]
- `[mlx-examples]` *Apple MLX examples*. https://github.com/ml-explore/mlx-examples [REFERENCE]
- `[ktransformers]` *KTransformers (CPU/GPU hybrid)*. kvcache-ai. https://github.com/kvcache-ai/ktransformers [REFERENCE] [2026-NEW]
- `[KTransformers-LMSYS]` *KTransformers LMSYS Blog*. 2025-10. https://lmsys.org/blog/2025-10-22-KTransformers/ [REFERENCE] [2026-NEW]
- `[LMCache-repo]` *LMCache/LMCache*. https://github.com/LMCache/LMCache [REFERENCE] [2026-NEW]
- `[LMCache-PS-K8s]` *vLLM Production Stack on K8s with LMCache*. https://blog.lmcache.ai/en/2025/01/21/high-performance-and-easy-deployment-of-vllm-in-k8s-with-vllm-production-stack/ [REFERENCE] [2026-NEW]
- `[Mooncake-repo]` *Mooncake (KVCache-centric)*. kvcache-ai. https://github.com/kvcache-ai/Mooncake [REFERENCE] [2026-NEW]
- `[NVIDIA-Dynamo-repo]` *ai-dynamo/dynamo + ai-dynamo/nixl*. https://github.com/ai-dynamo/dynamo [REFERENCE] [2026-NEW]
- `[llm-d-repo]` *llm-d*. https://github.com/llm-d/llm-d [REFERENCE] [2026-NEW]
- `[AIBrix-repo]` *AIBrix*. https://github.com/vllm-project/aibrix [REFERENCE] [2026-NEW]
- `[Llumnix-repo]` *Llumnix (open-source migration scheduler)*. https://github.com/AlibabaPAI/llumnix [REFERENCE]


## Past 12 months: at a glance (chronological)

This section re-lists every entry tagged `[2026-NEW]` (May 2025 – May 2026) in flat chronological order. Where exact month is unclear from a brief, the entry is placed at the broadest plausible point in the year.

### 2025-05

- `[GLA-GTA]` Hardware-Efficient Attention for Fast Decoding (Princeton).
- `[SageAttn3]` SageAttention3: Microscaling FP4 Attention (THU/Shengshu).
- `[Quartet]` Native FP4 Training (IST-DASLab).
- `[BeyondBuzz]` Beyond the Buzz: A Pragmatic Take on Inference Disaggregation (NVIDIA).
- `[MegaScale-MoE]` Large-Scale Communication-Efficient MoE Training (ByteDance).
- `[DeepSeek-V3-Insights]` Insights into DeepSeek-V3 (ISCA 2025).
- `[ELORA]` Multi-LoRA + KV management (HKUST).
- `[ServerlessLoRA]` Serverless multi-LoRA inference.
- `[SGLang-LargeEP]` DeepSeek with PD Disagg + Large-Scale EP on 96 H100 (LMSYS).
- `[HEXGEN-FLOW]` Hexgen-Text2SQL.
- `[Prism]` GPU sharing for cost-efficient multi-LLM serving.
- `[HydraInfer-2025]` Hybrid Disaggregated Scheduling for Multimodal LLM Inference.
- `[J1]` Incentivizing Thinking in LLM-as-a-Judge via RL.
- `[AReaL]` Large-Scale Asynchronous RL System for Language Reasoning.
- `[Theory-Scaling]` Scaling Laws for Speculative Decoding.
- `[Absolute-Zero]` Reinforced Self-play Reasoning with Zero Data.
- `[SOLA]` State-Aware SLO Scheduling (MLSys 2025).

### 2025-06

- `[BASE-Q]` Bias and Asymmetric Scaling Enhanced Rotational Quantization.
- `[AnTKV]` Anchor Token-Aware Sub-Bit Vector Quantization.
- `[OSP]` Outlier-Safe Pre-Training.
- `[MXFP8-Recipes]` Recipes for Pre-training LLMs with MXFP8.
- `[FastVLM-CVPR2025]` Apple FastVLM.
- `[Lookahead-Reasoning]` Step-level speculation for reasoning models.
- `[MoE-Cascade]` Utility-Driven Speculative Decoding for MoE.
- `[FlashDMoE]` Fast Distributed MoE in a Single Kernel.
- `[Truncated-PPO]` Truncated PPO.
- `[Infinite-Sampling]` Stable Grouped RL Training.
- `[SGLang-GB200-1]` DeepSeek on GB200 NVL72 Part I.
- `[BeyondBuzz]` (also referenced earlier in 2025).
- `[BeyondBuzz]` redundancy elided.
- `[Qwen3-Embedding-2025]` MMTEB-leading multilingual embedding model.
- `[MTEB-Maintenance-2025]` MTEB v2 disclaimer.

### 2025-07

- `[Falcon-H1]` Hybrid-head 0.5B–34B with 256K context (TII).
- `[SpecForge]` Open-source EAGLE-3 training framework.
- `[SGLang-K2-128]` Kimi K2 with PD Disagg + Large-Scale EP on 128 H200.
- `[Step-3]` Model-system Co-design for Cost-effective Decoding (StepFun).
- `[Kimi-K2]` 1.04T/32B-active MoE; 384 experts, MuonClip.
- `[KVFlow]` Multi-agent prefix caching.
- `[BucketServe]` Bucket-based dynamic batching.
- `[slime]` SGLang-Native RL Post-Training Framework.
- `[Edge-Cloud-Survey]` Collaborative Inference and Learning Survey.
- `[BrownoutServe]` SLO-Aware Inference under Bursty Workloads.
- `[AgenticRAG-KG-2025]` Agentic RAG with Knowledge Graphs.
- `[EdgeLoRA]` Multi-Tenant LLM Serving on Edge Devices.
- `[Toppings]` CPU-Assisted Rank-Aware Adapter Serving (USENIX ATC 2025).
- `[Reg-Furiosa]` Furiosa AI / LG.

### 2025-08

- `[Llama-Spec-Meta]` Production EAGLE for Llama at scale (Meta).
- `[SpecMoEOff]` MoE Inference + Speculative Decoding for offload latency.
- `[FlashCommV2]` Sub-byte MoE communication.
- `[MicroMix]` Mixed MXFP4/6/8 per channel.
- `[Optimal-LLM-Sched]` Throughput-optimality theory; RAD scheduler.
- `[TaiChi]` Aggregation vs Disaggregation unified.
- `[HeteroScale]` Coordinated Autoscaling for Heterogeneous and Disaggregated LLM Inference.
- `[CW-GB300]` CoreWeave GB300 NVL72 instances.
- `[NV-XGS-PR]` NVIDIA Spectrum-XGS scale-across.
- `[GPT-OSS-MXFP4]` OpenAI gpt-oss native MXFP4 release.
- `[GPT-OSS-Deploy]` GPT-OSS deployment analysis.
- `[CNCF-genai-platform]` Building cloud-native GenAI platform with OSS.
- `[Equinox]` Holistic Fair Scheduling.
- `[Strata]` Hierarchical Context Caching for Long Context.
- `[AIBrix-v0.4]` P/D Disagg, EP, KVCache v1, KV Event Sync.
- `[STH-CDNA4]` AMD CDNA 4 Hot Chips deep dive.
- `[UEC-Arch-Paper]` Ultra Ethernet's Design Principles.
- `[MoE-Inf-Bench]` MoE-Inference-Bench.

### 2025-09

- `[NVFP4-Pretraining]` NVIDIA NVFP4 pretraining (12B / 10T tokens).
- `[Bridging-MXFP4]` Bridging MXFP4-vs-NVFP4 gap.
- `[DSA-V32]` DeepSeek Sparse Attention.
- `[PreScope]` Prefetching for Resource-Constrained MoE Inference.
- `[Hetero-PD]` Disaggregated PD for Heterogeneous GPUs.
- `[PDTrim]` Targeted Pruning for PD (originally cited as PD-Pruning).
- `[Hetis]` Heterogeneous GPU Clusters with Dynamic Parallelism (SC 2025).
- `[Parallax]` Decentralized LLM Inference Service.
- `[NV-RubinCPX-Blog]` NVIDIA Rubin CPX announcement.
- `[Nova-2025]` Real-Time Agentic VLM Serving.
- `[GuidedDecoding-RAG-2025]` Guided Decoding in RAG.
- `[SGLang-GB200-2]` GB200 NVL72 Part II.
- `[Wide-EP-llmd]` DeepSeek-style MoEs with vLLM and llm-d using Wide EP.
- `[SpecVerify]` Speculative Verification.
- `[SGLang-HiCache]` Fast Hierarchical KV Caching (LMSYS).
- `[Perplexity-TE]` Disaggregated Prefill and Decode at Perplexity.
- `[Furiosa-Server]` NXT RNGD Server GA.
- `[DistributedANN-2025]` Microsoft Bing-scale 50B-vector index.
- `[MLPerf-Whisper-2025]` MLPerf Whisper benchmark.
- `[Outcome-Process-Harmonization]` Beyond Correctness PRM/ORM harmonization.
- `[Asys-Surveys]` Survey of RL for Large Reasoning Models.
- `[Rerankers-Bench-2025]` Top 8 Rerankers benchmark.
- `[VerlTool]` Holistic Agentic RL with Tool Use.
- `[SPEC-RL]` Accelerating On-Policy RL via Speculative Rollouts.
- `[Qwen3-Next-NVIDIA]` Qwen3-Next 80B-A3B announcement.
- `[RollPacker]` Mitigating Long-Tail Rollouts.
- `[TT-BH-Bench]` Tenstorrent Blackhole microbenchmarking.

### 2025-10

- `[ROLL-Flash]` Accelerating RLVR and Agentic Training with Asynchrony.
- `[FP16-RL]` Defeating Train-Inference Mismatch via FP16.
- `[NeuronMM]` Trainium2 matmul.
- `[Pitfalls-KV]` Pitfalls of KV Cache Compression.
- `[ExpertFlow-2025]` Adaptive Expert Scheduling.
- `[Tawa]` Automatic Warp Specialization.
- `[K-Merge]` Online Continual Merging of Adapters.
- `[zFLoRA]` Zero-Latency Fused Low-Rank Adapters (Samsung).
- `[L-MoE]` Gating Network Composes Adapter Parameters.
- `[BatchSpec-Right]` Batch Speculative Decoding Done Right.
- `[FairBatching]` Fairness-Aware Batch Formation.
- `[LMCache]` LMCache tech report.
- `[Reasoning-Serving-Empirical-2025]` RLLM serving empirical study.
- `[NV-Jetson-Thor]` Jetson AGX Thor benchmarks.
- `[KTransformers-LMSYS]` KTransformers LMSYS blog.
- `[Mirror-SD]` Mirror Speculative Decoding (Apple).
- `[NOSA]` Native and Offloadable Sparse Attention.
- `[KTransformers]` KTransformers SOSP 2025.
- `[DVI]` Draft, Verify, & Improve.
- `[LMCache-GKE]` LMCache on Google Kubernetes Engine.
- `[Granite-4]` IBM Granite 4.0 Hybrid Models.
- `[MX+]` Pushing the Limits of Microscaling Formats (MICRO 2025).
- `[vLLM-SleepMode]` Sleep Mode for RL.
- `[LangCopilot-Comparison]` OpenRLHF vs veRL.
- `[NP-NV-Groq]` Why Nvidia paid $20B for Groq.
- `[Twilight]` Hierarchical Top-p Adaptive Sparsity.
- `[Tinker]` Thinking Machines Lab Tinker API.
- `[NSA]` (also publication earlier; serving relevance pivots in 2025-10 ecosystem).
- `[Triton-Anatomy]` Anatomy of a Triton Attention Kernel.
- `[Bitwise-Consistent-RL]` vLLM × TorchTitan bitwise-consistent RL.
- `[Laminar]` Scalable Asynchronous RL Post-Training.

### 2025-11

- `[HipKittens]` AMD MI300 port of TK.
- `[ParallelKittens]` Multi-GPU AI kernels.
- `[Megakernels]` Hazy Research megakernel direction.
- `[MoE-SpeQ]` Speculative Quantized Decoding.
- `[Speculators]` v0.3.0 standardised SD format.
- `[Speculators-RH]` Speculators production-ready integration.
- `[Suffix-Decoding]` SuffixDecoding NeurIPS 2025 Spotlight (publication track).
- `[LoRAServe]` Heterogeneous LoRA Adapters in Distributed LLM Inference.
- `[Reg-TT]` Tenstorrent QuietBox review.
- `[NV-FY26Q3]` NVIDIA Q3 FY26 earnings on Blackwell ramp.
- `[NBF-CS3]` Cerebras CS-3 25 kW summary.
- `[Periodic-Async]` Periodic Asynchrony for On-Policy RL.
- `[Fused-FLoRA]` Fused forward-backward adapters.
- `[LoGo]` Instance-level Dynamic LoRA Selection.
- `[DuetServe]` Harmonizing Prefill and Decode.
- `[DOPD]` Dynamic PD-Disagg for Goodput.
- `[KServe-CNCF]` KServe joins CNCF Incubating.

### 2025-12

- `[DeepSeek-V3.2]` Frontier MoE with DSA.
- `[Janus]` Disaggregating Attention and Experts.
- `[vLLM-WideEP]` DeepSeek @ 2.2k tok/s/H200 with Wide-EP.
- `[Speculators]` v0.3.0 release blog (also referenced).
- `[Four-Over-Six]` More Accurate NVFP4 Quantization.
- `[SparseSpec-2025]` Sparse Self-Speculative Decoding for Reasoning.
- `[TokenScale]` Timely and Accurate Autoscaling with Token Velocity.
- `[TraCT]` CXL-shared-memory KV at Rack-Scale.
- `[A-3PO]` Staleness-aware Proximal Policy Approximation.
- `[Introl-Trn3]` Trainium3 deep-dive.
- `[RollMux]` RollMux phase-multiplexing.
- `[RollArt]` RollArt agentic disaggregated infra.
- `[P-LoRA]` Predictive-LoRA serverless inference system.
- `[CMX]` BlueField-4 Inference Context Memory Storage Platform (CES timing).
- `[NV-CPO-Blog]` NVIDIA Silicon Photonics in production switches.
- `[AutoRound-LLMC]` AutoRound × LLM Compressor.
- `[ASPLOS-Blackwell]` Microbenchmarking Blackwell.
- `[SGLang-ModelOpt]` SGLang × NVIDIA ModelOpt.

### 2026-01

- `[STH-Rubin]` Rubin "in full production" CES 2026.
- `[xAI-Colossus2]` 2 GW Colossus 2 hits 555,000 GPUs.
- `[veRL-0.7]` verl 0.7 release.
- `[MS-Maia200-Blog]` Microsoft Maia 200 announcement.
- `[Toms-MI400]` AMD MI400 / Helios at CES.
- `[Hummingbird]` SLO-Oriented GPU Preemption at Microsecond-scale.
- `[Quartet-II]` NVFP4 pre-training improved unbiased gradient.
- `[NP-VeraRubin]` Vera-Rubin obsoletes current AI iron.
- `[LLM-Compressor-09]` Attention Quantization, MXFP4 Support.

### 2026-02

- `[TK-2.0]` ThunderKittens 2.0 (Blackwell GA).
- `[GPT-OSS-vLLM]` GPT-OSS Performance Optimizations on Blackwell.
- `[llm-d-v0.5]` Sustaining Performance at Scale.
- `[vLLM-DSR1-GB200]` vLLM × GB200 DeepSeek-R1 Part 1.
- `[SPEED-Bench]` SPEED-Bench (NVIDIA).
- `[PrefillShare]` Shared Prefill Module for Multi-LLM Disagg.
- `[LongRoPE2]` Near-Lossless 128K scaling.
- `[NSA]` Native Sparse Attention (publication).
- `[Demystify-CostEff]` Demystifying Cost-Efficiency over Heterogeneous GPUs.
- `[SageServe]` Heterogeneous LLM Workloads at Scale.
- `[LMM-Characterization-2025]` Towards Efficient Large Multimodal Model Serving.
- `[AGoT-2025]` Adaptive Graph of Thoughts.
- `[Sample-Scrutinize-Scale-2025]` Scaling Verification beats scaling generation.
- `[MMTEB-2025]` Massive Multilingual Text Embedding Benchmark.
- `[Cache-Craft-2025]` Chunk-Caches for Efficient RAG.

### 2026-03

- `[FlashAttn-4]` FlashAttention-4 (CuTe-DSL, polynomial exp2).
- `[FA4-RE]` Modal team reverse-engineering of FA4.
- `[FA4-Together]` Together AI FA4 deployment.
- `[FA4-Colfax]` Colfax FA4 commentary.
- `[FA4-FlexAttn]` FlexAttention + FA4.
- `[ICLR-Evo]` Evolution of FlashAttention.
- `[vLLM-Triton]` vLLM Triton Backend Deep Dive.
- `[NV-Groq3LPX]` NVIDIA Groq 3 LPX inside-look.
- `[Decoder-LPX]` GTC 2026 Groq 3 LPX coverage.
- `[NCCL-EP]` Unified Expert Parallel Communication API.
- `[HF-AsyncRL]` HF survey of 16 OSS RL libraries.
- `[Elastic-EP-SGLang]` Elastic EP partial-failure tolerance for DeepSeek MoE.
- `[AdaFuse]` Token-Level Pre-Gating for Adapters.
- `[NVFP4-QAD]` Quantization-Aware Distillation for NVFP4.
- `[Hetero-MM]` Cost-Efficient Multimodal LLM Inference.
- `[ProRL-Agent]` Rollout-as-a-Service for Multi-Turn Agents.
- `[LLM-Compressor-010]` Faster Compression, Distributed GPTQ.
- `[FlashInfer-Bench]` MLSys 2026 FlashInfer Kernel Generation Contest.
- `[Anyscale-vLLM-WideEP]` Ray Serve LLM with Wide-EP and Disaggregated Serving.

### 2026-04

- `[Tessera]` Kernel-Granularity Heterogeneous Disaggregation.
- `[FastHetero]` Fast Heterogeneous Serving.
- `[PrefillAaS]` Prefill-as-a-Service Cross-Datacenter.
- `[KServe-llmd]` Combining KServe and llm-d.
- `[Llama-4-RL]` Llama 4 Multimodal Intelligence (RL pipeline disclosure).
- `[Qwen3.6-MTP]` Qwen3.6-27B with native MTP head.
- `[SambaN-Intel]` GPU/RDU/CPU Heterogeneous Inference Blueprint.

### 2026-05

- `[NeMo-RL-SpecDec]` Speculative Decoding inside NeMo-RL.
- `[Adaptive-Thinking-2026]` See `[Claude-Adaptive-Thinking-2026]` (Anthropic Adaptive Thinking GA).
- `[Qwen3-Next]` (Qwen3-Next reference card and recipes).
- `[Theory-AcceptDyn]` Acceptance Dynamics Across Cognitive Domains.
- ~~`[VSD]` Variational Speculative Decoding~~ — DROPPED 2026-05-05; arXiv ID does not resolve to the claimed paper.

