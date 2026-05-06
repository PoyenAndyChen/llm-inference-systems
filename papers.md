# Bibliography

## How to use this bibliography

This bibliography supports a textbook on LLM inference infrastructure. Entries are grouped topically to mirror the chapter structure (foundations, attention kernels, quantization, KV cache, batching, disaggregation, MoE, long-context, heterogeneous serving, LoRA, cluster systems, hardware, adjacent workloads, RL post-training, and OSS engines). Within each topic, entries are ordered chronologically (oldest first) so the lineage is visible.

Chapters cite into this file by citation key (e.g., `[FlashAttn-3]`). Where a paper has been cited under multiple keys in the literature, a single canonical key is used here and an "also cited as" line lists synonyms. A "Recent developments (2025–2026)" section at the end re-lists entries from the past year in flat chronological order so the reader can scan recent developments in one view.

Date format is YYYY-MM. Venues are listed where known; "preprint" indicates arXiv-only as of May 2026. URLs are direct links to papers, blog posts, or repositories.

## Topical index

### Foundations and roofline

- <a id="roofline-survey"></a>`[Roofline-Survey]` *LLM Inference Unveiled: Survey and Roofline Model Insights*. (multi-author). 2024-02 (rev v5 2025). preprint. https://arxiv.org/html/2402.16363v5
  - Roofline analysis of LLM inference across major GPUs; baseline reference for arithmetic-intensity arguments.
- <a id="scale-book-roofline"></a>`[Scale-Book-Roofline]` *All About Rooflines (How To Scale Your Model)*. JAX scaling book. 2024–25. blog. https://jax-ml.github.io/scaling-book/roofline/
  - Pedagogical roofline reference used widely in serving discussions.
- <a id="scale-book-tpu"></a>`[Scale-Book-TPU]` *How to Think About TPUs*. JAX scaling book. 2024–25. blog. https://jax-ml.github.io/scaling-book/tpus/
  - TPU-side roofline plus pod-topology discussion.

### Attention kernels

#### Lineage

- <a id="flashattn-1"></a>`[FlashAttn-1]` *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. Dao, Fu, Ermon, Rudra, Ré (Stanford). 2022-05. NeurIPS 2022, arXiv:2205.14135. https://arxiv.org/abs/2205.14135
  - Introduced IO-aware tiled attention and the online-softmax recurrence.
- <a id="flashattn-2"></a>`[FlashAttn-2]` *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. Dao (Princeton/Together). 2023-07. ICLR 2024, arXiv:2307.08691. https://arxiv.org/abs/2307.08691
  - Sequence-axis parallelism, reduced non-matmul FLOPs, 50–73% A100 peak.
- <a id="flashdecode"></a>`[FlashDecode]` *Flash-Decoding for long-context inference*. Dao, Haziza, Massa, Sizov (Stanford/Meta/Together). 2023-10. blog. https://crfm.stanford.edu/2023/10/12/flashdecoding.html
  - KV-axis split for batch-1 long-context decode; basis of every modern decode kernel.
- <a id="flashdecode++"></a>`[FlashDecode++]` *FlashDecoding++: Faster Large Language Model Inference on GPUs*. Hong et al. (Tsinghua/Infinigence). 2023-11. MLSys 2024, arXiv:2311.01282. https://arxiv.org/abs/2311.01282
  - Asynchronized softmax with unified max; flat-GEMM for query-length-1 decode.
- <a id="streamingllm"></a>`[StreamingLLM]` *Efficient Streaming Language Models with Attention Sinks*. Xiao, Tian, Chen, Han, Lewis (MIT/Meta/CMU/NVIDIA). 2023-09. ICLR 2024, arXiv:2309.17453. https://arxiv.org/abs/2309.17453
  - Discovered the attention-sink phenomenon; foundation for sink-aware kernels and sliding-window serving.
- <a id="fa-2-hopper"></a>`[FA-2-Hopper]` *A Case Study in CUDA Kernel Fusion: FlashAttention-2 on Hopper using CUTLASS*. Spector, Shah, Cazenavette, Thakkar (Colfax). 2023-12. arXiv:2312.11918. https://arxiv.org/abs/2312.11918
  - Blueprint for TMA + WGMMA + CuTe attention; precursor to FA3.
- <a id="flashattn-3"></a>`[FlashAttn-3]` *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*. Shah, Bikshandi, Zhang, Thakkar, Ramani, Ré, Dao (Colfax/Together/Meta/NVIDIA/Princeton). 2024-07. NeurIPS 2024, arXiv:2407.08608. https://arxiv.org/abs/2407.08608
  - Producer/consumer warp specialization, ping-pong scheduling, FP8 with block-quant.
- <a id="flexattn"></a>`[FlexAttn]` *FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention*. Dong, He, Reizenstein, et al. (PyTorch/Meta). 2024-08. blog → arXiv:2412.05496. https://pytorch.org/blog/flexattention/
  - score_mod / mask_mod Python functions lowered to fused Triton FA kernels.
- <a id="thunderkittens"></a>`[ThunderKittens]` *ThunderKittens: Simple, Fast, and Adorable AI Kernels*. Spector, Arora, Singhal, Fu, Ré (Hazy Research, Stanford). 2024-10. arXiv:2410.20399. https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk
  - Tile-primitive C++ DSL; basis for ThunderMLA, HipKittens, ParallelKittens.
- <a id="liger"></a>`[Liger]` *Liger Kernel: Efficient Triton Kernels for LLM Training*. Hsu et al. (LinkedIn). 2024-10. arXiv:2410.10989. https://arxiv.org/abs/2410.10989
  - Production Triton fused kernels (RMSNorm, RoPE, FLCE); now standard in HF/TRL/Axolotl.
- <a id="flashinfer"></a>`[FlashInfer]` *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving*. Ye, Chen, Lai, Lin et al. (UW/CMU/NVIDIA/OctoAI). 2025-01. MLSys 2025 best paper, arXiv:2501.01005. https://arxiv.org/abs/2501.01005
  - Unifies paged/ragged/radix/tree KV under a Block-Sparse-Row family; integrated into vLLM/SGLang/MLC.

#### Past 12 months SOTA

- <a id="gla-gta"></a>`[GLA-GTA]` *Hardware-Efficient Attention for Fast Decoding*. Zadouri et al. (Princeton). 2025-05. arXiv:2505.21487. https://arxiv.org/abs/2505.21487
  - GTA halves KV vs GQA; GLA kernel ≤2× FlashMLA in spec-decode.
- <a id="sageattn3"></a>`[SageAttn3]` *SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training*. Zhang, Wei et al. (THU/Shengshu). 2025-05. arXiv:2505.11594. https://arxiv.org/abs/2505.11594
  - First plug-and-play NVFP4/MXFP4 attention; 1038 TOPS on RTX 5090.
- <a id="flashmla"></a>`[FlashMLA]` *FlashMLA: Efficient Multi-head Latent Attention Kernels*. DeepSeek. 2025-02. open-source week day 1, https://github.com/deepseek-ai/FlashMLA
  - Canonical MLA decode kernel for Hopper; 3000 GB/s memory-bound on H800.
- <a id="thundermla"></a>`[ThunderMLA]` *ThunderMLA: FlashMLA, Faster and Fused-er!*. Hazy Research. 2025-03. blog. https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla
  - ThunderKittens re-implementation of FlashMLA with on-device scheduler.
- <a id="tk-blackwell"></a>`[TK-Blackwell]` *ThunderKittens Now on Blackwells!*. Hazy Research. 2025-03. blog. https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell
  - BF16 + FP8 GEMM and attention kernels for B200 written in TK; tcgen05 reference.
- <a id="tilelang"></a>`[TileLang]` *TileLang: A Composable Tiled Programming Model for AI Systems*. Wang et al. (PKU/Microsoft). 2025-04. arXiv:2504.17577. https://arxiv.org/abs/2504.17577
  - ~98% FlashMLA performance on H100 in ~70 lines of Python; basis of FlashQLA.
- <a id="cute-dsl"></a>`[CuTe-DSL]` *Achieve CUTLASS C++ Performance with Python APIs Using CuTe DSL*. NVIDIA. 2025. blog. https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/
  - The Python DSL FA4 is built on; FMHA examples for SM100 and SM80.
- <a id="flashinfer-nvidia"></a>`[FlashInfer-NVIDIA]` *Run High-Performance LLM Inference Kernels from NVIDIA Using FlashInfer*. NVIDIA Technical Blog. 2025. https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/
  - NVIDIA shipping TRT-LLM kernels through FlashInfer.
- <a id="xqa"></a>`[XQA]` *New XQA-kernel*. NVIDIA TRT-LLM blog. 2025. https://nvidia.github.io/TensorRT-LLM/blogs/XQA-kernel.html
  - TRT-LLM MQA/GQA-aware decode kernel; 2.4× Llama-70B at fixed latency.
- <a id="fa2-cudnn"></a>`[FA2-cuDNN]` *Accelerating Transformers with NVIDIA cuDNN 9*. NVIDIA. 2024–2025. https://developer.nvidia.com/blog/accelerating-transformers-with-nvidia-cudnn-9/
  - Fused SDPA in cuDNN 9.13.1+; up to 1.2 PFLOPS FP8 on H200; baseline FA4 measures against.
- <a id="vllm-torchcompile"></a>`[vLLM-torchcompile]` *Introduction to torch.compile and How It Works with vLLM*. vLLM blog. 2025-08. https://blog.vllm.ai/2025/08/20/torch-compile.html
  - Piecewise-compile + piecewise-CUDA-graph; attention stays eager.
- <a id="dsa-v32"></a>`[DSA-V32]` *DeepSeek Sparse Attention (DSA) in FlashMLA*. DeepSeek. 2025-09. https://api-docs.deepseek.com/news/news250929
  - Lightning-indexer + top-k drives O(L²)→O(Lk); 640 TFLOPs prefill, 410 TFLOPs decode; day-0 in vLLM and SGLang.
- <a id="triton-anatomy"></a>`[Triton-Anatomy]` *The Anatomy of a Triton Attention Kernel*. Ringlein et al. (IBM Research). 2025-11. arXiv:2511.11581. https://arxiv.org/abs/2511.11581
  - 800-line Triton kernel reaches 105.9% of FA3 on H100 after auto-tuning; basis of vLLM Triton backend.
- <a id="hipkittens"></a>`[HipKittens]` *HipKittens: Fast and Furious AMD Kernels*. Hazy Research. 2025-11. blog. https://hazyresearch.stanford.edu/blog/2025-11-09-hk
  - AMD-MI300 port of TK; AMD answer to FA's Hopper specificity.
- <a id="parallelkittens"></a>`[ParallelKittens]` *Systematic and Practical Simplification of Multi-GPU AI Kernels*. Hazy Research. 2025-11. blog.
  - Collective-aware tile primitives; multi-GPU attention experiments.
- <a id="megakernels"></a>`[Megakernels]` *Loads and Loads of Fluffy Kittens*. Hazy Research. 2025-11. https://hazyresearch.stanford.edu/blog/2025-11-17-fluffy-kittens
  - Megakernel idea (one persistent kernel per request); potential successor to piecewise compilation.
- <a id="tk-2-0"></a>`[TK-2.0]` *ThunderKittens 2.0: Even Faster Kernels for Your GPUs*. Hazy Research. 2026-02. https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2
  - Full Blackwell support; NVFP4/MXFP8 GEMMs at-or-above cuBLAS on B200.
- <a id="gpt-oss-vllm"></a>`[GPT-OSS-vLLM]` *GPT-OSS Performance Optimizations on NVIDIA Blackwell*. vLLM blog. 2026-02. https://blog.vllm.ai/2026/02/01/gpt-oss-optimizations.html
  - Attention-sink + 128-token sliding-window kernel work; hybrid KV allocator.
- <a id="flashattn-4"></a>`[FlashAttn-4]` *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*. Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao (Princeton/Together/Meta/xAI/NVIDIA). 2026-03. arXiv:2603.05451. https://tridao.me/blog/2026/flash4/
  - First FA in CuTe-DSL; software-emulated exp2 polynomial; 1605 TFLOPs/s BF16 on B200.
- <a id="fa4-re"></a>`[FA4-RE]` *We reverse-engineered Flash Attention 4*. Modal team. 2026. blog. https://modal.com/blog/reverse-engineer-flash-attention-4
  - Engineering walkthrough of the FA4 polynomial-exp trick and Cody-Waite range reduction.
- <a id="fa4-together"></a>`[FA4-Together]` *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design*. Together AI blog. 2026-03. https://www.together.ai/blog/flashattention-4
  - Production-engine view of FA4 deployment.
- <a id="fa4-colfax"></a>`[FA4-Colfax]` *FlashAttention-4 Algorithm and Kernel Pipelining Co-Design*. Colfax Research. 2026. https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/
  - Kernel-author commentary on TMEM, tcgen05, and pipeline scheduling.
- <a id="fa4-flexattn"></a>`[FA4-FlexAttn]` *FlexAttention + FlashAttention-4: Fast and Flexible*. PyTorch blog. 2026. https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/
  - ALiBi / sliding-window / score_mod reaching FA4-class speeds via DSL lowering.
- <a id="vllm-triton"></a>`[vLLM-Triton]` *vLLM Triton Attention Backend Deep Dive*. vLLM blog. 2026-03. https://vllm.ai/blog/vllm-triton-backend-deep-dive
  - Persistent kernels, 3D parallel-tiled-softmax decode; default for AMD.
- <a id="iclr-evo"></a>`[ICLR-Evo]` *The Evolution of FlashAttention*. ICLR 2026 blogposts track. 2026. https://iclr-blogposts.github.io/2026/blog/2026/the-evolution-of-flashattention/
  - Peer-reviewed synthesis of FA1→4.

#### Emerging

- <a id="deft"></a>`[DeFT]` *DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference*. ICLR 2025. https://arxiv.org/abs/2404.00242
  - Speculative-decoding-friendly flash-attention for tree-shaped Q with shared KV.
- <a id="kvax"></a>`[Kvax]` *Kvax: Fast and easy-to-use Flash Attention implementation for JAX*. Nebius. 2025.
  - FA for JAX/TPU/GPU.
- <a id="bitdecoding"></a>`[BitDecoding]` *BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache*. 2025-03. arXiv:2503.18773. https://arxiv.org/abs/2503.18773
  - Tensor-core-native low-bit KV decode kernel; complements SageAttn3.
- <a id="flash-d"></a>`[FLASH-D]` *FLASH-D: FlashAttention with Hidden Softmax Division*. 2025-05. arXiv:2505.14201. https://arxiv.org/abs/2505.14201
  - Removes the explicit normalization step.
- <a id="tawa"></a>`[Tawa]` *Tawa: Automatic Warp Specialization for Modern GPUs*. 2025-10. arXiv:2510.14719. https://arxiv.org/abs/2510.14719
  - Compiler-driven warp-spec generation; would automate FA3-style producer/consumer split.
- <a id="flashinfer-bench"></a>`[FlashInfer-Bench]` *MLSys 2026 NVIDIA Track: FlashInfer Kernel Generation Contest*. 2026. https://mlsys26.flashinfer.ai/
  - Institutionalizes AI-agent-written GPU kernels (CuTe-DSL/Triton/TileLang) as a benchmark.
- <a id="sageattn2"></a>`[SageAttn2]` *SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization*. Zhang et al. (THU). 2024-11. ICML 2025, arXiv:2411.10958. https://arxiv.org/abs/2411.10958
  - INT4(QK) + FP8(PV) thread-granularity quant; ~3× FA2 throughput on H100.

### Quantization

#### Lineage — weight-only PTQ

- <a id="gptq"></a>`[GPTQ]` *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. Frantar, Ashkboos, Hoefler, Alistarh (IST Austria/ETH). 2022-10. ICLR 2023, arXiv:2210.17323. https://arxiv.org/abs/2210.17323
  - Approximate-second-order layer-wise weight quantization; ancestor of every W4A16 path.
- <a id="awq"></a>`[AWQ]` *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang, Gan, Han (MIT-Han Lab). 2023-06. MLSys 2024 (Best Paper), arXiv:2306.00978. https://arxiv.org/abs/2306.00978
  - Salient channels by activation magnitude; ~1% protected via per-channel scaling.
- <a id="qlora"></a>`[QLoRA]` *QLoRA: Efficient Finetuning of Quantized LLMs*. Dettmers, Pagnoni, Holtzman, Zettlemoyer. 2023-05. NeurIPS 2023, arXiv:2305.14314. https://arxiv.org/abs/2305.14314
  - 4-bit NormalFloat (NF4); foundation of bitsandbytes 4-bit and 4-bit-base + bf16-adapter pattern.
- <a id="squeezellm"></a>`[SqueezeLLM]` *SqueezeLLM: Dense-and-Sparse Quantization*. Kim et al. (UC Berkeley). 2023-06. ICML 2024, arXiv:2306.07629. https://arxiv.org/abs/2306.07629
  - Sensitivity-based non-uniform quantization plus dense-and-sparse decomposition.
- <a id="omniquant"></a>`[OmniQuant]` *OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models*. OpenGVLab/Shanghai AI Lab. 2023-08. ICLR 2024, arXiv:2308.13137. https://arxiv.org/abs/2308.13137
  - LWC + LET on top of GPTQ; superior at extreme bit-widths.
- <a id="autoround"></a>`[AutoRound]` *AutoRound: Optimize Weight Rounding via Signed Gradient Descent*. Cheng et al. (Intel). 2023-09. arXiv:2309.05516. https://arxiv.org/abs/2309.05516
  - Block-wise signed-gradient descent; integrated into LLM Compressor in 2025.
- <a id="hqq"></a>`[HQQ]` *Half-Quadratic Quantization*. Badri (Mobius Labs). 2023-11. blog. https://mobiusml.github.io/hqq_blog/
  - Calibration-free closed-form weight quantization; 50× faster than GPTQ.
- <a id="gguf"></a>`[GGUF]` *GGUF k-/i-quants*. llama.cpp / GGML. ongoing. https://github.com/ggml-org/llama.cpp
  - k-quants and importance-matrix-aware i-quants; default for CPU/Apple/consumer GPU edge.

#### Lineage — weight + activation PTQ

- <a id="llm-int8"></a>`[LLM.int8]` *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. Dettmers, Lewis, Belkada, Zettlemoyer. 2022-08. NeurIPS 2022, arXiv:2208.07339. https://arxiv.org/abs/2208.07339
  - Mixed-precision decomposition; foundational characterization of emergent outlier features.
- <a id="fp8-formats"></a>`[FP8-Formats]` *FP8 Formats for Deep Learning*. Micikevicius et al. (NVIDIA, ARM, Intel). 2022-09. arXiv:2209.05433. https://arxiv.org/abs/2209.05433
  - Defines E4M3 and E5M2 and the standard hybrid recipe.
- <a id="smoothquant"></a>`[SmoothQuant]` *SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs*. Xiao, Lin, Seznec, Wu, Demouth, Han (MIT-Han Lab/NVIDIA). 2022-11. ICML 2023, arXiv:2211.10438. https://arxiv.org/abs/2211.10438
  - W8A8 PTQ via offline smoothing; recipe reappears in SpinQuant/FlatQuant.
- <a id="fp8-lm"></a>`[FP8-LM]` *FP8-LM: Training FP8 Large Language Models*. Microsoft. 2023-10. arXiv:2310.18313. https://arxiv.org/abs/2310.18313
  - Earliest end-to-end FP8 LLM training framework.
- <a id="atom"></a>`[Atom]` *Atom: Low-bit Quantization for Efficient and Accurate LLM Serving*. Zhao et al. 2023-10. MLSys 2024, arXiv:2310.19102. https://arxiv.org/abs/2310.19102
  - W4A4 with mixed-precision outlier handling; predecessor of QServe.
- <a id="qserve"></a>`[QServe]` *QServe: W4A8KV4 Quantization and System Co-design*. Lin, Tang, Yang, Han (MIT-Han Lab). 2024-05. MLSys 2025, arXiv:2405.04532. https://arxiv.org/abs/2405.04532
  - W4A8KV4 with progressive dequantization; 1.2–3.5× over TRT-LLM.
- <a id="deepseek-v3-fp8"></a>`[DeepSeek-V3-FP8]` *DeepSeek-V3 Technical Report*. DeepSeek-AI. 2024-12. arXiv:2412.19437. https://arxiv.org/abs/2412.19437
  - First very-large-scale (671B MoE) model trained natively in FP8 with fine-grained tile/block-wise scaling.
  - Also cited as: `[MLA-V3]`, `[DeepSeek-V3]`.

#### Lineage — KV cache quantization

- <a id="kivi"></a>`[KIVI]` *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*. Liu, Yuan, Yu et al. 2024-02. ICML 2024, arXiv:2402.02750. https://arxiv.org/abs/2402.02750
  - Per-channel K, per-token V at 2-bit; 2.6× memory, 2.35–3.47× throughput.
- <a id="kvquant"></a>`[KVQuant]` *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization*. Hooper, Kim, Mohtashami et al. (UC Berkeley). 2024-01. NeurIPS 2024, arXiv:2401.18079. https://github.com/SqueezeAILab/KVQuant
  - Pre-RoPE K quant, NUQ codebooks, 3-bit with <0.1 PPL hit; 10M-token-context demos.
- <a id="gear"></a>`[GEAR]` *GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Inference*. Kang et al. (Georgia Tech). 2024-03. COLM 2024, arXiv:2403.05527. https://arxiv.org/abs/2403.05527
  - Q (low-bit base) + L (low-rank residual) + S (sparse outliers); explicit hybrid.
- <a id="kcache"></a>`[KCache]` *Efficient LLM Inference with KCache*. He et al. 2024-04. arXiv:2404.18057. https://arxiv.org/abs/2404.18057
  - Argues some V cache can be dropped entirely; ~40% throughput at minimal quality loss.

#### Lineage — rotation / Hadamard family

- <a id="quarot"></a>`[QuaRot]` *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*. Ashkboos, Mohtashami, Croci et al. (ETH/IST Austria/MSR). 2024-04. NeurIPS 2024, arXiv:2404.00456. https://arxiv.org/abs/2404.00456
  - 4 inserted rotations (R1/R2 offline, R3/R4 online); foundation of rotation lineage.
- <a id="spinquant"></a>`[SpinQuant]` *SpinQuant: LLM Quantization with Learned Rotations*. Liu et al. (Meta). 2024-05. ICLR 2025, arXiv:2405.16406. https://arxiv.org/abs/2405.16406
  - Cayley-optimized learned rotations; +45.1% relative to QuaRot on Llama-3-8B.
- <a id="duquant"></a>`[DuQuant]` *DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs*. Lin, Xu et al. 2024-06. NeurIPS 2024 Oral, arXiv:2406.01721. https://arxiv.org/abs/2406.01721
  - Block-wise rotation + zigzag permutation; SOTA W4A4 at the time.
- <a id="flatquant"></a>`[FlatQuant]` *FlatQuant: Flatness Matters for LLM Quantization*. Liu et al. 2024-10. ICML 2025, arXiv:2410.09426. https://arxiv.org/abs/2410.09426
  - Per-layer Kronecker-factored learnable affine transforms; W4A4 LLaMA-3-70B with <1% drop.

#### Lineage — formats

- <a id="ocp-mx"></a>`[OCP-MX]` *OCP Microscaling Formats (MX) Specification v1.0*. Open Compute Project. 2023-09. https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  - Standardizes block-floating-point: MXFP8, MXFP6, MXFP4, MXINT8.
- <a id="mx-dataformats"></a>`[MX-DataFormats]` *Microscaling Data Formats for Deep Learning*. Rouhani et al. (Microsoft, AMD, Arm, Intel, Meta, NVIDIA, Qualcomm). 2023-10. arXiv:2310.10537. https://arxiv.org/abs/2310.10537
  - Empirical paper underlying the OCP spec; sub-8-bit accuracy parity.
- <a id="nvfp4-inference"></a>`[NVFP4-Inference]` *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference*. NVIDIA Technical Blog. 2025. https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
  - NVFP4 = 16-element micro-blocks with two-level scaling; 3.5× memory vs FP16.

#### Lineage — QAT and 1-bit

- <a id="lsq"></a>`[LSQ]` *Learned Step Size Quantization*. Esser et al. (IBM). 2019. ICLR 2020, arXiv:1902.08153. https://arxiv.org/abs/1902.08153
  - Pre-LLM but foundational; learnable step size with STE.
- <a id="llm-qat"></a>`[LLM-QAT]` *LLM-QAT: Data-Free Quantization Aware Training for LLMs*. Liu et al. (Meta). 2023-05. ACL Findings 2024, arXiv:2305.17888. https://arxiv.org/abs/2305.17888
  - QAT using model self-generations as distillation data.
- <a id="bitnet"></a>`[BitNet]` *BitNet: Scaling 1-bit Transformers for Large Language Models*. Wang, Ma, Dong et al. (Microsoft). 2023-10. arXiv:2310.11453. https://arxiv.org/abs/2310.11453
  - Replaces nn.Linear with BitLinear; native binary training.
- <a id="bitnet-1-58"></a>`[BitNet-1.58]` *The Era of 1-bit LLMs: All LLMs are in 1.58 Bits*. Ma, Wang, Ma et al. (Microsoft). 2024-02. arXiv:2402.17764. https://arxiv.org/abs/2402.17764
  - Ternary weights via absmean; matches FP at 3B+ on perplexity.
- <a id="efficientqat"></a>`[EfficientQAT]` *EfficientQAT: Efficient Quantization-Aware Training for LLMs*. 2024-07. ACL 2025, arXiv:2407.11062. https://arxiv.org/abs/2407.11062
  - Two-phase block-wise then end-to-end QAT.
- <a id="bitnet-a4-8"></a>`[BitNet-a4.8]` *BitNet a4.8: 4-bit Activations for 1-bit LLMs*. Microsoft Research. 2024-11. https://www.microsoft.com/en-us/research/publication/bitnet-a4-8-4-bit-activations-for-1-bit-llms/
  - Hybrid 4-bit-act / 8-bit-elsewhere with intermediate-state sparsification.
- <a id="bitnet-cpp"></a>`[bitnet.cpp]` *BitNet (inference framework)*. Microsoft. 2024–25. https://github.com/microsoft/BitNet
  - Official 1-bit inference framework; CPU-first kernels.

#### Past 12 months SOTA

- <a id="bitnet-2b-4t"></a>`[BitNet-2B-4T]` *BitNet b1.58 2B4T Technical Report*. Ma, Wang, Huang et al. (Microsoft). 2025-04. arXiv:2504.12285. https://arxiv.org/abs/2504.12285
  - First open-source native 1-bit LLM at 2B scale on 4T tokens.
- <a id="turboquant"></a>`[TurboQuant]` *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. Zandieh, Daliri, Hadian, Mirrokni (Google Research/NYU/DeepMind). 2025-04. ICLR 2026, arXiv:2504.19874. https://arxiv.org/abs/2504.19874
  - Random rotation + 1-bit residual VQ; 3.5-bit KV at parity, 2.5-bit marginal degradation.
- <a id="gbs-dimexpansion"></a>`[GBS-DimExpansion]` *Gradual Binary Search and Dimension Expansion: A general method for activation quantization in LLMs*. Maisonnave et al. (Inria/CEA). 2025-04. arXiv:2504.13989. https://arxiv.org/abs/2504.13989
  - Hadamard + Paley-extension to non-power-of-2 dims; pushes to 3-bit W/A/KV.
- <a id="million"></a>`[MILLION]` *MILLION: Mastering Long-Context LLM Inference Via Outlier-Immunized KV Product Quantization*. 2025-04. arXiv:2504.03661. https://arxiv.org/abs/2504.03661
  - KV product quantization at 4-bit with 2.09× e2e gain at 32K context.
- <a id="quartet"></a>`[Quartet]` *Quartet: Native FP4 Training Can Be Optimal for Large Language Models*. Castro, Panferov, Tabesh, Sieberling, Chen, Nikdan, Ashkboos, Alistarh (IST Austria/Red Hat AI/ETH). 2025-05. NeurIPS 2025, arXiv:2505.14669. https://arxiv.org/abs/2505.14669
  - End-to-end FP4 training using Random Hadamard + 2D quantization + stochastic rounding.
- <a id="oaken"></a>`[Oaken]` *Oaken: Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization*. 2025. ISCA 2025. https://dl.acm.org/doi/10.1145/3695053.3731019
  - Algo/HW co-design with offline outlier thresholds + online scale.
- <a id="base-q"></a>`[BASE-Q]` *BASE-Q: Bias and Asymmetric Scaling Enhanced Rotational Quantization*. 2025-06. arXiv:2506.15689. https://arxiv.org/abs/2506.15689
  - Bias correction and asymmetric scaling on top of rotation.
- <a id="antkv"></a>`[AnTKV]` *AnTKV: Anchor Token-Aware Sub-Bit Vector Quantization for KV Cache*. 2025-06. arXiv:2506.19505. https://arxiv.org/abs/2506.19505
  - 1-bit Mistral-7B PPL 6.32 vs KVQuant 15.36; 840K-token context on one A100.
- <a id="osp"></a>`[OSP]` *Outlier-Safe Pre-Training*. 2025-06. arXiv:2506.19697. https://arxiv.org/abs/2506.19697
  - Muon + Single-Scale RMSNorm trains 1.4B / 1T tokens without emergent outliers.
- <a id="mxfp8-recipes"></a>`[MXFP8-Recipes]` *Recipes for Pre-training LLMs with MXFP8*. 2025-06. arXiv:2506.08027. https://arxiv.org/abs/2506.08027
  - Round-to-infinity scale-factor rounding required for stable MXFP8 pretraining.
- <a id="micromix"></a>`[MicroMix]` *MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats*. 2025-08. arXiv:2508.02343. https://arxiv.org/abs/2508.02343
  - Mixes MXFP4/6/8 per channel; demonstrated on RTX 5090 / 5070Ti.
- <a id="gpt-oss-mxfp4"></a>`[GPT-OSS-MXFP4]` *Introducing gpt-oss*. OpenAI. 2025-08. https://openai.com/index/introducing-gpt-oss/
  - First major open-weight model shipped natively in MXFP4; gpt-oss-120B in 80GB.
- <a id="nvfp4-pretraining"></a>`[NVFP4-Pretraining]` *Pretraining Large Language Models with NVFP4*. NVIDIA. 2025-09. arXiv:2509.25149. https://arxiv.org/abs/2509.25149
  - 12B / 10T tokens; NVFP4 matches FP8 baseline.
- <a id="bridging-mxfp4"></a>`[Bridging-MXFP4]` *Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization*. 2025-09. arXiv:2509.23202. https://arxiv.org/abs/2509.23202
  - OAS + MBS closes MXFP4-vs-NVFP4 gap from ~10% to <1%.
- <a id="mx+"></a>`[MX+]` *MX+: Pushing the Limits of Microscaling Formats for Efficient LLM Serving*. MICRO 2025. arXiv:2510.14557. https://arxiv.org/abs/2510.14557
  - Repurposes outlier exponent field as extended mantissa; +42.15% accuracy over MXFP4.
- <a id="mx-plus-emulation"></a>`[MX-Plus-emulation]` Same as MX+, software emulation via CUTLASS/Triton.
- <a id="four-over-six"></a>`[Four-Over-Six]` *Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling*. 2025-12. arXiv:2512.02010. https://arxiv.org/abs/2512.02010
  - Adaptive (4 vs 6 bits) per-block scale-bit-width.
- <a id="quartet-ii"></a>`[Quartet-II]` *Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation*. IST-DASLab. 2026-01. arXiv:2601.22813
  - MS-EDEN unbiased micro-scale rounding (>2× lower quant error).
- <a id="nvfp4-qad"></a>`[NVFP4-QAD]` *Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery*. NVIDIA. 2026-03. https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf
  - Distillation recipe for NVFP4 inference checkpoints.
- <a id="gpt-oss-qat"></a>`[GPT-OSS-QAT]` *Fine-Tuning gpt-oss for Accuracy and Performance with QAT*. NVIDIA Tech Blog. 2025. https://developer.nvidia.com/blog/fine-tuning-gpt-oss-for-accuracy-and-performance-with-quantization-aware-training/
  - QAT recipe for OpenAI GPT-OSS on top of native MXFP4.
- <a id="marlin"></a>`[Marlin]` *Marlin: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models*. 2024-08. arXiv:2408.11743. https://arxiv.org/pdf/2408.11743
  - The Marlin FP16/INT4 mixed-precision GEMM kernel.
- <a id="marlin-rh"></a>`[Marlin-RH]` *How Marlin Pushes the Boundaries of Mixed-Precision LLM Inference*. Red Hat blog. 2024. https://developers.redhat.com/articles/2024/04/17/how-marlin-pushes-boundaries-mixed-precision-llm-inference
  - Production-context blog on Marlin in vLLM.
- <a id="llm-compressor"></a>`[LLM-Compressor]` *LLM Compressor*. Red Hat / vLLM. 2024-2026. https://github.com/vllm-project/llm-compressor
  - De-facto open compression toolchain for vLLM-targeted deployments.
- <a id="llm-compressor-09"></a>`[LLM-Compressor-09]` *LLM Compressor 0.9: Attention Quantization, MXFP4 Support, and More*. Red Hat. 2026-01. https://developers.redhat.com/articles/2026/01/16/llm-compressor-090-attention-quantization-mxfp4-support-and-more
- <a id="llm-compressor-010"></a>`[LLM-Compressor-010]` *LLM Compressor 0.10: Faster Compression, Distributed GPTQ*. Red Hat. 2026-03. https://developers.redhat.com/articles/2026/03/18/llm-compressor-010-faster-compression-distributed-gptq
- <a id="autoround-llmc"></a>`[AutoRound-LLMC]` *Advancing Low-Bit Quantization for LLMs: AutoRound × LLM Compressor*. Red Hat. 2025-12. https://developers.redhat.com/articles/2025/12/09/advancing-low-bit-quantization-llms-autoround-x-llm-compressor
- <a id="gptqmodel"></a>`[GPTQModel]` *GPTQModel*. modelcloud. ongoing. https://github.com/modelcloud/gptqmodel
  - Successor maintenance of GPTQ; integrates with vLLM/TRT-LLM.
- <a id="nvidia-modelopt"></a>`[NVIDIA-ModelOpt]` *NVIDIA Model Optimizer*. NVIDIA. ongoing. https://github.com/NVIDIA/Model-Optimizer
  - NVFP4/FP8 quantization toolchain; integrates with TRT-LLM/SGLang/vLLM.
- <a id="sglang-modelopt"></a>`[SGLang-ModelOpt]` *SGLang × NVIDIA ModelOpt Integration*. LMSYS. 2025-12. https://www.lmsys.org/blog/2025-12-02-modelopt-quantization/
- <a id="sglang-gpt-oss-mxfp4"></a>`[SGLang-GPT-OSS-MXFP4]` *SGLang GPT-OSS MXFP4 / QAT*. LMSYS. 2025-08. https://www.lmsys.org/blog/2025-08-28-gpt-oss-qat/
- <a id="vllm-fp8kv"></a>`[vLLM-FP8KV]` *The State of FP8 KV-Cache and Attention Quantization in vLLM*. vLLM project. 2024–25. https://vllm.ai/blog/fp8-kvcache
- <a id="trt-llm-kvq"></a>`[TRT-LLM-KVQ]` *TensorRT-LLM KV Cache Quantization*. NVIDIA. 2024–25. https://nvidia.github.io/TensorRT-LLM/features/kvcache.html
- <a id="amd-mxfp4"></a>`[AMD-MXFP4]` *AMD MXFP4/MXFP6 Quantization on ROCm*. AMD. 2025. https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html
- <a id="amd-mi325-mlperf"></a>`[AMD-MI325-MLPerf]` *MI325X Accelerates MLPerf Inference*. AMD. 2025. https://rocm.blogs.amd.com/artificial-intelligence/mi325x-accelerates-mlperf-inference/README.html
- <a id="vllm-dsr1-gb200"></a>`[vLLM-DSR1-GB200]` *vLLM × GB200 / DeepSeek-R1*. vLLM. 2026-02. https://blog.vllm.ai/2026/02/03/dsr1-gb200-part1.html

### Speculative decoding and multi-token prediction

#### Lineage

- <a id="specdec-leviathan"></a>`[SpecDec-Leviathan]` *Fast Inference from Transformers via Speculative Decoding*. Leviathan, Kalman, Matias (Google). 2022-11. ICML 2023, arXiv:2211.17192. https://arxiv.org/abs/2211.17192
  - Original modified-rejection-sampling speculative decoding; 2–3× T5-XXL.
- <a id="specsamp-chen"></a>`[SpecSamp-Chen]` *Accelerating Large Language Model Decoding with Speculative Sampling*. Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper (DeepMind). 2023-02. arXiv:2302.01318. https://arxiv.org/abs/2302.01318
  - Independently re-derived; 2–2.5× distributed on Chinchilla-70B.
- <a id="specinfer"></a>`[SpecInfer]` *SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification*. Miao, Oliaro et al. (CMU/FlexFlow). 2023-05. ASPLOS 2024, arXiv:2305.09781. https://arxiv.org/abs/2305.09781
  - Token-tree verification with tree attention; 1.5–3.5× speedup.
- <a id="osd"></a>`[OSD]` *Online Speculative Decoding*. Liu et al. (UC Berkeley). 2023-10. ICML 2024, arXiv:2310.07177. https://arxiv.org/abs/2310.07177
  - Continually retrains drafter on observed traffic; +0.1–0.65 acceptance.
- <a id="rest"></a>`[REST]` *REST: Retrieval-Based Speculative Decoding*. He, Zhong, Cai, Lee, He. 2023-11. NAACL 2024, arXiv:2311.08252. https://arxiv.org/abs/2311.08252
  - External datastore retrieval as drafter; 1.62–2.36×; plug-and-play.
- <a id="pld"></a>`[PLD]` *Prompt Lookup Decoding*. Saxena. 2023. https://github.com/apoorvumang/prompt-lookup-decoding
  - N-gram match against prompt history; ~2.4× on summarization/QA.
- <a id="medusa"></a>`[Medusa]` *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*. Cai et al. (Princeton/Together/UIUC). 2024-01. ICML 2024, arXiv:2401.10774. https://arxiv.org/abs/2401.10774
  - Parallel sequentially-independent draft heads; 2.2–3.6× with tree attention.
- <a id="eagle-1"></a>`[EAGLE-1]` *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*. Li, Wei, Zhang, Zhang. 2024-01. ICML 2024, arXiv:2401.15077. https://arxiv.org/abs/2401.15077
  - Feature-level draft with one-token shift; 2.7–3.5× Llama-2-Chat-70B.
- <a id="lookahead-decoding"></a>`[Lookahead-Decoding]` *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding*. Fu, Bailis, Stoica, Zhang (Hao AI Lab). 2024-02. ICML 2024, arXiv:2402.02057. https://arxiv.org/abs/2402.02057
  - Jacobi-iteration parallel n-gram extraction without a draft model.
- <a id="glide-cape"></a>`[GliDe-CaPE]` *GliDe with a CaPE: A Low-Hassle Method to Accelerate Speculative Decoding*. Du et al. 2024-02. ICML 2024, arXiv:2402.02082. https://arxiv.org/abs/2402.02082
  - Draft model reuses target's KV cache; 2.17–2.61× Vicuna.
- <a id="hydra"></a>`[Hydra]` *Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding*. Ankner, Parthasarathy, Nrusimha, Rinard, Ragan-Kelley, Brandon (MIT). 2024-02. COLM 2024, arXiv:2402.05109. https://arxiv.org/abs/2402.05109
  - 1.31× over Medusa, 2.7× autoregressive.
- <a id="specstream-apple"></a>`[SpecStream-Apple]` *Speculative Streaming: Fast LLM Inference Without Auxiliary Models*. Apple. 2024-02. NeurIPS 2024 ENLSP, arXiv:2402.11131. https://arxiv.org/abs/2402.11131
  - Multi-stream attention; 1.8–3.1× with ~10000× fewer extra params than Medusa.
- <a id="sequoia"></a>`[Sequoia]` *Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding*. Chen, May, Svirschevski, Huang, Ryabinin, Jia, Chen (CMU/Together). 2024-02. arXiv:2402.12374. https://arxiv.org/abs/2402.12374
  - DP-based optimal tree-shape selection; 4× on Llama-2-7B.
- <a id="redrafter"></a>`[ReDrafter]` *Recurrent Drafter for Fast Speculative Decoding in Large Language Models*. Cheng, Zhang et al. (Apple). 2024-03. arXiv:2403.09919. https://arxiv.org/abs/2403.09919
  - RNN drafter + dynamic tree attention + KD; 2.8× H100; in TensorRT-LLM.
- <a id="spec-bench"></a>`[Spec-Bench]` *Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding*. Xia et al. 2024-01. ACL Findings 2024, arXiv:2401.07851. https://arxiv.org/abs/2401.07851
  - Standard 6-task benchmark; basis for per-task numbers.
- <a id="bass"></a>`[BASS]` *BASS: Batched Attention-optimized Speculative Sampling*. Qian et al. (AWS). 2024-04. ACL Findings 2024, arXiv:2404.15778. https://arxiv.org/abs/2404.15778
  - First systematic batched-SD; 2.15× at batch 8 on A100.
- <a id="better-faster-mtp"></a>`[Better-Faster-MTP]` *Better & Faster Large Language Models via Multi-token Prediction*. Gloeckle, Idrissi, Rozière, Lopez-Paz, Synnaeve (Meta FAIR). 2024-04. ICML 2024, arXiv:2404.19737. https://arxiv.org/abs/2404.19737
  - Pre-training auxiliary loss with n parallel heads; intellectual antecedent of DeepSeek MTP.
- <a id="kangaroo"></a>`[Kangaroo]` *Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting*. Liu, Tang et al. (Huawei). 2024-04. NeurIPS 2024, arXiv:2404.18911. https://arxiv.org/abs/2404.18911
  - Shallow target sub-network as drafter; 2.04× walltime.
- <a id="smartspec"></a>`[SmartSpec]` *Optimizing Speculative Decoding for Serving LLMs Using Goodput*. Liu et al. (UC Berkeley/UCSD/Anyscale). 2024-06. arXiv:2406.14066. https://arxiv.org/abs/2406.14066
  - Goodput metric; dynamically modulates γ per request; renamed TurboSpec.
  - Also cited as: `[TurboSpec]`.
- <a id="eagle-2"></a>`[EAGLE-2]` *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees*. Same authors as EAGLE-1. 2024-06. EMNLP 2024, arXiv:2406.16858. https://arxiv.org/abs/2406.16858
  - Confidence-based dynamic tree expansion.
- <a id="token-recycling"></a>`[Token-Recycling]` *Turning Trash into Treasure: Accelerating Inference of LLMs with Token Recycling*. Luo et al. 2024-08. ACL 2025, arXiv:2408.08696. https://arxiv.org/abs/2408.08696
  - BFS over adjacency matrix of prior top-k; ~2× training-free, <2 MB extra memory.
- <a id="hass"></a>`[HASS]` *Learning Harmonized Representations for Speculative Sampling*. 2024-08. ICLR 2025, arXiv:2408.15766. https://arxiv.org/abs/2408.15766
  - Distillation + context-aligned training; +8–16% acceptance over EAGLE-2.
- <a id="magicdec"></a>`[MagicDec]` *MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding*. Chen et al. (CMU/Princeton). 2024-08. ICLR 2025, arXiv:2408.11049. https://arxiv.org/abs/2408.11049
  - Long-context KV-bound regime; 2.51× Llama-3.1-8B at batch 32–256.
- <a id="suffix-decoding"></a>`[Suffix-Decoding]` *SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications*. Snowflake/CMU. 2024-11. NeurIPS 2025 Spotlight, arXiv:2411.04975. https://arxiv.org/abs/2411.04975
  - Suffix tree over generated history; 5.3× for agentic workloads; in vLLM/Arctic Inference.
- <a id="theory-neurips24"></a>`[Theory-NeurIPS24]` *A Theoretical Perspective for Speculative Decoding Algorithm*. Yin et al. NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/e7349e785900b93d8b4971a3f2c1cefe-Paper-Conference.pdf
  - Formal expected-acceptance bounds.
- <a id="speckd"></a>`[SpecKD]` *Speculative Knowledge Distillation*. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/a2747a3844ca1e4667fbff3f558eb39b-Paper-Conference.pdf
  - Distillation algorithm tailored to the SD setting.
- <a id="survey-2024"></a>`[Survey-2024]` *A Comprehensive Survey of Speculative Decoding*. Xia et al. ACL Findings 2024. https://aclanthology.org/2024.findings-acl.456.pdf
- <a id="tutorial-2025"></a>`[Tutorial-2025]` *Tutorial Proposal: Speculative Decoding for Efficient LLM Inference*. 2025-03. arXiv:2503.00491. https://arxiv.org/abs/2503.00491

#### Past 12 months SOTA

- <a id="deepseek-v3-mtp"></a>`[DeepSeek-V3-MTP]` See `[DeepSeek-V3-FP8]` above. MTP1 acceptance >80%; ~1.8× generation TPS.
- <a id="jakiro"></a>`[Jakiro]` *Jakiro: Boosting Speculative Decoding with Decoupled Multi-Head via MoE*. 2025-02. arXiv:2502.06282. https://arxiv.org/abs/2502.06282
  - Drafter itself is MoE-decoupled.
- <a id="survey-2025"></a>`[Survey-2025]` *Speculative Decoding and Beyond: An In-Depth Survey of Techniques*. Hu et al. 2025-02. arXiv:2502.19732. https://arxiv.org/abs/2502.19732
  - Most current taxonomy.
- <a id="eagle-3"></a>`[EAGLE-3]` *EAGLE-3: Scaling Up Inference Acceleration of LLMs via Training-Time Test*. Li, Wei, Zhang, Zhang. 2025-03. NeurIPS 2025, arXiv:2503.01840. https://arxiv.org/abs/2503.01840
  - Up to 6.5× batch-1; 1.4× over EAGLE-2; default in vLLM/SGLang/TRT-LLM.
- <a id="pard"></a>`[PARD]` *PARD: Accelerating LLM Inference with Low-Cost PARallel Draft Model Adaptation*. An et al. (AMD). 2025-04. arXiv:2504.18583. https://arxiv.org/abs/2504.18583
  - Target-independent parallel drafters; 3.67× Llama-3.1-8B; 1.15× over EAGLE-3.
- <a id="theory-scaling"></a>`[Theory-Scaling]` *Scaling Laws for Speculative Decoding*. 2025-05. arXiv:2505.07858. https://arxiv.org/abs/2505.07858
  - Log-linear acceptance scaling with drafter capacity, pretraining tokens, batch size.
- <a id="lookahead-reasoning"></a>`[Lookahead-Reasoning]` *Scaling Speculative Decoding with Lookahead Reasoning*. Fu, Ge et al. (UCSD). 2025-06. NeurIPS 2025, arXiv:2506.19830. https://arxiv.org/abs/2506.19830
  - Step-level speculation for reasoning models; SD speedup 1.4×→2.1× on R1-style.
- <a id="moe-cascade"></a>`[MoE-Cascade]` *Utility-Driven Speculative Decoding for Mixture-of-Experts*. 2025-06. arXiv:2506.20675. https://arxiv.org/abs/2506.20675
  - Selectively enables/tunes γ to bound MoE slowdown to <5%.
- <a id="specforge"></a>`[SpecForge]` *SpecForge: Accelerating Speculative Decoding Training for SGLang*. LMSYS. 2025-07. https://www.lmsys.org/blog/2025-07-25-spec-forge/
  - Open-source EAGLE-3 training framework; 2.0–2.18× MT-Bench on Llama-4 Scout/Maverick.
- <a id="llama-spec-meta"></a>`[Llama-Spec-Meta]` *Efficient Speculative Decoding for Llama at Scale: Challenges and Solutions*. Meta. 2025-08. arXiv:2508.08192. https://arxiv.org/abs/2508.08192
  - Production EAGLE for Llama-3.3-70B and Llama-4 Maverick; 4 ms/token batch-1.
- <a id="specmoeoff"></a>`[SpecMoEOff]` *Accelerating MoE Inference by Hiding Offloading Latency with Speculative Decoding*. 2025-08. arXiv:2508.21706. https://arxiv.org/abs/2508.21706
  - SD enlarges per-expert workload while CPU↔GPU streams.
- <a id="specverify"></a>`[SpecVerify]` *Speculative Verification: Exploiting Information Gain for Speculative Decoding*. 2025-09. arXiv:2509.24328. https://arxiv.org/html/2509.24328v2
  - Information-gain-based verification scheduling.
- <a id="mirror-sd"></a>`[Mirror-SD]` *Mirror Speculative Decoding: Breaking the Serial Barrier in LLM Inference*. Apple. 2025-10 (rev 2026-01). arXiv:2510.13161. https://arxiv.org/abs/2510.13161
  - Bidirectional drafter↔target speculation; 2.8–5.8× SpecBench.
- <a id="dvi"></a>`[DVI]` *Draft, Verify, & Improve: Toward Training-Aware Speculative Decoding*. 2025-10. arXiv:2510.05421. https://arxiv.org/abs/2510.05421
  - On-the-fly drafter LoRA from accept/reject feedback.
- <a id="batchspec-right"></a>`[BatchSpec-Right]` *Batch Speculative Decoding Done Right*. 2025-10. arXiv:2510.22876. https://arxiv.org/abs/2510.22876
  - Re-examines correctness pitfalls of naive batched SD.
- <a id="moe-speq"></a>`[MoE-SpeQ]` *MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching*. 2025-11. arXiv:2511.14102. https://arxiv.org/abs/2511.14102
  - On-device draft predicts future expert sequence to drive prefetch.
- <a id="speculators"></a>`[Speculators]` *Speculators v0.3.0*. Red Hat / vLLM. 2025-11 → 2025-12. https://blog.vllm.ai/2025/12/13/speculators-v030.html
  - Standardised HF format for spec-dec drafters; one-button training/serving.
- <a id="speculators-rh"></a>`[Speculators-RH]` *Speculators: Standardized Production-Ready Speculative Decoding*. Red Hat. 2025-11. https://developers.redhat.com/articles/2025/11/19/speculators-standardized-production-ready-speculative-decoding
- <a id="speed-bench"></a>`[SPEED-Bench]` *SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding*. NVIDIA. 2026-02. arXiv:2604.09557. https://arxiv.org/abs/2604.09557
- <a id="theory-acceptdyn"></a>`[Theory-AcceptDyn]` *Acceptance Dynamics Across Cognitive Domains in Speculative Decoding*. 2026. arXiv:2604.14682. https://arxiv.org/abs/2604.14682
- <a id="google-sd-retro"></a>`[Google-SD-Retro]` *Looking Back at Speculative Decoding*. Google Research blog. 2024. https://research.google/blog/looking-back-at-speculative-decoding/
- <a id="p-eagle"></a>`[P-EAGLE]` *P-EAGLE: Faster LLM Inference with Parallel Speculative Decoding in vLLM*. AWS. 2025–2026. https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/
  - Pipelines drafter and verifier; 1.69× over EAGLE-3 on B200; in vLLM ≥0.16.
- <a id="suffixdec-arctic"></a>`[SuffixDec-Arctic]` *SuffixDecoding at Production Scale with Arctic Inference and vLLM*. Snowflake. 2025. https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/
- <a id="moe-spec-cohere"></a>`[MoE-Spec-Cohere]` *Why MoE Models Get More From Speculative Decoding*. Cohere blog. 2025. https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding
  - Bandwidth-bound sweet-spot argument; idle compute amortizes weight movement.
- <a id="speculative-diffusion"></a>`[Speculative-Diffusion]` *Speculative Diffusion Decoding*. NAACL 2025. https://aclanthology.org/2025.naacl-long.601/
- <a id="specextend"></a>`[SpecExtend]` *SpecExtend: Drop-in Enhancement for Long-Sequence SD*. 2025. arXiv:2505.20776. https://arxiv.org/html/2505.20776
- <a id="qwen3-next"></a>`[Qwen3-Next]` *Qwen3-Next-80B-A3B*. Alibaba Qwen. 2025. https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
  - Hybrid Gated DeltaNet + sparse MoE; ships native MTP head as drafter.
- <a id="qwen3-6-mtp"></a>`[Qwen3.6-MTP]` *Qwen3.6-27B*. Alibaba Qwen. 2026-04. https://huggingface.co/
  - Open-weight dense model with built-in MTP head.
- <a id="kimi-k2-5-eagle"></a>`[Kimi-K2.5-EAGLE]` *How We Built the Fastest Kimi K2.5 on Artificial Analysis*. Baseten. 2025. https://www.baseten.co/blog/how-we-built-the-fastest-kimi-k2-5-on-artificial-analysis/
  - Kimi K2.5 with custom-trained ~1B EAGLE-3 speculator + INT4→NVFP4; 340+ tok/s.
- <a id="gemma4-mtp"></a>`[Gemma4-MTP]` *Accelerating Gemma 4: Faster Inference with Multi-Token Prediction Drafters*. Google. 2026. https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/

### KV cache (memory mgmt, compression, tiered offload)

#### Lineage — memory management

- <a id="pagedattention"></a>`[PagedAttention]` *Efficient Memory Management for Large Language Model Serving with PagedAttention*. Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica (UC Berkeley/Stanford/UCSD). 2023-09. SOSP 2023 (best paper), arXiv:2309.06180. https://arxiv.org/abs/2309.06180
  - Block-structured KV; the OS-paging analogy that became vLLM.
  - Also cited as: `[vLLM-SOSP23]`.
- <a id="vattention"></a>`[vAttention]` *vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention*. Prabhu et al. (MSR India). 2024-05. ASPLOS 2025, arXiv:2405.04437. https://arxiv.org/abs/2405.04437
  - Contiguous virtual address space + CUDA VMM; up to 1.97× over vLLM.
- <a id="vllm-v1-prefix"></a>`[vLLM-V1-prefix]` *Automatic Prefix Caching design (V1)*. vLLM project. 2024–25. https://docs.vllm.ai/en/stable/design/prefix_caching/
  - Hash-chained block prefix cache; default-on in V1.

#### Lineage — prefix / prompt caching

- <a id="prompt-cache"></a>`[Prompt-Cache]` *Prompt Cache: Modular Attention Reuse for Low-Latency Inference*. Gim et al. (Yale/Google). 2023-11. MLSys 2024, arXiv:2311.04934. https://arxiv.org/abs/2311.04934
  - Modular reuse of non-prefix schema fragments; 8–60× TTFT.
- <a id="radixattention"></a>`[RadixAttention]` *SGLang: Efficient Execution of Structured Language Model Programs*. Zheng et al. (LMSYS/UCB/Stanford). 2023-12. NeurIPS 2024, arXiv:2312.07104. https://arxiv.org/abs/2312.07104
  - Token-level radix tree of KV with LRU eviction; default in SGLang.
- <a id="hydragen"></a>`[Hydragen]` *Hydragen: High-Throughput LLM Inference with Shared Prefixes*. Juravsky et al. (Stanford). 2024-02. ICML 2024, arXiv:2402.05099. https://arxiv.org/abs/2402.05099
  - Decompose into shared-prefix and unique-suffix; 32× over baselines for long shared prefix.
- <a id="chunkattention"></a>`[ChunkAttention]` *ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache*. Ye et al. (Microsoft). 2024-02. ACL 2024, arXiv:2402.15220. https://arxiv.org/abs/2402.15220
  - Prefix-tree of KV chunks + two-phase partitioned kernel.
- <a id="cachedattention"></a>`[CachedAttention]` *Cost-Efficient LLM Serving for Multi-turn Conversations*. Gao et al. (Bytedance). 2024-03. USENIX ATC 2024. https://www.usenix.org/conference/atc24/presentation/gao-bin-cost
  - Save inactive KV to cheaper tiers; pre-cursor to LMCache/Mooncake session caches.
- <a id="cacheblend"></a>`[CacheBlend]` *CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion*. Yao et al. (U. Chicago). 2024-05. EuroSys 2025, arXiv:2405.16444. https://arxiv.org/abs/2405.16444
  - Reuse precomputed KVs for non-prefix RAG chunks; 2.2–3.3× TTFT; in LMCache.
- <a id="cachegen"></a>`[CacheGen]` *CacheGen: KV Cache Compression and Streaming for Fast LLM Serving*. Liu et al. (U. Chicago). 2023-10. SIGCOMM 2024, arXiv:2310.07240. https://arxiv.org/abs/2310.07240
  - Per-layer-sensitive KV quant + arithmetic coding; 3.5–4.3× cache-size reduction.
- <a id="anthropic-pc"></a>`[Anthropic-PC]` *Prompt caching (API documentation)*. Anthropic. 2024-08+. https://docs.claude.com/en/docs/build-with-claude/prompt-caching
  - Production prompt-caching API contract: 5-min default, 1-hour extended TTL.

#### Lineage — eviction / sparsity

- <a id="h2o"></a>`[H2O]` *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs*. Zhang et al. 2023-06. NeurIPS 2023, arXiv:2306.14048. https://arxiv.org/abs/2306.14048
  - Recent + cumulative-attention heavy hitter eviction; eviction lineage starts here.
- <a id="scissorhands"></a>`[Scissorhands]` *Scissorhands: Exploiting the Persistence of Importance Hypothesis*. Liu et al. (Stevens/UMD). 2023-05. NeurIPS 2023, arXiv:2305.17118. https://arxiv.org/abs/2305.17118
  - 5× cache reduction; pivotal-token persistence.
- <a id="fastgen"></a>`[FastGen]` *Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs*. Ge et al. (UIUC/Microsoft). 2023-10. ICLR 2024 oral, arXiv:2310.01801. https://arxiv.org/abs/2310.01801
  - Per-head adaptive policy mix; precursor to DuoAttention.
- <a id="dejavu"></a>`[DejaVu]` *Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time*. Liu et al. (Rice/Stanford). 2023-10. ICML 2023, arXiv:2310.17157. https://arxiv.org/abs/2310.17157
  - Predict per-input head/MLP sparsity; basis for PowerInfer-style hot/cold.
- <a id="alisa"></a>`[ALISA]` *ALISA: Accelerating LLM Inference via Sparsity-Aware KV Caching*. Zhao et al. (NCSU). 2024-03. ISCA 2024, arXiv:2403.17312. https://arxiv.org/abs/2403.17312
  - Sparse Window Attention + 3-phase scheduler; 3× over FlexGen.
- <a id="snapkv"></a>`[SnapKV]` *SnapKV: LLM Knows What You Are Looking for Before Generation*. Li et al. (UIUC/Cohere). 2024-04. NeurIPS 2024, arXiv:2404.14469. https://arxiv.org/abs/2404.14469
  - Last-N observation tokens detect head-specific patterns; 380× compression at 380K.
- <a id="pyramidinfer"></a>`[PyramidInfer]` *PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inference*. Yang et al. (SJTU). 2024-05. ACL Findings 2024, arXiv:2405.12532. https://arxiv.org/abs/2405.12532
- <a id="pyramidkv"></a>`[PyramidKV]` *PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling*. Cai et al. (Pekin/Microsoft). 2024-06. ICLR 2025, arXiv:2406.02069. https://arxiv.org/abs/2406.02069
  - Layer-tapered KV budget; matches full-cache LongBench at 12% retention.
- <a id="quest"></a>`[Quest]` *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference*. Tang et al. (MIT-HAN-Lab). 2024-06. ICML 2024, arXiv:2406.10774. https://arxiv.org/abs/2406.10774
  - Per-page min/max + per-query criticality top-K; 2.23× attention speedup.
- <a id="loki"></a>`[Loki]` *Loki: Low-rank Keys for Efficient Sparse Attention*. Singhania et al. (UMD). 2024-06. NeurIPS 2024, arXiv:2406.02542. https://arxiv.org/abs/2406.02542
  - Top-K key selection in PCA-projected key space.
- <a id="infinigen"></a>`[InfiniGen]` *InfiniGen: Efficient Generative Inference of LLMs with Dynamic KV Cache Management*. Lee et al. (SNU). 2024-06. OSDI 2024, arXiv:2406.19707. https://arxiv.org/abs/2406.19707
  - Speculatively prefetch essential KV pages from CPU; 3× over offload.
- <a id="ada-kv"></a>`[Ada-KV]` *Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation*. Feng et al. (USTC). 2024-07. NeurIPS 2025, arXiv:2407.11550. https://arxiv.org/abs/2407.11550
  - Per-head adaptive budget; plug-in upgrade to SnapKV/PyramidKV.
- <a id="kvmerger"></a>`[KVMerger]` *Adaptive KV Cache Merging for LLMs on Long-Context Tasks*. Wang et al. 2024-07. arXiv:2407.08454. https://arxiv.org/abs/2407.08454
  - Token-merging instead of dropping; pivotal-token Gaussian-weighted merge sets.
- <a id="kv-bench"></a>`[KV-Bench]` *KV Cache Compression, But What Must We Give in Return?*. EMNLP 2024. arXiv:2407.01527. https://arxiv.org/abs/2407.01527
  - Honest benchmark across 10+ methods, 7 task families.
- <a id="nacl"></a>`[NACL]` *NACL: A General and Effective KV Cache Eviction Framework*. Chen et al. (Baidu). 2024-08. ACL 2024. https://aclanthology.org/2024.acl-long.428/
  - Proxy-token attention statistics + diversified random eviction; 5× cache reduction.
- <a id="retrievalattention"></a>`[RetrievalAttention]` *RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval*. Liu et al. (Microsoft). 2024-09. arXiv:2409.10516. https://arxiv.org/abs/2409.10516
  - CPU ANN over offloaded KV; 1–3% of cache accessed.
- <a id="duoattention"></a>`[DuoAttention]` *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads*. Xiao et al. (MIT-HAN-Lab). 2024-10. ICLR 2025, arXiv:2410.10819. https://arxiv.org/abs/2410.10819
  - Heads split into retrieval (full) vs streaming (sink+window); 2.55× MHA reduction.
- <a id="kv-survey-2024"></a>`[KV-Survey-2024]` *A Survey on LLM Acceleration based on KV Cache Management*. 2024-12. arXiv:2412.19442. https://arxiv.org/abs/2412.19442
- <a id="kv-survey-2025"></a>`[KV-Survey-2025]` *Key, Value, Compress: A Systematic Exploration of KV Cache Compression Techniques*. 2025-03. arXiv:2503.11816. https://arxiv.org/abs/2503.11816

#### Past 12 months SOTA — KV cache

- <a id="droidspeak"></a>`[DroidSpeak]` *DroidSpeak: KV Cache Sharing for Cross-LLM Communication*. Liu et al. (UChicago/Microsoft). 2024-11. arXiv:2411.02820. https://arxiv.org/abs/2411.02820
  - 4× throughput, 3.1× TTFT in compound-AI/agent settings.
- <a id="scbench"></a>`[SCBench]` *SCBench: A KV Cache-Centric Analysis of Long-Context Methods*. Microsoft/HKUST. 2024-12. ICLR 2025, arXiv:2412.10319. https://arxiv.org/abs/2412.10319
  - Evaluates compression/retrieval/loading across full KV lifecycle.
- <a id="strata"></a>`[Strata]` *Strata: Hierarchical Context Caching for Long Context LM Serving*. 2025-08. arXiv:2508.18572. https://arxiv.org/abs/2508.18572
  - GPU-assisted I/O against KV fragmentation; cache-aware request scheduling.
- <a id="lmcache"></a>`[LMCache]` *LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference*. LMCache team / U. Chicago. 2025-10. arXiv:2510.09665. https://arxiv.org/abs/2510.09665
  - Reference open-source KV layer; 8+ backends; 15× throughput in shared-prefix workloads.
- <a id="pitfalls-kv"></a>`[Pitfalls-KV]` *The Pitfalls of KV Cache Compression*. 2025-10. arXiv:2510.00231. https://arxiv.org/abs/2510.00231
  - Critical evaluation: instructions silently dropped under compression; system-prompt leakage.
- <a id="rethinking-kv"></a>`[Rethinking-KV]` *Rethinking Key-Value Cache Compression Techniques for LLM Serving*. MLSys 2025. arXiv:2503.24000. https://arxiv.org/abs/2503.24000
  - Eviction methods don't compose with FA/PagedAttention as deployed.
- <a id="lookaheadkv"></a>`[LookaheadKV]` *LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future*. Samsung Research. 2025. arXiv:2603.10899. https://arxiv.org/abs/2603.10899
  - Trained "lookahead" tokens + LoRA predict importance scores via KL.
- <a id="foresightkv"></a>`[ForesightKV]` *ForesightKV: Optimizing KV Cache Eviction for Reasoning Models*. 2025. arXiv:2602.03203. https://arxiv.org/abs/2602.03203
  - Lightweight scoring model trained to predict long-term contribution.
- <a id="lragent"></a>`[LRAgent]` *LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents*. 2025. arXiv:2602.01053. https://arxiv.org/abs/2602.01053
  - Tree-based KV-cache + LoRA-adapter co-management.
- <a id="cmx"></a>`[CMX]` *NVIDIA BlueField-4 Powered CMX (Inference Context Memory Storage Platform)*. NVIDIA. 2026-01. https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/
  - 4-tier KV pyramid down to NVMe-resident persistent KV.

#### Lineage — low-rank / latent / merging (KV-related)

- <a id="mla-v2"></a>`[MLA-V2]` *DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model*. DeepSeek-AI. 2024-05. arXiv:2405.04434. https://arxiv.org/abs/2405.04434
  - Multi-head Latent Attention; ~93.3% KV reduction vs MHA; 5.76× generation throughput.
- <a id="yoco"></a>`[YOCO]` *You Only Cache Once: Decoder-Decoder Architectures for Language Models*. Sun et al. (Microsoft). 2024-05. NeurIPS 2024 oral, arXiv:2405.05254. https://arxiv.org/abs/2405.05254
  - Cache one global KV; ~L× memory reduction; 1M-context prefill in seconds.
- <a id="minicache"></a>`[MiniCache]` *MiniCache: KV Cache Compression in Depth Dimension for LLMs*. Liu et al. (Monash). 2024-05. NeurIPS 2024, arXiv:2405.14366. https://arxiv.org/abs/2405.14366
  - Cross-layer KV merging via magnitude+direction; 1.53× compression alone.
- <a id="eigen-attn"></a>`[Eigen-Attn]` *Eigen Attention: Attention in Low-Rank Space for KV Cache Compression*. Saxena et al. 2024-08. EMNLP Findings 2024, arXiv:2408.05646. https://arxiv.org/abs/2408.05646
  - Joint low-rank approx of Q,K,V; 40% KV reduction.
- <a id="xquant"></a>`[XQuant]` *XQuant: Achieving Ultra-Low Bit KV Cache Quantization with Cross-Layer Compression*. 2025.
  - Cross-layer KV compression / sharing.

#### Past 12 months SOTA — tiered KV / disagg KV

- <a id="mooncake"></a>`[Mooncake]` *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*. Qin et al. (Moonshot AI / Tsinghua). 2024-06. FAST 2025, arXiv:2407.00079. https://arxiv.org/abs/2407.00079
  - Distributed KVCache pool over CPU/DRAM/SSD/NIC; Conductor scheduler; Transfer Engine.
- <a id="mooncake-store"></a>`[Mooncake-Store]` *Mooncake Store / Transfer Engine*. open-sourced 2024-11 / 2025-03. https://github.com/kvcache-ai/Mooncake
- <a id="dynamo"></a>`[Dynamo]` *NVIDIA Dynamo: Distributed Inference Framework*. NVIDIA. 2025-03 (GTC). https://docs.nvidia.com/dynamo/
  - KVBM + NIXL transfer + KV-aware Smart Router (radix-tree); disaggregated by default.
- <a id="dynamo-launch"></a>`[Dynamo-launch]` *Introducing NVIDIA Dynamo*. NVIDIA. GTC 2025-03. https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
- <a id="nixl"></a>`[NIXL]` *NVIDIA Inference Transfer Library*. NVIDIA / ai-dynamo. 2025-03. https://github.com/ai-dynamo/nixl
  - Transport for async KV transfers across NVLink/IB/RoCE/GDS/S3.
- <a id="dynamo-nixl"></a>`[Dynamo-NIXL]` *Enhancing Distributed Inference Performance with NIXL*. NVIDIA Tech Blog. https://developer.nvidia.com/blog/enhancing-distributed-inference-performance-with-the-nvidia-inference-transfer-library/
- <a id="dynamo-kvbm"></a>`[Dynamo-KVBM]` *KVBM Architecture*. NVIDIA Dynamo docs. https://docs.nvidia.com/dynamo/latest/kvbm/kvbm_architecture.html
- <a id="aibrix"></a>`[AIBrix]` *AIBrix: Towards Scalable, Cost-Effective LLM Inference Infrastructure*. Bytedance. 2025-04. arXiv:2504.03648. https://arxiv.org/abs/2504.03648
  - Distributed KV cache, multi-tier manager, pluggable eviction; 50% throughput improvement.
- <a id="sglang-hicache"></a>`[SGLang-HiCache]` *SGLang HiCache: Fast Hierarchical KV Caching*. SGLang/LMSYS. 2025-09. https://www.lmsys.org/blog/2025-09-10-sglang-hicache/
  - HiRadixTree as page table over GPU/CPU/storage; write-through/back policies.
- <a id="llm-d-fs"></a>`[llm-d-FS]` *Native KV Cache Offloading to Any Filesystem with llm-d*. Red Hat/IBM. 2025. https://llm-d.ai/blog/native-kv-cache-offloading-to-any-file-system-with-llm-d
  - Filesystem-agnostic POSIX connector with async I/O and GPU DMA.
- <a id="lmcache-gke"></a>`[LMCache-GKE]` *LMCache on Google Kubernetes Engine*. 2025-10. https://blog.lmcache.ai/en/2025/10/07/lmcache-on-google-kubernetes-engine-boosting-llm-inference-performance-with-kv-cache-on-tiered-storage/
- <a id="mla-v3"></a>`[MLA-V3]` See `[DeepSeek-V3-FP8]` above. MLA at scale (671B MoE, 37B activated); ~32× compression ratio.
- <a id="expertchoice"></a>`[ExpertChoice]` See MoE section. Cross-references KV via low-rank principles.

### Batching and scheduling

#### Lineage

- <a id="orca"></a>`[ORCA]` *Orca: A Distributed Serving System for Transformer-Based Generative Models*. Yu, Jeong, Kim, Kim, Chun (Seoul Nat'l U/FriendliAI). 2022-07. OSDI 2022. https://www.usenix.org/conference/osdi22/presentation/yu
  - Iteration-level scheduling and selective batching; canonical origin of continuous batching.
- <a id="flexgen"></a>`[FlexGen]` *FlexGen: High-Throughput Generative Inference of LLMs with a Single GPU*. Sheng et al. (Stanford/UCB). 2023-03. ICML 2023, arXiv:2303.06865. https://arxiv.org/abs/2303.06865
  - LP-derived schedule for tensor placement; foundational for offload lineage.
- <a id="fastserve"></a>`[FastServe]` *Fast Distributed Inference Serving for Large Language Models*. Wu, Bai, Liu, Zhang, Yi, Jin (PKU). 2023-05. arXiv:2305.05920. https://arxiv.org/abs/2305.05920
  - Skip-join multi-level feedback queue for token-level preemption.
- <a id="sarathi"></a>`[Sarathi]` *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*. Agrawal, Panwar, Mohan, Kwatra, Gulavani, Ramjee (MSR India/GaTech). 2023-08. arXiv:2308.16369. https://arxiv.org/abs/2308.16369
  - Chunked-prefill primitive; precursor to Sarathi-Serve.
- <a id="ds-fastgen"></a>`[DS-FastGen]` *DeepSpeed-FastGen: High-throughput Text Generation for LLMs*. Holmes et al. (Microsoft). 2024-01. arXiv:2401.08671. https://arxiv.org/abs/2401.08671
  - Dynamic SplitFuse; constant-forward-size composer.
- <a id="distserve"></a>`[DistServe]` *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving*. Zhong, Liu, Chen, Hu, Zhu, Liu, Jin, Zhang (PKU/UCSD/StepFun). 2024-01. OSDI 2024, arXiv:2401.09670. https://arxiv.org/abs/2401.09670
  - Goodput formalism; disaggregated scheduling with co-optimized parallelism.
- <a id="vtc"></a>`[VTC]` *Fairness in Serving Large Language Models*. Sheng, Cao, Li, Zhu, Zhuo, Gonzalez, Stoica. 2023-12. OSDI 2024, arXiv:2401.00588. https://arxiv.org/abs/2401.00588
  - Virtual Token Counter; first formally fair LLM scheduler with 2× bound.
- <a id="llumnix"></a>`[Llumnix]` *Llumnix: Dynamic Scheduling for Large Language Model Serving*. Sun et al. (Alibaba). 2024-07. OSDI 2024, arXiv:2406.03243. https://arxiv.org/abs/2406.03243
  - KV-cache live migration across replicas; OS-style context switch of LLM serving.
- <a id="sarathi-serve"></a>`[Sarathi-Serve]` *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*. Agrawal et al. (Microsoft/GaTech). 2024-03. OSDI 2024, arXiv:2403.02310. https://arxiv.org/abs/2403.02310
  - Production-ready chunked prefill + stall-free schedule; defines TTFT/ITL knobs.
- <a id="andes"></a>`[Andes]` *Andes: Defining and Enhancing Quality-of-Experience in LLM-Based Text Streaming Services*. Liu, Wu, et al. 2024-04. arXiv:2404.16283. https://arxiv.org/abs/2404.16283
  - Token-granularity preemptive scheduler with QoE function.
- <a id="ltr"></a>`[LTR]` *Efficient LLM Scheduling by Learning to Rank*. Fu, Zhu, Su, Sheng, Yang, Jiao, Zhang. 2024-08. NeurIPS 2024, arXiv:2408.15792. https://arxiv.org/abs/2408.15792
  - Predict pairwise ranks (not lengths); 2.8× chatbot latency reduction.
- <a id="pod-attn"></a>`[POD-Attn]` *POD-Attention: Unlocking Full Prefill-Decode Overlap*. Kamath, Mohan, Panwar, Ramjee, et al. 2024-10. ASPLOS 2025, arXiv:2410.18038. https://arxiv.org/abs/2410.18038
  - Single GPU kernel runs prefill-attention and decode-attention on same SM; 59% faster.
- <a id="mnemosyne"></a>`[Mnemosyne]` *Mnemosyne: Parallelization Strategies for Multi-Million Context Length LLM Inference*. Microsoft. 2024-09. arXiv:2409.17264. https://arxiv.org/abs/2409.17264
  - Adaptive chunking + Sequence Pipeline Parallelism + KV Cache Parallelism.

#### Past 12 months SOTA — batching/scheduling

- <a id="vllm-v1"></a>`[vLLM-V1]` *vLLM V1 Alpha: A major upgrade to vLLM's core architecture*. vLLM team. 2025-01. https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
  - Unified scheduler; prefill/decode phase distinction abolished; chunked prefill default.
- <a id="vllm-v1-rh"></a>`[vLLM-V1-RH]` *vLLM V1: A Major Upgrade*. Red Hat developer write-up. 2025-01. https://developers.redhat.com/articles/2025/01/28/vllm-v1-a-major-upgrade-vllms-core-architecture
- <a id="sglang-v0-4"></a>`[SGLang-v0.4]` *SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs*. LMSYS. 2024-12. https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/
  - CPU-side scheduling overlapped one batch ahead; Rust router with approximate radix tree.
- <a id="dlpm"></a>`[DLPM]` *Locality-aware Fair Scheduling in LLM Serving*. Cao, Wang, Mao, Hsu et al. 2025-01. arXiv:2501.14312. https://arxiv.org/abs/2501.14312
  - DLPM/D²LPM: fairness ⊥ prefix-locality tension; up to 2.87× over VTC.
- <a id="onlinekv"></a>`[OnlineKV]` *Online Scheduling for LLM Inference with KV Cache Constraints*. Jaillet et al. 2025-02. arXiv:2502.07115. http://www.mit.edu/~jaillet/general/2502.07115v5.pdf
  - Competitive-ratio analysis with KV memory as explicit resource.
- <a id="qpredllm"></a>`[QPredLLM]` *Queueing, Predictions, and Large Language Models*. arXiv:2503.07545. 2025-03. https://arxiv.org/abs/2503.07545
- <a id="throughputopt"></a>`[ThroughputOpt]` *Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents*. 2025-04. arXiv:2504.07347. https://arxiv.org/abs/2504.07347
  - SLAI scheduler for TBT-deadline awareness.
- <a id="slos-serve"></a>`[SLOs-Serve]` *SLOs-Serve: Optimized Serving of Multi-SLO LLMs*. 2025-04. MLSys 2025, arXiv:2504.08784. https://arxiv.org/abs/2504.08784
  - 2.2× per-GPU capacity over prior art.
- <a id="jitserve"></a>`[JITServe]` *JITServe: SLO-aware LLM Serving with Imprecise Request Information*. Zhang et al. 2025-04. NSDI 2026, arXiv:2504.20068. https://arxiv.org/abs/2504.20068
  - 1.4–6.3× goodput; grouped margin goodput allocation.
- <a id="flowkv"></a>`[FlowKV]` *FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer*. 2025-04. arXiv:2504.03775. https://arxiv.org/abs/2504.03775
- <a id="arrow"></a>`[Arrow]` *Arrow: Adaptive Scheduling Mechanisms for Disaggregated LLM Inference*. 2025-05. arXiv:2505.11916. https://arxiv.org/abs/2505.11916
- <a id="sola"></a>`[SOLA]` *SOLA: Optimizing SLO Attainment for LLM Serving with State-Aware Scheduling*. Tsinghua/Infinigence/SJTU/PKU. 2025-05. MLSys 2025. https://mlsys.org/virtual/2025/poster/3231
  - SLO attainment 45.5% → 99.4% on identical hardware.
- <a id="helix"></a>`[Helix]` *Helix: Serving LLMs over Heterogeneous GPUs and Network via Max-Flow*. Mei, Tang, Vinayak, Zaharia, Zhang. 2024-06. ASPLOS 2025, arXiv:2406.01566. https://arxiv.org/abs/2406.01566
  - Max-flow MILP joint placement + scheduling; 3.3× throughput on heterogeneous clusters.
- <a id="kvflow"></a>`[KVFlow]` *KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows*. 2025-07. arXiv:2507.07400. https://arxiv.org/abs/2507.07400
- <a id="bucketserve"></a>`[BucketServe]` *Bucket-based dynamic batching*. 2025-07. arXiv:2507.17120. https://arxiv.org/abs/2507.17120
- <a id="optimal-llm-sched"></a>`[Optimal-LLM-Sched]` *Optimal Scheduling Algorithms for LLM Inference: Theory and Practice*. 2025-08. arXiv:2508.01002. https://arxiv.org/abs/2508.01002
  - Throughput-optimality results; RAD scheduler.
- <a id="hotprefix"></a>`[HotPrefix]` *HotPrefix: Hotness-Aware KV Cache Scheduling*. 2025. https://dl.acm.org/doi/10.1145/3749168
- <a id="multi-bin"></a>`[Multi-bin]` *Multi-Bin Batching for Increasing LLM Inference Throughput*. 2024-12. arXiv:2412.04504. https://arxiv.org/abs/2412.04504
- <a id="don't-stop-me-now"></a>`[Don't-Stop-Me-Now]` *Embedding-based scheduling for LLMs*. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/9eb8b5ccb0de594a16548f7c058fdadf-Paper-Conference.pdf
- <a id="semi-clairvoyant"></a>`[Semi-Clairvoyant]` *Semi-Clairvoyant Scheduling of Speculative Decoding Requests*. IJCAI 2025. https://www.ijcai.org/proceedings/2025/0951.pdf
- <a id="fairbatching"></a>`[FairBatching]` *FairBatching: Fairness-Aware Batch Formation*. 2025-10. arXiv:2510.14392. https://arxiv.org/abs/2510.14392
- <a id="hummingbird"></a>`[Hummingbird]` *Hummingbird: SLO-Oriented GPU Preemption at Microsecond-scale*. 2026-01. arXiv:2601.04071. https://arxiv.org/abs/2601.04071
- <a id="nv-dynamo-04"></a>`[NV-Dynamo-04]` *Dynamo 0.4: 4× Faster Performance, SLO-based Autoscaling, Real-time Observability*. NVIDIA Tech Blog. 2025. https://developer.nvidia.com/blog/dynamo-0-4-delivers-4x-faster-performance-slo-based-autoscaling-and-real-time-observability/

### Prefill–decode disaggregation

#### Lineage

- <a id="splitwise"></a>`[Splitwise]` *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*. Patel, Choukse, Zhang, Shah, Goiri, Maleki, Bianchini (Microsoft). 2023-11. ISCA 2024 (best paper), arXiv:2311.18677. https://arxiv.org/abs/2311.18677
  - First systems paper articulating PD as deployment problem; 1.4× throughput at -20% cost.
- <a id="tetriinfer"></a>`[TetriInfer]` *Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads*. Hu et al. (PKU/ByteDance). 2024-01. arXiv:2401.11181. https://arxiv.org/abs/2401.11181
  - Three-pillar: chunked prompt + PD disagg + two-level scheduler.
- <a id="dejavu-ft"></a>`[DejaVu-FT]` *DéjàVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving*. Strati, McAllister, Phanishayee, Tarnawski, Klimovic (ETH/MSR). 2024-03. ICML 2024, arXiv:2403.01876. https://arxiv.org/abs/2403.01876
  - Prompt-token disaggregation, microbatch swap, per-stage KV replication; FT reference.
- <a id="p/d-serve"></a>`[P/D-Serve]` *P/D-Serve: Serving Disaggregated Large Language Model at Scale*. Huawei Cloud. 2024-08. arXiv:2408.08147. https://arxiv.org/html/2408.08147v1

#### Past 12 months SOTA — disaggregation

- <a id="beyondbuzz"></a>`[BeyondBuzz]` *Beyond the Buzz: A Pragmatic Take on Inference Disaggregation*. Mitra, Borkar, et al. (NVIDIA). 2025-06. arXiv:2506.05508. https://arxiv.org/abs/2506.05508
  - First systematic NVIDIA-internal study; disaggregation pays off for prefill-heavy + larger models.
- <a id="nexus"></a>`[Nexus]` *Nexus: Proactive Intra-GPU Disaggregation of Prefill and Decode in LLM Serving*. 2025-07. arXiv:2507.06608. https://arxiv.org/abs/2507.06608
  - Intra-GPU disagg via dynamic SM allocation; up to 2× over SGLang.
- <a id="taichi"></a>`[TaiChi]` *Prefill-Decode Aggregation or Disaggregation? Unifying Both for Goodput-Optimized LLM Serving*. Wang et al. 2025-08. arXiv:2508.01989. https://arxiv.org/abs/2508.01989
  - SLO-regime dependence; aggregation under tight TTFT, disagg under tight TPOT.
- <a id="heteroscale"></a>`[HeteroScale]` *Taming the Chaos: Coordinated Autoscaling for Heterogeneous and Disaggregated LLM Inference*. ByteDance. 2025-08. arXiv:2508.19559. https://arxiv.org/abs/2508.19559
  - Decode TPS as best autoscaling signal; +26.6 pp utilization on tens of thousands of GPUs.
- <a id="hetero-pd"></a>`[Hetero-PD]` *Disaggregated Prefill and Decoding Inference System for Heterogeneous GPUs*. 2025-09. arXiv:2509.17542. https://arxiv.org/abs/2509.17542
- <a id="pdtrim"></a>`[PDTrim]` *PDTrim: Targeted Pruning for Prefill-Decode Disaggregation*. 2025-09. arXiv:2509.04467. https://arxiv.org/abs/2509.04467
  - Verified system name is PDTrim.
- <a id="dopd"></a>`[DOPD]` *DOPD: A Dynamic PD-Disaggregation Architecture for Maximizing Goodput*. 2025-11. arXiv:2511.20982. https://arxiv.org/abs/2511.20982
  - Goodput +1.5×; P90 TTFT -67.5% vs vLLM/DistServe.
- <a id="duetserve"></a>`[DuetServe]` *DuetServe: Harmonizing Prefill and Decode for LLM Serving*. 2025-11. arXiv:2511.04791. https://arxiv.org/abs/2511.04791
- <a id="tokenscale"></a>`[TokenScale]` *TokenScale: Timely and Accurate Autoscaling for Disaggregated LLM Serving with Token Velocity*. 2025-12. arXiv:2512.03416. https://arxiv.org/abs/2512.03416
  - SLO attainment 50–88% → 80–96%; cost -4–14% vs DistServe/BlitzScale/AIBrix.
- <a id="tract"></a>`[TraCT]` *TraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale*. 2025-12. arXiv:2512.18194. https://arxiv.org/html/2512.18194v1
- <a id="togethercpd"></a>`[TogetherCPD]` *Cache-aware Prefill-Decode Disaggregation*. Together AI engineering blog. 2025. https://www.together.ai/blog/cache-aware-disaggregated-inference
  - Three-tier (pre-prefill / prefill / decode); +40% throughput.
- <a id="prefillshare"></a>`[PrefillShare]` *PrefillShare: A Shared Prefill Module for KV Reuse in Multi-LLM Disaggregated Serving*. 2026-02. arXiv:2602.12029. https://arxiv.org/abs/2602.12029
- <a id="prefillaas"></a>`[PrefillAaS]` *Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter*. 2026-04. arXiv:2604.15039. https://arxiv.org/html/2604.15039v1
- <a id="sglang-largeep"></a>`[SGLang-LargeEP]` *Deploying DeepSeek with PD Disaggregation and Large-Scale EP on 96 H100 GPUs*. LMSYS. 2025-05. https://www.lmsys.org/blog/2025-05-05-large-scale-ep/
- <a id="sglang-k2-128"></a>`[SGLang-K2-128]` *Deploying Kimi K2 with PD Disaggregation and Large-Scale EP on 128 H200 GPUs*. LMSYS. 2025-07. https://www.lmsys.org/blog/2025-07-20-k2-large-scale-ep/
- <a id="sglang-gb200-1"></a>`[SGLang-GB200-1]` *Deploying DeepSeek on GB200 NVL72 with PD and Large-Scale EP (Part I)*. LMSYS. 2025-06. https://www.lmsys.org/blog/2025-06-16-gb200-part-1/
- <a id="sglang-gb200-2"></a>`[SGLang-GB200-2]` *Deploying DeepSeek on GB200 NVL72 (Part II)*. LMSYS. 2025-09. https://www.lmsys.org/blog/2025-09-25-gb200-part-2/
- <a id="perplexity-te"></a>`[Perplexity-TE]` *Disaggregated Prefill and Decode at Perplexity*. https://research.perplexity.ai/articles/disaggregated-prefill-and-decode
- <a id="perplexity-rdma"></a>`[Perplexity-RDMA]` *RDMA Point-to-point Communication for LLM Systems*. Perplexity. https://research.perplexity.ai/articles/rdma-point-to-point-communication-for-llm-systems
- <a id="deepseek-v3-insights"></a>`[DeepSeek-V3-Insights]` *Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures*. DeepSeek-AI. 2025-05. ISCA 2025, arXiv:2505.09343. https://arxiv.org/abs/2505.09343
- <a id="ds-inference-overview"></a>`[DS-Inference-Overview]` *DeepSeek-V3/R1 Inference System Overview*. DeepSeek (Open Source Week Day 6). 2025-02. https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
- <a id="trt-llm-disagg"></a>`[TRT-LLM-Disagg]` *Disaggregated Serving in TensorRT-LLM*. NVIDIA. https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html
- <a id="vllm-disagg-doc"></a>`[vLLM-Disagg-Doc]` *Disaggregated Prefill in vLLM*. vLLM docs. https://docs.vllm.ai/en/latest/features/disagg_prefill/
- <a id="pytorch-disagginfer"></a>`[PyTorch-DisaggInfer]` *Disaggregated Inference at Scale with PyTorch + vLLM*. PyTorch blog. https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/

### MoE inference

#### Lineage — model

- <a id="shazeer-2017"></a>`[Shazeer-2017]` *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. Shazeer et al. (Google Brain). 2017-01. ICLR 2017, arXiv:1701.06538. https://arxiv.org/abs/1701.06538
  - The MoE-as-a-layer paper; top-k gating, load-balance loss, noisy gating.
- <a id="gshard"></a>`[GShard]` *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*. Lepikhin et al. (Google). 2020-06. ICLR 2021, arXiv:2006.16668. https://arxiv.org/abs/2006.16668
  - First production transformer MoE; established dispatch/combine pattern.
- <a id="switch"></a>`[Switch]` *Switch Transformers: Scaling to Trillion Parameter Models*. Fedus, Zoph, Shazeer (Google). 2021-01. JMLR 2022, arXiv:2101.03961. https://arxiv.org/abs/2101.03961
  - Top-1 routing simplification; canonical "MoE = capacity for free" reference.
- <a id="glam"></a>`[GLaM]` *GLaM: Efficient Scaling of Language Models with Mixture-of-Experts*. Du et al. (Google). 2021-12. ICML 2022, arXiv:2112.06905. https://arxiv.org/abs/2112.06905
  - 1.2T-param 64-expert decoder; ½ inference FLOPs of GPT-3 at better quality.
- <a id="st-moe"></a>`[ST-MoE]` *ST-MoE: Designing Stable and Transferable Sparse Expert Models*. Zoph et al. 2022-02. arXiv:2202.08906. https://arxiv.org/abs/2202.08906
  - 269B sparse with router-z-loss; first to top SuperGLUE/ARC.
- `[ExpertChoice]` *Mixture-of-Experts with Expert Choice Routing*. Zhou et al. (Google). 2022-02. NeurIPS 2022, arXiv:2202.09368. https://arxiv.org/abs/2202.09368
  - Inverts routing: experts pick top-k tokens; eliminates dropping.
- <a id="mixtral-8x7b"></a>`[Mixtral-8x7B]` *Mixtral of Experts*. Jiang et al. (Mistral AI). 2024-01. arXiv:2401.04088. https://arxiv.org/abs/2401.04088
  - 47B/13B-active 8-expert; first mainstream open-weight transformer MoE.
- <a id="mixtral-8x22b"></a>`[Mixtral-8x22B]` *Cheaper, Better, Faster, Stronger*. Mistral AI. 2024-04. https://mistral.ai/news/mixtral-8x22b
- <a id="grok-1"></a>`[Grok-1]` *Open Release of Grok-1*. xAI. 2024-03. https://x.ai/news/grok-os
- <a id="dbrx"></a>`[DBRX]` *Introducing DBRX*. Mosaic/Databricks. 2024-03. https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm
- <a id="snowflake-arctic"></a>`[Snowflake-Arctic]` *Snowflake Arctic: Open Efficient Foundation Models*. Snowflake. 2024-04. https://www.snowflake.com/en/blog/arctic-open-efficient-foundation-language-models-snowflake/
- <a id="deepseekmoe"></a>`[DeepSeekMoE]` *DeepSeekMoE: Towards Ultimate Expert Specialization in MoE Language Models*. Dai et al. (DeepSeek). 2024-01. ACL 2024, arXiv:2401.06066. https://arxiv.org/abs/2401.06066
  - Fine-grained segmentation + shared-expert isolation; foundation of V2/V3/R1.
- <a id="loss-free-balancing"></a>`[Loss-Free-Balancing]` *Auxiliary-Loss-Free Load Balancing Strategy for MoE*. Wang et al. (DeepSeek). 2024-08. arXiv:2408.15664. https://arxiv.org/abs/2408.15664
- <a id="olmoe"></a>`[OLMoE]` *OLMoE: Open Mixture-of-Experts Language Models*. Muennighoff et al. (AI2/Contextual). 2024-09. arXiv:2409.02060. https://arxiv.org/abs/2409.02060
  - Fully open (data, code, ckpts); canonical reproducible MoE baseline.
- <a id="phi-3-5-moe"></a>`[Phi-3.5-MoE]` *Phi-3.5-MoE-instruct*. Microsoft. 2024-08. https://huggingface.co/microsoft/Phi-3.5-MoE-instruct
- <a id="granite-3-moe"></a>`[Granite-3-MoE]` *IBM Granite 3.0*. IBM. 2024-10. https://www.ibm.com/new/announcements/ibm-granite-3-0-open-state-of-the-art-enterprise-models
- <a id="deepseek-r1"></a>`[DeepSeek-R1]` *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL*. DeepSeek-AI. 2025-01. arXiv:2501.12948. https://arxiv.org/abs/2501.12948
  - V3 backbone (671B/37B, 256+1) trained for reasoning.
- <a id="llama-4"></a>`[Llama-4]` *The Llama 4 Herd*. Meta. 2025-04. https://ai.meta.com/blog/llama-4-multimodal-intelligence/
  - Scout (109B/17B) and Maverick (400B/17B); Meta's first MoE.
- <a id="qwen3"></a>`[Qwen3]` *Qwen3 Technical Report*. Qwen Team (Alibaba). 2025-05. arXiv:2505.09388. https://arxiv.org/abs/2505.09388
- <a id="step-3"></a>`[Step-3]` *Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding*. StepFun. 2025-07. arXiv:2507.19427. https://arxiv.org/abs/2507.19427
  - 321B VLM with MFA + Attention-FFN Disaggregation; 4039 tok/s/GPU decode.
- <a id="kimi-k2"></a>`[Kimi-K2]` *Kimi K2: Open Agentic Intelligence*. Moonshot AI. 2025-07. arXiv:2507.20534. https://arxiv.org/abs/2507.20534
  - 1.04T/32B-active, 384 routed experts, MLA, MuonClip optimizer.
- <a id="gpt-oss-deploy"></a>`[GPT-OSS-Deploy]` *GPT-OSS Deployment Analysis*. 2025-08. arXiv:2508.16700. https://arxiv.org/abs/2508.16700
- <a id="qwen3-next-nvidia"></a>`[Qwen3-Next-NVIDIA]` *Qwen3-Next 80B-A3B*. Qwen + NVIDIA. 2025-09. https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform/
- <a id="deepseek-v3-2"></a>`[DeepSeek-V3.2]` *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*. DeepSeek-AI. 2025-12. arXiv:2512.02556. https://arxiv.org/abs/2512.02556
  - Adds DSA (lightning indexer + top-k); reduces attention to O(L·k) at 128K.

#### Lineage — systems

- <a id="fastmoe"></a>`[FastMoE]` *FastMoE: A Fast Mixture-of-Expert Training System*. He, Qiu, Zeng, Yang, Zhai, Tang. 2021-03. arXiv:2103.13262. https://arxiv.org/abs/2103.13262
- <a id="deepspeed-moe"></a>`[DeepSpeed-MoE]` *DeepSpeed-MoE: Advancing MoE Inference and Training to Next-Generation AI Scale*. Rajbhandari et al. (Microsoft). 2022-01. ICML 2022, arXiv:2201.05596. https://arxiv.org/abs/2201.05596
  - 7.3× lower MoE inference latency.
- <a id="tutel"></a>`[Tutel]` *Tutel: Adaptive Mixture-of-Experts at Scale*. Hwang et al. (Microsoft). 2022-06. MLSys 2023, arXiv:2206.03382. https://arxiv.org/abs/2206.03382
- <a id="hetumoe"></a>`[HetuMoE]` *HetuMoE: An Efficient Trillion-scale MoE Distributed Training System*. Nie, Zhao et al. (PKU). 2022-03. arXiv:2203.14685. https://arxiv.org/abs/2203.14685
- <a id="lina"></a>`[Lina]` *Accelerating Distributed MoE Training and Inference with Lina*. Li et al. 2022-10. USENIX ATC 2023, arXiv:2210.17223. https://arxiv.org/abs/2210.17223
  - All-to-all prioritization over allreduce; 1.73× / 1.63× speedups.
- <a id="megablocks"></a>`[MegaBlocks]` *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts*. Gale, Narayanan, Young, Zaharia. 2022-11. MLSys 2023, arXiv:2211.15841. https://arxiv.org/abs/2211.15841
  - MoE as block-sparse GEMMs (BCSR); basis of DBRX.
- <a id="brainstorm"></a>`[Brainstorm]` *Optimizing Dynamic Neural Networks with Brainstorm*. Cui et al. (MSRA). OSDI 2023. https://www.usenix.org/system/files/osdi23-cui.pdf
- <a id="pre-gated-moe"></a>`[Pre-gated-MoE]` *Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable MoE Inference*. 2023-08. ISCA 2024, arXiv:2308.12066. https://arxiv.org/abs/2308.12066
- <a id="mixtral-offload"></a>`[Mixtral-Offload]` *Fast Inference of Mixture-of-Experts Language Models with Offloading*. Eliseev, Mazur. 2023-12. arXiv:2312.17238. https://arxiv.org/abs/2312.17238
- <a id="moe-infinity"></a>`[MoE-Infinity]` *MoE-Infinity: Activation-Aware Expert Offloading*. Xue et al. (Edinburgh). 2024-01. arXiv:2401.14361. https://arxiv.org/abs/2401.14361
- <a id="exflow"></a>`[ExFlow]` *Exploiting Inter-Layer Expert Affinity for Accelerating MoE Inference*. 2024-01. arXiv:2401.08383. https://arxiv.org/abs/2401.08383
- <a id="fiddler"></a>`[Fiddler]` *Fiddler: CPU-GPU Orchestration for Fast Inference of MoE Models*. Kamahori, Tang, Gu, Zhu, Kasikci (UW). 2024-02. ICLR 2025, arXiv:2402.07033. https://arxiv.org/abs/2402.07033
- <a id="expertflow-2024"></a>`[ExpertFlow-2024]` *ExpertFlow: Predictive Expert Caching and Token Scheduling*. 2024-10. arXiv:2410.17954. https://arxiv.org/abs/2410.17954
- <a id="promoe"></a>`[ProMoE]` *ProMoE: Fast MoE-based LLM Serving using Proactive Caching*. 2024-10. arXiv:2410.22134. https://arxiv.org/abs/2410.22134
- <a id="hobbit"></a>`[HOBBIT]` *HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference*. 2024-11. arXiv:2411.01433. https://arxiv.org/abs/2411.01433

#### Past 12 months SOTA — MoE systems

- <a id="deepep"></a>`[DeepEP]` *DeepEP: Efficient Expert-Parallel Communication Library*. DeepSeek. 2025-02. https://github.com/deepseek-ai/DeepEP
  - Two-mode kernels (HT/LL); intranode NVLink + internode RDMA; native FP8 dispatch.
- <a id="dualpipe"></a>`[DualPipe]` *DualPipe: Bidirectional Pipeline Parallelism for V3/R1 Training*. DeepSeek. 2025-02. https://github.com/deepseek-ai/DualPipe
  - Forward and backward in opposing directions; minimizes pipeline bubbles.
- <a id="eplb"></a>`[EPLB]` *Expert Parallelism Load Balancer*. DeepSeek. 2025-02. https://github.com/deepseek-ai/EPLB
  - Redundant-experts strategy; hierarchical (prefill) and global (decode) policies.
- <a id="lplb"></a>`[LPLB]` *LPLB: Linear-Programming-based Expert-Parallel Load Balancer*. DeepSeek. 2025. https://github.com/deepseek-ai/LPLB
- <a id="deepgemm"></a>`[DeepGEMM]` *DeepGEMM: Clean and Efficient FP8 GEMM Kernels with Fine-Grained Scaling*. DeepSeek. 2025-02. https://github.com/deepseek-ai/DeepGEMM
  - Grouped FP8 GEMMs; ~1550 TFLOPS on H800.
- <a id="ds-profile"></a>`[DS-Profile]` *DeepSeek profile-data*. https://github.com/deepseek-ai/profile-data
- <a id="megascale-moe"></a>`[MegaScale-MoE]` *MegaScale-MoE: Large-Scale Communication-Efficient Training of MoE in Production*. ByteDance. 2025-05. arXiv:2505.11432. https://arxiv.org/abs/2505.11432
  - 1.88× MFU vs Megatron-LM on 352B MoE / 1,440 GPUs.
- <a id="megascale-infer"></a>`[MegaScale-Infer]` *MegaScale-Infer: Serving MoE at Scale with Disaggregated Expert Parallelism*. ByteDance + PKU. 2025-04. SIGCOMM 2025, arXiv:2504.02263. https://arxiv.org/abs/2504.02263
  - Disaggregates attention from FFN; 2.56× / 1.28× per-GPU vs vLLM/TRT-LLM.
- <a id="flashdmoe"></a>`[FlashDMoE]` *FlashDMoE: Fast Distributed MoE in a Single Kernel*. 2025-06. arXiv:2506.04667. https://arxiv.org/abs/2506.04667
- <a id="flashcommv2"></a>`[FlashCommV2]` *FlashCommunication V2: Bit Splitting and Spike Reserving for Any Bit Communication*. 2025-08. arXiv:2508.03760. https://arxiv.org/abs/2508.03760
- <a id="finemoe"></a>`[FineMoE]` *FineMoE: Taming Latency-Memory Trade-Off via Fine-Grained Expert Offloading*. 2025-02. arXiv:2502.05370. https://arxiv.org/abs/2502.05370
  - Verified system name is FineMoE (was incorrectly cited as fMoE).
- <a id="hybrimoe"></a>`[HybriMoE]` *HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management*. 2025-04. arXiv:2504.05897. https://arxiv.org/abs/2504.05897
- <a id="prescope"></a>`[PreScope]` *PreScope: Unleashing the Power of Prefetching for Resource-Constrained MoE Inference*. 2025-09. arXiv:2509.23638. https://arxiv.org/abs/2509.23638
- <a id="expertflow-2025"></a>`[ExpertFlow-2025]` *ExpertFlow: Adaptive Expert Scheduling and Memory Coordination*. 2025-10. arXiv:2510.26730. https://arxiv.org/abs/2510.26730
- <a id="janus"></a>`[Janus]` *Janus: Disaggregating Attention and Experts for Scalable MoE Inference*. 2025-12. arXiv:2512.13525. https://arxiv.org/abs/2512.13525
  - Up to 4.7× per-GPU vs SOTA; 25% GPU-cost reduction.
- <a id="nccl-ep"></a>`[NCCL-EP]` *NCCL EP: Towards a Unified Expert Parallel Communication API*. 2026-03. arXiv:2603.13606. https://arxiv.org/html/2603.13606
- <a id="vllm-wideep"></a>`[vLLM-WideEP]` *vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP*. vLLM. 2025-12. https://blog.vllm.ai/2025/12/17/large-scale-serving.html
- <a id="trt-llm-wideep-pt2"></a>`[TRT-LLM-WideEP-Pt2]` *Scaling Expert Parallelism in TensorRT-LLM (Part 2)*. NVIDIA. 2025. https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog8_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.html
- <a id="trt-llm-nvl72"></a>`[TRT-LLM-NVL72]` *Scaling Large MoE Models with Wide Expert Parallelism on NVL72*. NVIDIA. 2025. https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/
- <a id="nvidia-dynamo-moe"></a>`[NVIDIA-Dynamo-MoE]` *How NVIDIA GB200 NVL72 and Dynamo Boost MoE Inference*. NVIDIA. https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models/
- <a id="moe-inf-bench"></a>`[MoE-Inf-Bench]` *MoE-Inference-Bench: Performance Evaluation of MoE LLMs and VLMs*. SC 2025 Workshops. arXiv:2508.17467. https://arxiv.org/abs/2508.17467
- <a id="moe-survey-2024"></a>`[MoE-Survey-2024]` *A Survey on Inference Optimization Techniques for MoE Models*. 2024-12. ACM CSur 2025, arXiv:2412.14219. https://arxiv.org/abs/2412.14219
- <a id="elastic-ep-sglang"></a>`[Elastic-EP-SGLang]` *Elastic EP in SGLang: Achieving Partial Failure Tolerance for DeepSeek MoE Deployments*. LMSYS. 2026-03. https://lmsys.org/blog/2026-03-25-eep-partial-failure-tolerance/

### Long-context inference

#### Lineage — position encodings

- <a id="roformer"></a>`[RoFormer]` *RoFormer: Enhanced Transformer with Rotary Position Embedding*. Su, Lu, Pan, Murtadha, Wen, Liu (Zhuiyi). 2021-04. Neurocomputing 2024, arXiv:2104.09864. https://arxiv.org/abs/2104.09864
  - Origin of RoPE.
- <a id="pi"></a>`[PI]` *Extending Context Window of LLMs via Positional Interpolation*. Chen, Wong, Chen, Tian (Meta). 2023-06. arXiv:2306.15595. https://arxiv.org/abs/2306.15595
  - First practical RoPE extension via linear down-scaling.
- <a id="yarn"></a>`[YaRN]` *YaRN: Efficient Context Window Extension of Large Language Models*. Peng, Quesnelle, Fan, Shippole. 2023-08. ICLR 2024, arXiv:2309.00071. https://arxiv.org/abs/2309.00071
  - SOTA RoPE extension recipe; NTK-by-parts + temperature scaling.
- <a id="ntk-aware"></a>`[NTK-aware]` *NTK-aware RoPE scaling*. EleutherAI / community blog. 2023. https://blog.eleuther.ai/yarn/
- <a id="longrope"></a>`[LongRoPE]` *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens*. Ding et al. (Microsoft). 2024-02. ICML 2024, arXiv:2402.13753. https://arxiv.org/abs/2402.13753
- <a id="longrope2"></a>`[LongRoPE2]` *LongRoPE2: Near-Lossless LLM Context Window Scaling*. Microsoft. 2025-02. arXiv:2502.20082. https://arxiv.org/abs/2502.20082
  - 128K with 10B tokens (vs Meta's 800B); high RoPE dims under-trained hypothesis.

#### Lineage — sequence/context parallelism

- <a id="megatron-sp"></a>`[Megatron-SP]` *Reducing Activation Recomputation in Large Transformer Models*. Korthikanti et al. (NVIDIA). 2022-05. arXiv:2205.05198. https://arxiv.org/abs/2205.05198
- <a id="ring-attn"></a>`[Ring-Attn]` *Ring Attention with Blockwise Transformers for Near-Infinite Context*. Liu, Zaharia, Abbeel. 2023-10. ICLR 2024, arXiv:2310.01889. https://arxiv.org/abs/2310.01889
  - KV blocks ring with overlapped P2P; canonical long-sequence attention algorithm.
- <a id="striped-attn"></a>`[Striped-Attn]` *Striped Attention: Faster Ring Attention for Causal Transformers*. Brandon et al. (MIT). 2023-11. arXiv:2311.09431. https://arxiv.org/abs/2311.09431
  - Permutes token-to-device mapping to balance causal-mask workload.
- <a id="ds-ulysses"></a>`[DS-Ulysses]` *DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models*. Microsoft. 2023-09. arXiv:2309.14509. https://arxiv.org/abs/2309.14509
  - All-to-all on heads; constant comm volume with seq-length scaling.
- <a id="usp"></a>`[USP]` *USP: A Unified Sequence Parallelism Approach for Long Context Generative AI*. Fang, Zhao. 2024-05. arXiv:2405.07719. https://arxiv.org/abs/2405.07719
  - 2D hybrid (Ulysses outer, Ring inner).

#### Lineage — sparse / sliding-window attention

- <a id="longformer"></a>`[Longformer]` *Longformer: The Long-Document Transformer*. Beltagy, Peters, Cohan. 2020-04. arXiv:2004.05150. https://arxiv.org/abs/2004.05150
- <a id="bigbird"></a>`[BigBird]` *Big Bird: Transformers for Longer Sequences*. Zaheer et al. (Google). 2020-07. arXiv:2007.14062. https://arxiv.org/abs/2007.14062
- <a id="longlora"></a>`[LongLoRA]` *LongLoRA: Efficient Fine-tuning of Long-Context LLMs*. Chen et al. (CUHK/MIT). 2023-09. ICLR 2024, arXiv:2309.12307. https://arxiv.org/abs/2309.12307

#### Past 12 months SOTA — long-context

- <a id="qwen2-5-1m"></a>`[Qwen2.5-1M]` *Qwen2.5-1M Technical Report*. Qwen Team. 2025-01. arXiv:2501.15383. https://arxiv.org/abs/2501.15383
  - 7B/14B-Instruct-1M; sparse attention + chunked prefill; 3–7× faster prefill.
- <a id="nsa"></a>`[NSA]` *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*. DeepSeek-AI. 2025-02. ACL 2025, arXiv:2502.11089. https://arxiv.org/abs/2502.11089
  - Three branches (compressed/selected/sliding); up to 11.6× decode speedup at 64K.
- <a id="mminference"></a>`[MMInference]` *MMInference: Modality-Aware Permutation Sparse Attention*. Microsoft. 2025-04. ICML 2025.
- <a id="minference"></a>`[MInference]` *Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention*. Microsoft. 2024-07. NeurIPS 2024 Spotlight, arXiv:2407.02490. https://arxiv.org/abs/2407.02490
  - Three head-pattern templates; 10× prefill speedup at 1M on A100.
- <a id="rwkv-7"></a>`[RWKV-7]` *RWKV-7 "Goose"*. Peng et al. 2025-03. COLM 2025, arXiv:2503.14456. https://arxiv.org/abs/2503.14456
  - Generalized delta rule with vector-valued gating.
- <a id="mamba"></a>`[Mamba]` *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. Gu, Dao. 2023-12. arXiv:2312.00752. https://arxiv.org/abs/2312.00752
- <a id="mamba-2"></a>`[Mamba-2]` *Transformers are SSMs: Generalized Models and Efficient Algorithms*. Dao, Gu. 2024-05. ICML 2024, arXiv:2405.21060. https://arxiv.org/abs/2405.21060
- <a id="jamba"></a>`[Jamba]` *Jamba: A Hybrid Transformer-Mamba Language Model*. AI21. 2024-03. arXiv:2403.19887. https://arxiv.org/abs/2403.19887
- <a id="jamba-1-5"></a>`[Jamba-1.5]` *Jamba-1.5*. AI21. 2024-08. arXiv:2408.12570. https://arxiv.org/abs/2408.12570
- <a id="hymba"></a>`[Hymba]` *Hymba: A Hybrid-Head Architecture for Small Language Models*. Dong et al. (NVIDIA). 2024-11. ICLR 2025, arXiv:2411.13676. https://arxiv.org/abs/2411.13676
- <a id="zamba"></a>`[Zamba]` *Zamba: A Compact 7B SSM Hybrid Model*. Glorioso, Anthony. 2024-05. arXiv:2405.16712. https://arxiv.org/abs/2405.16712
- <a id="zamba2"></a>`[Zamba2]` *Zamba2-7B*. Zyphra. 2024-08. https://www.zyphra.com/post/zamba2-7b
- <a id="falcon-h1"></a>`[Falcon-H1]` *Falcon-H1: A Family of Hybrid-Head Language Models*. TII. 2025-07. arXiv:2507.22448. https://arxiv.org/abs/2507.22448
  - 0.5B–34B; 256K context; parallel attn+SSM concatenated.
- <a id="granite-4"></a>`[Granite-4]` *IBM Granite 4.0: Hyper-Efficient Hybrid Models*. IBM. 2025-10. https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models
  - 9:1 Mamba-2 : attention; >70% RAM reduction at long ctx.
- <a id="llama 3-1"></a>`[Llama 3.1]` *Llama 3.1 (multi-stage RoPE-base)*. Meta. 2024-07. https://huggingface.co/blog/llama31
- <a id="llama 3-2"></a>`[Llama 3.2]` *Llama 3.2*. Meta. 2024-09. https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
- <a id="gemini-1-5"></a>`[Gemini-1.5]` *Unlocking Multimodal Understanding Across Millions of Tokens of Context*. Google DeepMind. 2024-03. arXiv:2403.05530. https://arxiv.org/abs/2403.05530
- <a id="claude-1m"></a>`[Claude-1M]` *Claude Opus 4.6 / Sonnet 4.6 (1M GA)*. Anthropic. 2025. https://www.anthropic.com/news/claude-opus-4-6
- <a id="niah"></a>`[NIAH]` *Needle-in-a-Haystack benchmark*. Kamradt. 2023. https://github.com/gkamradt/LLMTest_NeedleInAHaystack
- <a id="ruler"></a>`[RULER]` *RULER: What's the Real Context Size of Your Long-Context Language Models?*. NVIDIA. 2024-04. COLM 2024, arXiv:2404.06654. https://arxiv.org/abs/2404.06654
- <a id="infinitebench"></a>`[InfiniteBench]` *InfiniteBench: 200K+ token tasks*. Zhang et al. 2024.
- <a id="longbench-v2"></a>`[LongBench-v2]` *LongBench v2*. 2024. https://aclanthology.org/2025.findings-acl.903.pdf
- <a id="mrcr"></a>`[MRCR]` *Multi-Round Co-reference Reasoning*. (Anthropic/OpenAI variants).
- <a id="twilight"></a>`[Twilight]` *Twilight: Hierarchical Top-p Adaptive Sparsity*. NeurIPS 2025 Spotlight. https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/twilight.neurips25.pdf
- <a id="mamba-3"></a>`[Mamba-3]` *Mamba-3: A Universal Architecture for State-Space Models*. (lineage promoted 2026-05-05 from emerging to canonical). 2026-03. ICLR 2026 oral, arXiv:2603.15569. https://arxiv.org/abs/2603.15569
  - The latest canonical SSM reference; relevant to long-context (`20/04`) and attention-variants (`30/03`) chapters.
- <a id="nosa"></a>`[NOSA]` *NOSA: Native and Offloadable Sparse Attention*. 2025-10. arXiv:2510.13602. https://arxiv.org/pdf/2510.13602
- <a id="rocketkv"></a>`[RocketKV]` *RocketKV: Two-stage KV compression at long ctx*. ICML 2025. arXiv:2502.14051. https://arxiv.org/abs/2502.14051
- <a id="unsloth-flex"></a>`[Unsloth-Flex]` *Long Context gpt-oss Training*. Unsloth. 2025. https://docs.unsloth.ai/models/gpt-oss-how-to-run-and-fine-tune/long-context-gpt-oss-training
- <a id="scbench-long"></a>`[SCBench-Long]` See `[SCBench]` above.
- <a id="hisparse"></a>`[HiSparse]` *HiSparse: SGLang CPU-offloaded sparse-attn KV*. LMSYS. 2026.
- <a id="sglang-pipeline-pp"></a>`[SGLang-Pipeline-PP]` *Chunked Pipeline Parallelism for Ultra-Long Context*. SGLang. 2026-01.

### Heterogeneous inference

#### Lineage

- <a id="zero-inference"></a>`[ZeRO-Inference]` *ZeRO-Inference (DeepSpeed blog)*. Aminabadi et al. (Microsoft). 2022-09. https://www.deepspeed.ai/2022/09/09/zero-inference.html
- <a id="petals"></a>`[Petals]` *Petals: Collaborative Inference and Fine-tuning of Large Models*. Borzunov et al. 2022-09. ACL 2023 demo, arXiv:2209.01188. https://arxiv.org/abs/2209.01188
- <a id="powerinfer"></a>`[PowerInfer]` *PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU*. Song, Xie, Zhang et al. (SJTU-IPADS). 2023-12. SOSP 2024, arXiv:2312.12456. https://arxiv.org/abs/2312.12456
  - Hot/cold neuron split via power-law activation; 11.69× over llama.cpp on RTX 4090.
- <a id="powerinfer-2"></a>`[PowerInfer-2]` *PowerInfer-2: Fast LLM Inference on a Smartphone*. Xue, Song, Chen et al. (SJTU-IPADS). 2024-06. arXiv:2406.06282. https://arxiv.org/abs/2406.06282
- <a id="ktransformers"></a>`[KTransformers]` *KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models*. Chen et al. (Tsinghua/kvcache-ai). 2025-10. SOSP 2025. https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/
  - DeepSeek-R1 671B on single 4090D + DRAM at ~14 tok/s decode.
- <a id="hexgen"></a>`[HexGen]` *HexGen: Generative Inference of LLM over Heterogeneous Environment*. Jiang, Yan, Yuan. 2023-11. ICML 2024, arXiv:2311.11514. https://arxiv.org/abs/2311.11514
- <a id="llm-pq"></a>`[LLM-PQ]` *LLM-PQ: Phase-Aware Partition and Adaptive Quantization*. Zhao et al. 2024-03. PPoPP 2024, arXiv:2403.01136. https://arxiv.org/abs/2403.01136
- <a id="melange"></a>`[Mélange]` *Mélange: Cost Efficient Large Language Model Serving by Exploiting GPU Heterogeneity*. Griggs, Liu, Ren, Patel, Stoica (UCB/Microsoft). 2024-04. arXiv:2404.14527. https://arxiv.org/abs/2404.14527
- <a id="alpaserve"></a>`[AlpaServe]` *AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving*. Li, Zheng, Zhuang, Yu, Stoica et al. OSDI 2023. https://www.usenix.org/conference/osdi23/presentation/li-zhuohan
- <a id="spotserve"></a>`[SpotServe]` *SpotServe: Serving Generative Large Language Models on Preemptible Instances*. Miao et al. ASPLOS 2024. arXiv:2311.15566. https://arxiv.org/abs/2311.15566
- <a id="serverlessllm"></a>`[ServerlessLLM]` *ServerlessLLM: Low-Latency Serverless Inference for LLMs*. Fu, Xue, Huang et al. (Edinburgh). OSDI 2024. https://www.usenix.org/conference/osdi24/presentation/fu
- <a id="greenllm"></a>`[GreenLLM]` *GreenLLM: Disaggregating LLM Serving on Heterogeneous GPUs for Lower Carbon Emissions*. Shi, Wu, Liu, Ding. 2024-12. arXiv:2412.20322. https://arxiv.org/abs/2412.20322

#### Past 12 months SOTA

- <a id="waferllm"></a>`[WaferLLM]` *WaferLLM: Large Language Model Inference at Wafer Scale*. He et al. 2025-02. OSDI 2025, arXiv:2502.04563. https://arxiv.org/abs/2502.04563
- <a id="hexgen-2"></a>`[HexGen-2]` *HexGen-2: Disaggregated Generative Inference of LLMs in Heterogeneous Environment*. Jiang, Yan, Yuan. 2025-02. ICLR 2025, arXiv:2502.07903. https://arxiv.org/abs/2502.07903
  - 2.0× throughput / 1.5× latency over SOTA at same price.
- <a id="demystify-costeff"></a>`[Demystify-CostEff]` *Demystifying Cost-Efficiency in LLM Serving over Heterogeneous GPUs*. Jiang, Fu, Yao, He, Miao, Klimovic, Cui, Yuan, Yoneki. 2025-02. ICML 2025, arXiv:2502.00722. https://arxiv.org/abs/2502.00722
- <a id="sageserve"></a>`[SageServe]` *Serving Models, Fast and Slow: Optimizing Heterogeneous LLM Inferencing Workloads at Scale*. Microsoft + UIUC + IISc. 2025-02. arXiv:2502.14617. https://arxiv.org/abs/2502.14617
  - 25% GPU-hour savings; ~$2M/month at provider scale.
- <a id="prism"></a>`[Prism]` *Prism: Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving*. 2025-05. arXiv:2505.04021. https://arxiv.org/abs/2505.04021
- <a id="hexgen-flow"></a>`[HEXGEN-FLOW]` *Optimizing LLM Inference Request Scheduling for Agentic Text-to-SQL*. Relaxed System Lab. 2025-05. arXiv:2505.05286. https://arxiv.org/abs/2505.05286
- <a id="hetis"></a>`[Hetis]` *Hetis: Serving LLMs in Heterogeneous GPU Clusters with Fine-grained and Dynamic Parallelism*. Mo, Liao, Xu, Zhou, Xu (UMacau). 2025-09. SC 2025, arXiv:2509.08309. https://arxiv.org/abs/2509.08309
- <a id="parallax"></a>`[Parallax]` *Parallax: Efficient LLM Inference Service over Decentralized Environment*. GradientHQ. 2025-09. arXiv:2509.26182. https://arxiv.org/abs/2509.26182
- <a id="cauchy"></a>`[Cauchy]` *Cauchy: Cost-Efficient LLM Serving through Adaptive Heterogeneous Deployment*. SoCC 2025. https://dl.acm.org/doi/10.1145/3772052.3772264
- <a id="neuronmm"></a>`[NeuronMM]` *NeuronMM: High-Performance Matrix Multiplication for LLM Inference on AWS Trainium*. 2025-10. arXiv:2510.25977. https://arxiv.org/abs/2510.25977
- <a id="flexllm-hetero"></a>`[FlexLLM-hetero]` *FlexLLM: Flexible and Cost-Efficient LLM Serving with Heterogeneous GPUs*. Kim et al. MASCOTS 2025. https://discos.sogang.ac.kr/file/2025/intl_conf/MASCOTS_2025_K_Kim.pdf
- <a id="hetero-mm"></a>`[Hetero-MM]` *Cost-Efficient Multimodal LLM Inference via Cross-Tier GPU Heterogeneity*. Yu. 2026-03. arXiv:2603.12707. https://arxiv.org/abs/2603.12707
- <a id="tessera"></a>`[Tessera]` *Tessera: Unlocking Heterogeneous GPUs through Kernel-Granularity Disaggregation*. 2026-04. arXiv:2604.10180. https://arxiv.org/abs/2604.10180
- <a id="fasthetero"></a>`[FastHetero]` *Fast Heterogeneous Serving: Scalable Mixed-Scale LLM Allocation*. 2026-04. arXiv:2604.07472. https://arxiv.org/abs/2604.07472
- <a id="edge-cloud-survey"></a>`[Edge-Cloud-Survey]` *Collaborative Inference and Learning between Edge SLMs and Cloud LLMs: A Survey*. 2025-07. arXiv:2507.16731. https://arxiv.org/abs/2507.16731
- <a id="sambanova-sn40l"></a>`[SambaNova-SN40L]` *SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts*. Prabhakar et al. 2024-05. arXiv:2405.07518. https://arxiv.org/html/2405.07518v1
- <a id="mlx"></a>`[MLX]` *Apple MLX framework*. Apple. https://machinelearning.apple.com/research/exploring-llms-mlx-m5

### LoRA and multi-tenant serving

#### Lineage

- <a id="lora"></a>`[LoRA]` *LoRA: Low-Rank Adaptation of Large Language Models*. Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen (Microsoft). 2021-06. ICLR 2022, arXiv:2106.09685. https://arxiv.org/abs/2106.09685
  - Defines W + (α/r)BA adapter; premise of the entire serving stack.
- <a id="ia3"></a>`[IA3]` *Few-Shot PEFT is Better and Cheaper than In-Context Learning (T-Few/IA³)*. Liu et al. 2022-05. NeurIPS 2022, arXiv:2205.05638. https://arxiv.org/abs/2205.05638
- <a id="lorahub"></a>`[LoraHub]` *LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition*. Huang et al. 2023-07. COLM 2024, arXiv:2307.13269. https://arxiv.org/abs/2307.13269
- <a id="punica"></a>`[Punica]` *Punica: Multi-Tenant LoRA Serving*. Chen, Ye, Jiang, Cao, Yang, Krishnamurthy (UW/Duke). 2023-10. MLSys 2024, arXiv:2310.18547. https://arxiv.org/abs/2310.18547
  - BGMV / SGMV kernels; first to batch heterogeneous-adapter requests in one launch.
- <a id="vera"></a>`[VeRA]` *VeRA: Vector-based Random Matrix Adaptation*. Kopiczko, Blankevoort, Asano. 2023-10. ICLR 2024, arXiv:2310.11454. https://arxiv.org/abs/2310.11454
- <a id="s-lora"></a>`[S-LoRA]` *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*. Sheng, Cao, Li, Hooper, Ho, Zhuo, Kasikci, Stoica et al. 2023-11. MLSys 2024, arXiv:2311.03285. https://arxiv.org/abs/2311.03285
  - Unified Paging memory pool; up to 4× over vLLM-naive.
- <a id="lorax"></a>`[LoRAX]` *LoRA eXchange: Serve 100s of fine-tuned LLMs*. Predibase. 2023-11. https://github.com/predibase/lorax
  - First production-grade open-source multi-LoRA server.
- <a id="mlora"></a>`[mLoRA]` *mLoRA: Fine-Tuning LoRA Adapters via Highly-Efficient Pipeline Parallelism*. Tang et al. (TUDB-Labs/Ant). 2023-12. VLDB 2025, arXiv:2312.02515. https://arxiv.org/abs/2312.02515
- <a id="flora-batched"></a>`[FLoRA-Batched]` *Batched Low-Rank Adaptation of Foundation Models*. Wen, Chaudhuri. 2023-12. ICLR 2024, arXiv:2312.05677. https://arxiv.org/abs/2312.05677
- <a id="caraserve"></a>`[CaraServe]` *CaraServe: CPU-Assisted and Rank-Aware LoRA Serving*. Li, Lu, Weng, Wu et al. 2024-01. arXiv:2401.11240. https://arxiv.org/abs/2401.11240
- <a id="dora"></a>`[DoRA]` *DoRA: Weight-Decomposed Low-Rank Adaptation*. Liu, Wang, Yin, Molchanov, Wang, Cheng, Chen (NVIDIA). 2024-02. ICML 2024 Oral, arXiv:2402.09353. https://arxiv.org/abs/2402.09353
- <a id="dlora"></a>`[dLoRA]` *dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving*. Wu, Zhu, Zhang, Sun, Liu, Jin (PKU/Shanghai AI Lab). 2024-07. OSDI 2024. https://www.usenix.org/conference/osdi24/presentation/wu-bingyang
  - Credit-based merge/unmerge batching; up to 1.8× over S-LoRA.

#### Past 12 months SOTA

- <a id="elora"></a>`[ELORA]` *Improving the Serving Performance of Multi-LoRA LLMs via Efficient LoRA and KV Cache Management*. Zhang, Shi et al. (HKUST). 2025-05. HPCA 2026, arXiv:2505.03756. https://arxiv.org/abs/2505.03756
  - Treats LoRA cache and KV cache as one dependency-aware pool; -63% TTFT.
- <a id="serverlesslora"></a>`[ServerlessLoRA]` *ServerlessLoRA: Minimizing Latency and Cost in Serverless Inference for LoRA-Based LLMs*. 2025-05. arXiv:2505.14468. https://arxiv.org/abs/2505.14468
  - 86% TTFT reduction, 89% cost reduction in serverless.
- <a id="edgelora"></a>`[EdgeLoRA]` *EdgeLoRA: An Efficient Multi-Tenant LLM Serving System on Edge Devices*. 2025-07. MobiSys 2025, arXiv:2507.01438. https://arxiv.org/abs/2507.01438
  - Adaptive adapter selection; 4× throughput vs llama.cpp.
- <a id="toppings"></a>`[Toppings]` *Toppings: CPU-Assisted, Rank-Aware Adapter Serving for LLM Inference*. Li, Lu, Weng et al. 2025-07. USENIX ATC 2025. https://www.usenix.org/conference/atc25/presentation/li-suyi-toppings
- <a id="equinox"></a>`[Equinox]` *Equinox: Holistic Fair Scheduling in Serving Large Language Models*. Wei et al. 2025-08. arXiv:2508.16646. https://arxiv.org/abs/2508.16646
  - Dual-counter (User + Resource) fairness; 1.3× throughput, 60% lower TTFT vs VTC.
- <a id="k-merge"></a>`[K-Merge]` *K-Merge: Online Continual Merging of Adapters for On-device LLMs*. 2025-10. arXiv:2510.13537. https://arxiv.org/abs/2510.13537
- <a id="zflora"></a>`[zFLoRA]` *zFLoRA: Zero-Latency Fused Low-Rank Adapters*. Gowda, Song et al. (Samsung). 2025-10. EMNLP 2025, arXiv:2510.25784. https://arxiv.org/abs/2510.25784
- <a id="l-moe"></a>`[L-MoE]` *L-MoE: Gating Network Composes Adapter Parameters per Token*. 2025-10. arXiv:2510.17898. https://arxiv.org/abs/2510.17898
- <a id="fused-flora"></a>`[Fused-FLoRA]` *FLoRA: Fused forward-backward adapters for parameter efficient fine-tuning*. 2025-11. arXiv:2511.00050. https://arxiv.org/abs/2511.00050
- <a id="logo"></a>`[LoGo]` *LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging*. 2025-11. arXiv:2511.07129. https://arxiv.org/abs/2511.07129
- <a id="loraserve"></a>`[LoRAServe]` *Serving Heterogeneous LoRA Adapters in Distributed LLM Inference Systems*. Jaiswal, Arun, Parayil, Mallick, Mastorakis et al. 2025-11. arXiv:2511.22880. https://arxiv.org/abs/2511.22880
  - Rank-induced performance skew + GPUDirect RDMA; 2× throughput, 9× lower TTFT.
- <a id="p-lora"></a>`[P-LoRA]` *Predictive-LoRA: A Proactive and Fragmentation-Aware Serverless Inference System*. 2025-12. arXiv:2512.20210. https://arxiv.org/abs/2512.20210
  - LSTM traffic predictor + page-based memory mgr; 1.52× throughput vs S-LoRA.
- <a id="adafuse"></a>`[AdaFuse]` *AdaFuse: Accelerating Dynamic Adapter Inference via Token-Level Pre-Gating*. 2026-03. arXiv:2603.11873. https://arxiv.org/abs/2603.11873
  - Adapter overhead reduced from 250–950% to ~29%.
- <a id="turbo-lora"></a>`[Turbo-LoRA]` *Turbo-LoRA: Joint LoRA + Draft-Head Training*. Predibase. 2024. https://predibase.com/blog/turbo-lora
- <a id="tt-lora-moe"></a>`[TT-LoRA-MoE]` *TT-LoRA MoE: Sparse-MoE Router Selects One Adapter Per Input*. SC 2025. https://dl.acm.org/doi/10.1145/3712285.3759888
- <a id="x-lora"></a>`[X-LoRA]` *X-LoRA*. https://github.com/EricLBuehler/xlora
- <a id="nim-lora"></a>`[NIM-LoRA]` *Seamlessly Deploying a Swarm of LoRA Adapters with NVIDIA NIM*. NVIDIA. https://developer.nvidia.com/blog/seamlessly-deploying-a-swarm-of-lora-adapters-with-nvidia-nim/
- <a id="cf-workers-lora"></a>`[CF-Workers-LoRA]` *Fine-tuned inference with LoRAs on Cloudflare Workers*. https://blog.cloudflare.com/fine-tuned-inference-with-loras/
- <a id="sagemaker-lora"></a>`[SageMaker-LoRA]` *Efficient and cost-effective multi-tenant LoRA serving with Amazon SageMaker*. https://aws.amazon.com/blogs/machine-learning/efficient-and-cost-effective-multi-tenant-lora-serving-with-amazon-sagemaker/
- <a id="together-multi-lora"></a>`[Together-Multi-LoRA]` *Together AI Serverless Multi-LoRA*. https://www.together.ai/blog/serverless-multi-lora-fine-tune-and-deploy-hundreds-of-adapters-for-model-customization-at-scale
- <a id="friendli-lora"></a>`[Friendli-LoRA]` *Multi-LoRA on Friendli Container*. https://friendli.ai/docs/guides/container/serving_multi_lora_models
- <a id="hf-peft"></a>`[HF-PEFT]` *HuggingFace PEFT*. https://github.com/huggingface/peft

### Cluster-level systems (router, gateway, autoscaling, observability)

#### Lineage

- <a id="gie-blog"></a>`[GIE-blog]` *Introducing Gateway API Inference Extension*. Kubernetes blog. 2025-06. https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/
- <a id="gie-repo"></a>`[GIE-repo]` *kubernetes-sigs/gateway-api-inference-extension*. https://github.com/kubernetes-sigs/gateway-api-inference-extension
  - InferencePool v1; Endpoint Picker Protocol.
- <a id="gie-docs"></a>`[GIE-docs]` *Kubernetes GIE docs*. https://gateway-api-inference-extension.sigs.k8s.io/
- <a id="istio-gie"></a>`[Istio-GIE]` *Istio Gateway API Inference Extension Support*. Istio blog. 2025. https://istio.io/latest/blog/2025/inference-extension-support/

#### Past 12 months SOTA — control planes

- <a id="llm-d-launch"></a>`[llm-d-launch]` *llm-d: Kubernetes-native distributed inferencing*. Red Hat. 2025-05. https://developers.redhat.com/articles/2025/05/20/llm-d-kubernetes-native-distributed-inferencing
- <a id="llm-d-cncf"></a>`[llm-d-CNCF]` *Donating llm-d to CNCF*. IBM Research. 2025. https://research.ibm.com/blog/donating-llm-d-to-the-cloud-native-computing-foundation
- <a id="llm-d-v0-5"></a>`[llm-d-v0.5]` *llm-d 0.5: Sustaining Performance at Scale*. 2026-02. https://llm-d.ai/blog/llm-d-v0.5-sustaining-performance-at-scale
  - Scale-to-zero with activator; UCCL transport (2.4× more congestion resilience).
- <a id="llm-d-epp-arch"></a>`[llm-d-EPP-arch]` *llm-d Inference Scheduler Architecture*. https://github.com/llm-d/llm-d-inference-scheduler/blob/main/docs/architecture.md
- <a id="llm-d-prefix-precise"></a>`[llm-d-prefix-precise]` *Precise Prefix Cache Aware Routing*. https://github.com/llm-d/llm-d/blob/main/guides/precise-prefix-cache-aware/README.md
- <a id="llm-d-pred-latency"></a>`[llm-d-pred-latency]` *Predicted-Latency Based Scheduling for LLMs*. https://llm-d.ai/blog/predicted-latency-based-scheduling-for-llms
- <a id="wide-ep-llmd"></a>`[Wide-EP-llmd]` *Scaling DeepSeek-style MoEs with vLLM and llm-d using Wide EP*. Red Hat. 2025-09. https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep
- <a id="aibrix-launch"></a>`[AIBrix-launch]` *Introducing AIBrix*. vLLM blog. 2025-02. https://blog.vllm.ai/2025/02/21/aibrix-release.html
- <a id="aibrix-v0-3"></a>`[AIBrix-v0.3]` *AIBrix v0.3.0: KVCache Offloading, Prefix Cache, Fairness Routing*. 2025-05. https://aibrix.github.io/posts/2025-05-21-v0.3.0-release/
- <a id="aibrix-v0-4"></a>`[AIBrix-v0.4]` *AIBrix v0.4.0: P/D Disaggregation, EP, KVCache v1, KV Event Sync*. 2025-08. https://aibrix.github.io/posts/2025-08-04-v0.4.0-release/
- <a id="aibrix-arch"></a>`[AIBrix-arch]` *AIBrix Architecture*. https://aibrix.readthedocs.io/latest/designs/architecture.html
- <a id="kserve-oip"></a>`[KServe-OIP]` *Open Inference Protocol (V2)*. KServe. https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol
- <a id="kserve-cncf"></a>`[KServe-CNCF]` *KServe joins CNCF as Incubating project*. 2025-11. https://kserve.github.io/website/
- <a id="kserve-llmd"></a>`[KServe-llmd]` *Combining KServe and llm-d for optimized generative AI inference*. Red Hat. 2026-04. https://developers.redhat.com/articles/2026/04/21/kserve-llm-d-optimized-gen-ai-inference
- <a id="aegaeon"></a>`[Aegaeon]` *Aegaeon: Effective GPU Pooling for Concurrent LLM Serving on the Market*. Xiang et al. (Alibaba). SOSP 2025. https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf
  - Token-granularity model auto-scaling; 1192 → 213 GPUs in real model marketplace.
- <a id="adaserve"></a>`[AdaServe]` *AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding*. CMU/Princeton/Purdue. EuroSys 2026, arXiv:2501.12162. https://arxiv.org/abs/2501.12162
  - 4.3× SLO-violation reduction.
- <a id="polyserve"></a>`[PolyServe]` *PolyServe: Efficient Multi-SLO Serving at Scale*. arXiv:2507.17769. https://arxiv.org/html/2507.17769
- <a id="brownoutserve"></a>`[BrownoutServe]` *BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads*. arXiv:2507.17133. https://arxiv.org/pdf/2507.17133
- <a id="dualmap"></a>`[DualMap]` *DualMap: Enabling Both Cache Affinity and Load Balancing*. arXiv:2602.06502. https://arxiv.org/abs/2602.06502
- <a id="routerwise"></a>`[RouterWise]` *RouterWise: Joint Resource Allocation and Routing for Latency-Aware Multi-Model LLM Serving*. arXiv:2604.10907. https://arxiv.org/abs/2604.10907
- <a id="kv-event-sync-aibrix"></a>`[KV-event-sync-AIBrix]` AIBrix KV Event Subscription System (v0.4.0).
- <a id="uccl-llm-d"></a>`[UCCL-llm-d]` UCCL host-resident transport in llm-d 0.5.

#### Engine-level routers / gateways

- <a id="vllm-router"></a>`[vLLM-router]` *vLLM Router: A High-Performance and Prefill/Decode Aware Load Balancer*. vLLM blog. 2025-12. https://blog.vllm.ai/2025/12/13/vllm-router-release.html
- <a id="vllm-prod-stack"></a>`[vLLM-prod-stack]` *vLLM Production Stack*. https://github.com/vllm-project/production-stack
- <a id="sgl-router"></a>`[sgl-router]` *SGLang sgl-router*. https://github.com/sgl-project/sglang
- <a id="envoyai-launch"></a>`[EnvoyAI-launch]` *Tetrate and Bloomberg Release Open Source Envoy AI Gateway*. Tetrate. 2025-02. https://tetrate.io/press/tetrate-and-bloomberg-release-open-source-envoy-ai-gateway-built-on-cncfs-envoy-gateway-project
- <a id="envoyai-site"></a>`[EnvoyAI-site]` *Envoy AI Gateway*. https://aigateway.envoyproxy.io/
- <a id="cncf-genai-platform"></a>`[CNCF-genai-platform]` *Building a scalable, flexible, cloud-native GenAI platform with open source*. CNCF. 2025-08. https://www.cncf.io/blog/2025/08/28/building-a-scalable-flexible-cloud-native-genai-platform-with-open-source-solutions/
- <a id="litellm"></a>`[LiteLLM]` *BerriAI/litellm*. https://github.com/BerriAI/litellm
- <a id="portkey"></a>`[Portkey]` *Portkey AI Gateway*. https://portkey.ai/buyers-guide/ai-gateway-solutions
- <a id="bifrost"></a>`[Bifrost]` *Maxim AI Bifrost*. https://github.com/maximhq/bifrost
- <a id="kong-aigw"></a>`[Kong-AIGW]` *Kong AI Gateway*. https://developer.konghq.com/ai-gateway/

#### Observability

- <a id="otel-genai"></a>`[OTel-GenAI]` *Semantic conventions for generative AI systems*. OpenTelemetry. https://opentelemetry.io/docs/specs/semconv/gen-ai/
- <a id="otel-genai-spans"></a>`[OTel-GenAI-spans]` *Semantic conventions for generative client AI spans*. https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
- <a id="otel-agent"></a>`[OTel-agent]` *GenAI agent and framework spans*. https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/
- <a id="vllm-metrics"></a>`[vLLM-metrics]` *vLLM Metrics*. https://docs.vllm.ai/en/stable/design/metrics/
- <a id="datadog-otel-genai"></a>`[Datadog-OTel-GenAI]` *Datadog LLM Observability OTel GenAI Semantic Conventions*. https://www.datadoghq.com/blog/llm-otel-semantic-convention/
- <a id="llm-d-observability"></a>`[llm-d-observability]` *Monitoring and Observability with llm-d*. https://llm-d.ai/docs/usage/monitoring

#### Managed runtimes

- <a id="rayserve-llm"></a>`[RayServe-LLM]` *Ray Serve LLM architecture*. https://docs.ray.io/en/latest/serve/llm/architecture/overview.html
- <a id="rayserve-async"></a>`[RayServe-async]` *Ray Serve: Async Inference, Custom Routing, Custom Autoscaling*. https://www.anyscale.com/blog/ray-serve-autoscaling-async-inference-custom-routing
- <a id="rayserve-prefix"></a>`[RayServe-prefix]` *Prefix-aware routing in Ray Serve*. https://docs.ray.io/en/latest/serve/llm/user-guides/prefix-aware-routing.html
- <a id="anyscale-llm"></a>`[Anyscale-LLM]` *Anyscale LLM Suite*. https://www.anyscale.com/product/platform/llm-suite
- <a id="bentoml-openllm"></a>`[BentoML-OpenLLM]` *bentoml/OpenLLM*. https://github.com/bentoml/OpenLLM
- <a id="modal"></a>`[Modal]` *Modal*. https://modal.com/
- <a id="modal-almanac"></a>`[Modal-Almanac]` *LLM Engineer's Almanac*. https://modal.com/llm-almanac/summary
- <a id="bedrock-cris"></a>`[Bedrock-CRIS]` *Cross-Region Inference for Anthropic Claude on Amazon Bedrock*. AWS. https://aws.amazon.com/blogs/machine-learning/unlock-global-ai-inference-scalability-using-new-global-cross-region-inference-on-amazon-bedrock-with-anthropics-claude-sonnet-4-5/
- <a id="bedrock-opus47"></a>`[Bedrock-Opus47]` *Introducing Anthropic's Claude Opus 4.7 model in Amazon Bedrock*. https://aws.amazon.com/blogs/aws/introducing-anthropics-claude-opus-4-7-model-in-amazon-bedrock/
- <a id="helicone-survey"></a>`[Helicone-Survey]` *Top 11 LLM API Providers in 2025*. Helicone. https://www.helicone.ai/blog/llm-api-providers
- <a id="anyscale-vllm-wideep"></a>`[Anyscale-vLLM-WideEP]` *Ray Serve LLM, Anyscale APIs, Wide-EP, Disaggregated Serving*. https://www.anyscale.com/blog/ray-serve-llm-anyscale-apis-wide-ep-disaggregated-serving-vllm

### Hardware (NVIDIA, AMD, ASIC, networking)

#### NVIDIA roadmap

- <a id="nv-h100-wp"></a>`[NV-H100-WP]` *NVIDIA H100 Tensor Core GPU Architecture (whitepaper)*. NVIDIA. 2022-03. https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c
- <a id="nv-hopper-blog"></a>`[NV-Hopper-Blog]` *NVIDIA Hopper Architecture In-Depth*. 2022-03. https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- <a id="nv-h200-page"></a>`[NV-H200-Page]` *NVIDIA H200 Tensor Core GPU*. 2024. https://www.nvidia.com/en-us/data-center/h200/
- <a id="nv-blackwell-arch"></a>`[NV-Blackwell-Arch]` *NVIDIA Blackwell Architecture*. NVIDIA. 2024–25. https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/
- <a id="nv-blackwell-pr"></a>`[NV-Blackwell-PR]` *NVIDIA Blackwell Platform Arrives to Power a New Era*. 2024-03. https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing
- <a id="nv-gb200-page"></a>`[NV-GB200-Page]` *NVIDIA GB200 NVL72*. https://www.nvidia.com/en-us/data-center/gb200-nvl72/
- <a id="nv-gb300-page"></a>`[NV-GB300-Page]` *NVIDIA GB300 NVL72*. https://www.nvidia.com/en-us/data-center/gb300-nvl72/
- <a id="nv-blackwell-ultra"></a>`[NV-Blackwell-Ultra]` *Inside NVIDIA Blackwell Ultra*. 2025-03. https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
- <a id="introl-b300"></a>`[Introl-B300]` *NVIDIA Blackwell Ultra and B300 Infrastructure Requirements*. 2025. https://introl.com/blog/nvidia-blackwell-ultra-b300-infrastructure-requirements-2025
- <a id="cw-gb300"></a>`[CW-GB300]` *CoreWeave Launches GB300 NVL72-Powered Cloud Instances*. 2025-08. https://docs.coreweave.com/docs/changelog/release-notes/gb300-nvl72
- <a id="nv-fy26q3"></a>`[NV-FY26Q3]` *NVIDIA Q3 FY 2026 Financial Results*. 2025-11. https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-third-quarter-fiscal-2026
- <a id="nv-vera-rubin-blog"></a>`[NV-Vera-Rubin-Blog]` *Inside the NVIDIA Vera Rubin Platform: Six New Chips*. 2026. https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/
- <a id="nv-rubin-wiki"></a>`[NV-Rubin-Wiki]` *Rubin (microarchitecture)*. https://en.wikipedia.org/wiki/Rubin_(microarchitecture)
- <a id="sth-rubin"></a>`[STH-Rubin]` *NVIDIA Launches Next-Generation Rubin AI Compute Platform at CES 2026*. 2026-01. https://www.servethehome.com/nvidia-launches-next-generation-rubin-ai-compute-platform-at-ces-2026/
- <a id="dcd-rubinultra"></a>`[DCD-RubinUltra]` *Nvidia's Rubin Ultra NVL576 rack expected to be 600kW*. 2025. https://www.datacenterdynamics.com/en/news/nvidias-rubin-ultra-nvl576-rack-expected-to-be-600kw-coming-second-half-of-2027/
- <a id="nv-rubincpx-blog"></a>`[NV-RubinCPX-Blog]` *NVIDIA Rubin CPX Accelerates 1M+ Token Context Workloads*. 2025-09. https://developer.nvidia.com/blog/nvidia-rubin-cpx-accelerates-inference-performance-and-efficiency-for-1m-token-context-workloads/
- <a id="reg-rubincpx"></a>`[Reg-RubinCPX]` *Nvidia's context-optimized Rubin CPX GPUs were inevitable*. 2025-09. https://www.theregister.com/2025/09/10/nvidia_rubin_cpx/
- <a id="nv-groq3lpx"></a>`[NV-Groq3LPX]` *Inside NVIDIA Groq 3 LPX*. 2026-03. https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/
- <a id="decoder-lpx"></a>`[Decoder-LPX]` *GTC 2026: With Groq 3 LPX, Nvidia adds dedicated inference hardware*. 2026-03. https://the-decoder.com/gtc-2026-with-groq-3-lpx-nvidia-adds-dedicated-inference-hardware-to-its-platform-for-the-first-time/
- <a id="toms-nv-groq"></a>`[Toms-NV-Groq]` *How Nvidia's $20 billion Groq 3 LPU deal reshapes Vera Rubin*. 2026. https://www.tomshardware.com/tech-industry/semiconductors/nvidias-20-billion-groq-deal-produces-its-first-chip
- <a id="nv-dgx-spark"></a>`[NV-DGX-Spark]` *NVIDIA DGX Spark*. https://www.nvidia.com/en-us/products/workstations/dgx-spark/
- <a id="nv-jetson-thor"></a>`[NV-Jetson-Thor]` *NVIDIA Jetson AGX Thor benchmarks*. 2025-10. https://jetsonhacks.com/2025/10/31/nvidia-jetson-agx-thor-vs-dgx-spark-benchmarks/
- <a id="np-verarubin"></a>`[NP-VeraRubin]` *Nvidia's Vera-Rubin Platform Obsoletes Current AI Iron*. 2026-01. https://www.nextplatform.com/ai/2026/01/06/nvidias-vera-rubin-platform-obsoletes-current-ai-iron-six-months-ahead-of-launch/4092179
- <a id="np-nv-groq"></a>`[NP-NV-Groq]` *Nvidia Finally Admits Why It Shelled Out $20B For Groq*. 2026-03. https://www.nextplatform.com/ai/2026/03/17/nvidia-finally-admits-why-it-shelled-out-20-billion-for-groq/5209495
- <a id="hopper-ms"></a>`[Hopper-MS]` *Comparing Blackwell vs Hopper*. https://www.exxactcorp.com/blog/hpc/comparing-nvidia-tensor-core-gpus
- <a id="asplos-blackwell"></a>`[ASPLOS-Blackwell]` *Microbenchmarking NVIDIA's Blackwell Architecture*. 2025. arXiv:2512.02189. https://arxiv.org/html/2512.02189v3
- <a id="nv-mlperf5-0"></a>`[NV-MLPerf5.0]` *NVIDIA Blackwell in MLPerf Inference v5.0*. 2025. https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/
- <a id="nv-mlperf5-1"></a>`[NV-MLPerf5.1]` *NVIDIA Blackwell Ultra Sets New Inference Records in MLPerf Debut*. 2025. https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/
- <a id="hpcwire-mlperf5-1"></a>`[HPCwire-MLPerf5.1]` *MLPerf Inference v5.1 Results Land*. 2025-09. https://www.hpcwire.com/2025/09/10/mlperf-inference-v5-1-results-land-with-new-benchmarks-and-record-participation/

#### AMD

- <a id="amd-cdna4-wp"></a>`[AMD-CDNA4-WP]` *Introducing AMD CDNA 4 Architecture (whitepaper)*. AMD. 2025. https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
- <a id="amd-mi350-blog"></a>`[AMD-MI350-Blog]` *AMD Instinct MI350 Series and Beyond*. 2025. https://www.amd.com/en/blogs/2025/amd-instinct-mi350-series-and-beyond-accelerating-the-future-of-ai-and-hpc.html
- <a id="amd-mi350-game"></a>`[AMD-MI350-Game]` *MI350 Series GPUs: A Game Changer*. 2025. https://www.amd.com/en/blogs/2025/amd-instinct-mi350-series-game-changer.html
- <a id="sth-cdna4"></a>`[STH-CDNA4]` *AMD Dives Deep on CDNA 4 at Hot Chips 2025*. 2025-08. https://www.servethehome.com/amd-dives-deep-on-cdna-4-architecture-and-mi350-accelerator-at-hot-chips-2025/
- <a id="toms-mi400"></a>`[Toms-MI400]` *AMD touts Instinct MI430X / MI440X / MI455X and Helios*. 2026-01. https://www.tomshardware.com/tech-industry/artificial-intelligence/amd-touts-instinct-mi430x-mi440x-and-mi455x-ai-accelerators-and-helios-rack-scale-ai-architecture-at-ces-full-mi400-series-family-fulfills-a-broad-range-of-infrastructure-and-customer-requirements
- <a id="semia-mi400"></a>`[SemiA-MI400]` *AMD Advancing AI: MI350X and MI400 UALoE72, MI500 UAL256*. 2025-06. https://semianalysis.com/2025/06/13/amd-advancing-ai-mi350x-and-mi400-ualoe72-mi500-ual256/
- <a id="amd-mlperf6"></a>`[AMD-MLPerf6]` *AMD Delivers Breakthrough MLPerf Inference 6.0 Results*. 2026. https://www.amd.com/en/blogs/2026/amd-delivers-breakthrough-mlperf-inference-6-0-results.html
- <a id="rocm-vllm-first"></a>`[ROCm-vLLM-First]` *ROCm Becomes a First-Class Platform in the vLLM Ecosystem*. 2025. https://rocm.blogs.amd.com/software-tools-optimization/vllm-omni/README.html
- <a id="rocm-vllm-doc"></a>`[ROCm-vLLM-Doc]` *vLLM inference - ROCm Documentation*. https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/vllm.html
- <a id="rocm-vllm-opt"></a>`[ROCm-vLLM-Opt]` *vLLM V1 performance optimization on ROCm*. https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html

#### ASIC ecosystem

- <a id="gcloud-tpuv7"></a>`[GCloud-TPUv7]` *TPU7x (Ironwood)*. 2025. https://docs.cloud.google.com/tpu/docs/tpu7x
- <a id="google-iron-blog"></a>`[Google-Iron-Blog]` *Ironwood: The first Google TPU for the age of inference*. 2025. https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/
- <a id="google-iron-stack"></a>`[Google-Iron-Stack]` *Inside the Ironwood TPU codesigned AI stack*. 2025-11. https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack
- <a id="gcloud-trillium"></a>`[GCloud-Trillium]` *Trillium TPU v6e in preview*. 2024. https://cloud.google.com/blog/products/compute/trillium-sixth-generation-tpu-is-in-preview
- <a id="gcloud-v5e"></a>`[GCloud-v5e]` *TPU v5e*. https://docs.cloud.google.com/tpu/docs/v5e
- <a id="gcloud-v5p"></a>`[GCloud-v5p]` *TPU v5p*. https://docs.cloud.google.com/tpu/docs/v5p
- <a id="jouppi-tpuv4"></a>`[Jouppi-TPUv4]` *TPU v4: An Optically Reconfigurable Supercomputer*. 2023-04. arXiv:2304.01433. https://arxiv.org/abs/2304.01433
- <a id="sth-iron"></a>`[STH-Iron]` *This is the Google TPU v7 Ironwood Chip*. 2025. https://www.servethehome.com/this-is-the-google-tpu-v7-ironwood-chip/
- <a id="fm-tpuocs"></a>`[FM-TPUOCS]` *Unveiling Google's TPU Architecture: OCS*. 2025. https://www.fibermall.com/blog/unveiling-google-tpu-architecture.htm
- <a id="anthropic-tpu"></a>`[Anthropic-TPU]` *Anthropic to Expand Use of Google Cloud TPUs*. 2025-10. https://www.anthropic.com/news/expanding-our-use-of-google-cloud-tpus-and-services
- <a id="anthropic-aws"></a>`[Anthropic-AWS]` *Anthropic and Amazon expand collaboration for up to 5 GW*. 2025. https://www.anthropic.com/news/anthropic-amazon-compute
- <a id="aws-trn2-page"></a>`[AWS-Trn2-Page]` *Amazon EC2 Trn2 Instances*. https://aws.amazon.com/ec2/instance-types/trn2/
- <a id="aws-trn2-blog"></a>`[AWS-Trn2-Blog]` *Trn2 Instances and Trn2 UltraServers GA*. 2024-12. https://aws.amazon.com/blogs/aws/amazon-ec2-trn2-instances-and-trn2-ultraservers-for-aiml-training-and-inference-is-now-available/
- <a id="semia-trn2"></a>`[SemiA-Trn2]` *Trainium2 Architecture & Networking*. 2025. https://newsletter.semianalysis.com/p/amazons-ai-self-sufficiency-trainium2-architecture-networking
- <a id="introl-trn3"></a>`[Introl-Trn3]` *Amazon's Trainium3 Throws Down the Gauntlet*. 2025-12. https://introl.com/blog/amazon-trainium3-aws-nvidia-ai-chip-competition-2025
- <a id="groq-lpu-page"></a>`[Groq-LPU-Page]` *Groq LPU Architecture*. https://groq.com/lpu-architecture
- <a id="groq-lpu-inside"></a>`[Groq-LPU-Inside]` *Inside the LPU: Deconstructing Groq's Speed*. 2024. https://groq.com/blog/inside-the-lpu-deconstructing-groq-speed
- <a id="semia-groq"></a>`[SemiA-Groq]` *Groq Inference Tokenomics*. 2024. https://newsletter.semianalysis.com/p/groq-inference-tokenomics-speed-but
- <a id="cerebras-cs3"></a>`[Cerebras-CS3]` *Cerebras CS-3*. 2024. https://www.cerebras.ai/blog/cerebras-cs3
- <a id="cerebras-hc24"></a>`[Cerebras-HC24]` *Cerebras Wafer-Scale AI (Hot Chips 2024)*. https://hc2024.hotchips.org/assets/program/conference/day2/72_HC2024.Cerebras.Sean.v03.final.pdf
- <a id="cerebras-compare"></a>`[Cerebras-Compare]` *Cerebras WSE Comparison with NVIDIA GPU-based Systems*. 2025. arXiv:2503.11698. https://arxiv.org/html/2503.11698v1
- <a id="nbf-cs3"></a>`[NBF-CS3]` *Cerebras CS-3 wafer-scale 25 kW*. 2025-11. https://armdevices.net/2025/11/27/cerebras-cs-3-wafer-scale-million-core-ai-chip-25kw-wse-3-125-pflops-inference-engine-tsunami-hpc/
- <a id="cerebras-press"></a>`[Cerebras-press]` *Cerebras inference release*. https://www.cerebras.ai/press-release/cerebras-launches-the-worlds-fastest-ai-inference
- <a id="samban-rdu"></a>`[SambaN-RDU]` *SN40L RDU product page*. https://sambanova.ai/products/rdu-ai-chips
- <a id="samban-rack"></a>`[SambaN-Rack]` *SambaRack SN40L-16 datasheet*. 2025. https://sambanova.ai/hubfs/SambaRack%20data%20sheet%20template%2007%2009%2025.pdf
- <a id="samban-intel"></a>`[SambaN-Intel]` *Heterogeneous Inference Blueprint: GPUs/RDUs/CPUs*. 2026-04. https://www.businesswire.com/news/home/20260408117878/
- <a id="etched-tweet"></a>`[Etched-Tweet]` *Etched Sohu announcement*. 2024-06. https://x.com/Etched/status/1805625693113663834
- <a id="etched-status"></a>`[Etched-Status]` *Etched Sohu — status review*. 2026. https://awesomeagents.ai/hardware/etched-sohu/
- <a id="dcd-etched"></a>`[DCD-Etched]` *Etched.ai raises $500m for $5bn valuation*. https://www.datacenterdynamics.com/en/news/etchedai-raises-500m-for-a-5bn-valuation-report/
- <a id="meta-mtia-v1"></a>`[Meta-MTIA-v1]` *MTIA v1*. 2023. https://ai.meta.com/blog/meta-training-inference-accelerator-AI-MTIA/
- <a id="meta-mtia-v2"></a>`[Meta-MTIA-v2]` *Our next generation MTIA*. 2024. https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/
- <a id="meta-mtia-2025"></a>`[Meta-MTIA-2025]` *Four MTIA Chips in Two Years*. 2025. https://ai.meta.com/blog/meta-mtia-scale-ai-chips-for-billions/
- <a id="sth-mtia"></a>`[STH-MTIA]` *Meta Outlines New MTIA Accelerator Roadmap*. 2025. https://www.servethehome.com/meta-outlines-new-mtia-accelerator-roadmap-for-its-next-gen-ai-compute-mix/
- <a id="meta-isca25"></a>`[Meta-ISCA25]` *Meta's Second Generation AI Chip: Model-Chip Co-Design (ISCA '25)*. 2025. https://dl.acm.org/doi/10.1145/3695053.3731409
- <a id="ms-maia200-blog"></a>`[MS-Maia200-Blog]` *Maia 200: The AI accelerator built for inference*. 2026-01. https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/
- <a id="ms-maia200-deep"></a>`[MS-Maia200-Deep]` *Deep dive into the Maia 200 architecture*. 2026. https://techcommunity.microsoft.com/blog/azureinfrastructureblog/deep-dive-into-the-maia-200-architecture/4489312
- <a id="ms-maia100"></a>`[MS-Maia100]` *Inside Maia 100*. 2024. https://techcommunity.microsoft.com/blog/azureinfrastructureblog/inside-maia-100-revolutionizing-ai-workloads-with-microsofts-custom-ai-accelerat/4229118
- <a id="tt-bh-specs"></a>`[TT-BH-Specs]` *Tenstorrent Blackhole Specifications*. https://docs.tenstorrent.com/aibs/blackhole/specifications.html
- <a id="tt-wh-specs"></a>`[TT-WH-Specs]` *Tenstorrent Wormhole Specifications*. https://docs.tenstorrent.com/aibs/wormhole/specifications.html
- <a id="tt-bh-bench"></a>`[TT-BH-Bench]` *Dissecting the Tenstorrent Blackhole Architecture via Microbenchmarking*. 2025-09. https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf
- <a id="reg-tt"></a>`[Reg-TT]` *Blackhole QuietBox review*. 2025-11. https://www.theregister.com/2025/11/27/tenstorrent_quietbox_review/
- <a id="furiosa-rngd"></a>`[Furiosa-RNGD]` *RNGD Specifications*. https://furiosa.ai/renegade-spec
- <a id="furiosa-server"></a>`[Furiosa-Server]` *Furiosa NXT RNGD Server*. 2025-09. https://furiosa.ai/blog/introducing-rngd-server-efficient-ai-inference-at-data-center-scale
- <a id="reg-furiosa"></a>`[Reg-Furiosa]` *How AI chip upstart FuriosaAI won over LG*. 2025-07. https://www.theregister.com/2025/07/22/sk_furiosa_ai_lg/

#### Networking and interconnect

- <a id="nv-nvlinkfusion-pr"></a>`[NV-NVLinkFusion-PR]` *NVIDIA Unveils NVLink Fusion*. 2025-05. https://investor.nvidia.com/news/press-release-details/2025/NVIDIA-Unveils-NVLink-Fusion-for-Industry-to-Build-Semi-Custom-AI-Infrastructure-With-NVIDIA-Partner-Ecosystem/default.aspx
- <a id="nv-nvlinkfusion-page"></a>`[NV-NVLinkFusion-Page]` *NVIDIA NVLink Fusion product page*. https://www.nvidia.com/en-us/data-center/nvlink-fusion/
- <a id="nv-nvlinkscaling"></a>`[NV-NVLinkScaling]` *Scaling AI Inference Performance with NVLink and NVLink Fusion*. https://developer.nvidia.com/blog/scaling-ai-inference-performance-and-flexibility-with-nvidia-nvlink-and-nvlink-fusion/
- <a id="nv-cx8-blog"></a>`[NV-CX8-Blog]` *NVIDIA ConnectX-8 SuperNICs*. https://developer.nvidia.com/blog/nvidia-connectx-8-supernics-advance-ai-platform-architecture-with-pcie-gen6-connectivity/
- <a id="sth-cx8"></a>`[STH-CX8]` *NVIDIA ConnectX-8 SuperNIC PCIe Gen6 800G NIC Detailed*. https://www.servethehome.com/nvidia-connectx-8-supernic-pcie-gen6-800g-nic-detailed/
- <a id="ibta-xdr"></a>`[IBTA-XDR]` *IBTA Launches XDR (800G) InfiniBand Specification*. 2024. https://www.fibermall.com/news/ibta-launches-xdr-800g-infiniband-spec.htm
- <a id="nv-qx-photonics"></a>`[NV-QX-Photonics]` *NVIDIA Spectrum-X / Quantum-X Photonics CPO*. 2025-03. https://investor.nvidia.com/news/press-release-details/2025/NVIDIA-Announces-Spectrum-X-Photonics-Co-Packaged-Optics-Networking-Switches-to-Scale-AI-Factories-to-Millions-of-GPUs/default.aspx
- <a id="nv-cpo-blog"></a>`[NV-CPO-Blog]` *A New Era in Data Center Networking with NVIDIA Silicon Photonics*. 2025. https://developer.nvidia.com/blog/a-new-era-in-data-center-networking-with-nvidia-silicon-photonics-based-network-switching/
- <a id="nv-xgs-pr"></a>`[NV-XGS-PR]` *NVIDIA Introduces Spectrum-XGS Ethernet*. 2025-08. https://nvidianews.nvidia.com/news/nvidia-introduces-spectrum-xgs-ethernet-to-connect-distributed-data-centers-into-giga-scale-ai-super-factories
- <a id="nv-xgs-blog"></a>`[NV-XGS-Blog]` *Connecting Distributed Data Centers Into Large AI Factories*. https://developer.nvidia.com/blog/how-to-connect-distributed-data-centers-into-large-ai-factories-with-scale-across-networking/
- <a id="uec-pr"></a>`[UEC-PR]` *Ultra Ethernet Consortium Launches Specification 1.0*. 2025-06. https://ultraethernet.org/ultra-ethernet-consortium-uec-launches-specification-1-0-transforming-ethernet-for-ai-and-hpc-at-scale/
- <a id="uec-spec"></a>`[UEC-Spec]` *UE Specification 1.0*. https://ultraethernet.org/wp-content/uploads/sites/20/2025/06/UE-Specification-6.11.25.pdf
- <a id="uec-arch-paper"></a>`[UEC-Arch-Paper]` *Ultra Ethernet's Design Principles and Architectural Innovations*. 2025-08. arXiv:2508.08906. https://arxiv.org/html/2508.08906v1
- <a id="phoronix-uec"></a>`[Phoronix-UEC]` *UEC 1.0 Specification*. 2025-06. https://www.phoronix.com/news/Ultra-Ethernet-1.0-UEC
- <a id="nv-gpudirect"></a>`[NV-GPUDirect]` *GPUDirect (developer page)*. https://developer.nvidia.com/gpudirect
- <a id="nv-gpudirect-op"></a>`[NV-GPUDirect-Op]` *GPUDirect RDMA and GPUDirect Storage — GPU Operator*. https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-rdma.html
- <a id="nv-quantum2"></a>`[NV-Quantum2]` *Quantum-2 InfiniBand Platform*. https://www.nvidia.com/en-us/networking/quantum2/
- <a id="nv-q800-switch"></a>`[NV-Q800-Switch]` *Quantum-X800 XDR InfiniBand Switch*. https://www.naddod.com/products/nvidia-networking/102612
- <a id="tweak-rubincpo"></a>`[Tweak-RubinCPO]` *NVIDIA GTC 2025: GB300, Rubin, CPO details*. 2025-03. https://www.tweaktown.com/news/103938/nvidia-gtc-2025-gb300-ai-gpu-with-1-4kw-power-new-details-on-rubin-cpo-tech-and-more/index.html
- <a id="xai-colossus2"></a>`[xAI-Colossus2]` *xAI Colossus Hits 2 GW: 555,000 GPUs*. 2026-01. https://introl.com/blog/xai-colossus-2-gigawatt-expansion-555k-gpus-january-2026
- <a id="semia-colossus2"></a>`[SemiA-Colossus2]` *xAI's Colossus 2*. https://newsletter.semianalysis.com/p/xais-colossus-2-first-gigawatt-datacenter
- <a id="nv-nvlink-intuit"></a>`[NV-NVLink-Intuit]` *NVIDIA NVLink Explained*. https://intuitionlabs.ai/articles/nvidia-nvlink-gpu-interconnect
- <a id="hopper-wiki"></a>`[Hopper-Wiki]` *Hopper (microarchitecture)*. https://en.wikipedia.org/wiki/Hopper_(microarchitecture)

#### Memory

- <a id="hbm-wiki"></a>`[HBM-Wiki]` *High Bandwidth Memory*. https://en.wikipedia.org/wiki/High_Bandwidth_Memory
- <a id="introl-hbm"></a>`[Introl-HBM]` *HBM evolution: HBM3 → HBM3E → HBM4*. 2025. https://introl.com/blog/hbm-evolution-hbm3-hbm3e-hbm4-memory-ai-gpu-2025
- <a id="kynix-hbm"></a>`[Kynix-HBM]` *HBM3e vs HBM4: 2026 Specs*. https://www.kynix.com/Blog/hbm3e-vs-hbm4-2026-specs-performance--supply-guide.html

### Test-time compute and reasoning serving

#### Lineage

- <a id="self-consistency-2022"></a>`[Self-Consistency-2022]` *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. Wang et al. (Google). 2022-03. ICLR 2023, arXiv:2203.11171. https://arxiv.org/abs/2203.11171
  - Establishes majority-vote-over-N as the dominant test-time aggregation.
- <a id="tot-2023"></a>`[ToT-2023]` *Tree of Thoughts: Deliberate Problem Solving with LLMs*. Yao et al. (Princeton). 2023-05. NeurIPS 2023. https://openreview.net/forum?id=5Xc1ecxO1h
  - Canonical tree-search reasoning structure.
- <a id="got-2023"></a>`[GoT-2023]` *Graph of Thoughts: Solving Elaborate Problems with LLMs*. Besta et al. 2023-08. arXiv:2308.09687. https://arxiv.org/abs/2308.09687
- <a id="openai-o1-2024"></a>`[OpenAI-o1-2024]` *Learning to Reason with LLMs*. OpenAI. 2024-09. https://openai.com/index/learning-to-reason-with-llms/
- <a id="snell-compute-optimal-2024"></a>`[Snell-Compute-Optimal-2024]` *Scaling LLM Test-Time Compute Optimally*. Snell et al. (DeepMind). 2024-08. arXiv:2408.03314. https://openreview.net/forum?id=4FWAwZtd2n
  - Compute-budget framework: shortest-CoT for low budget, beam search medium, voting high.
- <a id="tanay-its-2024"></a>`[Tanay-ITS-2024]` *OpenAI's o-1 and Inference-Time Scaling Laws*. T. Janakiraman. 2024. https://www.tanayj.com/p/openais-o-1-and-inference-time-scaling
- <a id="semia-o1-pro-2024"></a>`[SemiA-O1-Pro-2024]` *Scaling Laws – O1 Pro Architecture*. SemiAnalysis. 2024-12. https://semianalysis.com/2024/12/11/scaling-laws-o1-pro-architecture-reasoning-training-infrastructure-orion-and-claude-3-5-opus-failures/

#### Past 12 months SOTA — reasoning serving

- <a id="openai-o3-2025"></a>`[OpenAI-o3-2025]` *Introducing OpenAI o3 and o4-mini*. OpenAI. 2025-04. https://openai.com/index/introducing-o3-and-o4-mini/
- <a id="claude-adaptive-thinking-2026"></a>`[Claude-Adaptive-Thinking-2026]` *Adaptive thinking — Claude API*. Anthropic. 2026. https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
- <a id="lrm-survey-2025"></a>`[LRM-Survey-2025]` *Efficient Inference for Large Reasoning Models: A Survey*. Liu et al. 2025-03. arXiv:2503.23077. https://arxiv.org/abs/2503.23077
- <a id="stop-overthinking-2025"></a>`[Stop-Overthinking-2025]` *Stop Overthinking: A Survey on Efficient Reasoning for LLMs*. 2025-03. arXiv:2503.16419. https://arxiv.org/pdf/2503.16419
- <a id="agot-2025"></a>`[AGoT-2025]` *Adaptive Graph of Thoughts: Test-Time Adaptive Reasoning Unifying Chain, Tree, and Graph Structures*. 2025-02. arXiv:2502.05078. https://arxiv.org/abs/2502.05078
- <a id="sample-scrutinize-scale-2025"></a>`[Sample-Scrutinize-Scale-2025]` *Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification*. Google. 2025-02. ICML 2025, arXiv:2502.01839. https://arxiv.org/abs/2502.01839
- <a id="forest-of-thought-2025"></a>`[Forest-of-Thought-2025]` *Forest-of-Thought: Scaling Test-Time Compute*. ICLR 2025. https://openreview.net/forum?id=BMJ3pyYxu2
- <a id="hermes-2025"></a>`[Hermes-2025]` *Understanding and Optimizing Multi-Stage AI Inference Pipelines*. Krishna et al. (MIT CSAIL). 2025. https://people.csail.mit.edu/suvinay/pubs/2025.hermes.arxiv.pdf
- <a id="reasoning-serving-empirical-2025"></a>`[Reasoning-Serving-Empirical-2025]` *Reasoning Language Model Inference Serving Unveiled: An Empirical Study*. 2025-10. arXiv:2510.18672. https://arxiv.org/pdf/2510.18672
- <a id="sparsespec-2025"></a>`[SparseSpec-2025]` *Accelerating Large-Scale Reasoning Model Inference with Sparse Self-Speculative Decoding*. 2025-12. arXiv:2512.01278. https://arxiv.org/abs/2512.01278
  - Up to 2.13× throughput; co-designs scheduler, delayed verification, KV mgmt for self-speculation on long CoTs.

### Structured output and tool calling

#### Lineage

- <a id="outlines-2023"></a>`[Outlines-2023]` *Efficient Guided Generation for Large Language Models*. Willard, Louf (.txt). 2023-07. arXiv:2307.09702. https://arxiv.org/abs/2307.09702
- <a id="guidance-2023"></a>`[Guidance-2023]` *guidance (library)*. Microsoft. 2023. https://github.com/guidance-ai/llguidance
- <a id="lmfe-2023"></a>`[LMFE-2023]` *lm-format-enforcer*. N. Gat. 2023. https://github.com/noamgat/lm-format-enforcer
- <a id="llama-cpp-grammars"></a>`[Llama.cpp-Grammars]` *GBNF grammar support in llama.cpp*. ggerganov. 2023. https://github.com/ggerganov/llama.cpp
- <a id="domino-2024"></a>`[DOMINO-2024]` *Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation*. Beurer-Kellner et al. (ETH SRI). 2024-03. arXiv:2403.06988. https://arxiv.org/html/2403.06988v1

#### Past 12 months SOTA — structured output

- <a id="xgrammar-2024"></a>`[XGrammar-2024]` *XGrammar: Flexible and Efficient Structured Generation Engine for LLMs*. Dong, Ruan, Cai, Lai, Xu, Zhao, Chen (CMU/MLC, NVIDIA, SJTU, Berkeley). 2024-11. MLSys 2025, arXiv:2411.15100. https://arxiv.org/pdf/2411.15100
  - Default backend in vLLM/SGLang/TRT-LLM/MLC; up to 14× JSON-schema, 80–100× CFG.
- <a id="llguidance-2024"></a>`[llguidance-2024]` *llguidance: Super-fast Structured Outputs*. Microsoft / guidance-ai. 2024–25. https://github.com/guidance-ai/llguidance
  - ~50 µs CPU/token at 128K vocab.
- <a id="outlines-coalescence"></a>`[Outlines-Coalescence]` *Coalescence: making LLM inference 5x faster*. dottxt. 2024. https://blog.dottxt.ai/coalescence.html
- <a id="genoutputs-bench-2025"></a>`[GenOutputs-Bench-2025]` *Generating Structured Outputs from Language Models: Benchmark and Studies*. 2025-01. arXiv:2501.10868. https://arxiv.org/html/2501.10868v1
- <a id="flexgcd-icml2025"></a>`[FlexGCD-ICML2025]` *Flexible and Efficient Grammar-Constrained Decoding*. ICML 2025, arXiv:2502.05111. https://icml.cc/virtual/2025/poster/45613
- <a id="itergen-2025"></a>`[IterGen-2025]` *IterGen: Forward-Backward Grammar Generation*. ICLR 2025.
- <a id="guideddecoding-rag-2025"></a>`[GuidedDecoding-RAG-2025]` *Guided Decoding and Its Critical Role in RAG*. 2025-09. arXiv:2509.06631. https://arxiv.org/html/2509.06631v1
- <a id="squeezebits-bench-2025"></a>`[SqueezeBits-Bench-2025]` *Guided Decoding Performance on vLLM and SGLang*. SqueezeBits. 2025. https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang
- <a id="anthropic-structuredoutputs-2025"></a>`[Anthropic-StructuredOutputs-2025]` *Structured Outputs (Claude 4.5/Opus 4.1)*. Anthropic. 2025. https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- <a id="anthropic-advanced-tool-use-2025"></a>`[Anthropic-Advanced-Tool-Use-2025]` *Introducing advanced tool use on the Claude Developer Platform*. Anthropic. 2025. https://www.anthropic.com/engineering/advanced-tool-use
- <a id="brenndoerfer-2025-survey"></a>`[Brenndoerfer-2025-survey]` *Constrained Decoding: Grammar-Guided Generation for Structured LLM Output*. 2025. https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output
- <a id="vllm-structdec-blog"></a>`[vLLM-StructDec-Blog]` *Structured Decoding in vLLM: A Gentle Introduction*. vLLM/BentoML. 2025-01. https://blog.vllm.ai/2025/01/14/struct-decode-intro.html

### Multimodal serving

#### Lineage

- <a id="llava-2023"></a>`[LLaVA-2023]` *Visual Instruction Tuning*. Liu et al. 2023-04. NeurIPS 2023, arXiv:2304.08485. https://arxiv.org/abs/2304.08485
- <a id="internvl-2024"></a>`[InternVL-2024]` *InternVL: Scaling up Vision Foundation Models*. OpenGVLab. 2024. CVPR 2024 Oral. https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_InternVL_Scaling_up_Vision_Foundation_Models_and_Aligning_for_Generic_CVPR_2024_paper.pdf
- <a id="qwen2-vl-2024"></a>`[Qwen2-VL-2024]` *Qwen2-VL: Enhancing VLM's Perception of the World at Any Resolution*. Qwen Team. 2024.
- <a id="whisper-2022"></a>`[Whisper-2022]` *Robust Speech Recognition via Large-Scale Weak Supervision*. Radford et al. (OpenAI). 2022-12. arXiv:2212.04356. https://arxiv.org/abs/2212.04356
- <a id="gpt-4o-2024"></a>`[GPT-4o-2024]` *Hello GPT-4o*. OpenAI. 2024-05. https://openai.com/index/hello-gpt-4o/

#### Past 12 months SOTA — multimodal

- <a id="vllm-v1-multimodal"></a>`[vLLM-V1-Multimodal]` *vLLM V1 (multimodal cache + chunked prefill for VLMs)*. vLLM Team. 2025-01. https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
- <a id="hydrainfer-2025"></a>`[HydraInfer-2025]` *HydraInfer: Hybrid Disaggregated Scheduling for Multimodal LLM Inference*. 2025-05. arXiv:2505.12658. https://arxiv.org/pdf/2505.12658
- <a id="modserve-2025"></a>`[ModServe-2025]` *ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving*. Qiu et al. SoCC 2025. https://dl.acm.org/doi/pdf/10.1145/3772052.3772254
- <a id="nova-2025"></a>`[Nova-2025]` *Nova: Real-Time Agentic Vision-Language Model Serving with Adaptive Cross-Stage Parallelization*. 2025-09. arXiv:2509.21301. https://arxiv.org/abs/2509.21301
- <a id="spaceserve-2025"></a>`[SpaceServe-2025]` *SpaceServe: Spatial Multiplexing of Complementary Resources*. OpenReview 2025. https://openreview.net/pdf?id=...3b4c3201
- <a id="lmm-characterization-2025"></a>`[LMM-Characterization-2025]` *Towards Efficient Large Multimodal Model Serving*. Qiu et al. 2025-02. arXiv:2502.00937. https://arxiv.org/html/2502.00937v1
- <a id="fastvlm-cvpr2025"></a>`[FastVLM-CVPR2025]` *FastVLM: Efficient Vision Encoding for Vision Language Models*. Apple ML. 2025-06. CVPR 2025. https://machinelearning.apple.com/research/fast-vision-language-models
  - 85× TTFT vs LLaVA-OneVision-0.5B; 7.9× at 7B.
- <a id="hf-vlms-2025"></a>`[HF-VLMs-2025]` *Vision Language Models (Better, faster, stronger)*. Hugging Face. 2025. https://huggingface.co/blog/vlms-2025
- <a id="qwen2-5-omni-2025"></a>`[Qwen2.5-Omni-2025]` *Qwen2.5-Omni Technical Report*. Alibaba Qwen. 2025-03. https://github.com/QwenLM/Qwen2.5-Omni
- <a id="qwen3-omni-2025"></a>`[Qwen3-Omni-2025]` *Qwen3-Omni*. Alibaba Qwen. 2025. https://github.com/QwenLM/Qwen3-Omni
- <a id="f16-videollm-2025"></a>`[F16-VideoLLM-2025]` *Improving LLM Video Understanding with 16 Frames Per Second*. ICML 2025. https://icml.cc/virtual/2025/poster/46540
- <a id="streamingvideollm-2025"></a>`[StreamingVideoLLM-2025]` *Streaming VideoLLMs for Real-time Procedural Video Understanding*. ICCV 2025. https://www.openaccess.thecvf.com/content/ICCV2025/papers/Chatterjee_Streaming_VideoLLMs_for_Real-Time_Procedural_Video_Understanding_ICCV_2025_paper.pdf
- <a id="lmcache-mm-2025"></a>`[LMCache-MM-2025]` *LMCache Extends Its Turbo-Boost to Multimodal Models in vLLM V1*. 2025-07. https://blog.lmcache.ai/en/2025/07/03/lmcache-extends-its-turbo-boost-to-multimodal-models-in-vllm-v1/
- <a id="mlperf-whisper-2025"></a>`[MLPerf-Whisper-2025]` *Whisper: An MLPerf Inference Benchmark for ASR*. MLCommons. 2025-09. https://mlcommons.org/2025/09/whisper-inferencev5-1/
- <a id="fasterwhisper"></a>`[FasterWhisper]` *SYSTRAN/faster-whisper*. SYSTRAN. https://github.com/SYSTRAN/faster-whisper
- <a id="batched-whisper-2024"></a>`[Batched-Whisper-2024]` *Speeding up Whisper (ASR)*. Mobius ML. 2024. https://mobiusml.github.io/batched_whisper_blog/
- <a id="vllm-omni-2026"></a>`[vLLM-Omni-2026]` *vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models*. 2026. arXiv:2602.02204. https://arxiv.org/abs/2602.02204

### RAG infrastructure

#### Lineage

- <a id="faiss-2017"></a>`[FAISS-2017]` *Billion-scale similarity search with GPUs*. Johnson, Douze, Jégou (FAIR). 2017. arXiv:1702.08734. https://arxiv.org/abs/1702.08734
- <a id="hnsw-2018"></a>`[HNSW-2018]` *Efficient and robust ANN search using Hierarchical Navigable Small World graphs*. Malkov, Yashunin. 2018. arXiv:1603.09320. https://arxiv.org/abs/1603.09320
- <a id="scann-2020"></a>`[ScaNN-2020]` *Accelerating Large-Scale Inference with Anisotropic Vector Quantization*. Guo et al. (Google). 2020. ICML 2020, arXiv:1908.10396. https://arxiv.org/abs/1908.10396
- <a id="diskann-2019"></a>`[DiskANN-2019]` *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search*. Subramanya et al. (Microsoft). 2019. NeurIPS 2019. https://suhasjs.github.io/files/diskann_neurips19.pdf
- <a id="spann-2021"></a>`[SPANN-2021]` *SPANN: Highly-efficient Billion-scale ANN*. Chen et al. (Microsoft). 2021. NeurIPS 2021, arXiv:2111.08566. https://arxiv.org/abs/2111.08566
- <a id="colbert-2020"></a>`[ColBERT-2020]` *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*. Khattab, Zaharia. 2020-04. SIGIR 2020, arXiv:2004.12832. https://arxiv.org/abs/2004.12832
- <a id="colbertv2-2022"></a>`[ColBERTv2-2022]` *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction*. Santhanam et al. 2022. NAACL 2022, arXiv:2112.01488. https://arxiv.org/abs/2112.01488
- <a id="plaid-2022"></a>`[PLAID-2022]` *PLAID: An Efficient Engine for Late Interaction Retrieval*. Santhanam et al. 2022-05. CIKM 2022, arXiv:2205.09707. https://arxiv.org/abs/2205.09707
- <a id="splade-2021"></a>`[SPLADE-2021]` *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking*. Formal, Lassance, Piwowarski, Clinchant. 2021. SIGIR 2021/2022, arXiv:2107.05720. https://arxiv.org/abs/2107.05720
- <a id="bge-m3-2024"></a>`[BGE-M3-2024]` *BGE-M3: Multi-Functionality, Multi-Linguistic, Multi-Granular Embeddings*. BAAI. 2024. arXiv:2402.03216. https://arxiv.org/abs/2402.03216

#### Past 12 months SOTA — RAG

- <a id="colpali-2024"></a>`[ColPali-2024]` *ColPali: Efficient Document Retrieval with Vision Language Models*. Faysse et al. (Illuin). 2024-07. ICLR 2025, arXiv:2407.01449. https://arxiv.org/abs/2407.01449
- <a id="colqwen-2025"></a>`[ColQwen-2025]` *ColQwen2 / ColQwen2.5*. Illuin / community. 2025. https://github.com/illuin-tech/colpali
- <a id="cache-craft-2025"></a>`[Cache-Craft-2025]` *Cache-Craft: Managing Chunk-Caches for Efficient RAG*. Kejriwal et al. (Adobe Research). 2025-02. SIGMOD 2025, arXiv:2502.15734. https://arxiv.org/abs/2502.15734
  - -51% redundancy vs prefix caching; 1.6× lower production latency.
- <a id="agenticrag-survey-2025"></a>`[AgenticRAG-Survey-2025]` *Agentic Retrieval-Augmented Generation: A Survey*. Singh et al. 2025-01. arXiv:2501.09136. https://arxiv.org/abs/2501.09136
- <a id="agenticrag-kg-2025"></a>`[AgenticRAG-KG-2025]` *Agentic RAG with Knowledge Graphs for Complex Multi-Hop Reasoning*. 2025-07. arXiv:2507.16507. https://arxiv.org/abs/2507.16507
- <a id="distributedann-2025"></a>`[DistributedANN-2025]` *DistributedANN: Efficient Scaling of a Single DiskANN Graph Across Thousands of Computers*. Microsoft. 2025-09. arXiv:2509.06046. https://arxiv.org/abs/2509.06046
  - 26 ms median, >100K QPS over 50B-vector index.
- <a id="spfresh-2024"></a>`[SPFresh-2024]` *SPFresh: Incremental In-Place Update for Billion-Scale Vector Search*. 2024-10. arXiv:2410.14452. https://arxiv.org/pdf/2410.14452
- <a id="rerankers-bench-2025"></a>`[Rerankers-Bench-2025]` *Top 8 Rerankers: Quality vs Cost benchmark*. 2025-09. https://aimultiple.com/rerankers
- <a id="vectordbbench"></a>`[VectorDBBench]` *VectorDBBench*. Zilliz. https://github.com/zilliztech/VectorDBBench
- <a id="vespa-billion-2024"></a>`[Vespa-Billion-2024]` *Building Billion-Scale Vector Search (Vespa)*. https://medium.com/vespa/building-billion-scale-vector-search-part-two-94f0101d15dd
- <a id="pgvector-08"></a>`[pgvector-08]` *pgvector 0.8 on Aurora*. AWS. 2024–25. https://aws.amazon.com/blogs/database/supercharging-vector-search-performance-and-relevance-with-pgvector-0-8-0-on-amazon-aurora-postgresql/
- <a id="pgvectorscale-2024"></a>`[pgvectorscale-2024]` *pgvectorscale (Timescale)*. https://github.com/timescale/pgvectorscale
- <a id="mcp-donation-2025"></a>`[MCP-Donation-2025]` *Anthropic donates MCP to Linux Foundation Agentic AI Foundation*. 2025-12.

### Embedding and reranker serving

#### Lineage

- <a id="sentence-bert-2019"></a>`[Sentence-BERT-2019]` *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. Reimers, Gurevych. 2019-08. EMNLP 2019, arXiv:1908.10084. https://arxiv.org/abs/1908.10084
- <a id="e5-2022"></a>`[E5-2022]` *Text Embeddings by Weakly-Supervised Contrastive Pre-training*. Wang et al. (Microsoft). 2022. arXiv:2212.03533. https://arxiv.org/abs/2212.03533
- <a id="bge-2023"></a>`[BGE-2023]` *C-Pack / BGE: Packaged Resources To Advance General Chinese Embeddings*. Xiao et al. (BAAI). 2023. arXiv:2309.07597. https://arxiv.org/abs/2309.07597
- <a id="matryoshka-2022"></a>`[Matryoshka-2022]` *Matryoshka Representation Learning*. Kusupati et al. 2022. NeurIPS 2022, arXiv:2205.13147. https://arxiv.org/abs/2205.13147
- <a id="mteb-2022"></a>`[MTEB-2022]` *MTEB: Massive Text Embedding Benchmark*. Muennighoff et al. 2022. EACL 2023, arXiv:2210.07316. https://arxiv.org/abs/2210.07316

#### Past 12 months SOTA — embeddings/rerankers

- <a id="mmteb-2025"></a>`[MMTEB-2025]` *MMTEB: Massive Multilingual Text Embedding Benchmark*. Enevoldsen et al. 2025-02. ICLR 2025, arXiv:2502.13595. https://arxiv.org/abs/2502.13595
  - 250+ languages, 500+ tasks; new leaderboard default.
- <a id="mteb-maintenance-2025"></a>`[MTEB-Maintenance-2025]` *Maintaining MTEB*. 2025-06. arXiv:2506.21182. https://arxiv.org/html/2506.21182v1
- <a id="qwen3-embedding-2025"></a>`[Qwen3-Embedding-2025]` *Qwen3 Embedding: Advancing Text Embedding and Reranking through Foundation Models*. Qwen Team. 2025-06. arXiv:2506.05176. https://arxiv.org/pdf/2506.05176
  - Qwen3-Embedding-8B leads MMTEB multilingual at 70.58.
- <a id="nv-embed-v2"></a>`[NV-Embed-v2]` *NV-Embed-v2*. NVIDIA. 2024-25. https://huggingface.co/nvidia/NV-Embed-v2
- <a id="voyage-3-5"></a>`[Voyage-3.5]` *Voyage 3.5 / 3.5-Lite*. Voyage AI / MongoDB. 2025. https://docs.voyageai.com/docs/flexible-dimensions-and-quantization
- <a id="hf-embedding-quant-2024"></a>`[HF-Embedding-Quant-2024]` *Binary and Scalar Embedding Quantization*. HF / Sentence-Transformers. 2024. https://huggingface.co/blog/embedding-quantization
- <a id="vespa-matryoshka-binary"></a>`[Vespa-Matryoshka-Binary]` *Matryoshka 🤝 Binary vectors*. Vespa. 2024. https://blog.vespa.ai/combining-matryoshka-with-binary-quantization-using-embedder/
- <a id="tei-repo"></a>`[TEI-Repo]` *huggingface/text-embeddings-inference*. https://github.com/huggingface/text-embeddings-inference
- <a id="infinity-repo"></a>`[Infinity-Repo]` *michaelfeil/infinity*. https://github.com/michaelfeil/infinity
- <a id="cohere-rerank-docs"></a>`[Cohere-Rerank-Docs]` *Cohere Rerank*. https://docs.cohere.com/docs/rerank
- <a id="jina-rerank"></a>`[Jina-Rerank]` *Jina Reranker v2/v3*. Jina AI. 2024–25. https://jina.ai
- <a id="bge-reranker"></a>`[BGE-Reranker]` *BGE Reranker family*. BAAI. https://huggingface.co/BAAI
- <a id="modernbert-2024"></a>`[ModernBERT-2024]` *ModernBERT: A Modern Bidirectional Encoder*. Warner et al. (Answer.AI / LightOn). 2024-12. arXiv:2412.13663. https://arxiv.org/abs/2412.13663
- <a id="4bit-quant-rag-2025"></a>`[4bit-Quant-RAG-2025]` *4bit-Quantization in Vector-Embedding for RAG*. 2025-01. arXiv:2501.10534. https://arxiv.org/html/2501.10534v1
- <a id="qwen3-embedding-repo"></a>`[Qwen3-Embedding-Repo]` *QwenLM/Qwen3-Embedding*. Alibaba Qwen. 2025-06. https://github.com/QwenLM/Qwen3-Embedding
- <a id="aer-labs-2025-blog"></a>`[AER-Labs-2025-blog]` *Optimizing Embedding Model Inference*. AER Labs. 2025. https://aerlabs.tech/blogs/optimizing-embedding-model-inference
- <a id="filip-substack-2024"></a>`[Filip-Substack-2024]` *Comparing Embedding Inference Solutions: TEI, Infinity, FastEmbed*. F. Makraduli. https://open.substack.com/pub/filipmakraduli/p/comparing-embedding-inference-solutions
- <a id="hf-mteb-leaderboard"></a>`[HF-MTEB-Leaderboard]` *HF MTEB Leaderboard*. https://huggingface.co/spaces/mteb/leaderboard

### RL post-training infrastructure

#### Lineage

- <a id="instructgpt"></a>`[InstructGPT]` *Training language models to follow instructions with human feedback*. Ouyang et al. (OpenAI). 2022. arXiv:2203.02155. https://arxiv.org/abs/2203.02155
  - SFT → RM → PPO three-stage RLHF recipe template.
- <a id="cai"></a>`[CAI]` *Constitutional AI: Harmlessness from AI Feedback*. Bai et al. (Anthropic). 2022-12. arXiv:2212.08073. https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback
- <a id="deepspeed-chat"></a>`[DeepSpeed-Chat]` *DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training*. Yao et al. (Microsoft). 2023-08. arXiv:2308.01320. https://arxiv.org/abs/2308.01320
  - Hybrid Engine; 10× speedup over HF/Colossal-AI baselines.
- <a id="openrlhf"></a>`[OpenRLHF]` *OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework*. Hu et al. 2024-05. EMNLP-Demos 2025, arXiv:2405.11143. https://arxiv.org/abs/2405.11143
  - Ray + vLLM + DeepSpeed ZeRO; canonical NCCL broadcast and CUDA-IPC weight-sync.
- <a id="deepseekmath-grpo"></a>`[DeepSeekMath-GRPO]` *DeepSeekMath: Pushing the Limits of Mathematical Reasoning*. Shao et al. (DeepSeek). 2024-02. arXiv:2402.03300. https://arxiv.org/abs/2402.03300
  - Origin of GRPO.
- <a id="hybridflow-verl"></a>`[HybridFlow-veRL]` *HybridFlow: A Flexible and Efficient RLHF Framework*. Sheng et al. (ByteDance Seed + HKU). 2024-09. arXiv:2409.19256. https://arxiv.org/abs/2409.19256
  - Hybrid single+multi-controller; 3D-HybridEngine resharding; 1.53–20.57× over baselines.
- <a id="nemo-aligner"></a>`[NeMo-Aligner]` *NeMo-Aligner: Scalable Toolkit for Efficient Model Alignment*. Shen et al. (NVIDIA). 2024-05. arXiv:2405.01481. https://arxiv.org/abs/2405.01481
- <a id="async-rlhf"></a>`[Async-RLHF]` *Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models*. Noukhovitch et al. (MILA). 2024-10. ICLR 2025, arXiv:2410.18252. https://arxiv.org/abs/2410.18252
  - ~40–70% speedup; off-policy tolerance increases with policy-model scale.

#### Past 12 months SOTA — RL frameworks

- <a id="verl-0-7"></a>`[veRL-0.7]` *verl 0.7 release blog*. verl-project / ByteDance. 2026-01. https://verl.readthedocs.io/en/latest/blog/v0.7.html
  - Native server-mode rollout; Checkpoint Engine (NCCL + NIXL); on-policy / one-step-off / fully-async.
- <a id="areal"></a>`[AReaL]` *AReaL: A Large-Scale Asynchronous RL System for Language Reasoning*. Fu et al. (Tsinghua/Ant). 2025-05. arXiv:2505.24298. https://arxiv.org/abs/2505.24298
  - Fully asynchronous; modified PPO; 2.77× speedup over sync.
- <a id="slime"></a>`[slime]` *slime: An SGLang-Native Post-Training Framework for RL Scaling*. THUDM (Tsinghua) + LMSYS. 2025-07. https://www.lmsys.org/blog/2025-07-09-slime/
  - Powers GLM-4.5/4.6/4.7.
- <a id="skyrl"></a>`[SkyRL]` *SkyRL: A Modular Full-stack RL Library for LLMs*. NovaSky-AI (Berkeley) + Anyscale. 2025. https://github.com/NovaSky-AI/SkyRL
- <a id="nemo-rl"></a>`[NeMo-RL]` *NeMo-RL*. NVIDIA. 2025. https://github.com/NVIDIA-NeMo/RL
  - Megatron-Core backend; supports DAPO; v0.6 ships speculative decoding inside RL loop.
- <a id="roll-flash"></a>`[ROLL-Flash]` *ROLL Flash – Accelerating RLVR and Agentic Training with Asynchrony*. Alibaba. 2025-10. arXiv:2510.11345. https://arxiv.org/abs/2510.11345
  - 2.24× RLVR / 2.72× agentic; near-linear scaling at 100+ GPUs.
- <a id="rollmux"></a>`[RollMux]` *RollMux: Phase-level Multiplexing*. 2025. arXiv:2512.11306. https://arxiv.org/abs/2512.11306
- <a id="rollart"></a>`[RollArt]` *RollArt: Agentic Disaggregated Infra*. 2025. arXiv:2512.22560. https://arxiv.org/abs/2512.22560
- <a id="openrlhf-2025"></a>`[OpenRLHF-2025]` *OpenRLHF*. Hu et al. 2025. EMNLP-Demos 2025, https://github.com/OpenRLHF/OpenRLHF
  - Adds DAPO, REINFORCE++, async agentic RL, partial-rollout, TIS.
- <a id="awex"></a>`[Awex]` *Awex: An Ultra-Fast Weight Sync Framework for Trillion-Scale RL*. Ant Group. 2025. https://github.com/inclusionAI/asystem-awex
  - 1T params synced in 6s (RDMA) / 20s (NCCL) on thousand-GPU clusters.
- <a id="checkpoint-engine"></a>`[checkpoint-engine]` *Mooncake checkpoint-engine*. Moonshot AI. 2025-09. https://github.com/MoonshotAI/checkpoint-engine
  - Second-level updates of trillion-param models for vLLM/SGLang.
- <a id="tinker"></a>`[Tinker]` *Tinker*. Thinking Machines Lab. 2025-10. https://thinkingmachines.ai/tinker/
- <a id="pipelinerl"></a>`[PipelineRL]` *PipelineRL*. ServiceNow Research. 2025.
  - Per-forward-pass weight swaps; Redis-backed rollout stream.
- <a id="torchforge"></a>`[TorchForge]` *TorchForge*. Meta. 2025.

#### Past 12 months SOTA — RL algorithms / objectives

- <a id="dapo"></a>`[DAPO]` *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. ByteDance Seed. 2025-03. arXiv:2503.14476. https://arxiv.org/abs/2503.14476
  - Clip-Higher, Dynamic Sampling, Token-Level PG Loss, Overlong Reward Shaping.
- <a id="dr-grpo"></a>`[Dr.GRPO]` *Understanding R1-Zero-Like Training: A Critical Perspective*. Liu et al. (Sea AI Lab). 2025-03. arXiv:2503.20783. https://arxiv.org/abs/2503.20783
- <a id="open-reasoner-zero"></a>`[Open-Reasoner-Zero]` *Open-Reasoner-Zero*. StepFun. 2025-03. arXiv:2503.24290. https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero
- <a id="j1"></a>`[J1]` *J1: Incentivizing Thinking in LLM-as-a-Judge via RL*. 2025-05. arXiv:2505.10320. https://arxiv.org/abs/2505.10320
- <a id="truncated-ppo"></a>`[Truncated-PPO]` *Truncated Proximal Policy Optimization*. arXiv:2506.15050. https://arxiv.org/abs/2506.15050
- <a id="infinite-sampling"></a>`[Infinite-Sampling]` *Infinite Sampling: Efficient and Stable Grouped RL Training*. arXiv:2506.22950. https://arxiv.org/abs/2506.22950
- <a id="spec-rl"></a>`[SPEC-RL]` *SPEC-RL: Accelerating On-Policy RL via Speculative Rollouts*. 2025-09. arXiv:2509.23232. https://arxiv.org/abs/2509.23232
- <a id="asys-surveys"></a>`[Asys-Surveys]` *A Survey of Reinforcement Learning for Large Reasoning Models*. Tsinghua C3I. 2025-09. arXiv:2509.08827. https://arxiv.org/abs/2509.08827
- <a id="fp16-rl"></a>`[FP16-RL]` *Defeating the Training-Inference Mismatch via FP16*. arXiv:2510.26788. https://arxiv.org/abs/2510.26788
- <a id="bitwise-consistent-rl"></a>`[Bitwise-Consistent-RL]` *No More Train-Inference Mismatch: Bitwise Consistent On-Policy RL with vLLM and TorchTitan*. vLLM Blog. 2025-11. https://blog.vllm.ai/2025/11/10/bitwise-consistent-rl.html
- <a id="laminar"></a>`[Laminar]` *Laminar: A Scalable Asynchronous RL Post-Training Framework*. arXiv:2510.12633. https://arxiv.org/abs/2510.12633
- <a id="rollpacker"></a>`[RollPacker]` *RollPacker: Mitigating Long-Tail Rollouts*. arXiv:2509.21009. https://arxiv.org/abs/2509.21009
- <a id="periodic-async"></a>`[Periodic-Async]` *Periodic Asynchrony: An Effective Method for Accelerating On-Policy RL*. arXiv:2511.18871. https://arxiv.org/abs/2511.18871
- <a id="a-3po"></a>`[A-3PO]` *A-3PO: Accelerating Asynchronous LLM Training with Staleness-aware Proximal Policy Approximation*. arXiv:2512.06547. https://arxiv.org/abs/2512.06547
- <a id="tensorhub"></a>`[TensorHub]` *TensorHub: Scalable and Elastic Weight Transfer for LLM RL Training*. arXiv:2604.09107. https://arxiv.org/abs/2604.09107
- <a id="nemo-rl-specdec"></a>`[NeMo-RL-SpecDec]` *Speculative Decoding in NeMo-RL*. NVIDIA. 2026-05.
  - 1.8× rollout speedup at 8B; projected 2.5× E2E at 235B.
- <a id="hf-asyncrl"></a>`[HF-AsyncRL]` *Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries*. Hugging Face. 2026-03. https://huggingface.co/blog/async-rl-training-landscape
- <a id="anyscale-comparison"></a>`[Anyscale-Comparison]` *Open Source RL Libraries for LLMs*. Anyscale. 2025.
- <a id="langcopilot-comparison"></a>`[LangCopilot-Comparison]` *OpenRLHF vs veRL: Ray Framework Deep Dive*. 2025-11. https://langcopilot.com/posts/2025-11-06-openrlhf-vs-verl-ray-framework-deep
- <a id="absolute-zero"></a>`[Absolute-Zero]` *Absolute Zero: Reinforced Self-play Reasoning with Zero Data*. arXiv:2505.03335. https://arxiv.org/abs/2505.03335
- <a id="llama-4-rl"></a>`[Llama-4-RL]` *Llama 4 Multimodal Intelligence — RL pipeline*. Meta. 2025-04. https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- <a id="prorl-agent"></a>`[ProRL-Agent]` *ProRL Agent: Rollout-as-a-Service for RL Training of Multi-Turn LLM Agents*. NVIDIA. 2026-03. arXiv:2603.18815. https://arxiv.org/abs/2603.18815
- <a id="verltool"></a>`[VerlTool]` *VerlTool: Holistic Agentic RL with Tool Use*. arXiv:2509.01055. https://arxiv.org/abs/2509.01055
- <a id="opensandbox"></a>`[OpenSandbox]` *Alibaba's open-source sandbox runtime for agentic RL*. https://github.com/alibaba/OpenSandbox
- <a id="outcome-process-harmonization"></a>`[Outcome-Process-Harmonization]` *Beyond Correctness: Harmonizing Process and Outcome Rewards*. arXiv:2509.03403. https://arxiv.org/abs/2509.03403
- <a id="awesome-rlvr"></a>`[awesome-RLVR]` *Awesome-RLVR (curated list)*. https://github.com/opendilab/awesome-RLVR
- <a id="vllm-sleepmode"></a>`[vLLM-SleepMode]` *Zero-Reload Model Switching with vLLM Sleep Mode*. vLLM Blog. 2025-10. https://blog.vllm.ai/2025/10/26/vllm-sleep-mode.html
- <a id="vllm-rfc-31848"></a>`[vLLM-RFC-31848]` *Native Weight Syncing APIs (vLLM)*. https://github.com/vllm-project/vllm/issues/31848
- <a id="sglang-rl-doc"></a>`[SGLang-RL-Doc]` *SGLang for RL Systems*. https://sgl-project.github.io/advanced_features/sglang_for_rl.html
- <a id="perplexity-weighttransfer"></a>`[Perplexity-WeightTransfer]` *Weight Transfer for RL Post-Training in under 2 seconds*. Perplexity Research. https://research.perplexity.ai/articles/rdma-point-to-point-communication-for-llm-systems
- <a id="kimi-k2-rl"></a>`[Kimi-K2-RL]` See `[Kimi-K2]` above. RLVR + self-critique rubric reward; K8s sandbox >10k concurrent instances.

### Production engines and orchestrators (OSS deep-dive references)

- <a id="vllm"></a>`[vLLM]` *vLLM*. vLLM Project (Berkeley/UC). https://github.com/vllm-project/vllm
  - Reference open-source LLM serving engine; PagedAttention, V1 architecture, ecosystem hub.
- <a id="vllm-v1-blog"></a>`[vLLM-V1-blog]` *vLLM V1: A Major Upgrade to vLLM's Core Architecture*. vLLM Team. 2025-01. https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
- <a id="sglang"></a>`[SGLang]` *SGLang*. LMSYS / SGLang. https://github.com/sgl-project/sglang
  - RadixAttention, prefix-aware scheduler, native MoE/EP/disagg.
- <a id="tensorrt-llm"></a>`[TensorRT-LLM]` *TensorRT-LLM*. NVIDIA. https://github.com/NVIDIA/TensorRT-LLM
  - NVIDIA's official LLM serving stack; in-flight batching, FP8/NVFP4, Wide-EP.
- <a id="tgi"></a>`[TGI]` *Text Generation Inference*. Hugging Face. https://github.com/huggingface/text-generation-inference
- <a id="llama-cpp"></a>`[llama.cpp]` *llama.cpp*. ggerganov. https://github.com/ggerganov/llama.cpp
  - GGUF quants, CPU/Apple/edge baseline.
- <a id="mistral-rs"></a>`[mistral.rs]` *mistral.rs*. EricLBuehler. https://github.com/EricLBuehler/mistral.rs
- `[mlx]` *Apple MLX*. https://github.com/ml-explore/mlx
- <a id="mlx-lm"></a>`[mlx-lm]` *Apple MLX-LM*. https://github.com/ml-explore/mlx-lm
- <a id="mlx-examples"></a>`[mlx-examples]` *Apple MLX examples*. https://github.com/ml-explore/mlx-examples
- `[ktransformers]` *KTransformers (CPU/GPU hybrid)*. kvcache-ai. https://github.com/kvcache-ai/ktransformers
- <a id="ktransformers-lmsys"></a>`[KTransformers-LMSYS]` *KTransformers LMSYS Blog*. 2025-10. https://lmsys.org/blog/2025-10-22-KTransformers/
- <a id="lmcache-repo"></a>`[LMCache-repo]` *LMCache/LMCache*. https://github.com/LMCache/LMCache
- <a id="lmcache-ps-k8s"></a>`[LMCache-PS-K8s]` *vLLM Production Stack on K8s with LMCache*. https://blog.lmcache.ai/en/2025/01/21/high-performance-and-easy-deployment-of-vllm-in-k8s-with-vllm-production-stack/
- <a id="mooncake-repo"></a>`[Mooncake-repo]` *Mooncake (KVCache-centric)*. kvcache-ai. https://github.com/kvcache-ai/Mooncake
- <a id="nvidia-dynamo-repo"></a>`[NVIDIA-Dynamo-repo]` *ai-dynamo/dynamo + ai-dynamo/nixl*. https://github.com/ai-dynamo/dynamo
- <a id="llm-d-repo"></a>`[llm-d-repo]` *llm-d*. https://github.com/llm-d/llm-d
- <a id="aibrix-repo"></a>`[AIBrix-repo]` *AIBrix*. https://github.com/vllm-project/aibrix
- <a id="llumnix-repo"></a>`[Llumnix-repo]` *Llumnix (open-source migration scheduler)*. https://github.com/AlibabaPAI/llumnix

## Recent developments (2025–2026)

This section re-lists entries from May 2025 to May 2026 in flat chronological order. Where exact month is unclear from available sources, the entry is placed at the broadest plausible point in the year.

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
- `[PDTrim]` Targeted Pruning for PD.
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
- <a id="adaptive-thinking-2026"></a>`[Adaptive-Thinking-2026]` See `[Claude-Adaptive-Thinking-2026]` (Anthropic Adaptive Thinking GA).
- `[Qwen3-Next]` (Qwen3-Next reference card and recipes).
- `[Theory-AcceptDyn]` Acceptance Dynamics Across Cognitive Domains.

