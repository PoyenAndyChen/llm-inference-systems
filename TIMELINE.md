# Milestone Timeline: LLM Inference Infrastructure

A chronological reference covering key developments from late 2022 through mid-2026.
For depth, follow the cross-references to individual chapters.

---

## 2022

**2022-03:** **NVIDIA H100 (Hopper) announced** — First GPU with FP8 tensor cores and NVLink 4.0; sets the hardware baseline for the entire inference acceleration era. ([§70/01](70-hardware/01-nvidia-roadmap.md))

**2022-05:** **FlashAttention-1 (Dao et al.)** — IO-aware exact attention that fits the KV working set in SRAM, cutting memory traffic by ~5–10×; NeurIPS 2022. ([§10/01](10-engine-core/01-attention-kernels.md))

**2022-07:** **ORCA / Iteration-level Batching (Yu et al., OSDI '22)** — Replaces request-level batching with per-iteration scheduling, enabling continuous batching and dramatically improving GPU utilization. ([§10/03](10-engine-core/03-batching-scheduling.md))

**2022-08:** **LLM.int8() (Dettmers et al.)** — Mixed-precision INT8 inference via outlier decomposition; NeurIPS 2022. First practical post-training quantization for 100B-scale models. ([§10/04](10-engine-core/04-quantization.md))

**2022-09:** **FP8 Formats Paper (Micikevicius et al.)** — Defines E4M3 and E5M2 formats; lays theoretical groundwork for hardware-accelerated FP8 training and inference. ([§10/04](10-engine-core/04-quantization.md))

**2022-10:** **GPTQ (Frantar et al.)** — Layer-wise second-order weight quantization; ICLR 2023. Enables 3–4-bit quantization of GPT-scale models with near-lossless quality, making local inference viable. ([§10/04](10-engine-core/04-quantization.md))

**2022-11:** **Speculative Decoding — Leviathan et al.** — Proposes draft-then-verify decoding using a small draft model; ICML 2023. First paper to formalize the approach with lossless correctness guarantees. ([§10/05](10-engine-core/05-speculative-decoding.md))

**2022-11:** **SmoothQuant (Xiao et al.)** — Migrates quantization difficulty from activations to weights via per-channel scaling; ICML 2023. Enables smooth INT8 deployment across transformer layers. ([§10/04](10-engine-core/04-quantization.md))

---

## 2023 Q1–Q2

**2023-02:** **Speculative Decoding — Chen et al. (DeepMind)** — Independent, concurrent formulation of speculative decoding with empirical validation on large models.

**2023-03:** **FlexGen (Sheng et al., ICML 2023)** — Offloading-based inference for single-GPU hosts; enables 30B+ model inference on consumer hardware by tiling KV cache and weights across DRAM/disk. ([§20/05](20-distributed-inference/05-heterogeneous-inference.md))

**2023-06:** **AWQ: Activation-aware Weight Quantization (Lin et al.)** — Identifies salient weight channels via activation magnitudes; MLSys 2024 Best Paper. Achieves better accuracy than GPTQ at 4-bit with hardware-friendly kernels. ([§10/04](10-engine-core/04-quantization.md))

---

## 2023 Q3

**2023-07:** **FlashAttention-2 (Dao)** — Reduces non-matmul FLOPs ~4×, improves GPU utilization via better parallelism across query heads and sequence dimension; ICLR 2024. ([§10/01](10-engine-core/01-attention-kernels.md))

**2023-08:** **Sarathi (Preprint)** — Introduces chunked prefill: interleaves prefill chunks with decode tokens in the same micro-batch, eliminating decode stalls and improving throughput uniformity. ([§10/03](10-engine-core/03-batching-scheduling.md))

**2023-09:** **vLLM / PagedAttention (Kwon et al., SOSP '23)** — OS-style virtual memory for KV cache, eliminating fragmentation and enabling near-zero waste in dynamic batching; triggers broad industry adoption of paged KV management. ([§10/02](10-engine-core/02-paged-kv-memory.md))

**2023-09:** **OCP MX v1.0 Microscaling Specification** — Defines MXFP8, MXFP6, MXFP4 formats with shared block exponents; industry-standard foundation for sub-8-bit hardware. ([§10/04](10-engine-core/04-quantization.md))

**2023-10:** **FlashDecoding (Tri Dao et al., blog)** — Parallelizes attention over the KV sequence dimension for long-context decode; enables high-throughput single-request serving for long contexts. ([§10/01](10-engine-core/01-attention-kernels.md))

**2023-11:** **Splitwise (Patel et al.)** — First published proposal for prefill–decode disaggregation across separate GPU pools; ISCA 2024 Best Paper. ([§20/02](20-distributed-inference/02-prefill-decode-disagg.md))

---

## 2023 Q4 – 2024 Q1

**2024-01:** **DistServe (Zhong et al.)** — Implements PD disaggregation with KV migration; shows disaggregation eliminates prefill–decode SLO interference; OSDI 2024. ([§20/02](20-distributed-inference/02-prefill-decode-disagg.md))

**2024-01:** **TetriInfer** — Heterogeneous disaggregation approach co-submitted with DistServe; explores mixed-hardware PD scheduling.

**2024-01:** **Medusa (Cai et al., ICML 2024)** — Multi-head speculative decoding: auxiliary heads on the base model predict multiple tokens without a separate draft model. ([§10/05](10-engine-core/05-speculative-decoding.md))

**2024-01:** **EAGLE-1 (Li et al., ICML 2024)** — Draft-model speculative decoding via a lightweight feature-level autoregressive model; 3–4× speedup over vanilla decoding. ([§10/05](10-engine-core/05-speculative-decoding.md))

**2024-02:** **KIVI (Liu et al., ICML 2024)** — Asymmetric KV cache quantization (keys INT2 / values INT4 with full-precision pivots); reduces KV memory by ~2× with negligible quality loss. ([§30/01](30-kv-cache/01-kv-compression.md))

**2024-02:** **DeepSeekMath / GRPO** — Introduces Group Relative Policy Optimization for math reasoning; later becomes the training recipe behind DeepSeek-R1.

---

## 2024 Q2

**2024-03:** **Sarathi-Serve (Preprint; OSDI 2024)** — Production-grade chunked prefill system; demonstrates SLO improvements across diverse workloads and becomes the standard approach in major engines.

**2024-04:** **QuaRot (Ashkboos et al., NeurIPS 2024)** — Rotation-based quantization that distributes outliers uniformly; enables INT4 weight+activation quantization for large models. ([§10/04](10-engine-core/04-quantization.md))

**2024-04:** **Multi-Token Prediction / Gloeckle et al. (ICML 2024)** — Meta shows training with parallel prediction of N future tokens improves reasoning and enables inference-time multi-token generation. ([§10/06](10-engine-core/06-multi-token-prediction.md))

**2024-05:** **DeepSeek-V2 / MLA (Multi-head Latent Attention)** — Compresses KV cache via low-rank projection, reducing KV memory by ~5.75× relative to MHA at equivalent model quality. ([§10/01](10-engine-core/01-attention-kernels.md))

**2024-06:** **Mooncake (Qin et al.)** — Prefix-aware KV cache store for prefill–decode disaggregation across heterogeneous clusters; FAST 2025 Best Paper. ([§20/02](20-distributed-inference/02-prefill-decode-disagg.md))

**2024-06:** **EAGLE-2 (Li et al., EMNLP 2024)** — Context-aware dynamic tree drafting improves acceptance rate significantly over EAGLE-1. ([§10/05](10-engine-core/05-speculative-decoding.md))

**2024-06:** **Helix (Jiang et al., ASPLOS 2025)** — Heterogeneous GPU cluster inference via pipeline+tensor parallel search; handles mixed-generation hardware in a single serving pool. ([§20/05](20-distributed-inference/05-heterogeneous-inference.md))

---

## 2024 Q3

**2024-07:** **FlashAttention-3 (Shah et al., NeurIPS 2024)** — Exploits Hopper-specific hardware: WGMMA, TMA, and FP8 quantized attention; first attention kernel to reach 1.5+ PFLOPS sustained on H100. ([§10/01](10-engine-core/01-attention-kernels.md))

**2024-07 (approx.):** **H2O / Heavy Hitters Oracle** — KV cache eviction based on accumulative attention scores; enables streaming inference without full KV storage. ([§30/01](30-kv-cache/01-kv-compression.md))

**2024-07 (approx.):** **SnapKV** — Cluster-and-prune KV eviction at prefill time; reduces KV footprint for long-document prompts. ([§30/01](30-kv-cache/01-kv-compression.md))

---

## 2024 Q4

**2024-09:** **veRL / HybridFlow (Sheng et al.)** — Open-source RL training framework with hybrid actor-critic resource sharing; enables efficient GRPO/PPO at scale. ([§60/06](60-adjacent-workloads/06-rl-post-training-infrastructure.md))

**2024-09:** **OpenAI o1** — First commercially deployed chain-of-thought reasoning model; establishes the inference-time compute scaling paradigm that reshapes capacity planning. ([§60/01](60-adjacent-workloads/01-test-time-compute.md))

**2024-10:** **ThunderKittens (Spector et al.)** — CUDA DSL targeting Hopper's warpgroup primitives; used to implement production attention kernels at Google DeepMind and elsewhere.

**2024-10:** **POD-Attention (Agrawal et al., ASPLOS 2025)** — Unified attention kernel for heterogeneous prefill+decode batches; reduces kernel dispatch overhead in chunked prefill pipelines.

**2024-10 (approx.):** **H200 (Hopper refresh)** — 141 GB HBM3e; doubles memory bandwidth and capacity vs H100 SXM5; bridges GPU supply gap ahead of Blackwell.

**2024-11:** **LTR / Learning to Rank Preemptions (NeurIPS 2024)** — ML-based scheduler learns to predict request completion time, reducing preemptions and improving latency-SLO compliance. ([§10/03](10-engine-core/03-batching-scheduling.md))

**2024-12:** **SGLang v0.4 — Zero-overhead scheduler** — Decouples tokenization/detokenization from the inference loop with an async event-driven scheduler; halves scheduling overhead for high-QPS workloads. ([§80/02](80-oss-deep-dives/02-sglang.md))

**2024-12:** **DeepSeek-V3** — 671B MoE model with MTP training; introduces Multi-Token Prediction at inference, improving token throughput by ~1.8×. ([§10/06](10-engine-core/06-multi-token-prediction.md))

**2024-12:** **GB200 NVL72 — First Cloud Shipments (CoreWeave)** — 72-GPU NVLink-domain rack; 1.4 TB/s NVLink 5.0 bandwidth; marks commercial availability of Blackwell for inference. ([§70/01](70-hardware/01-nvidia-roadmap.md))

**2024-12:** **AWS Trainium2 GA** — Amazon's second-generation ML accelerator generally available; enables cost-competitive LLM training and inference on AWS. ([§70/03](70-hardware/03-asics-hyperscaler.md))

---

## 2025 Q1

**2025-01:** **FlashInfer (Ye et al., MLSys 2025 Best Paper)** — Composable, JIT-compiled attention kernel library with unified paged/ragged KV support and FP8 MLA kernels. ([§10/01](10-engine-core/01-attention-kernels.md))

**2025-01:** **vLLM V1 alpha** — Complete engine rewrite: async tokenizer, disaggregated scheduler, zero-copy KV transfer API, and prefix caching redesign. ([§80/01](80-oss-deep-dives/01-vllm.md))

**2025-01:** **DeepSeek-R1 / RLVR** — Open-weight reasoning model trained with GRPO; triggers large-scale deployment of long-chain-of-thought inference and motivates reasoning-aware scheduling. ([§60/01](60-adjacent-workloads/01-test-time-compute.md))

**2025-02:** **DeepSeek Open Source Week** — Single week releases: DeepEP (expert-parallel communication), DualPipe (compute-communication overlap), EPLB (expert load balancer), DeepGEMM (FP8 GEMM library), FlashMLA (MLA decode kernel for Hopper). ([§20/03](20-distributed-inference/03-moe-inference.md))

**2025-02:** **FlashMLA (DeepSeek)** — Hopper-optimized decode kernel for Multi-head Latent Attention; reaches ~3000 GB/s effective bandwidth at batch size 1. ([§10/01](10-engine-core/01-attention-kernels.md))

**2025-02:** **NSA / Native Sparse Attention (Yuan et al.)** — Block-sparse attention with hardware-aligned tile sizes; 9× faster than dense FlashAttention at 64K context; ACL 2025. ([§10/01](10-engine-core/01-attention-kernels.md))

**2025-02:** **AIBrix launch (ByteDance)** — Kubernetes-native LLM serving platform; provides KV cache routing, load balancing, and model management primitives for cloud-scale deployments. ([§80/07](80-oss-deep-dives/07-aibrix.md))

**2025-02:** **Envoy AI Gateway** — Open-source LLM-aware reverse proxy; adds token-rate limiting, semantic routing, and model failover on top of the Envoy data plane.

**2025-03:** **EAGLE-3 (arXiv:2503.01840; NeurIPS 2025)** — Training-free inter-layer feature aggregation removes the draft-model accuracy bottleneck; first speculative decoding system to reach 5× speedup on aligned models. ([§10/05](10-engine-core/05-speculative-decoding.md))

**2025-03:** **NVIDIA Dynamo GA (GTC)** — Distributed inference runtime with disaggregated prefill/decode, KV routing, and multi-node NIXL transport; open-sourced at GTC. ([§20/02](20-distributed-inference/02-prefill-decode-disagg.md))

**2025-03:** **NIXL Open-sourced (GTC)** — NVIDIA Inference Xfer Library; low-latency, zero-copy KV block transfer library across GPUs and nodes for disaggregated serving.

**2025-03:** **DGX Spark / DGX Workstation — Customer Shipments begin** — Blackwell-based desktop/workstation systems; democratizes high-bandwidth inference for researchers outside data centers. ([§70/01](70-hardware/01-nvidia-roadmap.md))

---

## 2025 Q2

**2025-04:** **MegaScale-Infer / AFD (Adaptive Flow Disaggregation, SIGCOMM 2025)** — ByteDance's production disaggregated inference system; dynamic prefill/decode ratio adjustment based on real-time load. ([§20/02](20-distributed-inference/02-prefill-decode-disagg.md))

**2025-04:** **TurboQuant / Google (arXiv:2504.19874; ICLR 2026)** — Rotation + Hadamard based quantization achieving INT4 quality parity with FP16 on Gemma-scale models. ([§10/04](10-engine-core/04-quantization.md))

**2025-04:** **BitNet b1.58 2B-4T** — Microsoft 1-bit language model at scale (2B params, 4T tokens); inference runs entirely in INT1 GEMM on CPU-scale hardware without GPU.

**2025-04:** **TileLang** — Tile-level IR for GPU kernel programming; extends beyond attention to fusion-friendly general tensor ops.

**2025-05:** **DeepSeek SGLang 96-GPU EP Demo** — 96× H100 all-expert-parallel serving; demonstrates near-linear throughput scaling for MoE with expert parallelism in SGLang. ([§20/03](20-distributed-inference/03-moe-inference.md))

**2025-05:** **NVLink Fusion announced (Computex)** — NVIDIA opens NVLink to third-party SoCs and CPUs; allows non-NVIDIA logic to attach to NVLink fabrics, enabling custom accelerator ecosystems. ([§70/01](70-hardware/01-nvidia-roadmap.md))

**2025-05:** **AReaL (arXiv:2505.24298)** — Asynchronous RLVR training with decoupled rollout and update workers; achieves higher GPU utilization than synchronous veRL at multi-node scale. ([§60/06](60-adjacent-workloads/06-rl-post-training-infrastructure.md))

**2025-05:** **Quartet (NeurIPS 2025)** — Joint weight-and-activation MXFP4 quantization method using block-wise scaling; targets Blackwell NVFP4 hardware instructions. ([§10/04](10-engine-core/04-quantization.md))

**2025-05:** **llm-d launch (Red Hat Summit)** — Kubernetes-native disaggregated LLM serving daemon; KV-aware request routing, P/D lifecycle management. ([§50/01](50-cluster-systems/01-router-gateway.md))

**2025-06:** **Ultra Ethernet 1.0 Specification released** — UEC consortium releases v1.0; defines RDMA-compatible, lossless 400G/800G Ethernet for AI cluster fabrics; first open alternative to InfiniBand at scale.

**2025-06:** **llm-d accepted to CNCF Sandbox** — Signals ecosystem commitment and standard governance; accelerates vendor integrations. ([§50/01](50-cluster-systems/01-router-gateway.md))

**2025-06 (approx.):** **GIE InferencePool v1 stable** — Gateway API Inference Extension adds InferencePool and InferenceModel CRDs to Kubernetes; vendor-neutral model routing becomes part of the cloud-native API surface. ([§50/01](50-cluster-systems/01-router-gateway.md))

---

## 2025 Q3

**2025-07:** **Kimi K2 (Moonshot AI)** — ~1T parameter MoE model serving with expert parallelism across 300+ GPU nodes; demonstrates MoE inference at hyperscaler scale from a non-US lab. ([§20/03](20-distributed-inference/03-moe-inference.md))

**2025-07:** **slime (RL infra)** — Lightweight async RL post-training library optimized for long-context rollouts and partial episode storage. ([§60/06](60-adjacent-workloads/06-rl-post-training-infrastructure.md))

**2025-07:** **SpecForge** — Automated speculative decoding draft-head generation and calibration tool; simplifies adding spec-dec to arbitrary base models.

**2025-08:** **GB300 / B300 — First Cloud Instances (CoreWeave)** — Blackwell refresh with enhanced FP4 throughput; first public instances Aug 19 2025. ([§70/01](70-hardware/01-nvidia-roadmap.md))

**2025-08:** **GPT-4o Open Weights (GPT-OSS) with MXFP4** — OpenAI releases production model weights quantized to MXFP4; first major public MXFP4 inference deployment at scale. ([§10/04](10-engine-core/04-quantization.md))

**2025-08:** **TaiChi** — Disaggregated inference scheduler with cross-cluster KV migration; extends Mooncake-style prefix routing to multi-datacenter scenarios.

**2025-08:** **HeteroScale** — Mixed-generation GPU (H100/H200/GB200) cluster inference; autoscales across hardware tiers with heterogeneity-aware batching. ([§20/05](20-distributed-inference/05-heterogeneous-inference.md))

**2025-09:** **AMD MI350X / MI355X (CDNA4) — Sampling** — First AMD GPUs with native FP4 matrix units; 288 GB HBM3e; direct Blackwell competition for inference deployments. ([§70/02](70-hardware/02-amd-and-non-nvidia-gpu.md))

**2025-09:** **SPEC-RL (arXiv:2509.23232)** — Speculative decoding integrated into the RL training loop; uses the draft model as a value baseline, reducing rollout cost during RLVR training. ([§60/06](60-adjacent-workloads/06-rl-post-training-infrastructure.md))

**2025-09:** **Rubin CPX rack announced** — NVIDIA's next-generation compute-plane switch for Rubin GPU racks; preview of the Rubin interconnect architecture. ([§70/01](70-hardware/01-nvidia-roadmap.md))

**2025-10:** **KTransformers (SOSP 2025)** — Heterogeneous CPU+GPU inference for MoE models; routes expert layers through large CPU DRAM while compute-intensive layers run on GPU; subsequently pivoted to SGLang kernel library (Oct 2025). ([§80/10](80-oss-deep-dives/10-others.md))

**2025-10 (approx.):** **Triton-Anatomy v1** — Open-source Triton kernel decomposition library; educational and production reference for writing fused attention/norm/quantize kernels in Triton.

**2025-11:** **TPU v7 Ironwood — Commercialized** — Google's seventh-generation TPU with doubled HBM capacity vs v6e; available in GKE and Vertex AI. ([§70/03](70-hardware/03-asics-hyperscaler.md))

**2025-11:** **Perplexity TransferEngine** — Production KV transfer component for disaggregated prefill; co-developed with Mooncake principles; open-sourced Nov 2025.

**2025-12:** **DeepSeek-V3.2 / DSA (Deep Seek Sparse Attention, arXiv:2512.02556)** — Extends NSA-style sparse attention to the production V3 stack; enables 128K+ context at MoE serving cost. ([§10/01](10-engine-core/01-attention-kernels.md))

**2025-12:** **Janus-AFD** — ByteDance follow-on to MegaScale-Infer AFD; adds joint prefill–decode–prefetch scheduling with look-ahead KV prefetch. ([§20/02](20-distributed-inference/02-prefill-decode-disagg.md))

**2025-12:** **vLLM Router v1 (Dec 2025)** — Standalone session-affinity router for vLLM clusters; KV-prefix-hash based routing to maximize prefix cache hit rates. ([§50/01](50-cluster-systems/01-router-gateway.md))

**2025-12:** **Speculators v0.3.0** — Hugging Face spec-dec integration library; supports EAGLE-2/3 heads, Medusa, and custom draft models across transformers-compatible engines.

---

## 2026 Q1

**2026-01:** **Microsoft Maia 200 — First Deployed** — In-house inference chip deployed in Azure data centers; targets cost-per-token for Azure OpenAI service workloads. ([§70/03](70-hardware/03-asics-hyperscaler.md))

**2026-01:** **AMD MI400 / Helios announced (CES)** — Next-generation AMD GPU with modular chiplet interconnect; ships H2 2026. ([§70/02](70-hardware/02-amd-and-non-nvidia-gpu.md))

**2026-01:** **Rubin R100 — Full Production (CES 2026)** — NVIDIA confirms Rubin chips in full production; rack-level GA expected H2 2026. ([§70/01](70-hardware/01-nvidia-roadmap.md))

**2026-02:** **llm-d v0.5** — Major release; adds NIXL-based KV transfer, full InferencePool conformance, and multi-model session affinity. ([§50/01](50-cluster-systems/01-router-gateway.md))

**2026-03:** **FlashAttention-4 (arXiv:2603.05451)** — Blackwell-native attention kernel exploiting FP4 MMA and Tensor Memory Accelerator v2; reaches 1605 TFLOPs on B200, 1.7× over FA-3. ([§10/01](10-engine-core/01-attention-kernels.md))

**2026-03:** **NCCL-EP (arXiv:2603.13606)** — NCCL extension for expert-parallel all-to-all communication; reduces MoE EP collective overhead by ~30% vs baseline NCCL on InfiniBand. ([§20/03](20-distributed-inference/03-moe-inference.md))

**2026-03:** **NVIDIA Groq 3 LPX announced (GTC)** — NVIDIA's first product based on acquired Groq IP; wafer-scale streaming architecture integrated into NVIDIA data-center rack; announced but not shipped as of May 2026. ([§70/01](70-hardware/01-nvidia-roadmap.md))

**2026-03:** **Rubin R100 Rack — Customer Shipments begin** — Early access customers receive Rubin rack systems with NVLink 6.0 fabric; widely available H2 2026 per roadmap. ([§70/01](70-hardware/01-nvidia-roadmap.md))

**2026-04:** **RouterWise** — Learned KV-cache-aware routing policy trained on historical prefix hit statistics; deployed in multi-tenant serving clusters.

**2026-04:** **PrefillAaS (Prefill as a Service)** — Decoupled prefill microservice with SLA contracts; extends disaggregated serving to cross-organizational prefill offloading. ([§20/02](20-distributed-inference/02-prefill-decode-disagg.md))

---

## 2026 Q2

**2026-05:** **NeMo-RL spec-dec v0.6** — NVIDIA's NeMo reinforcement learning library adds speculative decoding rollout acceleration; reduces rollout wall-clock time during RLVR training by up to 2×. ([§60/06](60-adjacent-workloads/06-rl-post-training-infrastructure.md))

**2026-05 (approx.):** **AMD MI400 / Helios — Engineering Samples** — Early silicon delivered to partners ahead of H2 2026 GA; first CDNA5 hardware visible in the field. ([§70/02](70-hardware/02-amd-and-non-nvidia-gpu.md))

---

*Entries marked "announced but not shipped as of May 2026" reflect hardware or software where public availability had not been confirmed at the time of writing. Quarters marked "(approx.)" indicate best-estimate placement from available sources; exact dates may differ.*
