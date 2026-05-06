# Quantization

**After reading this chapter, the reader will be able to:**

- Decompose any production quantization scheme along three orthogonal axes — *what* is quantized (weights, weights+activations, KV cache), *which numeric format* is used (INT4/8, FP8, FP4, NF4, MX-family, NVFP4), and *which calibration recipe* produced the scales (GPTQ, AWQ, HQQ, SmoothQuant, OmniQuant, AutoRound, plus rotation transforms as a preprocessing layer) — and reason about why each axis exists.
- Read a hardware-support matrix and a per-engine production matrix and predict, for a given (model, accelerator, engine, latency target) tuple, which quantization scheme will run, where it will be bandwidth- versus compute-limited, and what calibration toolchain the deployment must own.
- Place KV-cache quantization (KIVI, KVQuant, TurboQuant) and the 1-bit frontier (BitNet b1.58) in the production landscape — what is shipping, what is research, and what is on the public roadmap as of mid-2026.

Decode is bandwidth-bound: the per-token roofline in [§00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md) shows that on H100-class hardware a dense forward pass spends most of its time waiting on weight bytes streamed from HBM. Halving the bytes of every weight halves decode-step time up to a batch where the operation becomes compute-bound; in the prefill regime, where arithmetic intensity is already high, the same halving doubles peak compute throughput by feeding tensor cores a denser format. Quantization enters the engine at two distinct points — a memory-bandwidth lever in decode, a peak-FLOP lever in prefill and large-batch — and the literature has bifurcated accordingly.

Three axes generate the design space: weight-only versus weight+activation versus KV-only on the *what*; integer versus FP8 versus FP4 versus block-scaled MX/NVFP4 on the *format*; GPTQ-family versus AWQ-family versus rotation-preprocessed versus QAT-light on the *recipe*. A scheme like "AWQ-W4A16 with QuaRot preprocessing on Hopper" is one cell of the cross-product; "NVFP4 W4A4 on Blackwell with FP8 KV" is another. The chapter walks the three axes in turn, then circles back to the KV-specific subfield, the 1-bit research frontier, and the per-engine landing matrix.

## 1. Axis 1 — what is quantized

The first decision is which tensors leave full precision. The three production patterns differ in what they save and what they cost.

### 1.1 Weights only (W-only)

In a weight-only scheme the model parameters are quantized once, offline, and stored in their compressed format. At inference the kernel dequantizes on-the-fly into a higher-precision register tile or fuses dequant into a mixed-precision matmul; activations remain in BF16 throughout. The win is purely on the memory side: each weight matmul in decode reads roughly $N/2$ bytes for INT8 or $N/4$ bytes for INT4 instead of $2N$ bytes for BF16, which translates almost linearly into decode TPOT in the bandwidth-bound regime. Compute is not faster per se — the GEMM still runs in BF16 accumulators — and at the saturation batch where decode crosses into compute-bound, weight-only quantization stops paying.

W4A16 is the dominant production weight-only configuration: what Marlin, Machete, AWQ-Marlin, and GPTQ-Marlin kernels target on H100; what `llama.cpp`'s Q4_K_M and IQ4_XS map to on edge hardware; the universal fallback when an engine needs to fit a model in HBM but the calibration budget is small. W8A16 is an awkward middle. The natural endpoints are W4A16 (squeeze decode bandwidth) and W8A8 (squeeze prefill compute).

### 1.2 Weights and activations (W+A)

When activations are also quantized, the matmul runs on the low-precision tensor-core path: FP8 weights times FP8 activations on Hopper, FP4 weights times FP4 activations on Blackwell. The peak compute ceiling rises — H100 FP8 peak is roughly 2× the BF16 peak, B200 FP4 peak roughly 4× the BF16 peak — and prefill and large-batch decode benefit. The cost is at calibration: activation distributions are dynamic and contain *outlier channels* whose magnitude is orders of magnitude above the median; a single dimension can dominate the dynamic range and make naive quantization catastrophic. The smoothing-and-rotation literature (§3) exists almost entirely because of this outlier problem.

A subtlety on the roofline: FP8 doubles the compute peak relative to BF16, but FP8 weights are also half the bytes, so arithmetic intensity per matmul tile (FLOPs per byte loaded) is roughly preserved. In pure bandwidth-bound decode, FP8 W+A is therefore not magically twice as fast as BF16 — it is at most as fast as W8A16 at the same batch — with the benefit cashing in only when batch size pushes the operating point past the now-doubled compute ridge [see §00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md). The same logic at NVFP4 W4A4 on Blackwell gives 4× the compute peak and 4× the memory savings, with the ridge again shifting outward.

### 1.3 KV cache only

The third pattern targets the K and V tensors in the cache. Matmul-side activations stay full precision; only the cached K and V are stored in INT4/INT8/FP8/NVFP4 and are dequantized inside the attention kernel. The savings are twofold: cache memory shrinks (larger batch or longer context at fixed HBM), and bandwidth on the attention reduction shrinks (faster decode at long context, where the attention term dominates per the per-token FLOP/byte derivation). The KV cache has its own outlier structure — keys carry persistent per-channel outliers that values do not — which is why KIVI and KVQuant treat K and V asymmetrically. KV-only quantization is orthogonal to weight and activation quantization; production stacks routinely combine FP8 weights with FP8 KV, or NVFP4 weights with NVFP4 KV. Section 5 develops the KV-specific work in detail.

### 1.4 The decision tree

A regime-by-regime guide:

- **Bandwidth-bound decode at small batch.** W4A16 is the right default; every byte saved on the weight stream comes back as decode latency, and activations cost nothing extra at BF16. Add FP8 KV or NVFP4 KV when context is long enough that KV bandwidth dominates.
- **Compute-bound prefill or large-batch decode.** W+A is the answer: FP8 W8A8 on Hopper or NVFP4 W4A4 on Blackwell. KV quantization layers on top.
- **HBM-tight regimes (consumer GPUs, edge, fitting a 70B-class model on one node).** W4A16 with KV quantization is the only path that fits.
- **Test-time-compute / long-CoT serving.** KV memory dominates because each request sustains ten- to hundred-thousand-token traces; aggressive KV quantization (FP8 today, NVFP4 or sub-3-bit on the 2026 roadmap) is the dominant lever.

## 2. Axis 2 — numeric format

The second decision is which low-precision number system stores the quantized values. Four families: integer, IEEE-style FP8, FP4, and block-scaled MX/NVFP4.

### 2.1 Integer formats

INT8 and INT4 store fixed-point values: a $b$-bit signed integer with a per-tensor or per-channel scale $s$ and optional zero-point $z$, decoded as $\hat{w} = s \cdot (q - z)$. INT8 is the longest-supported low-precision format; every tensor core from Turing onward implements INT8 matmul. INT4 has no native tensor-core dot product on NVIDIA generations through Hopper — Marlin and Machete kernels achieve INT4 weight-only throughput by *packing* two INT4 values per byte and dequantizing on the fly into an FP16 register tile, which is why W4A16 is the natural INT4 configuration: only the weights are sub-byte, the GEMM itself stays in FP16. For a symmetric uniform quantizer at $b$ bits, the scale is $s = \max|w| / (2^{b-1} - 1)$ and per-element MSE is $\Delta^2/12 = s^2/12$ where $\Delta$ is the bin width; outliers blow $\Delta$ up and degrade the in-distribution mass, which is why group-wise quantization (one scale per 64 or 128 contiguous weights) is the universal fix.

### 2.2 FP8 (E4M3 and E5M2)

FP8 is the first floating-point format in the LLM-quantization toolbox. The OCP/NVIDIA/ARM/Intel paper standardized two variants:

- **E4M3** — 4 exponent bits, 3 mantissa bits, max $\approx 448$. Smaller dynamic range, finer precision; the standard for forward-pass weights and activations.
- **E5M2** — 5 exponent bits, 2 mantissa bits, max $\approx 57344$. Larger dynamic range, coarser precision; reserved for gradients in training and occasionally for KV cache.

The dynamic-range formula $\text{max} = (2 - 2^{-m}) \cdot 2^{2^e - 1 - \mathrm{bias}}$ gives the spec values directly. Hopper H100 introduced 4th-generation tensor cores with native FP8 matmul; the marketed peak is roughly 1978 TFLOP/s FP8 on H100 SXM5, double the BF16 peak. FP8 W8A8 has been the default production W+A precision on Hopper since 2023; AMD MI300X and MI325X both ship FP8 (E4M3 and E5M2) tensor cores natively. Critically, FP8's exponent gives it intrinsic outlier robustness that integer formats lack: an INT8 scale chosen to fit a +200 outlier quantizes a +0.05 in-distribution activation to zero, while FP8 E4M3 represents both with reasonable relative error. This is a major reason FP8 displaced INT8 in W+A production once Hopper arrived — the same accuracy is reached without nearly as much smoothing or rotation preprocessing.

### 2.3 FP4 (E2M1)

FP4 packs four bits into one sign, two exponent, and one mantissa bit, with a representable maximum of 6 and 16 distinct nonzero values. The dynamic range is small enough that FP4 cannot stand alone — every block of values needs a per-block scale — which is what motivated the block-scaled formats below rather than per-tensor scalar FP4. Hopper has no native FP4 tensor core; FP4 on Hopper is software-emulated. Blackwell B200's 5th-generation tensor cores are the first generation with native FP4 (and FP6) matmul, and the marketed peak on B200 is roughly 4× the BF16 peak. AMD MI300X does not ship native FP4.

### 2.4 NF4

NF4 (NormalFloat-4) is the 4-bit non-uniform scalar format introduced for QLoRA fine-tuning. Its 16 representable values are placed at the quantiles of a unit normal distribution: under the assumption that pre-trained weights are approximately normally distributed within a layer, this gives information-theoretically optimal bin placement for scalar quantization. NF4 is not a hardware tensor-core format; it exists as a software dequantization path (a 16-entry lookup table) inside `bitsandbytes`. Widely deployed for LoRA-style fine-tuning; rare in serving stacks because lookup-table dequant is slower than uniform dequant on tensor-core paths.

### 2.5 MX-family (Microscaling)

The OCP MX v1.0 specification standardized a family of *block floating-point* formats, each defined by block size $k$, per-element bit width $d$, and shared scale width $w$. A block of $k$ contiguous elements share one $w$-bit exponent scale; each element has its own $d$-bit private representation. The encoded value is $\hat{x}_i = X \cdot P_i$ where $X$ is the shared scale (E8M0 in MX) and $P_i$ is the private element.

Concrete formats: **MXFP8** ($k{=}32$, $d{=}8$ as E4M3 or E5M2, $w{=}8$, 264 bits per block); **MXFP6** ($d{=}6$ as E2M3 or E3M2, 200 bits per block); **MXFP4** ($d{=}4$ as E2M1, 136 bits or 17 bytes per 32-element block); **MXINT8** (block-scaled INT8). The shared scale lets each block locally adapt to its own magnitude, recovering the dynamic-range robustness that scalar 4-bit lacks. Blackwell ships native MXFP8, MXFP6, and MXFP4 tensor cores (5th generation); AMD MI300X supports MXFP8 natively and reaches MXFP4/MXFP6 through Quark and ROCm software paths; Hopper has no native MX-family tensor cores. The OCP MX paper reports accuracy parity with FP8 at 4-bit W+A for generative LMs; vendor-side data tends to claim under 1% accuracy drop on instruction-tuned baselines, hedged here as a vendor figure.

OpenAI's August-2025 GPT-OSS release shipped natively in MXFP4 — the first major open-weight model to do so, with `gpt-oss-120B` fitting in 80 GB and `gpt-oss-20B` in 16 GB. The release forced vLLM, SGLang, TRT-LLM, llama.cpp, and Ollama to ship MXFP4 inference paths within weeks.

### 2.6 NVFP4

NVFP4 is NVIDIA's variant of block-scaled FP4, distinct from MXFP4 in two ways: blocks are 16 elements instead of 32, and the per-block scale is FP8 (E4M3) rather than E8M0. NVFP4 also adds a per-tensor FP32 scale on top, giving two-level scaling $\hat{x}_i = S_{\mathrm{global}} \cdot S_{\mathrm{block}} \cdot P_i$. NVIDIA-reported accuracy on instruction-tuned baselines is within 1% of FP16; reported memory savings are 3.5× over FP16 and 1.8× over FP8; throughput claims of roughly 4× on B200 versus H100 FP8 for DeepSeek-R1-class workloads appear in NVIDIA and SGLang blog posts and should be read as vendor-supplied. NVFP4 is supported on B200/B300/GB200/GB300, RTX PRO 6000, DGX Spark, and Jetson Thor; it is the recommended W4A4 inference precision for Blackwell. The NVFP4 KV cache variant extends the format to KV tensors via `KvCacheConfig(dtype='nvfp4')` in TRT-LLM and an analogous path in vLLM.

NVFP4 also has a pre-training story. NVIDIA's *Pretraining Large Language Models with NVFP4* (2025-09, arXiv:2509.25149) trained a 12B model on 10T tokens in NVFP4 with parity to an FP8 baseline; IST-DASLab's Quartet and Quartet II (2026-01, arXiv:2601.22813) report similar results using random Hadamard transforms with stochastic or unbiased micro-scale rounding. The implication is that future frontier checkpoints may arrive natively in NVFP4, removing the PTQ step entirely. As of mid-2026, no public frontier-scale (>500B) model has shipped trained end-to-end in FP4.

### 2.7 Hardware-support matrix

The format axis maps onto hardware in a sparse way:

| Format | A100 | H100/H200 | B200/B300 | MI300X / MI325X |
|---|:---:|:---:|:---:|:---:|
| INT8 | ✓ | ✓ | ✓ | ✓ |
| FP8 (E4M3 / E5M2) | — | ✓ | ✓ | ✓ |
| FP4 / NVFP4 | — | — | ✓ | — |
| MXFP4 | — | — | ✓ | — (software) |
| MXFP8 | — | — | ✓ | ✓ |

Cells marked "—" indicate no native tensor-core support; some can be reached through software emulation (FP4 on Hopper through Marlin-style packing; MXFP4 on AMD through Quark) at reduced throughput. INT4 is omitted because no NVIDIA generation through Hopper ships an INT4 tensor-core dot product in a GEMM shape; INT4 weight-only paths pack in software and decode into FP16 inside the kernel. The *what* and *format* axes are therefore not freely composable: W4A4 NVFP4 requires Blackwell, and A100 has FP16 W+A as its only tensor-core option below BF16.

TPU v5/v6 (Trillium) and v7 (Ironwood) ship INT8 native and INT4 via the AQT path; Ironwood is *not* "v8" and does not ship FP4 tensor cores — earlier trade-press framing conflated v8 roadmap rumors with the actual v7 silicon. Intel Gaudi 3 supports INT8 and FP8.

## 3. Axis 3 — calibration recipe

The third decision is how scales (and zero-points, and per-block exponents) are *chosen*. PTQ algorithms operate offline on a calibration corpus and produce a quantized checkpoint; QAT algorithms fine-tune with fake-quantization in the loop. The recipe choices group into families along this PTQ/QAT line, with rotation transforms forming a preprocessing layer that any recipe can adopt.

### 3.1 GPTQ — second-order weight quantization

GPTQ ([Frantar et al., 2022](https://arxiv.org/abs/2210.17323), ICLR 2023) is the foundational weight-only PTQ algorithm; most W4A16 production checkpoints descend from it. It is a layer-wise application of the optimal-brain-surgeon principle: for a linear layer $Y = W X$ with calibration activations $X$, GPTQ minimizes $\| W X - \hat{W} X \|_F^2$ by quantizing one column of $W$ at a time, updating the remaining unquantized columns to compensate for the introduced error. The OBS update is

$$\delta_F \;=\; -\frac{w_q}{[\mathbf{H}^{-1}]_{qq}} \cdot \mathbf{H}^{-1}_{:,q},$$

where $\mathbf{H} = X X^{\top}$ is the Hessian of the calibration loss. The trick that makes GPTQ tractable at LLM scale is fixing the column ordering up-front so the inverse Hessian is computed once; without it, OBS would re-invert after each column. GPTQ's standard configuration is W4A16 with group size 128, calibrated on ~128 sequences of 2K tokens, finishing in minutes for 7B-class models. The Marlin and Machete kernel families ([Marlin, arXiv:2402.05137](https://arxiv.org/abs/2402.05137); Machete is Hopper-tuned) are the W4A16 inference kernels that pair with GPTQ checkpoints in production.

### 3.2 AWQ — activation-aware weight quantization

AWQ ([Lin et al., 2023](https://arxiv.org/abs/2306.00978), MLSys 2024 Best Paper) observes that not all weights are equally important — the weights aligned with high-magnitude activation channels matter most. AWQ identifies the salient channels (typically the top ~1% by activation magnitude), then scales the input channels to make those weights survive quantization at the cost of slightly larger error elsewhere. Concretely, $Y = WX$ is reparametrized as $Y = (W \cdot \mathrm{diag}(s)) \cdot (\mathrm{diag}(s)^{-1} X)$ for a chosen $s$; only $W \cdot \mathrm{diag}(s)$ is quantized, and the input scaling is fused into preceding layers. AWQ ships in vLLM (`awq`, `awq_marlin`) and TRT-LLM (`W4A16_AWQ`, `W4A8_AWQ`); calibration is faster than GPTQ's, and 4-bit quality is competitive. For W4A16 production, GPTQ and AWQ are interchangeable defaults.

### 3.3 HQQ — half-quadratic, no-calibration

HQQ (Half-Quadratic Quantization, [Badri & Shaji, 2023](https://mobiusml.github.io/hqq_blog/)) eliminates the calibration step. Per-element rounding is formulated as a convex optimization that minimizes $\|W - \hat{W}\|$ in a robust norm, solved in closed form per group via half-quadratic splitting. HQQ quantizes a 70B-class model in single-digit minutes on a single GPU, roughly 50× faster than GPTQ; quality is competitive at 4-bit and usable at 3-bit. HQQ is the default when calibration data is unavailable or not legally clean for the deployment.

### 3.4 AutoRound — sign-gradient learned rounding

AutoRound ([Cheng et al., 2023](https://arxiv.org/abs/2309.05516)) makes the rounding decision a learnable variable. For each weight $w$, instead of round-to-nearest the algorithm learns a per-weight rounding offset $v \in [-0.5, 0.5]$ so the dequantized value is $s \cdot \mathrm{round}(w/s + v)$, plus per-group clip parameters. Updates use signed gradient descent (SignSGD) on a small calibration loss. AutoRound is competitive at 2- and 3-bit weight-only configurations where vanilla GPTQ degrades; Intel's Neural Compressor and vLLM's LLM Compressor (since the 2025-12 integration) both ship AutoRound as a first-class recipe.

### 3.5 OmniQuant — QAT-light

OmniQuant ([Shang et al., 2023](https://arxiv.org/abs/2308.13137), ICLR 2024 Spotlight) sits in a distinct family from the GPTQ/AWQ/HQQ line. It introduces two learnable components — Learnable Weight Clipping (LWC, a per-channel clip threshold) and Learnable Equivalent Transformation (LET, a learnable smoothing transform more expressive than SmoothQuant's) — and trains both via gradient descent on a calibration set, block by block. Crucially, the base model weights are not updated; only the clipping and transformation parameters are. This places OmniQuant between PTQ and full QAT: it uses gradient steps and calibration data, but it does not fine-tune the model. The classification matters because OmniQuant's family is not GPTQ's: analytic versus learned. OmniQuant is computationally heavier (hours not minutes for 70B models) but pulls ahead at extreme bit-widths W2A16 and W4A4 where analytic methods leave accuracy on the floor.

### 3.6 SmoothQuant — outlier migration for W+A

SmoothQuant ([Xiao et al., 2022](https://arxiv.org/abs/2211.10438), ICML 2023) targets the W+A regime. Activation tensors in trained transformers contain a small number of channels whose magnitude is far above the median; per-tensor INT8 scales chosen to fit them crush the rest of the dynamic range, and per-channel scales break matmul algebra. SmoothQuant's fix is the equivalent transformation

$$Y = W X = \big(W \cdot \mathrm{diag}(s)\big) \cdot \big(\mathrm{diag}(s)^{-1} X\big),$$

which is mathematically identity and chooses $s_j = \max(|X_{:,j}|)^{\alpha} / \max(|W_{j,:}|)^{1-\alpha}$ with $\alpha \approx 0.5$. Quantization difficulty migrates from activation to weight: outlier activation channels are scaled down (easier to quantize), and the corresponding weight columns are scaled up (slightly harder, but weights are well-behaved). The scaling fuses offline into the preceding RMSNorm at zero runtime cost. SmoothQuant enabled INT8 W8A8 at near-FP16 quality and was the reference INT8 recipe; it has been largely displaced in production by FP8 W8A8 — FP8's exponent absorbs outliers without an explicit smoothing step — but the *idea* of migrating difficulty between weight and activation persists in OmniQuant's LET, in the rotation methods, and in every modern W4A4 recipe.

### 3.7 Rotation transforms as preprocessing — QuaRot, SpinQuant, DuQuant, FlatQuant

A separate body of work attacks the outlier problem by rotating the weight (and, where possible, the activation) into a basis where outliers are smeared away. For $x \in \mathbb{R}^d$ with bounded $\|x\|_2$, multiplication by a random orthogonal $Q$ makes each coordinate of $Qx$ concentrate with variance $O(\|x\|_2^2 / d)$ — outliers in any single coordinate are suppressed by $\sqrt{d}$, and a per-coordinate scalar quantizer becomes accurate. Random Hadamard matrices give the same effect deterministically and are cheap to apply. A second property: RMSNorm is rotation-invariant up to its own scaling, so an orthogonal $Q$ inserted at an RMSNorm boundary (with $Q^{\top} Q = I$ inserted on the other side and fused into the next weight) leaves the model output unchanged. The chain of RMSNorm-bordered linear layers therefore admits offline-fused rotations without retraining.

- **QuaRot** ([Ashkboos et al., 2024](https://arxiv.org/abs/2404.00456), NeurIPS 2024) is canonical. Four rotation positions: R1 and R2 fuse into weights offline; R3 (within attention) and R4 (down-projection input) apply online via cheap Hadamard transforms. End-to-end W4A4 KV4 on Llama-2-70B with at most 0.47 PPL increase.
- **SpinQuant** ([Liu et al., 2024](https://arxiv.org/abs/2405.16406), ICLR 2025) replaces random Hadamard with *learned* Cayley-parametrized rotations. Reports a 45% relative-PPL recovery over QuaRot on Llama-3-8B at W4A4.
- **DuQuant** ([Lin et al., 2024](https://arxiv.org/abs/2406.01721), NeurIPS 2024 Oral) applies block-wise rotations informed by outlier-channel priors plus a zigzag permutation to balance outlier mass across blocks.
- **FlatQuant** ([Liu et al., 2024](https://arxiv.org/abs/2410.09426), ICML 2025) uses per-layer Kronecker-factored learnable affine transforms, fused into a single runtime kernel. W4A4 on Llama-3-70B with under 1% accuracy drop.

These methods are *preprocessing transforms* applied before any quantization recipe, not standalone recipes. A production W4A4 stack is "AWQ + QuaRot" or "GPTQ + SpinQuant" or "round-to-nearest + FlatQuant"; the rotation reshapes weights and activations into a quantization-friendly basis, and a separate recipe assigns scales. Whether *outlier-safe pre-training* (arXiv:2506.19697, Muon optimizer + single-scale RMSNorm to suppress outlier emergence at training time) will eventually obviate rotation preprocessing is open; no frontier-scale model has yet shipped on this recipe.

### 3.8 PTQ versus QAT

Pure PTQ recipes — GPTQ, AWQ, HQQ, AutoRound, SmoothQuant, plus rotations — never update the base model's weights via gradient; they use calibration data only to choose quantizer parameters. QAT in the strict sense fine-tunes with fake-quantization in the forward pass. QAT-light methods sit between: OmniQuant updates only smoothing and clipping; LLM-QAT ([Liu et al., 2023](https://arxiv.org/abs/2305.17888)) and EfficientQAT ([arXiv:2407.11062](https://arxiv.org/abs/2407.11062), ACL 2025) do full-weight fine-tuning with quantization in the loop; NVIDIA's NVFP4-QAD (Quantization-Aware Distillation, 2026-03) recovers NVFP4 inference accuracy via short distillation runs. Rule of thumb: PTQ suffices for W4A16 and FP8 W8A8; QAT or QAT-light is the recovery path when PTQ leaves accuracy on the floor at W4A4 or below.

## 4. The 3-axis cross product

The three axes generate a design space in which any production scheme is one cell:

| Cell | Tensor axis | Format | Recipe | Example deployment |
|---|---|---|---|---|
| Universal weight-only | W-only | INT4 (W4A16) | GPTQ or AWQ, group 128 | Llama-3-70B on H100 with Marlin/Machete |
| Hopper W+A | W+A | FP8 (E4M3 W8A8) | dynamic per-tensor/per-token scales | DeepSeek-V3 native FP8 on H100/H200 |
| Blackwell W+A | W+A | NVFP4 (W4A4) | ModelOpt PTQ + rotation preprocessing | Llama-3.1-405B-FP4 on B200 |
| W+A with QAT-light | W+A | INT4 / FP4 | OmniQuant or NVFP4-QAD | Sub-4-bit recovery checkpoints |
| KV-only on Hopper | KV | FP8 KV | dynamic per-block | TRT-LLM `KvCacheConfig(dtype='fp8')` |
| KV-only on Blackwell | KV | NVFP4 KV | ModelOpt | TRT-LLM `KvCacheConfig(dtype='nvfp4')` |
| Edge / consumer | W-only | NF4 or GGUF k-/i-quants | bitsandbytes / llama.cpp imatrix | Llama-3-8B on Apple M-series, RTX 4090 |
| Mixed W4A8 | W+A | INT4 weights, FP8 activations | QServe / W4A8KV4 co-design | A100 / L40S 70B-class |

Production typically picks the cell whose toolchain is best-supported in a given engine, not the cell that is theoretically optimal. The toolchain question is the subject of §6.

## 5. KV-cache quantization

The KV cache deserves its own treatment because it differs structurally from weight quantization: weights are static and offline-quantizable, while KV is generated at runtime, lives for the duration of a request, and has its own outlier statistics. From the per-token bandwidth math in [§00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md) and [§10/02-paged-kv-memory](02-paged-kv-memory.md), KV bytes are $2 \cdot L \cdot H_{\mathrm{kv}} \cdot d_h \cdot B \cdot T \cdot b/8$; at $T = 128\text{k}$ and modest batch this is the dominant memory consumer, and cutting $b$ from 16 to 8 to 4 to 2 bits gives 2×, 4×, 8× reductions.

### 5.1 KIVI

KIVI ([Liu et al., 2024](https://arxiv.org/abs/2402.02750), ICML 2024) was the first widely-deployed sub-FP8 KV quantizer. Its key empirical finding: K and V have different outlier structure. Certain K channels carry persistent outliers across all tokens; V channels do not show this pattern. KIVI therefore quantizes K *per-channel* (capturing outlier channels accurately) and V *per-token* (capturing magnitude variation along the sequence axis). The asymmetry lets KIVI hit 2-bit KV with usable perplexity, 2.6× peak memory reduction, and reported throughput gains of 2.35×–3.47× at 4× larger usable batch.

### 5.2 KVQuant

KVQuant ([Hooper et al., 2024](https://github.com/SqueezeAILab/KVQuant), NeurIPS 2024) extends the asymmetric treatment with non-uniform per-vector quantization for K (pre-RoPE, since RoPE rotations smear the outlier structure across channels), sensitivity-driven scale selection, and dense-and-sparse outlier isolation (a small fraction of high-magnitude entries kept in FP16). The headline demo is 10-million-token-context inference at 4-bit and 3-bit KV; the technique remains a research baseline, but its decomposition of K outliers continues to inform the field.

### 5.3 TurboQuant

TurboQuant ([Zandieh et al., 2025-04, arXiv:2504.19874](https://arxiv.org/abs/2504.19874); Google Research / NYU / DeepMind, ICLR 2026) is the current near-frontier for KV quantization. The construction has three parts: first, each input vector is multiplied by a random orthogonal rotation $Q$, so rotated coordinates concentrate near a Beta-like distribution and outliers smear away. Second, a per-coordinate scalar quantizer captures the bulk of the energy. Third, a 1-bit Johnson-Lindenstrauss residual quantizes the remainder, providing an unbiased inner-product estimate via low-dimensional random projection. The combination is information-theoretically near-optimal: MSE distortion at a given bit-rate matches the rate-distortion lower bound within roughly 2.7×.

Applied numbers (verified at ICLR 2026): 3.5 bits per channel hits "absolute quality neutrality" on standard KV-quant benchmarks; 2.5 bits per channel produces marginal degradation. The method is tuning-free. TurboQuant landed in vLLM as the `turboquant/` quantization module and an associated attention-kernel backend; independent community Triton kernels (e.g., 3-bit K with 2-bit V) exist and should not be confused with the canonical Google paper.

### 5.4 Production KV layering

The production picture as of mid-2026 stacks KV quantization as a separate axis from weight quantization. FP8 KV (E4M3 or E5M2) is the default on Hopper for any engine that supports it: the FMHA, XQA, and FlashMLA kernels in TRT-LLM all ship FP8 inner loops, and vLLM and SGLang both expose `kv_cache_dtype=fp8`. NVFP4 KV is the default on Blackwell, supported in TRT-LLM and on vLLM's roadmap with kernel landings already merged. Sub-FP8 KV (KIVI, KVQuant, TurboQuant) is on the production roadmap but not yet a default. KV quantization also interacts with the paged block manager: for per-block scales (FP8 dynamic, MX-style) the scale lives alongside the data and the block size is unchanged; for per-channel scales (KIVI-style K) the scale is a per-layer side tensor pinned as an ordinary layer parameter.

SageAttention (1/2/3) — INT8, FP8, and NVFP4 quantization of the $QK^{\top}$ score matrix *inside* the attention computation — is a distinct technique from KV-cache quantization and is covered in [§10/01-attention-kernels](01-attention-kernels.md).

## 6. The 1-bit frontier

The natural lower bound on uniform scalar quantization is one bit per weight. The BitNet line (Microsoft Research, 2023–2025) has argued frontier-scale LLMs can in principle be trained natively at this bound. **BitNet** (Wang et al., 2023) replaced `nn.Linear` with `BitLinear` (binary $\{-1, +1\}$ weights, 8-bit activations). **BitNet b1.58** (Ma et al., 2024, [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)) quantized to ternary $\{-1, 0, +1\}$ via absmean rounding, $\log_2 3 \approx 1.58$ bits/weight, with parity to FP baselines at 3B+ scale on perplexity and downstream tasks. **BitNet b1.58 2B-4T** ([arXiv:2504.12285](https://arxiv.org/abs/2504.12285)) is the first open-source native ternary LLM at 2B parameters and 4T training tokens — the current proof point that ternary pre-training scales beyond toy. **BitNet a4.8** (2024-11) extends to 4-bit activations into attention/FFN with 8-bit elsewhere and 3-bit KV; activates only ~55% of parameters per token.

The BitNet program is treated here as research, not production. GPU tensor cores have no native ternary or binary path, so BitNet inference runs through the `bitnet.cpp` reference framework on CPU or through unoptimized custom kernels on GPU. There is no PTQ path — BitNet requires training-time quantization awareness, closing the door for deployments that start from an existing FP16 checkpoint. As of mid-2026, no frontier-scale (>100B) model has shipped on the BitNet recipe. The open watch-item is whether ASIC or FPGA accelerators that natively dot-product ternary tensors offer enough efficiency advantage at the 100B+ scale to incentivize the training-side investment.

## 7. Per-engine production matrix

The recipes and formats above matter only if the chosen engine ships the kernels. The following collapses the state of vLLM, SGLang, TensorRT-LLM, and llama.cpp as of mid-2026.

| Format / scheme | vLLM | SGLang | TensorRT-LLM | llama.cpp |
|---|---|---|---|---|
| GPTQ W4A16 | `gptq`, `gptq_marlin`, Machete | ✓ | `W4A16_GPTQ` | k-/i-quants approximate |
| AWQ W4A16 | `awq`, `awq_marlin` | ✓ | `W4A16_AWQ` | imatrix variants |
| INT8 W8A8 (SmoothQuant) | `compressed-tensors` W8A8 | ✓ | `W8A8_SQ_*` plugins | INT8 k-quants |
| FP8 W8A8 (E4M3) | `fp8`, `fbgemm_fp8`, `fp8_per_tensor` | ✓ (Hopper + Blackwell) | `FP8`, `FP8_PER_CHANNEL_PER_TOKEN`, `FP8_BLOCK_SCALES` | — |
| NVFP4 W4A4 | `modelopt_fp4`, `compressed-tensors` `quantization_w4a4_fp4` | ModelOpt path (2025-12+) | `NVFP4`, `NVFP4_AWQ`, `NVFP4_ARC` | NVFP4 ggml type (recent) |
| MXFP4 | `mxfp4`, `gpt_oss_mxfp4` (GPT-OSS path) | GPT-OSS MXFP4 (2025-08) | `W4A16_MXFP4`, `W4A8_MXFP4_FP8`, `W4A8_MXFP4_MXFP8` | MXFP4 ggml type |
| MXFP8 | `modelopt_mxfp8`, `mxfp8` | ✓ | (via ModelOpt) | — |
| FP8 KV cache | `kv_cache_dtype=fp8` (E4M3/E5M2) | ✓ | `KvCacheConfig(dtype='fp8')` | ✓ |
| NVFP4 KV cache | merged (`#40177`); roadmap default Q2 2026 | roadmap | `KvCacheConfig(dtype='nvfp4')` | — |
| INT8 KV cache | dynamic per-token | ✓ | `KvCacheConfig(dtype='int8')` | — |
| TurboQuant KV | `turboquant/` module + backend | — | — | — |
| W4A8 (QServe-style) | research-grade | research-grade | `W4A8_QSERVE_*` plugin | — |
| BitNet b1.58 | not supported | not supported | not supported | partial (`bitnet.cpp`-imported kernels) |
| Calibration toolchain | LLM Compressor (Red Hat / vLLM) | ModelOpt (NVIDIA) + LLM Compressor | ModelOpt (NVIDIA) | imatrix (in-tree) |

Three patterns in the table are worth naming. First, vLLM, SGLang, and TRT-LLM converge on the same set of formats but reach them through different toolchains: vLLM standardizes on Red Hat's LLM Compressor (the `compressed-tensors` checkpoint format), SGLang and TRT-LLM standardize on NVIDIA's ModelOpt, and the two formats are increasingly interconvertible. Second, llama.cpp's GGUF k-quants and i-quants occupy a parallel universe; importance-matrix i-quants (IQ4_XS, IQ3_S) carry the spirit of AWQ/GPTQ into the GGUF world. Third, MXFP4 production paths emerged unusually quickly because the GPT-OSS release forced the issue; before August 2025, NVFP4 was further along on every engine.

A second cut, by accelerator:

- **A100 / L40S / A10G.** FP8 unavailable; production is W4A16 GPTQ/AWQ on the weight side, BF16 activations, INT8 KV at most. QServe-style W4A8 (INT4 weights, FP8 activations) is the throughput-pushing alternative.
- **Hopper H100/H200.** FP8 W8A8 is the production default for compute-bound workloads; W4A16 remains the bandwidth-bound default. FP8 KV is universal. NVFP4 runs only in software emulation and is rarely chosen.
- **Blackwell B200/B300/GB200/GB300.** NVFP4 W4A4 is the recommended W+A precision; FP8 remains competitive. MXFP4 is supported and chosen when a model ships natively in MXFP4 (GPT-OSS). NVFP4 KV is supported.
- **AMD MI300X/MI325X.** FP8 W8A8 reaches NVIDIA-comparable throughput; INT4 weight-only is supported through ROCm Quark and AITER kernels. MXFP8 is native; MXFP4 runs through Quark. NVFP4 is not supported.

The implication is that *engine plus accelerator* defines the feasible cells in the 3-axis cross-product. A production decision is rarely exotic: pick the highest-throughput supported cell that hits the accuracy target, audit the calibration toolchain, run the eval suite. The remaining art is in the calibration choice within the supported set — where AWQ, GPTQ, AutoRound, OmniQuant, and rotation preprocessing reappear as practical knobs.

## Current production state

Production quantization in mid-2026 has settled into a small set of defaults that vary by accelerator generation. On Hopper H100/H200 — which still carries the majority of frontier-lab production tokens — the default for compute-bound prefill and large-batch workloads is FP8 W8A8 with FP8 (E4M3) KV cache, calibrated either by the Transformer-Engine recipe (dynamic per-tensor or per-token scales) or by SmoothQuant-lineage offline calibration through LLM Compressor or ModelOpt. The default for bandwidth-bound decode at small batch is W4A16 with GPTQ or AWQ scales and Marlin or Machete kernels; rotation preprocessing (QuaRot, SpinQuant) is layered when the deployment can afford the offline cost. Native FP8-trained checkpoints — DeepSeek-V3 with its 1×128 activation tile and 128×128 weight block scaling is canonical — are the cleanest production path because they avoid the PTQ step entirely.

On Blackwell B200/B300 the recommended W+A precision is NVFP4 with NVFP4 KV cache, deployed through ModelOpt with a small QAT-light recovery pass (NVIDIA's NVFP4-QAD distillation recipe) where accuracy budgets are tight. Public NVFP4 checkpoints — DeepSeek-R1-FP4, Llama-3.1-405B-FP4, GPT-OSS-MXFP4 — are the reference deployments. SGLang and TRT-LLM both ship NVFP4 in production; vLLM ships NVFP4 weights and has NVFP4 KV cache merged with the default-on flip on the Q2 2026 roadmap. MXFP4 production exists almost entirely because of GPT-OSS, with kernel paths in vLLM, SGLang, TRT-LLM, and llama.cpp; the accuracy gap to NVFP4, roughly 10% in early reports and narrowed to under 1% by techniques like Bridging-the-Gap MXFP4 (offline activation smoothing plus per-tensor bias scaling), is now small enough that the NVFP4-vs-MXFP4 choice is driven by hardware support rather than accuracy.

Sub-FP8 KV quantization remains a production frontier rather than a default: KIVI and KVQuant exist primarily in research and selective production; TurboQuant — verified at ICLR 2026 with information-theoretic optimality and 2.5–3.5 bits-per-channel quality — has landed in vLLM but is not yet on by default. The 1-bit frontier is treated as research; no frontier-scale model has shipped on it, no major engine has tensor-core support for ternary, and the path forward likely requires training-side commitment plus hardware that natively dot-products ternary tensors. The active production trajectory is downward in bit-width along the FP4 axis (NVFP4 today, possibly narrower with adaptive per-block scale-bit allocation — Four Over Six and similar — tomorrow) and the KV axis (NVFP4 KV today, sub-3-bit KV after TurboQuant matures), rather than along the BitNet axis. Rotation methods remain a universal preprocessing layer in the W4A4 stack; whether outlier-safe pre-training will eventually retire them is the open architectural question.
