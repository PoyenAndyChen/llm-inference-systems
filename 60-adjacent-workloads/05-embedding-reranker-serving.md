# Embedding and Reranker Serving

**After reading this chapter, the reader will be able to:**

- Explain why embedding inference is structurally a different serving
  problem from token-generation inference — single forward pass, short
  inputs, no autoregressive decode, embarrassingly batchable — and why this
  makes the optimisation surface (token-based batching, quantisation
  stacks, storage) look almost nothing like the LLM-engine optimisation
  surface developed in earlier chapters.
- Place the four serving regimes that share the embedding label — dense
  bi-encoder embeddings, late-interaction multi-vector models
  (ColBERTv2 / ColPali / ColQwen), cross-encoder rerankers, and
  Matryoshka-plus-quantisation storage stacks — onto a single map and
  reason about which engine, which index, and which quantisation
  recipe each demands.
- Read the 2025–2026 leaderboard transition from MTEB to MMTEB, the
  parity between top open-weight embedding models and the closed Voyage /
  Cohere / OpenAI APIs, and decide for a given deployment whether the
  bottleneck is embedding throughput, storage, or reranker latency.

Embedding serving is the unsung half of the modern retrieval stack.
The RAG pipeline assembled in [§60/04-rag-infrastructure](04-rag-infrastructure.md)
opens with an embedding step and closes with a reranker step; both
are model inferences, neither is autoregressive, and both run at orders
of magnitude higher throughput per GPU than the LLM-generation stage
they feed. The optimisation effort spent on token-generation engines
in [§10](../10-engine-core) does not transfer cleanly: there is no
KV cache to manage, no continuous batch to reorder, no chunked prefill
to schedule. What replaces it is a different set of levers — token-budgeted
batching, encoder quantisation, storage compression of the resulting
vectors, and a leaderboard that has shifted under the field's feet
during 2025.

## 1. Why embedding serving is not LLM serving

A dense embedding model maps an input sequence of $T$ tokens to a single
$d$-dimensional vector via one bidirectional forward pass. There is
no decode, no KV cache, no per-token loop. The arithmetic is closer to
prefill than to anything in [§00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md):
each request hits the matmul-heavy regime, batches concatenate cleanly
along the sequence axis, and the operating point sits firmly on the
compute-bound side of the roofline for any reasonable batch size.

Three structural consequences follow. First, throughput per GPU is one
to two orders of magnitude higher than for token generation: a
BGE-base-class encoder on H100 fp16 reaches into the 100K embeddings/sec
range for short text, and even a 7B-parameter embedding backbone hits
the 10K–30K embeddings/sec range under TensorRT-LLM in-flight batching
(order-of-magnitude figures from vendor and community blogs; the upper
end is unverified). Second, latency budgets are tight in absolute terms
but loose relative to LLM inference — a 5 ms embedding cost is invisible
inside a 2 s generate budget, but it is highly visible inside a
500 ms retrieval-only budget. Third, the optimisation surface compresses:
since there is no decode, the bandwidth-versus-compute distinction that
drives most of [§10/04-quantization](../10-engine-core/04-quantization.md)
collapses; the only meaningful axes are *batch construction* and
*encoder quantisation*, plus the post-encoding axis of *vector storage*.

The reranker case is similar in shape but different in magnitude. A
cross-encoder reranker takes (query, document) pairs and emits a relevance
score via a single forward pass per pair. Reranking the top-100 of a
retrieval set at 4 ms per pair on a 6B-class reranker gives a 400 ms
sequential cost; with cross-pair batching and early exit, production
deployments hit the 100–200 ms p95 range that closed APIs publish. The
serving primitive is again a single forward pass — same compute regime
as embedding, different input shape (paired) and output shape (scalar).

## 2. Token-budgeted batching: TEI, Infinity, TensorRT-LLM

Continuous batching as developed in
[§10/03-batching-scheduling](../10-engine-core/03-batching-scheduling.md)
is unnecessary here — there is no decode loop to inject new requests
into. What matters is how a *static batch* is constructed from a heterogeneous
queue of incoming requests of varying lengths. Two strategies dominate.

**Token-based batching** (Hugging Face Text Embeddings Inference, TEI)
builds each batch up to a fixed total token budget rather than a fixed
request count. A batch of 32 requests at 512 tokens each and a batch of
512 requests at 32 tokens each are equally well-served by the GPU; what
matters is the total token count fed to the encoder, since that drives
both the compute load and the activation memory footprint. TEI's queue
admits requests greedily until the next request would push the batch
past a configured token ceiling, at which point the batch is dispatched.
The scheme smooths GPU utilisation across heterogeneous request lengths
and is the reference implementation for production embedding serving;
TEI runs on NVIDIA and AMD GPUs and is the default backend in many
managed embedding stacks. [TEI-Repo]

**Request-based batching** (Infinity, [Feil, ongoing]) takes the
complementary path: batches are sized by request count, with per-thread
tokenisation overlapping the GPU compute. Infinity reports wins on
short-text workloads where tokenisation cost is non-trivial and where
length variance is small enough that fixed request-count batching does
not waste compute. Infinity also covers a wider range of model
families — text embeddings, cross-encoder rerankers, CLIP and CLAP
multimodal embeddings, and ColPali — which makes it the de-facto choice
for stacks that need uniform serving across modalities.

**TensorRT-LLM** treats embedding models as a special case of its
in-flight batching engine, with native FP8 quantisation kernels available
on Hopper and NVFP4 on Blackwell. Vendor and community blogs report
roughly 3× throughput over TEI and vLLM in some configurations, with the
upper end of these claims unverified; the win is largest when the
encoder is large enough to be compute-bound (Mistral-7B-class backbones)
and the activations dominate enough of the matmul time to benefit from
FP8 tensor cores. For the BGE-base / E5-base size class the throughput
is bounded by tokenisation and host-to-device copies more than by the
model itself, and TEI and Infinity remain competitive.

The throughput numbers worth carrying forward, hedged as
order-of-magnitude:

- BGE-base / E5-base on H100 fp16: 100K+ short-text embeddings/sec.
- 7B-parameter embedding backbone on H100 fp16 with TRT-LLM
  in-flight batching: 10K–30K short-text embeddings/sec.
- Same 7B model on A100 fp16: roughly half.

These figures dwarf the per-GPU rates of any generative serving stack.
The implication is that an embedding-heavy ingest pipeline rarely
needs the same number of GPUs as the generation pipeline it feeds;
provisioning is asymmetric.

## 3. Multi-vector serving: ColBERT, PLAID, ColPali, ColQwen

The bi-encoder model that emits one vector per passage is the
mainstream case but not the only one. Late-interaction retrieval, founded
by ColBERT [ColBERT-2020], emits *one vector per token* and computes
relevance as a sum of per-query-token max-similarities — the MaxSim
operator $\sum_{i} \max_{j} q_i \cdot d_j$ over query tokens $i$ and
document tokens $j$. The expressive power per query is higher because
information from individual tokens is preserved, but storage and
indexing footprints grow proportionally.

**ColBERTv2** [ColBERTv2-2022] introduced centroid-based residual
compression that drove storage down 6–10× from the original. A typical
ColBERTv2 passage carries on the order of 150 token vectors at 128 dims
fp16, totalling roughly 38 KB per passage. The serving engine that
pairs with this representation is **PLAID** [PLAID-2022], which uses
a centroid-pruning pre-filter, residual decompression on demand, and
fused MaxSim kernels to reach up to 7× GPU and 45× CPU latency
reductions over a vanilla ColBERTv2 implementation. PLAID is the
canonical late-interaction serving engine and continues to define the
performance envelope for this class.

**ColPali** [ColPali-2024] (ICLR 2025) extends late interaction to
visual document retrieval: a ViT-style image patch grid over a rendered
PDF page replaces the token grid, the ColBERT MaxSim operator runs over
patches, and the resulting "patch embedding per page" footprint is
roughly 1024 patches × 128 dims fp16 = 256 KB per page. ColPali eliminates
the OCR-and-chunk preprocessing step that traditional document-RAG
stacks need; the visual encoder reads the page directly. **ColQwen**
(2025) and ColQwen2.5 swap the visual backbone for Qwen2-VL / Qwen2.5-VL
and currently lead the ViDoRe benchmark for visual document retrieval.

The serving contract that multi-vector models impose on the index is
distinct from the bi-encoder case. A vector database must support
*tensor-valued* fields (one vector per token or patch, with arbitrary
counts per document) and a MaxSim aggregation operator at query time.
Three production engines have first-class multi-vector support:
**Vespa** with its tensor expressions, **Qdrant** with its multi-vector
field type, and **Milvus** with its multi-vector index. pgvector and
basic FAISS do not — they treat each vector as an independent point
and have no native MaxSim path.

The storage budget makes multi-vector deployments operationally heavier
than bi-encoder deployments at the same corpus size. At 10M passages,
a ColBERTv2 index occupies roughly 380 GB of fp16 multi-vector storage
versus 40 GB for a 1024-dim fp32 bi-encoder index. The Matryoshka and
quantisation stacks below are the lever that brings multi-vector
storage back into the bi-encoder range.

## 4. Quantisation taxonomy: int8, binary, 4-bit

Embedding quantisation is structurally simpler than the weight-quantisation
problem developed in [§10/04-quantization](../10-engine-core/04-quantization.md)
because the object being quantised is a value tensor, not a model parameter.
There is no calibration in the GPTQ/AWQ sense — only a choice of representation
and, for binary, a similarity-preserving binarisation function.

**Scalar int8.** Each of the $d$ vector components is represented as a signed
8-bit integer with a per-vector or per-tensor scale. Storage shrinks 4× from
fp32 (or 2× from fp16); retrieval quality preserves roughly 99% of fp32
performance on standard benchmarks; scan throughput improves by approximately
3.7× because SIMD lanes process 4× more values per cycle. Scalar int8 is
the safe default for production retrieval and the universal first quantisation
step. [HF-Embedding-Quant-2024]

**Binary.** Each component is reduced to a single bit, typically by sign
$b_i = \mathbb{1}[v_i > 0]$. Storage shrinks 32× from fp32; Hamming distance
replaces dot product as the similarity operator, and SIMD popcount instructions
make the scan up to 24× faster than fp32. The cost is a model-dependent
quality cliff. ModernBERT-based embeddings retain roughly 98% of full-precision
quality under binary [ModernBERT-2024]; older E5-family models retain 87–92%.
The model-dependence is large enough that binary quantisation should not be
adopted without per-model evaluation; the safe pattern is binary scan plus
a rescore stage against int8 or fp16 vectors for the top-K candidates.

**4-bit.** The 4-bit-per-component frontier ([4bit-Quant-RAG-2025])
sits between scalar int8 and binary, with roughly 8× storage reduction and
quality typically within 1–2% of fp32 for well-trained encoders. Hardware
support is software-only — there is no SIMD popcount-equivalent — so the
scan-throughput win is smaller than for binary. As of mid-2026 4-bit
embedding storage is emerging but not yet a default.

**Matryoshka Representation Learning** [Matryoshka-2022] (NeurIPS 2022) is
a complementary axis. The training loss is augmented so that for each prefix
length $k \in \{d/8, d/4, d/2, d\}$ the truncated vector is itself a valid
embedding. At inference the deployment chooses the smallest $k$ that meets
its quality bar — typically 1024 → 256 with roughly 2% quality loss, or
1024 → 128 with roughly 5% loss. Matryoshka is independent of the bit-width
axis and composes cleanly with scalar or binary quantisation.

The combined quantisation-plus-Matryoshka stack is the production
recipe. Working through the storage math for a representative 10M-document
corpus with 1024-dim base embeddings:

- **Baseline:** $10^7 \times 1024 \times 4 = 40$ GB at fp32.
- **fp16 baseline:** 20 GB (2×).
- **Matryoshka 1024 → 256 + int8:** $10^7 \times 256 \times 1 = 2.5$ GB
  (16×) at roughly 3% quality loss.
- **Matryoshka 1024 → 256 + binary + rescore:** $10^7 \times 256 / 8 = 320$ MB
  (125×) at roughly 5% loss after rescore. Scan is 32× faster; rescore
  is 5× slower per query than a pure int8 scan but applies only to the
  top-K candidates.

The 100–500× storage win at 3–5% quality cost flagged in HF and Vespa
production blogs comes from this Matryoshka-plus-binary-plus-rescore
stack. [HF-Embedding-Quant-2024, Vespa-Matryoshka-Binary]

For multi-vector models the same stack scales the per-document footprint
proportionally. A ColPali deployment dropping from 256 KB/page fp16 to
roughly 10–20 KB/page through Matryoshka truncation plus binary quantisation
moves the indexing budget from "expensive" to "comparable to a standard
bi-encoder corpus" and is the recipe that brought ColPali / ColQwen into
production-feasible territory.

## 5. Leaderboards: MTEB, MMTEB, and the open–closed parity

For most of the embedding era the **MTEB** benchmark [MTEB-2022] (EACL 2023)
defined the cross-model comparison standard: 56 datasets, 8 task families,
single-number averages that drove model selection. MTEB was English-centric
and assumed a static benchmark target.

**MMTEB** [MMTEB-2025] (ICLR 2025, arXiv:2502.13595) is the 2025 successor.
The expansion is substantial: 250+ languages, 500+ tasks, instruction
following, long-document retrieval, and a multilingual default headline
average. The community-maintenance paper ([MTEB-Maintenance-2025]) makes
explicit that MTEB v2 scores are not directly back-comparable to v1
because of overlapping-but-distinct task pools and revised aggregation;
historical MTEB-v1 numbers in older model cards should be read as a
different metric.

The 2024–2026 cycle saw open-weight embedding models reach and in some
cases exceed the closed APIs on these leaderboards. The headline open
models as of mid-2026:

- **Qwen3-Embedding-8B** [Qwen3-Embedding-2025] (Alibaba, June 2025,
  Apache 2.0): 70.58 average on MMTEB multilingual at release;
  flexible 32–4096 output dimensions; 32K context. Sets the open
  multilingual top.
- **NV-Embed-v2** (NVIDIA, Mistral-7B backbone): 72.31 average on
  MTEB English; 4096-dim output. Strong English-centric option.
- **BGE-en-ICL** (BAAI): in-context-learning embedding family;
  competitive on English and instruction-following tasks.
- **Llama-Embed-Nemotron-8B** (NVIDIA): Llama-based encoder line
  positioned alongside NV-Embed-v2.

The closed APIs remain competitive on cost and on flexible-dimension
support. **Voyage 3.5** and **Voyage 3.5-Lite** (Voyage AI, acquired
by MongoDB) are the cost/quality leaders among managed APIs at $0.06
and $0.02 per million tokens respectively, with native flexible
dimensions and quantisation as a serving knob. **Cohere Embed** and
**OpenAI text-embedding-3** sit in a similar quality band, with OpenAI's
text-embedding-3-large historically being one of the early flexible-dimension
production APIs.

The deployment question shifted accordingly. Pre-2024 the choice was
between "use a closed API for quality" and "self-host a smaller model
for cost." As of 2026 the choice is between "use a closed API for ops
simplicity" and "self-host a top-of-leaderboard open model for cost
and control"; the quality axis no longer separates the two cleanly.

Reranker leaderboards have not converged on a single MMTEB-equivalent
benchmark. Production reranker selection runs against task-specific
evaluation harnesses, with public latency–quality benchmarks
([Rerankers-Bench-2025]) covering Cohere Rerank (p95 ~210 ms,
closed), Jina Rerank v2 / v3 (p95 ~110 ms, closed), and BGE-Reranker
(BGE-base v2 ~92 ms p95, open). Cross-encoder rerankers add 100–600 ms
to a retrieval pipeline and recover 10–30% precision over the
first-stage retriever.

## 6. Production landscape

The production embedding-serving stack as of mid-2026 has converged on
a small set of components, with selection driven by deployment
constraints rather than capability gaps:

- **General-purpose dense embedding:** TEI for the NVIDIA + AMD
  default; Infinity when wider modality coverage (CLIP, CLAP, ColPali)
  is needed in one stack; TensorRT-LLM when the embedding backbone is
  large enough (≥7B) for FP8 / NVFP4 quantisation kernels to pay off.
- **Sentence-Transformers** as the offline / batch-processing backend
  inherited from the Sentence-BERT lineage [Sentence-BERT-2019]; fits
  ingest pipelines that do not need online serving SLAs.
- **Multi-vector text retrieval:** PLAID for ColBERT-style late
  interaction over passages.
- **Multi-vector visual retrieval:** Vespa or Qdrant for ColPali / ColQwen
  serving; Milvus as a third option; engines without multi-vector
  index types are excluded by construction.
- **Cross-encoder reranking:** Cohere Rerank or Jina Reranker v3 for
  closed-API simplicity; BGE-Reranker family for self-host at
  comparable latency.
- **Quantisation default:** int8 plus Matryoshka truncation
  (1024 → 256 typical) is the safe production stack; binary quantisation
  is reserved for ModernBERT-based models or deployed behind a rescore
  stage.

The selection matrix collapses to a small number of cells:

| Workload | Primary engine | Index | Quantisation default |
|---|---|---|---|
| Dense bi-encoder, ≤1B params | TEI or Infinity | HNSW / IVF-PQ | int8 + Matryoshka 1024→256 |
| Dense bi-encoder, 7B+ params | TRT-LLM | HNSW / DiskANN | FP8 encoder + int8 storage |
| Late-interaction text (ColBERTv2) | PLAID | PLAID centroid index | residual compression (built-in) |
| Visual late-interaction (ColPali / ColQwen) | Infinity or custom serving | Vespa / Qdrant multi-vector | Matryoshka + binary + rescore |
| Cross-encoder reranker (open) | TEI or vLLM | n/a | fp16 or FP8 |
| Cross-encoder reranker (closed) | Cohere / Jina API | n/a | provider-managed |

The ANN-side decisions — HNSW versus IVF-PQ versus DiskANN versus
SPANN at scale — are developed in the RAG chapter
[§60/04-rag-infrastructure](04-rag-infrastructure.md); embedding
serving and ANN indexing remain orthogonal axes in the stack.

## Current production state

Embedding serving in mid-2026 is a mature subfield with a small set of
defaults that vary mostly by encoder size and modality. For dense
bi-encoder workloads the TEI / Infinity / TRT-LLM trio covers
substantially the entire production surface, with TEI serving as the
de-facto default for moderate-sized models and TRT-LLM taking over at
the 7B+ scale where its FP8 and NVFP4 kernels begin to pay off. The
embedding-throughput gap to LLM generation remains roughly two orders of
magnitude, so embedding-side GPU provisioning is rarely the bottleneck
of a retrieval-and-generate pipeline; the harder questions are storage
and reranker latency.

The 2025 transition from MTEB to MMTEB and the parallel arrival of
Qwen3-Embedding-8B, NV-Embed-v2, and the BGE-en-ICL family completed the
open-weight catch-up to closed APIs on the quality axis; Voyage 3.5,
Cohere Embed, and OpenAI text-embedding-3 now compete primarily on
operational simplicity, native flexible-dimension and quantisation
controls, and cost per million tokens rather than on raw quality. The
deployment decision has correspondingly shifted from "API or self-host"
to "API for simplicity, self-host for cost and control," with both
choices reaching comparable retrieval quality.

The frontier in 2026 is in the storage and serving layers rather than
the encoder itself. Matryoshka truncation plus int8 quantisation is the
universal default; binary quantisation behind a rescore stage is the
production aggressive path, with the ModernBERT-versus-E5 quality
asymmetry under binarisation pushing newer deployments toward
ModernBERT-based encoders. Multi-vector serving — ColBERTv2 plus PLAID
for text, ColPali / ColQwen plus Vespa or Qdrant for visual documents —
remains the operationally heavier path but is the only viable serving
shape for visual document RAG and is increasingly common in production
where the document corpus is PDF-native. Sub-int8 frontiers (4-bit
scalar, learned quantisation) are emerging but not yet defaults; the
load-bearing recipe for the next year remains Matryoshka plus int8 plus
selective binary, with cross-encoder reranking on top.
