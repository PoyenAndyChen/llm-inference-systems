# RAG Infrastructure

**After reading this chapter, the reader will be able to:**

- Walk the canonical RAG pipeline — ingest, embed, index, retrieve,
  rerank, generate — and place each stage on a production latency
  budget, distinguishing index-bound from GPU-bound from
  LLM-forward-pass-bound work.
- Compare the dominant ANN index families (HNSW, IVF-PQ, ScaNN,
  DiskANN, SPANN, DistributedANN) along the recall-latency-memory
  Pareto, and reason about which is correct under different scale and
  tail-latency constraints.
- Connect retrieval-side techniques (hybrid, late interaction,
  cross-encoder reranking) to serving-side techniques (cross-query KV
  reuse, agentic multi-step retrieval) and locate each in the
  2025–2026 production stack.

Retrieval-augmented generation grew from a research pattern (Lewis
et al., 2020) into the dominant production architecture for grounding
LLM output in non-parametric knowledge. The infrastructure is a
pipeline of specialized services — embedding model, vector index,
optional sparse and learned-sparse indices, cross-encoder reranker,
LLM serving engine, increasingly an agent loop on top — each with its
own scaling regime and tail-latency story.

## 1. The pipeline and where time goes

The RAG request path has six canonical stages:

```
ingest → embed → index → retrieve → rerank → generate
```

The first three are offline: documents are chunked, embedded, and
written to one or more indices. The last three run on the request
critical path. A representative single-tenant budget for a 1k-document
RAG step:

| Stage          | Typical p50  | Typical p95  | Bound by                  |
|----------------|--------------|--------------|----------------------------|
| Embed query    | 5–20 ms      | 30–80 ms     | Embedding model GPU        |
| ANN retrieve   | 2–20 ms      | 5–50 ms      | Index structure, recall    |
| Sparse / hybrid| 5–15 ms      | 20–60 ms     | Inverted index, fusion     |
| Rerank top-k   | 30–200 ms    | 100–500 ms   | Cross-encoder GPU          |
| Generate       | 200 ms – 30 s| —            | LLM TTFT + decode          |

Generation dominates wall-clock for any non-trivial answer, so
RAG-specific work targets the *front* of the pipeline: making
retrieval and reranking parallel-friendly and cheap enough that all
five stages fit inside a one-second TTFT budget. Agentic RAG breaks
this implicit budget by issuing 5–20 retrieval rounds per user turn
(§8); the per-round budget shrinks accordingly.

Two ingest-time decisions set the ceiling. **Chunking strategy**
(fixed windows, semantic splits, hierarchical parent-child)
determines what the index can return and how much context the LLM
must consume. **Embedding model choice** sets recall ceiling and
storage cost: a 1024-dim fp16 embedding is 2 KB/vector, so a 100M-
vector corpus needs ~200 GB before index overhead. The embedding-
and-reranker serving stack is covered in
[see §60/05-embedding-reranker-serving](./05-embedding-reranker-serving.md).

## 2. ANN index families

Approximate nearest-neighbour search trades exact recall for
sub-linear query cost. Four families dominate production: graph
(HNSW), inverted-file with product quantization (IVF-PQ), learned
quantization (ScaNN), and disk-resident graph (DiskANN / SPANN). The
choice is driven by recall, latency, and memory footprint, which
trade off along a fairly clean Pareto frontier.

### 2.1 HNSW: the in-memory default

Hierarchical Navigable Small World graphs [HNSW-2018] structure the
corpus as a multi-layer proximity graph: upper layers are sparse with
long-range hops, lower layers are dense with local refinement. Search
greedily descends from the top entry point and collects the
$\text{efSearch}$ best candidates seen at the bottom layer. Two
knobs: $M$ (out-degree, density and build cost) and
$\text{efSearch}$ (search-time queue size). Pushing recall from 95%
to 97% via `efSearch` typically halves QPS — the curve is steep near
the top, where most deployments sit.

HNSW (via `hnswlib` and Faiss's `IndexHNSW*`) achieves ~95% recall@10
in 1–2 ms per query on SIFT1M on a single CPU core, and is the index
inside almost every managed vector database (Pinecone, Weaviate,
Qdrant, Milvus default). Its weak point is memory: the graph stores
full-precision vectors plus $O(M)$ neighbour pointers per node, ~10
KB per vector at $d{=}768, M{=}16$, fp32. At 100M vectors that is
~1 TB of RAM — affordable for a hyperscaler, limiting elsewhere.

### 2.2 IVF-PQ: cheap memory at the cost of latency

Inverted-file with product quantization [FAISS-2017] splits the
corpus into $K$ k-means Voronoi cells and represents each vector as
$m$ sub-quantizer codes (each subvector quantized to one of $2^k$
centroids). Search visits the $\text{nprobe}$ closest cells; recall
scales with $\text{nprobe}$, latency with
$\text{nprobe} \cdot$ avg-cell-size.

IVF-PQ matches HNSW on recall at ~10–50 ms per query (roughly 10×
slower) at 10–20× lower memory: a 768-dim vector compressed to PQ-64
costs 64 B instead of 3 KB. At 1B vectors that ratio is the
difference between feasible and not. GPU-Faiss reaches >100k QPS on a
single GPU at billion-scale and has been the batch-mode similarity-
search workhorse for nearly a decade.

### 2.3 ScaNN: anisotropic quantization

ScaNN [ScaNN-2020] refines PQ with one observation: when the
downstream task is *inner product* (cosine after normalization),
quantization error parallel to a vector's direction matters more than
orthogonal error. ScaNN's *anisotropic* loss weights parallel error
more heavily, yielding better recall at the same code length. ScaNN
underlies Google's Vertex AI Vector Search; its tree-AH variant wins
benchmarks at million-vector / dense-vector scale.

### 2.4 DiskANN and SPANN: disk-resident indexing

DiskANN [DiskANN-2019] places graph and full-precision vectors on SSD
with only a compressed in-memory navigation structure. Its Vamana
construction builds a single-layer graph with a long-edge bias
yielding short search paths (tens of hops). Reported on SIFT1B: >5k
QPS at <3 ms mean latency, 95%+ recall, single machine with 64 GB RAM
and NVMe SSD. DiskANN is the backbone of Bing's serving and the index
inside `pgvectorscale` [pgvectorscale-2024] (Timescale's
DiskANN-on-Postgres), making "DiskANN at the database tier" a
deployable pattern via `pgvector` 0.8 + `pgvectorscale`.

SPANN [SPANN-2021] generalizes the memory/disk split: cluster
centroids and a compact navigation index live in memory; per-cluster
posting lists with full vectors live on SSD. SPANN scales a single
index to 100B+ vectors on commodity hardware and has served Bing's
web-scale retrieval since 2021.

### 2.5 The Pareto picture

A summary with numbers to be read as orders of magnitude (workload-
and hardware-dependent):

| Index         | Vectors served      | Latency    | Memory per vec | Where deployed                |
|---------------|---------------------|------------|----------------|--------------------------------|
| HNSW          | $10^6$–$10^8$       | 1–5 ms     | 5–20 KB        | Most managed vector DBs        |
| IVF-PQ (CPU)  | $10^7$–$10^9$       | 10–50 ms   | 50–200 B       | Faiss-on-CPU baseline          |
| IVF-PQ (GPU)  | $10^8$–$10^{10}$    | <5 ms      | 50–200 B       | Faiss-GPU batch retrieval      |
| ScaNN         | $10^7$–$10^9$       | 2–20 ms    | 100–500 B      | Vertex AI; Google internal     |
| DiskANN       | $10^8$–$10^9$       | 2–10 ms    | ~50 B in RAM   | Bing; pgvectorscale            |
| SPANN         | $10^{10}$–$10^{11}$ | 10–50 ms   | small in RAM   | Bing 100B-vector index         |

Production selection rule: under 10M vectors on one machine, HNSW
with `efSearch` tuned to the latency budget; 10M–1B, IVF-PQ on GPU or
DiskANN on SSD; above 1B, SPANN or sharded DiskANN; above 10B, §3.

## 3. DistributedANN at hundred-billion scale

The single-machine boundary at 10–100B vectors has been a structural
limit for a decade — beyond it, the standard approach is sharding
into independent indices with query-time merge, where merge cost
grows in shard count and tail latency suffers. **DistributedANN**
[DistributedANN-2025] (Microsoft, Sept 2025) instead distributes a
*single* DiskANN graph across thousands of machines: graph nodes are
partitioned by a balanced clustering, each shard holds a local
subgraph and answers traversal queries from peers, and a coordinator
orchestrates the multi-hop walk.

Reported: 50B-vector index across 1000+ machines, 26 ms median
latency and >100k QPS, recall matching the single-machine DiskANN
baseline. The architectural significance is that it preserves
per-graph guarantees (single recall curve, no merge-and-rerank
correction) at a scale where sharded indices typically lose them. It
is now the backbone of Bing's web-scale retrieval and the canonical
existence proof that single-graph ANN can be carried to hyperscale
without ensemble-of-shards.

## 4. Hybrid retrieval

Dense retrieval is necessary but not sufficient. Lexical retrieval
(BM25) catches exact-match queries dense embeddings miss — proper
nouns, codes, rare acronyms — and learned-sparse retrieval (SPLADE
[SPLADE-2021], BGE-M3 [BGE-M3-2024]) bridges the two by emitting a
sparse vector over the vocabulary with *learned* term weightings.
IBM's "three-way retrieval" study and follow-ups consistently find
BM25 + dense + learned-sparse outperforms any single retriever; the
marginal gain from the third is smaller than from the first two but
consistently positive.

Fusion — combining ranked lists from heterogeneous retrievers — is
usually solved with **Reciprocal Rank Fusion (RRF)**: each document
$d$ scores $\sum_r \tfrac{1}{k + \text{rank}_r(d)}$ with smoothing
constant $k$ (typically 60). RRF is parameter-free at the score level
(BM25, cosine, and SPLADE scores are not directly comparable),
empirically robust, and the default in Weaviate, Vespa, and
Elasticsearch hybrid search. Weighted-sum with normalization can
squeeze more recall from well-calibrated retrievers; RRF wins when
calibration is hard.

BGE-M3 is the noteworthy 2024 development: one embedding model
emitting dense, sparse, and multi-vector representations in one
forward pass, letting all three retrieval modes share a GPU pool.
The multi-vector head connects to §5.

## 5. Late interaction: ColBERTv2 and PLAID

Single-vector dense retrieval compresses a passage into one vector.
**Late interaction** preserves a vector per token (or visual patch)
and computes similarity at query time as a sum of maxima:

$$
\text{sim}(q, d) \;=\; \sum_{i=1}^{|q|} \max_{j \in [1,|d|]} \big\langle q_i,\, d_j \big\rangle
$$

Each query token finds its best-matching document token; the MaxSim
operator gives token-level alignment without the cost of a cross-
encoder pass over $|q| \cdot |d|$ pairs jointly.

ColBERT [ColBERT-2020] introduced late interaction over BERT;
ColBERTv2 [ColBERTv2-2022] adds centroid-based compression — each
token vector encoded as (centroid id, residual) — yielding 6–10×
storage compression. A 150-token passage at 128-dim fp16 is ~38 KB
of multi-vector data; the residual codec brings that under 5 KB.
**PLAID** [PLAID-2022] is the canonical serving engine: multi-stage
retrieval that narrows candidates by centroid co-occurrence, refines
with residuals, then decompresses top candidates for exact MaxSim.
PLAID reports up to 7× GPU and 45× CPU latency reduction over the
original ColBERTv2 server, and is the production substrate for
late-interaction retrieval.

The visual analogue arrived in 2024–25. **ColPali** [ColPali-2024]
embeds document *pages* directly (no OCR pipeline) using ViT patches
as the multi-vector representation — ~1024 patches/page at 128-dim
fp16, ~256 KB/page. **ColQwen2/2.5** [ColQwen-2025] is the OSS SOTA
on the same recipe with a Qwen2-VL backbone. The storage cost is real
(an order of magnitude over single-vector dense) but adoption is
heavy in document-heavy verticals (legal, finance, medical) where
avoiding the OCR + chunking pipeline is decisive.

## 6. Reranking

First-stage retrieval returns top-100 (or top-1000); the reranker
scores each $(q, d)$ pair with a cross-encoder — a transformer that
*jointly* encodes the pair and emits a relevance score. Cross-
encoders are markedly more accurate than the bi-encoders powering
dense retrieval but cost $O(k)$ forward passes per query. The
canonical reranker family — Cohere Rerank, Jina Rerank v2/v3,
BGE-Reranker, MonoT5, Qwen3-Embedding-Rerank [Qwen3-Embedding-2025] —
has moved in 2026 to LLM-class backbones (3B–8B parameters).

Latency is the bottleneck. A 6B reranker scoring 100 candidates at
4 ms/pair costs 400 ms — unaffordable for interactive RAG. Production
rerankers reduce this with batching (single forward pass over packed
candidates), early-exit (terminate once top-$k'$ is unambiguous), and
cascading (cheap reranker to top-20, expensive reranker on those).
Public p95 from the Top-8 Rerankers benchmark [Rerankers-Bench-2025]:
BGE-Reranker ~92 ms, Jina v2/v3 ~110 ms, Cohere Rerank ~210 ms.

Reranking gives a 10–30 NDCG@10 point lift over retrieval-only,
consistently across domains, making it the single most reliable
quality lever in the pipeline and the reason "retrieve top-100,
rerank top-10, generate" is the default RAG topology in 2025–2026.
The embedding-and-reranker serving stack is covered in
[see §60/05-embedding-reranker-serving](./05-embedding-reranker-serving.md).

## 7. Cross-query KV reuse: Cache-Craft

RAG creates a structural KV-reuse opportunity prefix caching cannot
exploit. A chunk retrieved for query $q_1$ may also be retrieved for
$q_2$, but at a different position, alongside different sibling
chunks, and after a different system prompt. The shared content sits
in the *middle* of each prompt, not in a common prefix.

This is the chunk-level reuse problem, and it is the regime
**CacheBlend** [CacheBlend] (covered in
[§10/07](../10-engine-core/07-prompt-prefix-caching.md)) and
**Cache-Craft** [Cache-Craft-2025] target. Cache-Craft (Adobe
Research, SIGMOD 2025) is the canonical recent treatment specific to
RAG: chunk KVs computed in isolation are *almost* correct when
concatenated, with the missing cross-chunk attention contributions
concentrated on a small set of high-attention tokens. Cache-Craft
precomputes per-chunk KVs offline, recomputes ≈5–15% of tokens on the
fly to repair cross-chunk attention, and folds the recompute into the
prefill of the unique suffix.

Reported numbers — −51% redundancy versus prefix caching, 1.6× lower
production latency — are workload-conditional but substantial.
Cache-Craft, CacheBlend, and recent CacheRAG / CacheClip variants
share a theme: when the workload is *non-prefix shared* (the
canonical RAG shape), the correct caching primitive is chunk-level
with selective recompute, not prefix-level. As of mid-2026 these
techniques live in research and specialized RAG layers (LMCache ships
CacheBlend); none is default-on in vLLM, SGLang, or TRT-LLM, because
the recompute policy needs per-model tuning.

## 8. Agentic RAG

The single-shot pipeline of §1 is increasingly insufficient. A
multi-hop question decomposes naturally into multiple retrievals plus
a join. Reasoning models with tool-use (DeepSeek-R1, Claude Sonnet
4.6, GPT-o3) drive this further: the LLM plans retrieval, issues
5–20 sub-queries per user turn, and refines based on what it finds.
The 2025 **Agentic RAG survey** [AgenticRAG-Survey-2025] catalogues
four patterns: query decomposition, multi-hop chaining, planning
loops, multi-agent retrieval. Knowledge-graph variants
[AgenticRAG-KG-2025] add structured retrieval over entity-relation
graphs as a complement to vector retrieval.

The infrastructure consequences are real. A 5-round agentic loop
becomes 5 generation calls plus 5–20 retrieval calls; if each round
writes to a shared scratchpad, the LLM's prefix grows monotonically
and prefix caching becomes critical (the $G$-fold reuse pattern from
[§10/07](../10-engine-core/07-prompt-prefix-caching.md)). Retrieval
latency dominates the inner loop, so the pipeline must be tuned for
*median*, not peak — a 95-percentile slowdown on ANN search becomes a
per-loop multiplier.

The protocol layer is converging on **MCP** (Model Context Protocol)
[MCP-Donation-2025], introduced by Anthropic in 2024 and donated to
the Linux Foundation Agentic AI Foundation in December 2025. MCP
standardizes the surface a tool — including a retrieval tool —
exposes to an LLM agent. A vector database, hybrid retriever, or
late-interaction service can ship an MCP server and be consumed by
any MCP-aware agent without bespoke client integration — the same
kind of standardization moment OpenTelemetry was for observability.

## 9. The vector database landscape

Production teams choose among four classes of vector store:

- **Managed vector DBs**: Pinecone, Weaviate, Milvus (Zilliz Cloud),
  Qdrant. Hosted, multi-tenant, mostly HNSW under the hood, with
  hybrid search, namespaces, and metadata filtering. Milvus reports
  >100k QPS at sub-30 ms p95 on millions of vectors. Default for
  greenfield RAG up to ~1B vectors.
- **In-database vector**: `pgvector` 0.8 + `pgvectorscale`
  [pgvector-08, pgvectorscale-2024] (DiskANN-on-Postgres),
  Elasticsearch / OpenSearch HNSW, ClickHouse with USearch, MongoDB
  Atlas Search. The choice when "vectors live next to relational
  data" outweighs raw QPS — practical up to ~10M for `pgvector`,
  larger for `pgvectorscale`. Heavy adoption in Postgres-standard
  enterprises.
- **Search-platform extensions**: Vespa, Elasticsearch with
  learned-sparse, OpenSearch. Strong at hybrid retrieval because the
  BM25 side has been there for two decades. Vespa is a credible
  alternative to managed vector DBs at >1B vectors.
- **Hyperscale internal**: Microsoft Bing on DiskANN / DistributedANN
  (100Bs of vectors); Google internal on ScaNN variants. The
  existence proofs for the upper end of the scale curve.

Selection logic is mostly scale × freshness × hybrid needs. Sub-1B
batch-mostly no-hybrid: managed vector DB or `pgvector`. Sub-1B with
strong hybrid: Vespa or Weaviate. Multi-billion: Milvus distributed,
Vespa, or DiskANN. 100B+: only DistributedANN and SPANN have public
existence proofs.

## Current production state

As of mid-2026, the canonical production RAG stack has a
recognizable shape. Retrieval is hybrid by default — BM25 + dense +
(often) learned-sparse fused with RRF — with HNSW behind the dense
path at small scale and DiskANN or IVF-PQ at larger scale. Cross-
encoder reranking on top-100 is standard and the single most reliable
quality lever; the reranker family has migrated from BERT-base to
LLM-backed (Qwen3-Embedding-Rerank, Cohere v3, Jina v3). Late-
interaction retrieval via ColBERTv2 + PLAID and its visual cousins
ColPali / ColQwen has moved from research curiosity to production in
document-heavy verticals despite the 5–10× embedding storage
overhead.

The serving-side picture is more in motion. Cross-query KV reuse —
Cache-Craft, CacheBlend, CacheRAG — fits RAG's non-prefix sharing
pattern but has not become engine-default; production realizations
live in LMCache and custom layers. Agentic RAG with MCP-mediated
tool surfaces is the locus of architectural change: as reasoning
models drive more retrievals per turn, retrieval-side latency becomes
a first-order TTFT contributor, prefix caching of the agent
scratchpad becomes critical, and the boundary between "RAG" and
"agent with retrieval tool" continues to dissolve.

The hyperscale frontier is set by DistributedANN serving Bing's 50B+
vectors at tens of milliseconds median over a *single* logical graph
rather than an ensemble of shards. Whether the technique generalizes
beyond Bing's specific workload is an open question for 2026, but
the publication makes it the new ceiling against which web-scale RAG
infrastructure will be benchmarked.
