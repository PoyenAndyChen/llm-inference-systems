# Test-Time Compute

**After reading this chapter, the reader will be able to:**

- Quantify how reasoning models (o1/o3, DeepSeek-R1, Claude extended thinking,
  Gemini Deep Think) shift the serving regime from "balanced prefill+decode" to
  long-decode-dominated, and read off the KV-memory and scheduling consequences
  that follow from sequence-length distributions with P50 in the thousands and
  P90 above ten thousand tokens.
- Place the branching primitives — best-of-N / self-consistency, Tree of
  Thoughts, Graph of Thoughts, Adaptive GoT — onto a single fan-out cost model,
  derive when prefix-cache reuse for sibling completions pays, and identify the
  scheduler features (radix-shared KV, sibling-aware allocation, dynamic node
  expansion) that make this regime tractable.
- Reason about the new serving primitives that reasoning models introduce:
  thinking-budget control APIs, two-stage generate-then-verify topologies with
  PRM / ORM / LLM-as-judge as separately schedulable workloads, and the SLO
  reformulation needed when a single "request" runs for tens of seconds and
  emits tens of thousands of hidden tokens.

Reasoning models broke the assumption underlying most pre-2024 serving stacks:
that the average request has a few hundred decode tokens, that decode-bound
batch fills mostly through arrival aggregation, and that an SLO can be stated
as a single tail latency. A request to o3 or to Claude Sonnet 4.6 with extended
thinking enabled may emit four thousand to thirty thousand "thinking" tokens
before the user-visible answer begins, and the operator pays for every one of
them. The forward pass per token has not changed; the *number* of forward
passes per request has, by an order of magnitude. This chapter walks through
the consequences for the serving stack — KV-memory pressure, sibling-share
prefix caching, speculative decoding tuned for long chain-of-thought, the new
thinking-budget control surface, and the verification stage that increasingly
sits behind a generation stage as a separately scaled workload.

## 1. The long-decode shift

The single quantitative fact that anchors the rest of the chapter: reasoning
workloads have decode-token distributions an order of magnitude longer than
the chat workloads the serving stack was originally optimized for.
[SparseSpec] (December 2025, arXiv:2512.01278) measures GPT-OSS 120B on
UltraChat-style reasoning prompts and reports a *median* sequence length of
3,891 tokens with P90 at 10,800 tokens. Internal vendor disclosures for
extended-thinking deployments report similar shapes; the publicly published
distribution is the cleanest reference point. For comparison, ShareGPT-style
chat traces in the 2023 vLLM and SGLang papers had median decode under 200.

Three consequences follow.

**Decode-bound regime dominates.** Prefill-decode disaggregation
([see §20-distributed-inference](../20-distributed-inference/) for chunked
prefill and disaggregated topologies) was motivated in part by the observation
that prefill and decode are different workloads. Reasoning shifts the balance
hard towards decode: tokens-emitted per request now greatly exceed
tokens-prompted in many traces. A 30,000-token thinking budget against a
1,000-token prompt inverts the prefill/decode token ratio compared to chat.
Engines that auto-tune chunk sizes or split-K attention assuming
balanced workloads see those heuristics break.

**KV memory pressure constrains batch sizing.** A 70B-class model with grouped
query attention and FP16 KV holds roughly 160 KB of KV per token. A single
30,000-token reasoning request occupies ~4.7 GB of HBM in KV alone; an 80GB
H100 with weights at ~140GB sharded over 2 GPUs has room for only a handful of
such requests *concurrently in steady state*. Either the engine pages KV to
host memory ([see §30-kv-cache](../30-kv-cache/)) or the running batch shrinks.
The throughput-per-GPU number from chat-era benchmarks does not transfer.

**Standard scheduling assumptions break.** Continuous batching
([see §10/03-batching-scheduling](../10-engine-core/03-batching-scheduling.md))
relies on request churn to keep the running batch heterogeneous and well-sized.
A reasoning request that stays in the running batch for tens of seconds amplifies
head-of-line blocking, makes admission control bursty, and shifts the SLO of
interest from per-token latency to per-request *completion* latency.
Section 7 develops the SLO reformulation in detail.

The same SparseSpec study reports up to 2.13× throughput from sparse
self-speculative decoding tuned for long chain-of-thought; that is the
single-engine answer. The rest of this chapter covers the broader response:
sibling-cached fan-out, branching algorithms, thinking budgets, and a
generate-then-verify topology.

## 2. Branching primitives

Reasoning workloads do not just produce *long* sequences; many production
configurations produce *many* sequences per logical request and aggregate them.
The simplest and oldest is **best-of-N / self-consistency** ([Wang-SC],
ICLR 2023): sample $N$ completions from the same prompt, then majority-vote or
rerank. On AIME, the reported accuracy curve from a chain-of-thought model is
roughly: single-sample 74%, consensus-64 (majority vote across 64 samples)
83%, rerank-1000 (verifier-scored) 93%. Compute scales roughly linearly in $N$;
accuracy scales sub-linearly but monotonically, which is why operators ship
$N$-way self-consistency as a standard quality knob.

**Tree of Thoughts (ToT)** ([ToT-Yao], NeurIPS 2023) generalizes the flat
$N$-way fan-out to a tree: the model proposes several next-step thoughts at
each node, evaluates them, and explores promising branches deeper. **Graph of
Thoughts (GoT)** ([GoT-Besta], 2023, arXiv:2308.09687) generalizes further to
a directed acyclic graph, allowing intermediate results to be reused across
branches. **Adaptive Graph of Thoughts (AGoT)** ([AGoT-2025],
arXiv:2502.05078, February 2025) makes the expansion decision *learned and
selective* — only some nodes are expanded, others are pruned by an
internal scoring policy. AGoT reports +46.2% on GPQA over flat
chain-of-thought; the structural lesson is that the scheduler must support
*dynamic sparse expansion* — children are spawned and possibly pruned during
execution rather than declared in advance.

The cost model under fan-out is a first-order argument for prefix-cache reuse.
Consider $B$ logical requests, each with fan-out $K$, where each completion
shares a prompt prefix of length $L_p$ and emits a unique decode tail of
length $L_d$. Let $c_{\text{KV}}$ be the per-token KV cost. Without sibling
prefix-cache reuse, the engine pays

$$\text{KV}_{\text{naive}} \;=\; B \cdot K \cdot (L_p + L_d) \cdot c_{\text{KV}}.$$

With prefix-cache reuse — the prompt is computed once per logical request and
shared across its $K$ siblings — the cost drops to

$$\text{KV}_{\text{cached}} \;=\; B \cdot L_p \cdot c_{\text{KV}} \;+\; B \cdot K \cdot L_d \cdot c_{\text{KV}}.$$

The savings are $B \cdot (K - 1) \cdot L_p \cdot c_{\text{KV}}$. For any
$K > 1$ the cached path strictly dominates, and the gap widens with $L_p$.
A 2,000-token prompt with $K = 16$ siblings saves 30,000 token-slots of KV per
logical request — frequently more KV than the decode tail itself. This is why
sibling prefix-cache reuse is not optional in branching workloads; it is the
difference between fitting the fan-out on one node and sharding it across two.

## 3. Prefix-cache reuse for sibling completions

Sibling prefix-cache reuse is mechanically the same as cross-request prefix
caching ([see §10/07-prompt-prefix-caching](../10-engine-core/07-prompt-prefix-caching.md))
with one extra constraint: the siblings are spawned *together*, share a parent
in the request graph, and the engine should allocate KV for them with that
relationship visible to the scheduler.

**SGLang RadixAttention.** ([RadixAttn], NeurIPS 2024, arXiv:2312.07104).
SGLang's radix tree over token sequences is the canonical implementation: a
parent prompt becomes an internal node, sibling completions become leaves
hanging off it, and the parent's KV blocks are reference-counted with one
reference per active sibling. Sibling spawn is a tree-extend operation rather
than a fresh insertion. SGLang reports up to 6.4× throughput on prefix-heavy
reasoning workloads. The radix structure also lets the scheduler favour
batches that reuse the most cached prefix, which under fan-out means it
naturally co-schedules siblings.

**vLLM V1 prefix cache.** vLLM V1's hash-chained block table (keyed on
`(token_id, mm_hash)` per block) deduplicates KV blocks across requests; for
sibling completions sharing a parent prompt, the same hash chain is hit by all
$K$ requests. The structural difference from a radix tree — a flat hash table
over content-addressed blocks rather than a tree — is invisible at this layer:
both engines collapse the parent prompt's KV to a single physical copy.

**LMCache and Mooncake.** ([Mooncake], FAST 2025) tier KV across CPU memory,
RDMA-attached pools, and SSD, extending the cache hierarchy beyond a single
GPU. LMCache layered on vLLM reports up to 15× throughput on multi-round QA
workloads where prefixes are large and repeated. Mooncake's KVCache-centric
disaggregated architecture is the most aggressive published deployment: a
shared KV pool serves multiple inference instances. For test-time compute
specifically, the relevance is that fan-out KV (parent prompts of branching
requests) is exactly the kind of object that benefits from a shared cross-node
pool — many siblings, many readers, one writer.

The combination — radix or hash-indexed in-engine prefix cache plus a tiered
KV pool behind it — is the production response to fan-out. Branching
algorithms (ToT, GoT, AGoT) sit on top of these primitives; the algorithms
become tractable only because the underlying cache stops the fan-out cost from
scaling as $B \cdot K \cdot L_p$.

## 4. Speculative decoding for long chain-of-thought

Reasoning workloads are decode-bound in the bandwidth sense: the verifier's
HBM-read window dominates, the ALUs are idle for most of each step, and
speculative decoding ([see §10/05-speculative-decoding](../10-engine-core/05-speculative-decoding.md))
monetizes that idle compute. Two effects of long chain-of-thought reshape the
spec-dec design space.

**Within a coherent reasoning step, $\alpha$ is high.** The token-level
predictability inside one chain-of-thought step is similar to or higher than
code completion: the model is following its own pattern, the drafter trained
on similar reasoning data tracks well, and acceptance rates of 0.75–0.85 are
common.

**At branch points, $\alpha$ collapses.** When the model commits to a
direction ("Let me try a different approach…"), the drafter cannot predict the
shift. The geometric chain $\alpha^\gamma$ truncates exactly at the branch
point. Aggregate $\alpha$ over a long chain-of-thought is the weighted average
of these two regimes, with the weighting determined by how often the model
branches.

**[EAGLE-3]** ([Eagle-3], arXiv:2503.01840, NeurIPS 2025) is the dominant
spec-dec drafter for reasoning workloads as of mid-2026. It reports up to
6.5× speedup at batch-1 on reasoning-heavy code and math tasks; vendor numbers
should be read with batch size and workload disclosed. EAGLE-3 is merged into
vLLM, SGLang, and TensorRT-LLM as a default drafter family for non-MTP models.
[See §10/05-speculative-decoding](../10-engine-core/05-speculative-decoding.md)
for the full drafter zoo and the relationship to multi-token prediction.

**SparseSpec** (December 2025) goes further by co-designing the scheduler,
verification timing, and KV management for self-speculation on long
chain-of-thought specifically. Three ideas: sparse attention in the
self-speculative drafter (drafter and target share weights, drafter runs with
a sparse attention pattern); *delayed verification* — the verifier batches
multiple small steps before running, amortizing its cost; KV management that
recognizes the long-decode shape and avoids reallocation churn. The reported
2.13× throughput at P50 sequence length 3,891 is the most directly applicable
spec-dec number for reasoning workloads in the public literature. As with all
spec-dec results, the speedup is a function of $\alpha$, hardware, and batch
mix; the headline numbers are upper bounds on what a representative
deployment sees.

The structural point: spec-dec for reasoning is not a separate algorithm but
a tuning of the same modified-rejection-sampling framework around long-decode
and branching dynamics. Step-level speculation ([Lookahead-Reasoning],
NeurIPS 2025) — the drafter proposes a *future reasoning step* rather than a
token sequence, and the verifier checks semantic correctness — is an active
research direction; reported results push spec-dec speedup on GSM8K / AIME
from 1.4× (token-level) to 2.1× (step-level). It is not yet in mainline
production engines as of mid-2026.

## 5. Thinking-budget control APIs

The shift to reasoning models exposed a new control surface that did not exist
in the chat era: how much compute a request is allowed to spend *thinking*
before producing a user-visible answer. Three vendors have converged on
broadly similar APIs with telling differences in how the budget is owned.

**OpenAI o1 / o3.** Reasoning tokens are billed and budgeted as a serving
primitive. The `reasoning_effort` knob (low / medium / high) controls how
aggressively the model is allowed to extend its hidden thinking. The hidden
tokens are billed at output rate; the user does not see them, but they appear
on the invoice and contribute to context-window consumption.

**Anthropic.** The original `budget_tokens` parameter is a hard cap: the
client specifies how many thinking tokens a request may emit before being
forced to produce its final answer. As of 2026, Opus 4.6 / 4.7 and Sonnet 4.6
support *adaptive thinking* — the model itself decides how deep to think,
within an outer budget. The shift from client-controlled to model-controlled
budget is a recent and meaningful one; it moves the depth-allocation decision
from the request author (who does not know the problem difficulty) to the
model (which has at least a partial view).

**Google Gemini Deep Think.** Extends the path length of reasoning;
operationally similar to a thinking-token budget knob. Gemini's surface keeps
the reasoning behind a separate API contour rather than exposing per-request
token-level control to the same degree as OpenAI's `reasoning_effort` or
Anthropic's `budget_tokens`. (This characterization is hedged — public Gemini
API surfaces have evolved frequently and exact semantics depend on tier.)

The cost is real. Claude Sonnet 4.6 with extended thinking bills hidden
tokens at the standard output rate (approximately $15 per million output
tokens in the public price list as of 2026). A single complex query that
emits 30,000 thinking tokens incurs roughly $0.45 of hidden cost, against
perhaps a few hundred user-visible tokens. The serving stack must surface this
separately to operators (and to billing) because per-request cost is no
longer a function of visible response length.

The infrastructure implication is that the engine sees three token classes per
request — prompt (in), thinking (hidden out), answer (visible out) — and
schedulers, cost accounting, and SLO frameworks must distinguish all three.
This is the new minimum interface.

## 6. Verification-stage scaling

The third structural change reasoning brought to serving stacks: a generation
stage and a verification stage are increasingly *separate workloads* with
different scaling profiles, and the operator schedules them independently.

**Sample-Scrutinize-Scale** (Google, ICML 2025, arXiv:2502.01839) is the
clearest published statement of the principle: *scaling verification beats
scaling generation*. For a fixed compute budget, generating fewer candidates
and verifying each more carefully Pareto-dominates generating many candidates
with cheap verification. The operational consequence is that production
reasoning stacks ship a two-stage gen+verify pipeline.

The verifier comes in three flavors:

- **Process Reward Models (PRMs)** score *intermediate steps* of a reasoning
  trace. They are typically the same scale as the generator or smaller and
  produce per-step rewards that the search algorithm uses to prune.
- **Outcome Reward Models (ORMs)** score *complete completions*. Cheaper to
  invoke (one forward pass per candidate), less informative for tree search.
- **LLM-as-judge.** A separate large model scores candidates by prompting; the
  most flexible and most expensive option, frequently used for the final
  rerank step in best-of-N pipelines.

From a serving perspective each of these is a separately schedulable model.
PRMs must run during generation (the search algorithm needs per-step rewards
in the loop), so they share the latency budget with the generator. ORMs and
LLM-as-judge run after generation completes and can be batched aggressively —
$N$ candidates per logical request fan into a single ORM batch with no
ordering constraint. Production topologies separate them: the generation
cluster runs spec-dec-accelerated reasoning with prefix-cache-shared
siblings; the verification cluster runs ORM / judge models in throughput-mode
with very different batch shapes.

The second-order point is that *both* stages benefit from the same
infrastructure (prefix caching, paged KV, spec-dec) but tuned differently.
Verifier prompts often share large prefixes (the candidate prefix is identical
across $N$ siblings), so verification-cluster prefix caching is high-yield.
Verifier outputs are short (a score), so spec-dec gains are modest. The
generation cluster has the opposite shape. Treating them as one workload
under-utilizes both. [See §60/06-rl-post-training-infrastructure](06-rl-post-training-infrastructure.md)
for the relationship to RL training, where the same gen+verify split appears
inside the inner loop.

## 7. Scheduling implications

Reasoning requests stress the scheduler in ways the chat workload did not. A
short list of pathologies and their mitigations.

**Long-running request preemption.** A 30-second reasoning request that has
emitted 20,000 tokens has 20,000 tokens of KV state. If the scheduler must
preempt it (an admission-control burst, a higher-priority arrival), recomputing
20,000 tokens of prefill is expensive. Engines respond by *paging KV to host*
([see §30-kv-cache](../30-kv-cache/)) so a preempted request can be resumed
without recomputation. Selecting *which* request to preempt — the longest
runner has the most state and the largest sunk cost — turns into a non-trivial
policy decision.

**Timeout boundaries.** Per-request timeouts inherited from chat (a 30-second
hard limit) are too tight for reasoning. Operators raise them, but raising the
timeout enlarges the worst-case state a misbehaving request can hold. A
two-tier timeout — one for time-to-first-thinking-token, one for total
completion — is the typical compromise. The first remains short (the model
should produce *some* output quickly); the second is longer and may itself be
a function of the configured thinking budget.

**SLO formulation.** TTFT and ITL ([see §00/01-inference-landscape](../00-foundations/01-inference-landscape.md))
were the chat-era SLO primitives. For reasoning workloads, four numbers
matter: time to first thinking token (operationally similar to TTFT), inter-thinking-token
latency (similar to ITL but over a longer horizon), time to first
*answer* token (the user-visible latency), and total completion time. The
business SLO is usually time-to-first-answer-token (and total completion
time); the engine-internal SLO must include the thinking-token metrics
because they drive scheduler decisions. Hedge: there is no industry-standard
formalization of these four numbers as of mid-2026; the names and
operationalization vary across operators.

**Branching-aware admission control.** A request that will fan out to $K = 64$
siblings should not be admitted as if it were one request. Schedulers that
expose a fan-out hint can pre-allocate KV for the parent prefix once and admit
siblings as the parent completes prompt-prefill. Engines without this hint
under-allocate or thrash; the radix-tree-based engines (SGLang) handle this
more naturally than hash-table-based engines (vLLM V1), although both can be
adapted with sibling-aware allocation policies.

The scheduling story for reasoning is not a clean separate framework; it is a
re-tuning of continuous batching, paged KV, and prefix caching against a
sequence-length distribution that has shifted by an order of magnitude. The
primitives are the same; the operating point is different.

## Current production state

As of mid-2026, every major serving stack (vLLM, SGLang, TensorRT-LLM,
Anthropic's internal stack, OpenAI's reasoning stack to the extent it can be
inferred from API behavior, Gemini's serving fleet) ships the building blocks
this chapter covers: long-decode-tolerant paged KV with host paging, automatic
prefix caching with sibling-aware reuse, EAGLE-3 or MTP-based speculative
decoding tuned for the long-decode regime, and a thinking-budget control
surface exposed through the API. The convergence at the primitive level is
near-total. What differs is the API contour around them.

OpenAI's `reasoning_effort` (low / medium / high) and Anthropic's adaptive
thinking represent the two broad design choices for thinking-budget exposure:
discrete operator-controlled tiers versus model-controlled depth within an
outer cap. Both are settling as defaults with different ergonomic properties;
neither has clearly displaced the other. Google's Deep Think keeps the
reasoning surface more opaque, with the tradeoff that the operator gives up
some control in exchange for the vendor managing the depth-allocation
decision. The cost of thinking is a real and increasingly first-class line
item: a $0.45 hidden-token cost on a single Sonnet 4.6 query is unremarkable,
which would have been unthinkable in the chat era.

The verification-stage-as-separate-workload pattern is most visible in
research papers (Sample-Scrutinize-Scale and the broader process-reward and
LLM-as-judge literature) and in RL post-training stacks
([see §60/06-rl-post-training-infrastructure](06-rl-post-training-infrastructure.md))
where it sits at the heart of the rollout-and-score loop. In production
inference for end-user reasoning, the topology is most evident in best-of-N /
self-consistency pipelines and in the verifier-rerank stages of agentic
products; it is not yet a uniformly exposed feature in mainstream model APIs,
but the underlying serving infrastructure for it (prefix-shared sibling
generation, batched verifier inference, two-cluster topology) is in production
at the major labs. The trajectory is clear: reasoning workloads are the
dominant new shape of LLM inference, and the serving stack reflects the
shift across the entire stack rather than as a single feature flag.
