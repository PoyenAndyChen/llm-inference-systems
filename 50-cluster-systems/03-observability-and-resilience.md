# Observability and Resilience

**After reading this chapter, the reader will be able to:**

- Identify the LLM-specific signals that traditional APM tools miss, name
  the canonical engine metric set (TTFT, ITL, ITL p99, KV utilization,
  queue depth, prefix-hit rate, preemptions), and translate each metric
  into an operator-actionable alert threshold.
- Place the OpenTelemetry GenAI semantic conventions on the standards
  timeline, distinguish span attributes from event-attached prompts, and
  reason about prompt redaction and per-tenant cost attribution as
  privacy-and-billing problems with the same plumbing.
- Describe the fault-tolerance pattern stack — DéjàVu KV replication,
  request-level live migration (Llumnix lineage), elastic expert-parallel
  groups, StormService P/D lifecycle — and explain why a 30-minute
  reasoning rollout makes worker-failure recovery a first-class scheduler
  concern rather than a niche feature.

A serving cluster that cannot be observed cannot be operated. Through
2024 the community largely re-used standard web-tier telemetry —
request rate, error rate, p99 latency — and discovered those signals
miss the failure modes that matter: KV-pressure preemption, prefix-hit
collapse after a deploy, EP-group desync, a 30-minute reasoning
generation evaporating when one GPU faults. By 2025–2026 a canonical
engine metric set, a draft OpenTelemetry GenAI semconv, and a small
set of fault-tolerance primitives have emerged across vLLM, SGLang,
TRT-LLM, llm-d, AIBrix, and KServe. This chapter walks the layers
from metric to dashboard to chaos.

## 1. The observability gap

Standard APM treats a request as a single timed RPC: start timestamp,
end timestamp, status code. LLM serving breaks every assumption behind
that model. A request has three phase boundaries — queue admission,
prefill completion (first token), decode completion — and the durations
of prefill and decode are governed by *different* hardware bottlenecks
(compute-bound vs memory-bandwidth-bound;
[see §00/02-transformer-arithmetic-roofline](../00-foundations/02-transformer-arithmetic-roofline.md)).
A single-number "p99 latency" averages over those phases and tells the
operator nothing actionable.

The cluster also has shared state — the KV cache — whose utilization
governs whether new requests are admitted, preempted, or queued. KV
utilization is invisible to request-level telemetry; it must be sampled
directly from the engine. Prefix-cache hit rate is similarly an
engine-level gauge
[see §10/07-prompt-prefix-caching](../10-engine-core/07-prompt-prefix-caching.md).
And throughput as a scalar is misleading: a cluster can sustain high
tokens-per-second while individual requests miss TTFT SLA when batches
are large and queue-time dominates. The right scalar is
goodput-under-SLO — tokens delivered within latency target — which
requires per-request TTFT and ITL histograms, not request-rate counters.

## 2. OpenTelemetry GenAI semantic conventions

OpenTelemetry's GenAI semantic conventions (semconv) are the
cross-vendor standard for tracing LLM calls [OTel-GenAI]. As of v1.37
the spec is "In Development" — opt-in via the environment variable
`OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`. Stable
release is expected in 2026; consumers building on it today should
expect attribute renames between minor versions.

### 2.1 Span attributes

Every GenAI client span carries a small core attribute set
[OTel-GenAI-spans]:

- `gen_ai.system` — provider identifier (`anthropic`, `openai`, `aws.bedrock`).
- `gen_ai.request.model` and `gen_ai.response.model`.
- `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens`.
- `gen_ai.request.temperature`, `gen_ai.request.max_tokens`, etc.

Provider-specific extensions — Anthropic, AWS Bedrock, Azure AI
Inference, OpenAI, MCP — layer on additional attributes (cache-read
tokens, cache-creation tokens, tool-use spans). Agent and framework
spans (tools, tasks, agent teams, memory, artifacts) are under active
development and ship behind their own opt-in flags [OTel-agent].

### 2.2 Prompts and completions as events

The spec deliberately does not store prompt or completion text in
span attributes — long prompts blow per-attribute size budgets, and
defaulting raw user input into tracing backends is a privacy
non-starter. Prompt and completion content attaches as **events**, off
by default, opt-in via
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`. This is the seam
where redaction lives — see §5.

### 2.3 Engine metrics vs client tracing

The semconv covers client-side spans well; it does not yet cover
engine-internal metrics (KV utilization, prefix-hit rate, preemptions),
which remain in engine-specific Prometheus namespaces (vLLM's
`vllm:*`, SGLang's `sglang:*`, TRT-LLM's endpoint). Cross-engine
harmonization is an unsolved standardization problem.

## 3. The canonical engine metric set

vLLM v1's metrics endpoint defines the de facto standard [vLLM-metrics],
mirrored with minor naming differences in TGI [TGI], SGLang, TRT-LLM,
and the llm-d EPP gateway [llm-d-observability]. The list below uses
vLLM's names; the semantics carry across.

### 3.1 Counters and histograms (per-request)

- `vllm:request_prompt_tokens`, `vllm:request_generation_tokens` —
  input/output token distributions. The output distribution's long
  tail (reasoning rollouts) is the load-shape signal that broke
  2024-era schedulers.
- `vllm:e2e_request_latency_seconds` — wall-clock arrival-to-last-token
  histogram; percentiles taken at scrape time.
- `vllm:time_to_first_token_seconds` (TTFT) — arrival to first decoded
  token. p95 is the user-visible "is it alive?" metric and the primary
  scaling signal.
- `vllm:time_per_output_token_seconds` (ITL / TPOT) — inter-token
  latency during decode. p99 catches batch-induced stalls invisible to
  the median.
- `vllm:request_queue_time_seconds` — admission delay. Nonzero p50
  means the cluster is saturated.
- `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds`
  — phase splits, useful for diagnosing whether to scale prefill or
  decode workers under disaggregation
  [see §20/02-prefill-decode-disagg](../20-distributed-inference/02-prefill-decode-disagg.md).

### 3.2 Gauges (cluster state)

- `vllm:num_requests_running`, `vllm:num_requests_waiting` — batch
  occupancy and queue depth.
- `vllm:gpu_cache_usage_perc` — fraction of paged KV blocks in use;
  the single most important LLM-serving gauge (see §4).
- `vllm:cpu_cache_usage_perc` — host-tier KV offload usage when
  configured [see §30/05-kv-offload-and-tiering](../30-kv-cache/).
- `vllm:num_preemptions_total` — counter of requests evicted from the
  running batch; should be near zero in steady state.

### 3.3 Derived metrics

- **Goodput-under-SLO**: $\text{goodput} = \sum_i \mathbb{1}[\text{TTFT}_i \le S_{\text{ttft}}] \cdot \text{tokens}_i / T$
  — the right scalar when comparing engine versions.
- **Prefix-hit rate**: fraction of prompt tokens served from cache,
  via gateway-level KV-event subscription (AIBrix v0.4+ [AIBrix-v0.4],
  llm-d EPP) or per-engine probe.
- **Cost per million tokens**: gateway meter joining
  `gen_ai.usage.*_tokens` against per-tenant pricing.

## 4. Dashboard patterns and alerting

A four-panel standard has converged across vLLM Production Stack
[vLLM-prod-stack], llm-d [llm-d-observability], AIBrix [AIBrix-arch],
and KServe deployments:

```
┌─ Panel 1: Throughput ─────────┬─ Panel 2: Per-token latency ──┐
│ requests/s                    │ ITL p50 / p95 / p99           │
│ TTFT p50 / p95 / p99          │ TPOT distribution             │
├─ Panel 3: Cluster state ──────┼─ Panel 4: Saturation ─────────┤
│ KV utilization (%)            │ num_requests_running          │
│ prefix hit rate               │ num_requests_waiting          │
│ queue depth                   │ preemptions/min               │
└───────────────────────────────┴───────────────────────────────┘
```

Each panel maps to a distinct failure class: throughput catches
front-door issues, per-token catches batch jitter and scheduler
regressions, cluster-state catches KV-pressure preemption storms (the
most common LLM-specific incident) before they become user-visible,
and saturation closes the loop with the autoscaler
[see §50/02-autoscaling-cost-and-sustainability](02-autoscaling-cost-and-sustainability.md).

### 4.1 Alert thresholds (derive)

Thresholds are workload-dependent; the *form* of each rule generalizes:

- **TTFT p95 > $S_{\text{ttft}}$** for $\Delta t$ minutes → page; check
  KV utilization and queue depth before scaling, since the fix differs.
- **`gpu_cache_usage_perc` > 0.9** sustained → imminent preemption
  storm. The threshold is below 1.0 because chunked-prefill admission
  reserves headroom; pinned at 0.9 means the scheduler is already
  shedding work via the queue.
- **`queue_depth` > $Q_{\text{soft}}$** → admission control (return
  429) or autoscale; the choice is tenant policy. Gateways like
  LiteLLM and Portkey surface this as a routing rule.
- **Prefix-hit rate sudden drop** (e.g., 30 points in 1 minute) →
  eviction storm or cache invalidation, common after a system-prompt
  rotation. Prefix-hit telemetry catches this in seconds versus 10+
  minutes via TTFT alone.
- **`num_preemptions_total` rate > 0** in steady state → scheduling
  bug or under-provisioning; the legitimate steady-state rate is zero.

### 4.2 Grafana, Prometheus, and the llm-d pipeline

llm-d standardizes the pipeline: engines expose Prometheus metrics,
the EPP scrapes and aggregates per-pool, Grafana dashboards ship
in-repo, and OTel traces flow to a configurable collector
[llm-d-observability]. AIBrix's controller plane exports an analogous
set [AIBrix-launch]. A v0.5 llm-d deploy ships a working dashboard
out of the box [llm-d-v0.5].

## 5. Prompt and completion logging: redaction patterns

Debugging (where operators want full prompt text) and auditing/billing
(where operators want metadata only) collide in production. The OTel
`capture_message_content` flag is binary; real systems sit between via
one of three redaction architectures:

1. **Regex/classifier scrub at the SDK boundary** — PII patterns are
   stripped before prompts enter traces. Simple, fast, prone to silent
   miss-and-leak.
2. **Sampled retention** — full prompts logged for ~1% of requests
   stratified by tenant and model; the rest carry only token counts.
   The right default for most deployments.
3. **Vault/tokenization** — prompt text replaced with an opaque token
   at the gateway, content stored in a privacy-isolated vault behind
   separate access controls; spans carry only the token. LiteLLM,
   Portkey, Langfuse, and Helicone implement variants
   [LiteLLM, Portkey, Helicone-Survey]; Datadog's GenAI integration
   follows the same shape [Datadog-OTel-GenAI].

The decision is policy, not engineering — the underlying plumbing
(OTel events gated by `capture_message_content`, redaction at SDK or
collector) is uniform across the three.

## 6. Cost attribution via OTel baggage

Per-tenant metering rides the same telemetry stream as dashboards.
Identity headers (`x-tenant-id`, `x-user-id`, `x-team`) arrive at the
gateway, propagate as OTel **baggage** through prefill and decode
workers, and attach to every span. At aggregation time, baggage joins
`gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` to yield
per-tenant token counts; multiplying by per-million-token rates yields
spend. Billing and cost attribution are then one problem: metadata
propagation plus token-count joins. LiteLLM and Portkey expose this
as a first-class feature; DIY deployments must ensure the EPP
propagates baggage and that engines do not strip it across
prefill-decode boundaries
[see §50/01-router-gateway](01-router-gateway.md).

## 7. Fault tolerance: the 30-minute reasoning problem

Long-form reasoning makes worker failure expensive in a way it was
not in 2023. A 30-minute single-request decode generating 50K tokens
at 30 tokens/sec can accumulate tens-to-hundreds of GB of KV state on
a long-context model; if the GPU holding that KV faults at minute 28,
restarting from scratch costs 28 minutes of user-visible latency
*plus* the full prefill compute. Three primitives have emerged to
make this recoverable.

### 7.1 DéjàVu: KV-cache replication for fault tolerance

DéjàVu (ICML 2024) is the canonical paper for KV-replication as
fault-tolerance, introducing **KV-cache streaming with redundant
replication** for transparent worker-failure recovery [DejaVu-FT]. The
insight: the expensive per-request state is the KV cache, not the
model weights (already replicated across serving workers). Streaming
KV deltas to a replica — synchronously at block boundaries,
asynchronously between — means a worker failure surrenders only the
trailing $\Delta t$ since last replication. The replica takes over
decode, and the user sees a small ITL bump rather than a 30-minute
restart. Cost is non-trivial — KV blocks are large and replication
consumes interconnect bandwidth — so production systems typically
enable it for long-context or high-priority requests, not universally.

### 7.2 Request-level live migration

Complementary to KV replication: moving an in-flight request from a
draining or failing worker to a healthy peer. Llumnix (OSDI 2024)
introduced live request migration as a *scheduling* primitive — moving
requests for load balance — and the mechanism (KV-block transfer over
high-bandwidth interconnect plus scheduler hand-off) is reused for
fault tolerance [Llumnix]. llm-d 0.5 supports request-level
rescheduling on the EPP path [llm-d-v0.5]; KServe's controller
integration drains a node without dropping in-flight requests
[KServe-llmd]; AIBrix v0.4+ introduces the **StormService CRD** as a
Kubernetes-native lifecycle controller for prefill/decode pods,
supporting controlled failover and upgrades
[AIBrix-v0.4, AIBrix-arch]. Common pattern: a control-plane signal
(drain, fault, upgrade) triggers KV transfer to a target worker, the
scheduler updates routing, and the request continues without restart.

### 7.3 Elastic expert parallelism

MoE deployments raise the failure blast radius: DeepSeek-style models
distribute experts across 16–64+ GPUs, and a single GPU loss
historically tore down the whole EP group. SGLang's Elastic EP (March
2026) provides **partial-failure tolerance**: a failed node's experts
remap to surviving peers with reduced capacity, the EP group remains
live with degraded throughput, and recovery proceeds asynchronously
[Elastic-EP-SGLang]. The cluster loses some performance; it does not
lose the deployment. The published numbers cover specific
DeepSeek-MoE configurations; behavior under arbitrary expert layouts
and replication factors is less characterized.

## 8. Chaos engineering for LLM serving

LLM-serving chaos is less developed than web-tier chaos, but its
shape is becoming clear. Failures to inject:

- **GPU faults** — engine SIGKILL, PCIe link reset, NVLink partition.
  Measure: TTFT/ITL during recovery, fraction of in-flight requests
  rescued vs lost.
- **KV-eviction storms** — flood with long-prefix requests exceeding
  aggregate KV capacity. Measure: `num_preemptions_total` rate, p99
  ITL, queue-depth spike-and-drain time.
- **Network partitions** — split prefill from decode workers under
  P/D-disaggregation
  [see §20/02-prefill-decode-disagg](../20-distributed-inference/02-prefill-decode-disagg.md).
  Measure: fail-fast vs hang behavior, gateway retry-storm shape.
- **Prefix-cache invalidation** — bulk-evict the radix tree or block
  pool. Measure: TTFT degradation curve, re-warm time.
- **Expert-parallel group desync** — kill one GPU in an EP group.
  Measure (with elastic EP): degraded throughput and recovery time;
  (without): blast-radius and restart cost.

The measurement vocabulary is the canonical metric set from §3 — chaos
is useful when its impact registers on dashboards operators already
watch. Tooling is bespoke; no chaos-monkey-for-LLM-serving exists in
open source, and what runs lives inside hyperscaler platforms. The
material here reflects practitioner reports and emerging patterns
rather than a settled discipline; expect formalization through
2026–2027.

## Current production state

The metric set is settled. Across vLLM, SGLang, TRT-LLM, TGI, llm-d,
and AIBrix the vocabulary — TTFT, ITL, KV utilization, queue depth,
preemptions, prefix-hit rate — is identical up to naming, and the
four-panel dashboard ships out-of-the-box on llm-d 0.5 and AIBrix
v0.4+. Operators moving between deployments find the same gauges with
the same semantics, which is the strongest possible evidence that the
abstractions are correct. The OpenTelemetry GenAI semconv is the
remaining standardization gap: client-side span attributes are stable
in practice and likely to GA in 2026, but engine-internal metrics
(`vllm:gpu_cache_usage_perc` and its peers) are not yet covered, and
cross-engine harmonization remains a manual mapping.

Fault tolerance is mid-flight. DéjàVu's KV-replication design is
canonical at the paper level but not yet a default in any open-source
engine; production deployments running 30-minute reasoning rollouts
typically combine request-level migration (Llumnix lineage, llm-d 0.5,
AIBrix StormService) with application-level checkpointing of the
reasoning state itself. Elastic EP (SGLang, March 2026) is the most
recent and most consequential addition: by removing whole-group
failure as a mode, it makes large MoE deployments operationally
tractable for teams that cannot afford bespoke control planes. The
fault-tolerance numbers in vendor blog posts cover narrow
configurations; treat them as existence proofs rather than
specifications, and validate on the actual workload shape before
relying on them in an SLO.

Chaos engineering for LLM serving is the laggard. The metric vocabulary
is in place to *measure* chaos impact; the tooling to *inject* it
systematically — equivalents of Gremlin, Chaos Mesh, AWS FIS adapted
to GPU-and-KV failure modes — is bespoke at every shop running them.
Expect this to consolidate over the next 12–24 months as elastic-EP and
request-migration features create enough working fault-tolerance
substrate to make systematic injection worth the engineering investment.
