# RL Post-Training Infrastructure

**After reading this chapter, the reader will be able to:**

- Recognize an RL post-training stack as a system whose hot path is an
  *inference engine* — the rollout engine — operating under invariants (no
  per-request SLO, frequent weight churn, group sampling, long generations)
  that are systematically different from user-facing serving, and read off
  the consequences for batching, prefix caching, and weight sync.
- Place the major framework families — veRL HybridEngine, OpenRLHF, AReaL,
  TRL, NeMo-RL, slime, SkyRL — onto the 2×2 design space of (colocated vs
  disaggregated) × (synchronous vs asynchronous), derive the staleness and
  weight-sync cost models that decide which cell a deployment should sit in,
  and identify the failure modes (numerical mismatch, off-policyness,
  long-tail rollouts, MoE routing inconsistency) active in 2025–2026.
- Reason about ongoing shifts: GRPO and Dr.GRPO replacing PPO as the default
  objective, RLVR with verifier pools displacing learned reward models for
  reasoning, speculative decoding entering the rollout path (NeMo-RL v0.6,
  SPEC-RL) after years of being avoided, and the public glimpses of
  frontier-lab stacks (DeepSeek-R1/V3.2, Llama 4 online RL, Kimi K2) that
  show where the field is heading.

Reinforcement learning post-training is the dominant capability differentiator
of the 2025–2026 frontier-model generation. DeepSeek-R1, OpenAI o1/o3,
Anthropic's Claude reasoning stack, Google Gemini Deep Think, Llama 4, and
Kimi K2 all attribute their reasoning step-change to RL applied after a base
SFT model. From an inference-infrastructure perspective the relevant fact is
that an RL training pipeline contains an inference engine in its inner loop.
That engine — the *rollout engine* — runs the same model on the same hardware
with most of the same optimizations as a serving engine, but its operating
point is different enough that a subset of techniques behave differently or
break outright. This chapter is organized around that asymmetry: where
rollout-as-serving works, where it fails, and what the major frameworks do
about it.

## 1. Why an inference engineer needs to know RL infra

Three facts make RL post-training infrastructure a serving problem.

First, the rollout engine *is* an inference engine. veRL embeds vLLM, SGLang,
or TensorRT-LLM as its rollout backend. NeMo-RL embeds vLLM and TensorRT-LLM.
OpenRLHF and SkyRL embed vLLM. AReaL and slime embed SGLang. TRL embeds
vLLM V1. Every framework reuses the production serving stack rather than
writing a separate inference path, because rewriting paged attention, prefix
caching, and continuous batching for the RL inner loop would be duplicative.

Second, the operating point is different. A serving engine optimizes for
per-request latency under SLOs (TTFT, ITL); a rollout engine optimizes for
*aggregate throughput* of a batch of completions, with no per-request SLO.
A serving engine's weights are static for the lifetime of the deployment; a
rollout engine's weights change every training step. A serving engine sees a
mix of prompt lengths from arbitrary users; a rollout engine sees N
completions per prompt, with prefix-cache reuse density that serving rarely
sees.

Third, the failure modes are partially shared and partially novel. Rollout
shares with serving the long-tail latency problem from variable output length
([see §10/03-batching-scheduling](../10-engine-core/03-batching-scheduling.md))
and the KV-pressure problem from long generations
([see §30-kv-cache](../30-kv-cache/)). It introduces novel failure modes —
weight-sync overhead, off-policyness from staleness, BF16 numerical mismatch
between rollout and training, MoE routing inconsistency — with no analog in
serving.

## 2. The RL pipeline

The InstructGPT three-stage template ([InstructGPT], arXiv:2203.02155) — SFT,
reward-model training, then PPO against the reward model — established the
component decomposition every modern RL post-training framework still uses.
The pipeline contains six logical roles, which may be merged onto the same
physical workers or split across them.

- **Actor.** The policy being trained. Receives gradients; produces logits
  during training updates.
- **Reference.** A frozen copy (typically the SFT initialization) used for
  the KL penalty term. Forward only.
- **Reward.** A reward model, verifier, or rule-based scorer producing
  scalar rewards on completions or per-step rewards on traces.
- **Critic.** A value model used in PPO-style algorithms. Removed entirely
  in GRPO and its descendants.
- **Trainer.** Optimizer state and gradient computation; runs FSDP,
  DeepSpeed ZeRO, or Megatron-Core.
- **Rollout engine.** The inference engine that generates completions from
  the actor's current weights.

Two orchestration patterns exist. *SPMD* — every worker runs the same program
— is used in OpenRLHF and TRL: one driver invokes training and rollout
phases on a fixed set of workers. *Hybrid* — MPMD orchestration over SPMD
workers — is veRL's HybridFlow ([HybridFlow], arXiv:2409.19256, 2024): a
single controller issues phase-tagged operations to worker pools that may run
different programs (trainer vs rollout) on overlapping or distinct GPU sets.
The hybrid pattern is what enables HybridEngine's resharding-during-phase-
transition (see §5). DeepSpeed-Chat's Hybrid Engine — the first system to
combine training and inference on the same GPUs — is the lineage ancestor of
veRL's HybridEngine.

## 3. Rollout as special-case serving

Five differences from user-facing serving matter for infrastructure decisions.

**No per-request SLO.** The only quantity that matters is *batch completion
time* — when have all N completions in the rollout batch finished? This
permits batching, max-throughput-mode kernels, and admission-control choices
that a serving engine cannot make.

**Frequent weight churn.** The actor's weights change every training step.
Each weight update invalidates the rollout engine's parameter copy and may
invalidate compiled CUDA graphs and kernel autotuning state. A serving engine
amortizes startup tuning over weeks; a rollout engine cannot.

**Group sampling.** GRPO and its descendants generate G ∈ {4, 8, 16, 32, 64}
completions per prompt. Within a group all completions share the prompt
prefix exactly. The rollout engine sees a workload whose *native* shape is
many-to-one, pushing prefix caching from "nice to have" to "required" (§4).

**Long generations.** RL post-training on reasoning data routinely uses
max-generation lengths of 32k–64k tokens (DeepSeek-R1 trained at 32k extended
to 64k mid-run). The long-tail latency problem from
[§10/03-batching-scheduling](../10-engine-core/03-batching-scheduling.md)
becomes the dominant rollout cost (§6).

**Aggregate-throughput-only.** The rollout engine can exhaust KV memory by
packing a very large batch and accept that any individual completion may be
preempted; serving cannot make this trade because preempting a user request
is visible.

The architectural consequence is that vLLM, SGLang, and TensorRT-LLM ship
*explicit RL-rollout modes* — APIs like `update_weights_from_distributed`,
`sleep` / `wake_up`, and `collective_rpc` — that disable user-facing
invariants and expose internal state to the trainer (§5).

## 4. GRPO and prefix-cache implications

Group Relative Policy Optimization (GRPO), introduced in [DeepSeekMath]
(arXiv:2402.03300), removed PPO's critic by replacing learned baselines with
a *group-relative* baseline. For a prompt $x$, the policy generates $G$
completions $\{y_1, \ldots, y_G\}$ and the reward model produces scalars
$\{r_1, \ldots, r_G\}$. The advantage for token $t$ in completion $i$ is

$$A_{i,t} \;=\; \frac{r_i - \mathrm{mean}(r_1, \ldots, r_G)}{\mathrm{std}(r_1, \ldots, r_G)}.$$

The GRPO objective applies standard PPO clipping, with the KL penalty moved
to the loss (rather than the reward) and the per-group baseline:

$$\mathcal{L}_{\text{GRPO}} \;=\; \mathbb{E}\!\left[\sum_{i=1}^{G} \min\!\left(\rho_{i,t} A_{i,t},\; \mathrm{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon) A_{i,t}\right) - \beta \cdot \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})\right],$$

where $\rho_{i,t}$ is the importance ratio under the current vs behavior
policy. The critic disappears, halving the model count in the training mesh.

[Dr.GRPO] (arXiv:2503.20783, 2025) drops the standard-deviation normalization
and the implicit length normalization, removing a length bias that inflated
long completions' advantage. [DAPO] (arXiv:2503.14476, 2025) keeps the
group-relative baseline but adds Clip-Higher (asymmetric clipping ranges to
encourage exploration), Dynamic Sampling (continue sampling until the group
has both correct and incorrect completions), Token-Level PG Loss, and
Overlong Reward Shaping. veRL is the canonical reference implementation for
DAPO.

The serving implication is the prefix-cache math. With $G$ completions per
prompt of length $L_p$, each emitting decode tail $L_d$, KV cost without
sibling-prefix reuse is $G \cdot (L_p + L_d) \cdot c_{\text{KV}}$, versus
with reuse $(L_p + G \cdot L_d) \cdot c_{\text{KV}}$, saving $(G-1) \cdot L_p
\cdot c_{\text{KV}}$ per group. For DeepSeek-R1's training config — $G = 16$,
prompts of order 1k tokens — the saving is comparable to the entire decode
KV. The mechanism is the same as the sibling-cached fan-out for test-time
compute ([see §60/01-test-time-compute](01-test-time-compute.md)); the
difference is that in RL the fan-out is the *normal case*, not a feature
flag. SGLang's RadixAttention and vLLM V1's hash-chained block table both
handle this natively.

## 5. Weight synchronization

Every training step ends with a weight-sync phase: updated parameters must
reach the rollout engine before the next rollout batch. The cost of this
transfer is the principal hidden tax in colocated synchronous frameworks.

For a model of $N$ parameters at $b$ bytes per parameter synced across $K$
rollout replicas with effective bandwidth $B_{\text{eff}}$, plus one-time
overhead $L$ and per-rank overhead $R$:

$$T_{\text{sync}} \;\approx\; \frac{N \cdot b}{B_{\text{eff}}} + L + K \cdot R.$$

For a 1T-parameter BF16 model (2T bytes) on a thousand-GPU cluster, NCCL
broadcast over NVLink+IB measures around 20 seconds; an RDMA P2P transfer
through Mooncake's checkpoint engine measures around 6 seconds for the same
configuration ([Mooncake], FAST 2025). For smaller models the constant
overhead dominates: a 70B BF16 model is 140GB; on a single node with NVLink,
NCCL broadcast takes 100–500 ms naively, dropping to about 20 ms with veRL's
bucketing optimizations.

Five mechanisms cover the production landscape.

- **NCCL broadcast.** Canonical and well-debugged. The trainer's rank-0
  broadcasts each parameter (or shard) to the rollout replicas. Bucketing —
  packing multiple parameter tensors into one NCCL call — amortizes per-call
  overhead and is the difference between 100ms and 20ms for mid-sized models.
- **CUDA-IPC handles.** When rollout and trainer share GPUs, the trainer
  hands the rollout a CUDA IPC handle and the rollout reads parameters
  directly from the trainer's HBM. Near-zero latency, no copy.
- **RDMA P2P.** [Mooncake]'s checkpoint engine and Awex push parameters over
  RDMA between disaggregated trainer and rollout pools without going through
  host memory. Trillion parameters synced in ~6s on thousand-GPU clusters.
- **Checkpoint streaming.** Trainer writes a checkpoint to a fast shared
  filesystem; the rollout streams it in. Slowest but most decoupled.
- **In-process tensor handoff.** TRL's vLLM-V1 backend shares actor tensors
  with the rollout engine via reference rather than copy, exploiting a shared
  address space.

vLLM exposes `update_weights_from_disk`, `update_weights_from_tensor`,
`update_weights_from_distributed`, `wake_up(tags=[...])` and `sleep(level=1|2)`
for rollout-pause-during-update, and `collective_rpc(...)` for arbitrary
trainer-to-rollout coordination. SGLang mirrors `update_weights_from_*` and
adds an explicit `pause_generation → update → continue` invariant with a
`stop_all_requests` knob. The `sleep` / `wake_up` pair is the colocated-mode
specialty: the rollout releases its KV cache and weights from HBM
(level=1: weights only; level=2: weights+KV) while the trainer takes those
GPUs for a backward+step, then wakes the rollout back up after the weights
update in place. This eliminates the broadcast on a colocated deployment but
requires careful HBM accounting.

## 6. Co-located vs disaggregated, sync vs async

The architectural design space factors into a 2×2: where do trainer and
rollout sit (same GPUs vs different), and do they operate in lock-step or
independently?

```
              Synchronous              Asynchronous
            ┌──────────────────────┬──────────────────────┐
 Colocated  │ veRL HybridEngine    │ veRL one-step-off    │
            │ OpenRLHF colocated   │ veRL fully-async 0.7 │
            │ NeMo-RL colocated    │                      │
            ├──────────────────────┼──────────────────────┤
 Disagg     │ OpenRLHF separated   │ AReaL                │
            │ SkyRL disagg sync    │ ROLL Flash           │
            │                      │ PipelineRL           │
            │                      │ slime async mode     │
            └──────────────────────┴──────────────────────┘
```

**Colocated synchronous.** The default. Trainer and rollout share GPUs and
alternate phases. veRL HybridEngine's distinguishing feature is *resharding
between phases* — the trainer holds parameters in a 3D-parallel layout
(TP × PP × DP) optimized for backward, then HybridEngine transforms them
in-place into the rollout's preferred layout (often pure TP or TP × DP) on
phase switch. This is what makes "the same weights serve both phases on the
same GPUs without going through CPU" feasible at scale. The cost: total step
time is $T_{\text{train}} + T_{\text{rollout}} + T_{\text{sync}}$, with each
side idle during the other.

**Disaggregated synchronous.** Trainer and rollout occupy disjoint GPUs.
Same wall-clock dynamics as colocated synchronous (one phase active at a
time) but with weight-sync going over the network. Generally worse on a
single node; better at scale because trainer and rollout can be sized
independently.

**Asynchronous.** Rollout and trainer operate on separate clocks. Wall-clock
per step becomes

$$T_{\text{async}} \;\approx\; \max(T_{\text{rollout}},\, T_{\text{train}})$$

versus the synchronous $T_{\text{rollout}} + T_{\text{train}} +
T_{\text{sync}}$. The speedup is bounded by 2× when the phases are balanced.
AReaL ([AReaL], arXiv:2505.24298, 2025) reports up to 2.77× over synchronous
baselines on long-CoT workloads, exceeding the 2× ceiling because async also
unblocks the long-tail latency penalty:

$$\mathbb{E}[T_{\text{batch}}] \;=\; \mathbb{E}[\max_{i=1..B} L_i] \;\sim\; \log(B) \cdot \mathbb{E}[L]$$

for sub-exponential tails. At $B = 1024$, the synchronous rollout is
bottlenecked on the slowest completion (often 10× the median). Async
frameworks decouple batch boundary from step boundary: the trainer pulls
completions as they finish, with no requirement that all $B$ complete first.
This is the second source of async's gain beyond the $\max$-versus-sum
comparison.

## 7. The async stability problem

Async trades wall-clock speed for *off-policyness*: the rollout engine's
weights at generation time may lag the trainer's current weights by $k$
steps. Three controls bound the bias.

**Per-sample version rejection.** Drop completions whose generating-policy
version is older than a threshold; wastes rollout compute but stays on-policy.

**Depth bounding.** Cap staleness to small $k$ ($k = 1$ for "one-step-off";
$k \leq 4$ in AReaL's default). veRL 0.7 ships one-step-off as a stepping
stone between sync and fully async.

**Importance-sampling correction.** Reweight the gradient by current vs
behavior policy probability. PPO's clipped ratio handles small staleness;
larger staleness needs the decoupled-PPO formulation in AReaL or similar
reweightings. IS variance grows with staleness; depth bounding caps it.

A second class of stability problem appeared in 2025–2026 that is not about
algorithmic off-policyness but about *numerical mismatch* between rollout
and trainer.

**BF16 numerical mismatch.** vLLM/SGLang's BF16 attention kernels and
Megatron-Core's BF16 attention kernels do not produce bitwise-identical
logits even given identical weights and inputs. Different reduction orders,
different fused kernels, different epsilon placements in LayerNorm/RMSNorm
cause logit deltas of order $10^{-3}$. In synchronous RL this is invisible
(the rollout is treated as the ground-truth behavior policy); in async,
where the trainer recomputes log-probs over the rollout's output to compute
the gradient, the recomputed log-probs disagree with the rollout's and the
importance ratio is wrong. Loss curves diverge; runs collapse.

Two responses are emerging. *Bitwise-consistent rollout* — published in a
vLLM blog post (2025-11-10) — patches rollout kernels to match Megatron's,
eliminating the mismatch by construction. *FP16 RL fine-tuning*
([arXiv:2510.26788], 2025) argues FP16 numerics are more
implementation-portable than BF16 in this regime and recommends running RL
in FP16 even when the base model trained in BF16. Neither is fully landed as
of mid-2026.

**MoE routing inconsistency.** A subtler issue specific to MoE. Rollout's
router and trainer's router may pick different top-k experts for the same
token if gating logits straddle a threshold differently under different
numerics. The gradient is then computed with a different expert assignment
than the rollout used. Two proposals exist — "Keep Routing" (record rollout's
expert assignments and force trainer to use them) and "Keep Sampling Mask"
(the analog for stochastic top-k) — neither is shipped in any framework as
of May 2026 (hedged: large MoE RL deployments are an active area).

## 8. Reward and verifier serving

The reward signal is itself produced by an inference workload. Three
families exist with different serving profiles.

**Process Reward Models (PRMs).** Score *intermediate steps* of a reasoning
trace; produce per-step rewards. Run *during* generation in some search
designs, *after* completion in RL training. Batch shape is many-tokens-per-
completion. Typically the same scale as the actor or smaller.

**Outcome Reward Models (ORMs).** Score complete completions; one forward
pass per completion. Cheap, easy to batch. The classical InstructGPT RM is
an ORM. ORMs lost ground in 2025 to RLVR and judges for reasoning workloads
but remain dominant for preference-style RLHF on general-instruction data.

**LLM-as-judge.** A separate (often larger) model scores candidates by
prompting. [J1] (arXiv:2505.10320, 2025) trained judges with RL — J1-Qwen-32B
beats o1-mini on judge benchmarks despite being smaller. From a serving
perspective, a J1-style judge is a full reasoning model: long-decode,
prefix-shareable across the candidate prefix, batched aggressively.

**RLVR — verifier pools instead of learned reward.** The DeepSeek-R1 line
abandoned learned RM for math and code reasoning in favor of *rule-based or
deterministic verification*. A code-execution sandbox runs the program
against test cases; a symbolic math checker validates the answer. Reward is
binary (pass / fail) or graded by test-case count. The infrastructure
consequence is that "reward serving" becomes "verifier-pool serving" —
running thousands of concurrent code-execution sandboxes (typically
Firecracker microVMs or K8s pods), with snapshot/restore for sandbox state
and strict isolation for adversarial code. Kimi K2 is reported to run
>10,000 concurrent K8s sandbox instances during RL training (hedged: vendor-
disclosed, methodology not fully public).

The combined picture: production RL stacks ship a *generation cluster*
(rollout), a *training cluster* (trainer), and a *verifier cluster* (reward /
judge / RLVR sandboxes), each scaled and scheduled independently. The same
gen-then-verify topology that
[§60/01-test-time-compute](01-test-time-compute.md) developed for inference
appears inside the RL inner loop, with the twist that the generator's weights
are changing.

## 9. Long and agentic rollouts

Reasoning RL pushed rollout lengths from ~2k tokens (chat-era SFT) to 32k–64k
(DeepSeek-R1 and successors). Agentic RL goes further: a "completion" is a
sequence of (generation, tool-call, observation, …) steps that may span
minutes of wall-clock and require *external state* (a shell, a browser, a
code interpreter) alive for the trajectory's duration.

ProRL-Agent (NVIDIA, arXiv:2603.18815, 2026) introduces *rollout-as-a-service*
for multi-turn agent training: a rollout coordinator manages long-lived
sandbox sessions, snapshots their state at action boundaries, and restarts
them on the same or a different host if the rollout host fails. The sandbox
becomes a piece of inference infrastructure with its own lifecycle.

Two implications for the rollout engine. First, the engine sees *very*
heavy-tailed sequence lengths — an agent trajectory may be 100 tokens or
50,000 tokens depending on whether the task converges. The async-with-depth-
bounding regime is essentially mandatory; synchronous batching wastes the
entire batch on the longest trajectory. Second, the engine must support
pause / snapshot / resume of generation state cleanly enough that the rollout
can be re-entered after an external tool call without recomputing the prefix.
Modern paged-KV engines handle this naturally because KV blocks are already
content-addressed and host-pageable
([see §30-kv-cache](../30-kv-cache/)).

## 10. Speculative decoding during rollout

Speculative decoding ([see §10/05-speculative-decoding](../10-engine-core/05-speculative-decoding.md))
was historically *avoided* in rollout engines for two reasons. The drafter is
typically distilled from a fixed teacher; under weight churn it drifts from
the actor and acceptance $\alpha$ collapses. The training and rollout
codepaths must produce identical samples up to the rejection rule; the
introduced complexity multiplied the surface for numerical and reproducibility
bugs. Mainstream RL frameworks shipped without spec-dec until very recently.

Two 2025–2026 results reverse this position.

**NeMo-RL v0.6** (May 2026) ships speculative decoding for the rollout path
with three drafter families: MTP heads (when the actor has them), external
drafter models, and EAGLE-3 ([Eagle-3], arXiv:2503.01840). Reported: 1.8×
rollout-step speedup at 8B; *projected* 2.5× end-to-end at 235B (the 235B
number is projected, hence hedged). The key engineering piece is *drafter
co-update*: when the actor's weights update, the drafter updates synchronously
— either by re-distillation (rare) or by sharing layers (MTP heads, free;
EAGLE-3, requires a small adaptation step).

**SPEC-RL** (arXiv:2509.23232, 2025) takes a different angle: rather than
maintaining a separate drafter, it reuses *prior-epoch trajectory prefixes*
as the drafter. The model has already produced these tokens once; if the
policy hasn't moved much, they are likely re-accepted. Reported 2–3× rollout
speedup. SPEC-RL is essentially Lookahead-Decoding applied across rollout
epochs, and like Lookahead it requires no drafter training.

The infrastructure shift these represent is meaningful: spec-dec moves from
"only for serving" to "valid for both serving and training-time generation."
As of mid-2026 it is shipping in NeMo-RL only; veRL, OpenRLHF, AReaL, slime,
and SkyRL do not yet integrate it. The trajectory is clearly toward inclusion.

## 11. Framework comparison

The active framework landscape as of mid-2026:

| Framework | Org | Trainer | Rollout | Weight-sync | Async? | Notable |
|-----------|-----|---------|---------|-------------|--------|---------|
| **veRL** | ByteDance | FSDP / Megatron-Core | vLLM, SGLang, TRT-LLM | NCCL + bucketing (~20ms) | Sync, one-step-off, fully-async | 3D HybridEngine resharding; DAPO reference; largest codebase. |
| **OpenRLHF** | Independent | DeepSpeed ZeRO | vLLM | NCCL broadcast / CUDA-IPC | Sync + partial async | Simplest to adopt; PPO / GRPO / DAPO / REINFORCE++. |
| **TRL** | HuggingFace | Accelerate / DeepSpeed | vLLM V1 | HTTP / in-process | Sync (async in dev) | Reference / educational; cleanest GRPO source. |
| **NeMo-RL** | NVIDIA | DTensor → Megatron-Core | vLLM, TRT-LLM | NCCL / CUDA-IPC | Sync + async | 6D parallelism; spec-dec rollouts in v0.6 (May 2026). |
| **AReaL** | Ant Group + Tsinghua | FSDP2 / Megatron | SGLang | NCCL / Awex | Fully async | Decoupled-PPO; 2.77× vs sync on long-CoT. |
| **slime** | THUDM | Megatron-LM | SGLang sgl-router | NCCL bucketing | Sync + async | SGLang-native; powers GLM family. |
| **SkyRL** | NovaSky + Anyscale | FSDP / Megatron | vLLM, SGLang | NCCL broadcast | Both | Modular; Mini-SWE-Agent recipes. |

Patterns: the trainer side has consolidated onto FSDP/FSDP2 (smaller scales)
and Megatron-Core (frontier scales); DeepSpeed ZeRO remains in active use
only in OpenRLHF. The rollout side has consolidated onto vLLM and SGLang;
TensorRT-LLM is supported in veRL and NeMo-RL but is a minority choice. The
(a)synchronous distinction is the live frontier: veRL added one-step-off and
fully-async modes through 2025–2026, AReaL ships fully-async as the default,
and synchronous-only frameworks (OpenRLHF sync mode, TRL) are increasingly
used for smaller-scale work where async's complexity is not justified.

## 12. Frontier-lab snapshots

Public information about frontier-lab RL stacks is fragmented; the following
synthesizes disclosed details, hedged where the source is partial.

**DeepSeek-R1 / V3.2.** DeepSeek-R1 ([DeepSeek-R1], arXiv:2501.12948,
*Nature* 2025) is the most documented. Training: GRPO at lr=3e-6, KL
coefficient $\beta = 0.001$, 16 samples per prompt at temperature 1, max
generation 32k extended to 64k mid-run, batch 512, 10,400 steps, 85,000
prompts spanning 1,800 environments, rewards almost entirely rule-based
(RLVR) with limited rubric scoring. Post-training compute is disclosed as
more than 10% of pretraining compute — a ratio that would have been
unthinkable in 2023.

**Llama 4.** Meta has disclosed a three-stage post-training recipe:
"SFT-light", "online RL" (the bulk of post-training compute), and
"DPO-light". The online-RL phase is described as running on a "fully
asynchronous online RL training framework" — the framework is not publicly
named, but the description matches the AReaL / PipelineRL design pattern.

**Kimi K2.** Moonshot AI's Kimi K2 reportedly uses RLVR plus self-critique
rubric scoring. Infrastructure includes a verifier pool of >10,000 concurrent
K8s sandbox instances and uses Mooncake for weight sync between training and
rollout pools. Both numbers are vendor-disclosed and not independently
verified.

**Anthropic.** Constitutional AI and RLAIF are documented at the algorithmic
level ([CAI], arXiv:2212.08073, 2022); the infrastructure is not public.

**OpenAI o1 / o3.** The most opaque. The only public infrastructure-relevant
disclosure is that performance scales with *train-time RL compute* — i.e.,
the RL post-training phase is now its own scaling axis comparable to
pretraining compute.

The pattern across frontier labs is consistent at the level of recipe
(GRPO / DAPO / RLVR with rule-based or judge rewards, async at scale, very
long generations, verifier pools for code and math) and divergent at the
level of infrastructure (private trainers, private rollout coordinators,
private weight-sync paths). The OSS frameworks in §11 are the public analog
of those private stacks, and the convergence between them is substantial.

## Current production state

As of mid-2026, RL post-training is the dominant capability differentiator
for frontier reasoning models, and the underlying infrastructure has
crystallized into a recognizable shape. Every major OSS framework reuses a
production serving engine (vLLM or SGLang most often) as its rollout backend.
Every major framework has converged on GRPO or DAPO as the default objective
and on RLVR with rule-based or judge rewards as the default reward source for
reasoning workloads — the InstructGPT-style learned RM remains in use for
preference-style RLHF on general data but has been displaced for math and
code reasoning. Weight synchronization spans from CUDA-IPC handoff (colocated)
through NCCL broadcast with bucketing (most common) to RDMA P2P through
Mooncake or Awex (largest scales), with reported sync times of seconds at
trillion-parameter scale.

The active frontier is the synchronous-to-asynchronous shift. veRL's
HybridEngine remains the most-deployed colocated-synchronous design and is
likely to remain the default for small and mid-scale RL training where its
operational simplicity outweighs the wall-clock penalty. At frontier scale —
DeepSeek, Llama, Kimi K2 — asynchronous designs are taking over, with AReaL,
ROLL Flash, and PipelineRL serving as the public reference points and Llama
4's "fully asynchronous online RL training framework" being the most-cited
proprietary instance. The stability problems async surfaced — particularly
the BF16 numerical mismatch between rollout and trainer — are active
engineering problems with in-progress fixes (bitwise-consistent kernels,
FP16 RL fine-tuning); the MoE routing inconsistency variant is identified
but not yet broadly addressed.

Three trajectories are visible. Speculative decoding is moving into the
rollout path (NeMo-RL v0.6, SPEC-RL), reversing a years-long "too brittle for
training-time use" position; this likely generalizes within a release cycle
or two. Verifier-pool serving is becoming its own operational discipline —
sandbox lifecycle management, snapshot / restore, cross-trajectory state —
analogous to the rise of KV-cache management within serving. And the
rollout engine is increasingly diverging from the user-facing serving engine
they share code with, in the form of RL-specific APIs
(`update_weights_from_*`, `sleep` / `wake_up`, `collective_rpc`) that reflect
invariants with no analog in user-facing serving. Both directions in the
same codebase — a serving engine that is also a rollout engine — is the
durable shape; the specialization happens through configuration and API
surface, not through fork.
