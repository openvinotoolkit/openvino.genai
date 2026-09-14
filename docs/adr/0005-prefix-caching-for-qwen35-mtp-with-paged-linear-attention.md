# Prefix Caching for Qwen3.5 MTP with Paged Linear Attention

**Status:** Accepted
**Date:** 2026-08-29
**Last updated:** 2026-09-14
**Target:** Continuous Batching speculative decoding, Paged Attention backend,
Qwen3.5 MTP with hybrid KV and linear-attention state

**Implementation plan:** [Incremental Linear-Attention Live-Tail and Commit Refactor](../linear-attention-live-tail-commit-refactor-plan.md)

**Current implementation:** Uncommitted prompt-only paired predecessor replay passes
local Qwen3.5 VLM/media parity with observed restores. See the amendment below and
the [verification handoff](../exact-endpoint-continuation-progress.md). A measured
main-only versus paired-restore comparison remains open.

**Historical M0 decision:** Defer restore-policy selection until the common foundation provides
real prepared per-sequence restores and the minimum endpoint sidecar. The accepted M0
run observed zero restores and found generated draft state path-dependent; M0b must
observe real restores before selecting coherent-pair, main-only, or no MTP enablement.

## Implementation Progress

| Phase | Step | Status | Result |
|---|---|---|---|
| P4-P5 | Scoped LFM prefix-verification foundation | Verified locally, uncommitted, 2026-09-14 | Prepared coherent predecessor restore, private verifier KV rows, prefix LA promotion, protected next-window headroom and independent-draft restore/recompute are integrated. Static greedy single-sequence token-input prompt lookup and independent drafting pass with observed restores, both split modes, cancellation/reuse and a hard LA ceiling. 755 selected C++ tests, 18 isolated cache tests and 17 LFM Python cases passed. Exact-live-boundary optional checkpoint sets remain private when publication would consume reusable headroom. Earlier guarded-phase entries below are historical. MTP policy, adapter and Qwen3.5 VLM/media acceptance remain pending. |
| P0 | Base step failure boundary and output visibility | Accepted | Worker joining precedes cleanup; active and awaiting handles receive the original failure and the pipeline rejects reuse. Owned accepted-output snapshots are delivered after coherent base-step updates, excluding prompt-lookup candidates. Newly terminal sequences participate in LA promotion before free. See the [P0 verification report](../p0-failure-visibility-report.md) for test evidence and scope limits. |
| P1-P3 | Live handle, scratch lease and coordinated accepted-state commit | Exit gates passed locally, 2026-09-11 | Prepared KV releases and LA promotions precede counter mutation for the selected static greedy path; sampler deferral is opt-in. 749 existing C++ tests, six isolated host-allocation-failure tests and five local LFM hybrid Python cases pass. The failure sweeps cover heap-owned scratch leases, two-sequence KV/LA preparation and actual pipeline commit; prepared apply and cleanup allocate nothing. Prefix verification remains guarded. No MTP adapter or Qwen3.5 VLM/media acceptance is claimed. |
| M0 | Restore-policy feasibility and draft-state probe | Accepted | Output parity passed, but no restore executed. Generated draft state was path-dependent, so policy selection and generated-state publication remain deferred. |
| F1 | Orthogonal LA history and step modes | Accepted | `LIVE_ONLY` / `PREFIX_CHECKPOINTS` and `PREFILL` / `DECODE` / `VERIFY` are represented independently. `PREFIX_CHECKPOINTS + VERIFY` remains guarded before scheduler or cache mutation. |
| F1 | Logical LA read-row lookup | Accepted | Logical checkpoint positions map through `block_table_logical_start`. Multi-row prefix tables now select the latest represented read row instead of the front checkpoint; writable-tail ownership is not yet separated. |
| F1 | Latest-row ownership invariant | Accepted | Existing `append_slots` copy-on-write remains responsible for separating forked `LIVE_ONLY` rows. Plan construction verifies the resulting aliased read/write row is private, while shared prefix checkpoints remain observation-only. |
| F1 | Prefix verification guard cleanup | Accepted | Scheduler diagnostics were collapsed to one authoritative `PREFIX_CHECKPOINTS + VERIFY` guard before scheduling or cache mutation; strategy and pipeline guards remain intact. |
| F1 | Scheduler paging-mode hygiene | Accepted | Removed an unreachable classifier assertion and aligned paging-plan declarations without changing mode mapping or guard behavior. Post-edit validation passed 48 LA tests and 106 broader scheduler/cache tests. |
| F1 | Prefix reference COW acquisition and rollback | Accepted | Shared incomplete checkpoints are copied into unpublished rows on every prefix scheduling path. Acquisition uses fresh or evicted overwriteable rows, capacity accounts for growth plus COW, and group staging rolls back atomically on failure without changing non-prefix beam accounting. |
| F1 | COW row republication | Accepted | The base pipeline post-sampling hook republishes completed KV+LA COW boundaries using each cache's own granularity. Prefix-off managers are skipped, speculative child KV publication is disabled, and incomplete COW rows remain unpublished. |
| F1 | Per-granularity hash memoization | Accepted | KV and LA prefix hashes are memoized independently by block granularity, and token rollback truncates each Sequence-owned memo chain to the retained identity length. |
| F1 | BlockManager prefix-registry rollback | Accepted | Partial divergent rollback unregisters an invalid retained-row identity, preserves or transfers physical duplicate ownership, and unregisters on overwrite only when the evicted physical row owns the registration. Tests cover zero and exact retained lengths preserving warm completed rows, multilayer state, nonzero logical offsets, duplicate rollback orders, and release/restore/eviction: 70 cache-selection tests including 8 rollback tests, 72 scheduler tests, and 4 focused publication checks passed in the freshly built `tests_continuous_batching` binary; 2 publication checks overlap the scheduler set, for 144 distinct tests. An independent supervisor check passed 12/12 rollback and publication checks, overlapping these results. |

Historical F1 boundary, before P1-P3: existing live-only copy-on-write is enforced but an explicit
live-state handle is not yet represented independently from checkpoint chains. Prefix reference COW
is reference-safe and capacity-bounded, and accepted completed KV+LA COW rows return to
prefix reuse. Scoped completed KV+LA COW republication and tested rollback ownership
behavior are accepted. Transactional scratch leases, general speculative
allocation/publication separation, prepared per-sequence restore/apply, bounded prefix-LA
capacity, and the MTP adapter remain unimplemented or guarded. Alternative-owner
discovery scans active tables under the cache mutex; performance is unmeasured.

## Decision Summary

### Local Paired-Replay Implementation (2026-09-14)

The implemented adapter negotiates a common local processed length `D`, bounded
below the end of both prompts. Main state covers original `[0,D)` and draft state
covers shifted `[1,D+1)`. Main replay from `D` exports the hidden suffix needed by
the next draft input at original `D+1`. Neither child samples directly from a cache
hit; no endpoint sidecar is implemented. This differs from the exact semantic
pair `P`/`P-1` proposal below and uses the permitted predecessor-replay alternative.

Draft identity at `D` delegates to the parent prefix identity at `D+1`, including
the first original token. Main identity combines complete aligned prompt IDs with
the existing bounded embedding samples. Allocation is private; completed prompt
boundaries alone become hash-visible in both roles. Generated-state publication
remains disabled. Both child configurations must agree on prefix caching.

Admission and scheduling use the same strategy mutex. Restored references remain
owned by each child during negotiation. Failed admission removes that request from
both awaiting queues and frees its active cache rows; other request IDs remain.
This is rollback-based paired admission, not cross-child prepared atomic apply.

All 16 local MTP VLM cases pass with actual draft execution and observed warm paired
restores, including built-in images, video and video metadata. Four-candidate warm
cases cover 80-token decoding, cancellation/reuse and prompt extension. Prompt
lookup's 16-case media matrix passes separately. No speedup or optimal-policy claim
is made. Arbitrary caller-supplied position-ID rejection and a complete paired
admission allocation-failure sweep remain verification/design gaps; the current
aligned-token-ID guard alone does not establish built-in-input provenance.

### Milestone Boundary Update (2026-09-11)

The user permits predecessor-checkpoint reuse for the three-row pressure regression.
The updated test proves safe capacity deferral and preserved ownership, not exact-P
restoration. Endpoint-logits caching and zero-forward full-hit sampling are separate
feature-branch work, not P1-P3 requirements. This supersedes the conflicting portions
of the 2026-09-09 amendment below; restoring state and counters coherently still applies.
Final MTP acceptance requires Qwen3.5 through the VLM pipeline with built-in media
support. The LFM foundation tests do not establish that later integration gate.

### Exact-Endpoint Restore Amendment (2026-09-09)

The current restore milestone must support direct exact-endpoint continuation
without predecessor recomputation. Earlier predecessor/replay alternatives in
this record are superseded for this work. State after `[0, P)`, processed-token
counters, and restored KV/LA state must agree on `P`; replay into state already
at `P` is forbidden. Full-prompt generation needs retained, sampling-independent
endpoint output, with coherent publication and cache-identity lifetime.

The original three-row logical-offset rollback test remains an acceptance gate,
with scheduling adjusted only to preserve its absolute target endpoint under
the corrected counter semantics. Pipeline output parity and zero replay must
also be verified. This is a requirement, not an implementation acceptance or
permission to enable guarded strategies. See the implementation plan's
exact-endpoint continuation decision for the detailed gates.

Deliver prefix caching in two layers:

1. A strategy-neutral hybrid/LA speculative-verification foundation in Continuous
  Batching, the scheduler, and cache management.
2. An MTP adapter in `MtpDecodingImpl` for shifted main/draft alignment, paired
  admission, hidden-state handoff, and draft-state publication.

The common foundation separates three main-model linear-attention (LA) state lifetimes:

1. Persistent, hash-visible prefix checkpoints.
2. One exact live-state handle per running sequence. It may reference an immutable
  checkpoint until an exclusively writable destination is required.
3. Per-step speculative scratch at token precision.

The MTP adapter supports Qwen3.5 text, image, and video prompts prepared by the built-in
embedder. Do not commit yet to restoring its draft prefix. M0b compares, after real
restore and endpoint-sidecar prerequisites exist:

- **Coherent-pair restore:** main at semantic boundary `P`, draft at shifted local
  boundary `P - 1`.
- **Main-only restore:** main at `P`, draft restarted with a fresh private cache at
  semantic boundary `P - 1` under an explicit cold-context position contract.

Both preserve target-model output correctness. Measurements with observed restores of coherent hit rate,
cold-draft acceptance recovery, TTFT, and total latency select one policy before MTP
pair admission is refactored.

MTP delivery remains split by publication risk:

- **M1:** reuse prompt state, then keep all generated state private and
  hash-invisible. This accelerates repeated prompts, not chat continuation through
  generated history.
- **M2:** publish canonical accepted generated state for both roles. This is
  conditional on proving or constructing canonical draft KV state.

## Scope

The common foundation supports a hybrid/LA main model verifying a static candidate
window with one return sequence. It owns immutable shared checkpoints, token-precise
rollback, pre-inference scratch reservation, accepted-state commit, and existing
non-prefix behavior. It contains no MTP shift, child-pair, or hidden-state assumptions.

The first product adapter supports greedy Qwen3.5 MTP with text, image, and video
prompts prepared by the built-in input embedder. Prompt lookup and an ordinary
independent draft model are validation targets for the common foundation, not MTP
dependencies. Eagle3 and DFlash require separate strategy adapters.

The following remain unsupported and guarded by the common foundation: dynamic
candidate counts, tree or beam search, multiple return sequences, cross-request scratch
sharing, and forking while a speculative lease is active. The MTP adapter additionally
guards independently supplied embedding/position-ID pairs and an LA-bearing draft.
MTP Milestone M1 excludes publication of generated continuations and per-request
switching between restore policies.

## Current Constraints

- **Conflicting LA layouts.** Prefix caching uses a variable checkpoint chain with
  `block_table_logical_start`; speculative verification assumes one committed row and
  promotes a temporary row into `block_table[0]`. Combining them would replace or free
  shared state.
- **Two independently admitted roles.** `MtpDecodingImpl` owns separate main and draft
  pipelines. The main owns hybrid KV+LA state and exports hidden states. The shipped
  one-layer draft owns KV only, consumes `main_embeddings[1:]`, and requires imported
  hidden-state length to match scheduled draft length.
- **Shifted restore geometry.** Main state covering original positions `[0, P)` aligns
  with draft state covering `[1, P)`, whose local length is `P - 1`. Independent hits
  can therefore represent different semantic boundaries.
- **Incomplete cache identity lifecycle.** Both roles are embedding-backed, so current
  hashing uses reduced embedding samples and ignores populated `prompt_ids`.
  Sequence hash memoization is now scoped by block granularity and truncated on token
  rollback. BlockManager can still retain prefix-registry entries for identities that
  a sequence rolls back and subsequently rewrites. The existing bounded embedding
  reduction is intentional and remains unchanged.
- **Allocation publishes too early.** KV allocation currently registers hashes before
  speculative acceptance is known. MTP M1 needs a prompt-only publication cutoff.
- **No prefix-mode LA admission bound.** Prefix mode sets `max_total_la_blocks` to zero,
  making the existing fixed-pool floor and growth paths inert.
- **Generated draft state may be path-dependent.** Full acceptance and rejection repair
  can pair the same token prefix with different hidden-state histories.

## Required Contracts

### Common foundation

1. **Ownership:** persistent checkpoints are immutable after publication. A live-state
  handle records the exact endpoint after `[0,P)` and may reference an immutable
  checkpoint or an exclusively writable private tail. Scratch is transaction-local and
  never part of the checkpoint chain before commit.
2. **Commit:** a window reserves `N + 1` writable rows before scheduling and retains
  the row selected by an explicit per-sequence acceptance result. Rejected rows never
  enter a prefix registry. The same commit may retain all eligible crossed boundaries.
3. **Identity lifecycle:** hash memoization is scoped by block granularity and truncated
  after token rollback. Strategy adapters supply the semantic identity material.
4. **Publication:** allocation and publication are separate. Only accepted canonical
  state may be published; rejected and scratch state remains hash-invisible.
5. **Boundary publication:** accepted-state commit may publish a policy-selected,
  no-gap set of crossed LA boundaries. Accepted live state is mandatory; optional
  historical publication may be skipped under pressure. MTP M1 publishes prompt state
  only, including when generated state reaches a normal interval.
6. **Transition and failure:** `CacheOrchestrator` prepares every expected-to-fail KV,
  LA, reference, metadata, and counter transition per sequence. Apply allocates nothing
  and has no expected failure. Unexpected post-sampling/precommit failure terminates an
  affected request only where P0 proves isolation; there is no transparent retry or
  complete sampler snapshot. Rewind restores an exact endpoint or a predecessor plus
  recomputation before processed-token counts change.
7. **Capacity:** unique physical rows reachable through checkpoints, pins, live-state
  handles, and scratch fit a hard prefix-LA ceiling, while reserved budget is accounted
  separately. `max_num_seqs` is not used as a live-sequence bound.

### MTP adapter

1. **Alignment:** main cache state, draft cache state, position IDs, processed-token
  counts, and hidden-state handoff describe one semantic endpoint before either child
  runs. A cold draft is not represented as restored state.
2. **Identity:** text positions use complete token IDs and embedding-backed positions
  keep the existing bounded samples. Draft endpoint `D` is keyed from the parent main
  prefix identity at `P = D + 1`, not from shifted embeddings alone. M-RoPE is not
  hashed independently on the built-in Qwen3.5 embedder path.
3. **Boundary hidden state:** restoring main endpoint `P` and draft endpoint `P - 1`
  must provide `h[P - 1]` before the draft consumes `e(x[P])`. The adapter obtains it
  from a retained endpoint sidecar or an explicitly valid predecessor replay; cache
  state alone is not treated as a hidden-state output.
4. **Paired publication:** MTP M1 publishes prompt state only. Generated main and draft
  state is published together only after draft canonicality is established.
5. **Model shape:** prefix-enabled MTP rejects a draft graph containing LA state until
   restore negotiation and scratch ownership support its additional checkpoint grid.

## Proposed Design

## A. Common Hybrid/LA Foundation

This layer is implemented in `ContinuousBatchingImpl`, `Scheduler`, `BlockManager`,
`CacheOrchestrator`, and `Sequence`. It must not depend on an MTP shift, a draft model,
or hidden-state pairing.

### A1. Transactional recurrent state

Model LA paging with independent history (`LIVE_ONLY` or `PREFIX_CHECKPOINTS`) and
step (`PREFILL`, `DECODE`, or `VERIFY`) dimensions. This permits
`PREFIX_CHECKPOINTS + VERIFY` without weakening validation of other combinations.

The block manager owns three disjoint structures:

| State | Addressing | Visibility | Lifetime |
|---|---|---|---|
| Checkpoint chain | LA interval | Hash-visible, shareable | Prefix cache |
| Live-state handle | Exact token | Immutable reference or private writable row | Running sequence |
| Scratch | Exact token | Never hash-visible | One verification window |

`m_block_table` remains the checkpoint chain; the live-state handle is not encoded as
`block_table[0]`, because latest-only restore may start at any logical checkpoint offset
and a growing chain may contain many rows. A restored or hash-visible base may be
referenced read-only by several sequences, because each verification writes to a
distinct private destination. Publication makes a row immutable regardless of its
current reference count. Strategy policy decides whether accepted state stays private
or becomes publishable through the common transaction.

Replace pipeline-level temporary-row promotion with a move-only RAII lease owned by the
cache layer. The lease records sequence and base positions, the read block,
`block_table_logical_start`, and all `N + 1` scratch rows. It is reserved before the
sequence enters scheduler output, commits or aborts exactly once, and aborts on
destruction if still active.

The indexing contract is explicit:

```text
read block:       immutable checkpoint or private live tail
write block k:    scratch[k], k in [0, N]
acceptance:       explicit zero or accepted scratch position
new live state:   scratch selected by explicit acceptance result
```

The acceptance result represents zero and nonzero advancement explicitly. Commit
remains after sampling and before fork/free processing, and handle notification is
deferred until commit is coherent.

### A2. Restore, identity lifecycle, and publication

Expose internal prepare and apply operations while keeping public APIs unchanged. The
common layer prepares one sequence across its cache hierarchies at a caller-provided
semantic ceiling; a strategy adapter decides whether several prepared sequences form
one admission unit. Abandoned preparations release their pins and reservations.

Replace append-only `Sequence::m_prefix_hashes` with memoization scoped by hashing
granularity. Every token rollback truncates affected chains to the retained length.
The common layer hashes adapter-provided identity material and does not infer shifted or
hidden-state-dependent identity.

Separate allocation from publication. Accepted-state commit retains the policy-selected
eligible crossed LA boundaries as a no-gap set; rejected rows remain uncached. Logical
placement respects `block_table_logical_start`:

```text
physical_position = logical_checkpoint_position - block_table_logical_start
```

Publication cannot create a logical gap or overwrite another sequence's registered row.
For an existing identity with matching metadata, the registered row is canonical;
retain/reference it and release a replaceable staged duplicate without hot-path tensor
comparison. A final equivalent duplicate may remain the private live row. Remove a
mapping or row only after registry, sequence, pin, and live-state references reach zero.
Metadata mismatch is an error; the hash has only the equality strength of the existing
prefix identity contract.

Cache endpoint `P` means recurrent state after positions `[0, P)`. Restoring that state
must not set the execution boundary to `P - 1` and replay position `P - 1`; doing so
would apply the last token twice to LA state. Full-prompt output recovery must instead
restore a real predecessor endpoint or use adapter-owned endpoint output metadata.

### A3. Capacity, admission, and failure

Prefix mode needs new admission control because its current LA manager is unbounded.
Use the normalized LA allocation as a hard ceiling and account for:

```text
unique resident rows referenced by checkpoints, pins, live handles, or scratch
  <= prefix LA ceiling
```

Reserve scratch headroom from actual admitted verifiers:

```text
scratch_headroom = sum(1 + num_assistant_tokens[i]) for i in admitted_verifiers
```

This resolves ADR-0004's scratch-reservation question with allocator-backed,
borrow-per-step rows plus an admission-time **budget reservation**. It does not revive
own-upfront: no scratch rows are preallocated per sequence or placed in the committed
checkpoint table.

Do not substitute `max_num_seqs`; `dynamic_split_fuse` can make every submitted request
live. At admission, reserve writable execution and scratch headroom first, evict only unreferenced
and unpinned LRU checkpoints, then plan and pin restoration from the remainder. Defer
the adapter-defined admission unit if protected capacity does not fit; fail early only
when one request and its window can never fit.

Scratch consumes reserved headroom and never triggers per-step checkpoint eviction.
Referenced checkpoints, pins, live-state rows, and in-flight scratch are not evictable.
Reserved budget is tracked separately from unique physical residency. Publication uses
remaining capacity on a best-effort all-or-none boundary-set basis and otherwise keeps
accepted state private. The existing borrow-pool floor remains the non-prefix special
case; endpoint sidecar bytes require later accounting.

The lease owns scratch cleanup. Expected pressure defers before forward. An unexpected
post-sampling/precommit failure aborts the lease and terminates the affected request only
where the audited worker/output path proves request isolation; shared corruption or a
backend-wide failure is propagated at its actual scope. There is no transparent retry.
Rewind restores an exact saved endpoint or predecessor plus recomputation and updates all
cache types and hash memoization before processed-token counts change.

## B. MTP Adapter

This layer is implemented in `MtpDecodingImpl` plus MTP-specific endpoint metadata. It
uses the common primitives but owns all main/draft relationships.

### B1. Select the restore policy

Phase M0 measures both policies before production MTP APIs change:

- **Coherent pair:** record jointly restorable main endpoints `M`, draft local lengths
  `D`, and the selected maximum `P` where `P in M` and `P - 1 in D`.
- **Main only:** restore main at `P`, start a short private cold draft at shifted local
  semantic boundary `P - 1`, and record acceptance recovery and draft catch-up cost.

The cold draft cannot prefill `[0, P - 1)` because skipped main forwards produced none
of its required hidden states. Starting at `P - 1` is valid only when boundary hidden
state `h[P - 1]` is available. This affects draft quality, not target-model correctness.

Use representative prompt lengths and scheduler configurations. Report cache-hit
distribution, TTFT, total latency, acceptance by generation step, and output parity.
Choose coherent-pair restore, main-only restore, or stop if neither improves latency.

Alignment uses semantic endpoints rather than processed-token counters:

```text
main endpoint P:  state after original positions [0, P)
draft endpoint D: state after pairs [h[j], e(x[j + 1])] for j in [0, D)
coherent pair:    D = P - 1
```

The grids may align sparsely. For example, a main LA checkpoint at `P = 1024` requires
draft length `1023`, which is not a regular boundary for draft KV block size 32. Exact
prompt endpoints may still align through retained partial blocks.

### B2. Pair admission, identity, and boundary synthesis

The MTP parent prepares both child requests before either is visible to `step()`. In
coherent-pair mode it asks the common layer to plan both roles at candidate `P`. If
either plan falls lower, repeat convergence before pinning and applying both plans
atomically. If no coherent endpoint can be pinned, both roles prefill. Main-only mode
applies only the main plan and labels the draft explicitly cold. Either policy prevents
draft-first partial admission.

MTP supplies these identities without changing model inputs:

```text
main_key(P)  = hash(main namespace, token IDs [0, P), sampled embeddings [0, P))
draft_key(D) = hash(draft namespace, main_key(D + 1), D)
```

The draft key is derived from the parent main prefix because draft row `j` depends on
`h[j]`, which includes original token `x[0]`; hashing shifted embeddings alone is
insufficient. Preserve `_reduce_embedding()` for embedding material: sample at most 10
values per vector with stride 50. This retains current compute cost and collision
characteristics. Store identity through existing `SequenceGroup` data rather than a new
public `PromptCacheIdentity`. Amend ADR-0002 before approval because its current
shifted-prefix wording does not capture the parent hidden-state dependency.

Do not add Qwen3.5 M-RoPE position IDs to the key. On the built-in embedder path,
positions and `rope_delta` are derived with embeddings from the same prompt and media
geometry and cannot vary independently. Cumulative matching sees changed media
embeddings before later equal text embeddings. Guard independently supplied embedding
and position-ID pairs until position identity is represented. Assert that the shipped
draft graph is LA-free.

Every MTP-restorable main endpoint `P` also needs `h[P - 1]`. Retain it as an
MTP-owned, reference-counted sidecar keyed by `main_key(P)`, or restore a genuine
KV+LA predecessor endpoint `P - 1` and replay `x[P - 1]` exactly once. For an exact
prompt hit, the sidecar also retains target logits or supports applying the target LM
head to `h[P - 1]`. Never combine LA state at `P` with a processed boundary at `P - 1`.

When extending with available `x[P]`, eagerly evaluate the draft pair
`[h[P - 1], e(x[P])]` to complete the next draft row before candidate generation. At
an exact prompt endpoint, recover target next-token output from the sidecar, produce
`x[P]`, and then use the same operation. Sidecars are pinned and evicted with their main
endpoint and count toward prefix-cache capacity.

Draft entries carry their parent main identity and are restored only as part of an
eligible pair. Mandatory cross-pool cascade eviction is unnecessary: an orphaned draft
entry remains unusable until the same main identity exists and may expire through the
draft LRU.

### B3. Gate MTP generated-state publication

Prompt-prefill draft KV is canonical because each position uses the corresponding main
hidden state. Generated draft KV may not be: normal drafting and rejection repair can
reach the same token prefix through different hidden-state histories.

Phase M0 reaches the same semantic position through full acceptance and repeated
first-token rejection, then compares draft KV and next-step logits. Record bitwise
results for identical device and execution shape, with normal output tolerances as the
functional criterion.

MTP M2 proceeds only if draft state is path-independent or a bounded canonical
construction is selected, such as recomputation from target hidden states. Otherwise
M1 remains the prompt-only feature. Publishing main generated state alone is
insufficient for later MTP restore because it cannot recreate draft hidden-state
history.

A path-sensitive identity is valid only if a later request can reproduce that exact
path. When M2 is enabled, the MTP adapter atomically publishes main KV+LA, canonical
draft KV, and required endpoint sidecars through the common publication transaction.

## C. Applicability Beyond MTP

| Strategy | Placement after the common foundation |
|---|---|
| Prompt lookup | Validate directly; no draft cache or MTP adapter is involved. |
| Independent draft model | Validate independent same-prefix restore and catch-up; no shifted identity or main hidden-state sidecar is required. |
| Eagle3 | Reuse the common verifier foundation, but keep pair alignment and hidden-state rules in an Eagle3 adapter. |
| DFlash | Keep guarded until a DFlash adapter defines target-hidden-delta restore. |
| LA-bearing draft | Keep guarded until that strategy supports another recurrent checkpoint grid and scratch transaction. |

## Implementation Ownership

| Layer | Owner | Change |
|---|---|---|
| Common | Continuous Batching pipeline | Supply decisions, apply prepared transitions after sampling and before fork/free, then publish output. |
| Common | `BlockManager` | Own physical references for checkpoints, live-state rows, pins, and scratch; separate allocation from publication; publish by logical offset. |
| Common | `CacheOrchestrator` | Prepare and apply per-sequence KV+LA transitions; expose leases; coordinate rewind and protected/evictable capacity. |
| Common | Scheduler | Model history and step modes independently; reserve complete windows; defer preemption during a lease; queue adapter-defined admission units. |
| Common | `SequenceGroup` / `Sequence` | Hash adapter-provided identity and maintain rollback-safe chains per granularity. |
| MTP | `MtpDecodingImpl` | Select restore policy, derive parent-bound draft identities, coordinate `P` / `P - 1` admission, and assert an LA-free draft. |
| MTP | Endpoint metadata | Retain, pin, account, and evict boundary hidden-state sidecars with their main-prefix endpoints. |
| MTP | Publication policy | Keep generated state private in M1; publish main, draft, and sidecar state atomically in M2. |

## Failure Contract

### Common foundation

| Event | Required behavior |
|---|---|
| Expected capacity shortage | Defer before forward after prepare releases reservations and pins. |
| Optional publication shortage | Keep accepted live state and skip the complete optional checkpoint set. |
| Unexpected post-sampling/precommit request-local failure | Abort the lease, join workers, terminate and unblock the affected request, and expose no output; no transparent retry. |
| Shared corruption or backend-wide failure | Propagate at the audited failure scope; do not assume request isolation. |
| Cancellation | At a post-inference safe point, retain in-flight ownership until lease cleanup, then release live and checkpoint references. |
| Preemption or recompute | Defer while a lease is active; otherwise restore an exact endpoint or predecessor plus recomputation before changing processed-token counts. |
| Fork | Reject while a lease is active; independent child private writable tails remain future work. |
| Output delivery failure after commit | Do not roll back committed inference state. |

### MTP adapter

| Event | Required behavior |
|---|---|
| No coherent pair | In coherent-pair mode, restore neither child and prefill both. |
| Pair pin or admission failure | Release both plans and sidecar pins before fallback or deferral; never expose partial admission. |
| Missing boundary sidecar | Fall back to a valid predecessor replay or reject the hit; never synthesize a cold draft without `h[P - 1]`. |
| Independent position IDs or LA-bearing draft | Reject prefix-enabled MTP before either child is enqueued. |

## Validation Plan

Common tests use synthetic hybrid models where practical. MTP tests use a tiny Qwen3.5
MTP model, followed by the real model for performance. Tests must assert restoration
and registry state directly; successful generation alone is insufficient.

| Layer | Required coverage |
|---|---|
| Common hash lifecycle | Adapter-provided identities; rollback across a block boundary followed by a divergent suffix; fresh-sequence equality for KV and LA granularities; independent memoization granularities. |
| Common LA lease | Shared complete and incomplete restore; copy-on-write; first, partial, and full acceptance; abort and injected failure; nonzero logical start; crossed-boundary publication; no rejected-row or scratch leakage. |
| Common scheduler/orchestrator | Atomic `PREFIX_CHECKPOINTS + VERIFY`; partial-window deferral; mixed paging modes; cancellation before commit; recompute between windows; preemption deferral; protected scratch under pressure. |
| Common applicability | Prompt lookup and ordinary independent-draft verification with a hybrid/LA main model; existing non-prefix behavior unchanged. |
| MTP identity | Parent-derived draft keys distinguish equal shifted embeddings with different original prefixes; bounded embedding samples remain unchanged; image/video prompts work without M-RoPE in the key. |
| MTP admission | Exact and extending prompt endpoints without LA double application; sidecar hit/miss/eviction; hidden-state/scheduled-length equality; no partial child admission; concurrent pair pins; cancellation while another request retains a shared endpoint; LA-bearing draft rejection. |
| MTP policy experiment | Compare coherent-pair and cold-draft runs at the same main endpoint: hit distribution, first-step and recovery acceptance, TTFT, draft catch-up, total latency, and target output parity. |
| MTP end to end | First/repeated prompt, shared-prefix extension, chunked prefill, concurrency, first/partial/full acceptance, parity with target-only PA and prefix-off MTP, observed restore count, and stable scratch/sidecar usage. |
| Performance | At least 20% lower median repeated-prompt TTFT when restoring one main LA interval; no more than 3% regression in prefix-off speculative throughput or inter-token latency, with variance reported. |

## Delivery Plan

The dependency-ordered, review-sliced delivery plan is maintained in
[Incremental Linear-Attention Live-Tail and Commit Refactor](../linear-attention-live-tail-commit-refactor-plan.md).
It starts with the worker-future, termination, and output-visibility audit; introduces
the common live-state, scratch, acceptance, capacity, and rewind primitives in guarded
slices; validates prompt lookup and an independent draft; and only then runs a
restore-capable text MTP M0b before policy selection. M2 remains a separate optional
decision.

Until F1 and the relevant adapter pass, keep actionable prefix-plus-verification guards
and never silently disable `enable_prefix_caching`. If M2 is abandoned, M1 remains a
complete repeated-prompt feature; generated chat history continues to prefill.

## Risks and Rejected Shortcuts

Common risks are state/processed-token off-by-one errors, nonzero
`block_table_logical_start`, accidental publication of rejected state, and LA-capacity
thrashing. MTP-specific risks are sparse coherent endpoints, collisions from the
intentionally reduced embedding identity, missing or stale boundary sidecars, partial
child admission, and path-dependent draft KV.

Historical commit `139a2e7a7` is design evidence for reserve/commit/rollback and boundary
publication, but it predates MTP, assumes zero-based checkpoint tables, and lacks paired
admission and end-to-end coverage. Do not cherry-pick it.

Do not restore the deleted own-upfront policy: it places scratch in the committed table
and cannot represent both interval checkpoints and token-precision candidates. Do not
remove only the current guards: promotion would replace `block_table[0]`, ignore logical
offsets, and may free shared state.

## Ship Gates

| Gate | Requirement |
|---|---|
| F1 foundation | Common identity lifecycle, immutability, lease, failure, preemption, publication, and capacity tests pass with strategy guards retained. |
| M1 correctness | Qwen3.5 text and built-in VLM MTP accept prefix caching; the selected policy and boundary sidecar are observed; output matches target-only PA and prefix-off MTP. |
| M1 value | Repeated-prompt TTFT improves by at least 20% for a restored main LA interval; prefix-off throughput and inter-token latency regress by no more than 3%. |
| F2 reuse | Prompt lookup and ordinary independent-draft suites pass with a hybrid/LA main before their guards are removed. |
| M2 | Extended chat restores canonical main, draft, and sidecar state across multiple LA intervals and acceptance histories while retaining F1/M1 guarantees. |

M1 claims repeated-prompt and shared-endpoint acceleration only. It does not claim reuse
through generated chat history.

## References

- `docs/adr/0001-read-only-restored-linear-attention-checkpoints.md`
- `docs/adr/0002-align-mtp-prefix-restore-with-shifted-draft-prefix.md`
- `docs/adr/0004-linear-attention-speculative-scratch-ownership.md`
- `docs/implementation-report-linear-attention-paging-refactor.md`
- `docs/linear-attention-live-block-removal-implementation-report.md`
- `docs/linear-attention-live-tail-commit-refactor-plan.md`
- `temp/qwen35_mtp/M0-restore-policy-report.md`
