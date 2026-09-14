# Incremental Linear-Attention Live-Tail and Commit Refactor Plan

**Status:** P0 accepted; P1-P3 committed; scoped LFM P4-P5 foundation verified locally, uncommitted; Qwen3.5 MTP and VLM/media gates pending
**Date:** 2026-09-07
**Last updated:** 2026-09-14
**Decision record:** [ADR-0005](adr/0005-prefix-caching-for-qwen35-mtp-with-paged-linear-attention.md)

P1-P3 currently includes explicit greedy acceptance, move-only LA scratch leases,
prepared LA promotion, prepared non-prefix KV tail release, deferred notification, and
atomic counter/cache application for the selected one-sequence greedy static verifier.
Sampler counter deferral is explicit opt-in, so KV-only speculative, prompt lookup,
DFlash, MTP, and standalone sampler callers retain their previous finalization behavior.
The final implementation builds in `build/vscode-__unspec__`. Independent direct
GoogleTest execution passed 749 selected tests from 108 suites; five local LFM2.5
Python regressions passed using the rebuilt native library. CTest discovery is
empty by repository convention, not a validation blocker. An integrated real-sampler
failure test verifies missing KV-plan rejection before apply, shared exception
propagation to active/awaiting handles, and full physical-cache cleanup. Six additional
isolated allocation-failure tests now pass: each ordinary C++ heap acquisition on the
tested scratch-lease and combined KV/LA preparation paths is failed in turn until the
first successful attempt. The actual pipeline commit is also swept, checking both
request counters, endpoints, generations, references, capacity and lease ownership.
The tests cover one/two allocator layers, two-sequence commits, and accepted depths
1/2/4 in the prepared apply tests. Prepared apply and lease cleanup run with allocations
disabled. This closes the remaining P1-P3 host-allocation gate for the selected paths;
it does not claim device-allocation or general process-OOM recovery coverage.

Qwen3.5 MTP acceptance must use the VLM pipeline and include built-in media support.
Execution order confirmed September 14: finish LFM common-foundation support first,
including observed prefix restores and strategy-specific guard gates, then check
Qwen3.5 VLM/media. Do not treat existing LFM smoke/parity cases as full enablement.
The five LFM cases are common-foundation tests, not Qwen3.5 VLM/media tests. The user
permits predecessor-checkpoint behavior for the three-row pressure test; that case
now verifies safe capacity deferral and unchanged published ownership, not exact-P
continuation or cached-logit sampling.

## Goal and Boundary

### Current P4 Progress (2026-09-14)

The scoped LFM P4-P5 milestone now passes: ordered prepared hybrid restore,
predecessor full hits, private verifier KV rows, prefix-aware LA promotion and
protected next-window headroom are integrated. Greedy single-sequence token-input
verification is enabled for static windows. Draft alignment metadata uses ordinary
paging, and actual independent-draft rejection restores/recomputes recurrent state.
Explicit prefix-LA ceilings also survive `cache_size` normalization.

Optional exact-live-boundary checkpoint sets stay unpublished when retained
headroom needs a reusable private live row. This preserves continuation under
competition rather than maximizing newly published checkpoints. It does not add
endpoint-logits caching or remove MTP/embedding-input guards.

Final local gates: 755 selected C++ tests, 18 isolated cache tests and 17 LFM Python
cases passed. Real-model tests cover both split modes, candidate counts 1/4, actual
restores, nonzero independent-draft acceptance, extensions and cancellation/reuse
under a 12-row LA ceiling. CSV-backed suites were excluded. See the
[current handoff](exact-endpoint-continuation-progress.md) for scope and evidence.
LFM tests are not MTP tests. The Qwen3.5 adapter/policy experiment and VLM/media
acceptance remain the next milestones; no new commit has been made.

### Previous Publication Slice

Prepared scratch eviction is committed as `74c622c38`. The subsequent uncommitted
slice prepares all represented crossed-boundary publication metadata before setting
any row's published flag. Both production publication loops use the batch API.
Allocation failure preserves the whole new set; canonical duplicates remain private.
This is per-cache metadata atomicity, not cross-cache publication or the optional
physical-row retention policy. Legacy allocation still registers fresh rows early.

Validation: 755 selected C++ tests, ten allocation-failure tests, and five LFM Python
regressions passed using the rebuilt native library. The new tests cover deterministic
multi-candidate LRU and failure at each allocation while publishing two boundaries.
One independent review completed; scoped diagnostics and whitespace checks passed.
CSV-backed model suites remained excluded. Retained next-window headroom, optional
no-gap retention, pinned atomic KV+LA restore/rewind and P5 guard decisions remain.

### Earlier Capacity and Eviction Slices

P1-P3 was committed as `0ce159919`. The first subsequent P4 slice preserves the
explicit `num_linear_attention_blocks` ceiling in production prefix-LA registration.
Dynamic scheduler admission no longer grows KV indefinitely when capped LA capacity
is the limiting resource. Both prompt scheduling modes defer without touching a
protected checkpoint, then resume after its owner is freed. Split-fuse still permits
a partial prefill that fits the remaining LA capacity. Unspecified limits retain
their existing dynamic-growth behavior.

The capacity slice is committed as `f06a67b11`.
Validation: three new focused tests passed, followed by 752 selected C++ tests,
six allocation-failure tests, and five LFM hybrid Python regressions using the rebuilt
native library. CSV-backed model suites were excluded. Reservation accounting, checkpoint-set publication/canonicalization,
prepared pinned restore and rewind remain unimplemented P4 gates. No prefix-verifier
guard was removed; Qwen3.5 VLM/media remains mandatory for later MTP acceptance.

The scratch-admission correction committed as `119c6a40c` distinguishes raw writable capacity
from free-but-cached checkpoint capacity. Admission and both scratch acquisition paths
now agree before any row is taken. The regression covers capped one/two-layer pools,
rejection without mutation, cached restore while scratch is reserved, blocked competing
continuation and successful continuation after release. Validation passed 753 selected
C++ tests, six allocation-failure tests and five real-hybrid Python cases.

The next uncommitted slice adds prepared batch eviction for scratch reservations.
Both acquisition paths prefer fresh rows, then reclaim only unowned LRU checkpoints.
Allocations and row validation complete before any ownership or registry mutation;
eviction also removes content-length metadata for erased canonical registrations.
Admission can now include evictable capacity. Failed preparation preserves cached
lookup and references, while release after successful eviction returns unpublished
rows without resurrecting stale hashes. This guarantee is per reservation, not a
rollback of successful eviction if a later group reservation fails.

Validation passed 754 selected C++ tests, eight allocation-failure tests and five
real-hybrid Python cases with the rebuilt native library. CSV-backed suites were
excluded. One independent review was completed. Multi-candidate LRU and duplicate-owner
eviction need dedicated coverage. Retained next-window headroom, optional no-gap
publication/canonicalization, pinned atomic restore and rewind remain P4 work;
prefix-verifier guards stay in place.

Refactor internal linear-attention (LA) state coordination incrementally so prefix
checkpoints, a sequence's current recurrent state, and speculative state have separate
lifetimes. This is not a pipeline rebuild. Preserve:

- LA kernels and their read/write row layout;
- the sampler's token-selection algorithms and public generation APIs;
- existing hashing, copy-on-write (COW), registry ownership, and rollback work;
- prefix-off behavior and existing non-prefix, multinomial, beam, and other sampler
  consumers until each path is explicitly integrated and verified.

The first enabled scope is one sequence, greedy decoding, and a static speculative
window. Keep guards for unsupported combinations until their own exit gates pass. Do
not add an unused abstraction in anticipation of a later phase: every new primitive
must have an integrated call site in the same review slice.

## Retained Evidence

The accepted rows in ADR-0005 remain the implementation baseline. In particular, retain
per-granularity `Sequence` hashes and rollback truncation, logical latest-row lookup with
nonzero `block_table_logical_start`, reference-safe unpublished COW with reversible
eviction, accepted KV+LA COW republication, and duplicate registry-owner rollback and
overwrite-owner eviction.

The last accepted validation comprised 70 cache-filter tests, 72 scheduler tests, and
4 publication checks, with 2 publication checks overlapping the scheduler set: 144
distinct tests. The 70 cache tests included 8 rollback tests. Alternative-owner lookup
still performs an unmeasured active-table scan under the cache mutex.

M0 is evidence, not a restore-policy decision: the real-model probe observed zero
restores and path-dependent generated draft state. No phase below may call future work
accepted or implemented until its exit gate has passed and ADR-0005 records the result.

## Agreed State Model

### Endpoints and ownership

- LA checkpoint chains contain immutable, cache-visible historical rows.
- A **live-state handle** is the explicit sequence-local endpoint `P` plus the complete
  row set across LA layers representing recurrent state after positions `[0, P)`.
- `P` is absolute within the sequence. It does not depend on LA checkpoint interval or
  `block_table_logical_start`. Generated tokens can exist without having been processed.
- Empty state is represented explicitly; it is not inferred from a missing or zero
  checkpoint index.
- A live-state handle may initially reference an immutable checkpoint. A **private live
  tail** means an exclusively writable row set, not merely a row referenced by one
  sequence. A published row remains immutable even when only one sequence references it.
- A shared immutable read row may serve several sequences while each forward uses a
  distinct private write destination. The kernel's first row reference is read and its
  second is write; no copy of the shared base is required when those references differ.
- Speculative scratch is sequence-owned, hash-invisible, and separate from both the
  checkpoint chain and the live-state handle until commit.

### Component responsibilities

- `BlockManager` owns physical row references, immutable checkpoint registrations,
  private write rows, pins, and scratch rows.
- `CacheOrchestrator` coordinates KV and LA preparation, logical transitions, commit,
  abort, rewind, and capacity reservations.
- The pipeline supplies scheduling and acceptance decisions, invokes coherent apply,
  and publishes output only after commit.
- A move-only RAII scratch lease transitions exactly once from `active` to `committed`
  or `aborted`. Destruction aborts an active lease. Forking and preemption are forbidden
  while a lease is active.

### Commit and publication

- Reserve writable execution and scratch budget before forward.
- Sampling returns an explicit per-sequence acceptance result. Commit does not derive
  acceptance from mutable processed-token or rejection counters.
- Copy-free scratch promotion retains the accepted scratch row as the new live row.
- One commit may retain every eligible crossed checkpoint boundary. The final boundary
  row may simultaneously back the live-state handle and an immutable checkpoint; one
  physical allocation can have several ownership references and is counted once for
  physical residency.
- Policy chooses canonical publication endpoints. MTP M1 publishes prompt state only;
  generated state stays private even when it lands on a normal interval.
- On duplicate publication, the existing canonical registry row wins. Retain/reference
  it and release the staged duplicate where replacement is valid. Do not compare tensors
  in the hot path. A hash is not stronger proof of tensor equality than existing prefix
  identity assumptions. Metadata mismatch is an error. A final equivalent duplicate may
  remain the private live row when replacing it would require a copy or violate commit.

### Prepare/apply transition

Preparation and apply are per sequence across cache types and counters, not batch-wide
atomic. A future MTP adapter may compose a main/draft pair where required.

Prepare performs every operation expected to fail:

- validate the base endpoint and a simple per-sequence generation where serialized
  scheduling alone is insufficient for stale-lease detection;
- validate identities, metadata, placement, and no-gap checkpoint representation;
- acquire references and pins;
- reserve capacity and allocate all mandatory physical rows and transition metadata
  required for forward and apply before forward begins.

Apply performs no expected allocation and has no expected failure. No user may observe
an intermediate KV/LA/counter transition. A new optional checkpoint set is retained
best effort as one set initially: if it cannot fit, skip the optional set rather than
creating holes that a later restore would cross. Post-sampling prepare may allocate
optional publication retention or metadata, but it must not perform the first allocation
of mandatory execution scratch.

### Capacity

The hard prefix-LA ceiling includes the unique physical rows reachable through pins,
live handles, checkpoints, and scratch. Reserved budget is tracked separately from
physical residency so one row with several ownership references is not charged several
times. Admission preserves headroom for the next verification window. Sidecar bytes are
an acknowledged later accounting requirement.

Accepted live state is mandatory. Historical publication is optional. Expected memory
pressure defers the request before forward; optional checkpoint publication may be
skipped after accepted state is secured.

## Agreed Failure and Visibility Contract

- Expected memory pressure is handled during prepare by deferral before forward.
- An unexpected failure after sampling starts and before coherent commit terminates the
  affected request after lease cleanup. There is no transparent retry and no framework
  for snapshotting and restoring the complete sampler state.
- Request-local termination is valid only after the worker-future and cleanup audit
  proves no task can continue against freed request state. Shared-state corruption or a
  backend-wide failure must not be assumed request-isolatable; P0 must record the actual
  failure scope and stop for review if isolation cannot be established.
- `notify_handle`, finish visibility, echo/stop output, and partial streaming for the
  affected sequence are deferred until commit is coherent. Output publication follows
  commit. Failure to deliver an output after commit does not roll inference state back.
- Termination must notify and unblock existing generation handles using supported API
  state. If the existing API cannot express this without inventing a status, report the
  gap at P0 and stop.
- Cancellation after inference is a safe point only while in-flight rows remain owned.
  Preemption defers an active lease.
- Rewind restores an exact saved endpoint or restores a predecessor and recomputes. It
  never relabels recurrent state. Existing LA state preservation is not equivalent to a
  logical rollback guarantee.

## Execution Protocol

### Exact-Endpoint Continuation Decision (2026-09-09)

Direct exact-endpoint continuation is required. Predecessor-and-recompute is not
an implementation option for this restore work; this decision supersedes the
earlier alternatives below. The minimum exact-restore work is brought forward
into the current milestone. It is not yet implemented or accepted.

**Historical exact-resume gates:** The requirements below record the earlier scope.
Endpoint-logits caching and zero-forward sampling were subsequently separated onto
`feature/prefix-cache-endpoint-logits`; predecessor-checkpoint reuse for the three-row
pressure case is now permitted. These are not current P1-P3 acceptance gates.

- A restored state after `[0, P)` has processed endpoint `P`, including full-prompt
  hits. Never set its counter to `P - 1` or replay a token already represented.
- Additional prompt tokens continue from `P` with separate writable state while
  preserving the immutable checkpoint and existing kernel endpoint mapping.
- Immediate generation from a full-prompt hit must use retained endpoint output.
  Raw logits must be captured before sampler mutation and published with the
  corresponding committed cache identity. Their lifetime and invalidation must
  follow that identity. Caching a sampled token is not equivalent.
- Missing endpoint output must not silently trigger predecessor replay. Handling
  unavailable or unsupported exact-resume artifacts must be explicit; no new
  public failure policy or strategy-guard change is approved by this decision.

`PrefixCachingRollbackInvalidatesLatestOnlyEndpointWithLogicalOffset` is an
acceptance criterion: retain its three-row budget, logical offset, and stale and
divergent restore endpoint of 12. Scheduling must express the same absolute end
14, which requires two tokens after an exact restore at 12 rather than three
after the old counter value 11. Do not weaken checkpoint-preservation assertions.
Also require full-prompt restored/cold output parity, no forward before the first
restored sample, immutable raw endpoint output across sampling configurations,
and coherent KV/LA/counter endpoints. Passing the block-manager test alone does
not establish inference correctness. These historical exact-resume gates do not block
the now-verified P1-P3 foundation; the current boundary is recorded at the top of this plan.

Work is delivered in approved milestones: P0, P1-P3, P4-P5, and P6-P7. P8 remains
separately optional. Small edits and focused checks remain internal to each milestone;
routine repairs do not require a new approval. Stop for public-contract changes,
architectural constraint changes, or a genuine blocker. Delegated implementation work uses
GPT-5.6 Sol for hard implementation and design and GPT-5.6 Luna for straightforward
exploration or documentation. No nested agents are used.

After every milestone:

1. Stop delegated work.
2. Perform supervisor code review against the milestone invariants and out-of-scope list.
3. Run an independent focused verification from the matching build tree.
4. Report changed files, review findings, exact test command/counts, remaining risks,
   and whether the exit gate passed.
5. Continue only after explicit approval. Do not mark future rows accepted in advance.

## Dependency-Ordered Phases

### P0: Failure Boundary and Output Audit

**Accepted 2026-09-08:** The base Continuous Batching step joins sampler workers before
failure cleanup, terminates active and awaiting handles with the original exception,
and permanently rejects reuse after unexpected shared-state failure. Request-local
isolation is not claimed. Successfully cleaned requests are released; requests whose
cleanup fails remain owned by the unusable pipeline until destruction.

Accepted output is prepared after sampling, before prompt-lookup candidates can mutate
sequence token vectors, and delivered only after cache publication, terminal-inclusive
LA promotion, fork/free, candidate generation, and embedding updates. Echo uses ranges
captured before processed-token advancement. Precommit failure discards pending output;
postcommit delivery failure never rolls inference state back. Already finalized handles
remain final. User STOP/CANCEL intent remains immediately observable; pipeline-produced
stop output is deferred.

Verification and limits are recorded in [the P0 report](p0-failure-visibility-report.md).
P0 does not establish cross-pipeline main/draft atomicity, recoverable backend failure,
or the future prepared-transition guarantees. Existing strategy guards remain unchanged.

**Ownership/files:** `sampling/sampler.cpp`, `sequence_group.hpp`, base Continuous
Batching `pipeline_impl.cpp/.hpp`, generation stream/handle implementation where the
audit leads, and test files for the same paths. This phase begins read-only and changes
production code only after the audit result is reviewed.

**Slices:**

1. Trace sampler futures from submission through every `future::get()`, including what
   happens when one future throws and whether later futures complete before cleanup.
2. Trace `notify_handle`, streaming, echo, finish/stop visibility, post-sampling cache
   publication/commit, fork/free, and `_free_non_running_requests` for terminal and
   failed sequences.
3. Record a concrete request-local termination/unblock path and classify failures that
   must remain backend-wide. Resolve whether terminal sequences are currently omitted
   from cache cleanup or commit by running-only iteration.
4. Add the minimum failure boundary needed to join all worker futures safely and delay
   sequence visibility until a later coherent commit point. Do not introduce live-state
   or scratch behavior yet.

**Invariant:** No request is freed while a sampler task can access it; no generated,
terminal, echo, or stop output becomes visible before the future commit boundary.

**Focused tests:** injected failure in one sequence with another future in flight;
streaming and non-streaming unblock; greedy finish/stop and echo; terminal-sequence
cleanup; no queued partial output on precommit failure; backend-wide failure propagation.

**Exit gate/review checkpoint:** The supervisor can name the supported termination
status/path, prove all futures are joined, and classify request-local versus shared
failure. Stop if public handles cannot be unblocked without a new public contract.

**Out of scope:** sampler algorithm changes, full sampler snapshots, transparent retry,
LA ownership changes, batch-wide transactional sampling.

### P1: Explicit Live-State Handle

**Ownership/files:** `BlockManager`, `CacheOrchestrator`, scheduler/sequence cache-state
records, base pipeline model-input mapping, and focused BlockManager/orchestrator tests.

**Slices:**

1. Add the endpoint/row-set value type with explicit empty state, move/reference
   semantics, absolute `P`, and per-sequence generation. Integrate it into existing
   lookup without changing execution.
2. Route LA read-row selection through the handle while preserving logical checkpoint
   placement and current guards.
3. Support shared immutable read plus distinct private write references in model input.
   Remove a copy only after tests prove read/write separation.
4. Move physical-reference ownership into `BlockManager`; keep orchestration in
   `CacheOrchestrator`. Do not add a second owner in the pipeline.

**Invariant:** The handle always names exact state after `[0,P)` across every LA layer;
published rows are immutable regardless of reference count; checkpoint offsets do not
alter `P`.

**Focused tests:** explicit empty state; restore at zero/nonzero logical start; multilayer
row sets; shared checkpoint read by two sequences with separate writes; unique published
row remains immutable; generated-but-unprocessed token does not advance `P`; stale
generation rejection.

**Exit gate/review checkpoint:** Existing live-only and prefix lookup paths pass through
the handle with no kernel/layout or public API change. Guards remain enabled.

**Out of scope:** speculative lease, checkpoint-set publication, MTP pairing, guard
removal.

### P2: Scratch Lease and Prepared Transition

**Ownership/files:** `BlockManager` physical allocation/reference APIs,
`CacheOrchestrator` lease and transition APIs, scheduler reservation/preemption rules,
and focused cache/scheduler tests.

**Slices:**

1. Add a move-only active/committed/aborted lease with destructor cleanup and one
   integrated guarded call site.
2. Reserve all writable rows for one static verification window before forward; forbid
   fork and defer preemption while active.
3. Introduce per-sequence prepare data containing validated base endpoint/generation,
   references, metadata, allocation, and capacity reservation.
4. Add allocation-free/no-expected-failure apply and copy-free accepted-row promotion;
   retain all-or-none optional crossed boundaries without publishing them yet.

**Invariant:** Scratch is never hash-visible; prepare owns every expected failure;
apply only swaps prepared ownership/metadata; abort leaves the base endpoint unchanged.

**Focused tests:** move/destruction and exactly-once transition; first/partial/full
acceptance; injected prepare failures at each acquisition; abort after forward; stale
base/generation; multilayer cleanup; fork rejection; preemption deferral; no allocation
or expected failure in apply.

**Exit gate/review checkpoint:** A guarded one-sequence greedy static window can prepare,
apply, or abort with stable reference/capacity counts. No user-visible output is yet
routed through this path.

**Out of scope:** enabling prefix verification, MTP, dynamic windows, tree/beam/multiple
returns, sampler snapshots.

### P3: Explicit Acceptance and Coherent Pipeline Integration

**Ownership/files:** sampler result types, base pipeline step ordering,
`CacheOrchestrator` apply/abort, `SequenceGroup` visibility hooks, and integrated
Continuous Batching publication tests.

**Slices:**

1. Return an explicit per-sequence acceptance result without changing token selection;
   preserve existing result paths for all non-integrated sampler consumers.
2. Integrate acceptance, cache/counter apply, finish state, and delayed notification as
   one coherent request transition. Do not enable a halfway path that samples with the
   new lease but publishes through the old ordering.
3. Handle unexpected post-sampling/precommit request-local failure using the P0 path;
   abort the lease, join workers, notify/unblock, then clean up. Propagate non-isolatable
   failures at their audited scope.
4. Process fork/free and publish outputs only after apply. Keep finish/stop/echo behavior
   equivalent after successful commit.

**Invariant:** Acceptance is not reconstructed from mutable counters; internal
intermediate states are invisible; cache state, counters, finish status, and output
describe the same committed endpoint.

**Focused tests:** explicit acceptance independent of counter mutation; partial/full
acceptance; finish and stop on the accepted token; streaming and non-streaming; echo;
failure before and after apply; delivery failure after commit does not roll back;
terminal cleanup; unaffected request completes when isolation is proven.

**Exit gate/review checkpoint:** Integrate the first scope through the existing
non-prefix or guarded internal test path: one sequence, greedy, static window. Prefix
verification, including prefix plus VERIFY, remains guarded until P4 supplies restore
and capacity gates and P5 removes each strategy guard independently. Preserve old paths
and guards for multinomial, beam/tree, multiple return sequences, dynamic candidates,
and non-prefix consumers not selected for integration.

**Out of scope:** generalized sampler staging, batch-wide atomicity, MTP pair admission.

### P4: Capacity, Publication, Restore, and Exact Rewind

**Ownership/files:** `BlockManager` unique-row accounting and registry operations,
`CacheOrchestrator` budget/no-gap publication/restore/rewind, scheduler admission, and
focused cache-pressure tests.

**Slices:**

1. Enforce a hard prefix-LA ceiling over unique resident rows and separately tracked
   reservations. Reserve current writes and next-window headroom before forward.
2. Publish the eligible crossed checkpoint set best effort and without logical holes.
   Preserve mandatory live state when optional publication cannot fit.
3. Resolve duplicate publication with the existing registry owner as canonical, without
   hot-path tensor comparison; cover final private duplicate and metadata mismatch.
4. Add the common per-sequence non-mutating restore plan, bounded by the caller's restore
  ceiling. Validate the endpoint and generation, pin the complete KV+LA endpoint before
  mutation, then revalidate the endpoint generation and apply KV, LA, hashes, live
  endpoint, and counters as one transition with no partial state. Release every pin and
  reservation when planning, pinning, revalidation, cancellation, or apply is abandoned;
  partial pin acquisition must unwind completely.
5. Implement exact saved-endpoint rewind or predecessor-plus-recompute. For a full-prompt
  LA hit, never replay the last prompt token into recurrent state already at `P`: select
  a true predecessor for recompute or consume caller-provided endpoint output metadata.
  Keep MTP shift and its hidden-state sidecar out of the common restore plan.

**Invariant:** Protected rows are never evicted; unique physical residency is counted
once; reservations cannot overcommit; restore never crosses a missing logical boundary;
restore exposes either the complete pinned KV+LA endpoint with matching live endpoint and
counters or no mutation; rewind never relabels recurrent state.

**Focused tests:** shared row under multiple ownership references; pins/live/checkpoints/
scratch pressure; reservation versus residency; optional-set skip; no-gap placement;
duplicate winner and final private duplicate; metadata mismatch; cancellation after
inference and during restore; exact and recompute rewind; nonzero logical start; stale
plan between planning and pinning; endpoint-generation change before apply; pinned rows
survive pressure; partial pin failure and abandoned-plan cleanup; full-prompt LA restore
does not replay the last token into state at `P`; live endpoint and counters remain
consistent on success and every abort path.

**Exit gate/review checkpoint:** Guarded/internal preparation for prefix verification in
the first scope remains within the configured ceiling under concurrent pressure, with
deterministic deferral before forward, atomic restore, and no rejected/scratch registry
entries. This gate does not externally enable prefix verification; P5 decides each
strategy guard independently.

**Out of scope:** sidecar byte accounting beyond recording the future requirement,
cross-request scratch sharing, policy-specific MTP publication.

### P5: Common-Foundation Validation and Guard Decisions

**Ownership/files:** prompt-lookup and independent-draft adapters only where needed,
their existing guards, common feature tests, and Python feature tests when a path becomes
enabled.

**Slices:**

1. Validate prompt lookup on the new common foundation.
2. Validate an ordinary independent draft model with a hybrid/LA main model.
3. Remove each guard independently only after its focused tests pass; retain all other
   strategy and non-prefix paths unchanged.

**Invariant:** Common LA transactions contain no MTP shift, pair-admission, or hidden-
state-sidecar assumptions.

**Focused tests:** first/repeated prompt, shared-prefix extension, first/partial/full
acceptance, output parity, observed nonzero restores, cancellation/preemption, prefix-off
behavior, and applicable Python feature tests.

**Exit gate/review checkpoint:** Each guard-removal decision cites an observed restore
and its focused correctness/failure suite. A successful no-restore generation run is not
a model pass.

**Out of scope:** Qwen3.5 MTP restore-policy selection and media.

### P6: Minimum Text MTP Adapter and M0b

**Dependency resolution:** Do not repeat old M0 before restore primitives and endpoint
metadata exist. P4 establishes the common atomic restore plan/pin/revalidate/apply path,
and P5 validates it and makes strategy-specific guard decisions before P6 adds the
minimum MTP endpoint sidecar needed for a restore-capable M0b experiment. Policy is
selected only after M0b observes real restores.

**Ownership/files:** `MtpDecodingImpl`, MTP parent/child admission, parent-derived draft
identity, minimum endpoint hidden-state/logit sidecar, and text MTP feature tests.

**Slices:**

1. Compose prepared main/draft transitions only where MTP needs pair semantics. Keep
   common transitions per sequence. Assert the draft graph is LA-free.
2. Add parent-derived draft identity and minimum text endpoint sidecar. Reject a hit
   lacking a valid `h[P-1]` or predecessor replay; never label a cold draft restored.
3. Enforce M1 prompt-only publication for both roles. Generated state remains private,
   including normal checkpoint intervals.
4. Run M0b on the real text model with counters proving nonzero restore. Compare coherent
   pair and main-only where both are feasible; measure parity, hit distribution, draft
   recovery, sidecar cost, TTFT, and total latency.
5. Stop for policy review. Select coherent pair, main only, or stop MTP enablement. Do
   not infer policy from a no-restore run.

**Invariant:** Main endpoint `P`, draft local endpoint `P-1`, counters, position data,
and `h[P-1]` describe one semantic boundary before either child runs.

**Focused tests:** shifted identity with equal suffixes/different parents; exact and
extending prompt; sidecar hit/miss/eviction; no partial pair visibility; real restore
counter; full/partial/rejected windows; target-only PA and prefix-off MTP parity.

**Exit gate/review checkpoint:** A restore policy is recorded only after M0b demonstrates
real text restores and acceptable correctness/lifetime behavior. M1 performance must
show at least 20% repeated-prompt TTFT improvement with variance reported.

**Out of scope:** image/video, generated-state publication, LA-bearing draft, per-request
policy switching.

### P7: Media and Remaining M1 Gates

**Ownership/files:** built-in Qwen3.5 input/embedder integration, identity/sidecar
metadata needed by image and video prompts, MTP tests, and Python feature tests.

**Slices:** text policy hardening, image enablement, video enablement, concurrency and
chunked-prefill coverage, then prefix-off performance measurement.

**Invariant:** Embedding-backed identity preserves the existing bounded sampling
assumption; independently supplied embedding/position-ID pairs remain guarded unless
their identity contract is explicitly added. Generated MTP state remains unpublished.

**Focused tests:** first/repeated/shared-prefix text, image, and video prompts; media
identity distinction; sidecar lifetime; concurrency; cancellation; chunked prefill;
observed restore; Python API feature tests.

**Exit gate/review checkpoint:** Text and built-in media paths meet correctness and
lifetime gates. Prefix-off speculative throughput and inter-token latency regress by no
more than 3%, with methodology and variance reported.

**Out of scope:** externally supplied position identity and M2 publication.

### P8: Optional M2 Generated-State Publication

This is a separate decision and implementation plan after M1 ships. Proceed only if
generated draft state is proven path-independent or a bounded canonical construction is
selected. Publication must compose canonical main KV+LA, draft KV, and sidecars; it must
not weaken the M1 prompt-only contract by default.

**Ownership/files:** Defined by the separate M2 decision after canonicality evidence is
reviewed; expected owners are the MTP adapter, cache publication transaction, endpoint
sidecars, and generated-history tests.

**Invariant:** A restorable generated endpoint contains one canonical and coherent main,
draft, and sidecar state for its identity, independent of the acceptance path.

**Focused tests:** Multiple acceptance/rejection histories reaching equal identities;
cross-request restore; atomic duplicate publication; sidecar lifetime; extended-chat
parity; capacity and performance regression.

**Exit gate/review checkpoint:** Approve a separate M2 plan only after canonicality or a
bounded canonical construction is demonstrated. Otherwise M1 remains the shipped scope.

**Out of scope:** Any opportunistic generated-state publication before this decision.

## Build and Validation Commands

Use the available VS Code CMake Tools integration for C++ builds. Build target:
`tests_continuous_batching`. CMake Tools currently has no worker-count argument; when a
direct documented build is required, use exactly:

```bash
cmake --build ./build --target tests_continuous_batching -j18
```

CMake Tools builds under `build/vscode-__unspec__/`. Run the freshly rebuilt
`tests_continuous_batching` binary from that same tree, never a stale `build/bin` binary.
There are no CTest registrations for this suite, so run the gtest binary directly.

Build `tests_cache_allocation_failure` with CMake Tools and run
`build/vscode-__unspec__/bin/tests_cache_allocation_failure` as an additional P1-P3
gate. Its allocation replacement is isolated from the ordinary test executable.
The new executable is installed with the test component and invoked by the existing
Linux, macOS and Windows unit-test CI steps. Local validation was on Linux; the other
platform runs remain CI checks.

For matching Python and C++ integration, only when the phase needs it, use exactly:

```bash
pip install --pre -U . --no-deps --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
```

Baseline filters are examples, not substitutes for slice-specific tests:

```text
TestBlockManager.*:TestBlockAllocator.*:*PrefixCachingCopyOnWrite*:*CacheOrchestrator*:*LinearAttention*:*MtpPrefix*
TestScheduler.*
CBPublicationTest.*:TestScheduler.hybrid_prefix_caching_republishes_cow*:CBForSDTest.DraftPipelineKeepsKvOnlyCompletedBlocksUnpublished
```

Report the exact selected, passed, failed, and skipped counts. With `CACHE_TYPES_CSV`
unset, environment-gated suites are excluded; do not report them as passes. Add focused
tests for every changed area and Python feature tests when a feature is enabled.

Performance gates are at least 20% repeated-prompt TTFT improvement for actual M1
restores and no more than 3% prefix-off throughput or inter-token-latency regression,
both with run-to-run variance. A no-restore run cannot satisfy a model gate.

## Unresolved Risks and Review Decisions

- P0 may find that sampler-future failure is not request-isolatable or that existing
  generation status cannot terminate and unblock one request without an API change.
- The exact apply boundary may still contain an operation that allocates or can fail;
  this must be moved to prepare before first-scope enablement.
- Duplicate canonical publication can preserve a private final row and therefore consume
  more capacity than the ideal canonical-reference path.
- No-gap checkpoint retention may reduce hit density under pressure; policy must prefer
  correctness and explicit deferral over sparse unusable chains.
- Sidecar byte accounting and eviction coupling are unresolved until P6, and can change
  MTP value even when row accounting is correct.
- The existing alternative-owner scan under the cache mutex is unmeasured and may need a
  separate optimization after correctness is stable.
- Sparse main/draft endpoint alignment or cold-draft recovery may make both MTP restore
  policies unattractive. M0b may legitimately stop MTP enablement.
- Generated draft state remains path-dependent in existing evidence. M2 is optional and
  must not block M1 prompt reuse.
