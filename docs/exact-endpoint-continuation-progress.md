# Exact-Endpoint Continuation Progress

## Current Handoff (2026-09-11)

P1-P3 committed at user request as `0ce159919`:
`Prepare coherent KV and linear-attention commits with allocation failure coverage`.
Hooks passed after end-of-file formatting. Unrelated worktree changes were excluded.

The next P4 capacity slice is implemented and uncommitted:

- Production prefix-LA registration now retains an explicitly configured hard row
  ceiling; inferred limits keep their previous growth behavior.
- Dynamic admission stops unrelated KV growth when the capped LA pool blocks the
  target. Both prompt modes defer before forward without changing protected rows,
  hashes, counters or ownership; scheduling resumes when the owner releases its row.
- Split-fuse can still schedule a partial prefill within the available LA capacity.
- Three focused tests, 752 selected C++ tests, six allocation-failure tests and five
  local LFM Python regressions pass. The Python process loaded the rebuilt native
  library through `LD_PRELOAD`; CSV-backed model suites remained excluded.

P4 is not complete: explicit reservations/headroom, optional no-gap publication,
duplicate canonicalization, pinned atomic restore and rewind gates remain. P5 guard
decisions have not started. Final Qwen3.5 MTP acceptance still requires VLM/media.
The sections below preserve the earlier P1-P3 handoff and validation history.

## Active Milestone Contract

Updated 2026-09-11. Baseline: `d06001c43`. Execute P1-P3 as one implementation
cycle with one independent review and integrated validation, not per-slice
approval stops. Do not commit without authorization.

P1-P3 acceptance checklist:

- Prepared accepted KV/LA/counter transition for the selected static greedy
  verifier path, without sampler snapshots or selection-algorithm changes.
- Preparation failure leaves committed cache state intact; P0 fails/unblocks the
  entire base pipeline and cleans speculative resources on unexpected errors.
- Accepted terminal/stop state and deferred output agree with the committed
  endpoint; fork/free and notification ordering preserve that boundary.
- Unsupported verifier modes have explicit guards or a documented preserved
  legacy path; prefix+verification remains guarded until P4-P5 acceptance.
- Focused failure and acceptance tests, selected C++ suite, and real-model
  prompt-lookup validation pass. Report excluded model suites explicitly.

P1-P3 coordinated-transition implementation updated locally on 2026-09-11:

- Greedy single-sequence verifier sampling now returns accepted depth and the
  absolute processed-token endpoint without committing the request counter.
- The pipeline validates acceptance, KV scheduling metadata, the deferred
  counter endpoint, scratch lease, and LA promotion. KV tail preparation takes
  explicit accepted endpoints and preallocates release nodes; lock acquisition
  order is KV then LA. Apply then commits the counter, trims rejected KV tail blocks,
  promotes LA state, and releases scratch before fork/free, candidate generation,
  embedding synchronization, or external notification delivery.
- Missing or stale acceptance/KV metadata fails before cache mutation. P0 remains
  the terminal whole-pipeline failure boundary and releases speculative scratch.
  Non-greedy verification retains the previously supported processed-depth path;
  prefix caching plus verification remains guarded.
- The three-row logical-offset regression now asserts supported capacity
  deferral: predecessor-backed rows and published hashes remain unchanged when
  continuation requires unavailable COW capacity. It does not claim an exact-P
  cache hit.

Validation:

- `tests_continuous_batching` built with CMake Tools. Fifteen focused publication,
  notification, failure-boundary, and rollback tests passed.
- The final independent selected C++ suite passed 749 tests from 108 suites with
  `CACHE_TYPES_CSV` unset. CSV-backed cache-type and backend-routing model suites
  were excluded. Log: `temp/p1_p3_cpp.log`.
- `openvino_genai` built with CMake Tools. Five selected local LFM2.5 hybrid
  prompt-lookup/cache Python cases passed; 201 tests were deselected. The test
  process used `LD_PRELOAD`, and `/proc/self/maps` confirmed
  `build/vscode-__unspec__/openvino_genai/libopenvino_genai.so.2026.5.0.0`.
  The final five-case run was independently repeated after rebuilding the library.
- VS Code diagnostics are clean for all eight changed source/test files. Scoped
  `git diff --check` passes. Global `git diff --check` still reports pre-existing
  trailing whitespace in the preserved generated Python stub.
- Independent review rejected the initial throwing KV cleanup after counter
  mutation. Repair added actual prepared KV releases. Subsequent checks repaired
  opt-in sampler finalization, accepted-endpoint release targets, and prepared
  object misuse checks. The integrated step test now reaches real sampler output,
  invalidates its KV transition metadata, and verifies original-exception delivery
  to active/awaiting handles with no deferred output and complete cache cleanup.
- The remaining P1-P3 host-allocation gate passed on 2026-09-11: six isolated
  `tests_cache_allocation_failure` cases sweep each allocation through scratch-lease
  acquisition, combined KV/LA preparation and the actual pipeline commit. Failures
  preserve both counters, endpoints, generations, references, free capacity and active
  leases. Successful apply and lease cleanup allocate nothing. One/two allocator
  layers, two sequences and first/partial/full acceptance are covered. Injection is
  scoped to ordinary C++ heap allocations in these paths, not device allocations or
  general process-OOM recovery. No production-code changes were needed for this gate.
- The final selection was rerun: 749 existing C++ tests plus six isolated tests and
  five LFM Python cases passed. The isolated target is installed and added to all three
  unit-test CI workflows; only Linux was executed locally. P1-P3 exit gates have now
  passed locally; P4-P8 remain outside this completed milestone.
- No staging, commit, push, branch, reset, or revert was performed.

MTP adapter work and P4 capacity publication remain outside this milestone.
Final MTP acceptance still requires media through the Qwen3.5 VLM pipeline; the
local text-only LFM validation above is not a substitute for that later gate.

The user now permits predecessor-checkpoint reuse for the three-row pressure
case. Update that test to describe legitimate supported behavior; do not bypass
write protection or relabel recurrent state. Zero-forward full-hit sampling and
endpoint-logits caching remain on their separate feature branch.

Final MTP acceptance MUST include media through the Qwen3.5 VLM pipeline, which
is the user's existing Qwen3.5 coverage path. A text-only LLM test is not a
substitute for that coverage. Shared P1-P3 work must preserve embedding-backed
paths; MTP image/video identity, sidecar alignment and actual restore tests remain
required later integration work, not optional features or reasons to expand P1-P3.

## Branch Separation

On 2026-09-10, the endpoint-logits feature was separated at the user's request:

- `feature/prefix-cache-endpoint-logits` retains implementation `f682076db` and
  Python regression `f3b57159d` (commit hooks applied formatting only).
- The active `linear-attention-prefix-cache` branch is back at `e67da1e42`.
  The feature commit's exact-endpoint counter changes also moved with it; any
  required MTP checkpoint-continuation corrections must be implemented separately.
- Cached first-token sampling is not a prerequisite for MTP prefix continuation
  with new input. Do not carry over a zero-forward full-prompt-hit requirement
  merely to implement MTP checkpoint restore.
- The validation below belongs to the feature branch. Existing build outputs and
  the installed Python package still contain that feature and must be rebuilt
  before validating the active MTP branch. Its earlier rollback limitation is
  not fixed by the historical results below.
- Unrelated tracked diffs were verified unchanged by checksum; local untracked
  documentation and other files were preserved. Nothing was pushed.

## Prepared LA Batch Promotion

On 2026-09-11, a failure regression reproduced partial LA commit: missing
acceptance for a second sequence threw after the first sequence's live endpoint,
generation, row ownership, and scratch had already changed.

The uncommitted implementation now collects acceptance metadata before mutation
and prepares all LA promotions under one BlockManager lock. A move-only prepared
object owns that lock and validated map iterators. Apply performs reference
updates, ownership swaps, and allocation-free scratch release. Lease states are
marked committed after apply. Invalid slots, duplicate sequence IDs, stale base
metadata, and mismatched lease/request identities are checked before mutation.
The lock-owning prepared object must be released before aborting its leases;
the pipeline applies immediately and the abandonment test observes this order.

Independent verification:

- Nine focused C++ tests passed, covering missing second acceptance, invalid
  second slot, successful two-sequence retry, move/abandonment, moved-from and
  repeated apply rejection, and real sampler acceptance/terminal promotion.
- A two-LA-layer fixture exercises the production convention of one shared
  logical LA row; it is not a test of separate allocator tables per model layer.
- The selected suite passed 744 of 745 tests. The existing
  `PrefixCachingRollbackInvalidatesLatestOnlyEndpointWithLogicalOffset` still
  fails before promotion, during insufficient-capacity append setup. CSV-backed
  model suites were excluded. Log: `temp/prepared_promotion_cpp.log`.
- All five local real-hybrid Python regressions passed against the rebuilt
  library via process-local `LD_PRELOAD`; `/proc/self/maps` verified the library
  came from `build/vscode-__unspec__/openvino_genai`. No package reinstall was
  performed; ordinary Python invocations still use the installed package.

This closes validation-before-mutation for the LA promotion batch only. It does
not yet make KV rollback, sampler-mutated counters, finish state, and LA promotion
one prepared request transition. Allocation-failure injection at every acquisition
and full precommit pipeline failure coverage remain required. Non-greedy fallback
behavior is unchanged. Prefix+verification stays guarded, and no endpoint-logits
feature was introduced. Nothing was staged or committed.

Next: extend the prepared contract to accepted KV/counter state and test failures
at that boundary, using prompt lookup before any MTP-specific integration.

## MTP Branch Prompt-Lookup Baseline

On 2026-09-10, follow-up validation on `linear-attention-prefix-cache` at
`e67da1e42` used the user's installed Python package, reporting
`2026.5.0.0-3417-e67da1e4245-linear-attention-prefix-cache`. No reinstall or
production changes were made. New test changes remain uncommitted.

- Five Python cases passed on local LFM2.5-350M FP16, CPU. Static prompt-lookup
  windows of one and four tokens match ordinary greedy decoding with prefix
  caching off, including repeated requests on the same pipeline. With prefix
  caching on, both windows raise the existing verifier guard as expected.
- Separately, a 128-token cached prefix followed by either of two new suffixes
  produces the same 16 generated tokens as cold decoding without prompt lookup.
  This is output-parity evidence, not a measured no-replay trace or proof of
  real-model checkpoint reuse on its own.
- Six focused C++ tests passed independently with the rebuilt matching binary.
  The partial-prefix scheduler test explicitly verifies restored LA endpoint
  and processed count at P, past length P, one scheduled suffix token, and a
  private successor row.
- Real sampler decisions, rather than fabricated acceptance records, exercise
  accepted depths 1, 2, and 4 plus a stop-token case at depth 2. Pipeline commit
  promotes the corresponding LA row, advances to the absolute expected endpoint,
  and releases scratch. KV tables remain present; this does not prove all KV
  rollback or exception-atomicity requirements.

The partial-hit restore path already uses P when new input follows the cached
prefix. The full-hit decrement and rollback limitation remain separate.
No cached-logit code was restored, and prefix+verification remains disabled.
Next work is the prepared-transition/failure-injection contract for the guarded
static greedy verifier path, before enabling combined prefix+prompt lookup.

## Endpoint-Logits Feature Validation

Updated 2026-09-10. Implementation committed as `f682076db`
(`Resume prefix cache hits at exact endpoints`), on top of `e67da1e42`.
This is a verification handoff, not acceptance of the entire P1-P3 milestone.

## Implemented and Checked

- Restored cache counters name the exact endpoint. The original three-row
  logical-offset rollback test passes without predecessor replay.
- Full-prompt hits with reusable raw endpoint output sample without a forward;
  the following generated token is processed by normal forward inference.
- Raw output is copied before sampling. Artifact identities are prepared after
  cache publication and installed only after coherent LA, fork/free, candidate,
  and embedding updates, before external notification delivery.
- Partial-prefix prompt history is retained for echo and non-echo producers.
  Multi-chunk echo consumes restored history once, without repeating the prefix.
- Mixed cached/forward batches and decode-before-prompt row offsets are tested.
- Artifact lookup rejects reused backing physical identities. Candidate failure
  does not install reusable endpoint output.
- Base admission restore is serialized with step execution. Malformed partial
  endpoint metadata releases restored KV/LA ownership, resets the processed
  counter, and preserves the original error.

## Independent Verification

CMake Tools built `tests_continuous_batching` in `build/vscode-__unspec__`.
The final independent run passed **754 tests from 107 suites**:

```bash
env -u CACHE_TYPES_CSV ./build/vscode-__unspec__/bin/tests_continuous_batching \
  --gtest_filter='-*GoogleTestVerification.UninstantiatedParameterizedTestSuite*'
```

CSV-dependent real-model cache-type and backend-routing suites are excluded,
not counted as passing. Pipeline acceptance uses deterministic C++ test models;
these results do not establish real hybrid-model accuracy or performance.

The Python package was rebuilt and installed after the commit. Its version is
`2026.5.0.0-3418-f682076db3e-linear-attention-prefix-cache`, loaded from the local
`venv`. The generated Python stub diff was preserved.

### Real Hybrid Exact-Hit Validation

The local real LFM2.5-350M FP16 hybrid export at `temp/lfm2.5-350m-fp16-ov`
passed `test_exact_endpoint_real_hybrid` in
`tests/python_tests/test_continuous_batching.py` on CPU. The test uses an encoded
128-token prompt, prefix caching, static prefill, greedy decoding, and two generated
tokens with EOS ignored. Cold and repeated full-hit token IDs match; generated
log probabilities match with `atol=1e-5, rtol=1e-5`.

GDB traced the resolved entry address of `ov::genai::ModelRunner::forward` in the
installed native library, without modifying production instrumentation:

| Step | Forward calls |
| --- | ---: |
| Cold first sample | 1 |
| Cold second sample | 1 |
| Cached first sample | 0 |
| Cached second sample | 1 |

Trace: `temp/exact_endpoint_lfm25_gdb_entry.log`. Both the real-model test and the
generation-status enum contract test passed; the inferior exited normally. The
initial name-based breakpoint reported duplicate hits, so the final trace used
the resolved function entry to count each call once. No-forward evidence is from
this debugger trace, not a timing threshold or a public Python metric.

This validates one real hybrid model/configuration, not Qwen SSM/MTP parity,
broader model accuracy, or performance. The new Python regression is uncommitted.

## Remaining Gates

- Review full-prompt missing-artifact behavior and unsupported request shapes;
  do not silently replay or weaken existing strategy guards.
- Close the original P2/P3 prepared-transition and coherent-apply gates, including
  acquisition-failure coverage and the allocation-free apply contract. Current
  scratch preparation does not by itself establish the combined KV/LA/counter
  guarantee required by the plan.
- Resolve non-greedy verifier acceptance before milestone approval: the current
  commit path still derives non-greedy acceptance from processed counters, and
  verifier validation does not enforce the plan's greedy-only first scope.
- No MTP adapter, strategy enablement, or complete P4 capacity/restore guarantee
  is claimed. The three-row rollback test already schedules two tokens from 12
  to 14; an audit suggestion that this continuation was missing was rejected
  after inspecting the test.

The implementation commit contains only ten C++ implementation/test files and passed commit hooks.
Unrelated site, sample, tokenizer, measurement-tool, stub, and local documentation
changes were excluded and preserved. The Python regression was subsequently
committed on the feature branch as noted above; this handoff remains local/untracked.
