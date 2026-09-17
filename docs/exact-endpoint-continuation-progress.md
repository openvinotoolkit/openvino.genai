# Exact-Endpoint Continuation Progress

## Current Handoff (2026-09-15)

LFM is committed as `43b1c3df1`; Qwen3.5 paired prefix replay is committed as
`80c5236b1`. The explicit `ChatHistory` follow-up is committed as `751f910ea`:

- Assisted history requests now forward the aligned prompt IDs without requiring
  Omni outputs, and MTP initializes its strategy-level vision registry.
- Qwen's single-media merger copies the stored encoded tensor before adding
  positional embeddings in place. Previously, repeated history calls mutated the
  registry-owned tensor, changed prefix identity, and lost warm cache reuse.
- Four real-model MTP cases passed with text, image, video and video metadata.
  Each checks explicit-history parity, observed warm hybrid/paired restores, and
  a follow-up turn using retained media. Existing string-input long-generation,
  cancellation/reuse and changed-media checks also run in these cases.
- The native library builds and scoped C++ diagnostics are clean. The broader
  native and model-suite counts below describe the preceding committed milestone.

### Converted Tiny Qwen3.5-MTP Tests (2026-09-17)

The rollback fixes and real-model validation below are committed as `98e678c92`.
The subsequent test-portability change replaces `OV_GENAI_QWEN35_MODEL` in the
three Qwen test functions with a module-scoped fixture using `_get_ov_model` and
`optimum-intel-internal-testing/tiny-random-qwen3.5-mtp`. The fixture requires
`openvino_mtp_model.xml`; the existing Qwen3.5 Transformers minimum-version gate
also applies to this model.

All 35 migrated cases passed in 116.96 seconds, including conversion. The existing
parity, warm hybrid/paired replay, draft activity, changed media, cancellation,
prompt extension, retained chat media, direct admission, and prefix-mismatch
assertions are unchanged. The local-model environment variable was unset.

The installed exporter's `Qwen3_5DynamicCache` import fails with the active
Transformers package. Validation used the approved Transformers 5.2.0 overlay
without modifying the venv or overriding its activated OpenVINO runtime:

```bash
source venv/bin/activate
env -u LD_PRELOAD -u OV_GENAI_QWEN35_MODEL OV_CACHE="$PWD/ov_cache0" \
  uv run --active --no-project --with transformers==5.2.0 python -m pytest \
  tests/python_tests/test_vlm_pipeline.py \
  -k 'test_qwen35_mtp_text_add_request or test_qwen35_mtp_rejects_prefix_mismatch or test_qwen35_vlm_verifier_cache_contract' \
  -x -q --tb=short --show-capture=no --disable-warnings
```

### Converted Tiny Hybrid CB Tests (2026-09-17)

Both CB tests previously gated by `OV_GENAI_HYBRID_MODEL` now use the existing
`llm_model` conversion fixture, parameterized by `LINEAR_ATTENTION_MODELS_LIST`
(`optimum-intel-internal-testing/tiny-random-lfm2`). This preserves independent
speculative decoding rather than substituting MTP. All existing assertions remain
unchanged, including draft acceptance, warm hybrid restore, cancellation/reuse,
and cached-prefix continuation with new input.

All 17 cases passed in 33.90 seconds with `OV_GENAI_HYBRID_MODEL` unset, using
the activated runtime and the Transformers 5.2.0 overlay above. The installed
exporter rejects LFM2 export with Transformers 5.5.0 (maximum supported: 5.4.0).
The pytest selection was
`tests/python_tests/test_continuous_batching.py -k 'test_hybrid_cached_prefix_with_new_input or test_hybrid_verifier_cache_contract'`.

### Interval-4 Benchmark Rerun (2026-09-17)

The benchmark was rerun successfully in the explicitly activated environment.
`venv/bin/activate` intentionally sources `setupvars.sh`; its
`PYTHONPATH=/home/apaniuko/openvino/python` and
`LD_LIBRARY_PATH=/home/apaniuko/openvino/runtime/...` were preserved. In that
shell, `openvino` imported from `/home/apaniuko/openvino/python/openvino/__init__.py`
and `ov.get_version()` returned `2026.5.0-22974-1af7b61853c`. Installed GenAI
imported from
`venv/lib/python3.10/site-packages/openvino_genai/__init__.py`; its metadata is
`2026.5.0.0`. The pip `openvino` metadata is `2026.3.1`, but it is not the
runtime selected by the activated `PYTHONPATH` and was not used as a blocker.
The obsolete `build/vscode-__unspec__/openvino_genai/libopenvino_genai.so`
Debug `LD_PRELOAD` was removed when present. `LD_PRELOAD` was empty during the
run, and `OPENVINO_LOG_LEVEL` and `OV_GENAI_MTP_TIMING_JSONL` were unset.

The available CMake provenance reports `CMAKE_BUILD_TYPE=Release` and
`ENABLE_LTO=OFF` for `build/` and `build-review/`; the installed wheel's full
build flags were not independently exposed. This is a provenance limitation,
not a timing blocker. No environment packages were changed.

Conditions match the prior interval-4 experiment: local Qwen3.5-2B INT4
actual-MTP model, CPU with 18 inference threads and one stream, text-only
`VLMPipeline` with PA, greedy generation with `ignore_eos=True`, 64 output
tokens and four assistant candidates. Main cache uses 1 GiB, 32 linear-
attention rows, interval multiplier 4, batch-token limit 256 and one sequence;
the draft uses 256 KV blocks. Three repetitions rotate mode order. Each length
has a distinct-prefix untimed warmup, followed by measured first and exact
repeat submissions. Interval 1 remains excluded because its long prompt
exceeded capacity. Model loading is excluded from timings.

| Input tokens | Mode | TTFT median (min-max) ms | Wall median (min-max) ms |
| ---: | --- | ---: | ---: |
| 187 | Main-only, prefix off | 659.9 (640.7-678.0) | 2622.8 (2600.3-2636.6) |
| 187 | MTP, prefix off | 649.4 (642.1-653.5) | 2810.4 (2797.1-2868.3) |
| 187 | MTP, prefix on cold | 676.2 (648.3-676.8) | 2842.8 (2800.2-2854.6) |
| 187 | MTP, prefix on warm | 229.5 (226.6-239.7) | 2392.5 (2386.2-2406.1) |
| 668 | Main-only, prefix off | 2280.7 (2191.7-2350.9) | 4244.8 (4155.5-4319.4) |
| 668 | MTP, prefix off | 2237.0 (2208.5-2267.3) | 4433.4 (4374.4-4459.0) |
| 668 | MTP, prefix on cold | 2244.3 (2230.7-2370.9) | 4400.5 (4393.3-4568.1) |
| 668 | MTP, prefix on warm | 136.7 (135.2-141.6) | 2305.4 (2302.1-2334.0) |
| 2588 | Main-only, prefix off | 8734.4 (8496.9-8899.3) | 10720.3 (10509.9-10879.0) |
| 2588 | MTP, prefix off | 8538.3 (8499.8-8625.7) | 10794.2 (10754.0-10941.4) |
| 2588 | MTP, prefix on cold | 8737.2 (8599.9-9117.4) | 11003.1 (10852.5-11414.9) |
| 2588 | MTP, prefix on warm | 174.2 (173.4-184.3) | 2461.2 (2438.1-2466.2) |

Warm MTP-prefix versus MTP-off TTFT speedups are **2.83x/16.36x/49.02x**;
wall-time speedups are **1.17x/1.92x/4.39x**. Compared with main-only,
warm MTP TTFT speedups are **2.88x/16.68x/50.15x** and wall-time speedups are
**1.10x/1.84x/4.36x** for 187/668/2588 tokens. MTP acceptance was 37.5% in
every MTP cell. All 54 measured outputs generated exactly 64 tokens and matched
by SHA-256 within each scale; each of the 18 cells has three samples.

These results measure a synthetic repetitive-text workload only. They do not
measure media latency, model loading, main-only prefix-on reuse, or general MTP
speedups. "Cold" means the first measured prompt after the distinct-prefix
warmup, not cold startup.

Exact reproduction, with each fresh shell explicitly activated:

```bash
source /home/apaniuko/cpp/openvino.genai/venv/bin/activate
unset LD_PRELOAD OPENVINO_LOG_LEVEL OV_GENAI_MTP_TIMING_JSONL
python /home/apaniuko/cpp/openvino.genai/temp/qwen35_mtp/prefix_replay_bench.py \
  --model /home/apaniuko/cpp/openvino.genai/temp/qwen35_mtp/Qwen3.5-2B-int4 \
  --repetitions 3 --scales 32 128 512 --tokens 64 \
  --output /home/apaniuko/cpp/openvino.genai/temp/qwen35_mtp/prefix-replay-benchmark-interval4-20260917.json
```

Raw ignored artifacts are
`temp/qwen35_mtp/prefix-replay-benchmark-interval4-20260917.json` and
`temp/qwen35_mtp/prefix-replay-benchmark-interval4-20260917.log`. The prior
2026-09-15 measurements below remain historical baselines and were not
overwritten.

### API Tests and Diagnostic Performance (2026-09-15)

Added local-model Python tests for direct string `add_request` (token parity,
request-ID reuse, observed warm hybrid/paired restore) and both directions of
explicit MTP prefix-setting mismatch. All three cases passed. The MTP-only
linear-attention draft guard was extracted unchanged into a protected method;
one/two-layer native cases exercise KV-only versus hybrid state, prefix on/off,
and main versus draft roles. All 34 isolated tests passed. Eight existing LFM
independent-draft cases also passed across prefix settings, split modes and
candidate counts, confirming that independent hybrid drafting remains allowed.
The LA guard matrix tests the production predicate with synthetic schedulers,
not construction of an unsupported real LA-bearing MTP export.

Diagnostic benchmark: local Qwen3.5-2B INT4, actual MTP submodel, CPU
Intel Core i9-10980XE, 18 inference threads, one stream, text-only `VLMPipeline`
with PA. Each request generates 64 tokens, greedy, ignoring EOS, with four MTP
candidates. Main-only has prefix caching disabled. Main cache budget is 1 GiB,
32 LA rows, checkpoint interval multiplier 4, batch-token limit 256; draft uses
256 KV blocks. Three repetitions rotate the mode order, each using a newly
constructed pipeline and untimed per-length warmup with a different prefix.
The measured prompt is then submitted twice. "Cold" means first prompt use,
not cold model loading; "warm" means the immediate exact repeat. Prompts repeat
"one two three four." at three lengths. All 54 measured outputs matched by text
SHA-256 and had exactly 64 generated tokens. Draft acceptance was 37.5% throughout.

Median timings, three samples per cell. Off/main rows use repeat submissions
to match the warm run; cold prefix rows use first submissions. Times are ms.

| Input tokens | Mode | TTFT | Total request | Total min-max |
| ---: | --- | ---: | ---: | ---: |
| 187 | Main-only, prefix off | 712.3 | 3038.8 | 2994.3-3039.6 |
| 187 | MTP, prefix off | 691.9 | 4021.6 | 3976.2-4118.7 |
| 187 | MTP, prefix on cold | 682.3 | 3973.6 | 3942.8-4106.5 |
| 187 | MTP, prefix on warm | 251.5 | 3595.9 | 3531.6-3716.0 |
| 668 | Main-only, prefix off | 2249.4 | 4560.0 | 4539.6-4687.4 |
| 668 | MTP, prefix off | 2311.2 | 5766.2 | 5522.9-5893.9 |
| 668 | MTP, prefix on cold | 2321.0 | 5714.7 | 5626.3-5825.7 |
| 668 | MTP, prefix on warm | 182.7 | 3627.1 | 3469.5-3637.9 |
| 2588 | Main-only, prefix off | 8516.8 | 10864.3 | 10847.3-11266.7 |
| 2588 | MTP, prefix off | 8920.7 | 12511.6 | 12132.4-12555.9 |
| 2588 | MTP, prefix on cold | 8830.7 | 12288.8 | 12275.8-12466.9 |
| 2588 | MTP, prefix on warm | 275.3 | 3833.2 | 3783.4-3885.0 |

Warm versus MTP-off TTFT speedups are 2.75x/12.65x/32.41x; total-time speedups
are 1.12x/1.59x/3.26x. Compared with main-only, warm MTP is slower at 187 tokens
and 1.26x/2.83x faster at 668/2588. Cold MTP does not beat main-only on this
workload. These results isolate a repeated-prefix benefit, not a general MTP
speedup, and do not measure image/video latency or main-only prefix-on reuse.

Historical setup limits: the 2026-09-15 run used a preloaded GenAI Debug
library, while its provenance check recorded pip OpenVINO metadata 2026.3.1
and the activated native runtime 2026.5.0-22974-1af7b61853c. The benchmark
completed through `openvino_genai`; metadata recorded GenAI 2026.5.0.0 and
Transformers 5.5.0. The 2026-09-17 run above intentionally used the activated
nightly paths without that obsolete preload. An initial interval-1 attempt
exhausted the LA budget on the long prompt; those partial results are excluded.

Local reproduction and raw data (ignored, not intended for commit):
`temp/qwen35_mtp/prefix_replay_bench.py` and
`temp/qwen35_mtp/prefix-replay-benchmark-interval4-20260915.json`.
Run with `venv/bin/python`, `LD_PRELOAD` pointing to
`build/vscode-__unspec__/openvino_genai/libopenvino_genai.so`, logging/timing
instrumentation disabled, and `--output` naming a new JSON file. The harness
defaults reproduce the table's three lengths and three repetitions.

### Paired-Admission Failure Testing (2026-09-15)

Implemented and uncommitted on top of `751f910ea`. The allocation sweep found and
fixed two production defects:

- Cleanup of restored rows allocated temporary vectors and cache-store nodes.
  Under sustained allocation failure, rollback threw and left the failed pair
  queued. Prefix preparation now reserves release records before acquiring rows;
  freeing an unchanged restored table uses allocation-free prepared release.
  Records are removed with their table and validated against current row identity
  and the saved hash before reuse.
- The KV-only draft used incremental restore, which could leave a partially
  initialized table and crash cleanup. Single-cache orchestrator restore now uses
  the same prepared restore primitive as the hybrid main.

The production MTP admission/negotiation path is exercised through synthetic
token-backed child ingress. One/two-layer cases cover matching checkpoints at 8
and negotiation back to a draft checkpoint at 4, both with a pinned neighboring
pair and with no neighbor. The allocator keeps failing during production cleanup.
Checks cover both awaiting queues, draft handles, neighbor state, row references,
published hashes, capacity, temporary/headroom ownership, and successful reuse of
the failed request ID. Failures after both children are queued and after main
restore are explicitly observed.

Validation:

- Eight sweeps exercised 1,662 failed admissions and eight successful retries.
  One-layer counts: 228/176 failures for endpoints 4/8, for each ownership case;
  two-layer counts: 242/185 respectively.
- All 32 isolated allocation-failure tests and 755 selected native tests passed.
  CSV-backed model/cache-routing suites remain excluded.
- Four real Qwen3.5 MTP prefix-enabled, four-candidate text/image/video/video-metadata
  cases passed against the rebuilt library, including explicit history, retained
  media, warm restore, long generation, cancellation/reuse and prompt extension.
- Scoped C++ diagnostics and changed-file whitespace checks are clean.

Limits: injection covers host `new` allocations on the admission thread, not
device OOM, concurrent scheduling, or real embedding/position construction. The
real-model checks validate generation separately, without allocation injection.
No cross-child prepared-apply transaction or performance baseline was added.

### Paired-Replay Milestone (2026-09-14)

VLM prompt lookup and the actual Qwen3.5 MTP submodel now pass local prefix-on/off
greedy parity checks with built-in text, images, video and video metadata.

The MTP adapter restores both children to local processed length `D`: main state
covers original `[0,D)`, while draft state covers shifted embeddings `[1,D+1)`.
The main replays from `D` to regenerate the hidden-state suffix before draft work.
Negotiation moves to earlier checkpoints until both processed lengths match;
no paired hit restarts both at zero. This is predecessor replay, not endpoint-logit
or endpoint-hidden-state caching. Draft identity uses the main prefix hash at
`D+1`. Main identity includes complete aligned prompt IDs and the existing bounded
embedding samples. Both roles publish completed prompt rows only; generated rows
remain private. Admission rollback removes the failed pair's awaiting requests.
Scheduling and admission share the strategy mutex; per-child hybrid restore remains
prepared and atomic, but there is no cross-child prepared-apply transaction.

Validation using the rebuilt native library:

- 755 selected native tests passed; CSV-backed model/cache-routing suites excluded.
- All 30 isolated cache tests passed after the complete-ID identity change.
- All 32 local Qwen VLM cases passed; all 16 MTP cases passed again after complete
  prompt-ID hashing and admission validation. Warm prefix-on calls require observed
  hybrid and paired MTP restores, not merely output parity.
- Four-candidate prefix-on MTP cases also cover 80-token decoding, cancellation,
  reuse, prompt extension and changed-media parity for all four media modes.
- Tiny Qwen3.5 prompt lookup and all 17 LFM regressions passed with Transformers
  5.2.0 and Optimum Intel `2.3.0.dev0+fdad637`.
- Tiny Qwen2-VL passed with `uv run --active --no-project --with
  transformers==4.57.6 python -m pytest`, leaving the main venv unchanged.

Acceptance limits at that milestone: no latency/TTFT comparison against main-only
restore, no cross-child admission allocation-failure sweep (now covered above), and no dedicated negative test
for arbitrary caller-supplied position IDs. The aligned-ID admission guard is not
proof that inputs came from the built-in embedder. LA-bearing drafts are rejected
when prefix caching is enabled. Generated-state publication stays out of scope.

### Previous LFM Handoff

The scoped LFM common-foundation milestone is implemented and verified locally,
uncommitted on top of `3b52846fa`. This enables prefix verification for greedy
token-input requests with one sequence and a static candidate window. Both prompt
lookup and ordinary independent-draft decoding are covered. LFM has no MTP submodel;
none of these results establish Qwen3.5 MTP, VLM, or media support.

Implemented behavior:

- Hybrid restore prepares KV and LA under ordered locks before applying either.
  Both select the same endpoint; full-prompt hits use a real predecessor, not a
  counter rewind into state that already includes the last prompt token.
- Verification uses private, hash-invisible KV allocations and token-precise LA
  scratch. Accepted promotion preserves published bases and retains physical
  next-window headroom, which competing requests cannot allocate.
- Optional crossed LA checkpoints are retained only when headroom permits. At an
  exact live boundary with retained headroom, the optional publication set stays
  private so the next window can reuse that live row. Earlier checkpoints remain
  usable; this deliberately trades cache coverage for guaranteed continuation.
- Draft alignment metadata no longer selects verifier paging: scheduler routing
  uses the pipeline's sampler validation mode. A rejected independent hybrid draft
  restores a prompt predecessor (or starts at zero without prefix caching) and
  recomputes accepted suffix tokens instead of relabeling recurrent state.
- Explicit LA limits survive byte-budget normalization; prefix pools can be
  pre-sized, and admission includes the extra published-base continuation row.
- Dynamic candidate counts, parallel returns, non-greedy prefix verification and
  embedding-input prefix verification retain their guards. MTP stays separate.

Validation on the rebuilt native library:

- 755 selected C++ tests passed; CSV-backed model/cache-routing suites excluded.
- 18 isolated cache tests passed, including host-allocation failure sweeps,
  no-allocation promotion, stale/abandoned preparation and forced draft rewind.
- 17 local LFM2.5 Python cases passed. The 16-case verifier matrix covers prefix
  on/off, both split modes, candidate counts 1/4, a hard 12-row LA ceiling, repeated
  and extended prompts, cancellation and reuse. Prefix-enabled warm/extended calls
  require observed hybrid restore events; independent drafts require nonzero
  drafted and accepted tokens. All outputs match a prefix-off greedy reference.
- The tight six-row C++ case covers acceptance depths 1/2/3/4, exact-boundary
  publication and next-window admission while a competitor owns the last free row.
- C++ diagnostics are clean. Python diagnostics outside the changed test remain.

Next milestone: Qwen3.5 text MTP adapter and observed-restore policy experiment,
then mandatory Qwen3.5 VLM/image/video acceptance. Endpoint-logits caching is not
part of this work. No commit, push, or branch change was performed.

### Previous Publication Handoff

User direction: finish LFM common-foundation support first, then validate Qwen3.5
through the VLM pipeline with media. Existing LFM smoke/parity cases do not establish
full prefix-verification support. Prefix+VERIFY guards remain until P4/P5 gates pass.

Prepared scratch eviction is committed as `74c622c38`:
`Prepare atomic checkpoint eviction for scratch reservations`. Hooks passed; unrelated
worktree changes were preserved.

The next uncommitted slice makes completed-boundary publication allocation-safe as
one represented set per cache manager. Registry entries and content-length counts
are staged in local map nodes before any row becomes published. Node transfer and
row flags are applied only after preparation succeeds. A failure leaves the entire
new set unpublished; an existing canonical owner is retained. Scheduler and common
orchestrator publication use the batch API rather than publishing boundaries one by
one. This is not cross-cache atomic publication, optional physical-row retention,
or a change to legacy early registration of freshly allocated rows.

A one/two-layer allocation sweep covers two unpublished boundaries at 8 and 12:
every failed preparation keeps both unpublished and restore at 6, while success
restores at 12. A deterministic multi-candidate LRU regression also confirms that
scratch takes the oldest unowned checkpoint and leaves the newer checkpoint restorable.

Validation on September 14: 755 selected C++ tests, ten allocation-failure tests and
five LFM Python regressions passed using the rebuilt native library. CSV-backed model
suites were excluded. Scoped diagnostics and whitespace checks passed. One independent
review found no actionable defect in publication preparation. Whole-set failure
coverage was added after the review's test-gap check.

Next: retained next-window headroom and optional no-gap checkpoint retention, followed
by pinned atomic KV+LA restore/rewind and observed-restore LFM strategy tests before
guard removal. Duplicate-owner eviction still needs dedicated coverage. No Qwen3.5
VLM/media validation was attempted in this slice.

### Previous Eviction Handoff (2026-09-11)

The first P4 capacity slice is now committed as `f06a67b11`:
`Bound prefix linear-attention capacity and defer blocked admission`.

The conservative raw-only reservation-admission correction is committed as
`119c6a40c`: `Align scratch reservation admission with writable row capacity`.
Its validation passed 753 selected C++ tests, six allocation-failure tests and five
LFM Python regressions.

The next slice, prepared checkpoint eviction for scratch, is implemented and
uncommitted. Admission and both acquisition paths now count reclaimable unowned
checkpoints because the batch allocator can acquire them. Fresh rows are preferred;
any remaining need selects unowned checkpoints by LRU. All allocation and row
validation precedes mutation. Eviction removes owned hash registrations and their
content-length metadata. Referenced rows remain protected. Zero-row and oversized
reservations are rejected.

One/two-layer regressions cover fresh-only preservation, competing continuation,
reclaim under a hard ceiling, and no stale restore after release. An allocation-failure
sweep checks that every failed preparation preserves capacity, references and cached
lookup; allocation-disabled release succeeds after eviction. This is per-reservation
preparation atomicity, not rollback of successful evictions when a later group
reservation fails. Released scratch stays unpublished rather than resurrecting an
evicted checkpoint.

Validation: 754 selected C++ tests, eight allocation-failure tests and five LFM Python
regressions passed against the rebuilt native library. CSV-backed model suites were
excluded. Scoped editor diagnostics were clean. One independent review was completed;
its alleged apply allocations were ruled out by reserved vectors, nonallocating
same-allocator list splices and integer-only reference increments. Multi-candidate
LRU and duplicate-owner eviction remain dedicated coverage gaps.

Retained next-window headroom, optional publication/canonicalization, pinned atomic
restore and rewind remain P4 gates. Prefix verification is still guarded. Final
Qwen3.5 MTP acceptance still requires VLM/media; the local LFM tests do not cover it.

### Previous Capacity Handoff

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
