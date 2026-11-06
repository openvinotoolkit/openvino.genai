# [Bug] Stateful (SDPA) pipeline: batched `generate()` corrupts KV cache after a request finishes early

## Describe the bug

When `LLMPipeline` runs on the stateful (SDPA) backend with a batch of prompts, the per-request
KV-cache row offsets (`beam_offets`) are recomputed with a wrong key once a request that is not the
last one in the batch finishes. On the following decode steps, the remaining requests reorder the
KV cache via `beam_idx` using wrong rows: they either read another request's KV cache or index past
the end of the (shrunk) batch. Generation continues without any error, but the output of the
affected requests becomes wrong / garbage.

## Root cause

`src/cpp/src/lm_encoding.cpp` (in `get_lm_encoded_results`):

```cpp
std::map<size_t, size_t> beam_offets;                  // key = request_id
for (size_t i = 0; i < sequence_groups.size(); i++)
    beam_offets.insert({sequence_groups.at(i)->get_request_id(), i});
...
for (size_t i = 0; i < active_sequence_groups.size(); i++) {
    beam_offets[active_sequence_groups.at(i)->get_request_id()] =
        i == 0 ? 0 : (active_sequence_groups.at(i - 1)->num_running_seqs() + beam_offets[i - 1]);
    //                                                                               ^^^^^^^^^^^^^^^
    //                         indexed by *position* i - 1, but the map is keyed by *request_id*
}
```

`beam_offets` is keyed by `request_id`, but the recurrence reads `beam_offets[i - 1]` where `i - 1`
is the position in `active_sequence_groups`. Finished requests are erased from
`active_sequence_groups` (`free_non_running_requests`), so after the first early finish the
position no longer equals the request id, and a stale offset of an already finished request is
used.

### Worked example

Batch `[req0, req1, req2]`, greedy, `req1` hits EOS / a stop condition first.

| Step | Active groups | Offsets computed | Correct offsets |
|------|---------------|------------------|-----------------|
| t    | `[0, 1, 2]`   | `{0:0, 1:1, 2:2}` | `{0:0, 1:1, 2:2}` |
| t+1  | `[0, 2]`      | `{0:0, 2: running(req0) + offsets[1] = 1 + 1 = 2}` | `{0:0, 2:1}` |
| t+2  | `[0, 2]`      | `req2` gathers KV row **2** from a **2-row** cache → out of range | row 1 |

With 4+ prompts (e.g. `req0` finishes first while `req1..req3` continue) the wrong offset can point
at a valid row of a *different* request, so the request silently continues from another prompt's
KV cache. Beam search (`num_beams > 1`) is affected the same way.

## Affected code paths

`get_lm_encoded_results` is used by:

- `StatefulLLMPipeline` (`src/cpp/src/llm/pipeline_stateful.cpp`), i.e. `LLMPipeline` with
  `ATTENTION_BACKEND="SDPA"`, on platforms without PagedAttention support, or when the PA backend
  falls back to stateful.
- `VLMPipeline` (`src/cpp/src/visual_language/pipeline.cpp`), when used with batch > 1.

The continuous batching / PA backend is **not** affected (it uses a different code path).

## To reproduce

```bash
optimum-cli export openvino --model Qwen/Qwen2.5-0.5B-Instruct --weight-format fp16 qwen2.5-0.5b-ov
```

```python
import itertools
import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline("qwen2.5-0.5b-ov", "CPU", ATTENTION_BACKEND="SDPA")

prompts = [
    "Write a long story about a dragon who",
    "1 + 1 =",                                   # expected to stop early
    "Explain in detail how a car engine works:",
    "List ten facts about the Moon:",
]

config = ov_genai.GenerationConfig()
config.max_new_tokens = 64
config.stop_strings = {"\n"}                     # makes requests finish at different steps
config.include_stop_str_in_output = False

reference = {p: pipe.generate(p, config) for p in prompts}

for order in itertools.permutations(prompts, 3):
    batch = pipe.generate(list(order), config).texts
    for prompt, text in zip(order, batch):
        if text != reference[prompt]:
            print(f"MISMATCH for order={order}\n  prompt:     {prompt!r}\n"
                  f"  batched:    {text!r}\n  individual: {reference[prompt]!r}\n")
```

## Expected behavior

Each request in a batch produces the same output as when generated individually (greedy),
regardless of the order in which requests in the batch finish.

## Actual behavior

`"1 + 1 ="` stops after 8 tokens while the other prompts keep generating. From the second decode
step on, `req2` gathers KV row 2 out of a 2-row cache, so its output becomes garbage (note the
`S?_%?^` bytes read from another request's KV cache):

```
MISMATCH order=('List ten facts about the Moon:', 'Explain in detail how a car engine works:', '1 + 1 =')
  prompt:     '1 + 1 ='
  batched:    '\xef\xbf\xbd_\xef\xbf\xbd`S\xef\xbf\xbd_%\xef\xbf\xbd^\xef\xbf\xbd`\xef\xbf\xbd,"\xef\xbf\xbd3...'
  individual: '1 + 1 = 2'
```

With a 4-prompt batch whose second request finishes first, 24 of 24 batch orders produce wrong
output for the remaining requests. Measured on `Qwen/Qwen2-0.5B-Instruct-int8-ov`:

| Backend | mismatching outputs (96 outputs over 24 batch orders) |
|---------|------------------------------------------------------|
| SDPA    | 30 (garbage / wrong content)                         |
| PA      | 0                                                    |

## Environment

- OpenVINO GenAI version: 2026.5.0.dev (master @ `8f92495`)
- OpenVINO version: 2026.5.0.dev
- OS: Windows 11
- Device: CPU

## Proposed fix

Look up the previous group's offset by its request id instead of by position:

```cpp
for (size_t i = 0; i < active_sequence_groups.size(); i++) {
    beam_offets[active_sequence_groups.at(i)->get_request_id()] =
        i == 0 ? 0
               : active_sequence_groups.at(i - 1)->num_running_seqs() +
                 beam_offets.at(active_sequence_groups.at(i - 1)->get_request_id());
}
```

## Why existing tests don't catch it

`test_batch_string_inputs` (`tests/python_tests/test_llm_pipeline.py`) uses at most 3 prompts on
tiny random models that almost never finish before `max_new_tokens`, so all requests finish on the
same step and the stale-offset branch is never exercised. A regression test should force requests
to finish at different steps (e.g. via `stop_strings` / `stop_token_ids`) with the middle request
finishing first, and compare batched output against individual generation on
`PipelineType.STATEFUL`.
