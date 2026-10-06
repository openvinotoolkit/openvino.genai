# Continuous batching trace smoke test

`continuous_batching_benchmark` accepts a ShareGPT-style JSON array. Each item
must have `conversations[0].value` (the prompt) and
`conversations[1].value` (used to size `max_new_tokens`, not as a generated
answer). Entries with fewer than four input or answer tokens are skipped. The
included `sample_prompts.json` contains four short requests. Use
`--sequential_dataset` flag to submit them in file order; without it the
benchmark samples with replacement as before.

From the `openvino.genai.fork` repository root in PowerShell, after configuring
and building with `setup_RelWithDebInfo.bat` and `build_RelWithDebInfo.bat`:

```powershell
$env:PATH = "D:\repos\openvino_2026_0\install\openvino\runtime\bin\intel64\RelWithDebInfo;D:\repos\openvino_2026_0\openvino.genai.fork\buildRelWithDebInfo\openvino_genai;D:\repos\openvino_2026_0\openvino.genai.fork\buildRelWithDebInfo\bin\RelWithDebInfo;$env:PATH"
.\buildRelWithDebInfo\tools\continuous_batching\benchmark\RelWithDebInfo\continuous_batching_benchmark.exe --model D:\models\Qwen2.5-0.5B-Instruct --dataset .\tools\continuous_batching\benchmark\sample_prompts.json --num_prompts 4 --sequential_dataset --max_batch_size 64 --max_output_len 32 --request_rate inf --cache_size 1 --device CPU
```

Change `--model` to the directory containing your exported language model and
tokenizer. `--request_rate inf` queues all requests before stepping the engine,
which makes overlapping continuous batching requests easy to observe. The
benchmark prints throughput and TTFT/TPOT, but does not emit a trace by itself.

Prefix caching remains off by default. Add `--enable_prefix_caching` to opt in.
Requests can share cached KV blocks only when their tokenized prompts have an
identical prefix long enough to contain complete cache blocks; `--cache_size`
controls the available cache capacity, not whether prefix caching is enabled.

To capture `genai.cb.*` ITT regions, configure and rebuild the fork with:

```powershell
cmake -S . -B buildRelWithDebInfo -DENABLE_PROFILING_ITT=ON
cmake --build buildRelWithDebInfo --config RelWithDebInfo --target continuous_batching_benchmark -j8
```

Verify the generated build enables `ENABLE_PROFILING_ITT`, then record the
benchmark process with an ITT-capable collector such as Intel VTune or
`ut-tool-ext`. Export a Chrome/Perfetto trace JSON. The report uses ITT events
directly.

From the fork root, export the trace to a temporary location and convert it:

```powershell
$trace = Join-Path $env:TEMP 'cb.json'
$analysis = Join-Path $env:TEMP 'analysis.json'
python .\tools\continuous_batching\perfetto_to_analysis.py $trace --out $analysis --model Qwen2.5-0.5B-Instruct --device CPU
Start-Process .\tools\continuous_batching\trace_viz.html
```

Load the temporary `analysis.json` with the HTML page's file picker. The
converter expects `genai.cb.step` spans and associated metadata events for step
counts, per-sequence scheduling, request completion, and KV-block lifecycle.
Check that the output reports nonzero steps and sequences, and that memory
snapshots are present when KV-block events were captured.

If conversion reports CB events but no complete `genai.cb.step` span, the trace
is incomplete for report reconstruction. Start capture before launching the
benchmark and stop only after it prints `Benchmark finished`; do not filter the
`ov.genai` events during export.