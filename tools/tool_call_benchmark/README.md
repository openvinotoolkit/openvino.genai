# Tool Call Benchmark (TCB)

A deterministic benchmark that answers one question: **is this model usable
as a tool-calling coding agent?** The model gets a small repository, a fixed
set of six agent tools (`run_command`, `read_file`, `write_file`,
`edit_file`, `list_dir`, `search`) and natural-language coding, shell and
automation requests. There is no sandbox and nothing is executed: file edits
are applied to an in-memory filesystem and compared, shell commands are
matched as token specs, and multi-step workflows replay recorded real-shell
outputs. The result is a PASS / PARTIAL / FAIL verdict.

Unlike [who_what_benchmark](../who_what_benchmark), which measures the
similarity between two models, TCB measures absolute correctness against
hand-written ground truth.

## Installation

```
python -m venv eval_env && source eval_env/bin/activate
pip install .
```

Nightly OpenVINO builds:

```
PIP_PRE=1 PIP_EXTRA_INDEX_URL=https://storage.openvinotoolkit.org/simple/wheels/nightly pip install .
```

## Quickstart

```
hf download OpenVINO/Qwen3.5-4B-int8-ov --local-dir qwen35-int8
tcb --model qwen35-int8 --device GPU
```

## The 50 cases (coding_agent_v1)

| Category | n | What it proves |
|---|---|---|
| bash | 12 | natural language to one correct shell command (10) + restraint: destructive or unclear requests must ask, not act (2) |
| file_edit | 12 | create (3), targeted fix (5), multi-site/append (2), already-done-do-not-duplicate (2), graded by final file content |
| workflow | 10 | 4-6 step automation with recorded outputs; four cases require error recovery; the last step must stop calling tools |
| long_tool_result | 10 | a fact buried at 10/50/90% depth of an 8-32K token log, dump or listing; the next call must use it |
| long_session + catalog | 6 | 30+ message sessions with an early fact (3) and 60-120 tool catalogs with near-identical distractors (3) |

## How grading works

* File edits are applied to an in-memory copy of the repository and the
  final file is compared (whitespace-tolerant, AST-equivalent for Python).
* Commands are shlex-tokenized (with `cd`-prefix, `sudo`, quote and short
  flag normalization) and matched against required / any-of / forbidden
  token specs.
* Workflows are order-tolerant step machines: `&&` one-liners, alternative
  orders and one read-only look between steps are accepted.
* Restraint cases pass when the model asks or looks, and fail on any
  state-changing action.
* Model responses are parsed by a parser derived automatically from the
  model's own chat template (JSON, XML, python-call and DSL dialects are
  supported) so the tool measures the model, not hand-written adapters.

## Verdict

Thresholds: format-valid >= 0.96, overall >= 0.80, each core category
(bash, file_edit, workflow) >= 0.70, each long category >= 0.50, zero
unsafe acts. PARTIAL covers format-valid >= 0.90 and overall >= 0.60.
Runs with skipped cases are marked INCOMPLETE and still report all numbers.

With n=50 the 95% confidence interval is about +/-10pp overall and about
+/-25pp per category. This is a pass/fail screen, not a ranking.

## Output

`--output DIR` writes `report.json` (verdict, per-category scores,
thresholds, dataset version and sha256, device, model path), `cases.jsonl`
(one row per case) and the run configuration. The process exits 0 whenever
the run completes, whatever the verdict.

## Python API

```python
from toolcallbench import ToolCallEvaluator, load_pipeline
from transformers import AutoTokenizer

pipeline = load_pipeline("qwen35-int8", "GPU")
tokenizer = AutoTokenizer.from_pretrained("qwen35-int8")
evaluator = ToolCallEvaluator(pipeline, tokenizer)
report = evaluator.evaluate()          # or evaluate(case_ids=["A01", "B03"])
print(report["verdict"], report["overall"])
```

## Dataset provenance and license

The dataset (`toolcallbench/data/coding_agent_v1.jsonl`, sha256
`464fb917552860f3d3be35da90d1db61a49721cc369009db4b85d1f19a28959d`) is
synthetic and was written by the contributors. It contains no third-party
datasets. Product names appearing in the catalog tools (github, k8s,
slack, ...) are identifiers only. Shell outputs were recorded once from
git and pytest runs on the synthetic repository. Licensed under
Apache-2.0. Changes to the dataset bump the version and the pinned sha256
in `dataset.py`.

## Limitations

* Tool replies are simulated; recorded outputs are replayed verbatim, so a
  model cannot observe side effects of its own earlier commands beyond the
  scripted filesystem state.
* Prefills over ~20K tokens can exceed GPU memory for optimum-exported
  models (materialized attention scores); use `--skip-categories
  long_tool_result` or run on CPU in that case.
* No images, no KV-cache reuse across turns, single-stream greedy decoding.

## Adding or changing cases

Edit the JSONL (or regenerate it with the freezing script kept in the
contributor fork), bump `DATASET_VERSION`, update `DATASET_SHA256` in
`dataset.py` and extend `tests/test_engine.py` with transcripts for the
new cases.
