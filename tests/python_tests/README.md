# OpenVINO™ GenAI Tests

This tests aim to validate support for vanilla and continuous batching GenAI APIs.

## Setup environment

In order to run tests first of all build or install OpenVINO GenAI library, follow instructions [GenAI Library README](../../src/README.md).

Then install requirements for tests:
```sh
pip install -r tests/python_tests/requirements.txt
```

## Run Tests

```sh
python -m pytest tests/python_tests/
```

If you have built GenAI library by yourself instead of using wheel please set `PYTHONPATH` so that test could find library, e.g.
```sh
PYTHONPATH=$PYTHONPATH:.../openvino.genai/build-Release/ python -m pytest tests/python_tests/
```

## Customize tests run

Tests have different sets of models for different purposes. If you wish to run specific tests, you can use `-k` option, for example to run only multibatch and chat tests:
```sh
python -m pytest tests/python_tests/ -k "test_multibatch and test_chat"
```

If you wish to run all tests except beam search do the following:
```sh
python -m pytest tests/python_tests/ -k "not test_beam_search"
```

Argument `--model_ids` can be used to run tests selectively only for specific models. HF model ids should be separated by space, e.g:
```sh
python -m pytest tests/python_tests/ -k "test_multibatch" --model_ids "TinyLlama/TinyLlama-1.1B-Chat-v1.0 Qwen/Qwen2-0.5B-Instruct"
```

List of currently supported models can be found in tests/python_tests/data/models.py:get_models_list

## GGUF Frontend Tests

The Linux precommit GGUF jobs include a tiny-model suite comparing the frontend with
the regular Optimum export path:

```sh
python -m pip install transformers==5.0.0
python -m pytest tests/python_tests/test_gguf_frontend.py -v
```

It downloads pinned Llama, Qwen3, Phi-3 and Gemma-3 GGUFs (about 79 MB total), plus
their source tokenizer/configuration files, into the Hugging Face cache. The reference
is exported through Optimum from the same GGUF weights. No model binaries are stored
in this repository; reference IRs and serialization outputs use pytest temporary directories.
Tests cover PA, SDPA and automatic backend selection, exact generation comparisons,
batched prompts, beam search, encoded inputs, tokenizer whitespace, streaming and saved-IR reloads. Batch and beam tests
use PA/default; native SDPA attention masks currently support batch one. The Optimum
Gemma-3 reference requires Transformers 5.0, matching the regular LLM job. The existing reader
suite separately retains its model/quantization and native-tokenizer coverage.

This suite validates GenAI integration rather than standalone frontend translation.
OpenVINO frontend tests should check graph conversion, operators and metadata handling
in the OpenVINO repository. These tests additionally exercise GenAI's reader selection,
metadata-to-tokenizer construction, attention backend selection, generation APIs,
streaming and serialization/reload behavior.

For each pinned model, the suite:

1. Downloads the selected GGUF and the matching source tokenizer/configuration files.
2. Reconstructs a Hugging Face model from those GGUF weights and checks that no weights
   are missing, unexpected or mismatched.
3. Exports that reconstructed model and tokenizer to reference IRs with Optimum Intel.
4. Loads the original GGUF through GenAI with `GGUF_READER="FRONTEND"` for SDPA, PA and
   automatic pipeline selection, using the same reference inference settings.
5. Compares tokenizer IDs and decoded text, greedy generation, batches, beam tokens and
   scores, encoded inputs and streamed output against the reference path. It also checks
   the standalone GGUF tokenizer.
6. Saves model/tokenizer IRs beside a temporary copy of the GGUF, reloads them through
   the regular IR path, and checks that generation is unchanged.

The same precommit jobs also run generated-model architecture tests:

```sh
python -m pip install cmake==3.26.4
python -m pytest tests/python_tests/test_gguf_generated.py -v
```

These tests fetch a SHA-256-verified source archive of llama.cpp at
`03fa73cb27f5c251b9528489b18d303b1366aca4` and build only its CPU generator and a
reference runner (CMake >=3.24 and a C++17 compiler are required). Both use the same
revision. `generate.cpp` reuses the upstream `test-llama-archs` factory for architecture
metadata, tensor shapes and seeded random weights. It removes unrelated metadata and
optional bias/RoPE/quantization-scale tensors, retains required biases, and centers
normalization weights around one (Gemma applies its own offset). MoE fixtures select
two of four experts. The runner reloads the finalized GGUFs and independently produces
greedy token IDs and log probabilities using llama.cpp CPU, with OpenVINO disabled in
that build. Both sides use F32 inference/KV caches and no dynamic quantization.
GenAI checks prefill/decode with SDPA, PA and automatic selection, plus concurrent
PA requests with chunked prefill. Inputs include a sliding-window-boundary case.
The fixtures have no text vocabulary, so an auxiliary tokenizer only satisfies the
pipeline constructor; comparisons use token IDs. The Optimum suite above separately
checks real tokenizer behavior.

Build outputs, GGUFs and references are generated in pytest temporary directories,
not committed or downloaded as model binaries. For repeated local runs,
`GGUF_LLAMA_BUILD_DIR=/absolute/path` reuses the reference build, rebuilding as needed.
A local run took 87 seconds including the clean four-job CPU build (about 79 seconds),
and about 10 seconds with that build cached. The combined Optimum/generated suite took
117 seconds with build/download caches warm; CI timings depend on the runner.
On the Linux precommit runner for [PR #4318 at `af609b39`](https://github.com/openvinotoolkit/openvino.genai/actions/runs/38698456185/job/116311682631),
the combined suites took 295 seconds in pytest (189 passed, 36 skipped and 10 xfailed).
The full job took about 9 minutes including setup, dependency installation and cleanup;
the preceding run took about 10 minutes. The job has a 30-minute timeout. These timings
precede the standalone-tokenizer case added here and can vary with download caches and
runner load.

The 21 model variants cover 18 architecture IDs: BailingMoE2, ERNIE4.5-MoE, EXAONE4,
Gemma, Gemma2, GPT-OSS, Hunyuan-Dense, Llama, Maincoder, MiniCPM, Mistral3, OLMoE,
Phi3, Qwen2, Qwen3, Qwen3.5, Qwen3MoE and SmolLM3. Llama, MiniCPM and Mistral3 have
separate dense and MoE cases. Each two-layer F32 model is about 4-12 MiB.
The architecture list is explicit: missing exports fail setup rather than silently
skipping coverage. The pinned factory/saver cannot export Gemma3, Gemma4,
DeepSeek2-OCR, Mellum or Muse-Glimmer; Gemma3 remains covered by the Optimum suite.

Known limitations are isolated from the passing cases:

| Cases | Status |
| --- | --- |
| Qwen3.5 PA/default/continuous batching | Skipped: recurrent state needs SDPA |
| Gemma2 explicit PA/continuous batching | Strict xfail: soft-capped attention has no SDPA node for PA conversion; automatic fallback passes |
| GPT-OSS SDPA/PA/default/continuous batching | Strict xfail: first-token mismatch against llama.cpp CPU |

Expected failures still execute and become failures on unexpected passes, so fixes
require removing the corresponding markers. Generator/build/reference failures are
never xfailed. Run with `--runxfail` to reproduce the underlying failures.

Real-model comparisons with llama.cpp remain opt-in and require the WWB GGUF dependencies:

```sh
WWB_GGUF_TESTS=1 python -m pytest tools/who_what_benchmark/tests/test_cli_text_gguf.py -m gguf_small -v
```

Omit `-m gguf_small` to include the larger checkpoints. Tiny-model coverage is not a
claim that all frontend architectures or production quantization formats are tested.

### Adding GGUF model coverage

When adding a new architecture or fixing its integration:

1. Add conversion/operator/metadata regressions in the OpenVINO frontend tests for the
   frontend change itself.
2. Add an explicit case to `test_gguf_generated.py:ARCHITECTURES` / `MODEL_CASES` when the
   pinned llama.cpp factory supports it. For a newly supported factory, update its pinned
   revision and source hash together and verify the generator and reference runner.
   Run every applicable SDPA, PA, automatic and continuous-batching case against the
   independent llama.cpp reference; include dense/MoE variants where applicable.
3. For tokenizer or end-to-end integration coverage, add a small GGUF to
   `test_gguf_frontend.py:TINY_GGUF_MODELS`, pin both the GGUF repository and the matching
   source tokenizer/configuration revision, and confirm the reference model is rebuilt
   from the same GGUF weights. This runs the tokenizer, generation, streaming and saved-IR
   checks above. Prefer small fixtures that keep precommit downloads and runtime bounded.
4. For production weights or quantization coverage, add a real-checkpoint case to
   `tools/who_what_benchmark/tests/test_cli_text_gguf.py:GGUF_MODELS` and run the opt-in
   llama.cpp comparison. Mark it `gguf_small` only when the download is under about 1 GB.
5. Document any unsupported case with a specific skip reason or strict xfail, and remove
   the marker once fixed. Keep setup, generator and reference failures visible.

Run both precommit suites after updating coverage and record pytest counts and elapsed
runtime. A new architecture is not fully covered by a tokenizer-only or single-backend run.

## Test Samples
To test samples, set the `SAMPLES_PY_DIR` and `SAMPLES_CPP_DIR` environment variables to the directories containing your Python samples and built C++ samples respectively. The `SAMPLES_CPP_DIR` should point to the folder with built C++ samples, which can be installed using `cmake --component samples_bin`. For example:
```sh
SAMPLES_PY_DIR=openvino.genai/samples/python SAMPLES_CPP_DIR=openvino.genai/samples_bin python -m pytest tests/python_tests -m samples
```

You can also use markers such as `llm` and `whisper` to run specific sets of tests. For example, to run only the `llm` tests:
```sh
python -m pytest tests/python_tests/samples -m llm
```

Or to run only the `whisper` tests:
```sh
python -m pytest tests/python_tests/samples -m whisper
```

If the `OV_CACHE` environment variable is set, all downloaded and converted models will be saved to the specified directory. This allows the models to be reused between runs, saving time and resources. For example:
```sh
OV_CACHE=$HOME/ov_cache python -m pytest tests/python_tests -m samples
```

If the `CLEANUP_CACHE` environment variable is set, all downloaded and converted models will be removed right after the tests have stopped using them. Note that this does not affect the HuggingFace (HF) cache. For example:
```sh
CLEANUP_CACHE=1 python -m pytest tests/python_tests -m samples
```

Test images are saved to pytest's default cache dir. It can be changed with `--override-ini cache_dir=new_path`. `-p no:cacheprovider` disables the cache.

## GGUF multimodal acceptance

`test_gguf_mmproj_acceptance.py` runs pinned Gemma3, Gemma4, Muse and Qwen3.5
checkpoint conversions against the CPU-only llama.cpp oracle built by
`gguf_reference`. The OpenVINO Ubuntu 22 workflow calls this harness with its
own build artifacts and a pinned companion GenAI commit. Its separate family
steps select compatible Transformers versions and retain JUnit and modality
reports. The harness uses its own fixtures; `--noconftest` avoids initializing
unrelated pipeline test fixtures.

For a focused local image/chat/API check after installing the test dependencies,
CMake >=3.24, a C++17 compiler, sentencepiece and ffmpeg:

```sh
python -m pip install transformers==5.0.0
python -m pytest --noconftest tests/python_tests/test_gguf_mmproj_acceptance.py \
  -k 'gemma3 and square and SDPA' -v
```

Set `GGUF_MMPROJ_REPORT_DIR` to retain JSON reports and oracle logs and
`GGUF_RUNTIME_MANIFEST` to attach exact runtime source/build provenance. The
harness requires every requested modality/API case and checks greedy choices
on the same token history. Tiny fixtures establish numerical agreement, not
text quality. Real quantized acceptance expands the pinned Qwen3.5 checkpoint
to represented F16 and requantizes it to pure Q4_0, recording tensor inventories
and hashes. This creates a new fixture and does not establish parity for the
original mixed-weight checkpoint. Q4_K_M accuracy loss remains a plugin gap.
Optimum directory/map loading is checked separately from llama.cpp agreement.
The tiny Qwen fixture uses explicit RoPE sections `[1, 1, 0]` scaled to its head
size, and omits MTP export; the converter's production defaults do not fit it.
Muse materializes the tied output tensor for the reference loader. These fixture
adaptations are recorded in the source manifest and apply to both engines.
