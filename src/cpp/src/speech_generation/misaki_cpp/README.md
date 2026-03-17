# Misaki C++ (Embedded in OpenVINO GenAI)

This directory contains the embedded C++ Misaki G2P implementation used by the OpenVINO GenAI Kokoro speech backend.

## Project Layout

- `CMakeLists.txt` — standalone component build/install config
- `include/misaki/g2p.hpp` — public C++ API (`G2P`, `make_engine`)
- `src/` — English + espeak backends and dispatch
- `python/misaki_cpp_py.cpp` — optional Python bindings for parity runs
- `samples/` — small usage examples
- `tools/run_parity_dataset_misaki_cpp_vs_python.py` — primary parity benchmark
- `data/` — required English lexicon data (`us_*`, `gb_*`)

## Required Lexicon Data

English backend requires all of:
- `us_gold.json`
- `us_silver.json`
- `gb_gold.json`
- `gb_silver.json`

If auto-discovery is not enough, pass lexicon root via:
- env var: `MISAKI_DATA_DIR`
- Python binding API: `engine.set_lexicon_data_root(...)`
- parity CLI: `--lexicon-data-root <path>`

## E2E Flow: Build Bindings + Run Parity

### 1) Build Python bindings

From workspace root:

```powershell
cmake -S openvino.genai/src/cpp/src/speech_generation/misaki_cpp -B misaki_cpp-build -DMISAKI_CPP_BUILD_PYTHON_BINDINGS=ON
cmake --build misaki_cpp-build --config Release --target misaki_cpp_py
```

### 2) Expose module to Python

Use `PYTHONPATH` pointing at the built module output folder (platform/build dependent), then verify import:

```powershell
python -c "import misaki_cpp_py; print('misaki_cpp_py import: OK')"
```

### 3) Run dataset parity benchmark (recommended)

Single dataset baseline:

```powershell
python openvino.genai/src/cpp/src/speech_generation/misaki_cpp/tools/run_parity_dataset_misaki_cpp_vs_python.py --profile single --variant en-us --max-items 300 --lexicon-data-root "<path-to-lexicon-data>"
```

Curated realistic profiles:

```powershell
python openvino.genai/src/cpp/src/speech_generation/misaki_cpp/tools/run_parity_dataset_misaki_cpp_vs_python.py --profile realistic --variant en-us --max-items 300 --lexicon-data-root "<path-to-lexicon-data>"
python openvino.genai/src/cpp/src/speech_generation/misaki_cpp/tools/run_parity_dataset_misaki_cpp_vs_python.py --profile mixed --variant en-us --max-items 300 --lexicon-data-root "<path-to-lexicon-data>"
python openvino.genai/src/cpp/src/speech_generation/misaki_cpp/tools/run_parity_dataset_misaki_cpp_vs_python.py --profile chatty --variant en-us --max-items 300 --lexicon-data-root "<path-to-lexicon-data>"
python openvino.genai/src/cpp/src/speech_generation/misaki_cpp/tools/run_parity_dataset_misaki_cpp_vs_python.py --profile adversarial --variant en-us --max-items 300 --lexicon-data-root "<path-to-lexicon-data>"
```

For `en-gb`, switch `--variant en-gb`.

## Expected Results / How to Judge Runs

The benchmark prints:
- `Exact matches`
- `Average ratio`
- `Minimum ratio`
- `Token Exact Match`
- `Avg Token Ratio`
- per-dataset summaries

Interpretation guidance:
- `Average ratio` and `Avg Token Ratio` should stay close to your established branch baseline.
- `Token Exact Match` is usually more stable for regression tracking than strict full-string exactness.
- `Minimum ratio` is useful for spotting catastrophic outliers; inspect worst mismatches printed at the end.
- If `Skipped prompts` rises unexpectedly, treat as a reliability regression.

Recommended regression practice:
- keep one or more baseline runs (same profile/seed/max-items),
- compare deltas rather than enforcing one universal absolute threshold,
- investigate when any metric drifts materially from baseline.

## Helpful Options

`run_parity_dataset_misaki_cpp_vs_python.py` supports:
- `--profile single|realistic|chatty|adversarial|mixed`
- `--variant en-us|en-gb`
- `--max-items`, `--seed`, `--min-chars`
- `--normalize-input-escapes`
- `--show-diffs`
- `--analyze-normalization`
- `--analyze-categories` and `--analyze-example-limit`

## Minimal Python Binding Usage

```python
import misaki_cpp_py

engine = misaki_cpp_py.Engine("en", "en-us")
phonemes = engine.phonemize("[Kokoro](/kˈOkəɹO/) is a model.")
result = engine.phonemize_with_tokens("Hello world")
```
