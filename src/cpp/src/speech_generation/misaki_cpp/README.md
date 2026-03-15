# Misaki C++ (Embedded in OpenVINO GenAI)

This directory contains the embedded C++ Misaki G2P implementation used by the OpenVINO GenAI Kokoro speech backend.

Current status:
- English backend (`en-us`, `en-gb`) is implemented with corpus-driven parity coverage and exactness tests.
- Non-English C++ backends are intentionally out of scope here.

## Project Layout

- `CMakeLists.txt` — component build/test configuration
- `include/misaki/g2p.hpp` — public C++ API (`G2P`, `make_engine`)
- `src/`
  - `en_g2p.cpp` — English backend logic
  - `english_lexicon.cpp` / `english_lexicon.hpp` — English lexicon loading
  - `factory_stub.cpp` — top-level language dispatch
- `tests/`
  - `test_english.cpp` — focused English behavior tests
  - `test_english_exactness.cpp` — English corpus exactness
  - `english_golden_cases.jsonl` / `english_gb_golden_cases.jsonl` — generated corpora used by tests
- `samples/`
  - `basic_usage.cpp` — minimal API usage + token output
  - `fallback_and_unknown.cpp` — fallback hook + unknown-token behavior
- `tools/gen_golden.py` — Python golden corpus generator
- `tools/run_parity_dataset_misaki_cpp_vs_python.py` — dataset-driven parity runner (`misaki_cpp` vs Python `misaki`)
- `data/`
  - English lexicon data (`us_*.json`, `gb_*.json`)

## Required Lexical Data Files

The English C++ backend requires the following lexicon JSON files at runtime:

- `us_gold.json`
- `us_silver.json`
- `gb_gold.json`
- `gb_silver.json`

These files are required for both `en-us` and `en-gb` phonemization behavior.

You can obtain these files from the official Misaki repository: https://github.com/hexgrad/misaki/tree/main/misaki/data

If needed, point the engine to the directory containing these files via `set_lexicon_data_root(...)`.

## Typical Workflow in OpenVINO GenAI

### Build OpenVINO GenAI (includes embedded Misaki)

From `openvino.genai/` root:

```powershell
cmake -S . -B ../openvino.genai-build
cmake --build ../openvino.genai-build --config Release --target openvino_genai
```

If you only want to rebuild the embedded Misaki library target:

```powershell
cmake --build ../openvino.genai-build --config Release --target openvino_genai_misaki_cpp
```

For standalone `misaki_cpp` tests, set lexicon data root at configure time if `data/` is not present:

```powershell
cmake -S openvino.genai/src/cpp/src/speech_generation/misaki_cpp -B misaki_cpp-build -DMISAKI_ENGLISH_LEXICON_DATA_DIR="<path-to-lexicon-data>"
```

### Run tests

If your configured build enables these tests, run from build directory:

```powershell
ctest --test-dir ../openvino.genai-build --output-on-failure -R "English|misaki|speech"
```

For standalone `misaki_cpp` test builds (without bundled `data/`), provide lexicon data via either:

- `-DMISAKI_ENGLISH_LEXICON_DATA_DIR="<path>"` at CMake configure time, or
- `MISAKI_DATA_DIR=<path>` in the environment.

## Standalone Python Bindings (for G2P parity work)

To compare `misaki_cpp` directly against Python `misaki` without going through the OpenVINO GenAI speech pipeline,
build the optional `misaki_cpp_py` module.

From workspace root (standalone configure of embedded project):

```powershell
cmake -S openvino.genai/src/cpp/src/speech_generation/misaki_cpp -B misaki_cpp-build -DMISAKI_CPP_BUILD_PYTHON_BINDINGS=ON
cmake --build misaki_cpp-build --config Release --target misaki_cpp_py
```

Then point `PYTHONPATH` at the built module folder and import `misaki_cpp_py` in Python.

Minimal usage:

```python
import misaki_cpp_py

engine = misaki_cpp_py.Engine("en", "en-us")
phonemes = engine.phonemize("[Kokoro](/kˈOkəɹO/) is a model.")
result = engine.phonemize_with_tokens("Hello world")
```

If lexicon JSON files are not auto-located in your environment, set data root explicitly:

```python
engine.set_lexicon_data_root("/path/to/misaki_data")
```

## Regenerating Golden Corpora

Use this when refreshing corpus-driven parity cases.

From `openvino.genai/src/cpp/src/speech_generation/misaki_cpp/`:

```powershell
python tools/gen_golden.py --profile english --english-sample-size 200 --english-seed 1337 --out tests/english_golden_cases.jsonl
python tools/gen_golden.py --profile english-gb --english-sample-size 200 --english-seed 1337 --out tests/english_gb_golden_cases.jsonl
```

Supported profiles:
- `english`
- `english-gb`

## Dataset-driven parity check (`misaki_cpp` vs Python `misaki`)

Run the parity harness in single-dataset mode (default: `wikitext/wikitext-103-raw-v1`):

```powershell
python tools/run_parity_dataset_misaki_cpp_vs_python.py --profile single --variant en-us --max-items 300 --lexicon-data-root "<path-to-lexicon-data>"
```

Or use curated profiles for more realistic TTS-style prompts:

```powershell
python tools/run_parity_dataset_misaki_cpp_vs_python.py --profile realistic --variant en-us --max-items 300 --lexicon-data-root "<path-to-lexicon-data>"
python tools/run_parity_dataset_misaki_cpp_vs_python.py --profile mixed --variant en-us --max-items 300 --lexicon-data-root "<path-to-lexicon-data>"
python tools/run_parity_dataset_misaki_cpp_vs_python.py --profile chatty --variant en-us --max-items 300 --lexicon-data-root "<path-to-lexicon-data>"
python tools/run_parity_dataset_misaki_cpp_vs_python.py --profile adversarial --variant en-us --max-items 300 --lexicon-data-root "<path-to-lexicon-data>"
```

The preset profiles currently use text-only datasets (`xsum`, `ag_news`, `yelp_polarity`, `tweet_eval/sentiment`, `glue/sst2`, `wikitext`) to avoid audio/script loader dependencies.

Useful options:
- `--profile single|realistic|chatty|adversarial|mixed`
- `--dataset`, `--config`, `--split`, `--field`
- `--variant en-us|en-gb`
- `--lexicon-data-root`
- `--max-items`, `--seed`, `--min-chars`, `--normalize-input-escapes`, `--show-diffs`
- `--analyze-normalization` (strict/basic/loose/unknown-collapsed normalization stats)
- `--analyze-categories` and `--analyze-example-limit` (mismatch category summary + examples)

If your source text includes escaped Unicode literals (e.g. `\u2019`, `\u002c`), use:

```powershell
python tools/run_parity_dataset_misaki_cpp_vs_python.py --profile chatty --variant en-us --normalize-input-escapes
```

For deeper diagnostics in a single run:

```powershell
python tools/run_parity_dataset_misaki_cpp_vs_python.py --profile adversarial --variant en-us --analyze-normalization --analyze-categories
```

The parity output reports both:
- character-level metrics (`Exact matches`, `Average ratio`)
- token-level metrics (`Token Exact Match`, `Avg Token Ratio`)

## Public API (current)

```cpp
auto engine = misaki::make_engine("en", "en-us");
auto result = engine->phonemize_with_tokens("[Misaki](/misˈɑki/) is a G2P engine.");
auto phonemes = result.phonemes;
auto tokens = result.tokens;

// Optional unresolved-token fallback:
engine->set_fallback_hook([](const misaki::MToken& token) -> std::optional<std::string> {
  if (token.text == "rare_token") return std::string{"ɹˈɛɹ tˈOkən"};
  return std::nullopt;
});

// Optional unknown marker override:
engine->set_unknown_token("<UNK>");
```

`tokens` mirrors Python `MToken` fields:
- `text`, `tag`, `whitespace`, `phonemes`
- `start_ts`, `end_ts`
- `_` metadata (`is_head`, `alias`, `stress`, `currency`, `num_flags`, `prespace`, `rating`)

Token behavior notes:
- English backend (`make_engine("en", "en-us|en-gb")`) returns populated token streams.
- Non-English espeak backend (`make_engine("espeak", "es|fr-fr|hi|it|pt-br")`) is parity-aligned with Python `misaki.espeak.EspeakG2P` and returns phonemes with an empty token list.

Inline directives are propagated similarly to Python preprocessing:
- `[word](/phonemes/)`
- `[word](<number>)`
- `[word](#flags#)`

## Notes

- This component is parity-oriented; exactness is validated against Python-generated golden cases.
- Test exactness is intentionally strict (`REQUIRE(got == expected)`).
