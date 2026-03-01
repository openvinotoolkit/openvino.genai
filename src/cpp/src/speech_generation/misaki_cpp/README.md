# Misaki C++ (G2P Parity Project)

This directory is the C++ port/parity project for Misaki G2P.

Current status:
- English backend (`en-us`, `en-gb`) has corpus-driven parity coverage and dedicated exactness tests.
- Non-English C++ backends are intentionally not included at this stage.

## Project Layout

- `CMakeLists.txt` — build and test configuration
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
  - `fallback_and_unknown.cpp` — fallback hook + unknown-token behavior example
  - `README.md` — sample notes
- `tools/gen_golden.py` — Python golden corpus generator

## Prerequisites

### C++ build/test
- CMake >= 3.20
- C++20 compiler (MSVC/Clang/GCC)
- Network access on first configure (FetchContent pulls Catch2 and nlohmann/json)

### Golden corpus generation (optional for consumers, required for corpus refresh)
- Python 3.12 recommended
- Misaki Python dependencies installed in a venv

## Build and Run Tests

From repository root:

```powershell
cmake -S cpp -B cpp/build
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
```

Or from `cpp/` using presets:

Windows (Visual Studio):

```powershell
cmake --preset windows-vs-debug
cmake --build --preset windows-vs-debug
ctest --preset windows-vs-debug
```

Linux/macOS (Ninja):

```bash
cmake --preset ninja-debug
cmake --build --preset ninja-debug
ctest --preset ninja-debug
```

One-command workflow presets are also available:

```powershell
cmake --workflow --preset windows-vs-debug-all
```

## Install and Consume via `find_package`

Install the library/package files:

Windows (PowerShell):

```powershell
cmake -S cpp -B cpp/build
cmake --build cpp/build --config Debug
cmake --install cpp/build --config Debug --prefix <install-prefix>
```

Linux/macOS (Bash/Zsh):

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build
cmake --install cpp/build --prefix <install-prefix>
```

In a downstream CMake project:

```cmake
find_package(MisakiCpp CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE MisakiCpp::misaki_cpp)
```

If installed to a custom prefix, set `CMAKE_PREFIX_PATH` accordingly.

Example configure for a downstream project:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="<install-prefix>"
```

Useful subsets:

```powershell
ctest --test-dir cpp/build --output-on-failure -R "English"
```

## Regenerating Golden Corpora

Example commands (from repository root, with your Python environment):

```powershell
.\.venv312\Scripts\python.exe cpp/tools/gen_golden.py --profile english --english-sample-size 200 --english-seed 1337 --out cpp/tests/english_golden_cases.jsonl
.\.venv312\Scripts\python.exe cpp/tools/gen_golden.py --profile english-gb --english-sample-size 200 --english-seed 1337 --out cpp/tests/english_gb_golden_cases.jsonl
```

Profiles currently supported:
- `english`
- `english-gb`

English corpus notes:
- `english` (`cpp/tests/english_golden_cases.jsonl`) includes curated numeric/symbol coverage
  (e.g. `%`, `+`, `@`, currency, decimals) plus deterministic lexicon sampling.
- `english-gb` (`cpp/tests/english_gb_golden_cases.jsonl`) mirrors the same coverage pattern for `en-gb`.

Quick refresh (English corpora only):

```powershell
.\.venv312\Scripts\python.exe cpp/tools/gen_golden.py --profile english --english-sample-size 200 --english-seed 1337 --out cpp/tests/english_golden_cases.jsonl
.\.venv312\Scripts\python.exe cpp/tools/gen_golden.py --profile english-gb --english-sample-size 200 --english-seed 1337 --out cpp/tests/english_gb_golden_cases.jsonl
```

## Standalone Migration Guide

This directory is designed to be movable as a standalone project.

### What you need to copy
At minimum, keep these inside the standalone `cpp/` project:
- `CMakeLists.txt`
- `include/`
- `src/`
- `tests/` (including generated JSONL corpora)
- `samples/`
- `tools/`

For English runtime lexicon support, you also need:
- `data/us_gold.json`
- `data/us_silver.json`
- `data/gb_gold.json`
- `data/gb_silver.json`

English lexicon resolution order is:
1. `cpp/data/{us,gb}_{gold,silver}.json`
2. fallback to monorepo-style `../misaki/data/...`

So in standalone mode, place those files under `cpp/data/`.

### Standalone usage modes

1. **Consumer mode (no Python required)**
  - Keep English generated JSONL corpora in `tests/` and build/run C++ tests only.

2. **Corpus-refresh mode (Python required)**
   - Set up Python env, install Misaki dependencies, run `tools/gen_golden.py` to regenerate corpora.

## Public API (current)

```cpp
auto engine = misaki::make_engine("en", "en-us");
auto result = engine->phonemize_with_tokens("[Misaki](/misˈɑki/) is a G2P engine.");
auto phonemes = result.phonemes;
auto tokens = result.tokens;

// Backward-compatible convenience still available:
auto phonemes_only = engine->phonemize("[Misaki](/misˈɑki/) is a G2P engine.");

// Optional unresolved-token fallback (Python-like hook point):
engine->set_fallback_hook([](const misaki::MToken& token) -> std::optional<std::string> {
  if (token.text == "rare_token") return std::string{"ɹˈɛɹ tˈOkən"};
  return std::nullopt;
});

// Python-like unknown token marker (default is "❓"):
engine->set_unknown_token("<UNK>");

// Built-in espeak-ng fallback (Python EspeakFallback equivalent):
// Requires runtime availability of libespeak-ng (no build-time link required).
#include "misaki/fallbacks.hpp"
misaki::EspeakFallback espeak_fallback(/*british=*/false);
engine->set_fallback_hook(espeak_fallback.as_hook());

if (!espeak_fallback.backend_available()) {
  // Helpful for deployment diagnostics (missing DLL/.so/.dylib, etc.)
  auto err = espeak_fallback.backend_error();
}
```

`tokens` is a C++ mirror of Python `MToken`, including:
- `text`, `tag`, `whitespace`, `phonemes`
- `start_ts`, `end_ts`
- `_` metadata (`is_head`, `alias`, `stress`, `currency`, `num_flags`, `prespace`, `rating`)

Some metadata values are currently defaults in C++ where equivalent runtime signals are not yet implemented.
Inline directives are propagated similarly to Python preprocess behavior:
- `[word](/phonemes/)` sets token phonemes and high-confidence rating
- `[word](<number>)` stores `_ .stress` and applies stress override to emitted phonemes
- `[word](#flags#)` stores `_ .num_flags`

If no fallback hook is configured, unresolved English tokens emit the configured unknown marker
(default `❓`) rather than throwing.

English lexicon data root is resolved at runtime in this order:
1. `set_english_lexicon_data_root(...)` override
2. `MISAKI_DATA_DIR` environment variable
3. source-tree development paths (`cpp/data`, `../misaki/data`)
4. installed layout relative to module (`../share/misaki/data`, then `../data`)

Runtime override helpers:

```cpp
misaki::set_english_lexicon_data_root("/path/to/misaki/data");
misaki::clear_english_lexicon_data_root();
```

Supported language/variant pairs in this project:
- `en` / `en-us`
- `en` / `en-gb`

## Notes

- This project is parity-oriented; exactness is validated against Python-generated goldens.
- The test harness is intentionally strict (`REQUIRE(got == expected)` exact match).
