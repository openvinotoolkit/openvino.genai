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
- `data/`
  - English lexicon data (`us_*.json`, `gb_*.json`)

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

### Run tests

If your configured build enables these tests, run from build directory:

```powershell
ctest --test-dir ../openvino.genai-build --output-on-failure -R "English|misaki|speech"
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

Inline directives are propagated similarly to Python preprocessing:
- `[word](/phonemes/)`
- `[word](<number>)`
- `[word](#flags#)`

## Notes

- This component is parity-oriented; exactness is validated against Python-generated golden cases.
- Test exactness is intentionally strict (`REQUIRE(got == expected)`).
