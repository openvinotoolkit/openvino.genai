# Text-to-speech Python samples

This folder contains Python examples for `openvino_genai.Text2SpeechPipeline`:

- `text2speech.py`: basic text → audio generation (SpeechT5 and Kokoro)
- `kokoro_generate_from_phonemes.py`: Kokoro `generate_from_tokens` / `generate_from_phonemes` flow
- `kokoro_phonemize_fallback.py`: Kokoro unknown-word fallback behavior

## Supported Models

- **SpeechT5**
    - Requires exported SpeechT5 model and vocoder.
    - Usually uses a speaker embedding file.
- **Kokoro**
    - Uses a Kokoro model directory.
    - Uses `--speaker_embedding_file_path` and `--language` options.
    - End-to-end Kokoro language support includes:
        - `en-us` (English, United States)
        - `en-gb` (English, United Kingdom)
        - `es` (Spanish)
        - `fr-fr` (French, France)
        - `hi` (Hindi)
        - `it` (Italian)
        - `pt-br` (Portuguese, Brazil)
    - Not yet supported for end-to-end text generation in this flow: `ja` (Japanese), `zh` (Chinese/Mandarin).
    - For `ja` / `zh`, you can still generate speech by phonemizing with an external G2P and then using `generate_from_phonemes` (see `kokoro_generate_from_phonemes.py`).

## Install dependencies

- Export-time deps:

    `pip install --upgrade-strategy eager -r ../../export-requirements.txt`

- Runtime deps:

    `pip install -r ../../deployment-requirements.txt`

## SpeechT5 setup

Export SpeechT5 with vocoder:

```sh
optimum-cli export openvino --model microsoft/speecht5_tts --model-kwargs "{\"vocoder\": \"microsoft/speecht5_hifigan\"}" speecht5_tts
```

Create a speaker embedding file (SpeechT5-specific):

`python create_speaker_embedding.py`

## Kokoro setup

optimum-cli export openvino -m hexgrad/Kokoro-82M ov_Kokoro-82M --trust-remote-code

Kokoro can use `espeak-ng` in a couple of different ways:

- English (`en-us`, `en-gb`): `espeak-ng` is used as fallback for unknown/out-of-dictionary words. See `kokoro_phonemize_fallback` sample to understand how to use an OpenVINO fallback model to avoid use of `espeak-ng` for English.
- Non-English (`es`, `fr-fr`, `hi`, `it`, `pt-br`): `espeak-ng` is the primary engine used for G2P (phonemization) step. So, it is required to be installed for E2E text-to-speech generation cases for non-english languages. Note that application can replace default G2P step with another phonemizer. See `kokoro_generate_from_phonemes` sample for more details.

You can install `espeak-ng` by following the official guide [here](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md).

## Run samples

### 1) `text2speech.py`

SpeechT5:

`python text2speech.py --speaker_embedding_file_path speaker_embedding.bin speecht5_tts "Hello from OpenVINO GenAI"`

Kokoro:

`python text2speech.py --speaker_embedding_file_path ov_Kokoro-82M/voices/af_heart.bin --language en-us ov_Kokoro-82M "Hello, and welcome to speech generation using OpenVINO GenAI."`

Kokoro (non-English):

`python text2speech.py --speaker_embedding_file_path ov_Kokoro-82M/voices/ef_dora.bin --language es ov_Kokoro-82M "Hola y bienvenidos a la generación de voz utilizando OpenVINO GenAI."`

### 2) `kokoro_generate_from_phonemes.py` (Kokoro only)

This sample uses kokoro python module to perform G2P (phonemization) step, so install it into your environment:

`pip install kokoro`

Run:

`python kokoro_generate_from_phonemes.py ov_Kokoro-82M --language en-us --speaker_embedding_file_path ov_Kokoro-82M/voices/am_michael.bin --api auto`

This sample uses predefined texts per language and initializes Misaki for the selected `--language`.
`--api auto` prefers `generate_from_tokens` when Misaki returns token objects, and falls back to
`generate_from_phonemes` for languages where tokens are unavailable (for example, `ja`).
Pass `--speaker_embedding_file_path` with a prepared Kokoro embedding. If you want a blended speaker, mix the source voice binaries in your application before running the sample.

### 3) `kokoro_phonemize_fallback.py` (Kokoro only)

Prepare OV fallback models:

US:

`optimum-cli export openvino --model PeterReid/graphemes_to_phonemes_en_us --task text2text-generation graphemes_to_phonemes_en_us-ov`

GB:

`optimum-cli export openvino --model PeterReid/graphemes_to_phonemes_en_gb --task text2text-generation graphemes_to_phonemes_en_gb-ov`

Use OV fallback model:

US model + `en-us`:

`python kokoro_phonemize_fallback.py ov_Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --speaker_embedding_file_path ov_Kokoro-82M/voices/af_heart.bin --language en-us --phonemize_fallback_model_dir graphemes_to_phonemes_en_us-ov`

GB model + `en-gb`:

`python kokoro_phonemize_fallback.py ov_Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --speaker_embedding_file_path ov_Kokoro-82M/voices/bf_emma.bin --language en-gb --phonemize_fallback_model_dir graphemes_to_phonemes_en_gb-ov`

Use default `espeak-ng` fallback (omit `--phonemize_fallback_model_dir`):

`python kokoro_phonemize_fallback.py ov_Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --speaker_embedding_file_path ov_Kokoro-82M/voices/af_heart.bin --language en-us`

Set `--language` to match the fallback model variant (`en-us` with `..._en_us-ov`, `en-gb` with `..._en_gb-ov`).
OpenVINO fallback models above are an English-only feature (`en-us` / `en-gb`). For non-English Kokoro languages, phonemization is handled directly by `espeak-ng` as the primary G2P path (this fallback-model feature is not used).

All samples produce WAV output.

Refer to [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-generation-models) for model details.

# Text-to-speech API usage

```python
import openvino_genai

pipe = openvino_genai.Text2SpeechPipeline(model_dir, device)

result = pipe.generate("Hello OpenVINO GenAI", speaker_embedding)

# Kokoro generation with an application-prepared embedding tensor
result = pipe.generate("Hello from Kokoro", speaker_embedding, language="en-us")

# Kokoro generation from token (Kokoro backend only)
tokens = [
        openvino_genai.SpeechToken(phonemes="həlˈoʊ", whitespace=True, text="Hello"),
        openvino_genai.SpeechToken(phonemes="wˈɝld", whitespace=False, text="world"),
]
result = pipe.generate_from_tokens(tokens, speaker_embedding, language="en-us")

# Kokoro unknown-word fallback via config
cfg = pipe.get_generation_config()
cfg.phonemize_fallback_model_dir = "graphemes_to_phonemes_en_us-ov"  # set -> OV fallback
# cfg.phonemize_fallback_model_dir = None  # unset -> espeak-ng fallback
pipe.set_generation_config(cfg)
result = pipe.generate("Vellorin traded copperchimes for rainmint at Candlehaven.", speaker_embedding, language="en-us")
```
