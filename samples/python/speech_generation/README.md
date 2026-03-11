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
    - Uses `--voice` and `--language` options.
    - For end-to-end Kokoro pipeline, only english (`en-us`, `en-gb`) language is supported.

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

**TODO! Add optimum-intel cmd here when available!**

Kokoro phonemization can encounter unknown/out-of-dictionary words. For those words, Kokoro uses a fallback phonemizer.

- Default fallback: `espeak-ng`
- Optional alternative: OpenVINO fallback model via `phonemize_fallback_model_dir`

In order to make use of `espeak-ng` fallback method, you must install `espeak-ng`. Please follow the official installation guide, [here](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md)

## Run samples

### 1) `text2speech.py`

SpeechT5:

`python text2speech.py --speaker_embedding_file_path speaker_embedding.bin speecht5_tts "Hello OpenVINO GenAI"`

Kokoro:

`python text2speech.py --voice af_heart --language en-us Kokoro-82M "Hello from Kokoro in OpenVINO GenAI"`

### 2) `kokoro_generate_from_phonemes.py` (Kokoro only)

Install Kokoro if needed:

`pip install kokoro`

Run:

`python kokoro_generate_from_phonemes.py Kokoro-82M --language es --api auto`

This sample uses predefined texts per language and initializes Misaki for the selected `--language`.
`--api auto` prefers `generate_from_tokens` when Misaki returns token objects, and falls back to
`generate_from_phonemes` for languages where tokens are unavailable (for example, `ja`).
If `--voice` is omitted, the sample picks a language-appropriate default voice.

### 3) `kokoro_phonemize_fallback.py` (Kokoro only)

Prepare OV fallback models:

US:

`optimum-cli export openvino --model PeterReid/graphemes_to_phonemes_en_us --task text2text-generation graphemes_to_phonemes_en_us-ov`

GB:

`optimum-cli export openvino --model PeterReid/graphemes_to_phonemes_en_gb --task text2text-generation graphemes_to_phonemes_en_gb-ov`

Use OV fallback model:

US model + `en-us`:

`python kokoro_phonemize_fallback.py Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --voice af_heart --language en-us --phonemize_fallback_model_dir graphemes_to_phonemes_en_us-ov`

GB model + `en-gb`:

`python kokoro_phonemize_fallback.py Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --voice af_heart --language en-gb --phonemize_fallback_model_dir graphemes_to_phonemes_en_gb-ov`

Use default `espeak-ng` fallback (omit `--phonemize_fallback_model_dir`):

`python kokoro_phonemize_fallback.py Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --voice af_heart --language en-us`

Set `--language` to match the fallback model variant (`en-us` with `..._en_us-ov`, `en-gb` with `..._en_gb-ov`).

All samples produce WAV output.

Refer to [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-generation-models) for model details.

# Text-to-speech API usage

```python
import openvino_genai

pipe = openvino_genai.Text2SpeechPipeline(model_dir, device)
result = pipe.generate("Hello OpenVINO GenAI", speaker_embedding)

# Kokoro voice-based generation (speaker embedding not required)
result = pipe.generate("Hello from Kokoro", None, voice="af_heart", language="en-us")

# Kokoro generation from token (Kokoro backend only)
tokens = [
        openvino_genai.SpeechToken(phonemes="həlˈoʊ", whitespace=True, text="Hello"),
        openvino_genai.SpeechToken(phonemes="wˈɝld", whitespace=False, text="world"),
]
result = pipe.generate_from_tokens(tokens, None, voice="af_heart", language="en-us")

# Kokoro unknown-word fallback via config
cfg = pipe.get_generation_config()
cfg.phonemize_fallback_model_dir = "graphemes_to_phonemes_en_us-ov"  # set -> OV fallback
# cfg.phonemize_fallback_model_dir = None  # unset -> espeak-ng fallback
pipe.set_generation_config(cfg)
result = pipe.generate("Vellorin traded copperchimes for rainmint at Candlehaven.", None, voice="af_heart", language="en-us")
```
