# Text-to-speech Python samples

This folder contains Python examples for `openvino_genai.Text2SpeechPipeline`:

- `text2speech.py`: basic text → audio generation (SpeechT5 and Kokoro)
- `kokoro_phonemize_fallback.py`: Kokoro unknown-word fallback behavior
- `qwen3_customvoice.py`: Qwen3-TTS CustomVoice (built-in speaker + optional style instruct)
- `qwen3_voice_design.py`: Qwen3-TTS VoiceDesign (new voice from a natural-language instruct)
- `qwen3_base.py`: Qwen3-TTS Base voice cloning from reference audio (x-vector and ICL modes)

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
- **Qwen3-TTS**
    - Three model variants, each with its own dedicated sample:
        - `qwen3_customvoice.py`: speak with one of the model's built-in speaker identities, with an optional style `instruct`.
        - `qwen3_voice_design.py`: create a brand-new voice from a natural-language `instruct` description.
        - `qwen3_base.py`: clone a voice from a short reference recording (x-vector and ICL modes).
    - See the [Qwen3-TTS samples](#qwen3-tts-samples) section below for setup and run commands.

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
```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
pip install kokoro
optimum-cli export openvino -m hexgrad/Kokoro-82M ov_Kokoro-82M --trust-remote-code
```

> **Note:**
> After export is complete, you will find the available speaker embedding `.bin` files in `ov_Kokoro-82M/voices`.

## Qwen3-TTS setup

Export a Qwen3-TTS model to OpenVINO (choose the variant matching the sample you want to run), for example:

```sh
optimum-cli export openvino --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --trust-remote-code qwen3_tts_customvoice_ov
optimum-cli export openvino --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --trust-remote-code qwen3_tts_voicedesign_ov
optimum-cli export openvino --model Qwen/Qwen3-TTS-12Hz-0.6B-Base         --trust-remote-code qwen3_tts_base_ov
```

`--language` accepts the model's language names (for example `english`, `chinese`). Pass `auto` (or omit) to
let the model adapt automatically.

## Use of `espeak-ng` within the Kokoro Pipeline

Within the Kokoro Text-to-Speech pipeline, `espeak-ng` is an external dependency used for the grapheme-to-phoneme (G2P) stage. Its role varies depending on the selected language:

- **English (`en-us`, `en-gb`)**:
  `espeak-ng` is used as a fallback for words that are not found in the built-in dictionary.
  See the `kokoro_phonemize_fallback` sample for an example of using an OpenVINO-based fallback model to avoid relying on `espeak-ng` for English.

- **Non-English (`es`, `fr-fr`, `hi`, `it`, `pt-br`)**:
  `espeak-ng` serves as the primary G2P (phonemization) engine. As such, it must be installed to enable end-to-end text-to-speech generation for these languages.

> **Note:**
> `espeak-ng` is licensed under GPLv3 and must be installed separately. OpenVINO GenAI detects its presence automatically at runtime.

To install `espeak-ng`, follow the official guide:
https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md

## Run samples

### 1) `text2speech.py`

SpeechT5:
```
python text2speech.py --speaker_embedding_file_path speaker_embedding.bin speecht5_tts "Hello from OpenVINO GenAI"
```

Kokoro:
```
python text2speech.py --speaker_embedding_file_path ov_Kokoro-82M/voices/af_heart.bin --language en-us ov_Kokoro-82M "Hello, and welcome to speech generation using OpenVINO GenAI."
```

Kokoro (non-English):
```
python text2speech.py --speaker_embedding_file_path ov_Kokoro-82M/voices/ef_dora.bin --language es ov_Kokoro-82M "Hola y bienvenidos a la generación de voz utilizando OpenVINO GenAI."
```

Text2speech with speed control:
```
python text2speech.py --speaker_embedding_file_path ov_Kokoro-82M/voices/af_heart.bin --language en-us --speed 1.15 ov_Kokoro-82M "Hello from OpenVINO GenAI with a faster speaking rate."
```

> **Note:** `text2speech.py` targets SpeechT5 and Kokoro. Qwen3-TTS is covered by its own dedicated
> samples — see the [Qwen3-TTS samples](#qwen3-tts-samples) section.

### 2) `kokoro_phonemize_fallback.py` (Kokoro only)

This sample demonstrates how to use an OpenVINO-based fallback model for phonemization, allowing you to avoid relying on `espeak-ng` when working with English languages.

**Why use a fallback model instead of `espeak-ng`?**

While `espeak-ng` provides robust phonemization, using an OpenVINO-based fallback model avoids the need for external dependencies and consideration of their associated licensing requirements, enabling a more self-contained and uniformly licensed deployment.

#### Export OV fallback models:

US:
```
optimum-cli export openvino --model PeterReid/graphemes_to_phonemes_en_us --task text2text-generation graphemes_to_phonemes_en_us-ov
```

GB:
```
optimum-cli export openvino --model PeterReid/graphemes_to_phonemes_en_gb --task text2text-generation graphemes_to_phonemes_en_gb-ov
```

#### Run using fallback models:

US model + `en-us`:
```
python kokoro_phonemize_fallback.py ov_Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --speaker_embedding_file_path ov_Kokoro-82M/voices/af_heart.bin --language en-us --phonemize_fallback_model_dir graphemes_to_phonemes_en_us-ov
```

GB model + `en-gb`:
```
python kokoro_phonemize_fallback.py ov_Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --speaker_embedding_file_path ov_Kokoro-82M/voices/bf_emma.bin --language en-gb --phonemize_fallback_model_dir graphemes_to_phonemes_en_gb-ov
```

Use default `espeak-ng` fallback (omit `--phonemize_fallback_model_dir`):
```
python kokoro_phonemize_fallback.py ov_Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --speaker_embedding_file_path ov_Kokoro-82M/voices/af_heart.bin --language en-us
```

Set `--language` to match the fallback model variant (`en-us` with `..._en_us-ov`, `en-gb` with `..._en_gb-ov`).
OpenVINO fallback models above are an English-only feature (`en-us` / `en-gb`). For non-English Kokoro languages, phonemization is handled directly by `espeak-ng` as the primary G2P path (this fallback-model feature is not used).

## Qwen3-TTS samples

Qwen3-TTS ships as three model variants, and this folder provides one focused sample for each:

| Sample | Model variant | What it showcases |
| --- | --- | --- |
| `qwen3_customvoice.py` | CustomVoice | Built-in speaker identities, optional style `instruct` |
| `qwen3_voice_design.py` | VoiceDesign | A new voice created from a natural-language `instruct` description |
| `qwen3_base.py` | Base | Voice cloning from reference audio (x-vector and ICL modes) |

See the [Qwen3-TTS setup](#qwen3-tts-setup) section above for export commands.

### 3) `qwen3_customvoice.py`

Speak with one of the model's built-in speakers. `--instruct` is optional and steers tone/emotion/pace.

```
python qwen3_customvoice.py qwen3_tts_customvoice_ov "Hello from Qwen3 CustomVoice." --speaker ryan --language english
```

With a style instruction:
```
python qwen3_customvoice.py qwen3_tts_customvoice_ov "Hello from Qwen3 CustomVoice." --speaker ryan --language english --instruct "Speak in a calm, professional tone."
```

For `Qwen3-TTS-12Hz-1.7B/0.6B-CustomVoice` models, the supported speaker list and speaker descriptions are provided below. We recommend using each speaker's native language for the best quality. Of course, each speaker can speak any language supported by the model.

| Speaker | Voice Description | Native language |
| --- | --- | --- |
| Vivian | Bright, slightly edgy young female voice. | Chinese |
| Serena | Warm, gentle young female voice. | Chinese |
| Uncle_Fu | Seasoned male voice with a low, mellow timbre. | Chinese |
| Dylan | Youthful Beijing male voice with a clear, natural timbre. | Chinese (Beijing Dialect) |
| Eric | Lively Chengdu male voice with a slightly husky brightness. | Chinese (Sichuan Dialect) |
| Ryan | Dynamic male voice with strong rhythmic drive. | English |
| Aiden | Sunny American male voice with a clear midrange. | English |
| Ono_Anna | Playful Japanese female voice with a light, nimble timbre. | Japanese |
| Sohee | Warm Korean female voice with rich emotion. | Korean |

### 4) `qwen3_voice_design.py`

Design a new voice purely from a natural-language description. There is no speaker list; `--instruct` is required.

```
python qwen3_voice_design.py qwen3_tts_voicedesign_ov "Hello from Qwen3 VoiceDesign." --language english --instruct "A male voice with a thick French accent."
```

### 5) `qwen3_base.py`

Clone a voice from a short reference recording. Two modes are selected automatically from the inputs:

- **x-vector mode** (fast, identity only): provide reference audio (or a pre-saved speaker embedding). `--ref_text` is not required.
- **ICL mode** (higher fidelity): additionally provide the reference transcript via `--ref_text`.

Clone directly from reference audio (x-vector mode):
```
python qwen3_base.py qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path reference_24k.wav --language english
```

Clone from reference audio + transcript (ICL mode):
```
python qwen3_base.py qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path reference_24k.wav --ref_text "This is the reference transcript." --language english
```

> **Note:** reference audio must already be mono/stereo at 24000 Hz. OV GenAI does not resample reference audio.

#### Reusing a reference prompt (save once, reuse many times)

Extracting the speaker embedding and reference codes from audio is the expensive part of cloning. To avoid
recomputing them on every run, clone once from reference audio and save the artifacts that `generate(...)`
returns on the result (`speaker_embedding` and `voice_clone_ref_codec_ids`):

Save from a first x-vector run:
```
python qwen3_base.py qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path reference_24k.wav --language english --save_speaker_embedding_file_path qwen_speaker_embedding.bin
```

Save from a first ICL run (also emits reference codes):
```
python qwen3_base.py qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path reference_24k.wav --ref_text "This is the reference transcript." --language english --save_speaker_embedding_file_path qwen_speaker_embedding.bin --save_ref_codec_ids_file_path qwen_ref_code.bin
```

Reuse the saved speaker embedding (x-vector mode, no encoder pass):
```
python qwen3_base.py qwen3_tts_base_ov "Hello again." --speaker_embedding_file_path qwen_speaker_embedding.bin --language english
```

Reuse saved embedding + reference codes (ICL mode, no encoder pass):
```
python qwen3_base.py qwen3_tts_base_ov "Hello again." --speaker_embedding_file_path qwen_speaker_embedding.bin --ref_text "This is the reference transcript." --ref_codec_ids_file_path qwen_ref_code.bin --language english
```

The saved files use simple flat-binary layouts owned by this sample (the speaker embedding is raw
float32; the reference codes carry a small shape header), matching the C++ `qwen3_base` sample so the
files are interchangeable between the two.

All samples produce WAV output.

Refer to [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-generation-models) for model details.

# Text-to-speech API usage

```python
import openvino_genai

pipe = openvino_genai.Text2SpeechPipeline(model_dir, device)

result = pipe.generate("Hello OpenVINO GenAI", speaker_embedding)

# Kokoro generation with an application-prepared embedding tensor
result = pipe.generate("Hello from Kokoro", speaker_embedding, language="en-us")

# Qwen3 CustomVoice generation (no external embedding required)
result = pipe.generate("Hello from Qwen3 CustomVoice", None, speaker="ryan", language="english")

# Qwen3 VoiceDesign generation (no external embedding required)
result = pipe.generate("Hello from Qwen3 VoiceDesign", None, language="english", instruct="A warm, deep male narrator voice")

# Qwen3 Base voice clone from an application-prepared speaker embedding tensor
result = pipe.generate("Hello from Qwen3 Base", speaker_embedding, language="english")

# Kokoro unknown-word fallback via config
cfg = pipe.get_generation_config()
cfg.phonemize_fallback_model_dir = "graphemes_to_phonemes_en_us-ov"  # set -> OV fallback
# cfg.phonemize_fallback_model_dir = None  # unset -> espeak-ng fallback
pipe.set_generation_config(cfg)
result = pipe.generate("Vellorin traded copperchimes for rainmint at Candlehaven.", speaker_embedding, language="en-us")
```
