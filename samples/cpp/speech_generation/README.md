# Text-to-speech C++ samples

This folder contains C++ examples for `ov::genai::Text2SpeechPipeline`.

## Supported Models

- **SpeechT5**
	- Requires exported SpeechT5 model and vocoder.
	- Usually uses a speaker embedding file.
- **Kokoro**
	- Uses a Kokoro model directory.
	- Uses a speaker embedding file and language options.
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
		- `qwen3_customvoice`: speak with one of the model's built-in speaker identities, with an optional style `instruct`.
		- `qwen3_voice_design`: create a brand-new voice from a natural-language `instruct` description.
		- `qwen3_base`: clone a voice from a short reference recording (x-vector and ICL modes).
	- See the [Qwen3-TTS samples](#qwen3-tts-samples) section below for setup and run commands.

## SpeechT5 setup

Install export tools and export SpeechT5 with vocoder:

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
optimum-cli export openvino --model microsoft/speecht5_tts --model-kwargs "{\"vocoder\": \"microsoft/speecht5_hifigan\"}" speecht5_tts
```

Create a speaker embedding file (SpeechT5-specific):

`python ../../python/speech_generation/create_speaker_embedding.py`

## Kokoro setup
```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
pip install kokoro
optimum-cli export openvino -m hexgrad/Kokoro-82M ov_Kokoro-82M --trust-remote-code
```

> **Note:**
> After export is complete, you will find the available speaker embedding `.bin` files in `ov_Kokoro-82M/voices`.

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

### 1) `text2speech`

SpeechT5:
```
text2speech speecht5_tts "Hello from OpenVINO GenAI" speaker_embedding.bin
```

Kokoro:
```
text2speech ov_Kokoro-82M "Hello, and welcome to speech generation using OpenVINO GenAI." ov_Kokoro-82M/voices/af_heart.bin --language en-us
```

Kokoro (non-English):
```
text2speech ov_Kokoro-82M "Hola y bienvenidos a la generación de voz utilizando OpenVINO GenAI." ov_Kokoro-82M/voices/ef_dora.bin --language es
```

Text2speech with speed control:
```
text2speech ov_Kokoro-82M "Hello from OpenVINO GenAI with a faster speaking rate." ov_Kokoro-82M/voices/af_heart.bin --language en-us --speed 1.15
```

> **Note:** `text2speech` targets SpeechT5 and Kokoro. Qwen3-TTS is covered by its own dedicated
> samples — see the [Qwen3-TTS samples](#qwen3-tts-samples) section.

### 2) `kokoro_phonemize_fallback` (Kokoro only)

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
kokoro_phonemize_fallback ov_Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --speaker_embedding_file_path ov_Kokoro-82M/voices/af_heart.bin --language en-us --phonemize_fallback_model_dir graphemes_to_phonemes_en_us-ov
```

GB model + `en-gb`:
```
kokoro_phonemize_fallback ov_Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --speaker_embedding_file_path ov_Kokoro-82M/voices/bf_emma.bin --language en-gb --phonemize_fallback_model_dir graphemes_to_phonemes_en_gb-ov
```

To run with default `espeak-ng` fallback:
```
kokoro_phonemize_fallback ov_Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --speaker_embedding_file_path ov_Kokoro-82M/voices/af_heart.bin --language en-us
```

Set `--language` to match the fallback model variant (`en-us` with `..._en_us-ov`, `en-gb` with `..._en_gb-ov`).
OpenVINO fallback models above are an English-only feature (`en-us` / `en-gb`). For non-English Kokoro languages, phonemization is handled directly by `espeak-ng` as the primary G2P path (this fallback-model feature is not used).

## Qwen3-TTS samples

Qwen3-TTS ships as three model variants, and this folder provides one focused sample for each:

| Sample | Model variant | What it showcases |
| --- | --- | --- |
| `qwen3_customvoice` | CustomVoice | Built-in speaker identities, optional style `instruct` |
| `qwen3_voice_design` | VoiceDesign | A new voice created from a natural-language `instruct` description |
| `qwen3_base` | Base | Voice cloning from reference audio (x-vector and ICL modes) |

### Qwen3-TTS setup

Convert a Qwen3-TTS model to OpenVINO (choose the variant matching the sample you want to run), for example:

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
optimum-cli export openvino --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --trust-remote-code qwen3_tts_customvoice_ov
optimum-cli export openvino --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --trust-remote-code qwen3_tts_voicedesign_ov
optimum-cli export openvino --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --trust-remote-code qwen3_tts_base_ov
```

`--language` accepts the model's language names (for example `english`, `chinese`). Pass `auto` (or omit) to
let the model adapt automatically.

### 3) `qwen3_customvoice`

Speak with one of the model's built-in speakers. `--instruct` is optional and steers tone/emotion/pace.

```
qwen3_customvoice qwen3_tts_customvoice_ov "Hello from Qwen3 CustomVoice." --speaker ryan --language english
```

With a style instruction:
```
qwen3_customvoice qwen3_tts_customvoice_ov "Hello from Qwen3 CustomVoice." --speaker ryan --language english --instruct "Speak in a calm, professional tone."
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

### 4) `qwen3_voice_design`

Design a new voice purely from a natural-language description. There is no speaker list; `--instruct` is required.

```
qwen3_voice_design qwen3_tts_voicedesign_ov "Hello from Qwen3 VoiceDesign." --language english --instruct "A male voice with a thick French accent."
```

### 5) `qwen3_base`

Clone a voice from a short reference recording. Two modes are selected automatically from the inputs:

- **x-vector mode** (fast, identity only): provide reference audio (or a pre-saved speaker embedding). `--ref_text` is not required.
- **ICL mode** (higher fidelity): additionally provide the reference transcript via `--ref_text`.

Clone directly from reference audio (x-vector mode):
```
qwen3_base qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path reference_24k.wav --language english
```

Clone from reference audio + transcript (ICL mode):
```
qwen3_base qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path reference_24k.wav --ref_text "This is the reference transcript." --language english
```

> **Note:** reference audio must already be mono/stereo at 24000 Hz. OV GenAI does not resample reference audio.

#### Reusing a reference prompt (save once, reuse many times)

Extracting the speaker embedding and reference codes from audio is the expensive part of cloning. To avoid
recomputing them on every run, clone once from reference audio and save the artifacts that `generate(...)`
returns on the result (`speaker_embedding` and `voice_clone_ref_codec_ids`):

Save from a first x-vector run:
```
qwen3_base qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path reference_24k.wav --language english --save_speaker_embedding_file_path qwen_speaker_embedding.bin
```

Save from a first ICL run (also emits reference codes):
```
qwen3_base qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path reference_24k.wav --ref_text "This is the reference transcript." --language english --save_speaker_embedding_file_path qwen_speaker_embedding.bin --save_ref_codec_ids_file_path qwen_ref_code.bin
```

Reuse the saved speaker embedding (x-vector mode, no encoder pass):
```
qwen3_base qwen3_tts_base_ov "Hello again." --speaker_embedding_file_path qwen_speaker_embedding.bin --language english
```

Reuse saved embedding + reference codes (ICL mode, no encoder pass):
```
qwen3_base qwen3_tts_base_ov "Hello again." --speaker_embedding_file_path qwen_speaker_embedding.bin --ref_text "This is the reference transcript." --ref_codec_ids_file_path qwen_ref_code.bin --language english
```

The saved files use simple flat-binary layouts owned by this sample (the speaker embedding is raw
float32; the reference codes carry a small shape header), so no external tooling is required.

All samples produce WAV output.

Refer to [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-generation-models) for model details.

# Text-to-speech API usage

```c++
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

ov::genai::Text2SpeechPipeline pipe(models_path, device);

gen_speech = pipe.generate(prompt, speaker_embedding);

// Kokoro generation with an application-prepared embedding tensor
gen_speech = pipe.generate(prompt,
                           speaker_embedding,
                           ov::AnyMap{{"language", "en-us"}});

// Qwen3 CustomVoice generation (no external embedding required)
gen_speech = pipe.generate(prompt,
                           ov::Tensor(),
                           ov::AnyMap{{"speaker", "ryan"}, {"language", "english"}});

// Qwen3 VoiceDesign generation (no external embedding required)
gen_speech = pipe.generate(prompt,
                           ov::Tensor(),
                           ov::AnyMap{{"language", "english"}, {"instruct", "A warm, deep male narrator voice"}});

// Qwen3 Base voice clone from an application-prepared speaker embedding tensor
gen_speech = pipe.generate(prompt,
                           speaker_embedding,
                           ov::AnyMap{{"language", "english"}});

auto speech = gen_speech.speeches[0];
// speech tensor contains the waveform of the spoken phrase
```
