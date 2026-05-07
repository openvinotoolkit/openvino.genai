# Text-to-speech C++ samples

This folder contains C++ examples for `ov::genai::Text2SpeechPipeline`.

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

## SpeechT5 setup

Install export tools and export SpeechT5 with vocoder:

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
optimum-cli export openvino --model microsoft/speecht5_tts --model-kwargs "{\"vocoder\": \"microsoft/speecht5_hifigan\"}" speecht5_tts
```

Create a speaker embedding file (SpeechT5-specific):

`python ../../python/speech_generation/create_speaker_embedding.py`

## Kokoro setup

optimum-cli export openvino -m hexgrad/Kokoro-82M ov_Kokoro-82M --trust-remote-code

> **Note:**
> After export is complete. you will find the available speaker embedding `.bin` files in `ov_Kokoro-82M/voices`.

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

`text2speech speecht5_tts "Hello from OpenVINO GenAI" speaker_embedding.bin`

Kokoro:

`text2speech ov_Kokoro-82M "Hello, and welcome to speech generation using OpenVINO GenAI." ov_Kokoro-82M/voices/af_heart.bin --language en-us`

Kokoro (non-English):

`text2speech ov_Kokoro-82M "Hola y bienvenidos a la generación de voz utilizando OpenVINO GenAI." ov_Kokoro-82M/voices/ef_dora.bin --language es`

### 2) `kokoro_phonemize_fallback` (Kokoro only)

This sample demonstrates how to use an OpenVINO-based fallback model for phonemization, allowing you to avoid relying on `espeak-ng` when working with English languages.

**Why use a fallback model instead of `espeak-ng`?**

While `espeak-ng` provides robust phonemization, it is licensed under GPLv3, which can introduce complications for redistribution in certain applications. Using an OpenVINO-based fallback model avoids this dependency entirely, enabling a more self-contained and permissively licensed deployment.

#### Export OV fallback models:

US:

`optimum-cli export openvino --model PeterReid/graphemes_to_phonemes_en_us --task text2text-generation graphemes_to_phonemes_en_us-ov`

GB:

`optimum-cli export openvino --model PeterReid/graphemes_to_phonemes_en_gb --task text2text-generation graphemes_to_phonemes_en_gb-ov`

#### Run using fallback models:

US model + `en-us`:

`kokoro_phonemize_fallback ov_Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --speaker_embedding_file_path ov_Kokoro-82M/voices/af_heart.bin --language en-us --phonemize_fallback_model_dir graphemes_to_phonemes_en_us-ov`

GB model + `en-gb`:

`kokoro_phonemize_fallback ov_Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --speaker_embedding_file_path ov_Kokoro-82M/voices/bf_emma.bin --language en-gb --phonemize_fallback_model_dir graphemes_to_phonemes_en_gb-ov`

To run with default `espeak-ng` fallback:

`kokoro_phonemize_fallback ov_Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --speaker_embedding_file_path ov_Kokoro-82M/voices/af_heart.bin --language en-us`

Set `--language` to match the fallback model variant (`en-us` with `..._en_us-ov`, `en-gb` with `..._en_gb-ov`).
OpenVINO fallback models above are an English-only feature (`en-us` / `en-gb`). For non-English Kokoro languages, phonemization is handled directly by `espeak-ng` as the primary G2P path (this fallback-model feature is not used).

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

auto speech = gen_speech.speeches[0];
// speech tensor contains the waveform of the spoken phrase
```
