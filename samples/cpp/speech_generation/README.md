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
	- Supports `custom_voice`, `voice_design`, and `base` variants.
	- `custom_voice`: pass `--speaker <name>` with one of the predefined speaker ids.
	- `voice_design`: pass `--instruct <text>` with natural language voice description.
	- `base`: pass an external speaker embedding `.bin` file as the positional speaker embedding argument
	  or with `--speaker_embedding_file_path <PATH>`.
	  Expected shape is returned by `pipe.get_speaker_embedding_shape()`.

## SpeechT5 setup

Install export tools and export SpeechT5 with vocoder:

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
optimum-cli export openvino --model microsoft/speecht5_tts --model-kwargs "{\"vocoder\": \"microsoft/speecht5_hifigan\"}" speecht5_tts
```

Create a speaker embedding file (SpeechT5-specific):

`python ../../python/speech_generation/create_speaker_embedding.py`

Create a speaker embedding file for Qwen3 Base from reference audio:

`python ../../python/speech_generation/create_qwen3_speaker_embedding.py qwen3_tts_base_ov ref_audio.wav --output qwen_speaker_embedding.bin`

Note: this utility expects reference audio already at the model sample rate (typically 24000 Hz)
and does not resample internally.

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

Qwen3-TTS CustomVoice:
```
text2speech qwen3_tts_customvoice_ov "Hello from Qwen3 CustomVoice" --speaker ryan --language english --instruct "speak in a calm style"
```

Qwen3-TTS VoiceDesign:
```
text2speech qwen3_tts_voicedesign_ov "Hello from Qwen3 VoiceDesign" --language english --instruct "A male voice with a thick french accent."
```

Qwen3-TTS Base (x-vector style voice clone):
```
text2speech qwen3_tts_base_ov "Hello from Qwen3 Base" qwen_speaker_embedding.bin --language english
```

Qwen3-TTS Base (named speaker embedding argument):
```
text2speech qwen3_tts_base_ov "Hello from Qwen3 Base" --speaker_embedding_file_path qwen_speaker_embedding.bin --language english
```

Qwen3-TTS Base (ICL mode via `generate(...)` properties):
```
text2speech qwen3_tts_base_ov "Hello from Qwen3 Base ICL" qwen_speaker_embedding.bin --language english --qwen_x_vector_only_mode false --qwen_ref_text "Reference transcript for prompt conditioning" --qwen_ref_code_file_path ref_code.npy
```

If speaker embedding is omitted for a Base model, the sample now fails early with a clear message
that includes the expected shape (for example, `{1, 1, 1024}`).

For Qwen3 ICL mode in the C++ sample:
- Set `--qwen_x_vector_only_mode false`
- Pass `--qwen_ref_text <TEXT>`
- Pass `--qwen_ref_code_file_path <PATH.npy>` (expects int64 `.npy`, for example produced by the Python utility)

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

// Qwen3 Base generation with an application-prepared embedding tensor
gen_speech = pipe.generate(prompt,
						   speaker_embedding,
						   ov::AnyMap{{"language", "english"}});

// Qwen3 CustomVoice generation (no external embedding required)
gen_speech = pipe.generate(prompt,
						   ov::Tensor(),
						   ov::AnyMap{{"speaker", "ryan"}, {"language", "english"}});

// Qwen3 VoiceDesign generation (no external embedding required)
gen_speech = pipe.generate(prompt,
						   ov::Tensor(),
						   ov::AnyMap{{"language", "english"}, {"instruct", "A warm, deep male narrator voice"}});

auto speech = gen_speech.speeches[0];
// speech tensor contains the waveform of the spoken phrase
```
