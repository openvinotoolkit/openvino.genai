# Text-to-speech C++ samples

This folder contains C++ examples for `ov::genai::Text2SpeechPipeline`.

## Supported Models

- **SpeechT5**
	- Requires exported SpeechT5 model and vocoder.
	- Usually uses a speaker embedding file.
- **Kokoro**
	- Uses a Kokoro model directory.
	- Uses `--voice` and `--language` options.
	- End-to-end Kokoro language support includes:
		- `en-us` (English, United States)
		- `en-gb` (English, United Kingdom)
		- `es` (Spanish)
		- `fr-fr` (French, France)
		- `hi` (Hindi)
		- `it` (Italian)
		- `pt-br` (Portuguese, Brazil)
	- Not yet supported for end-to-end text generation in this flow: `ja` (Japanese), `zh` (Chinese/Mandarin).
	- For `ja` / `zh`, you can still generate speech by phonemizing with an external G2P and then using `generate_from_phonemes` (see `kokoro_generate_from_phonemes.py` in Python samples for a reference flow).

## SpeechT5 setup

Install export tools and export SpeechT5 with vocoder:

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
optimum-cli export openvino --model microsoft/speecht5_tts --model-kwargs "{\"vocoder\": \"microsoft/speecht5_hifigan\"}" speecht5_tts
```

Create a speaker embedding file (SpeechT5-specific):

`python ../../python/speech_generation/create_speaker_embedding.py`

## Kokoro setup

**TODO! Add optimum-intel cmd here when available!**

Kokoro can use `espeak-ng` in a couple of different ways:

- English (`en-us`, `en-gb`): `espeak-ng` is used as fallback for unknown/out-of-dictionary words. See `kokoro_phonemize_fallback` sample to understand how to use an OpenVINO fallback model to avoid use of `espeak-ng` for English.
- Non-English (`es`, `fr-fr`, `hi`, `it`, `pt-br`): `espeak-ng` is the primary engine used for G2P (phonemization) step. So, it is required to be installed for E2E text-to-speech generation cases for non-english languages. Note that application can replace default G2P step with another phonemizer and call `generate_from_phonemes` API directly.

You can install `espeak-ng` by following the official guide [here](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md).

## Run samples

Follow [Get Started with Samples](https://docs.openvino.ai/2026/get-started/learn-openvino/openvino-samples/get-started-demos.html)
to run the sample.

Text-to-speech sample (SpeechT5):

`text-to-speech speecht5_tts "Hello OpenVINO GenAI" speaker_embedding.bin`

Text-to-speech sample (Kokoro):

`text-to-speech standalone_python_ov/Kokoro-82M "Hello from Kokoro in OpenVINO GenAI" --voice af_heart --language en-us`

Text-to-speech sample (Kokoro, non-English):

`text-to-speech standalone_python_ov/Kokoro-82M "Los partidos políticos tradicionales compiten con los populismos." --voice ef_dora --language es`

Kokoro fallback sample with OV fallback model:

Prepare OV fallback models:

US:

`optimum-cli export openvino --model PeterReid/graphemes_to_phonemes_en_us --task text2text-generation graphemes_to_phonemes_en_us-ov`

GB:

`optimum-cli export openvino --model PeterReid/graphemes_to_phonemes_en_gb --task text2text-generation graphemes_to_phonemes_en_gb-ov`

US model + `en-us`:

`kokoro_phonemize_fallback standalone_python_ov/Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --voice af_heart --language en-us --phonemize_fallback_model_dir graphemes_to_phonemes_en_us-ov`

GB model + `en-gb`:

`kokoro_phonemize_fallback standalone_python_ov/Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --voice af_heart --language en-gb --phonemize_fallback_model_dir graphemes_to_phonemes_en_gb-ov`

Kokoro fallback sample with default `espeak-ng` fallback:

`kokoro_phonemize_fallback standalone_python_ov/Kokoro-82M "Vellorin traded copperchimes for rainmint at Candlehaven." --voice af_heart --language en-us`

Set `--language` to match the fallback model variant (`en-us` with `..._en_us-ov`, `en-gb` with `..._en_gb-ov`).
OpenVINO fallback models above are an English-only feature (`en-us` / `en-gb`). For non-English Kokoro languages, phonemization is handled directly by `espeak-ng` as the primary G2P path (this fallback-model feature is not used).

Refer to [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-generation-models) for model details.

# Text-to-speech API usage

```c++
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

ov::genai::Text2SpeechPipeline pipe(models_path, device);
gen_speech = pipe.generate(prompt, speaker_embedding);

// Kokoro voice-based generation (speaker embedding not required)
gen_speech = pipe.generate(prompt,
													 ov::Tensor(),
													 ov::AnyMap{{"voice", "af_heart"},
																			{"language", "en-us"}});

auto speech = gen_speech.speeches[0];
// speech tensor contains the waveform of the spoken phrase
```
