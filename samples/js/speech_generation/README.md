# Text-to-speech samples (JavaScript)

This folder contains JavaScript examples for `Text2SpeechPipeline` from `openvino-genai-node`:

- `text2speech.js`: basic text-to-audio generation (SpeechT5 and Kokoro)
- `qwen3_tts.js`: unified Qwen3-TTS sample with subcommands for Base, CustomVoice, and VoiceDesign

## Supported models

- **SpeechT5**
	- Requires exported SpeechT5 model and vocoder.
	- Usually uses a speaker embedding file.
- **Kokoro**
	- Uses a Kokoro model directory.
	- Uses `--speaker_embedding_file_path` and `--language` options.
	- End-to-end Kokoro language support includes: `en-us`, `en-gb`, `es`, `fr-fr`, `hi`, `it`, `pt-br`.
	- Not yet supported for end-to-end text generation in this flow: `ja`, `zh`.
- **Qwen3-TTS**
	- Three model variants are covered by one sample with subcommands:
		- `qwen3_tts.js customvoice`: built-in speaker identities, with optional style `instruct`.
		- `qwen3_tts.js voice-design`: create a brand-new voice from a natural-language `instruct` description.
		- `qwen3_tts.js base`: clone a voice from a short reference recording (x-vector and ICL modes).

## Export dependencies

```sh
pip install -r <GENAI_ROOT_DIR>/samples/export-requirements.txt
```

Some newer models may require the latest Optimum Intel. If needed, run:

```sh
pip install "optimum-intel @ git+https://github.com/huggingface/optimum-intel.git@main"
```
## SpeechT5 setup

```sh
optimum-cli export openvino --model microsoft/speecht5_tts --model-kwargs "{\"vocoder\": \"microsoft/speecht5_hifigan\"}" speecht5_tts
```

Create a speaker embedding file (SpeechT5-specific):

```sh
python ../python/speech_generation/create_speaker_embedding.py
```

## Kokoro setup

```sh
pip install kokoro
optimum-cli export openvino -m hexgrad/Kokoro-82M ov_Kokoro-82M --trust-remote-code
```

After export, speaker embedding `.bin` files are in `ov_Kokoro-82M/voices`.

## Qwen3-TTS setup

Export a Qwen3-TTS model to OpenVINO (choose the variant matching the sample you want to run), for example:

```sh
pip install qwen-tts transformers==4.57.3 torch==2.11.0 torchaudio==2.11.0
optimum-cli export openvino --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --trust-remote-code qwen3_tts_customvoice_ov
optimum-cli export openvino --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --trust-remote-code qwen3_tts_voicedesign_ov
optimum-cli export openvino --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --trust-remote-code qwen3_tts_base_ov
```

Use `--language` to set the language. Supported values are model-dependent; commonly: `auto`, `chinese`, `english`, `japanese`, `korean`, `german`, `french`, `russian`, `portuguese`, `spanish`, `italian`.

## Run samples

### 1) `text2speech.js`

SpeechT5:

```sh
node speech_generation/text2speech.js speecht5_tts "Hello from OpenVINO GenAI" --speaker_embedding_file_path speaker_embedding.bin
```

Kokoro:

```sh
node speech_generation/text2speech.js ov_Kokoro-82M "Hello, and welcome to speech generation using OpenVINO GenAI." --speaker_embedding_file_path ov_Kokoro-82M/voices/af_heart.bin --language en-us
```

Kokoro (non-English):

```sh
node speech_generation/text2speech.js ov_Kokoro-82M "Hola y bienvenidos a la generación de voz utilizando OpenVINO GenAI." --speaker_embedding_file_path ov_Kokoro-82M/voices/ef_dora.bin --language es
```

Speed control:

```sh
node speech_generation/text2speech.js ov_Kokoro-82M "Hello from OpenVINO GenAI with a faster speaking rate." --speaker_embedding_file_path ov_Kokoro-82M/voices/af_heart.bin --language en-us --speed 1.15
```

> `text2speech.js` targets SpeechT5 and Kokoro. Qwen3-TTS is covered by `qwen3_tts.js` below.

## Qwen3-TTS samples

Qwen3-TTS ships as three model variants, and this folder provides one unified sample:

| Sample | Subcommand | Model variant | What it showcases |
| --- | --- | --- | --- |
| `qwen3_tts.js` | `customvoice` | CustomVoice | Built-in speaker identities, optional style `instruct` |
| `qwen3_tts.js` | `voice-design` | VoiceDesign | A new voice created from a natural-language `instruct` description |
| `qwen3_tts.js` | `base` | Base | Voice cloning from reference audio (x-vector and ICL modes) |

### 2) `qwen3_tts.js customvoice`

```sh
node speech_generation/qwen3_tts.js customvoice qwen3_tts_customvoice_ov "Hello from Qwen3 CustomVoice." --speaker ryan
```

`--instruct` is optional and steers tone/emotion/pace:

```sh
node speech_generation/qwen3_tts.js customvoice qwen3_tts_customvoice_ov "Hello from Qwen3 CustomVoice." --speaker ryan --instruct "Speak in a calm, professional tone."
```

### 3) `qwen3_tts.js voice-design`

```sh
node speech_generation/qwen3_tts.js voice-design qwen3_tts_voicedesign_ov "H-hey! You dropped your... uh... calculus notebook? I mean, I think it's yours? Maybe?" --instruct "Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous"
```

### 4) `qwen3_tts.js base`

Clone a voice from a short reference recording.

You can download this 24 kHz reference audio file to use for following examples:
https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav

Save this file locally as `clone.wav`, then pass it via `--ref_audio_wav_path`.

Note: The reference audio must be 24 kHz.

Two modes are selected automatically from inputs:

- **x-vector mode** (fast, identity only): provide reference audio (or a pre-saved speaker embedding). `--ref_text` is not required.
- **ICL mode** (higher fidelity): additionally provide the reference transcript via `--ref_text`.

`--ref_text` is the transcript for the reference recording given in `--ref_audio_wav_path`.

If you're using the example audio file given above, then `--ref_text` should be set to: "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

Clone directly from reference audio (x-vector mode):

```sh
node speech_generation/qwen3_tts.js base qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path clone.wav
```

Clone from reference audio + transcript (ICL mode):

```sh
node speech_generation/qwen3_tts.js base qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path clone.wav --ref_text "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
```

#### Reusing a reference prompt (save once, reuse many times)

Extracting the speaker embedding and reference codes from audio is the expensive part of cloning. To avoid
recomputing them on every run, clone once from reference audio and save the artifacts that `generate(...)`
returns on the result (`speakerEmbedding` and `refCodecIds`):

For x-vector mode: Save speaker embedding:

```sh
node speech_generation/qwen3_tts.js base qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path clone.wav --save_speaker_embedding_file_path qwen_speaker_embedding.bin
```

For ICL mode: Save speaker embedding & ref codec ids:

```sh
node speech_generation/qwen3_tts.js base qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path clone.wav --ref_text "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you." --save_speaker_embedding_file_path qwen_speaker_embedding.bin --save_ref_codec_ids_file_path qwen_ref_code.bin
```

For x-vector mode: Reuse the saved speaker embedding (ref-audio not needed anymore):

```sh
node speech_generation/qwen3_tts.js base qwen3_tts_base_ov "Hello again." --speaker_embedding_file_path qwen_speaker_embedding.bin
```

For ICL mode: Reuse saved embedding + reference codes (ref-audio not needed anymore):
Note: `--ref_text` is still required here.

```sh
node speech_generation/qwen3_tts.js base qwen3_tts_base_ov "Hello again." --speaker_embedding_file_path qwen_speaker_embedding.bin --ref_codec_ids_file_path qwen_ref_code.bin --ref_text "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
```

## Notes

- `result.outputSampleRate` is used when writing WAV, so output sample rate comes from the model/backend.
- For Qwen3 Base reference audio, `--ref_audio_wav_path` must point to a 24 kHz WAV file.
