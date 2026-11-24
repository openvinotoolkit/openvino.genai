# Text-to-speech Python samples

This folder contains Python examples for `openvino_genai.Text2SpeechPipeline`:

- `text2speech.py`: basic text → audio generation (SpeechT5 and Kokoro)
- `kokoro_phonemize_fallback.py`: Kokoro unknown-word fallback behavior
- `qwen3_tts.py`: unified Qwen3-TTS sample with subcommands for Base, CustomVoice, and VoiceDesign
- `qwen3_tts_input_streaming.py`: Qwen3-TTS speaking text while it arrives (input streaming with `Text2SpeechStreamingSession`)

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
    - Three model variants are covered by one sample with subcommands:
        - `qwen3_tts.py customvoice`: built-in speaker identities, with optional style `instruct`.
        - `qwen3_tts.py voice-design`: create a brand-new voice from a natural-language `instruct` description.
        - `qwen3_tts.py base`: clone a voice from a short reference recording (x-vector and ICL modes).
    - See the [Qwen3-TTS samples](#qwen3-tts-samples) section below for setup and run commands.

## Install dependencies

- Export-time deps:

    `pip install -r ../../export-requirements.txt`

    Some newer models may require the latest Optimum Intel. If needed, run:

    ```sh
    pip install "optimum-intel @ git+https://github.com/huggingface/optimum-intel.git@main"
    ```

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
pip install kokoro
optimum-cli export openvino -m hexgrad/Kokoro-82M ov_Kokoro-82M --trust-remote-code
```

> **Note:**
> After export is complete, you will find the available speaker embedding `.bin` files in `ov_Kokoro-82M/voices`.

## Qwen3-TTS setup

Export a Qwen3-TTS model to OpenVINO (choose the variant matching the sample you want to run), for example:

```sh
pip install qwen-tts transformers==4.57.3 torch==2.11.0 torchaudio==2.11.0
optimum-cli export openvino --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --trust-remote-code qwen3_tts_customvoice_ov
optimum-cli export openvino --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --trust-remote-code qwen3_tts_voicedesign_ov
optimum-cli export openvino --model Qwen/Qwen3-TTS-12Hz-0.6B-Base         --trust-remote-code qwen3_tts_base_ov
```

Use `--language` to set the language. The supported languages are: ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", and "Italian"]. "Auto" will be used by default if not specified.

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

> **Note:** `text2speech.py` targets SpeechT5 and Kokoro. Qwen3-TTS is covered by
> `qwen3_tts.py` — see the [Qwen3-TTS samples](#qwen3-tts-samples) section.

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

Qwen3-TTS ships as three model variants, and this folder provides one unified sample:

| Sample | Subcommand | Model variant | What it showcases |
| --- | --- | --- | --- |
| `qwen3_tts.py` | `customvoice` | CustomVoice | Built-in speaker identities, optional style `instruct` |
| `qwen3_tts.py` | `voice-design` | VoiceDesign | A new voice created from a natural-language `instruct` description |
| `qwen3_tts.py` | `base` | Base | Voice cloning from reference audio (x-vector and ICL modes) |

See the [Qwen3-TTS setup](#qwen3-tts-setup) section above for export commands.

### 3) `qwen3_tts.py customvoice`

Speak with one of the model's built-in speakers.

```
python qwen3_tts.py customvoice qwen3_tts_customvoice_ov "Hello from Qwen3 CustomVoice." --speaker ryan
```

`--instruct` is optional and steers tone/emotion/pace.
Note: Only supported in 1.7B variant of CustomVoice -- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`

```
python qwen3_tts.py customvoice qwen3_tts_customvoice_ov "Hello from Qwen3 CustomVoice." --speaker ryan --instruct "Speak in a calm, professional tone."
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

### 4) `qwen3_tts.py voice-design`

Design a new voice purely from a natural-language description. There is no speaker list; `--instruct` is required.

```
python qwen3_tts.py voice-design qwen3_tts_voicedesign_ov "H-hey! You dropped your... uh... calculus notebook? I mean, I think it's yours? Maybe?" --instruct "Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous"
```

### 5) `qwen3_tts.py base`

Clone a voice from a short reference recording.

You can download this 24 kHz reference audio file to use for following examples:
https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav

Save this file locally as clone.wav, then pass it via `--ref_audio_wav_path`.

Note: The reference audio must be 24 kHz.

Two modes are selected automatically from the inputs:

- **x-vector mode** (fast, identity only): provide reference audio (or a pre-saved speaker embedding). `--ref_text` is not required.
- **ICL mode** (higher fidelity): additionally provide the reference transcript via `--ref_text`.

`--ref_text` is the transcript for the reference recording given in `--ref_audio_wav_path`.

If you're using the example audio file given above, then `--ref_text` should be set to: "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

Clone directly from reference audio (x-vector mode):
```
python qwen3_tts.py base qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path clone.wav
```

Clone from reference audio + transcript (ICL mode):
```
python qwen3_tts.py base qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path clone.wav --ref_text "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
```

#### Reusing a reference prompt (save once, reuse many times)

Extracting the speaker embedding and reference codes from audio is the expensive part of cloning. To avoid
recomputing them on every run, clone once from reference audio and save the artifacts that `generate(...)`
returns on the result (`speaker_embedding` and `ref_codec_ids`):

For x-vector mode: Save speaker embedding:
```
python qwen3_tts.py base qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path clone.wav --save_speaker_embedding_file_path qwen_speaker_embedding.bin
```

For ICL mode: Save speaker embedding & ref codec ids:
```
python qwen3_tts.py base qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path clone.wav --ref_text "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you." --save_speaker_embedding_file_path qwen_speaker_embedding.bin --save_ref_codec_ids_file_path qwen_ref_code.bin
```

For x-vector mode: Reuse the saved speaker embedding (ref-audio not needed anymore):
```
python qwen3_tts.py base qwen3_tts_base_ov "Hello again." --speaker_embedding_file_path qwen_speaker_embedding.bin
```

For ICL mode: Reuse saved embedding + reference codes (ref-audio not needed anymore):
Note: `--ref_text` is still required here.
```
python qwen3_tts.py base qwen3_tts_base_ov "Hello again." --speaker_embedding_file_path qwen_speaker_embedding.bin --ref_codec_ids_file_path qwen_ref_code.bin --ref_text "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
```

For Qwen3-TTS Base examples above, output is 24 kHz WAV.

### 6) `qwen3_tts_input_streaming.py`: speak text while it arrives

`Text2SpeechStreamingSession` speaks text that arrives in pieces, for example an LLM reply while it is being generated,
and returns the audio as soon as it is ready. `push()` only buffers text, `read()` returns the next audio (and runs the
inference that produces it on the calling thread), `end()` closes the input, and the last result has `is_final == True`.
The sample pushes a text word by word from a producer thread at a chosen rate, and reads from the main thread:

```
python qwen3_tts_input_streaming.py qwen3_tts_customvoice_ov "Hello from Qwen3 CustomVoice. This text is pushed word by word, and spoken while it arrives." --speaker ryan --tokens_per_second 10
```

It prints a timeline of the audio it gets back, the time to first audio, and writes `output_streaming_audio.wav`. `--compare`
also runs `generate()` on the whole text and checks that the audio is identical. The voice options are those of
`qwen3_tts.py` (`--speaker`, `--instruct`, `--ref_audio_wav_path` with `--ref_text` for voice cloning, ...).

```python
import openvino_genai

pipe = openvino_genai.Text2SpeechPipeline(model_dir, "CPU", ATTENTION_BACKEND="SDPA")
session = openvino_genai.Text2SpeechStreamingSession(pipe, speaker="ryan")

for text_piece in llm_reply_stream():      # for example, from another thread
    session.push(text_piece)
    while (result := session.read()) is not None:
        play(result.speech)                # float32 [N] at result.output_sample_rate
session.end()
while (result := session.read()) is not None:
    play(result.speech)                    # the last one has result.is_final == True
```

Things to know:

- Input streaming needs the SDPA backend: create the pipeline with `ATTENTION_BACKEND="SDPA"` (the default
  PagedAttention backend does not support it yet).
- One session at a time per pipeline, and `generate()` cannot be called while a session is alive.
- A session always uses the streaming prompt layout (`non_streaming_mode=False`). With the same seed it produces the
  same audio as `generate(..., non_streaming_mode=False)` on the whole text, however the text is split into `push()` calls,
  as long as the pieces end at word boundaries and the stateful codec decoder is used (the default when the model has it).
  The default `non_streaming_mode=True` of `generate()` gives the model the whole text up front, which a session cannot do,
  so its audio differs.
- The model is only allowed to end the utterance once the text has ended, so an early end-of-speech cannot cut the text short.
- Words are held back until a following space shows where they end (`Text2SpeechStreamingConfig.max_holdback_chars`
  bounds this for text without spaces). Voice cloning needs enough text to cover the reference transcript before it can
  start, or `end()`.

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
