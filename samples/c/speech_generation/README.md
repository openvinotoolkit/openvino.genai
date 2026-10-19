# Text to speech in C

Download an exported model such as [OpenVINO/Kokoro-82M-int8-ov](https://huggingface.co/OpenVINO/Kokoro-82M-int8-ov). Keep `config.json`, `openvino_model.xml`, `openvino_model.bin`, the selected `voices/*.bin` file, and the English lexicon files in `data/` together in the model directory.

Build this sample with `ENABLE_SAMPLES=ON`, then run:

```sh
text2speech_c MODEL_DIR "Hello from OpenVINO" MODEL_DIR/voices/af_heart.bin output_audio.wav --device CPU
```

To generate two waveforms in one call with the same speaker embedding (optionally with `--speed`, `--language` and `--device` flags):

```sh
text2speech_c --batch MODEL_DIR "Hello from OpenVINO" "A second sentence" \
  MODEL_DIR/voices/af_heart.bin first.wav second.wav --speed 1.1 --language en-us
```

The sample reads a float32 speaker embedding and writes a mono 16-bit PCM WAV file. Kokoro requires an embedding. Pass `-` in its place to use the default voice of a model that provides one, such as SpeechT5. The returned waveform tensor is owned by the caller and must be released with `ov_tensor_free`.

For SpeechT5, [llmware/speech-t5-tts-ov](https://huggingface.co/llmware/speech-t5-tts-ov) provides an exported model with `openvino_tokenizer.xml` and `openvino_tokenizer.bin`, which are required for text input.

## Qwen3-TTS in C

Qwen3-TTS ships as three model variants, and this folder provides one unified C sample:

| Sample | Subcommand | Model variant | What it showcases |
| --- | --- | --- | --- |
| `qwen3_tts_c` | `customvoice` | CustomVoice | Built-in speaker identities, optional style `instruct` |
| `qwen3_tts_c` | `voice-design` | VoiceDesign | A new voice created from a natural-language `instruct` description |
| `qwen3_tts_c` | `base` | Base | Voice cloning from reference audio (x-vector and ICL modes) |

### Qwen3-TTS setup

Convert a Qwen3-TTS model to OpenVINO (choose the variant matching the sample you want to run), for example:

```sh
pip install qwen-tts transformers==4.57.3 torch==2.11.0 torchaudio==2.11.0
pip install --upgrade-strategy eager -r ../../export-requirements.txt
optimum-cli export openvino --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --trust-remote-code qwen3_tts_customvoice_ov
optimum-cli export openvino --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --trust-remote-code qwen3_tts_voicedesign_ov
optimum-cli export openvino --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --trust-remote-code qwen3_tts_base_ov
```

Use `--language` to set the language. The supported languages are: ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", and "Italian"]. "Auto" will be used by default if not specified.

### `qwen3_tts_c customvoice`

Speak with one of the model's built-in speakers.

```sh
qwen3_tts_c customvoice qwen3_tts_customvoice_ov "Hello from Qwen3 CustomVoice." --speaker ryan
```

`--instruct` is optional and steers tone/emotion/pace.
Note: Only supported in 1.7B variant of CustomVoice -- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`

```sh
qwen3_tts_c customvoice qwen3_tts_customvoice_ov "Hello from Qwen3 CustomVoice." --speaker ryan --instruct "Speak in a calm, professional tone."
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

### `qwen3_tts_c voice-design`

Design a new voice purely from a natural-language description. There is no speaker list; `--instruct` is required.

```sh
qwen3_tts_c voice-design qwen3_tts_voicedesign_ov "H-hey! You dropped your... uh... calculus notebook? I mean, I think it's yours? Maybe?" --instruct "Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous"
```

### `qwen3_tts_c base`

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
```sh
qwen3_tts_c base qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path clone.wav
```

Clone from reference audio + transcript (ICL mode):
```sh
qwen3_tts_c base qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path clone.wav --ref_text "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
```

#### Reusing a reference prompt (save once, reuse many times)

Extracting the speaker embedding and reference codes from audio is the expensive part of cloning. To avoid recomputing them on every run, clone once from reference audio and save the artifacts that the result returns (`speaker_embedding` and `ref_codec_ids`):

For x-vector mode: Save speaker embedding:
```sh
qwen3_tts_c base qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path clone.wav --save_speaker_embedding_file_path qwen_speaker_embedding.bin
```

For ICL mode: Save speaker embedding & ref codec ids:
```sh
qwen3_tts_c base qwen3_tts_base_ov "Hello from Qwen3 Base." --ref_audio_wav_path clone.wav --ref_text "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you." --save_speaker_embedding_file_path qwen_speaker_embedding.bin --save_ref_codec_ids_file_path qwen_ref_code.bin
```

For x-vector mode: Reuse the saved speaker embedding (ref-audio not needed anymore):
```sh
qwen3_tts_c base qwen3_tts_base_ov "Hello again." --speaker_embedding_file_path qwen_speaker_embedding.bin
```

For ICL mode: Reuse saved embedding + reference codes (ref-audio not needed anymore):
Note: `--ref_text` is still required here.
```sh
qwen3_tts_c base qwen3_tts_base_ov "Hello again." --speaker_embedding_file_path qwen_speaker_embedding.bin --ref_codec_ids_file_path qwen_ref_code.bin --ref_text "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
```

The saved files use simple flat-binary layouts owned by this sample (the speaker embedding is raw float32; the reference codes carry a small shape header).

For Qwen3-TTS Base examples above, output is 24 kHz WAV.

Refer to [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-generation-models) for model details.
