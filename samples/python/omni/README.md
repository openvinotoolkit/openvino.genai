# Qwen3-Omni Chat Sample (Python)

> **Preview:** The Qwen3-Omni API (`OmniPipeline` and related types) is a preview feature and is subject to change in future releases.

This example demonstrates interactive multimodal chat with Qwen3-Omni models: text, image, and audio input producing text and optionally synthesized speech output. The sample features `openvino_genai.OmniPipeline` and configures it for the chat scenario using the `ChatHistory` API.

The following are sample files:
 - [`qwen3_omni_chat.py`](./qwen3_omni_chat.py) demonstrates multimodal chat with optional speech synthesis.
 - [`audio_placement.py`](./audio_placement.py) shows where an audio lands in the prompt, using `<ov_genai_audio_N>` tags.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
```

Then export a Qwen3-Omni model to OpenVINO format using the Optimum Intel CLI or Python API.

Install [deployment-requirements.txt](../../deployment-requirements.txt) via `pip install -r ../../deployment-requirements.txt` to run the sample.

## Run the sample

```sh
python qwen3_omni_chat.py <MODEL_DIR> <IMAGE_FILE_OR_DIR> [--audio AUDIO_WAV]
```

**Parameters:**
- `<MODEL_DIR>` — Path to the exported Qwen3-Omni OpenVINO model directory.
- `<IMAGE_FILE_OR_DIR>` — Path to an input image or a directory of images for visual context.
- `--audio AUDIO_WAV` — Path to an input audio file (16kHz mono WAV). Repeat the flag to pass several; refer to them in a question as `<ov_genai_audio_0>`, `<ov_genai_audio_1>`, and so on.

**Example:**

```sh
python qwen3_omni_chat.py ./qwen3-omni-ov ./coco.jpg --audio ./audio.wav
```

## Audio placement

Every audio you pass gets an index, starting at zero. Write `<ov_genai_audio_N>` in the prompt to
put audio N at that exact position:

```sh
python audio_placement.py ./qwen3-omni-ov ./first.wav ./second.wav
```

Omit the tags and the audio is prepended to your text instead. That default is also the layout
Qwen3-Omni was trained on, so prefer it unless you specifically need the audio elsewhere. Placing
audio between spans of text works, but it is off the training distribution and answers may be
weaker.

Media attaches to the turn you supply it on. On later turns pass an empty audio list and refer to
earlier audio through the chat history rather than sending the tensors again.

Images are loaded once at startup and available to all turns. Type questions and press Enter; the model responds with streaming text and, when speech output is enabled, 24kHz mono PCM samples in `OmniDecodedResults.speech_result.waveforms`. Press Ctrl+D to exit.

## Speech synthesis

Text decoding (the "thinker" phase) and speech output (the "talker" phase) are configured separately:

```python
text_config = openvino_genai.GenerationConfig()
text_config.max_new_tokens = 256

talker_speech_config = openvino_genai.OmniTalkerSpeechConfig(model_dir)
talker_speech_config.return_audio = True  # Enable speech synthesis
talker_speech_config.speaker = "Cherry"   # Select voice (optional)
```

Available voices vary by checkpoint. MoE models typically expose `"Ethan"`, `"Chelsie"`, `"Aiden"`, `"Cherry"`. Check `talker_config.speaker_id` in the model's `config.json` for the full list. Leaving `speaker` empty selects the default voice. Set `talker_speech_config.return_audio = False` for text-only responses.

## GPU inference

Qwen3-Omni speech synthesis requires FP32 precision on GPU — FP16 causes numerical drift that corrupts codec tokens and distorts audio. The pipeline enforces `INFERENCE_PRECISION_HINT=f32` automatically for GPU devices. Change the device by passing it to the `OmniPipeline` constructor:

```python
pipe = openvino_genai.OmniPipeline(model_dir, "GPU")
```

MoE models (30B-A3B) may exceed available GPU memory; use CPU for MoE inference.

## See Also

- [C++ Qwen3-Omni chat sample](../../cpp/omni/)
- [Qwen3-Omni model documentation](https://github.com/QwenLM/Qwen3-Omni)
