# Qwen3-Omni Chat Sample

> **Preview:** The Qwen3-Omni API (`OmniPipeline` and related types) is a preview feature and is subject to change in future releases.

Interactive multimodal chat with Qwen3-Omni models supporting text, image, and speech input/output.

## Description

Demonstrates `ov::genai::OmniPipeline` for end-to-end multimodal conversations. Qwen3-Omni models accept text, images, and audio inputs, generating text and optionally synthesized speech responses.

**Key features:**
- Multi-turn conversations with image context
- Optional speech synthesis (talker mode)
- Separate generation configs for text decoding and speech output
- Interactive CLI interface

## Prerequisites

Prepare image files (JPG, PNG) and audio files (16kHz mono WAV) for testing.

## Usage

```bash
omni_chat <MODEL_DIR> <IMAGE_FILE_OR_DIR> <AUDIO_FILE>
```

**Parameters:**
- `<MODEL_DIR>` — Path to exported Qwen3-Omni OpenVINO model directory
- `<IMAGE_FILE_OR_DIR>` — Path to input image(s) for visual context
- `<AUDIO_FILE>` — Path to input audio file (16kHz mono WAV)

**Example:**

```bash
./build/samples/cpp/omni/omni_chat /models/qwen3-omni-ov ./coco.jpg ./audio.wav
```

**Interactive usage:**
1. Images are loaded once at startup and available to all turns
2. Type questions and press Enter
3. Model responds with streaming text; speech output indicated when enabled
4. Continue the conversation across multiple turns
5. Press Ctrl+D to exit

## Model Support

Compatible with Qwen3-Omni models exported to OpenVINO format.

Export using [Optimum Intel](https://github.com/huggingface/optimum-intel) CLI or Python API.

## Configuration

### Text Generation

`GenerationConfig` controls text decoding (the "thinker" phase):

```cpp
ov::genai::GenerationConfig text_config;
text_config.max_new_tokens = 256;
```

### Speech Synthesis

`OmniTalkerSpeechConfig` controls talker and speech output:

```cpp
ov::genai::OmniTalkerSpeechConfig talker_speech_config(models_path);
talker_speech_config.return_audio = true;  // Enable speech synthesis
talker_speech_config.speaker = "Cherry";   // Select voice (optional)
```

**Available voices** vary by checkpoint. MoE models typically expose: `"Ethan"`, `"Chelsie"`, `"Aiden"`, `"Cherry"`. Check `talker_config.speaker_id` in the model's `config.json` for the full list. Leaving `speaker` empty selects the default voice.

**Speech output:** 24kHz mono PCM samples returned in `OmniDecodedResults.speech_result.waveforms`.

## GPU Inference

**Important:** Qwen3-Omni speech synthesis requires FP32 precision on GPU. FP16 causes numerical drift → codec token corruption → distorted audio. The pipeline enforces `INFERENCE_PRECISION_HINT=f32` automatically for GPU devices. Change the device argument in `OmniPipeline` constructor:

```cpp
ov::genai::OmniPipeline pipe(models_path, "GPU");
```

**MoE models** (30B-A3B) may exceed available GPU memory. Use CPU for MoE inference.

## Speaker API Features

The main sample demonstrates speaker enumeration and voice blending at startup:

1. **List speakers** — shows all named voices in the model
2. **Voice blending** — when 2+ speakers are available, creates a 50/50 blend and uses it for generation

**Custom blending:** Retrieve speaker embeddings via `pipe.get_talker()->get_speaker_embedding(name)`, mix with custom ratios (e.g., 0.75A + 0.25B), and pass via `OmniTalkerSpeechConfig::speaker`.

## See Also

- [Python Qwen3-Omni chat sample](../../python/omni/)
- [OmniPipeline C++ API reference](../../../src/cpp/include/openvino/genai/omni/pipeline.hpp)
- [Qwen3-Omni model documentation](https://github.com/QwenLM/Qwen3-Omni)
