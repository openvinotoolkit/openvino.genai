# Qwen3-Omni CLI Chat

Interactive CLI chat using the Qwen3-Omni-4B model with PyTorch.
Supports text, image, audio, and video inputs. Generates text and audio outputs.

## Requirements

- Python 3.10+
- PyTorch
- transformers: `pip install git+https://github.com/huggingface/transformers@3d1a4f5e34753e51cb85052539c6ef10cab9a5c1`
- qwen-omni-utils: `pip install qwen-omni-utils -U`
- numpy

## Model Download

```bash
wget https://multimodal-dialog.oss-cn-hangzhou.aliyuncs.com/yiru/Qwen3-Omni-Release/Qwen3-Omni-4B-Instruct-multilingual.tar \
  -P temp/llm_cache/hf_models/

tar xf temp/llm_cache/hf_models/Qwen3-Omni-4B-Instruct-multilingual.tar \
  -C temp/llm_cache/hf_models/
```

## Usage

### Interactive Chat

```bash
python -m tools.qwen3.qwen3_chat \
  --model_path temp/llm_cache/hf_models/Qwen3-Omni-4B-Instruct-multilingual
```

### Text-only (no audio output, saves ~2GB GPU memory)

```bash
python -m tools.qwen3.qwen3_chat \
  --model_path temp/llm_cache/hf_models/Qwen3-Omni-4B-Instruct-multilingual \
  --no_audio
```

### Demo Mode (smoke test)

Runs 3 predefined text prompts and exits:

```bash
python -m tools.qwen3.qwen3_chat \
  --model_path temp/llm_cache/hf_models/Qwen3-Omni-4B-Instruct-multilingual \
  --demo
```

### Select Speaker Voice

```bash
python -m tools.qwen3.qwen3_chat \
  --model_path temp/llm_cache/hf_models/Qwen3-Omni-4B-Instruct-multilingual \
  --speaker Chelsie
```

## CLI Arguments

| Argument       | Default | Description                                          |
|----------------|---------|------------------------------------------------------|
| `--model_path` | —       | Path to the model directory (required)               |
| `--device`     | `auto`  | Device map for model loading                         |
| `--no_audio`   | `false` | Disable audio output at load time (saves GPU memory) |
| `--output_dir` | `output`| Directory for saved audio files                      |
| `--speaker`    | —       | Speaker voice name (model default if not set)        |
| `--demo`       | `false` | Run text-only smoke test and exit                    |

## Chat Commands

| Command          | Description                                         |
|------------------|-----------------------------------------------------|
| `/image <path>`  | Attach an image (use quotes for paths with spaces)  |
| `/audio <path>`  | Attach an audio file                                |
| `/video <path>`  | Attach a video file                                 |
| `/clear`         | Clear chat history                                  |
| `/help`          | Show available commands                             |
| `/quit`          | Exit chat                                           |

### Example Session

```
You: Hello!
Qwen: Hello! How can I help you today?

You: /image photos/cat.jpg What do you see in this image?
Qwen: I can see a cute cat sitting on a windowsill...
[Audio saved to output/turn_001.wav]

You: /audio recording.wav What is being said?
Qwen: The audio contains someone saying...
[Audio saved to output/turn_002.wav]

You: /clear
Chat history cleared.

You: /quit
Goodbye!
```
