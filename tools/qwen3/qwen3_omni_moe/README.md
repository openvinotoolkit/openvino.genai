# Qwen3-Omni-MOE to OpenVINO Converter

Converts Qwen3-Omni-MOE and Qwen3-Omni models to OpenVINO IR format.

## Prerequisites

- Python 3.12+
- transformers: `pip install git+https://github.com/huggingface/transformers@3d1a4f5e34753e51cb85052539c6ef10cab9a5c1`
- openvino
- torch
- nncf (for int8/int4 quantization)

## Usage

```bash
python3 -m tools.qwen3.qwen3_omni_moe.convert --model_id <MODEL_ID> --output_dir <OUTPUT_DIR> [--weight_format fp16|fp32|int8|int4] [--use_local_dir]
```

## Usage Examples

### Basic conversion (FP16)

```bash
python3 -m tools.qwen3.qwen3_omni_moe.convert \
    --model_id Qwen/Qwen3-Omni-MOE \
    --output_dir ./qwen3-omni-ov/FP16
```

### INT4 quantized conversion

```bash
python3 -m tools.qwen3.qwen3_omni_moe.convert \
    --model_id Qwen/Qwen3-Omni-MOE \
    --output_dir ./qwen3-omni-ov/INT4 \
    --weight_format int4
```

### Local model path

```bash
python3 -m tools.qwen3.qwen3_omni_moe.convert \
    --model_id /path/to/local/model \
    --output_dir ./qwen3-omni-ov/FP16
```

### Dense (non-MoE) variant

```bash
python3 -m tools.qwen3.qwen3_omni_moe.convert \
    --model_id /path/to/Qwen3-Omni-4B-Instruct \
    --output_dir ./qwen3-omni-4b-ov/FP16
```

## Output Structure

The converter produces these OpenVINO IR models:

- `openvino_text_embeddings_model.xml` - Thinker token embeddings
- `openvino_audio_encoder_model.xml` - Audio encoder (AuT)
- `openvino_vision_embeddings_model.xml` - Vision patch embeddings
- `openvino_vision_embeddings_merger_model.xml` - Vision merger with deepstack
- `openvino_language_model.xml` - Thinker language model (stateful, with KV-cache)
- `openvino_talker_embeddings_model.xml` - Talker token embeddings
- `openvino_talker_model.xml` - Talker language model (stateful)
- `openvino_code_predictor_model.xml` - Code predictor (stateful)
- `openvino_code2wav_model.xml` - Code-to-waveform vocoder

Config files: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `processor_config.json`, `chat_template.jinja`

## Arguments

| Argument        | Required | Default | Description                              |
|----------------|----------|---------|------------------------------------------|
| `--model_id`   | Yes      | -       | HuggingFace model ID or local path       |
| `--output_dir` | Yes      | -       | Output directory for converted models    |
| `--weight_format` | No    | fp16    | Weight format: fp16, fp32, int8, int4    |
| `--use_local_dir` | No   | false   | Download model to output_dir/ckpt first  |

## Notes

- Quantization (int8/int4) only applies to language models (thinker, talker, code predictor)
- The converter skips already-converted models (checks for existing .xml files)
- Dense models (e.g. Qwen3-Omni-4B) are supported natively via `Qwen3OmniForConditionalGeneration`
