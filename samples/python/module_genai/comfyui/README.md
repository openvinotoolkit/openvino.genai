# OpenVINO GenAI ModulePipeline for ComfyUI

This tool converts and runs ComfyUI JSON workflows using OpenVINO GenAI ModulePipeline.

## Features

- **ComfyUI JSON Support**: Supports both workflow and API JSON formats
- **Auto Model Detection**: Automatically detects model path based on JSON filename
- **Parameter Override**: Command line arguments override JSON values
- **Multiple Output Formats**: Supports video (saved_video), image (saved_image), and raw tensor outputs

## Usage

```bash
python ./samples/python/module_genai/comfyui/md_comfyui.py \
    --json <comfyui.json> [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--json` | ComfyUI JSON file (required) | - |
| `--model_path` | Model path base | Auto-detected |
| `--device` | Device to run on | GPU |
| `--prompt` | Text prompt (overrides JSON) | From JSON |
| `--width` | Image/video width | From JSON |
| `--height` | Image/video height | From JSON |
| `--num_frames` | Number of video frames | From JSON |
| `--steps` | Inference steps | From JSON |
| `--guidance` | Guidance scale | From JSON |
| `--seed` | Random seed | From JSON |
| `--tile_size` | VAE decoder tile size | None |
| `--verbose` | Verbosity (0-3) | 2 |
| `--debug` | Enable debug output | False |

## Examples

### Wan 2.1 Text-to-Video

```bash
# Basic usage
python ./samples/python/module_genai/comfyui/md_comfyui.py \
    --json ./samples/cpp/module_genai/comfyui/wan2.1_t2v.json \
    --model_path ./samples/cpp/module_genai/ut_pipelines/Wan2.1-T2V-1.3B-Diffusers \
    --device GPU

# With custom parameters (reduce for lower VRAM)
python ./samples/python/module_genai/comfyui/md_comfyui.py \
    --json ./samples/cpp/module_genai/comfyui/wan2.1_t2v.json \
    --model_path ./samples/cpp/module_genai/ut_pipelines/Wan2.1-T2V-1.3B-Diffusers \
    --device GPU \
    --width 480 --height 272 --num_frames 17

# Override prompt
python ./samples/python/module_genai/comfyui/md_comfyui.py \
    --json ./samples/cpp/module_genai/comfyui/wan2.1_t2v.json \
    --model_path ./samples/cpp/module_genai/ut_pipelines/Wan2.1-T2V-1.3B-Diffusers \
    --device GPU \
    --prompt "A cat playing piano"
```

### Z-Image Generation

```bash
# Basic usage
python ./samples/python/module_genai/comfyui/md_comfyui.py \
    --json ./samples/cpp/module_genai/comfyui/z_image_turbo_2k_non_tiled_api.json \
    --model_path ./samples/cpp/module_genai/ut_pipelines/Z-Image-Turbo-fp16-ov \
    --device GPU

# High resolution with tiling
python ./samples/python/module_genai/comfyui/md_comfyui.py \
    --json ./samples/cpp/module_genai/comfyui/z_image_turbo_2k_tiled.json \
    --model_path ./samples/cpp/module_genai/ut_pipelines/Z-Image-Turbo-fp16-ov \
    --device GPU \
    --tile_size 128

# Custom size
python ./samples/python/module_genai/comfyui/md_comfyui.py \
    --json ./samples/cpp/module_genai/comfyui/z_image_turbo_2k_non_tiled_api.json \
    --model_path ./samples/cpp/module_genai/ut_pipelines/Z-Image-Turbo-fp16-ov \
    --device GPU \
    --width 1024 --height 1024 --steps 4
```

### Debug Mode

```bash
# Enable debug output to see YAML configuration
python ./samples/python/module_genai/comfyui/md_comfyui.py \
    --json ./samples/cpp/module_genai/comfyui/wan2.1_t2v.json \
    --model_path ./samples/cpp/module_genai/ut_pipelines/Wan2.1-T2V-1.3B-Diffusers \
    --device GPU --debug
```

## Auto Model Path Detection

The tool automatically detects model paths based on JSON filename patterns:

| Filename Pattern | Model Path |
|-----------------|------------|
| `wan2.1*` | `ut_pipelines/Wan2.1-T2V-1.3B-Diffusers` |
| `z_image*`, `zimage*` | `ut_pipelines/Z-Image-Turbo-fp16-ov` |

You can always override with `--model_path`.

## Output Handling

The tool checks outputs in this priority order:

1. **saved_video** - Video file saved by SaveVideoModule
2. **saved_image** - Image file saved by SaveImageModule
3. **image** - Raw tensor, saved as PNG

## Generated Files

When running, the tool generates:

- `<json_file>.generated.yaml` - The converted YAML pipeline configuration

## GPU Memory Requirements (Wan 2.1)

| Resolution | Frames | Estimated VRAM |
|------------|--------|----------------|
| 480×272 | 17 | ~7-9 GB |
| 512×288 | 17 | ~8-10 GB |
| 640×360 | 17 | ~10-12 GB |
| 832×480 | 33 | ~16+ GB |

For Arc 770 (16GB), use:
```bash
python ./samples/python/module_genai/comfyui/md_comfyui.py \
    --json ./samples/cpp/module_genai/comfyui/wan2.1_t2v.json \
    --model_path ./samples/cpp/module_genai/ut_pipelines/Wan2.1-T2V-1.3B-Diffusers \
    --device GPU \
    --width 480 --height 272 --num_frames 17
```

## Troubleshooting

### OOM (Out of Memory) Error

If you see `CL_OUT_OF_RESOURCES`, reduce resolution and frames:
```bash
python ./samples/python/module_genai/comfyui/md_comfyui.py \
    --json ./samples/cpp/module_genai/comfyui/wan2.1_t2v.json \
    --model_path ./samples/cpp/module_genai/ut_pipelines/Wan2.1-T2V-1.3B-Diffusers \
    --device GPU \
    --width 480 --height 272 --num_frames 17
```

### NaN in Transformer Output

This is usually caused by T5 encoder running on GPU. The tool automatically forces T5/UMT5 encoders to CPU for Wan 2.1 models.

### Model Not Found

Ensure the model path is correct. Use `--model_path` to specify explicitly:
```bash
python ./samples/python/module_genai/comfyui/md_comfyui.py \
    --json ./samples/cpp/module_genai/comfyui/wan2.1_t2v.json \
    --model_path ./samples/cpp/module_genai/ut_pipelines/Wan2.1-T2V-1.3B-Diffusers
```
