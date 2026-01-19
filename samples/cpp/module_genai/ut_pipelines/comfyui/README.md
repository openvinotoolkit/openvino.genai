## module_pipeline_comfyui Sample Applications

A standalone tool to run OpenVINO GenAI ModulePipeline from ComfyUI JSON or YAML configuration files.

#### Features

- Convert ComfyUI API JSON workflow to GenAI YAML pipeline config
- Load and run pipelines from YAML config files
- Command-line parameter overrides for prompt, image size, steps, etc.
- Automatic image saving with timestamp

#### Usage

```bash
# Run from ComfyUI JSON
module_pipeline_comfyui --json ut_pipelines/comfyui/z_image_turbo_2k_tiled_api.json
module_pipeline_comfyui --json ut_pipelines/comfyui/z_image_turbo_2k_tiled.json

# Full options
module_pipeline_comfyui --json workflow.json \
    --model-path ./models \
    --device GPU \
    --prompt "a cat sitting on a sofa" \
    --width 1024 \
    --height 1024 \
    --steps 4 \
    --guidance 0.0
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--json <file>` | ComfyUI JSON file (both Workflow and API format) | - |
| `--yaml <file>` | YAML pipeline config file | - |
| `--model-path <path>` | Model path base directory | `./models/` |
| `--device <device>` | Device to run on (CPU, GPU, GPU.0, etc.) | `CPU` |
| `--prompt <text>` | Text prompt for generation | - |
| `--output <file>` | Output image filename | Auto-generated |
| `--width <int>` | Image width | From config |
| `--height <int>` | Image height | From config |
| `--steps <int>` | Number of inference steps | From config |
| `--guidance <float>` | Guidance scale | From config |
| `--max-seq-len <int>` | Max sequence length | From config |
| `--help` | Show help message | - |

---

## License

Copyright (C) 2018-2026 Intel Corporation. SPDX-License-Identifier: Apache-2.0