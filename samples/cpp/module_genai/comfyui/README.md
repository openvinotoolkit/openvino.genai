# ComfyUI Pipeline Sample

This sample runs OpenVINO GenAI ModulePipeline from ComfyUI JSON or YAML configuration files.

Build **Module GenAI** [Refer](../../../../README_Module_GenAI.md)

## Features

- Convert ComfyUI API JSON workflow to GenAI YAML pipeline config
- Load and run pipelines from YAML config files
- Command-line parameter overrides for prompt, image size, steps, etc.
- Automatic image/video saving with timestamp

## Run from ComfyUI JSON

<details>
	<summary>Z-Image (Text-to-Image)</summary>

```bash
cd openvino.genai
./build/samples/cpp/module_genai/comfyui/md_comfyui \
    --json samples/cpp/module_genai/comfyui/z_image_turbo_2k_tiled_api.json \
    --model_path samples/cpp/module_genai/ut_pipelines/Z-Image-Turbo-fp16-ov \
    --device GPU \
    --prompt "a cat sitting on a sofa" \
    --width 1024 \
    --height 1024 \
    --steps 4
```

</details>

<details>
	<summary>Wan 2.1 (Text-to-Video)</summary>

```bash
cd openvino.genai
./build/samples/cpp/module_genai/comfyui/md_comfyui \
    --json samples/cpp/module_genai/comfyui/wan2.1_t2v_api.json \
    --model_path samples/cpp/module_genai/ut_pipelines/Wan2.1-T2V-1.3B-Diffusers \
    --device GPU \
    --prompt "a cat walking in the garden" \
    --width 480 \
    --height 272 \
    --num_frames 17 \
    --steps 20
```

</details>

## Options

Run with `--help` to see all available options:

```bash
./build/samples/cpp/module_genai/comfyui/md_comfyui --help
```

---

## License

Copyright (C) 2018-2026 Intel Corporation. SPDX-License-Identifier: Apache-2.0