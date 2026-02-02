# Module GenAI CPP Samples

This folder contains CPP samples for **Module GenAI** based pipelines.

Build **Module GenAI** [Refer](../../../README_Module_GenAI.md)

## Visual language chat

This sample runs a VLM pipeline (e.g. Qwen2.5-VL-3B-Instruct) using a ModulePipeline config YAML.

<details>
	<summary>Command</summary>

```
cd openvino.genai
./build/samples/cpp/module_genai/md_visual_language_chat \
    -cfg ./samples/cpp/module_genai/config_yaml/Qwen2.5-VL-3B-Instruct/config.yaml \
    -prompt "Please describe the image" \
    -img ./tests/module_genai/cpp/test_data/cat_120_100.png
```
Notes:

- Update **model_path** inside the config YAML if you keep models in a different location.
- `image_path` / `video_path` are optional; the sample maps inputs based on `ParameterModule.outputs`.

</details>


## Image generation

Use the Z-Image pipeline Python sample.

<details>
	<summary>Command</summary>

```
cd openvino.genai
./build/samples/cpp/module_genai/md_image_generation    \
    -cfg ./samples/cpp/module_genai/config_yaml/Z-Image-Turbo-fp16-ov/config.yaml   \
    -prompt "A beautiful landscape painting by Claude Monet"    \ 
    --height 512    \
    --width 512     \
    --num_inference_steps 9 \
    --guidance_scale 2.5    \
    --max_sequence_length 512
```
`Note:` Update **model_path** inside the config YAML if you keep models in a different location.

</details>