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

This sample runs an Image Generation pipeline (e.g. Z-Image) using a ModulePipeline config YAML.

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

``Z-Image with tiling``: sample_size(tiling_size) 256
```
cd openvino.genai
./build/samples/cpp/module_genai/md_image_generation    \
    -cfg ./samples/cpp/module_genai/config_yaml/Z-Image-Turbo-fp16-ov/config_tiling.yaml   \
    -prompt "A beautiful landscape painting by Claude Monet"    \
    --height 512    \
    --width 512     \
    --num_inference_steps 9 \
    --guidance_scale 2.5    \
    --max_sequence_length 512
```
</details>


## Video generation

This sample runs a Video Generation pipeline (e.g. Wan2.1-T2V-1.3B) using a ModulePipeline config YAML.

<details>
	<summary>Command</summary>

```
cd openvino.genai
export cfg=./samples/cpp/module_genai/config_yaml/Wan2.1-T2V-1.3B-Diffusers/config.yaml
# export  cfg=./samples/cpp/module_genai/config_yaml/Wan2.1-T2V-1.3B-Diffusers/config_split_transformer.yaml
export prompt="A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
export neg_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

./build/samples/cpp/module_genai/md_video_generation \
    -cfg "$cfg" \
    -prompt "$prompt" \
    --negative_prompt $neg_prompt \
    --num_frames 16 --height 128 --width 128 --num_inference_steps 9
```
`Note:` Update **model_path** inside the config YAML if you keep models in a different location.

</details>