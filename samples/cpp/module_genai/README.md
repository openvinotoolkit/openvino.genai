# Module GenAI Samples

Build samples, refer: openvino.genai/README_Module_GenAI.md

<details>
<summary>visual_language_chat</summary>


#### Test model: Qwen2.5-VL-3B-Instruct
```
cd openvino.genai
app=./build/samples/cpp/module_genai/md_visual_language_chat
cfg=./samples/cpp/module_genai/config_yaml/Qwen2.5-VL-3B-Instruct/config.yaml
prompt="Please describe the image"
img=./tests/module_genai/cpp/test_data/cat_120_100.png

"$app" -cfg "$cfg" -prompt "$prompt" -img "$img"
```
`Note:` update model_path based on your local path.

</details>

<details>
<summary>image_generation</summary>

#### Test model: Z-Image-Turbo
```
cd openvino.genai
app=./build/samples/cpp/module_genai/md_image_generation
cfg=./samples/cpp/module_genai/config_yaml/Z-Image-Turbo-fp16-ov/config.yaml
prompt="A beautiful landscape painting by Claude Monet"

"$app" -cfg "$cfg" -prompt "$prompt" --height 512 --width 512 --num_inference_steps 9 --guidance_scale 2.5 --max_sequence_length 512
```
`Note:` update model_path based on your local path.

</details>