
# Module GenAI Python samples

This folder contains Python samples for **Module GenAI** based pipelines.

Build **Module GenAI** [Refer](../../../README_Module_GenAI.md)

## Setup

From the repo root:

```bash
cd openvino.genai
# Set OpenVINO runtime environment (repo helper script)
source [PATH]/openvino/setupvars.sh

export GENAI_ROOT_DIR=[PATH]/openvino.genai/install/python/
export OV_PYTHON_DIR=${INTEL_OPENVINO_DIR}/python

export PYTHONPATH=${OV_PYTHON_DIR}:${GENAI_ROOT_DIR}:$PYTHONPATH
export LD_LIBRARY_PATH=${GENAI_ROOT_DIR}/../runtime/lib/intel64/:$LD_LIBRARY_PATH
export OV_TOKENIZER_PREBUILD_EXTENSION_PATH=${GENAI_ROOT_DIR}/../runtime/lib/intel64/libopenvino_tokenizers.so
```

## Visual language chat

This sample runs a VLM pipeline (e.g. Qwen2.5-VL-3B-Instruct) using a ModulePipeline config YAML.

<details>
	<summary>Command</summary>

```bash
python3 ./samples/python/module_genai/md_visual_language_chat.py	\
	--cfg ./samples/cpp/module_genai/config_yaml/Qwen2.5-VL-3B-Instruct/config.yaml \
	--prompt "Please describe the image" 	\
	--img ./tests/module_genai/cpp/test_data/cat_120_100.png
```
Notes:

- Update **model_path** inside the config YAML if you keep models in a different location.
- `image_path` / `video_path` are optional; the sample maps inputs based on `ParameterModule.outputs`.

</details>


## Image generation

Use the Z-Image pipeline Python sample.

<details>
	<summary>Command</summary>

```bash
prompt="Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
python3 ./samples/python/module_genai/md_image_generation.py \
	--model_path ./samples/cpp/module_genai/ut_pipelines/Z-Image-Turbo-fp16-ov \
	--prompt "${prompt}" \
	--height 1040 \
	--width 1040
```

</details>

## More samples

- `md_video_generation.py` (video generation)
- `md_cowork_with_torch.py` (mix ModulePipeline with Torch steps)

