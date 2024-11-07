# Text to Image Python Generation Pipeline

Examples in this folder showcase inference of text to image models like Stable Diffusion 1.5, 2.1, LCM. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `openvino_genai.Text2ImagePipeline` and uses a text prompt as input source.

There are two sample files:
 - [`main.py`](./main.py) demonstrates basic usage of the text to image pipeline
 - [`lora.py`](./lora.py) shows how to apply LoRA adapters to the pipeline
 - [`lora_fuse.py`](./lora_fuse.py) shows how to maximize performance of LoRA adapters by fusing them into base model weights

Users can change the sample code and play with the following generation parameters:

- Change width or height of generated image
- Generate multiple images per prompt
- Adjust a number of inference steps
- Play with [guidance scale](https://huggingface.co/spaces/stabilityai/stable-diffusion/discussions/9) (read [more details](https://arxiv.org/abs/2207.12598))
- (SD 1.x, 2.x only) Add negative prompt when guidance scale > 1
- Apply multiple different LoRA adapters and mix them with different blending coefficients

## Download and convert the models and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
optimum-cli export openvino --model dreamlike-art/dreamlike-anime-1.0 --task stable-diffusion --weight-format fp16 dreamlike_anime_1_0_ov/FP16
```

## Run

Install [deployment-requirements.txt](../../deployment-requirements.txt) via `pip install -r ../../deployment-requirements.txt` and then, run a sample:

`python main.py ./dreamlike_anime_1_0_ov/FP16 "cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting"`

### Examples

Prompt: `cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting`

   ![](./image.bmp)


## Run with optional LoRA adapters

LoRA adapters can be connected to the pipeline and modify generated images to have certain style, details or quality. Adapters are supported in Safetensors format and can be downloaded from public sources like [Civitai](https://civitai.com) or [HuggingFace](https://huggingface.co/models) or trained by the user. Adapters compatible with a base model should be used only. A weighted blend of multiple adapters can be applied by specifying multiple adapter files with corresponding alpha parameters in command line. Check `lora.cpp` source code to learn how to enable adapters and specify them in each `generate` call.

Here is an example how to run the sample with a single adapter. First download adapter file from https://civitai.com/models/67927/soulcard page manually and save it as `soulcard.safetensors`. Or download it from command line:

`wget -O soulcard.safetensors https://civitai.com/api/download/models/72591`

Then run `lora.py`:

`python lora.py ./dreamlike_anime_1_0_ov/FP16 "curly-haired unicorn in the forest, anime, line" soulcard.safetensors 0.7`

The sample generates two images with and without adapters applied using the same prompt:
   - `lora.bmp` with adapters applied
   - `baseline.bmp` without adapters applied

Check the difference:

With adapter | Without adapter
:---:|:---:
![](./lora.bmp) | ![](./baseline.bmp)


# Fuse LoRA adapters into model weights

To maximize inference performance using a LoRA adapter, refer to `lora_fuse.py`, which demonstrates fusing the adapter into the model weights. This approach achieves the same performance as the base model without a LoRA adapter but removes the flexibility to switch adapters between generate calls. This mode is ideal when performing multiple generations with the same LoRA adapters and blending alpha parameters, and when model recompilation on adapter changes is feasible. The example outputs the resulting image as `lora.bmp`.