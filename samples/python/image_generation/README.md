# Text to Image Python Generation Pipeline

Examples in this folder showcase inference of text to image models like Stable Diffusion 1.5, 2.1, LCM. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `openvino_genai.Text2ImagePipeline` and uses a text prompt as input source.

There are several sample files:
 - [`text2image.py`](./text2image.py) demonstrates basic usage of the text to image pipeline
 - [`lora_text2image.py`](./lora.py) shows how to apply LoRA adapters to the pipeline
 - [`heterogeneous_stable_diffusion.py`](./heterogeneous_stable_diffusion.py) shows how to assemble a heterogeneous text2image pipeline from individual subcomponents (scheduler, text encoder, unet, vae decoder)
 - [`image2image.py`](./image2image.py) demonstrates basic usage of the image to image pipeline
 - [`inpainting.py`](./inpainting.py) demonstrates basic usage of the inpainting pipeline

Users can change the sample code and play with the following generation parameters:

- Change width or height of generated image
- Generate multiple images per prompt
- Adjust a number of inference steps
- Play with [guidance scale](https://huggingface.co/spaces/stabilityai/stable-diffusion/discussions/9) (read [more details](https://arxiv.org/abs/2207.12598))
- (SD 1.x, 2.x; SD3, SDXL) Add negative prompt when guidance scale > 1
- (SDXL, SD3, FLUX) Specify other positive prompts like `prompt_2`
- Apply multiple different LoRA adapters and mix them with different blending coefficients
- (Image to image and inpainting) Play with `strength` parameter to control how initial image is noised and reduce number of inference steps

> [!NOTE]  
> All samples use `openvino_genai.TorchGenerator` as an argument to `generate` call to align random numbers generation with Diffusers library where `torch.Generator` is used underhood.
> To have the same results in HuggingFace, pass manually created `torch.Generator(device='cpu').manual_seed(42)` to Diffusers generation pipelines to fix `seed` parameter.

## Download and convert the models and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
optimum-cli export openvino --model dreamlike-art/dreamlike-anime-1.0 --task stable-diffusion --weight-format fp16 dreamlike_anime_1_0_ov/FP16
```

## Run text to image

Install [deployment-requirements.txt](../../deployment-requirements.txt) via `pip install -r ../../deployment-requirements.txt` and then, run a sample:

`python text2image.py ./dreamlike_anime_1_0_ov/FP16 "cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting"`

### Examples

Prompt: `cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting`

   ![](./text2image.bmp)

### Run with callback

You can also add a callback to the `text2image.py` file to interrupt the image generation process earlier if you are satisfied with the intermediate result of the image generation or to add logs.

Please find the template of the callback usage below.

```python
pipe = openvino_genai.Text2ImagePipeline(model_dir, device)

def callback(step, num_steps, latent):
   print(f"Image generation step: {step} / {num_steps}")
   image_tensor = pipe.decode(latent) # get intermediate image tensor
   if your_condition: # return True if you want to interrupt image generation
      return True
   return False

image = pipe.generate(
   ...
   callback = callback
)
```

## Run with optional LoRA adapters

LoRA adapters can be connected to the pipeline and modify generated images to have certain style, details or quality. Adapters are supported in Safetensors format and can be downloaded from public sources like [Civitai](https://civitai.com) or [HuggingFace](https://huggingface.co/models) or trained by the user. Adapters compatible with a base model should be used only. A weighted blend of multiple adapters can be applied by specifying multiple adapter files with corresponding alpha parameters in command line. Check `lora_text2image.py` source code to learn how to enable adapters and specify them in each `generate` call.

Here is an example how to run the sample with a single adapter. First download adapter file from https://civitai.com/models/67927/soulcard page manually and save it as `soulcard.safetensors`. Or download it from command line:

`wget -O soulcard.safetensors https://civitai.com/api/download/models/72591`

Then run `lora_text2image.py`:

`python lora_text2image.py ./dreamlike_anime_1_0_ov/FP16 "curly-haired unicorn in the forest, anime, line" soulcard.safetensors 0.7`

The sample generates two images with and without adapters applied using the same prompt:
   - `lora.bmp` with adapters applied
   - `baseline.bmp` without adapters applied

Check the difference:

With adapter | Without adapter
:---:|:---:
![](./lora.bmp) | ![](./baseline.bmp)

## Run text to image with multiple devices

The `heterogeneous_stable_diffusion.py` sample demonstrates how a Text2ImagePipeline object can be created from individual subcomponents - scheduler, text encoder, unet, & vae decoder. This approach gives fine-grained control over the devices used to execute each stage of the stable diffusion pipeline.

The usage of this sample is:

`heterogeneous_stable_diffusion.py [-h] model_dir prompt [text_encoder_device] [unet_device] [vae_decoder_device]`

For example:

`python heterogeneous_stable_diffusion.py ./dreamlike_anime_1_0_ov/FP16 'cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting' CPU NPU GPU`

The sample will create a stable diffusion pipeline such that the text encoder is executed on the CPU, UNet on the NPU, and VAE decoder on the GPU.

## Run image to image pipeline

The `image2mage.py` sample demonstrates basic image to image generation pipeline. The difference with text to image pipeline is that final image is denoised from initial image converted to latent space and noised with image noise according to `strength` parameter. `strength` should be in range of `[0., 1.]` where `1.` means initial image is fully noised and it is an equivalent to text to image generation.
Also, `strength` parameter linearly affects a number of inferenece steps, because lower `strength` values means initial latent already has some structure and it requires less steps to denoise it. 

To run the sample, download initial image first:

`wget -O cat.png https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png`

And then run the sample:

`python image2image.py ./dreamlike_anime_1_0_ov/FP16 'cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k' cat.png`

The resuling image is:

   ![](./../../cpp/image_generation/imageimage.bmp)

Note, that LoRA, heterogeneous execution and other features of `Text2ImagePipeline` are applicable for `Image2ImagePipeline`.

## Run inpainting pipeline

The `inpainting.py` sample demonstrates usage of inpainting pipeline, which can inpaint initial image by a given mask. Inpainting pipeline can work on typical text to image models as well as on specialized models which are oftenly named `space/model-inpainting`, e.g. `stabilityai/stable-diffusion-2-inpainting`. 

Such models can be converted in the same way as regular ones via `optimum-cli`:

`optimum-cli export openvino --model stabilityai/stable-diffusion-2-inpainting --weight-format fp16 stable-diffusion-2-inpainting`

Let's also download input data:

`wget -O image.png https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png`

`wget -O mask_image.png https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png`

And run the sample:

`python inpainting.py ./stable-diffusion-2-inpainting 'Face of a yellow cat, high resolution, sitting on a park bench' image.png mask_image.png`

Note, that LoRA, heterogeneous execution and other features of `Text2ImagePipeline` are applicable for `InpaintingPipeline`.
