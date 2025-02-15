# Text to Image Python Generation Pipeline

Examples in this folder showcase inference of text to image models like Stable Diffusion 1.5, 2.1, LCM. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `openvino_genai.Text2ImagePipeline` and uses a text prompt as input source.

There are several sample files:
 - [`text2image.py`](./text2image.py) demonstrates basic usage of the text to image pipeline
 - [`lora_text2image.py`](./lora.py) shows how to apply LoRA adapters to the pipeline
 - [`heterogeneous_stable_diffusion.py`](./heterogeneous_stable_diffusion.py) shows how to assemble a heterogeneous text2image pipeline from individual subcomponents (scheduler, text encoder, unet, vae decoder)
 - [`image2image.py`](./image2image.py) demonstrates basic usage of the image to image pipeline
 - [`inpainting.py`](./inpainting.py) demonstrates basic usage of the inpainting pipeline
 - [`benchmark_image.py`](./benchmark_image.py) demonstrates how to benchmark the text to image / image to image / inpainting pipeline

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
> OpenVINO GenAI is written in C++ and uses `CppStdGenerator` random generator in Image Generation pipelines, while Diffusers library uses `torch.Generator` underhood.
> To have the same results with HuggingFace, pass manually created `torch.Generator(device='cpu').manual_seed(seed)` to Diffusers generation pipelines and `openvino_genai.TorchGenerator(seed)` to OpenVINO GenAI pipelines as value for `generator` kwarg.

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

   ![](./../../cpp/image_generation/512x512.bmp)

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
![](./../../cpp/image_generation/lora.bmp) | ![](./../../cpp/image_generation/baseline.bmp)

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

The resuling image is:

   ![](./../../cpp/image_generation/inpainting.bmp)

Note, that LoRA, heterogeneous execution and other features of `Text2ImagePipeline` are applicable for `InpaintingPipeline`.

## benchmarking sample for text to image / image to image / inpainting pipeline

This `benchmark_image.py` sample script demonstrates how to benchmark text to image / image to image / inpainting pipeline. The script includes functionality for warm-up iterations, generating image, and calculating various performance metrics.

The usage of this sample is:
```bash
python benchmark_image.py [OPTIONS]
```
Options:
- `-t, --pipeline_type`: Pipeline type: text2image/image2image/inpainting.
- `-m, --model`: Path to the model and tokenizers base directory.
- `-p, --prompt` (default: `"The Sky is blue because"`): The prompt to generate text.
- `-nw, --num_warmup` (default: `1`): Number of warmup iterations.
- `-n, --num_iter` (default: `3`): Number of iterations.
- `-d, --device` (default: `"CPU"`): Device to run the model on.
- `-wh, --width` (default: `512`): The width of the output image.
- `-ht, --height` (default: `512`): The height of the output image.
- `-is, --num_inference_steps` (default: `20`): The number of inference steps.
- `-ni, --num_images_per_prompt` (default: `1`): The number of images to generate per generate() call.
- `-o, --output_dir` (default: `""`): Path to save output image.
- `-i, --image`: Path to input image.
- `-mi, --mask_image`: Path to the mask image.
- `-s, --strength`: Indicates extent to transform the reference `image`. Must be between 0 and 1.

For example:

`python benchmark_image.py -t text2image -m dreamlike_anime_1_0_ov/FP16 -n 10 -d CPU`

Performance output:

```
[warmup-0] generate time: 66005.00 ms, total infer time: 65993.44 ms
[warmup-0] encoder infer time: 264.00 ms
[warmup-0] unet iteration num: 20, first iteration time: 3471.94 ms, other iteration avg time: 3154.51 ms
[warmup-0] unet inference num: 20, first inference time: 3471.44 ms, other inference avg time: 3154.00 ms
[warmup-0] vae decoder infer time: 2332.00 ms

[iter-0] generate time: 65725.00 ms, total infer time: 65712.61 ms
[iter-0] encoder infer time: 153.00 ms
[iter-0] unet iteration num: 20, first iteration time: 3167.23 ms, other iteration avg time: 3165.03 ms
[iter-0] unet inference num: 20, first inference time: 3166.56 ms, other inference avg time: 3164.53 ms
[iter-0] vae decoder infer time: 2267.00 ms

[iter-1] generate time: 65715.00 ms, total infer time: 65704.02 ms
[iter-1] encoder infer time: 146.00 ms
[iter-1] unet iteration num: 20, first iteration time: 3185.16 ms, other iteration avg time: 3162.04 ms
[iter-1] unet inference num: 20, first inference time: 3184.67 ms, other inference avg time: 3161.54 ms
[iter-1] vae decoder infer time: 2304.00 ms

Test finish, load time: 3604.00 ms
Warmup number: 1, first generate warmup time: 66005.00 ms, infer warmup time: 65993.44 ms
Generate iteration number: 2, for one iteration, generate avg time: 65720.00 ms, infer avg time: 65708.31 ms, total encoder infer avg time: 149.50 ms, decoder infer avg time: 2285.50 ms
```
