# Text to Image Python Generation Pipeline

Examples in this folder showcase inference of text to image models like Stable Diffusion 1.5, 2.1, LCM. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `openvino_genai.Text2ImagePipeline` and uses a text prompt as input source.

There are several sample files:
 - [`text2image.py`](./text2image.py) demonstrates basic usage of the text to image pipeline
 - [`lora_text2image.py`](./lora_text2image.py) shows how to apply LoRA adapters to the pipeline
 - [`heterogeneous_stable_diffusion.py`](./heterogeneous_stable_diffusion.py) shows how to assemble a heterogeneous text2image pipeline from individual subcomponents (scheduler, text encoder, unet, vae decoder)
 - [`image2image.py`](./image2image.py) demonstrates basic usage of the image to image pipeline
 - [`inpainting.py`](./inpainting.py) demonstrates basic usage of the inpainting pipeline
 - [`benchmark_image_gen.py`](./benchmark_image_gen.py) demonstrates how to benchmark the text to image / image to image / inpainting pipeline
 - [`stable_diffusion_export_import.py`](./stable_diffusion_export_import.py) demonstrates how to export and import compiled models in the text to image pipeline. Only the Stable Diffusion XL model is supported.

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
```

Then, run the export with Optimum CLI:

```sh
optimum-cli export openvino --model dreamlike-art/dreamlike-anime-1.0 --task stable-diffusion --weight-format fp16 dreamlike_anime_1_0_ov/FP16
```

Alternatively, do it in Python code (FP16 is used by default). If NNCF is installed, the model will be compressed to INT8 automatically.

```python
from optimum.exporters.openvino.convert import export_tokenizer
from optimum.intel import OVPipelineForText2Image

output_dir = "dreamlike_anime_1_0_ov/FP16"

pipeline = OVPipelineForText2Image.from_pretrained("dreamlike-art/dreamlike-anime-1.0", export=True)
pipeline.save_pretrained(output_dir)
export_tokenizer(pipeline.tokenizer, output_dir + "/tokenizer")
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
   print(f"Image generation step: {step + 1} / {num_steps}")
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

> [!NOTE]
> ### LoRA `alpha` interpretation in OpenVINO GenAI
> The OpenVINO GenAI implementation merges the traditional LoRA parameters into a **single effective scaling factor** used during inference.
>
> In this context, the `alpha` value already includes:
> - normalization by LoRA rank (`alpha / rank`)
> - any user-defined scaling factor (`weight`)
>
> This means `alpha` in GenAI should be treated as the **final scaling weight** applied to the LoRA update — not the raw `alpha` parameter from training.

### Example: Running with a LoRA Adapter

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

## benchmarking sample for image generation pipelines

This `benchmark_image_gen.py` sample script demonstrates how to benchmark text to image / image to image / inpainting pipeline. The script includes functionality for warm-up iterations, generating image, and calculating various performance metrics.

The usage of this sample is:
```bash
python benchmark_image_gen.py [OPTIONS]
```
Options:
- `-t, --pipeline_type`: Pipeline type: text2image/image2image/inpainting.
- `-m, --model`: Path to the model and tokenizers base directory.
- `-p, --prompt` (default: `"The Sky is blue because"`): The prompt to generate text.
- `-nw, --num_warmup` (default: `1`): Number of warmup iterations.
- `-n, --num_iter` (default: `3`): Number of iterations.
- `-d, --device` (default: `"CPU"`): Device(s) to run the pipeline with.
- `-w, --width` (default: `512`): The width of the output image.
- `-ht, --height` (default: `512`): The height of the output image.
- `-is, --num_inference_steps` (default: `20`): The number of inference steps.
- `-ni, --num_images_per_prompt` (default: `1`): The number of images to generate per generate() call.
- `-o, --output_dir` (default: `""`): Path to save output image.
- `-i, --image`: Path to input image.
- `-mi, --mask_image`: Path to the mask image.
- `-s, --strength`: Indicates extent to transform the reference `image`. Must be between 0 and 1.
- `-r, --reshape': Reshape pipeline before compilation. This can improve image generation performance.

For example:

`python benchmark_image_gen.py -t text2image -m dreamlike_anime_1_0_ov/FP16 -n 10 -d CPU`

Performance output:

```
[warmup-0] generate time: 85008.00 ms, total infer time:84999.88 ms
[warmup-0] text encoder infer time: 98.00 ms
[warmup-0] unet iteration num:21, first iteration time:4317.94 ms, other iteration avg time:3800.91 ms
[warmup-0] unet inference num:21, first inference time:4317.71 ms, other inference avg time:3800.61 ms
[warmup-0] vae encoder infer time:0.00 ms, vae decoder infer time:4572.00 ms

[iter-0] generate time: 84349.00 ms, total infer time:84340.97 ms
[iter-0] text encoder infer time: 76.00 ms
[iter-0] unet iteration num:21, first iteration time:3805.63 ms, other iteration avg time:3799.68 ms
[iter-0] unet inference num:21, first inference time:3805.42 ms, other inference avg time:3799.38 ms
[iter-0] vae encoder infer time:0.00 ms, vae decoder infer time:4472.00 ms

[iter-1] generate time: 84391.00 ms, total infer time:84384.36 ms
[iter-1] text encoder infer time: 78.00 ms
[iter-1] unet iteration num:21, first iteration time:3801.15 ms, other iteration avg time:3802.17 ms
[iter-1] unet inference num:21, first inference time:3800.93 ms, other inference avg time:3801.87 ms
[iter-1] vae encoder infer time:0.00 ms, vae decoder infer time:4468.00 ms

[iter-2] generate time: 84377.00 ms, total infer time:84366.51 ms
[iter-2] text encoder infer time: 76.00 ms
[iter-2] unet iteration num:21, first iteration time:3783.31 ms, other iteration avg time:3802.25 ms
[iter-2] unet inference num:21, first inference time:3783.09 ms, other inference avg time:3801.82 ms
[iter-2] vae encoder infer time:0.00 ms, vae decoder infer time:4471.00 ms

Test finish, load time: 9356.00 ms
Warmup number:1, first generate warmup time:85008.00 ms, infer warmup time:84999.88 ms
Generate iteration number:3, for one iteration, generate avg time: 84372.34 ms, infer avg time:84363.95 ms, all text encoders infer avg time:76.67 ms, vae encoder infer avg time:0.00 ms, vae decoder infer avg time:4470.33 ms
```

### Image Generation Pipeline reuse

To extend the pipeline's capabilities, we provide an interface that allows a specific image generation pipeline to reuse models from another pipeline that has already loaded them. The table below shows the support scope.

| Image Generation pipeline | Model can be reused from |
|:---|:---|
| `Text2ImagePipeline` | `Image2ImagePipeline` or `InpaintingPipeline` |
| `Image2ImagePipeline` | `InpaintingPipeline` |
| `InpaintingPipeline` | `Image2ImagePipeline` |

This example shows how `Text2ImagePipeline` reuses models from `Image2ImagePipeline` and executes a different pipeline depending on whether an initial image is provided.

```py
img2img_pipe = openvino_genai.Image2ImagePipeline(models_path, device)
text2img_pipe = openvino_genai.Text2ImagePipeline(img2img_pipe)

if image_path:
   image = read_image(image_path)
   image_tensor = img2img_pipe.generate(prompt, image, strength=0.8)
else:
   image_tensor = text2img_pipe.generate(prompt, strength=1.0)
```

## Export and import compiled models

`openvino_genai.Image2ImagePipeline` supports exporting and importing compiled models to and from a specified directory. This API can significantly reduce model load time, especially for large models like UNet. Only the Stable Diffusion XL model is supported.

```python
# export models
pipeline = openvino_genai.Text2ImagePipeline(models_path, device)
pipeline.export_model(models_path / "blobs")

# import models
imported_pipeline = openvino_genai.Text2ImagePipeline(models_path, device, blob_path=models_path / "blobs")
```
