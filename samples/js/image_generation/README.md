# Text to Image JavaScript Generation Pipeline

This example showcases inference of text-to-image diffusion models like Stable Diffusion 1.5, 2.1, FLUX, and LCM. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `Text2ImagePipeline` from `openvino-genai-node` and uses a text prompt as input source.

Sample files:
 - [`text2image.js`](./text2image.js) demonstrates basic usage of the text-to-image pipeline with a step callback and saves the result as a BMP file using `bmp-js`

Users can change the sample code and play with the following generation parameters:

- Change width or height of generated image
- Generate multiple images per prompt (`num_images_per_prompt`)
- Adjust a number of inference steps (`num_inference_steps`)
- Play with [guidance scale](https://huggingface.co/spaces/stabilityai/stable-diffusion/discussions/9) (read [more details](https://arxiv.org/abs/2207.12598))
- (SD 1.x, 2.x; SD3, SDXL) Add negative prompt when guidance scale > 1
- (SDXL, SD3, FLUX) Specify other positive prompts like `prompt_2`
- Add a per-step callback to monitor progress or stop generation early

## Download and convert the model

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

It's not required to install [../../export-requirements.txt](../../export-requirements.txt) for deployment if the model has already been exported.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
```

Then, run the export with Optimum CLI:

```sh
optimum-cli export openvino --model dreamlike-art/dreamlike-anime-1.0 --task stable-diffusion --weight-format fp16 dreamlike_anime_1_0_ov/FP16
```

## Run

From the `samples/js` directory, install dependencies (if not already done):

```bash
npm install
```

If you use the master branch, you may need to [build openvino-genai-node from source](../../../src/js/README.md#build-bindings) first.

Run the sample:

```bash
node image_generation/text2image.js dreamlike_anime_1_0_ov/FP16 "cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting"
```

The result is saved as `image.bmp` in the current directory.

### Optional: change device

The device is hardcoded to `CPU` in the sample. To use `GPU`, edit `text2image.js` and change:

```js
const device = "CPU"; // GPU can be used as well
```

### Examples

Prompt: `cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting`

   ![](./../../cpp/image_generation/512x512.bmp)

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#image-generation-models) for the list of supported models.

## Run with a step callback

The sample registers a callback that prints progress to stdout on each denoising step. The callback can also stop generation early by returning `true`:

```js
function callback(step, numSteps) {
  process.stdout.write(`Step ${step + 1}/${numSteps}\r`);
  return false; // return true to stop early
}

const imageTensor = await pipeline.generate(prompt, {
  width: 512,
  height: 512,
  num_inference_steps: 20,
  callback,
});
```
