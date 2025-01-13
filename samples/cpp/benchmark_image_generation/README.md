# LLMs benchmarking sample

This sample script demonstrates inference of text to image models in OpenVINO GenAI. The script includes functionality for warm-up iterations, generating image, and calculating various performance metrics.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

It's not required to install [../../export-requirements.txt](../../export requirements.txt) for deployment if the model has already been exported.

```sh
pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --model dreamlike-art/dreamlike-anime-1.0 --task stable-diffusion --weight-format fp16 dreamlike_anime_1_0_ov/FP16
```

## Usage

Follow [Get Started with Samples](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/get-started-demos.html) to run the sample.

```sh
benchmark_text2image [OPTIONS]
```

### Options

- `-m, --model`: Path to the model and tokenizers base directory.
- `-p, --prompt` (default: `"The Sky is blue because"`): The prompt to generate text.
- `-nw, --num_warmup` (default: `1`): Number of warmup iterations.
- `-mt, --max_new_tokens` (default: `20`): Number of warmup iterations.
- `-n, --num_iter` (default: `3`): Number of iterations.
- `-d, --device` (default: `"CPU"`): Device to run the model on.
- `-wh, --width` (default: `512`): The width of the output image.
- `-ht, --height` (default: `512`): The height of the output image.
- `-is, --num_inference_steps` (default: `20`): The number of inference steps.
- `-ni, --num_images_per_prompt` (default: `1`): The number of images to generate per generate() call.
- `-o, --output_dir` (default: `""`): Path to save output image.

### Output:

```
benchmark_text2image -m dreamlike_anime_1_0_ov/FP16 -n 10
```

```
Load time: 1433.69 ms
One generate avg time: 1430.77 ms
Total inference for one generate avg time:: 1429 ms
```

For more information how performance metrics are calculated please follow [performance-metrics tutorial](../../../src/README.md#performance-metrics).
