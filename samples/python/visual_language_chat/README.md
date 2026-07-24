# Python vlm_chat_sample that supports VLM models

This example showcases inference of text-generation Vision Language Models (VLMs): `Qwen3-VL-2B-Instruct` and other models with the same signature. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `openvino_genai.VLMPipeline` and configures it for the chat scenario. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/qwen3-vl/qwen3-vl.ipynb) which provides an example of Visual-language assistant.

The following are sample files:
 - [`visual_language_chat.py`](./visual_language_chat.py) demonstrates basic usage of the VLM pipeline which supports accelerated inference using prompt lookup decoding.
 - [`video_to_text_chat.py`](./video_to_text_chat.py) demonstrates video to text usage of the VLM pipeline.
 - [`benchmark_vlm.py`](./benchmark_vlm.py) shows how to benchmark a VLM in OpenVINO GenAI. The script includes functionality for warm-up iterations, generating text and calculating various performance metrics.
 - [`visual_language_lora.py`](./visual_language_lora.py) demonstrates how to apply one or more LoRA adapters to a VLM at runtime.
 - [`milebench_eval_vlm.py`](./milebench_eval_vlm.py) provides MileBench validation for VLMs, enabling evaluation of image–text reasoning and visual QA tasks across multiple subsets designed to assess the MultImodal Long-contExt capabilities of MLLMs.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
```

Then, run the export with Optimum CLI:

```sh
optimum-cli export openvino --model Qwen/Qwen3-VL-2B-Instruct --trust-remote-code Qwen3-VL-2B-Instruct
```

Alternatively, you can do it in Python code:

```python
from optimum.exporters.openvino.convert import export_tokenizer
from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoTokenizer

output_dir = "Qwen3-VL-2B-Instruct"

model = OVModelForVisualCausalLM.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", export=True, trust_remote_code=True)
model.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
export_tokenizer(tokenizer, output_dir)
```

Install [deployment-requirements.txt](../../deployment-requirements.txt) via `pip install -r ../../deployment-requirements.txt` to run VLM samples.

## Run image-to-text chat sample:

[This image](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11) can be used as a sample image.

`python visual_language_chat.py ./Qwen3-VL-2B-Instruct/ 319483352-d5fbbd1a-d484-415c-88cb-9986625b7b11.jpg`

See https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md#supported-models for the list of supported models.

## Run image-to-text sample with LoRA adapters:

This sample runs generation twice for the same prompt and image: first with LoRA adapter(s) applied, then without any adapters (base model).

Export `Qwen/Qwen2.5-VL-7B-Instruct` to OpenVINO as [described above](#download-and-convert-the-model-and-tokenizers), then download LoRA `Mouad2004/qwen2.5-vl-lora-diagrams`:

```sh
wget -O adapter_model.safetensors \
	https://huggingface.co/Mouad2004/qwen2.5-vl-lora-diagrams/resolve/main/adapter_model.safetensors
```

This OpenVINO overview diagram can be used as a convenient image input:

```sh
wget -O openvino-overview-diagram.jpg \
	https://docs.openvino.ai/2026/_images/openvino-overview-diagram.jpg
```

`python visual_language_lora.py ./Qwen2.5-VL-7B-Instruct ./openvino-overview-diagram.jpg "What is shown in this diagram?" ./adapter_model.safetensors 4.0`

> You can run with multiple LoRA adapters by providing multiple `<LORA_SAFETENSORS> <ALPHA>` pairs.

> [!NOTE]
> ### LoRA `alpha` interpretation in OpenVINO GenAI
> The OpenVINO GenAI implementation merges the traditional LoRA parameters into a **single effective scaling factor** used during inference.
>
> In this context, the `alpha` value already includes:
> - normalization by LoRA rank (`alpha / rank`)
> - any user-defined scaling factor (`weight`)
>
> This means `alpha` in GenAI should be treated as the **final scaling weight** applied to the LoRA update — not the raw `alpha` parameter from training.

## Run video-to-text chat sample:

A model that supports video input is required to run this sample, for example `llava-hf/LLaVA-NeXT-Video-7B-hf`.

[This video](https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4) can be used as a sample video.

`python video_to_text_chat.py ./LLaVA-NeXT-Video-7B-hf/ sample_demo_1.mp4`

Supported models with video input are listed in [this section](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/visual-processing/#use-image-or-video-tags-in-prompt).

Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM.
Modify the source code to change the device for inference to the GPU.

## Run benchmark:

```sh
python benchmark_vlm.py [OPTIONS]
```

### Options

- `-m, --model` (required): Path to the model and tokenizers base directory.
- `-D, --draft_model` (default: `None`): Path to the draft model and tokenizers base directory. Not supported with `-d NPU`.
- `-A, --num_assistant_tokens` (default: `5`): Number of assistant tokens used for speculative decoding when `--draft_model` is provided.
- `-p, --prompt` (default: `None`): The prompt to generate text. If without `-p, --prompt`, and `-F, --prompt_file`, the default prompt is `"What is on the image?"`
- `-F, --prompt_file`: Read prompt from file.
- `-i, --image` (default: `image.jpg`): Path to image. Can be a single image or a directory of images.
- `-H, --image_height` (default: `None`): Target image height for resizing. Must be a positive value and provided together with `-W, --image_width`.
- `-W, --image_width` (default: `None`): Target image width for resizing. Must be a positive value and provided together with `-H, --image_height`.
- `-N, --num_warmup` (default: `1`): Number of warmup iterations.
- `-M, --max_new_tokens` (default: `20`): Maximal number of new tokens.
- `-n, --num_iter` (default: `2`): Number of iterations.
- `-d, --device` (default: `"CPU"`): Device to run the model on.
- `-P, --pruning_ratio` (optional): Percentage of visual tokens to prune (valid range: 0-100). If this option is not provided, pruning is disabled.
- `-R, --relevance_weight` (optional): Float value from 0 to 1, controls the trade-off between diversity and relevance for visual tokens pruning; a value of 0 disables relevance weighting, while higher values (up to 1.0) emphasize relevance, making pruning more conservative on borderline tokens.

### Output:

```
python benchmark_vlm.py -m Qwen3-VL-2B-Instruct -i 319483352-d5fbbd1a-d484-415c-88cb-9986625b7b11.jpg -n 3 -d GPU
```

```
Number of images: 1, Prompt token size: 6
Input token size: 667
Output token size: 20
Load time: 17628.00 ms
Generate time: 487.58 ± 6.96 ms
Tokenization time: 16.46 ± 0.08 ms
Detokenization time: 0.24 ± 0.02 ms
Embeddings preparation time: 143.88 ± 0.00 ms
TTFT: 229.26 ± 7.68 ms
TPOT: 13.52 ± 3.03 ms/token
Throughput: 73.97 ± 16.59 tokens/s
```

* With different image size

	```
	python benchmark_vlm.py -m Qwen3-VL-2B-Instruct -i 319483352-d5fbbd1a-d484-415c-88cb-9986625b7b11.jpg -n 3 -d GPU -H 224 -W 224
	```

	```
	Number of images: 1, Prompt token size: 6
	Image is resized to: 224x224
	Input token size: 80
	Output token size: 20
	Load time: 4460.00 ms
	Generate time: 323.61 ± 8.40 ms
	Tokenization time: 15.80 ± 0.05 ms
	Detokenization time: 0.24 ± 0.01 ms
	Embeddings preparation time: 36.70 ± 0.00 ms
	TTFT: 72.10 ± 4.37 ms
	TPOT: 13.21 ± 3.34 ms/token
	Throughput: 75.69 ± 19.10 tokens/s
	```

For more information on how performance metrics are calculated please follow [performance-metrics tutorial](../../../src/README.md#performance-metrics).

### Troubleshooting

#### Unicode characters encoding error on Windows

Example error:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u25aa' in position 0: character maps to <undefined>
```

If you encounter the error described in the example when sample is printing output to the Windows console, it is likely due to the default Windows encoding not supporting certain Unicode characters. To resolve this:
1. Enable Unicode characters for Windows cmd - open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.
2. Enable UTF-8 mode by setting environment variable `PYTHONIOENCODING="utf8"`.
