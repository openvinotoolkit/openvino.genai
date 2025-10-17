# C++ visual language chat

This example showcases inference of Visual language models (VLMs). The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `ov::genai::VLMPipeline` and runs the simplest deterministic greedy sampling algorithm. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/minicpm-v-multimodal-chatbot) which provides an example of Visual-language assistant.


There are two sample files:
 - [`visual_language_chat.cpp`](./visual_language_chat.cpp) demonstrates basic usage of the VLM pipeline.
 - [`benchmark_vlm.cpp`](./benchmark_vlm.cpp) shows how to benchmark a VLM in OpenVINO GenAI. The script includes functionality for warm-up iterations, generating text and calculating various performance metrics.


## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

It's not required to install [../../export-requirements.txt](../../export-requirements.txt) for deployment if the model has already been exported.

```sh
pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --model openbmb/MiniCPM-V-2_6 --trust-remote-code MiniCPM-V-2_6
```

## Run

Follow [Get Started with Samples](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/get-started-demos.html) to run the sample.

[This image](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11) can be used as a sample image.

`visual_language_chat miniCPM-V-2_6 319483352-d5fbbd1a-d484-415c-88cb-9986625b7b11.jpg`

Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM. For example, the model `llava-hf/llava-v1.6-mistral-7b-hf` can benefit from being run on a dGPU. Modify the source code to change the device for inference to the `GPU`.

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#visual-language-models-vlms) for more details.

## Run benchmark:

```sh
benchmark_vlm [OPTIONS]
```

### Options

- `-m, --model`(default: `.`): Path to the model and tokenizers base directory.
- `-p, --prompt` (default: ''): The prompt to generate text. If without `-p` and `--pf`, the default prompt is `"What is on the image?"`
- `--pf, --prompt_file` Read prompt from file.
- `-i, --image` (default: `image.jpg`): Path to the image.
- `-nw, --num_warmup` (default: `1`): Number of warmup iterations.
- `-mt, --max_new_tokens` (default: `20`): Maximal number of new tokens.
- `-n, --num_iter` (default: `3`): Number of iterations.
- `-d, --device` (default: `"CPU"`): Device to run the model on.

### Output:

```
benchmark_vlm -m miniCPM-V-2_6 -i 319483352-d5fbbd1a-d484-415c-88cb-9986625b7b11.jpg -n 3
```

```
Load time: 1982.00 ms
Generate time: 13820.99 ± 64.62 ms
Tokenization time: 1.26 ± 0.09 ms
Detokenization time: 0.33 ± 0.05 ms
Embeddings preparation time: 5733.85 ± 26.34 ms
TTFT: 11246.98 ± 80.55 ms
TPOT: 135.45 ± 4.73 ms/token 
Throughput: 7.38 ± 0.26 tokens/s
```

For more information how performance metrics are calculated please follow [performance-metrics tutorial](../../../src/README.md#performance-metrics).

### Troubleshooting

#### Unicode characters encoding error on Windows

Example error:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u25aa' in position 0: character maps to <undefined>
```

If you encounter the error described in the example when sample is printing output to the Windows console, it is likely due to the default Windows encoding not supporting certain Unicode characters. To resolve this:
1. Enable Unicode characters for Windows cmd - open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.
2. Enable UTF-8 mode by setting environment variable `PYTHONIOENCODING="utf8"`.
