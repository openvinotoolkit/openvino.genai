# VLMs benchmarking sample

This sample script demonstrates how to benchmark a VLM in OpenVINO GenAI. The script includes functionality for warm-up iterations, generating text, and calculating various performance metrics.

## Download and convert the model and tokenizers and download an image

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

It's not required to install [../../export-requirements.txt](../../export requirements.txt) for deployment if the model has already been exported.

```sh
pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --model openbmb/MiniCPM-V-2_6 --trust-remote-code MiniCPM-V-2_6
```

[This image](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11) can be used as a sample image.

## Usage

Follow [Get Started with Samples](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/get-started-demos.html) to run the sample.

```sh
benchmark_vlm [OPTIONS]
```

### Options

- `-m, --model`: Path to the model and tokenizers base directory.
- `-p, --prompt` The prompt to generate text.
- `-i, --image`: Path to the image.
- `-nw, --num_warmup` (default: `1`): Number of warmup iterations.
- `-mt, --max_new_tokens` (default: `20`): Number of warmup iterations.
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
