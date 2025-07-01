# Python vlm_chat_sample that supports VLM models

This example showcases inference of text-generation Vision Language Models (VLMs): `miniCPM-V-2_6` and other models with the same signature. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `openvino_genai.VLMPipeline` and configures it for the chat scenario. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/minicpm-v-multimodal-chatbot) which provides an example of Visual-language assistant.

There are two sample files:
 - [`visual_language_chat.py`](./visual_language_chat.py) demonstrates basic usage of the VLM pipeline.
 - [`benchmark_vlm.py`](./benchmark_vlm.py) shows how to benchmark a VLM in OpenVINO GenAI. The script includes functionality for warm-up iterations, generating text and calculating various performance metrics.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
```

Then, run the export with Optimum CLI:

```sh
optimum-cli export openvino --model openbmb/MiniCPM-V-2_6 --trust-remote-code MiniCPM-V-2_6
```

Alternatively, you can do it in Python code:

```python
from optimum.exporters.openvino.convert import export_tokenizer
from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoTokenizer

output_dir = "MiniCPM-V-2_6"

model = OVModelForVisualCausalLM.from_pretrained("openbmb/MiniCPM-V-2_6", export=True, trust_remote_code=True)
model.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-2_6")
export_tokenizer(tokenizer, output_dir)
```

## Run:

[This image](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11) can be used as a sample image.

Install [deployment-requirements.txt](../../deployment-requirements.txt) via `pip install -r ../../deployment-requirements.txt` and then, run a sample:

`python visual_language_chat.py ./miniCPM-V-2_6/ 319483352-d5fbbd1a-d484-415c-88cb-9986625b7b11.jpg`


Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM. # TODO: examples of larger models
Modify the source code to change the device for inference to the GPU.

See https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md#supported-models for the list of supported models.

## Run benchmark:

```sh
python benchmark_vlm.py [OPTIONS]
```

### Options

- `-m, --model`(default: `.`): Path to the model and tokenizers base directory.
- `-p, --prompt` (default: `None`): The prompt to generate text. If without `-p` and `-pf`, the default prompt is `"What is on the image?"`
- `-pf, --prompt_file` Read prompt from file.
- `-i, --image` (default: `image.jpg`): Path to the image.
- `-nw, --num_warmup` (default: `1`): Number of warmup iterations.
- `-mt, --max_new_tokens` (default: `20`): Maximal number of new tokens.
- `-n, --num_iter` (default: `3`): Number of iterations.
- `-d, --device` (default: `"CPU"`): Device to run the model on.

### Output:

```
python benchmark_vlm.py -m miniCPM-V-2_6 -i 319483352-d5fbbd1a-d484-415c-88cb-9986625b7b11.jpg -n 3
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
