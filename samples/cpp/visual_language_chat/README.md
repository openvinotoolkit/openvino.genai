# C++ visual language chat

This example showcases inference of Visual language models (VLMs): [`openbmb/MiniCPM-V-2_6`](https://huggingface.co/openbmb/MiniCPM-V-2_6). The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `ov::genai::VLMPipeline` and runs the simplest deterministic greedy sampling algorithm. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/minicpm-v-multimodal-chatbot) which provides an example of Visual-language assistant.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

It's not required to install [../../export-requirements.txt](../../export requirements.txt) for deployment if the model has already been exported.

```sh
pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --model openbmb/MiniCPM-V-2_6 --trust-remote-code MiniCPM-V-2_6
```

## Run

Follow [Get Started with Samples](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/get-started-demos.html) to run the sample.

[This image](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11) can be used as a sample image.

`visual_language_chat miniCPM-V-2_6 319483352-d5fbbd1a-d484-415c-88cb-9986625b7b11.jpg`

Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM. For example, the model `llava-hf/llava-v1.6-mistral-7b-hf` can benefit from being run on a dGPU. Modify the source code to change the device for inference to the `GPU`.

See [SUPPORTED_MODELS.md](../../../src/docs/SUPPORTED_MODELS.md#visual-language-models) for the list of supported models.

### Troubleshooting

#### Unicode characters encoding error on Windows

Example error:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u25aa' in position 0: character maps to <undefined>
```

If you encounter the error described in the example when sample is printing output to the Windows console, it is likely due to the default Windows encoding not supporting certain Unicode characters. To resolve this:
1. Enable Unicode characters for Windows cmd - open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.
2. Enable UTF-8 mode by setting environment variable `PYTHONIOENCODING="utf8"`.
