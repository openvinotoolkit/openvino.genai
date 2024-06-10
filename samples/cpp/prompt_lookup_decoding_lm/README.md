# prompt_lookup_decoding_lm C++ sample that supports most popular models like LLaMA 2

[Prompt Lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) is [assested-generation](https://huggingface.co/blog/assisted-generation#understanding-text-generation-latency) technique where the draft model is replaced with simple string matching the prompt to generate candidate token sequences. This method highly effective for input grounded generation (summarization, document QA, multi-turn chat, code editing), where there is high n-gram overlap between LLM input (prompt) and LLM output. This could be entity names, phrases, or code chunks that the LLM directly copies from the input while generating the output. Prompt lookup exploits this pattern to speed up autoregressive decoding in LLMs. This results in significant speedups with no effect on output quality.

This example showcases inference of text-generation Large Language Models (LLMs): `chatglm`, `LLaMA`, `Qwen` and other models with the same signature. The application don't have many configuration options to encourage the reader to explore and modify the source code.  Loading `openvino_tokenizers` to `ov::Core` enables tokenization. Run `optimum-cli` to generate IRs for the samples. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot) which provides an example of LLM-powered Chatbot in Python.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

```sh
source <INSTALL_DIR>/setupvars.sh
python3 -m pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --trust-remote-code --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
```

## Run

`prompt_lookup_decoding_lm ./TinyLlama-1.1B-Chat-v1.0/ "return 0;"`

To enable Unicode characters for Windows cmd open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.

Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM. For example, the model meta-llama/Llama-2-13b-chat-hf can benefit from being run on a dGPU. Modify the source code to change the device for inference to the GPU.

See https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md#supported-models for the list of supported models.
