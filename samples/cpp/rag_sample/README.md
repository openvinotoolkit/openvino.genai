# C++ rag sample that supports most popular models like LLaMA 2

This example showcases for Retrieval-Augmented Generation based on text-generation Large Language Models (LLMs): `chatglm`, `LLaMA`, `Qwen` and other models with the same signature and bert model for embedding feature extraction. The sample fearures `ov::genai::LLMPipeline` and configures it for the chat scenario. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot) which provides an example of LLM-powered RAG in Python.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

```sh
python3 -m pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --trust-remote-code --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
```

## Run:
### Launch RAG Server
`rag_sample_server --llm_model_path TinyLlama-1.1B-Chat-v1.0 --llm_device CPU`

### Lanuch RAG Client
`rag_sample_client`

To enable Unicode characters for Windows cmd open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.

Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM. For example, the model meta-llama/Llama-2-13b-chat-hf can benefit from being run on a dGPU. Modify the source code to change the device for inference to the GPU.

See https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md#supported-models for the list of supported models.
