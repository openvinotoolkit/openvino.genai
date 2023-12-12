# LLM

This application showcases inference of a large language model (LLM). It doesn't have much of configuration options to encourage the reader to explore and modify the source code. There's a Jupyter notebook which corresponds to this pipeline and discusses how to create an LLM-powered Chatbot: https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot.

> [!NOTE]
> This pipeline is not for production use.

## How it works

The program loads a model a tokenizer, and a detokenizer (`.xml` and `.bin`) to OpenVINO. The model is reshaped to variable batch size and prompt length. A prompt is tokenized and passed to the model. The model predicts a distribution over the next tokens and group beam search samples from that distribution to explore possible sequesnses. The result is converted to chars and printed.

## Install OpenVINO Runtime

Install OpenVINO Runtime from an archive: [Linux](https://docs.openvino.ai/2023.2/openvino_docs_install_guides_installing_openvino_from_archive_linux.html). `<INSTALL_DIR>` below refers to the extraction location.

## Build `Casual LM` and `user_ov_extensions`

```sh
git submodule update --init
source <INSTALL_DIR>/setupvars.sh
cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/ && cmake --build ./build/ --config Release -j
```

## Supported models

1. LLaMA 2
   1. https://huggingface.co/meta-llama/Llama-2-7b-hf
   2. https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
   3. https://huggingface.co/meta-llama/Llama-2-13b-hf
   4. https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
   5. https://huggingface.co/meta-llama/Llama-2-70b-hf
   6. https://huggingface.co/meta-llama/Llama-2-70b-chat-hf
2. OpenLLaMA
   1. https://huggingface.co/openlm-research/open_llama_3b
   2. https://huggingface.co/openlm-research/open_llama_7b
   3. https://huggingface.co/openlm-research/open_llama_13b
   4. https://huggingface.co/openlm-research/open_llama_3b_v2
   5. https://huggingface.co/openlm-research/open_llama_7b_v2
4. TinyLlama
   1. https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T
   2. https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.6
4. [Llama2-7b-WhoIsHarryPotter](https://huggingface.co/microsoft/Llama2-7b-WhoIsHarryPotter)

This pipeline can work with other similar topologies produced by `optimum-intel` with the same model signature.

### Download and convert the model and tokenizers

```sh
source <OpenVINO dir>/setupvars.sh
python -m pip install --upgrade-strategy eager transformers==4.35.2 "optimum[openvino]>=1.14" ../../thirdparty/openvino_contrib/modules/custom_operations/[transformers] --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip uninstall --yes optimum-intel
python -m pip install git+https://github.com/huggingface/optimum-intel.git@5dac93d6e8d15c96fe061c653d82b7afd54954db
optimum-cli export openvino -m meta-llama/Llama-2-7b-hf ./Llama-2-7b-hf/
python ./llm/cpp/convert_tokenizers.py ./Llama-2-7b-hf/
```

## Run

Usage: `llm <openvino_model.xml> <tokenizer.xml> <detokenizer.xml> "<prompt>"`

Example: `./build/llm/cpp/llm ./Llama-2-7b-hf/openvino_model.xml ./tokenizer.xml ./detokenizer.xml "Why is the Sun yellow?"`
