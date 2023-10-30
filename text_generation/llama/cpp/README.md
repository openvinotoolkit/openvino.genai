# LLaMA

This application showcases inference of a large language model (LLM). It doesn't have much of configuration options to encourage the reader to explore and modify the source code. There's a Jupyter notebook which corresponds to this pipeline and discusses how to create an LLM-powered Chatbot: https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot.

> Note  
This pipeline is not for production use.

## How it works

The program loads a tokenizer, detokenizer and a model (`.xml` and `.bin`) to OpenVINO™. The model is reshaped to batch 1 and variable prompt length. A prompt is tokenized and passed to the model. The model greedily generates token by token until the special end of sequence (EOS) token is obtained. The predicted tokens are converted to chars and printed in a streaming fashion.

## Install OpenVINO Runtime

Install OpenVINO Runtime from an archive: [Linux](https://docs.openvino.ai/2023.1/openvino_docs_install_guides_installing_openvino_from_archive_linux.html).
`<INSTALL_DIR>` refers to the extraction location below.

## Build `LLaMA` and `user_ov_extensions`

```sh
git submodule update --init
mkdir ./build/ && cd ./build/
source <OpenVINO dir>/setupvars.sh
cmake -DCMAKE_BUILD_TYPE=Release ../ && cmake --build ./ --config Release -j
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
3. [Llama2-7b-WhoIsHarryPotter](https://huggingface.co/microsoft/Llama2-7b-WhoIsHarryPotter)

### Download and convert the model and tokenizers

> Note  
Windows isn't currently supported by `user_ov_extensions`.

```sh
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu ../../../thirdparty/openvino_contrib/modules/custom_operations/user_ie_extensions/tokenizer/python/[transformers] onnx git+https://github.com/huggingface/optimum-intel.git
source <INSTALL_DIR>/setupvars.sh
optimum-cli export openvino -m meta-llama/Llama-2-7b-hf ./Llama-2-7b-hf/
python ./convert_tokenizers.py ./custom_operations/user_ie_extensions/libuser_ov_extensions.so ./Llama-2-7b-hf/
```

## Run

Usage: `llama <openvino_model.xml> <tokenizer.xml> <detokenizer.xml> "<prompt>"`

Example: `./build/llama ./Llama-2-7b-hf/openvino_model.xml ./tokenizer.xml ./detokenizer.xml "Why is the Sun yellow?"`

To enable non ASCII characters for Windows cmd open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.
