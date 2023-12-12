# Casual LM

This application showcases inference of casual language model (LM). It doesn't have much of configuration options to encourage the reader to explore and modify the source code. There's a Jupyter notebook which corresponds to this pipeline and discusses how to create an LLM-powered Chatbot: https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot.

> [!NOTE]
> This project is not for production use.

## How it works

The program loads a tokenizer, detokenizer and a model (`.xml` and `.bin`) to OpenVINO. The model is reshaped to batch 1 and variable prompt length. A prompt is tokenized and passed to the model. The model greedily generates token by token until the special end of sequence (EOS) token is obtained. The predicted tokens are converted to chars and printed in a streaming fashion.

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
3. [Llama2-7b-WhoIsHarryPotter](https://huggingface.co/microsoft/Llama2-7b-WhoIsHarryPotter)

This pipeline can work with other similar topologies produced by `optimum-intel` with the same model signature.

### Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

```sh
source <INSTALL_DIR>/setupvars.sh
python -m pip install --upgrade-strategy eager transformers==4.35.2 "optimum[openvino]>=1.14" ../../../thirdparty/openvino_contrib/modules/custom_operations/[transformers] --extra-index-url https://download.pytorch.org/whl/cpu
optimum-cli export openvino -m meta-llama/Llama-2-7b-hf ./Llama-2-7b-hf/
python ./convert_tokenizers.py ./Llama-2-7b-hf/
```

## Run

Usage: `casual_lm <openvino_model.xml> <tokenizer.xml> <detokenizer.xml> "<prompt>"`

Example: `./build/casual_lm ./Llama-2-7b-hf/openvino_model.xml ./tokenizer.xml ./detokenizer.xml "Why is the Sun yellow?"`

To enable Unicode characters for Windows cmd open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.
