# Causal LM

These applications showcase inference of a causal language model (LM). They don't have many configuration options to encourage the reader to explore and modify the source code. There's a Jupyter notebook which corresponds to these pipelines and discusses how to create an LLM-powered Chatbot: https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot.

> [!NOTE]
> This project is not for production use.

## How it works

### causal_lm

The program loads a tokenizer, a detokenizer and a model (`.xml` and `.bin`) to OpenVINO. A prompt is tokenized and passed to the model. The model greedily generates token by token until the special end of sequence (EOS) token is obtained. The predicted tokens are converted to chars and printed in a streaming fashion.

### beam_search_causal_lm

The program loads a tokenizer, a detokenizer and a model (`.xml` and `.bin`) to OpenVINO. A prompt is tokenized and passed to the model. The model predicts a distribution over the next tokens and group beam search samples from that distribution to explore possible sequesnses. The result is converted to chars and printed.

## Install OpenVINO Runtime

Install OpenVINO Runtime from an archive: [Linux](https://docs.openvino.ai/2023.2/openvino_docs_install_guides_installing_openvino_from_archive_linux.html). `<INSTALL_DIR>` below refers to the extraction location.

## Build `causal_lm`, `beam_search_causal_lm` and `user_ov_extensions`

```sh
git submodule update --init
source <INSTALL_DIR>/setupvars.sh
cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/ && cmake --build ./build/ --config Release -j
```

## Supported models

1. LLaMA 2
   4. https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
   3. https://huggingface.co/meta-llama/Llama-2-13b-hf
   2. https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
   1. https://huggingface.co/meta-llama/Llama-2-7b-hf
   6. https://huggingface.co/meta-llama/Llama-2-70b-chat-hf
   5. https://huggingface.co/meta-llama/Llama-2-70b-hf
2. [Llama2-7b-WhoIsHarryPotter](https://huggingface.co/microsoft/Llama2-7b-WhoIsHarryPotter)
3. OpenLLaMA
   3. https://huggingface.co/openlm-research/open_llama_13b
   1. https://huggingface.co/openlm-research/open_llama_3b
   4. https://huggingface.co/openlm-research/open_llama_3b_v2
   2. https://huggingface.co/openlm-research/open_llama_7b
   5. https://huggingface.co/openlm-research/open_llama_7b_v2
4. TinyLlama
   1. https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.6
   2. https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T

This pipeline can work with other similar topologies produced by `optimum-intel` with the same model signature.

### Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.
`beam_search_causal_lm` requires ommiting `--streaming-detokenizer` for `convert_tokenizers.py`.

```sh
source <INSTALL_DIR>/setupvars.sh
python -m pip install --upgrade-strategy eager "optimum[openvino]>=1.14" -r ../../../llm_bench/python/requirements.txt ../../../thirdparty/openvino_contrib/modules/custom_operations/[transformers] --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip uninstall openvino  # Uninstall openvino from PyPI because there's one from the archive installed
python ../../../llm_bench/python/convert.py --model_id meta-llama/Llama-2-7b-hf --output_dir ./Llama-2-7b-hf/ --precision FP16 --stateful
python ./convert_tokenizers.py --streaming-detokenizer ./Llama-2-7b-hf/pytorch/dldt/FP16/
```

## Run

Usage:
1. `greedy_causal_lm <MODEL_DIR> "<PROMPT>"`
2. `beam_search_causal_lm <MODEL_DIR> "<PROMPT>"`

Examples:
1. `./build/greedy_causal_lm ./Llama-2-7b-hf/pytorch/dldt/FP16/ "Why is the Sun yellow?"`
2. `./build/beam_search_causal_lm ./Llama-2-7b-hf/pytorch/dldt/FP16/ "Why is the Sun yellow?"`

To enable Unicode characters for Windows cmd open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.
