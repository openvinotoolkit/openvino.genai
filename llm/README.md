# LLM

This application showcases inference of a large language model (LLM). It doesn't have much of configuration options to encourage the reader to explore and modify the source code.

It's required to install https://github.com/openvinotoolkit/openvino.

## How it works

The program loads a model (`.xml` and `.bin`) to OpenVINO™ and a provided vocab (`.gguf`) to use for tokenization. The model is reshaped to batch 1 and variable prompt length. A prompt is tokenized and passed to the model. The model greedily generates token by token until the special end of sequence (EOS) token is obtained or the application is killed for example because of out of memory error. The predicted tokens are converted to chars and printed in a streaming fashion.

## Supported models

### LLaMA 2

Follow https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot to convert the model to intermediate representation (IR) (`.xml` and `.bin`) format. Use `thirdparty/llama.cpp/models/ggml-vocab-llama.gguf` as a vocab.

### OpenLLaMA

1. Install dependencies

   `python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu onnx git+https://github.com/huggingface/optimum-intel.git`

2. Download and convert the model

   ```sh
   source <OpenVINO dir>/setupvars.sh
   optimum-cli export openvino -m openlm-research/open_llama_3b_v2 open_llama_3b_v2/
   ```

3. Convert the vocab

   ```sh
   rm open_llama_3b_v2/added_tokens.json  # The file added by optimum-cli confuses llama.cpp
   python demos/thirdparty/llama.cpp/convert.py open_llama_3b_v2/ --vocab-only --outfile open_llama_3b_v2/vocab.gguf
   ```

## Build

```sh
mkdir build
cd build
source <OpenVINO dir>/setupvars.sh
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release -j
```

## Running

Usage: `llm_demo <model_path> <vocab_path> "<prompt>"`

Example: `llm_demo openvino_model.xml vocab.gguf "Why is the Sun yellow?"`

To enable non ASCII characters for Windows cmd open `Region` settings from `Control panel`. `Adiministrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.
