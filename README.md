# OpenVINO™ GenAI

OpenVINO™ GenAI is a library of most popular Generative AI model pipelines, optimized execution methods and samples that runs on top of highly performant [OpenVINO Runtime](https://github.com/openvinotoolkit/openvino).

Library is friendly to PC and laptop execution, optimized for resource consumption and requires no external dependencies to run generative models and includes all required functionality (e.g. tokenization via openvino-tokenizers).

(TBD plug small gif of generation)

## Supported Generative AI scenarios

OpenVINO™ GenAI library provides very lightweight C++ and Python APIs to run following Generative Scenarios:
 - Text generation using Large Language Models. For example, chat with local LLaMa model
 - Image generation using Diffuser models, for example generation using Stable Diffusion models
 - Speech recognition using Whisper family models
 - Text generation using Large Visual Models, for instance Image analysis using LLaVa models family

Library efficiently supports LoRA adapters for Text and Image generation scenarios:
- Load multiple adapters per model
- Select active adapters for every generation
- Mix multiple adapters with coefficients via alpha blending

All scenarios are run on top of OpenVINO Runtime that supports inference on CPU, GPU and NPU. See [here](https://github.com/openvinotoolkit/openvino) for platform support matrix.

## Supported Generative AI optimization methods

OpenVINO™ GenAI library provides transparent way to use state of the art generation optimizations:
- Prefix caching that caches fragments of previous generation requests and corresponding KVCache entries internally and uses them in case of repeated query. See [here](https://google.com) for more detailed overview
- Speculative decoding that employs two models of different size and uses large model to periodically correct results of small model. See [here](https://google.com) for more detailed overview
- Lookahead decoding that attempts to guess multiple tokens based on historical context and corrects this guess using LLMs. See [here](https://google.com) for more detailed overview
- KVCache token eviction algorithm that reduces size of the KVCache by pruning less impacting tokens. See [here](https://google.com) for more detailed overview

Additionally, OpenVINO™ GenAI library implements continuous batching approach that allows processing multiple generation requests simultaneously and efficiently use compute resources. 

## Installing OpenVINO GenAI

```sh
    # Installing OpenVINO GenAI via pip
    pip install openvino-genai

    # Install optimum-intel to be able to download, convert and optimize LLMs from Hugging Face
    # Optimum is not required to run models, only to convert and compress
    pip install optimum[openvino]

    # (Optional) Install (TBD) to be able to download models from Model Scope
    #pip install optimum[openvino]
```

## Performing text generation 
<details open>
For more examples check out our [LLM Cheat Sheet](https://docs.openvino.ai)

### Converting and compressing text generation model from Hugging Face library

```sh
#(Basic) download and convert to OpenVINO TinyLlama-Chat-v1.0 model
optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --weight-format fp16 --trust-remote-code "TinyLlama-1.1B-Chat-v1.0"

#(Recommended) download, convert to OpenVINO and compress to int4 TinyLlama-Chat-v1.0 model
optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --weight-format int4 --trust-remote-code "TinyLlama-1.1B-Chat-v1.0"
```

### Run generation using LLMPipeline API in Python

```python
import openvino_genai as ov_genai
#Will run model on CPU, GPU or NPU are possible options
pipe = ov_genai.LLMPipeline("./TinyLlama-1.1B-Chat-v1.0/", "CPU")
print(pipe.generate("The Sun is yellow because", max_new_tokens=100))
```

### Run generation using LLM Pipeline in C++

Code below requires installation of C++ compatible package (see [here](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-genai.html#archive-installation) for more details)

```cpp
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
   std::string model_path = argv[1];
   ov::genai::LLMPipeline pipe(model_path, "CPU");
   std::cout << pipe.generate("The Sun is yellow because", ov::genai::max_new_tokens(100));
}
```
</details>

## Performing image generation (TBD)

<details>
For more examples check out our [LLM Cheat Sheet](https://docs.openvino.ai)

### Converting and compressing text generation model from Hugging Face library

```sh
#(Basic) download and convert to OpenVINO TinyLlama-Chat-v1.0 model
optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --weight-format fp16 --trust-remote-code "TinyLlama-1.1B-Chat-v1.0"

#(Recommended) download, convert to OpenVINO and compress to int4 TinyLlama-Chat-v1.0 model
optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --weight-format int4 --trust-remote-code "TinyLlama-1.1B-Chat-v1.0"
```

### Run generation using LLMPipeline API in Python

```python
import openvino_genai as ov_genai
#Will run model on CPU, GPU or NPU are possible options
pipe = ov_genai.LLMPipeline("./TinyLlama-1.1B-Chat-v1.0/", "CPU")
print(pipe.generate("The Sun is yellow because", max_new_tokens=100))
```

### Run generation using LLM Pipeline in C++

Code below requires installation of C++ compatible package (see [here](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-genai.html#archive-installation) for more details)

```cpp
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
   std::string model_path = argv[1];
   ov::genai::LLMPipeline pipe(model_path, "CPU");
   std::cout << pipe.generate("The Sun is yellow because", ov::genai::max_new_tokens(100));
}
```
</details>

## License

The OpenVINO™ GenAI repository is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release
your contribution under these terms.
