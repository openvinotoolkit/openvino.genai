# OpenVINO™ GenAI Library

OpenVINO™ GenAI is a flavor of OpenVINO™, aiming to simplify running inference of generative AI models.
It hides the complexity of the generation process and minimizes the amount of code required.

## Install OpenVINO™ GenAI

> **NOTE**: Please make sure that you are following the versions compatibility rules, refer to the [OpenVINO™ GenAI Dependencies](#openvino-genai-dependencies) for more information.

The OpenVINO™ GenAI flavor is available for installation via Archive and PyPI distributions.
To install OpenVINO™ GenAI, refer to the [Install Guide](https://docs.openvino.ai/2024/get-started/install-openvino.html).

To build OpenVINO™ GenAI library from source, refer to the [Build Instructions](https://github.com/openvinotoolkit/openvino.genai/tree/releases/2024/3/src/docs/BUILD.md).

### OpenVINO™ GenAI Dependencies

OpenVINO™ GenAI depends on [OpenVINO](https://github.com/openvinotoolkit/openvino) and [OpenVINO Tokenizers](https://github.com/openvinotoolkit/openvino_tokenizers).

When installing OpenVINO™ GenAI from PyPi, the same versions of OpenVINO and OpenVINO Tokenizers are used (e.g. for `openvino-genai==2024.3.0` will be used `openvino==2024.3.0` and `openvino-tokenizers==2024.3.0.0`).
If you update one of the dependency packages (e.g. install `openvino-nightly`), versions might be incompatible due to different ABI and running OpenVINO GenAI can result to errors (e.g. `ImportError: libopenvino.so.2440: cannot open shared object file: No such file or directory`).

`openvino::runtime` exports `_GLIBCXX_USE_CXX11_ABI=0` on CentOS7. It needs to be propagated to every library openvino_tokenizers links with. That prohibits linkage with prebuilt libraries because they aren't compiled with `_GLIBCXX_USE_CXX11_ABI=0`.

Having packages version in format `<MAJOR>.<MINOR>.<PATCH>.<REVISION>`, only `<REVISION>` part of the full version can be varied to ensure ABI compatibility, while changing `<MAJOR>`, `<MINOR>` or `<PATCH>` parts of the version might break ABI.

If you want to try OpenVINO GenAI with different dependencies versions (e.g. nightly or pre-release versions), build OpenVINO GenAI library from source.

## Usage

### Prerequisites

1. Installed OpenVINO™ GenAI

    > If OpenVINO GenAI is installed via archive distribution or built from source, you will need to install additional python dependencies (e.g. `optimum-cli` for simplified model downloading and exporting, it's not required to install [./samples/requirements.txt](./samples/requirements.txt) for deployment if the model has already been exported):
    > 
    > ```sh
    > # (Optional) Clone OpenVINO GenAI repository if it does not exist
    > git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
    > cd openvino.genai
    > # Install python dependencies
    > python -m pip install ./thirdparty/openvino_tokenizers/[transformers] --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release
    > python -m pip install --upgrade-strategy eager -r ./samples/requirements.txt
    > ```

2. A model in OpenVINO IR format

    Download and convert a model with `optimum-cli`:
    ``` sh
    optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --trust-remote-code "TinyLlama-1.1B-Chat-v1.0"
    ```

`LLMPipeline` is the main object used for decoding. You can construct it straight away from the folder with the converted model. It will automatically load the main model, tokenizer, detokenizer and default generation configuration.

### Python

A simple example:
```python
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline(model_path, "CPU")
print(pipe.generate("The Sun is yellow because", max_new_tokens=100))
```

Calling generate with custom generation config parameters, e.g. config for grouped beam search:
```python
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline(model_path, "CPU")

result = pipe.generate("The Sun is yellow because", max_new_tokens=100, num_beam_groups=3, num_beams=15, diversity_penalty=1.5)
print(result)
```

output:
```
'it is made up of carbon atoms. The carbon atoms are arranged in a linear pattern, which gives the yellow color. The arrangement of carbon atoms in'
```

A simple chat in Python:
```python
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline(model_path)

config = {'max_new_tokens': 100, 'num_beam_groups': 3, 'num_beams': 15, 'diversity_penalty': 1.5}
pipe.set_generation_config(config)

pipe.start_chat()
while True:
    print('question:')
    prompt = input()
    if prompt == 'Stop!':
        break
    print(pipe(prompt, max_new_tokens=200))
pipe.finish_chat()
```

Test to compare with Huggingface outputs

### C++

A simple example:
```cpp
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string model_path = argv[1];
    ov::genai::LLMPipeline pipe(model_path, "CPU");
    std::cout << pipe.generate("The Sun is yellow because", ov::genai::max_new_tokens(256));
}
```

Using group beam search decoding:
```cpp
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string model_path = argv[1];
    ov::genai::LLMPipeline pipe(model_path, "CPU");

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 256;
    config.num_beam_groups = 3;
    config.num_beams = 15;
    config.diversity_penalty = 1.0f;

    std::cout << pipe.generate("The Sun is yellow because", config);
}
```

A simple chat in C++ using grouped beam search decoding:
```cpp
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string prompt;

    std::string model_path = argv[1];
    ov::genai::LLMPipeline pipe(model_path, "CPU");
    
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    config.num_beam_groups = 3;
    config.num_beams = 15;
    config.diversity_penalty = 1.0f;
    
    pipe.start_chat();
    for (;;;) {
        std::cout << "question:\n";
        std::getline(std::cin, prompt);
        if (prompt == "Stop!")
            break;

        std::cout << "answer:\n";
        auto answer = pipe(prompt, config);
        std::cout << answer << std::endl;
    }
    pipe.finish_chat();
}
```

Streaming example with lambda function:
```cpp
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string model_path = argv[1];
    ov::genai::LLMPipeline pipe(model_path, "CPU");
        
    auto streamer = [](std::string word) { 
        std::cout << word << std::flush; 
        // Return flag corresponds whether generation should be stopped.
        // false means continue generation.
        return false;
    };
    std::cout << pipe.generate("The Sun is yellow bacause", ov::genai::streamer(streamer), ov::genai::max_new_tokens(200));
}
```

Streaming with a custom class:
```cpp
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

class CustomStreamer: public ov::genai::StreamerBase {
public:
    bool put(int64_t token) {
        bool stop_flag = false; 
        /* 
        custom decoding/tokens processing code
        tokens_cache.push_back(token);
        std::string text = m_tokenizer.decode(tokens_cache);
        ...
        */
        return stop_flag;  // flag whether generation should be stoped, if true generation stops.
    };

    void end() {
        /* custom finalization */
    };
};

int main(int argc, char* argv[]) {
    CustomStreamer custom_streamer;

    std::string model_path = argv[1];
    ov::genai::LLMPipeline pipe(model_path, "CPU");
    std::cout << pipe.generate("The Sun is yellow because", ov::genai::streamer(custom_streamer), ov::genai::max_new_tokens(200));
}
```

## How It Works

For information on how OpenVINO™ GenAI works, refer to the [How It Works Section](https://github.com/openvinotoolkit/openvino.genai/tree/releases/2024/3/src/docs/HOW_IT_WORKS.md).

## Supported Models

For a list of supported models, refer to the [Supported Models Section](https://github.com/openvinotoolkit/openvino.genai/tree/releases/2024/3/src/docs/SUPPORTED_MODELS.md).
