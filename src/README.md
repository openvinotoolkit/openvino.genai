# OpenVINO™ GenAI Library

OpenVINO™ GenAI is a flavor of OpenVINO™, aiming to simplify running inference of generative AI models.
It hides the complexity of the generation process and minimizes the amount of code required.

## Install OpenVINO™ GenAI

> **NOTE**: Please make sure that you are following the versions compatibility rules, refer to the [OpenVINO™ GenAI Dependencies](#openvino-genai-dependencies) for more information.

The OpenVINO™ GenAI flavor is available for installation via Archive and PyPI distributions.
To install OpenVINO™ GenAI, refer to the [Install Guide](https://docs.openvino.ai/2025/get-started/install-openvino.html).

To build OpenVINO™ GenAI library from source, refer to the [Build Instructions](./docs/BUILD.md).

### OpenVINO™ GenAI Dependencies

OpenVINO™ GenAI depends on [OpenVINO](https://github.com/openvinotoolkit/openvino) and [OpenVINO Tokenizers](https://github.com/openvinotoolkit/openvino_tokenizers).

When installing OpenVINO™ GenAI from PyPi, the same versions of OpenVINO and OpenVINO Tokenizers are used (e.g. `openvino==2024.3.0` and `openvino-tokenizers==2024.3.0.0` are installed for `openvino-genai==2024.3.0`).
If you update one of the dependency packages (e.g. `pip install openvino --pre --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly`), versions might be incompatible due to different ABI and running OpenVINO GenAI can result in errors (e.g. `ImportError: libopenvino.so.2430: cannot open shared object file: No such file or directory`).
Having packages version in format `<MAJOR>.<MINOR>.<PATCH>.<REVISION>`, only `<REVISION>` part of the full version can be varied to ensure ABI compatibility, while changing `<MAJOR>`, `<MINOR>` or `<PATCH>` parts of the version might break ABI.

GenAI, Tokenizers, and OpenVINO wheels for Linux on PyPI are compiled with `_GLIBCXX_USE_CXX11_ABI=0` to cover a wider range of platforms. In contrast, C++ archive distributions for Ubuntu are compiled with `_GLIBCXX_USE_CXX11_ABI=1`. It is not possible to mix different Application Binary Interfaces (ABIs) because doing so results in a link error. This incompatibility prevents the use of, for example, OpenVINO from C++ archive distributions alongside GenAI from PyPI.

If you want to try OpenVINO GenAI with different dependencies versions (**not** prebuilt packages as archives or python wheels), build OpenVINO GenAI library from source.

## Usage

### Prerequisites

1. Installed OpenVINO™ GenAI

    > To use OpenVINO GenAI with models that are already in OpenVINO format, no additional python dependencies are needed. To
    > convert models with optimum-cli and to run the examples, install the dependencies in [./samples/requirements.txt](./samples/requirements.txt):
    ```sh
    # (Optional) Clone OpenVINO GenAI repository if it does not exist
    git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
    cd openvino.genai
    # Install python dependencies
    python -m pip install ./thirdparty/openvino_tokenizers/[transformers] --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    python -m pip install --upgrade-strategy eager -r ./samples/requirements.txt
    ```

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
pipe = ov_genai.LLMPipeline(models_path, "CPU")
print(pipe.generate("The Sun is yellow because", max_new_tokens=100))
```

Calling generate with custom generation config parameters, e.g. config for grouped beam search:
```python
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline(models_path, "CPU")

result = pipe.generate("The Sun is yellow because", max_new_tokens=100, num_beam_groups=3, num_beams=15, diversity_penalty=1.5)
print(result)
```

output:
```
'it is made up of carbon atoms. The carbon atoms are arranged in a linear pattern, which gives the yellow color. The arrangement of carbon atoms in'
```

>**Note**: The chat_template from tokenizer_config.json or from tokenizer/detokenizer model will be automatically applied to the prompt at the generation stage. If you want to disable it, you can do it by calling pipe.get_tokenizer().set_chat_template("").

A simple chat in Python:
```python
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline(models_path)

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
    std::string models_path = argv[1];
    ov::genai::LLMPipeline pipe(models_path, "CPU");
    std::cout << pipe.generate("The Sun is yellow because", ov::genai::max_new_tokens(256));
}
```

Using group beam search decoding:
```cpp
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string models_path = argv[1];
    ov::genai::LLMPipeline pipe(models_path, "CPU");

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

    std::string models_path = argv[1];
    ov::genai::LLMPipeline pipe(models_path, "CPU");

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
    std::string models_path = argv[1];
    ov::genai::LLMPipeline pipe(models_path, "CPU");

    auto streamer = [](std::string word) {
        std::cout << word << std::flush;
        // Return flag corresponds whether generation should be stopped.
        return ov::genai::StreamingStatus::RUNNING;
    };
    std::cout << pipe.generate("The Sun is yellow because", ov::genai::streamer(streamer), ov::genai::max_new_tokens(200));
}
```

Streaming with a custom class:

C++ template for a streamer.
```cpp
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

class CustomStreamer: public ov::genai::StreamerBase {
public:
    bool put(int64_t token) {
        // Custom decoding/tokens processing logic.

        // Returns a flag whether generation should be stopped, if true generation stops.
        return false;
    };

    void end() {
        // Custom finalization logic.
    };
};

int main(int argc, char* argv[]) {
    CustomStreamer custom_streamer;

    std::string models_path = argv[1];
    ov::genai::LLMPipeline pipe(models_path, "CPU");
    std::cout << pipe.generate("The Sun is yellow because", ov::genai::max_new_tokens(15), ov::genai::streamer(custom_streamer));
}
```

Python template for a streamer.
```py
import openvino_genai as ov_genai

class CustomStreamer(ov_genai.StreamerBase):
    def __init__(self):
        super().__init__()
        # Initialization logic.

    def write(self, token_id) -> ov_genai.StreamingStatus:
        # Custom decoding/tokens processing logic.

        # Returns a status whether generation should be stopped or continue.
        return ov_genai.StreamingStatus.RUNNING

    def end(self):
        # Custom finalization logic.

pipe = ov_genai.LLMPipeline(models_path, "CPU")
custom_streamer = CustomStreamer()

pipe.generate("The Sun is yellow because", max_new_tokens=15, streamer=custom_streamer)
```
For fully implemented iterable CustomStreamer please refer to [multinomial_causal_lm](../samples/python/text_generation/README.md) sample.


Continuous batching with LLMPipeline:

To activate continuous batching please provide additional property to LLMPipeline config: ov::genai::scheduler_config. This property contains struct SchedulerConfig.
```cpp
#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) {
    ov::genai::SchedulerConfig scheduler_config;
    // fill other fields in scheduler_config with custom data if required
    scheduler_config.cache_size = 1;    // minimal possible KV cache size in GB, adjust as required

    ov::genai::LLMPipeline pipe(models_path, "CPU", ov::genai::scheduler_config(scheduler_config));
}
```

### Performance Metrics

Refer to the [Performance Metrics](https://openvinotoolkit.github.io/openvino.genai/docs/guides/performance-metrics) page for details and usage examples.

### Structured Output generation
OpenVINO™ GenAI supports structured output generation, which allows you to generate outputs in a structured format such as JSON, regex, or according to EBNF (Extended Backus–Naur form) grammar.

Below is a minimal example that demonstrates how to use OpenVINO™ GenAI to generate structured JSON output for a single item type (e.g., `person`). This example uses a Pydantic schema to define the structure and constraints of the generated output.

```python
import json
from openvino_genai import LLMPipeline, GenerationConfig, StructuredOutputConfig
from pydantic import BaseModel, Field

# Define the schema for a person
class Person(BaseModel):
    name: str = Field(pattern=r"^[A-Z][a-z]{1,20}$")
    surname: str = Field(pattern=r"^[A-Z][a-z]{1,20}$")
    age: int
    city: str

pipe = LLMPipeline(models_path, "CPU")

config = GenerationConfig()
config.max_new_tokens = 100
# If backend is not specified, it will use the default backend which is "xgrammar" for the moment.
config.structured_output_config = StructuredOutputConfig(json_schema=json.dumps(Person.model_json_schema()), backend="xgrammar")

# Generate structured output
result = pipe.generate("Generate a JSON for a person.", config)
print(json.loads(result))
```

This will generate a JSON object matching the `Person` schema, for example:
```json
{
  "name": "John",
  "surname": "Doe",
  "age": 30,
  "city": "Dublin"
}
```
**Note:**
Structured output enforcement guarantees correct JSON formatting, but does not ensure the factual correctness or sensibility of the content. The model may generate implausible or nonsensical data, such as `{"name": "John", "age": 200000}` or `{"model": "AbrakaKadabra9999######4242"}`. These are valid JSONs but may not make sense. For best results, use the latest or fine-tuned models for this task to improve the quality and relevance of the generated output.


### Tokenization

Refer to the [Tokenization](https://openvinotoolkit.github.io/openvino.genai/docs/guides/tokenization) page for details and usage examples.

## How It Works

For information on how OpenVINO™ GenAI works, refer to the [How It Works](https://openvinotoolkit.github.io/openvino.genai/docs/concepts/how-it-works) page.

## Supported Models

For a list of supported models, refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/) page.

## Debug Log

For using debug log, refer to the [Debug Logging](https://openvinotoolkit.github.io/openvino.genai/docs/guides/debug-logging) page.
