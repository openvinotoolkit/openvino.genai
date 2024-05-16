# OpenVINO Generate API

## Usage 

First of all you need to convert your model with optimum-cli
``` sh
optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --weight-format fp16 --trust-remote-code "TinyLlama-1.1B-Chat-v1.0"
pip install openvino-genai
```

LLMPipeline is the main object used for decoding. You can initiliza it straigh away from the folder with the converted model. It will automanically load the main model, tokenizer, detokenizer and default generation configuration.

### Python

A minimalist example:
```python
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline(model_path, "CPU")
print(pipe.generate("The Sun is yellow bacause"))
```

Calling generate with custom generation config parameters, e.g. config for grouped beam search
```python
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline(model_path, "CPU")

res = pipe.generate("The Sun is yellow bacause", max_new_tokens=30, num_groups=3, group_size=5)
print(res)
```

output:
```
'it is made up of carbon atoms. The carbon atoms are arranged in a linear pattern, which gives the yellow color. The arrangement of carbon atoms in'
```

A simples chat in python:
```python
import openvino_genai as ov_genai
pipe = ov_ov_genai.LLMPipeline(model_path)

config = {'num_groups': 3, 'group_size': 5, 'diversity_penalty': 1.1}
pipe.set_generation_cofnig(config)

pipe.start_chat()
while True:
    print('question:')
    prompt = input()
    if prompt == 'Stop!':
        break
    print(pipe(prompt))
pipe.finish_chat()

```

Test to compare with Huggingface outputs

### C++

Minimalistc example
```cpp
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string model_path = argv[1];
    ov::LLMPipeline pipe(model_path, "CPU");
    std::cout << pipe.generate("The Sun is yellow bacause");
}
```

Using Group Beam Search Decoding
```cpp
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string model_path = argv[1];
    ov::LLMPipeline pipe(model_path, "CPU");

    ov::GenerationConfig config = pipe.get_generation_config();
    config.max_new_tokens = 256;
    config.num_groups = 3;
    config.group_size = 5;
    config.diversity_penalty = 1.0f;

    std::cout << pipe.generate("The Sun is yellow bacause", config);
}
```

A simplest chat in C++
``` cpp
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string prompt;

    std::string model_path = argv[1];
    ov::LLMPipeline pipe(model_path, "CPU");

    pipe.start_chat();
    for (size_t i = 0; i < questions.size(); i++) {
        std::cout << "question:\n";
        std::getline(std::cin, prompt);

        std::cout << pipe(prompt) << std::endl>>;
    }
    pipe.finish_chat();
}
```

Specifying generation_config to use grouped beam search
``` cpp
int main(int argc, char* argv[]) {
    std::string prompt;

    std::string model_path = argv[1];
    ov::LLMPipeline pipe(model_path, "CPU");
    
    ov::GenerationConfig config = pipe.get_generation_config();
    config.max_new_tokens = 256;
    config.num_groups = 3;
    config.group_size = 5;
    config.diversity_penalty = 1.0f;
    
    auto streamer = [](std::string word) { std::cout << word << std::flush; };

    pipe.start_chat();
    for (size_t i = 0; i < questions.size(); i++) {
        
        std::cout << "question:\n";
        cout << prompt << endl;

        auto answer = pipe(prompt, config, streamer);
        // no need to print answer, streamer will do that
    }
    pipe.finish_chat();
}
```

Streaming exapmle with lambda function

``` cpp

#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string model_path = argv[1];
    ov::LLMPipeline pipe(model_path, "CPU");
        
    auto streamer = [](std::string word) { std::cout << word << std::flush; };
    std::cout << pipe.generate("The Sun is yellow bacause", streamer);
}
```

Streaming with custom class
``` cpp
#include <streamer_base.hpp>
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

class CustomStreamer: publict StreamerBase {
public:
    void put(int64_t token) {
        /* custom decoding/tokens processing code
        tokens_cache.push_back(token);
        std::string text = m_tokenizer.decode(tokens_cache);
        ...
        */
    };

    void end() {
        /* custom finalization */
    };
};

int main(int argc, char* argv[]) {
    CustomStreamer custom_streamer;

    std::string model_path = argv[1];
    ov::LLMPipeline pipe(model_path, "CPU");
    cout << pipe.generate("The Sun is yellow bacause", custom_streamer);
}
```
