# OpenVINO Generate API

## Usage 

Firs of all you need to convert your model with optimum-cli
``` sh
optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --weight-format fp16 --trust-remote-code "TinyLlama-1.1B-Chat-v1.0"
pip install openvino-genai
```

LLMPipeline is the main object used for decoding. You can initiliza it straigh away from the folder with the converted model. It will automanically load the main model, tokenizer, detokenizer and default generation configuration.

### In Python

A minimalist example:
```python
import py_generate_pipeline as genai # set more friendly module name
pipe = genai.LLMPipeline(model_path, "CPU")
print(pipe.generate("The Sun is yellow bacause"))
```

A simples chat in python:
```python
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline(model_path)

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
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

max_new_tokens = 32
prompt = 'table is made of'

encoded_prompt = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)
hf_encoded_output = model.generate(encoded_prompt, max_new_tokens=max_new_tokens, do_sample=False)
hf_output = tokenizer.decode(hf_encoded_output[0, encoded_prompt.shape[1]:])
print(f'hf_output: {hf_output}')

import sys
sys.path.append('build-Debug/')
import py_generate_pipeline as genai # set more friendly module name

pipe = genai.LLMPipeline('text_generation/causal_lm/TinyLlama-1.1B-Chat-v1.0/pytorch/dldt/FP16/')
ov_output = pipe(prompt, max_new_tokens=max_new_tokens)
print(f'ov_output: {ov_output}')

assert hf_output == ov_output

```

### In C++

Minimalistc example
```cpp
int main(int argc, char* argv[]) {
    std::string model_path = argv[1];
    ov::LLMPipeline pipe(model_path, "CPU");
    cout << pipe.generate("The Sun is yellow bacause");
}
```

Using Group Beam Search Decoding
```cpp
int main(int argc, char* argv[]) {
    std::string model_path = argv[1];
    ov::LLMPipeline pipe(model_path, "CPU");
    ov::GenerationConfig config = pipe.get_generation_config();
    config.max_new_tokens = 256;
    config.num_groups = 3;
    config.group_size = 5;
    config.diversity_penalty = 1.0f;

    cout << pipe.generate("The Sun is yellow bacause", config);
}
```

A simplest chat in C++
``` cpp
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
int main(int argc, char* argv[]) {
    auto streamer = [](std::string word) { std::cout << word << std::flush; };

    std::string model_path = argv[1];
    ov::LLMPipeline pipe(model_path, "CPU");
    cout << pipe.generate("The Sun is yellow bacause", streamer);
}
```

Streaming with custom class
``` cpp
#include <streamer_base.hpp>

class CustomStreamer: publict StreamerBase {
public:
    void put(int64_t token) {/* decode tokens and do process them*/};

    void end() {/* decode tokens and do process them*/};
};

int main(int argc, char* argv[]) {
    CustomStreamer custom_streamer;

    std::string model_path = argv[1];
    ov::LLMPipeline pipe(model_path, "CPU");
    cout << pipe.generate("The Sun is yellow bacause", custom_streamer);
}
```

