# OpenVINO Generate API

## Usage 

### In C++


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

### In Python
   

``` python
pip install openvino-genai
optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --weight-format fp16 --trust-remote-code "TinyLlama-1.1B-Chat-v1.0"
```


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