# OpenVINO™ GenAI Library

OpenVINO™ GenAI is a flavor of OpenVINO™, aiming to simplify running inference of generative AI models.
It hides the complexity of the generation process and minimizes the amount of code required.

## Install OpenVINO™ GenAI

> **NOTE**: Please make sure that you are following the versions compatibility rules, refer to the [OpenVINO™ GenAI Dependencies](#openvino-genai-dependencies) for more information.

The OpenVINO™ GenAI flavor is available for installation via Archive and PyPI distributions.
To install OpenVINO™ GenAI, refer to the [Install Guide](https://docs.openvino.ai/2024/get-started/install-openvino.html).

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
        // false means continue generation.
        return false;
    };
    std::cout << pipe.generate("The Sun is yellow because", ov::genai::streamer(streamer), ov::genai::max_new_tokens(200));
}
```

Streaming with a custom class:

C++ template for a stremer.
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

    def put(self, token_id) -> bool:
        # Custom decoding/tokens processing logic.

        # Returns a flag whether generation should be stopped, if true generation stops.
        return False

    def end(self):
        # Custom finalization logic.

pipe = ov_genai.LLMPipeline(models_path, "CPU")
custom_streamer = CustomStreamer()

pipe.generate("The Sun is yellow because", max_new_tokens=15, streamer=custom_streamer)
```
For fully implemented iterable CustomStreamer please refer to [multinomial_causal_lm](https://github.com/openvinotoolkit/openvino.genai/tree/releases/2024/3/samples/python/multinomial_causal_lm/README.md) sample.


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

`openvino_genai.PerfMetrics` (referred as `PerfMetrics` for simplicity) is a structure that holds performance metrics for each generate call. `PerfMetrics` holds fields with mean and standard deviations for the following metrics:
- Time To the First Token (TTFT), ms
- Time per Output Token (TPOT), ms/token
- Generate total duration, ms
- Tokenization duration, ms
- Detokenization duration, ms
- Throughput, tokens/s

and:
- Load time, ms
- Number of generated tokens
- Number of tokens in the input prompt

Performance metrics are stored either in the `DecodedResults` or `EncodedResults` `perf_metric` field. Additionally to the fields mentioned above, `PerfMetrics` has a member `raw_metrics` of type `openvino_genai.RawPerfMetrics` (referred to as `RawPerfMetrics` for simplicity) that contains raw values for the durations of each batch of new token generation, tokenization durations, detokenization durations, and more. These raw metrics are accessible if you wish to calculate your own statistical values such as median or percentiles. However, since mean and standard deviation values are usually sufficient, we will focus on `PerfMetrics`.

```python
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline(models_path, "CPU")
result = pipe.generate(["The Sun is yellow because"], max_new_tokens=20)
perf_metrics = result.perf_metrics

print(f'Generate duration: {perf_metrics.get_generate_duration().mean:.2f}')
print(f'TTFT: {perf_metrics.get_ttft().mean:.2f} ms')
print(f'TPOT: {perf_metrics.get_tpot().mean:.2f} ms/token')
print(f'Throughput: {perf_metrics.get_throughput().mean:.2f} tokens/s')
```

```cpp
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string models_path = argv[1];
    ov::genai::LLMPipeline pipe(models_path, "CPU");
    auto result = pipe.generate("The Sun is yellow because", ov::genai::max_new_tokens(20));
    auto perf_metrics = result.perf_metrics;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Generate duration: " << perf_metrics.get_generate_duration().mean << " ms" << std::endl;
    std::cout << "TTFT: " << metrics.get_ttft().mean  << " ms" << std::endl;
    std::cout << "TPOT: " << metrics.get_tpot().mean  << " ms/token " << std::endl;
    std::cout << "Throughput: " << metrics.get_throughput().mean  << " tokens/s" << std::endl;
}
```
output:
```sh
mean_generate_duration: 76.28
mean_ttft: 42.58
mean_tpot 3.80
```

>**Note**: If the input prompt is just a string, the generate function returns only a string without perf_metrics. To obtain perf_metrics, provide the prompt as a list with at least one element or call generate with encoded inputs.

#### Accumulating metrics
Several `perf_metrics` can be added to each other. In that case `raw_metrics` are concatenated and mean/std values are recalculated. This accumulates statistics from several `generate()` calls

```cpp
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string models_path = argv[1];
    ov::genai::LLMPipeline pipe(models_path, "CPU");
    auto result_1 = pipe.generate("The Sun is yellow because", ov::genai::max_new_tokens(20));
    auto result_2 = pipe.generate("The Sun is yellow because", ov::genai::max_new_tokens(20));
    auto perf_metrics = result_1.perf_metrics + result_2.perf_metrics

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Generate duration: " << perf_metrics.get_generate_duration().mean << " ms" << std::endl;
    std::cout << "TTFT: " << metrics.get_ttft().mean  << " ms" << std::endl;
    std::cout << "TPOT: " << metrics.get_tpot().mean  << " ms/token " << std::endl;
    std::cout << "Throughput: " << metrics.get_throughput().mean  << " tokens/s" << std::endl;
}
```

```python
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline(models_path, "CPU")
res_1 = pipe.generate(["The Sun is yellow because"], max_new_tokens=20)
res_2 = pipe.generate(["Why Sky is blue because"], max_new_tokens=20)
perf_metrics = res_1.perf_metrics + res_2.perf_metrics

print(f'Generate duration: {perf_metrics.get_generate_duration().mean:.2f}')
print(f'TTFT: {perf_metrics.get_ttft().mean:.2f} ms')
print(f'TPOT: {perf_metrics.get_tpot().mean:.2f} ms/token')
print(f'Throughput: {perf_metrics.get_throughput().mean:.2f} tokens/s')
```

#### Using raw performance metrics
In addition to mean and standard deviation values, the `perf_metrics` object has a `raw_metrics` field. This field stores raw data, including:

- Timestamps for each batch of generated tokens
- Batch sizes for each timestamp
- Tokenization durations
- Detokenization durations
- Other relevant metrics

These metrics can be use for more fine grained analysis, such as getting exact calculating median values, percentiles, etc. Below are a few examples of how to use raw metrics.

Getting timestamps for each generated token:
```python
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline(models_path, "CPU")
result = pipe.generate(["The Sun is yellow because"], max_new_tokens=20)
perf_metrics = result.perf_metrics
raw_metrics = perf_metrics.raw_metrics

print(f'Generate duration: {perf_metrics.get_generate_duration().mean:.2f}')
print(f'Throughput: {perf_metrics.get_throughput().mean:.2f} tokens/s')
print(f'Timestamps: {" ms, ".join(f"{i:.2f}" for i in raw_metrics.m_new_token_times)}')
```

Getting pure inference time without tokenizatin and detokenization duration:
```python
import openvino_genai as ov_genai
import numpy as np
pipe = ov_genai.LLMPipeline(models_path, "CPU")
result = pipe.generate(["The Sun is yellow because"], max_new_tokens=20)
perf_metrics = result.perf_metrics
print(f'Generate duration: {perf_metrics.get_generate_duration().mean:.2f} ms')

raw_metrics = perf_metrics.raw_metrics
generate_duration = np.array(raw_metrics.generate_durations)
tok_detok_duration = np.array(raw_metrics.tokenization_durations) - np.array(raw_metrics.detokenization_durations)
pure_inference_duration = np.sum(generate_duration - tok_detok_duration) / 1000 # in milliseconds
print(f'Pure Inference duration: {pure_inference_duration:.2f} ms')
```

Example of using raw metrics to calculate median value of generate duration:
```python
import openvino_genai as ov_genai
import numpy as np
pipe = ov_genai.LLMPipeline(models_path, "CPU")
result = pipe.generate(["The Sun is yellow because"], max_new_tokens=20)
perf_metrics = result.perf_metrics
raw_metrics = perf_metrics.raw_metrics

print(f'Generate duration: {perf_metrics.get_generate_duration().mean:.2f}')
print(f'Throughput: {perf_metrics.get_throughput().mean:.2f} tokens/s')
durations = np.array(raw_metrics.m_new_token_times[1:]) - np.array(raw_metrics.m_new_token_times[:-1])
print(f'Median from token to token duration: {np.median(durations):.2f} ms')
```

For more examples of how metrics are used, please refer to the Python [benchmark_genai.py](../samples/python/benchmark_genai/README.md) and C++ [benchmark_genai](../samples/cpp/benchmark_genai/README.md) samples.

## How It Works

For information on how OpenVINO™ GenAI works, refer to the [How It Works Section](./docs/HOW_IT_WORKS.md).

## Supported Models

For a list of supported models, refer to the [Supported Models Section](./docs/SUPPORTED_MODELS.md).
