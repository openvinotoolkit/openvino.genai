# OpenVINO™ GenAI

OpenVINO™ GenAI is a library of the most popular Generative AI model pipelines, optimized execution methods, and samples that run on top of highly performant [OpenVINO Runtime](https://github.com/openvinotoolkit/openvino).

This library is friendly to PC and laptop execution, and optimized for resource consumption. It requires no external dependencies to run generative models as it already includes all the core functionality (e.g. tokenization via openvino-tokenizers).

![Text generation using LLaMa 3.2 model running on Intel ARC770 dGPU](./samples/generation.gif)

## Getting Started

Please follow the following blogs to setup your first hands-on experience with C++ and Python samples.

* [How to Build OpenVINO™ GenAI APP in C++](https://medium.com/openvino-toolkit/how-to-build-openvino-genai-app-in-c-32dcbe42fa67)
* [How to run Llama 3.2 locally with OpenVINO™](https://medium.com/openvino-toolkit/how-to-run-llama-3-2-locally-with-openvino-60a0f3674549)


## Supported Generative AI scenarios

OpenVINO™ GenAI library provides very lightweight C++ and Python APIs to run following Generative Scenarios:
 - Text generation using Large Language Models. For example, chat with local LLaMa model
 - Image generation using Diffuser models, for example, generation using Stable Diffusion models
 - Speech recognition using Whisper family models
 - Text generation using Large Visual Models, for instance, Image analysis using LLaVa or miniCPM models family

Library efficiently supports LoRA adapters for Text and Image generation scenarios:
- Load multiple adapters per model
- Select active adapters for every generation
- Mix multiple adapters with coefficients via alpha blending

All scenarios are run on top of OpenVINO Runtime that supports inference on CPU, GPU and NPU. See [here](https://docs.openvino.ai/2024/about-openvino/release-notes-openvino/system-requirements.html) for platform support matrix.

## Supported Generative AI optimization methods

OpenVINO™ GenAI library provides a transparent way to use state-of-the-art generation optimizations:
- Speculative decoding that employs two models of different sizes and uses the large model to periodically correct the results of the small model. See [here](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/) for more detailed overview
- KVCache token eviction algorithm that reduces the size of the KVCache by pruning less impacting tokens.

Additionally, OpenVINO™ GenAI library implements a continuous batching approach to use OpenVINO within LLM serving. Continuous batching library could be used in LLM serving frameworks and supports the following features:
- Prefix caching that caches fragments of previous generation requests and corresponding KVCache entries internally and uses them in case of repeated query. See [here](https://google.com) for more detailed overview

Continuous batching functionality is used within OpenVINO Model Server (OVMS) to serve LLMs, see [here](https://docs.openvino.ai/2024/ovms_docs_llm_reference.html) for more details.

## Installing OpenVINO GenAI

```sh
    # Installing OpenVINO GenAI via pip
    pip install openvino-genai

    # Install optimum-intel to be able to download, convert and optimize LLMs from Hugging Face
    # Optimum is not required to run models, only to convert and compress
    pip install optimum-intel@git+https://github.com/huggingface/optimum-intel.git

    # (Optional) Install (TBD) to be able to download models from Model Scope
```

## Performing text generation 
<details>

For more examples check out our [LLM Inference Guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html)

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

### Run generation using LLMPipeline in C++

Code below requires installation of C++ compatible package (see [here](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-genai.html#archive-installation) for more details)

```cpp
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string model_path = argv[1];
    ov::genai::LLMPipeline pipe(model_path, "CPU");
    std::cout << pipe.generate("The Sun is yellow because", ov::genai::max_new_tokens(100)) << '\n';
}
```

### Sample notebooks using this API

See [here](https://openvinotoolkit.github.io/openvino_notebooks/?search=Create+an+LLM-powered+Chatbot+using+OpenVINO+Generate+API)

</details>

## Performing visual language text generation
<details>

For more examples check out our [LLM Inference Guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html)

### Converting and compressing the model from Hugging Face library

```sh
optimum-cli export openvino --model openbmb/MiniCPM-V-2_6 --trust-remote-code MiniCPM-V-2_6
```

### Run generation using VLMPipeline API in Python

```python
import openvino_genai as ov_genai
#Will run model on CPU, GPU
pipe = ov_genai.VLMPipeline("./MiniCPM-V-2_6/", "CPU")
rgb = read_image("cat.jpg")
print(pipe.generate(prompt, image=rgb))
```

### Run generation using VLMPipeline in C++

Code below requires installation of C++ compatible package (see [here](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-genai.html#archive-installation) for more details)

```cpp
#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    std::string model_path = argv[1];
    ov::genai::VLMPipeline pipe(model_path, "CPU");
    ov::Tensor rgb = utils::load_image(argv[2]);
    std::cout << pipe.generate(prompt, ov::genai::image(rgb)) << '\n';
}
```

### Sample notebooks using this API

See [here](https://openvinotoolkit.github.io/openvino_notebooks/?search=Visual-language+assistant+with+MiniCPM-V2+and+OpenVINO)

</details>

## Performing image generation

<details>

For more examples check out our [LLM Inference Guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html)

### Converting and compressing image generation model from Hugging Face library

```sh
#Download and convert to OpenVINO dreamlike-anime-1.0 model
optimum-cli export openvino --model dreamlike-art/dreamlike-anime-1.0 --task stable-diffusion --weight-format fp16 dreamlike_anime_1_0_ov/FP16
```

### Run generation using Text2Image API in Python

```python

#WIP

```

### Run generation using Text2Image API in C++

Code below requires installation of C++ compatible package (see [here](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-genai.html#archive-installation) for additional setup details, or this blog for full instruction [How to Build OpenVINO™ GenAI APP in C++](https://medium.com/openvino-toolkit/how-to-build-openvino-genai-app-in-c-32dcbe42fa67)

```cpp
#include "openvino/genai/text2image/pipeline.hpp"
#include "imwrite.hpp"
int main(int argc, char* argv[]) {

   const std::string models_path = argv[1], prompt = argv[2];
   const std::string device = "CPU";  // GPU, NPU can be used as well

   ov::genai::Text2ImagePipeline pipe(models_path, device);
   ov::Tensor image = pipe.generate(prompt,
        ov::genai::width(512),
        ov::genai::height(512),
        ov::genai::num_inference_steps(20));

   imwrite("image.bmp", image, true);
}
```
### Sample notebooks using this API

(TBD)

</details>

## Speech-to-text processing using Whisper Pipeline
<details>

For more examples check out our [LLM Inference Guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html)

NOTE: Whisper Pipeline requires preprocessing of audio input (to adjust sampling rate and normalize)
 
 ### Converting and compressing image generation model from Hugging Face library
```sh
#Download and convert to OpenVINO whisper-base model
optimum-cli export openvino --trust-remote-code --model openai/whisper-base whisper-base
```

### Run generation using Whisper Pipeline API in Python

NOTE: This sample is a simplified version of the full sample that is available [here](./samples/python/whisper_speech_recognition/whisper_speech_recognition.py)

```python
import argparse
import openvino_genai
import librosa

def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=16000)
    return raw_speech.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("wav_file_path")
    args = parser.parse_args()

    raw_speech = read_wav(args.wav_file_path)

    pipe = openvino_genai.WhisperPipeline(args.model_dir)

    def streamer(word: str) -> bool:
        print(word, end="")
        return False

    pipe.generate(
        raw_speech,
        max_new_tokens=100,
        # 'task' and 'language' parameters are supported for multilingual models only
        language="<|en|>",
        task="transcribe",
        streamer=streamer,
    )

    print()
```

 
### Run generation using Whisper Pipeline API in C++

NOTE: This sample is a simplified version of the full sample that is available [here](./samples/cpp/whisper_speech_recognition/whisper_speech_recognition.cpp)

```cpp
#include "audio_utils.hpp"
#include "openvino/genai/whisper_pipeline.hpp"

int main(int argc, char* argv[]) try {

    std::string model_path = argv[1];
    std::string wav_file_path = argv[2];

    ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);

    ov::genai::WhisperPipeline pipeline{model_path};

    ov::genai::WhisperGenerationConfig config{model_path + "/generation_config.json"};
    config.max_new_tokens = 100;
    // 'task' and 'language' parameters are supported for multilingual models only
    config.language = "<|en|>";
    config.task = "transcribe";

    auto streamer = [](std::string word) {
        std::cout << word;
        return false;
    };

    pipeline.generate(raw_speech, config, streamer);

    std::cout << std::endl;
}
```

 ### Sample notebooks using this API

See [here](https://openvinotoolkit.github.io/openvino_notebooks/?search=Automatic+speech+recognition+using+Whisper+and+OpenVINO+with+Generate+API)

</details>


## Additional materials

- [List of supported models](https://github.com/openvinotoolkit/openvino.genai/blob/master/src/docs/SUPPORTED_MODELS.md) (NOTE: models can work, but were not tried yet)
- [OpenVINO LLM inference Guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html)
- [Optimum-intel and OpenVINO](https://huggingface.co/docs/optimum/intel/openvino/export)

## License

The OpenVINO™ GenAI repository is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release
your contribution under these terms.
