# OpenVINO™ GenAI

![](src/docs/openvino_genai.svg)

OpenVINO™ GenAI is a library of the most popular Generative AI model pipelines, optimized execution methods, and samples that run on top of highly performant [OpenVINO Runtime](https://github.com/openvinotoolkit/openvino).

This library is friendly to PC and laptop execution, and optimized for resource consumption. It requires no external dependencies to run generative models as it already includes all the core functionality (e.g. tokenization via openvino-tokenizers).

![Text generation using LLaMa 3.2 model running on Intel ARC770 dGPU](./samples/generation.gif)

## Getting Started

* [Introduction to OpenVINO™ GenAI](https://openvinotoolkit.github.io/openvino.genai/docs/getting-started/introduction)
* [Install OpenVINO™ GenAI](https://openvinotoolkit.github.io/openvino.genai/docs/getting-started/installation)
* [Build OpenVINO™ GenAI](./src/docs/BUILD.md)

Please follow the following blogs to setup your first hands-on experience with C++ and Python samples.

* [How to Build OpenVINO™ GenAI APP in C++](https://medium.com/openvino-toolkit/how-to-build-openvino-genai-app-in-c-32dcbe42fa67)
* [How to run Llama 3.2 locally with OpenVINO™](https://medium.com/openvino-toolkit/how-to-run-llama-3-2-locally-with-openvino-60a0f3674549)


## Supported Generative AI scenarios

OpenVINO™ GenAI library provides very lightweight C++ and Python APIs to run following Generative Scenarios:
 - Text generation using Large Language Models. For example, chat with local LLaMa model
 - Image generation using Diffuser models, for example, generation using Stable Diffusion models
 - Speech recognition using Whisper family models
 - Text generation using Large Visual Models, for instance, Image analysis using LLaVa or miniCPM models family
 - Text-to-speech generation using SpeechT5 TTS models
 - Text embedding for Retrieval-Augmented Generation (RAG). For example, compute embeddings for documents and queries to enable efficient retrieval in RAG workflows.

Library efficiently supports LoRA adapters for Text and Image generation scenarios:
- Load multiple adapters per model
- Select active adapters for every generation
- Mix multiple adapters with coefficients via alpha blending

All scenarios are run on top of OpenVINO Runtime that supports inference on CPU, GPU and NPU. See [here](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html) for platform support matrix.

## Supported Generative AI optimization methods

OpenVINO™ GenAI library provides a transparent way to use state-of-the-art generation optimizations:
- Speculative decoding that employs two models of different sizes and uses the large model to periodically correct the results of the small model. See [here](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/) for more detailed overview
- KVCache token eviction algorithm that reduces the size of the KVCache by pruning less impacting tokens.

Additionally, OpenVINO™ GenAI library implements a continuous batching approach to use OpenVINO within LLM serving. Continuous batching library could be used in LLM serving frameworks and supports the following features:
- Prefix caching that caches fragments of previous generation requests and corresponding KVCache entries internally and uses them in case of repeated query. See [here](https://google.com) for more detailed overview

Continuous batching functionality is used within OpenVINO Model Server (OVMS) to serve LLMs, see [here](https://docs.openvino.ai/2025/openvino-workflow/model-server/ovms_what_is_openvino_model_server.html) for more details.

## Installing OpenVINO GenAI

```sh
    # Installing OpenVINO GenAI via pip
    # export-requirements are not required to run models, only to convert and compress
    pip install openvino-genai --requirement ./samples/export-requirements.txt --requirement ./samples/deployment-requirements.txt

    # (Optional) Install (TBD) to be able to download models from Model Scope
```

## Performing text generation 
<details>

For more examples check out our [Generative AI workflow](https://docs.openvino.ai/2025/openvino-workflow-generative.html)

### Converting and compressing text generation model from Hugging Face library

```sh
#(Basic) download and convert to OpenVINO TinyLlama-Chat-v1.0 model
optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --weight-format fp16 --trust-remote-code "TinyLlama-1.1B-Chat-v1.0"

#(Recommended) download, convert to OpenVINO and compress to int4 TinyLlama-Chat-v1.0 model
optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --weight-format int4 --trust-remote-code "TinyLlama-1.1B-Chat-v1.0"
```

### Run generation using LLMPipeline API in Python

```python
import openvino_genai
#Will run model on CPU, GPU or NPU are possible options
pipe = openvino_genai.LLMPipeline("./TinyLlama-1.1B-Chat-v1.0/", "CPU")
print(pipe.generate("The Sun is yellow because", max_new_tokens=100))
```

### Run generation using LLMPipeline API in JavaScript

```js
import { LLMPipeline } from 'openvino-genai-node';

main();

async function main() {
    const pipe = await LLMPipeline("./TinyLlama-1.1B-Chat-v1.0/", "CPU");
    const result = await pipe.generate("The Sun is yellow because", { 'max_new_tokens': 100 });
    console.log(result);
}
```

### Run generation using LLMPipeline in C++

Code below requires installation of C++ compatible package (see [here](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-genai.html#archive-installation) for more details)

```cpp
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string models_path = argv[1];
    ov::genai::LLMPipeline pipe(models_path, "CPU");
    std::cout << pipe.generate("The Sun is yellow because", ov::genai::max_new_tokens(100)) << '\n';
}
```

### Sample notebooks using this API

See [here](https://openvinotoolkit.github.io/openvino_notebooks/?search=Create+an+LLM-powered+Chatbot+using+OpenVINO+Generate+API)

</details>

## Performing visual language text generation
<details>

For more examples check out our [Generative AI workflow](https://docs.openvino.ai/2025/openvino-workflow-generative.html)

### Converting and compressing the model from Hugging Face library

To convert the [OpenGVLab/InternVL2-1B](https://huggingface.co/OpenGVLab/InternVL2-1B) model, `timm` and `einops` are required: `pip install timm einops`.

```sh
# Download and convert the OpenGVLab/InternVL2-1B model to OpenVINO with int4 weight-compression for the language model
# Other components are compressed to int8
optimum-cli export openvino -m OpenGVLab/InternVL2-1B --trust-remote-code --weight-format int4 InternVL2-1B
```

### Run generation using VLMPipeline API in Python

See [Visual Language Chat](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/visual_language_chat) for a demo application.

Run the following command to download a sample image:

```sh
curl -O "https://storage.openvinotoolkit.org/test_data/images/dog.jpg"
```

```python
import numpy as np
import openvino as ov
import openvino_genai
from PIL import Image

# Choose GPU instead of CPU in the line below to run the model on Intel integrated or discrete GPU
pipe = openvino_genai.VLMPipeline("./InternVL2-1B", "CPU")

image = Image.open("dog.jpg")
image_data = np.array(image)
image_data = ov.Tensor(image_data)

prompt = "Can you describe the image?"
result = pipe.generate(prompt, image=image_data, max_new_tokens=100)
print(result.texts[0])
```

### Run generation using VLMPipeline in C++

Code below requires installation of C++ compatible package (see [here](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-genai.html#archive-installation) for more details). See [Visual Language Chat](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/cpp/visual_language_chat) for a demo application.

```cpp
#include "openvino/genai/visual_language/pipeline.hpp"
#include "load_image.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string models_path = argv[1];
    ov::genai::VLMPipeline pipe(models_path, "CPU");
    ov::Tensor rgb = utils::load_image(argv[2]);
    std::cout << pipe.generate(
        prompt,
        ov::genai::image(rgb),
        ov::genai::max_new_tokens(100)
    ) << '\n';
}
```

### Sample notebooks using this API

See [here](https://openvinotoolkit.github.io/openvino_notebooks/?search=Visual-language+assistant+with+MiniCPM-V2+and+OpenVINO)

</details>

## Performing image generation

<details>

For more examples check out our [Generative AI workflow](https://docs.openvino.ai/2025/openvino-workflow-generative.html)

### Converting and compressing image generation model from Hugging Face library

```sh
#Download and convert to OpenVINO dreamlike-anime-1.0 model
optimum-cli export openvino --model dreamlike-art/dreamlike-anime-1.0 --weight-format fp16 dreamlike_anime_1_0_ov/FP16

#You can also use INT8 hybrid quantization to further optimize the model and reduce inference latency
optimum-cli export openvino --model dreamlike-art/dreamlike-anime-1.0 --weight-format int8 --dataset conceptual_captions dreamlike_anime_1_0_ov/INT8
```

### Run generation using Text2Image API in Python

```python
import argparse
from PIL import Image
import openvino_genai

device = 'CPU'  # GPU can be used as well
pipe = openvino_genai.Text2ImagePipeline("./dreamlike_anime_1_0_ov/INT8", device)
image_tensor = pipe.generate("cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting")

image = Image.fromarray(image_tensor.data[0])
image.save("image.bmp")
```

### Run generation using Text2Image API in C++

Code below requires installation of C++ compatible package (see [here](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-genai.html#archive-installation) for additional setup details, or this blog for full instruction [How to Build OpenVINO™ GenAI APP in C++](https://medium.com/openvino-toolkit/how-to-build-openvino-genai-app-in-c-32dcbe42fa67)

```cpp
#include "openvino/genai/image_generation/text2image_pipeline.hpp"
#include "imwrite.hpp"

int main(int argc, char* argv[]) {
   const std::string models_path = argv[1], prompt = argv[2];
   const std::string device = "CPU";  // GPU can be used as well

   ov::genai::Text2ImagePipeline pipe(models_path, device);
   ov::Tensor image = pipe.generate(prompt);

   imwrite("image.bmp", image, true);
}
```

### Run generation using Image2Image API in Python

```python
import argparse
from PIL import Image
import openvino_genai
import openvino as ov

device = 'CPU'  # GPU can be used as well
pipe = openvino_genai.Image2ImagePipeline("./dreamlike_anime_1_0_ov/INT8", device)

image = Image.open("small_city.jpg")
image_data = np.array(image)[None]
image_data = ov.Tensor(image_data)

image_tensor = pipe.generate(
    "cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting",
    image=image_data,
    strength=0.8
)

image = Image.fromarray(image_tensor.data[0])
image.save("image.bmp")
```

### Run generation using Image2Image API in C++

Code below requires installation of C++ compatible package (see [here](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-genai.html#archive-installation) for additional setup details, or this blog for full instruction [How to Build OpenVINO™ GenAI APP in C++](https://medium.com/openvino-toolkit/how-to-build-openvino-genai-app-in-c-32dcbe42fa67)

```cpp
#include "openvino/genai/image_generation/image2image_pipeline.hpp"
#include "load_image.hpp"
#include "imwrite.hpp"

int main(int argc, char* argv[]) {
   const std::string models_path = argv[1], prompt = argv[2], image_path = argv[3];
   const std::string device = "CPU";  // GPU can be used as well

   ov::Tensor image = utils::load_image(image_path);

   ov::genai::Image2ImagePipeline pipe(models_path, device);
   ov::Tensor generated_image = pipe.generate(prompt, image, ov::genai::strength(0.8f));

   imwrite("image.bmp", generated_image, true);
}
```

### Run generation using Inpainting API in Python

```python
import argparse
from PIL import Image
import openvino_genai
import openvino as ov

def read_image(path: str) -> openvino.Tensor:
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic)[None]
    return openvino.Tensor(image_data)

device = 'CPU'  # GPU can be used as well
pipe = openvino_genai.InpaintingPipeline(args.model_dir, device)

image = read_image("image.jpg")
mask_image = read_image("mask.jpg")

image_tensor = pipe.generate(
    "Face of a yellow cat, high resolution, sitting on a park bench",
    image=image,
    mask_image=mask_image
)

image = Image.fromarray(image_tensor.data[0])
image.save("image.bmp")
```

### Run generation using Inpainting API in C++

Code below requires installation of C++ compatible package (see [here](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-genai.html#archive-installation) for additional setup details, or this blog for full instruction [How to Build OpenVINO™ GenAI APP in C++](https://medium.com/openvino-toolkit/how-to-build-openvino-genai-app-in-c-32dcbe42fa67)

```cpp
#include "openvino/genai/image_generation/inpainting_pipeline.hpp"
#include "load_image.hpp"
#include "imwrite.hpp"

int main(int argc, char* argv[]) {
   const std::string models_path = argv[1], prompt = argv[2];
   const std::string device = "CPU";  // GPU can be used as well

   ov::Tensor image = utils::load_image(argv[3]);
   ov::Tensor mask_image = utils::load_image(argv[4]);

   ov::genai::InpaintingPipeline pipe(models_path, device);
   ov::Tensor generated_image = pipe.generate(prompt, image, mask_image);

   imwrite("image.bmp", generated_image, true);
}
```

### Sample notebooks using this API

See [here](https://openvinotoolkit.github.io/openvino_notebooks/?search=Text+to+Image+pipeline+and+OpenVINO+with+Generate+API)

</details>

## Speech-to-text processing using Whisper Pipeline
<details>

For more examples check out our [Generative AI workflow](https://docs.openvino.ai/2025/openvino-workflow-generative.html)

NOTE: Whisper Pipeline requires preprocessing of audio input (to adjust sampling rate and normalize)
 
 ### Converting and quantizing speech-to-text model from Hugging Face library
```sh
#Download and convert to OpenVINO whisper-base model
optimum-cli export openvino --model openai/whisper-base whisper-base

#Download, convert and apply int8 static quantization to whisper-base model
optimum-cli export openvino --model openai/whisper-base --disable-stateful \
--quant-mode int8 --dataset librispeech --num-samples 32 whisper-base-int8
```

### Run generation using Whisper Pipeline API in Python

NOTE: This sample is a simplified version of the full sample that is available [here](./samples/python/whisper_speech_recognition/whisper_speech_recognition.py)

```python
import openvino_genai
import librosa

def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=16000)
    return raw_speech.tolist()

device = "CPU" # GPU can be used as well
pipe = openvino_genai.WhisperPipeline("whisper-base", device)
raw_speech = read_wav("sample.wav")
print(pipe.generate(raw_speech))
```

 
### Run generation using Whisper Pipeline API in C++

NOTE: This sample is a simplified version of the full sample that is available [here](./samples/cpp/whisper_speech_recognition/whisper_speech_recognition.cpp)

```cpp
#include <iostream>

#include "audio_utils.hpp"
#include "openvino/genai/whisper_pipeline.hpp"

int main(int argc, char* argv[]) {
    std::filesystem::path models_path = argv[1];
    std::string wav_file_path = argv[2];
    std::string device = "CPU"; // GPU can be used as well

    ov::genai::WhisperPipeline pipeline(models_path, device);

    ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);

    std::cout << pipeline.generate(raw_speech, ov::genai::max_new_tokens(100)) << '\n';
}
```

 ### Sample notebooks using this API

See [here](https://openvinotoolkit.github.io/openvino_notebooks/?search=Automatic+speech+recognition+using+Whisper+and+OpenVINO+with+Generate+API)

</details>

## Performing text-to-speech generation
<details>

For more examples check out our [Generative AI workflow](https://docs.openvino.ai/2025/openvino-workflow-generative.html)

NOTE: Currently, text-to-speech in OpenVINO GenAI supports the SpeechT5 TTS model. The generated audio signal is a single-channel (mono) waveform with a sampling rate of 16 kHz.
 
### Converting text-to-speech model from Hugging Face library
```sh
# Download and convert to OpenVINO
optimum-cli export openvino --model microsoft/speecht5_tts --model-kwargs "{\"vocoder\": \"microsoft/speecht5_hifigan\"}" ov_speecht5_tts
```

### Run generation using Text-to-speech API in Python

NOTE: This sample is a simplified version of the full sample that is available [here](./samples/python/speech_generation/text2speech.py)

```python
import openvino_genai
import soundfile as sf

pipe = openvino_genai.Text2SpeechPipeline("ov_speecht5_tts", "CPU")

# additionally, a speaker embedding can be specified as the target voice input to the generate method
result = pipe.generate("Hello OpenVINO GenAI")
speech = result.speeches[0]
sf.write("output_audio.wav", speech.data[0], samplerate=16000)
```

 
### Run generation using Text-to-speech API in C++

NOTE: This sample is a simplified version of the full sample that is available [here](./samples/cpp/speech_generation/text2speech.cpp)

```cpp
#include "audio_utils.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

int main(int argc, char* argv[]) {
    ov::genai::Text2SpeechPipeline pipe("ov_speecht5_tts", "CPU");

    // additionally, a speaker embedding can be specified as the target voice input to the generate method
    auto gen_speech = pipe.generate("Hello OpenVINO GenAI");

    auto waveform_size = gen_speech.speeches[0].get_size();
    auto waveform_ptr = gen_speech.speeches[0].data<const float>();
    auto bits_per_sample = gen_speech.speeches[0].get_element_type().bitwidth();
    utils::audio::save_to_wav(waveform_ptr, waveform_size, "output_audio.wav", bits_per_sample);

    return 0;
}
```

</details>

## Text Embeddings
<details>

### Converting and preparing a text embedding model from Hugging Face library

```sh
# Download and convert the BAAI/bge-small-en-v1.5 model to OpenVINO format
optimum-cli export openvino --trust-remote-code --model BAAI/bge-small-en-v1.5 BAAI/bge-small-en-v1.5
```

### Compute embeddings using TextEmbeddingPipeline API in Python

```python
import openvino_genai

pipeline = openvino_genai.TextEmbeddingPipeline("./BAAI/bge-small-en-v1.5", "CPU")

documents = ["Document 1", "Document 2"]
embeddings = pipeline.embed_documents(documents)

query = "The Sun is yellow because"
query_embedding = pipeline.embed_query(query)
```

### Compute embeddings using TextEmbeddingPipeline API in JavaScript

```js
import { TextEmbeddingPipeline } from 'openvino-genai-node';

main();

async function main() {
    const pipeline = await TextEmbeddingPipeline("./BAAI/bge-small-en-v1.5", "CPU")

    const documents = ["Document 1", "Document 2"];
    const embeddings = await pipeline.embedDocuments(documents);

    const query = "The Sun is yellow because";
    const query_embedding = await pipeline.embedQuery(query);
}

```

### Compute embeddings using TextEmbeddingPipeline API in C++

```cpp
#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string models_path = argv[1];
    std::vector<std::string> documents(argv + 2, argv + argc);
    std::string device = "CPU";  // GPU can be used as well

    ov::genai::TextEmbeddingPipeline pipeline(models_path, device);

    const ov::genai::EmbeddingResults embeddings = pipeline.embed_documents(documents);
}
```
</details>

## Additional materials

- [List of supported models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/)
- [OpenVINO Generative AI workflow](https://docs.openvino.ai/2025/openvino-workflow-generative.html)
- [Optimum-intel and OpenVINO](https://huggingface.co/docs/optimum/intel/openvino/export)

## License

The OpenVINO™ GenAI repository is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release
your contribution under these terms.
