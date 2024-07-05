# OpenVINO Stable Diffusion (with LoRA) C++ Image Generation Pipeline

The pure C++ text-to-image pipeline, driven by the OpenVINO native C++ API for Stable Diffusion v1.5 with LMS Discrete Scheduler, supports both static and dynamic model inference. It includes advanced features like [LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora#lora) integration with [safetensors](https://huggingface.co/docs/safetensors/index#format) and [OpenVINO Tokenizers](https://github.com/openvinotoolkit/openvino_tokenizers). Loading `openvino_tokenizers` to `ov::Core` enables tokenization. The sample uses [diffusers](../../common/diffusers) for image generation and [imwrite](../../common/imwrite) for saving `.bmp` images. This demo has been tested on Windows and Unix platforms. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/stable-diffusion-text-to-image) which provides an example of image generation in Python.

> [!NOTE]
>This tutorial assumes that the current working directory is `<openvino.genai repo>/image_generation/stable_diffusion_1_5/cpp/` and all paths are relative to this folder.

## Step 1: Prepare Build Environment

Prerequisites:
- Conda ([installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html))

C++ Packages:
* [CMake](https://cmake.org/download/): Cross-platform build tool
* [OpenVINO](https://docs.openvino.ai/install): Model inference. `master` and possibly the latest `releases/*` branch correspond to not yet released OpenVINO versions. https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/ can be used for these branches early testing.

Prepare a python environment and install dependencies:

```shell
conda create -n openvino_sd_cpp python==3.10
conda activate openvino_sd_cpp
conda install -c conda-forge openvino=2024.2.0 c-compiler cxx-compiler git make cmake
# Ensure that Conda standard libraries are used
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Step 2: Obtain Stable Diffusion Model

1. Install dependencies to import models from HuggingFace:

    ```shell
    git submodule update --init
    # Reactivate Conda environment after installing dependencies and setting env vars
    conda activate openvino_sd_cpp
    python -m pip install -r ../../requirements.txt
    python -m pip install ../../../thirdparty/openvino_tokenizers/[transformers]
    ```

2. Download the model from Huggingface and convert it to OpenVINO IR via [optimum-intel CLI](https://github.com/huggingface/optimum-intel).

    Example models to download:
    - [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
    - [dreamlike-art/dreamlike-anime-1.0](https://huggingface.co/dreamlike-art/dreamlike-anime-1.0)

    Example command for downloading [dreamlike-art/dreamlike-anime-1.0](https://huggingface.co/dreamlike-art/dreamlike-anime-1.0) model and exporting it with FP16 precision:

    `optimum-cli export openvino --model dreamlike-art/dreamlike-anime-1.0 --task stable-diffusion --weight-format fp16 models/dreamlike_anime_1_0_ov/FP16`

    You can also choose other precision and export FP32 or INT8 model.

    Please, refer to the official website for [🤗 Optimum](https://huggingface.co/docs/optimum/main/en/index) and [optimum-intel](https://github.com/huggingface/optimum-intel) to read more details.

    If https://huggingface.co/ is down, the script won't be able to download the model.

> [!NOTE]
> Now the pipeline support batch size = 1 only, i.e. static model `(1, 3, 512, 512)`

### (Optional) Enable LoRA Weights with Safetensors

Low-Rank Adaptation (LoRA) is a technique introduced to deal with the problem of fine-tuning Diffusers and Large Language Models (LLMs). In the case of Stable Diffusion fine-tuning, LoRA can be applied to the cross-attention layers for the image representations with the latent described.

LoRA weights can be enabled for Unet model of Stable Diffusion pipeline to generate images with different styles.

In this sample LoRA weights are used in [safetensors]((https://huggingface.co/docs/safetensors/index#format)) format.
Safetensors is a serialization format developed by Hugging Face that is specifically designed for efficiently storing and loading large tensors. It provides a lightweight and efficient way to serialize tensors, making it easier to store and load machine learning models.

The LoRA safetensors model is loaded via [safetensors.h](https://github.com/hsnyder/safetensors.h). The layer name and weight are modified with `Eigen` library and inserted into the SD models with `ov::pass::MatcherPass` in the file [common/diffusers/src/lora.cpp](https://github.com/openvinotoolkit/openvino.genai/blob/master/image_generation/common/diffusers/src/lora.cpp).

There are various LoRA models on https://civitai.com/tag/lora and on HuggingFace, you can consider to choose your own LoRA model in safetensor format. For example, you can use LoRA [soulcard model](https://civitai.com/models/67927?modelVersionId=72591).
Download and put LoRA safetensors model into the models directory. When running the built sample provide the path to the LoRA model with `-l, --loraPath arg` argument.

## Step 3: Build the SD Application

```shell
conda activate openvino_sd_cpp
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build --parallel
```

## Step 4: Run Pipeline
```shell
./build/stable_diffusion [-p <posPrompt>] [-n <negPrompt>] [-s <seed>] [--height <output image>] [--width <output image>] [-d <device>] [-r <readNPLatent>] [-l <lora.safetensors>] [-a <alpha>] [-h <help>] [-m <modelPath>] [-t <modelType>] [--guidanceScale <guidanceScale>] [--dynamic]

Usage:
  stable_diffusion [OPTION...]
```

* `-p, --posPrompt arg` Initial positive prompt for SD (default: "cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting")
* `-n, --negPrompt arg` The prompt to guide the image generation away from. Ignored when not using guidance (`--guidanceScale` is less than `1`) (default: "")
* `-d, --device arg`    AUTO, CPU, or GPU. Doesn't apply to Tokenizer model, OpenVINO Tokenizers can be inferred on a CPU device only (default: CPU)
* `--step arg`          Number of diffusion step ( default: 20)
* `-s, --seed arg`      Number of random seed to generate latent (default: 42)
* `--guidanceScale arg` A higher guidance scale value encourages the model to generate images closely linked to the text prompt at the expense of lower image quality (default: 7.5)
* `--num arg`           Number of image output(default: 1)
* `--height arg`        Height of output image (default: 512)
* `--width arg`         Width of output image (default: 512)
* `-c, --useCache`      Use model caching
* `-r, --readNPLatent`  Read numpy generated latents from file
* `-m, --modelPath arg` Specify path of SD model IR (default: ../models/dreamlike_anime_1_0_ov)
* `-t, --type arg`      Specify the type of SD model IRs (FP32, FP16 or INT8) (default: FP16)
* `--dynamic`           Specify the model input shape to use dynamic shape
* `-l, --loraPath arg`  Specify path of lora file. (*.safetensors). (default: )
* `-a, --alpha arg`     alpha for lora (default: 0.75)
* `-h, --help`          Print usage

> [!NOTE]
> The tokenizer model will always be loaded to CPU: [OpenVINO Tokenizers](https://github.com/openvinotoolkit/openvino_tokenizers) can be inferred on a CPU device only.

#### Examples

Positive prompt: cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting

Negative prompt: (empty, check the [Notes](#negative-prompt) for details)

To read the numpy latent instead of C++ std lib for the alignment with Python pipeline, use `-r, --readNPLatent` argument.

* Generate image without lora `./build/stable_diffusion -r`

   ![](./without_lora.bmp)

* Generate image with soulcard lora `./build/stable_diffusion -r -l path/to/soulcard.safetensors`

   ![](./soulcard_lora.bmp)

* Generate different size image with dynamic model (C++ lib generated latent): `./build/stable_diffusion -m ./models/dreamlike_anime_1_0_ov -t FP16 --dynamic --height 448 --width 704`

   ![](./704x448.bmp)

## Notes

For the generation quality, be careful with the negative prompt and random latent generation. C++ random generation with MT19937 results differ from `numpy.random.randn()`. Hence, please use `-r, --readNPLatent` for the alignment with Python (this latent file is for output image 512X512 only).

#### Guidance Scale

Guidance scale controls how similar the generated image will be to the prompt. A higher guidance scale means the model will try to generate an image that follows the prompt more strictly. A lower guidance scale means the model will have more creativity.
`guidance_scale` is a way to increase the adherence to the conditional signal that guides the generation (text, in this case) as well as overall sample quality. It is also known as [classifier-free guidance](https://arxiv.org/abs/2207.12598).

#### Negative Prompt

To improve image generation quality, model supports negative prompting. Technically, positive prompt steers the diffusion toward the images associated with it, while negative prompt steers the diffusion away from it. 
In other words, negative prompt declares undesired concepts for generation image, e.g. if we want to have colorful and bright image, gray scale image will be result which we want to avoid, in this case gray scale can be treated as negative prompt.
The positive and negative prompt are in equal footing. You can always use one with or without the other. More explanation of how it works can be found in this [article](https://stable-diffusion-art.com/how-negative-prompt-work/).

> [!NOTE]
> Negative prompting is applicable only for high guidance scale (at least > 1).

#### LoRA Weights Enabling

Refer to the [OpenVINO blog](https://blog.openvino.ai/blog-posts/enable-lora-weights-with-stable-diffusion-controlnet-pipeline) to get more information on enabling LoRA weights.
