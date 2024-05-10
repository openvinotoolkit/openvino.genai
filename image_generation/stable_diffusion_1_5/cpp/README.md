# OpenVINO Stable Diffusion (with LoRA) C++ image generation pipeline
The pure C++ text-to-image pipeline, driven by the OpenVINO native C++ API for Stable Diffusion v1.5 with LMS Discrete Scheduler, supports both static and dynamic model inference. It includes advanced features like [LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora) integration with [safetensors](https://huggingface.co/docs/safetensors/index#format) and [OpenVINO Tokenizers](https://github.com/openvinotoolkit/openvino_tokenizers). Loading `openvino_tokenizers` to `ov::Core` enables tokenization. The sample uses [diffusers](../../common/diffusers) for image generation and [imwrite](../../common/imwrite) for saving `.bmp` images. This demo has been tested on Windows and Unix platforms. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/225-stable-diffusion-text-to-image/225-stable-diffusion-text-to-image.ipynb) which provides an example of image generation in Python.

> [!NOTE]
>This tutorial assumes that the current working directory is `<openvino.genai repo>/image_generation/stable_diffusion_1_5/cpp/` and all paths are relative to this folder.

## Step 1: Prepare build environment

Prerequisites:
- Conda ([installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html))


C++ Packages:
* [CMake](https://cmake.org/download/): Cross-platform build tool
* [OpenVINO](https://docs.openvino.ai/install): Model inference. `master` and possibly the latest `releases/*` branch correspond to not yet released OpenVINO versions. https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/ can be used for these branches early testing.

Prepare a python environment and install dependencies:
```shell
conda create -n openvino_sd_cpp python==3.10
conda activate openvino_sd_cpp
conda install -c conda-forge openvino=2024.1.0 c-compiler cxx-compiler git make cmake
# Ensure that Conda standard libraries are used
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Step 2: Convert Stable Diffusion v1.5 and Tokenizer models

### Stable Diffusion v1.5 model:

1. Install dependencies to import models from HuggingFace:
```shell
git submodule update --init
# Reactivate Conda environment after installing dependencies and setting env vars
conda activate openvino_sd_cpp
python -m pip install -r requirements.txt
python -m pip install ../../../thirdparty/openvino_tokenizers/[transformers]
```
2. Download a huggingface SD v1.5 model like:
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [dreamlike-anime-1.0](https://huggingface.co/dreamlike-art/dreamlike-anime-1.0) to run Stable Diffusion with LoRA adapters.

   Example command for downloading and exporting FP16 model:

   `optimum-cli export openvino --model dreamlike-art/dreamlike-anime-1.0 --task stable-diffusion --weight-format fp16 models/dreamlike_anime_1_0_ov/FP16`

   You can also choose other precision and export FP32 or INT8 model.

   Please, refer to the official website for [🤗 Optimum](https://huggingface.co/docs/optimum/main/en/index) and [optimum-intel](https://github.com/huggingface/optimum-intel) to read more details.

   If https://huggingface.co/ is down, the script won't be able to download the model.

> [!NOTE]
> Now the pipeline support batch size = 1 only, i.e. static model `(1, 3, 512, 512)`

### LoRA enabling with safetensors

Refer to [python pipeline blog](https://blog.openvino.ai/blog-posts/enable-lora-weights-with-stable-diffusion-controlnet-pipeline).
The safetensor model is loaded via [safetensors.h](https://github.com/hsnyder/safetensors.h). The layer name and weight are modified with `Eigen` library and inserted into the SD models with `ov::pass::MatcherPass` in the file [common/diffusers/src/lora.cpp](https://github.com/openvinotoolkit/openvino.genai/blob/master/image_generation/common/diffusers/src/lora.cpp).

SD model [dreamlike-anime-1.0](https://huggingface.co/dreamlike-art/dreamlike-anime-1.0) and LoRA [soulcard](https://civitai.com/models/67927?modelVersionId=72591) are tested in this pipeline.

Download and put safetensors and model IR into the models folder.

## Step 3: Build the SD application

```shell
conda activate openvino_sd_cpp
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build --parallel
```

## Step 4: Run Pipeline
```shell
./build/stable_diffusion [-p <posPrompt>] [-n <negPrompt>] [-s <seed>] [--height <output image>] [--width <output image>] [-d <device>] [-r <readNPLatent>] [-l <lora.safetensors>] [-a <alpha>] [-h <help>] [-m <modelPath>] [-t <modelType>] [--dynamic]

Usage:
  stable_diffusion [OPTION...]
```

* `-p, --posPrompt arg` Initial positive prompt for SD  (default: cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting)
* `-n, --negPrompt arg` Default is empty with space (default: )
* `-d, --device arg`    AUTO, CPU, or GPU. Doesn't apply to Tokenizer model, OpenVINO Tokenizers can be inferred on a CPU device only (default: CPU)
* `--step arg`          Number of diffusion step ( default: 20)
* `-s, --seed arg`      Number of random seed to generate latent (default: 42)
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

Negative prompt: (empty, here couldn't use OV tokenizer, check the issues for details)

Read the numpy latent instead of C++ std lib for the alignment with Python pipeline

* Generate image without lora `./build/stable_diffusion -r`

   ![](./without_lora.bmp)

* Generate image with soulcard lora `./build/stable_diffusion -r`

   ![](./soulcard_lora.bmp)

* Generate different size image with dynamic model (C++ lib generated latent): `./build/stable_diffusion -m ./models/dreamlike_anime_1_0_ov -t FP16 --dynamic --height 448 --width 704`

   ![](./704x448.bmp)

## Notes:

For the generation quality, be careful with the negative prompt and random latent generation. C++ random generation with MT19937 results is differ from `numpy.random.randn()`. Hence, please use `-r, --readNPLatent` for the alignment with Python (this latent file is for output image 512X512 only)
