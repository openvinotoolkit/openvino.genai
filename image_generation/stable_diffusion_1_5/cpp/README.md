# OpenVINO Stable Diffusion (with LoRA) C++ image generation pipeline
The pure C++ text-to-image pipeline, driven by the OpenVINO native C++ API for Stable Diffusion v1.5 with LMS Discrete Scheduler, supports both static and dynamic model inference. It includes advanced features like [LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora) integration with [safetensors](https://huggingface.co/docs/safetensors/index#format) and [OpenVINO extension for tokenizers](https://github.com/openvinotoolkit/openvino_contrib/blob/master/modules/custom_operations/user_ie_extensions/tokenizer/python/README.md). Loading `user_ov_extensions` provided by `openvino-tokenizers` to `ov::Core` enables tokenization. The sample uses [diffusers](../../common/diffusers) for image generation and [imwrite](../../common/imwrite) for saving `.bmp` images. This demo has been tested on Windows and Unix platforms. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/225-stable-diffusion-text-to-image/225-stable-diffusion-text-to-image.ipynb) which provides an example of image generation in Python.

> [!NOTE]
>This tutorial assumes that the current working directory is `<openvino.genai repo>/image_generation/stable_diffusion_1_5/cpp/` and all paths are relative to this folder.

## Step 1: Prepare build environment

C++ Packages:
* [CMake](https://cmake.org/download/): Cross-platform build tool
* [OpenVINO](https://docs.openvino.ai/install): Model inference
* [Eigen3](https://anaconda.org/conda-forge/eigen): LoRA enabling

Prepare a python environment and install dependencies:
```shell
conda create -n openvino_sd_cpp python==3.10
conda activate openvino_sd_cpp
conda install openvino eigen c-compiler cxx-compiler make
```

## Step 2: Convert Stable Diffusion v1.5 and Tokenizer models

### Stable Diffusion v1.5 model:

1. Install dependencies to import models from HuggingFace:
```shell
conda activate openvino_sd_cpp
python -m pip install -r scripts/requirements.txt
python -m pip install ../../../thirdparty/openvino_contrib/modules/custom_operations/[transformers]
```
2. Download a huggingface SD v1.5 model like:
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [dreamlike-anime-1.0](https://huggingface.co/dreamlike-art/dreamlike-anime-1.0) to run Stable Diffusion with LoRA adapters.


Example command:
```shell
huggingface-cli download --resume-download --local-dir-use-symlinks False dreamlike-art/dreamlike-anime-1.0 --local-dir models/dreamlike-anime-1.0
```

Please, refer to the official website for [model downloading](https://huggingface.co/docs/hub/models-downloading) to read more details.

3. Run model conversion script to convert PyTorch model to OpenVINO IR via [optimum-intel](https://github.com/huggingface/optimum-intel). Please, use the script `scripts/convert_model.py` to convert the model into `FP16_static` or `FP16_dyn`, which will be saved into the `models` folder:
```shell
cd scripts
python convert_model.py -b 1 -t FP16 -sd ../models/dreamlike-anime-1.0 # to convert to models with static shapes
python convert_model.py -b 1 -t FP16 -sd ../models/dreamlike-anime-1.0 -dyn True # to keep models with dynamic shapes
python convert_model.py -b 1 -t INT8 -sd ../models/dreamlike-anime-1.0 -dyn True # to compress the models to INT8
```

> [!NOTE]
>Now the pipeline support batch size = 1 only, i.e. static model `(1, 3, 512, 512)`

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
./stable_diffusion [-p <posPrompt>] [-n <negPrompt>] [-s <seed>] [--height <output image>] [--width <output image>] [-d <device>] [-r <readNPLatent>] [-l <lora.safetensors>] [-a <alpha>] [-h <help>] [-m <modelPath>] [-t <modelType>]

Usage:
  stable_diffusion [OPTION...]
```

* `-p, --posPrompt arg` Initial positive prompt for SD  (default: cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting)
* `-n, --negPrompt arg` Default is empty with space (default: )
* `-d, --device arg`    AUTO, CPU, or GPU (default: CPU)
* `--step arg`          Number of diffusion step ( default: 20)
* `-s, --seed arg`      Number of random seed to generate latent (default: 42)
* `--num arg`           Number of image output(default: 1)
* `--height arg`        Height of output image (default: 512)
* `--width arg`         Width of output image (default: 512)
* `-c, --useCache`      Use model caching
* `-r, --readNPLatent`  Read numpy generated latents from file
* `-m, --modelPath arg` Specify path of SD model IR (default: ../models/dreamlike-anime-1.0)
* `-t, --type arg`      Specify the type of SD model IR (FP16_static or FP16_dyn) (default: FP16_static)
* `-l, --loraPath arg`  Specify path of lora file. (*.safetensors). (default: )
* `-a, --alpha arg`     alpha for lora (default: 0.75)
* `-h, --help`          Print usage

#### Examples

Positive prompt: cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting

Negative prompt: (empty, here couldn't use OV tokenizer, check the issues for details)

Read the numpy latent instead of C++ std lib for the alignment with Python pipeline

* Generate image without lora `./stable_diffusion -r`

   ![](./without_lora.bmp)

* Generate image with soulcard lora `./stable_diffusion -r`

   ![](./soulcard_lora.bmp)

* Generate different size image with dynamic model (C++ lib generated latent): `./stable_diffusion -m ../models/dreamlike-anime-1.0 -t FP16_dyn --height 448 --width 704`

   ![](./704x448.bmp)

## Notes:

For the generation quality, be careful with the negative prompt and random latent generation. C++ random generation with MT19937 results is differ from `numpy.random.randn()`. Hence, please use `-r, --readNPLatent` for the alignment with Python (this latent file is for output image 512X512 only)
