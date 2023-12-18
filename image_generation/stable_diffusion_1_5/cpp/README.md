# OpenVINO Stable Diffusion (with LoRA) C++ pipeline
The pure C++ text-to-image pipeline, driven by the OpenVINO native API for Stable Diffusion v1.5 with LMS Discrete Scheduler, supports both static and dynamic model inference. It includes advanced features like LoRA integration with safetensors and [OpenVINO extension for tokenizers](https://github.com/openvinotoolkit/openvino_contrib/blob/master/modules/custom_operations/user_ie_extensions/tokenizer/python/README.md). This demo has been tested on Windows and Linux platform.

> [!NOTE]
>This tutorial assumes that the current working directory is `<openvino.genai repo>/image_generation/stable_diffusion_1_5/cpp/` and all paths are relative to this folder.

## Step 1: Prepare build environment

C++ Packages:
* [CMake](https://cmake.org/download/): Cross-platform build tool
* [OpenVINO](https://docs.openvino.ai/2023.2/openvino_docs_install_guides_overview.html): Model inference
* Eigen3: Lora enabling

SD preparation could be auto implemented with `scripts/build_dependencies.sh`. This script provides 2 ways to install `OpenVINO 2023.1.0`: [conda-forge](https://anaconda.org/conda-forge/openvino) and [Download archives](https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.1/windows/).
```shell
./scripts/build_dependencies.sh
...
"use conda-forge to install OpenVINO Toolkit 2023.1.0 (C++), or download from archives? (yes/no): "
```

## Step 2: Convert Stable Diffusion v1.5 and Tokenizer models

### Stable Diffusion v1.5 model:

1. Prepare a conda python environment and install dependencies:
    ```shell
    conda create -n SD-CPP python==3.10
    conda activate SD-CPP
    python -m pip install -r scripts/requirements.txt
    ```
2. Download a huggingface SD v1.5 model like:
   - [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
   - [dreamlike-anime-1.0](https://huggingface.co/dreamlike-art/dreamlike-anime-1.0) to run Stable Diffusion with LoRA adapters.


    Example command:
    ```shell
    huggingface-cli download --resume-download --local-dir-use-symlinks False dreamlike-art/dreamlike-anime1.0 --local-dir models
    ```

    Please, refer to the official website for [model downloading](https://huggingface.co/docs/hub/models-downloading) to read more details.

3. Run model conversion script to convert PyTorch model to OpenVINO IR via [optimum-intel](https://github.com/huggingface/optimum-intel). Please, use the script `scripts/convert_model.py` to convert the model into `FP16_static` or `FP16_dyn`, which will be saved into the `models` folder:
    ```shell
    cd scripts
    python convert_model.py -b 1 -t FP16 -sd ../models/dreamlike-anime-1.0 # to convert to a model with static shapes
    python convert_model.py -b 1 -t FP16 -sd ../models/dreamlike-anime-1.0 -dyn True # to keep a model with dynamic shapes
    ```

> [!NOTE]
>Now the pipeline support batch size = 1 only, i.e. static model `(1, 3, 512, 512)`

### LoRA enabling with safetensors

Refer to [python pipeline blog](https://blog.openvino.ai/blog-posts/enable-lora-weights-with-stable-diffusion-controlnet-pipeline).
The safetensor model is loaded via [src/safetensors.h](https://github.com/hsnyder/safetensors.h). The layer name and weight are modified with
`Eigen Lib` and inserted into the SD model with `ov::pass::MatcherPass` in the file `src/lora_cpp.hpp`. 

SD model [dreamlike-anime-1.0](https://huggingface.co/dreamlike-art/dreamlike-anime-1.0) and Lora [soulcard](https://civitai.com/models/67927?modelVersionId=72591) are tested in this pipeline. Here, Lora enabling only for FP16. 

Download and put safetensors and model IR into the models folder. 

### Tokenizer model

There two steps to convert a tokenizer from HuggingFace format to OpenVINO model:

1. Install OpenVINO tokenizers using the following command:
    ```shell
    python -m pip install ../../../thirdparty/openvino_contrib/modules/custom_operations/[transformers]
    ```
2. Use the `scripts/convert_tokenizer.py` script to convert and serialize the tokenizer to OpenVINO IR format:
    ```shell
    cd scripts
    python convert_tokenizer.py --model_id ../models/dreamlike-anime-1.0/tokenizer/ -o ../models/tokenizer/
    ```

> [!NOTE]
When the negative prompt is empty, use the default tokenizer without any configuration (`-e` or `--useOVExtension`).

## Step 3: Build the SD application

```shell
conda activate SD-CPP
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

Example:

Positive prompt: cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting

Negative prompt: (empty, here couldn't use OV tokenizer, check the issues for details)  

Read the numpy latent instead of C++ std lib for the alignment with Python pipeline 

* Generate image without lora `./stable_diffusion -r -l ""`

![image](https://github.com/intel-sandbox/OV_SD_CPP/assets/102195992/66047d66-08a3-4272-abdc-7999d752eea0)

* Generate image with soulcard lora `./stable_diffusion -r`

![image](https://github.com/intel-sandbox/OV_SD_CPP/assets/102195992/0f6e2e3e-74fe-4bd4-bb86-df17cb4bf3f8)

* Generate different size image with dynamic model (C++ lib generated latent): `./stable_diffusion -m ../models/dreamlike-anime-1.0 -t FP16_dyn --height 448 --width 704`

![image](https://github.com/yangsu2022/OV_SD_CPP/assets/102195992/9bd58b64-6688-417e-b435-c0991247b97b)

## Benchmark:

The performance and image quality of C++ pipeline are aligned with Python

To align the performance with [Python SD pipeline](https://github.com/FionaZZ92/OpenVINO_sample/tree/master/SD_controlnet), C++ pipeline will print the duration of each model inferencing only

For the diffusion part, the duration is for all the steps of Unet inferencing, which is the bottleneck

For the generation quality, be careful with the negative prompt and random latent generation. C++ random generation with MT19937 results is differ from `numpy.random.randn()`. Hence, please use `-r, --readNPLatent` for the alignment with Python (this latent file is for output image 512X512 only)

Program optimization: In addition to inference optimization, now parallel optimization with `std::for_each`` only and `add_compile_options(-O3 -march=native -Wall)` with CMake 

## Setup in Windows 10 with VS2019:

1. Download [Anaconda3](https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Windows-x86_64.exe) and setup Conda env `SD-CPP` for OpenVINO with conda-forge and use the anaconda prompt terminal for CMake
2. C++ dependencies:
  * OpenVINO:
    To deployment without Conda: [Download archives* with OpenVINO](https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.1/windows/), unzip and setup environment vars with `.\setupvars.bat`
  * Eigen:
      ```bat
      1. Download from https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip 
      2. Unzip to path C:/Eigen3/eigen-3.4.0 
      3. Run next step's build.bat will report error: not found Eigen3Config.cmake/eigen3-config.cmake
      - Create build folder for Eigen and Open VS in this path C:/Eigen3/eigen-3.4.0/build
      - Open VS's developer PS terminal to do "cmake .." and redo the CMake 
      ```

    Ref: [not found Eigen3Config.cmake/eigen3-config.cmake](https://stackoverflow.com/questions/48144415/not-found-eigen3-dir-when-configuring-a-cmake-project-in-windows)
3. CMake with command lines, create a script build.bat:
    ```bat
    rmdir /Q /S build
    cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release -B build -S .
    cmake --build build --config Release --parallel
    ```
4. Put safetensors and model IR into the models folder with the following default path:
`models\dreamlike-anime-1.0\FP16_static` 
`models\soulcard.safetensors`
5. Run with prompt:  
    ```bat
    cd PROJECT_SOURCE_DIR\build
    .\Release\stable_diffusion.exe -l ''  // without lora
    .\Release\stable_diffusion.exe -l ../models/soulcard.safetensors
    ```
6. Debug within Visual Studio(open .sln file in the `build` folder)
