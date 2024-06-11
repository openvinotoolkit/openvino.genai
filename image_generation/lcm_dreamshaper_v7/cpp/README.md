# OpenVINO Latent Consistency Model C++ image generation pipeline with LoRA

The pure C++ text-to-image pipeline, driven by the OpenVINO native API for SD v1.5 Latent Consistency Model with LCM Scheduler. It includes advanced features like LoRA integration with safetensors and [OpenVINO Tokenizers](https://github.com/openvinotoolkit/openvino_tokenizers). Loading `openvino_tokenizers` to `ov::Core` enables tokenization. [The common folder](../../common/) contains schedulers for image generation and `imwrite()` for saving `bmp` images. This demo has been tested for Linux platform only. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/263-latent-consistency-models-image-generation/263-lcm-lora-controlnet.ipynb) which provides an example of image generaztion in Python.

> [!NOTE]
> This tutorial assumes that the current working directory is `<openvino.genai repo>/image_generation/lcm_dreamshaper_v7/cpp/` and all paths are relative to this folder.

## Step 1: Prepare build environment

C++ Packages:
* [CMake](https://cmake.org/download/): Cross-platform build tool
* [OpenVINO](https://docs.openvino.ai/2024/get-started/install-openvino.html): Model inference

Prepare a python virtual environment and install dependencies:

To create a virtual environment, open your terminal or command prompt and navigate to the directory where you want to create the environment. Then, run the following command:
```shell
python3 -m venv lcm_cpp_env
```

This will create a new virtual environment named "lcm_cpp_env" in the current directory.

Activating the Environment
Activate the virtual environment using the following command:

```shell
source lcm_cpp_env/bin/activate   # For Unix-based operating system such as Linux or macOS
```

NOTE: If you are using Windows, use `lcm_cpp_env\Scripts\activate`command instead.

### Cloning the Repository

To clone the repository, run the following command:

```shell
git clone https://github.com/openvinotoolkit/openvino.genai.git
```

The above will clone the repository into a directory named "openvino.genai" in the current directory. Then, navigate into the directory using the following command: 

```shell
cd <openvino.genai repo>/image_generation/lcm_dreamshaper_v7/cpp/
```

## Step 2: Latent Consistency Model and Tokenizer models

### Latent Consistency Model model

1. Install dependencies to import models from HuggingFace:

    ```shell
    git submodule update --init
    python -m pip install -r requirements.txt
    ```

2. Download the model from Huggingface and convert it to OpenVINO IR via [optimum-intel CLI](https://github.com/huggingface/optimum-intel). Example command for downloading and exporting FP16 model:

    `optimum-cli export openvino --model SimianLuo/LCM_Dreamshaper_v7 --weight-format fp16 models/lcm_dreamshaper_v7/FP16`

If https://huggingface.co/ is down, the script won't be able to download the model.

> [!NOTE]
> Only static model is currently supported for this sample.

### LoRA enabling with safetensors

Refer to [python pipeline blog](https://blog.openvino.ai/blog-posts/enable-lora-weights-with-stable-diffusion-controlnet-pipeline).
The safetensor model is loaded via [safetensors.h](https://github.com/hsnyder/safetensors.h). The layer name and weight are modified with `Eigen Lib` and inserted into the LCM model with `ov::pass::MatcherPass` in the file [common/diffusers/src/lora.cpp](https://github.com/openvinotoolkit/openvino.genai/blob/master/image_generation/common/diffusers/src/lora.cpp).

LCM model [lcm_dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) and Lora [soulcard](https://civitai.com/models/67927?modelVersionId=72591) are tested in this pipeline.

Download and put safetensors and model IR into the models folder.

## Step 3: Build the LCM application

```shell
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build --config Release --parallel
```

## Step 4: Run Pipeline
```shell
./build/lcm_dreamshaper [-p <posPrompt>] [-s <seed>] [--height <output image>] [--width <output image>] [-d <device>] [-r <readNPLatent>] [-a <alpha>] [-h <help>] [-m <modelPath>] [-t <modelType>]

Usage:
  lcm_dreamshaper [OPTION...]
```

* `-p, --posPrompt arg` Initial positive prompt for LCM (default: a beautiful pink unicorn)
* `-d, --device arg`    AUTO, CPU, or GPU. Doesn't apply to Tokenizer model, OpenVINO Tokenizers can be inferred on a CPU device only (default: CPU)
* `--step arg`          Number of diffusion step (default: 4)
* `-s, --seed arg`      Number of random seed to generate latent (default: 42)
* `--num arg`           Number of image output (default: 1)
* `--height arg`        Height of output image (default: 512)
* `--width arg`         Width of output image (default: 512)
* `-c, --useCache`      Use model caching
* `-r, --readNPLatent`  Read numpy generated latents from file, only supported for one output image
* `-m, --modelPath arg` Specify path to LCM model IRs (default: ./models/lcm_dreamshaper_v7)
* `-t, --type arg`      Specify the type of LCM model IRs (e.g., FP32, FP16 or INT8) (default: FP16)
* `--dynamic`           Specify the model input shape to use dynamic shape
* `-l, --loraPath arg`  Specify path to LoRA file (*.safetensors) (default: )
* `-a, --alpha arg`     Specify alpha for LoRA (default: 0.75)
* `-h, --help`          Print usage

> [!NOTE]
> The tokenizer model will always be loaded to CPU: [OpenVINO Tokenizers](https://github.com/openvinotoolkit/openvino_tokenizers) can be inferred on a CPU device only.

Example:

Positive prompt: a beautiful pink unicorn

Read the numpy latent input and noise for scheduler instead of C++ std lib for the alignment with Python pipeline.

* Generate image with random data generated by Python: `./build/Release/lcm_dreamshaper -r`

![image](./python_random.bmp)

* Generate image with C++ lib generated latent and noise: `./build/Release/lcm_dreamshaper`

![image](./cpp_random.bmp)

* Generate image with soulcard lora and C++ generated latent and noise: `./build/Release/lcm_dreamshaper -r -l path/to/soulcard.safetensors`

![image](./lora_cpp_random.bmp)

## Benchmark:

For the generation quality, C++ random generation with MT19937 results is differ from `numpy.random.randn()` and `diffusers.utils.randn_tensor`. Hence, please use `-r, --readNPLatent` for the alignment with Python (this latent file is for output image 512X512 only)
