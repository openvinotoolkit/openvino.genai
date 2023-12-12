<!---
- [Repository Description](#repository-description))
   - [Pipeline Samples](#pipeline-samples)
   - [License](#license)
- [Requirements](#requirements)
-->
## GenAI Pipeline Repository

The GenAI repository contains pipelines that implement image and text generation tasks.
The implementation uses OpenVINO capabilities to optimize the pipelines. Each sample covers
a family of models and suggests certain modifications to adapt the code to specific needs.

> Note
This project is not for production use.

### Pipeline Samples 
 - [image generation by stable diffusion](image_generation/stable_diffusion_1_5/cpp)
 - [large language model demo](llm/cpp)
 - [Benchmarking script for large language models](llm_bench/python)

### License

The GenAI repository is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release
your contribution under these terms.

## Requirements

Requirements may vary for different samples. See respective readme files for more details.
Every pipeline requires https://github.com/openvinotoolkit/openvino for C++ to be installed.

Build the pipelines and the `user_ov_extensions` submodule

```sh
git submodule update --init
mkdir ./build/ && cd ./build/
source <OpenVINO dir>/setupvars.sh
cmake -DCMAKE_BUILD_TYPE=Release ../ && cmake --build ./ --config Release -j
```

To enable non ASCII characters for Windows cmd open `Region` settings from `Control panel`.
`Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`.
Reboot for the change to take effect.
