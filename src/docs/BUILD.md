# How to Build OpenVINO™ GenAI

> **NOTE**: There is a known Python API issue with `ov::Tensor`. The issue is reproduced when building OpenVINO GenAI from sources while using OpenVINO from archives. Using `ov::Tensor` with OpenVINO GenAI fails. Possible errors: `TypeError: generate(): incompatible function arguments.`, `TypeError: __init__(): incompatible constructor arguments.`, `TypeError: Unregistered type : ov::Tensor`.
The preferred approach is to build both OpenVINO and OpenVINO GenAI from sources using the same build environment. Or to install prebuilt OpenVINO GenAI from [distribution channels](https://docs.openvino.ai/2024/get-started/install-openvino.html).

## Build for Linux Systems

### Software Requirements 

- [CMake](https://cmake.org/download/) 3.23 or higher
- GCC 7.5 or higher
- Python 3.8 or higher

### Build Instructions

1. Build and install OpenVINO from sources following the [instructions](https://github.com/openvinotoolkit/openvino/wiki#how-to-build).  
The path to the openvino install directory is referred as <INSTALL_DIR> throughout the document.
2. Clone OpenVINO GenAI repository and init submodules:
    ```sh
    git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
    cd openvino.genai
    ```
3. Build the project:
    ```sh
    source <INSTALL_DIR>/setupvars.sh
    cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
    cmake --build ./build/ --config Release --target package -j
    cmake --install ./build/ --config Release --prefix ov
    ```

## Build for Windows Systems

### Software Requirements 

- [CMake](https://cmake.org/download/) 3.23 or higher
- Microsoft Visual Studio 2019 or higher, version 16.3 or later
- Python 3.8 or higher
- Git for Windows

### Build Instructions

1. Build and install OpenVINO from sources following the [instructions](https://github.com/openvinotoolkit/openvino/wiki#how-to-build)  
The path to the openvino install directory is referred as <INSTALL_DIR> throughout the document.
2. Clone OpenVINO GenAI repository and init submodules:
    ```sh
    git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
    cd openvino.genai
    ```
3. Build the project:
    ```sh
    call <INSTALL_DIR>\setupvars.bat
    cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
    cmake --build ./build/ --config Release --target package -j
    cmake --install ./build/ --config Release --prefix ov
    ```

## Build for macOS Systems

### Software Requirements

- [CMake](https://cmake.org/download/) 3.23 or higher
- [brew](https://brew.sh/) package manager to install additional dependencies:
    ```sh
    brew install coreutils scons
    ```
- Clang compiler and other command line tools from Xcode 10.1 or higher:
    ```sh
    xcode-select --install
    ```
- Python 3.8 or higher

### Build Instructions

1. Build and install OpenVINO from sources following the [instructions](https://github.com/openvinotoolkit/openvino/wiki#how-to-build)  
The path to the openvino install directory is referred as <INSTALL_DIR> throughout the document.
2. Clone OpenVINO GenAI repository and init submodules:
    ```sh
    git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
    cd openvino.genai
    ```
3. Build the project:
    ```sh
    source <INSTALL_DIR>/setupvars.sh
    cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
    cmake --build ./build/ --config Release --target package -j
    cmake --install ./build/ --config Release --prefix ov
    ```
