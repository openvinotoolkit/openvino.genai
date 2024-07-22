# How to Build OpenVINO™ GenAI

## Build for Linux Systems

### Software Requirements 

- [CMake](https://cmake.org/download/) 3.23 or higher
- GCC 7.5 or higher
- Python 3.8 or higher

### Build Instructions

1. Clone OpenVINO GenAI repository and init submodules:
    ```sh
    git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
    cd openvino.genai
    ```
2. Download OpenVINO archive and install dependencies:
    ```sh
    mkdir ./ov/
    curl https://storage.openvinotoolkit.org/repositories/openvino/packages/pre-release/2024.3.0rc1/linux/l_openvino_toolkit_ubuntu20_2024.3.0.dev20240711_x86_64.tgz | tar --directory ./ov/ --strip-components 1 -xz
    sudo ./ov/install_dependencies/install_openvino_dependencies.sh
    ```
3. Build the project:
    ```sh
    source ./ov/setupvars.sh
    cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
    cmake --build ./build/ --config Release --target package -j
    cmake --install ./build/ --config Release --prefix ov
    ```

> **NOTE**: There is a known Python API issue with `ov::Tensor`. The issue is reproduced when building OpenVINO GenAI from sources while using OpenVINO from archives. Using `ov::Tensor` with OpenVINO GenAI will fail on Windows and Linux Systems. Possible errors: `TypeError: generate(): incompatible function arguments.`, `TypeError: __init__(): incompatible constructor arguments.`, `TypeError: Unregistered type : ov::Tensor`.
The preferred approach is to build both OpenVINO and OpenVINO GenAI from sources using the same build environment. Or to install OpenVINO GenAI from [distribution channels](https://docs.openvino.ai/2024/get-started/install-openvino.html).

## Build for Windows Systems

### Software Requirements 

- [CMake](https://cmake.org/download/) 3.23 or higher
- Microsoft Visual Studio 2019 or higher, version 16.3 or later
- Python 3.8 or higher
- Git for Windows

### Build Instructions

1. Clone OpenVINO GenAI repository and init submodules:
    ```sh
    git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
    cd openvino.genai
    ```
2. Download OpenVINO archive and install dependencies:
    ```sh
    mkdir ./ov/
    curl --output ov.zip https://storage.openvinotoolkit.org/repositories/openvino/packages/pre-release/2024.3.0rc1/windows/w_openvino_toolkit_windows_2024.3.0.dev20240711_x86_64.zip
    unzip ov.zip
    mklink /D ov w_openvino_toolkit_windows_2024.3.0.dev20240711_x86_64
    ```
3. Build the project:
    ```sh
    call ov\setupvars.bat
    cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
    cmake --build ./build/ --config Release --target package -j
    cmake --install ./build/ --config Release --prefix ov
    ```

> **NOTE**: There is a known Python API issue with `ov::Tensor`. The issue is reproduced when building OpenVINO GenAI from sources while using OpenVINO from archives. Using `ov::Tensor` with OpenVINO GenAI will fail on Windows and Linux Systems. Possible errors: `TypeError: generate(): incompatible function arguments.`, `TypeError: __init__(): incompatible constructor arguments.`, `TypeError: Unregistered type : ov::Tensor`.
The preferred approach is to build both OpenVINO and OpenVINO GenAI from sources using the same build environment. Or to install OpenVINO GenAI from [distribution channels](https://docs.openvino.ai/2024/get-started/install-openvino.html).

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

1. Clone OpenVINO GenAI repository and init submodules:
    ```sh
    git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
    cd openvino.genai
    ```
2. Download OpenVINO archive and install dependencies:
    ```sh
    mkdir ./ov/
    curl https://storage.openvinotoolkit.org/repositories/openvino/packages/pre-release/2024.3.0rc1/macos/m_openvino_toolkit_macos_12_6_2024.3.0.dev20240711_x86_64.tgz | tar --directory ./ov/ --strip-components 1 -xz
    ```
3. Build the project:
    ```sh
    source ./ov/setupvars.sh
    cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
    cmake --build ./build/ --config Release --target package -j
    cmake --install ./build/ --config Release --prefix ov
    ```
