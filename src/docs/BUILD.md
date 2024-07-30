# How to Build OpenVINO™ GenAI

> **NOTE**: There is a known Python API issue with `ov::Tensor`. The issue is reproduced when building OpenVINO GenAI from sources while using OpenVINO from archives. Using `ov::Tensor` with OpenVINO GenAI fails. Possible errors: `TypeError: generate(): incompatible function arguments.`, `TypeError: __init__(): incompatible constructor arguments.`, `TypeError: Unregistered type : ov::Tensor`.
The preferred approach is to build both OpenVINO and OpenVINO GenAI from sources using the same build environment. Or to install prebuilt OpenVINO GenAI from [distribution channels](https://docs.openvino.ai/2024/get-started/install-openvino.html).

## Software Requirements

### Linux

- [CMake](https://cmake.org/download/) 3.23 or higher
- GCC 7.5 or higher
- Python 3.8 or higher
- Git

### Windows

- [CMake](https://cmake.org/download/) 3.23 or higher
- Microsoft Visual Studio 2019 or higher, version 16.3 or later
- Python 3.8 or higher
- Git for Windows

### macOS

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
- Git


## Build Instructions

### Build OpenVINO, OpenVINO Tokenizers, and OpenVINO GenAI from sources

1. Build and install OpenVINO from sources following the [instructions](https://github.com/openvinotoolkit/openvino/wiki#how-to-build).  
The path to the OpenVINO install directory is referred as `<INSTALL_DIR>` throughout the document.
2. Clone OpenVINO GenAI repository and init submodules:
    ```sh
    git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
    cd openvino.genai
    ```
3. Build the project:
    ```sh
    source <INSTALL_DIR>/setupvars.sh
    cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
    cmake --build ./build/ --config Release -j
    cmake --install ./build/ --config Release --prefix <INSTALL_DIR>
    ```
    > **NOTE**: For running setupvars script on Windows cmd, use command `call <INSTALL_DIR>\setupvars.bat`

To optimize the package size, you can reduce the ICU (International Components for Unicode) data size when building OpenVINO Tokenizers.
For more information please refer to the [OpenVINO Tokenizers instructions](https://github.com/openvinotoolkit/openvino_tokenizers?tab=readme-ov-file#reducing-the-icu-data-size).

### Build OpenVINO GenAI only

Assuming that you have OpenVINO installed at `<INSTALL_DIR>`:

1. Clone OpenVINO GenAI repository and init submodules:
    ```sh
    git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
    cd openvino.genai
    ```
2. Build the project:
    ```sh
    export OpenVINO_DIR=<INSTALL_DIR>/runtime
    cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
    cmake --build ./build/ --config Release --target package -j
    ```
3. Set environment variables:
    ```sh
    export PYTHONPATH=<INSTALL_DIR>/python:./build/:${PYTHONPATH}
    export LD_LIBRARY_PATH=<INSTALL_DIR>/runtime/lib/intel64:${LD_LIBRARY_PATH}
    ```

### Build OpenVINO GenAI Wheel

1. Clone OpenVINO GenAI repository and init submodules:
    ```sh
    git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
    cd openvino.genai
    ```
2. Set up the environment:
    - Option 1 - using OpenVINO `setupvars.sh` script:
        ```sh
        source <INSTALL_DIR>/setupvars.sh
        ```
    - Option 2 - setting environment variable manually:
        ```sh
        export OpenVINO_DIR=<INSTALL_DIR>/runtime
        export PYTHONPATH=<INSTALL_DIR>/python:./build/:${PYTHONPATH}
        export LD_LIBRARY_PATH=<INSTALL_DIR>/runtime/lib/intel64:${LD_LIBRARY_PATH}
        ```
3. Upgrade pip to ensure you have the latest version:
    ```sh
    python -m pip install --upgrade pip
    ```
4. Build the wheel in the `dist` directory:
    ```sh
    python -m pip wheel . -w dist/ --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release
    ```

### Install OpenVINO GenAI From Source

1. Clone OpenVINO GenAI repository and init submodules:
    ```sh
    git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
    cd openvino.genai
    ```
2. Set up the environment:
    - Option 1 - using OpenVINO `setupvars.sh` script:
        ```sh
        source <INSTALL_DIR>/setupvars.sh
        ```
    - Option 2 - setting environment variable manually:
        ```sh
        export OpenVINO_DIR=<INSTALL_DIR>/runtime
        export PYTHONPATH=<INSTALL_DIR>/python:./build/:${PYTHONPATH}
        export LD_LIBRARY_PATH=<INSTALL_DIR>/runtime/lib/intel64:${LD_LIBRARY_PATH}
        ```
3. Upgrade pip to ensure you have the latest version:
    ```sh
    python -m pip install --upgrade pip
    ```
4. Install the package directly from source:
    ```sh
    python -m pip install .
    ```
5. To verify the installation, run a simple Python script:
    ```python
    import openvino_genai
    print(openvino_genai.__version__)
    ```
