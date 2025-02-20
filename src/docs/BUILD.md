# How to Build OpenVINO™ GenAI

> **NOTE**: There is a known Python API issue with `ov::Tensor`. The issue is reproduced when building OpenVINO GenAI from sources while using OpenVINO from archives. Using `ov::Tensor` with OpenVINO GenAI fails. Possible errors: `TypeError: generate(): incompatible function arguments.`, `TypeError: __init__(): incompatible constructor arguments.`, `TypeError: Unregistered type : ov::Tensor`.
The preferred approach is to build both OpenVINO and OpenVINO GenAI from sources using the same build environment. Or to install prebuilt OpenVINO GenAI from [distribution channels](https://docs.openvino.ai/2025/get-started/install-openvino.html).

## Software Requirements

### Linux

- [CMake](https://cmake.org/download/) 3.23 or higher
- GCC 7.5 or higher
- Python 3.9 or higher
- Git

### Windows

- [CMake](https://cmake.org/download/) 3.23 or higher
- Microsoft Visual Studio 2019 or higher, version 16.3 or later
- Python 3.9 or higher
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
- Python 3.9 or higher
- Git


## Build Instructions

### Build OpenVINO GenAI as OpenVINO Extra Module

OpenVINO GenAI can be built as an extra module during the OpenVINO build process. This method simplifies the build process by integrating OpenVINO GenAI directly into the OpenVINO build.

1. Clone OpenVINO and OpenVINO GenAI repositories:
    ```sh
    git clone --recursive https://github.com/openvinotoolkit/openvino.git
    git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
    ```
2. Configure CMake with OpenVINO extra modules:
    ```sh
    cmake -DOPENVINO_EXTRA_MODULES=./openvino.genai -DCPACK_ARCHIVE_COMPONENT_INSTALL=OFF -S ./openvino -B ./build
    ```
3. Build OpenVINO archive with GenAI:
    ```sh
    cmake --build ./build --target package -j
    ```

After the build process completes, you should find the packaged OpenVINO with GenAI in the `build` directory.
Follow the OpenVINO [build instructions](https://github.com/openvinotoolkit/openvino/wiki#how-to-build) and [install instructions](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/installing.md) for additional information.

### Build OpenVINO, OpenVINO Tokenizers, and OpenVINO GenAI From Source

1. Build and install OpenVINO from sources following the [instructions](https://github.com/openvinotoolkit/openvino/wiki#how-to-build).  
The path to the OpenVINO install directory is referred as `<INSTALL_DIR>` throughout the document.
2. Clone OpenVINO GenAI repository and init submodules:
    ```sh
    git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
    cd openvino.genai
    ```
3. Set up the environment:

    #### Option 1 - using OpenVINO `setupvars` script:

    Linux and macOS:
    ```sh
    source <INSTALL_DIR>/setupvars.sh
    ```

    Windows Command Prompt:
    ```cmd
    call <INSTALL_DIR>\setupvars.bat
    ```

    Windows PowerShell:
    ```cmd
    . <INSTALL_DIR>/setupvars.ps1
    ```

    #### Option 2 - setting environment variables manually:

    Linux:
    ```sh
    export OpenVINO_DIR=<INSTALL_DIR>/runtime
    export PYTHONPATH=<INSTALL_DIR>/python:./build/:$PYTHONPATH
    export LD_LIBRARY_PATH=<INSTALL_DIR>/runtime/lib/intel64:$LD_LIBRARY_PATH
    ```

    macOS:
    ```sh
    export OpenVINO_DIR=<INSTALL_DIR>/runtime
    export PYTHONPATH=<INSTALL_DIR>/python:./build/:$PYTHONPATH
    export DYLD_LIBRARY_PATH=<INSTALL_DIR>/runtime/lib/intel64:$LD_LIBRARY_PATH
    ```

    Windows Command Prompt:
    ```cmd
    set OpenVINO_DIR=<INSTALL_DIR>\runtime
    set PYTHONPATH=<INSTALL_DIR>\python;%CD%\build;%PYTHONPATH%
    set OPENVINO_LIB_PATHS=<INSTALL_DIR>\bin\intel64\Release;%OPENVINO_LIB_PATHS%
    set PATH=%OPENVINO_LIB_PATHS%;%PATH%
    ```
    
    Windows PowerShell:
    ```sh
    $env:OpenVINO_DIR = "<INSTALL_DIR>\runtime"
    $env:PYTHONPATH = "<INSTALL_DIR>\python;$PWD\build;$env:PYTHONPATH"
    $env:OPENVINO_LIB_PATHS = "<INSTALL_DIR>\bin\intel64\Release;$env:OPENVINO_LIB_PATHS"
    $env:PATH = "$env:OPENVINO_LIB_PATHS;$env:PATH"
    ```

4. Build the project:
    ```sh
    cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
    cmake --build ./build/ --config Release -j
    ```

5. Install OpenVINO GenAI:

    #### Option 1 - using cmake:
    
    The following command will store built OpenVINO GenAI artifacts along with OpenVINO in `<INSTALL_DIR>`:

    ```sh
    cmake --install ./build/ --config Release --prefix <INSTALL_DIR>
    ```

    #### Option 2 - setting paths to built OpenVINO GenAI artifacts manually:

    The path to the OpenVINO GenAI root directory is referred as `<GENAI_ROOT_DIR>` throughout the document.

    Linux:
    ```sh
    export PYTHONPATH=<GENAI_ROOT_DIR>/build/:$PYTHONPATH
    export LD_LIBRARY_PATH=<GENAI_ROOT_DIR>/build/openvino_genai/:$LD_LIBRARY_PATH
    ```

    macOS:
    ```sh
    export PYTHONPATH=<GENAI_ROOT_DIR>/build:$PYTHONPATH
    export DYLD_LIBRARY_PATH=<GENAI_ROOT_DIR>/build/openvino_genai:$DYLD_LIBRARY_PATH
    ```

    Windows Command Prompt:
    ```cmd
    set PYTHONPATH=<GENAI_ROOT_DIR>\build;%PYTHONPATH%
    set PATH=<GENAI_ROOT_DIR>\build\openvino_genai;%PATH%
    ```

    Windows PowerShell:
    ```sh
    $env:PYTHONPATH = "<GENAI_ROOT_DIR>\build;$env:PYTHONPATH"
    $env:PATH = "<GENAI_ROOT_DIR>\build\openvino_genai;$env:PATH"
    ```

To optimize the package size, you can reduce the ICU (International Components for Unicode) data size when OpenVINO Tokenizers are built as a submodule of OpenVINO GenAI.
For more information please refer to the [OpenVINO Tokenizers instructions](https://github.com/openvinotoolkit/openvino_tokenizers?tab=readme-ov-file#reducing-the-icu-data-size).


### Build OpenVINO GenAI Wheel

1. Build and install OpenVINO from sources following the [instructions](https://github.com/openvinotoolkit/openvino/wiki#how-to-build)  
The path to the openvino install directory is referred as <INSTALL_DIR> throughout the document.
2. Clone OpenVINO GenAI repository and init submodules:
    ```sh
    git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
    cd openvino.genai
    ```
2. Set up the environment:
    - Option 1 - using OpenVINO `setupvars.sh` script:
        ```sh
        source <INSTALL_DIR>/setupvars.sh
        ```
    - Option 2 - setting environment variables manually:
        ```sh
        export OpenVINO_DIR=<INSTALL_DIR>/runtime
        export PYTHONPATH=<INSTALL_DIR>/python:./build/:$PYTHONPATH
        export LD_LIBRARY_PATH=<INSTALL_DIR>/runtime/lib/intel64:$LD_LIBRARY_PATH
        ```
3. Upgrade pip to ensure you have the latest version:
    ```sh
    python -m pip install --upgrade pip
    ```
4. Build the wheel in the `dist` directory:
    ```sh
    python -m pip wheel . -w dist/ --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    ```

> **NOTE**: You'd need to build ABI compatible OpenVINO and OpenVINO Tokenizers for Ubuntu instead of downloading them from PyPI. See [OpenVINO™ GenAI Dependencies](../README.md#openvino-genai-dependencies) for the explanation.

### Build OpenVINO GenAI JavaScript Bindings

Build OpenVINO GenAI JavaScript Bindings from sources following the [instructions](../js/BUILD.md).

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
    - Option 2 - setting environment variables manually:
        ```sh
        export OpenVINO_DIR=<INSTALL_DIR>/runtime
        export PYTHONPATH=<INSTALL_DIR>/python:./build/:$PYTHONPATH
        export LD_LIBRARY_PATH=<INSTALL_DIR>/runtime/lib/intel64:$LD_LIBRARY_PATH
        ```
3. Upgrade pip to ensure you have the latest version:
    ```sh
    python -m pip install --upgrade pip
    ```
4. Install the package directly from source:
    ```sh
    python -m pip install .
    ```
5. Verify the installation:
    ```sh
    python -c "import openvino_genai; print(openvino_genai.__version__)"
    ```
