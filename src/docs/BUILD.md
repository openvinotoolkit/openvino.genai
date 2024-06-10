# How To build OpenVINO™ GenAI

## Build for Linux systems

### Software requirements 

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
    <!-- TODO Update link to OV Archive -->
    ```sh
    mkdir ./ov/
    curl https://storage.openvinotoolkit.org/repositories/openvino/packages/pre-release/2024.2.0rc1/linux/l_openvino_toolkit_ubuntu20_2024.2.0.dev20240524_x86_64.tgz | tar --directory ./ov/ --strip-components 1 -xz
    sudo ./ov/install_dependencies/install_openvino_dependencies.sh
    ```
3. Build the project:
    ```sh
    source ./ov/setupvars.sh
    cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
    cmake --build ./build/ --config Release --target package -j
    cmake --install ./build/ --config Release --prefix ov
    ```

## Build for Windows systems

TBD

## Build for macOS systems (Intel CPU)

TBD
