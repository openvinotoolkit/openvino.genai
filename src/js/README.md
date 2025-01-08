# OpenVINO™ GenAI Node.js bindings (preview)

## DISCLAIMER

This is preview version, do not use it in production!

## Install and Run

### Requirements

- Node.js v21+
- Tested on Ubuntu, another OS didn't tested yet

### Build Bindings

#### Build OpenVINO GenAI as OpenVINO Extra Module

OpenVINO GenAI Node.js bindings can be built as an extra module during the OpenVINO build process. This method simplifies the build process by integrating OpenVINO GenAI directly into the OpenVINO build.

1. Clone OpenVINO repository:
   ```sh
   git clone --recursive https://github.com/openvinotoolkit/openvino.git
   ```
1. Configure CMake with OpenVINO extra modules:
   ```sh
   cmake -DOPENVINO_EXTRA_MODULES=*path to genai repository directory* -DCPACK_ARCHIVE_COMPONENT_INSTALL=OFF \
         -DCPACK_GENERATOR=NPM \
         -DENABLE_PYTHON=OFF \
         -DENABLE_WHEEL=OFF \
         -DCPACK_PACKAGE_FILE_NAME=genai_nodejs_bindings \
         -S ./openvino -B ./build
   ```
1. Build OpenVINO archive with GenAI:
   ```sh
   cmake --build ./build --target package -j
   ```

1. Put Node.js bindings into npm package `bin` directory and install dependencies:
   ```sh
   mkdir ./src/js/bin/
   tar -xvf ./build/genai_nodejs_bindings.tar.gz --directory ./src/js/bin/
   cd ./src/js/
   npm install
   ```
1. Run tests to be sure that everything works:
   ```sh
   npm test
   ```

### Using as npm Dependency

To use this package locally use `npm link` in `src/js/` directory
and `npm link genai-node` in the folder where you want to add this package as a dependency

To extract this package and use it as distributed npm package run `npm package`.
This command creates archive that you may use in your projects.
