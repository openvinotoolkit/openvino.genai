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
   cmake -DOPENVINO_EXTRA_MODULES=*relative (from openvino folder) path to genai repository* -DCPACK_ARCHIVE_COMPONENT_INSTALL=OFF \
         -DCPACK_GENERATOR=NPM -DENABLE_JS=ON -UTBB* -DENABLE_SYSTEM_TBB=OFF \
         -DENABLE_PYTHON=OFF \
         -DENABLE_WHEEL=OFF \
         -DCPACK_PACKAGE_FILE_NAME=genai_nodejs_bindings \
         -S ./openvino -B ./build
   ```
1. Build OpenVINO archive with GenAI:
   ```sh
   cmake --build ./build --target package -j
   ```

1. In `build` folder you will find `genai_nodejs_bindings.tar.gz`.
   Create `bin` directory by path `src/js/` and unarchive archive content to it.
1. Run tests to be sure that everything works:
   `npm test`

### Perform Test Run

- To run sample you should have prepared model.
  Use this instruction [to download model](https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/js/chat_sample/README.md#download-and-convert-the-model-and-tokenizers)
- Go to [samples/js/chat_sample/](../../samples/js/chat_sample/)
- Read [README.md](../../samples/js/chat_sample/README.md) and follow steps there
  to run **chat sample**.

### Using as npm Dependency

To use this package locally use `npm link` in this directory
and `npm link genai-node` in the folder where you want add this package as dependency

To extract this package and use it as distributed npm package run `npm package`.
This command creates archive that you may use in your projects.
