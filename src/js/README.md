# OpenVINO™ GenAI Node.js bindings (preview)

## DISCLAIMER

This is preview version, do not use it in production!

## Install and Run

### Requirements

- Node.js v21+
- Tested on Ubuntu, another OS didn't tested yet

### Build Bindings

Build OpenVINO™ GenAI JavaScript Bindings from sources following the [instructions](../js/BUILD.md).

### Using the package from your project

Since the OpenVINO GenAI NodeJS package depends on the OpenVINO NodeJS package, these packages must be of the same version.
If you intend to use one of the released versions, please check and install the correct version of the `openvino-node` package in your project.
If you want to use an unstable version of the OpenVINO GenAI NodeJS package, you will also need to build and use OpenVINO™ JavaScript Bindings from source
following the [instructions](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/docs/README.md#build) to use it correctly.

Then you can use OpenVINO™ GenAI JavaScript Bindings in one of the following ways:

#### Option 1 - using npm:

To use this package locally use `npm link` in `src/js/` directory
and `npm link openvino-genai-node` in the folder where you want to add this package as a dependency

#### Option 2 - using package.json:

Add the `openvino-genai-node` package manually by specifying the path to the `src/js/` directory in your `package.json`:

```
"openvino-genai-node": "file:*path-to-current-directory*"
```

### Verify the installation:
```sh
node -e "const { Pipeline } = require('openvino-genai-node'); console.log(Pipeline);"
```