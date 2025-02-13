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

#### Option 1 - using npm:

To use this package locally use `npm link` in `src/js/` directory
and `npm link genai-node` in the folder where you want to add this package as a dependency

#### Option 2 - using package.json:

Add the `genai-node` package manually by specifying the path to the `src/js/` directory in your `package.json`:

```
"genai-node": "file:*path-to-current-directory*"
```

### Installation verification

Run a simple NodeJS script:
```nodejs
const { Pipeline } = require('genai-node');
```