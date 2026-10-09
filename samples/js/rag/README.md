# Retrieval Augmented Generation Sample

This example showcases inference of Text Embedding and Text Rerank Models. The application has limited configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `TextEmbeddingPipeline` and `TextRerankPipeline`, which use text as an input source.

## Download and Convert the Model and Tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
```

To export text embedding model run Optimum CLI command:

```sh
optimum-cli export openvino --task feature-extraction --model BAAI/bge-small-en-v1.5 BAAI/bge-small-en-v1.5
```

To export text reranking model run Optimum CLI command:

```sh
optimum-cli export openvino --task text-classification --model cross-encoder/ms-marco-MiniLM-L6-v2 cross-encoder/ms-marco-MiniLM-L6-v2
```

## Run

Compile GenAI JavaScript bindings archive first using [the instructions](../../../src/js/README.md#build-bindings).

Run `npm install` from the `samples/js` directory and the text samples (1 and 2) will be ready to run. The image and video sample (3) additionally requires building a native OpenCV addon — follow its [environment setup](#3-image-and-video-embedding-sample-image_video_embeddingjs) steps.

### 1. Text Embedding Sample (`text_embeddings.js`)
- **Description:**
  Demonstrates inference of text embedding models using OpenVINO GenAI. Converts input text into vector embeddings for downstream tasks such as retrieval or semantic search.
- **Run Command:**
  ```sh
  node text_embeddings.js <MODEL_DIR> "Document 1" "Document 2"
  ```
Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#embedding-models) for more details.

### 2. Text Rerank Sample (`text_rerank.js`)
- **Description:**
  Demonstrates inference of text rerank models using OpenVINO GenAI. Reranks a list of candidate documents based on their relevance to a query using a cross-encoder or reranker model.
- **Run Command:**
  ```sh
  node text_rerank.js <MODEL_DIR> "<QUERY>" "<TEXT 1>" ["<TEXT 2>" ...]
  ```

### 3. Image and Video Embedding Sample (`image_video_embedding.js`)
- **Description:**
  Demonstrates multimodal retrieval with OpenVINO GenAI `EmbeddingPipeline`. Embeds a user text query and multiple image or video inputs, ranks the inputs by cosine similarity, and prints the most similar image or video. Image decoding reuses the shared [`image_utils.js`](../image_utils.js) helper; video decoding uses the native [`@u4/opencv4nodejs`](https://www.npmjs.com/package/@u4/opencv4nodejs) addon.

Unlike the text samples, this one needs the native `@u4/opencv4nodejs` addon, which has to be compiled against an OpenCV **4.6** installation. The autobuild triggered by a plain `npm install` is slow and frequently fails (no system OpenCV available, or a broken system `node-gyp`), so build the addon explicitly with the steps below. The commands target Linux; adapt the paths for other platforms.

#### Step 1. Prerequisites
- Node.js >= 22 and `npm`.
- A C/C++ toolchain (`gcc`/`g++` or `clang`) and Python 3 — required by `node-gyp` to compile the addon.
- The OpenVINO GenAI JavaScript bindings, built as described in [the bindings instructions](../../../src/js/README.md#build-bindings).

#### Step 2. Install OpenCV 4.6
The addon must link against OpenCV **4.6**. Pick one option and export the three location variables; they are reused in Steps 4 and 5.

**Option A — conda-forge (no admin rights, recommended):**
```sh
conda create -y -n ov-genai-opencv -c conda-forge "libopencv=4.6.0" pkg-config
# Path to the created environment (adjust to your conda/miniforge location):
export OPENCV_DIR="$HOME/miniforge3/envs/ov-genai-opencv"
export OPENCV_INCLUDE_DIR="$OPENCV_DIR/include/opencv4"
export OPENCV_LIB_DIR="$OPENCV_DIR/lib"
export OPENCV_BIN_DIR="$OPENCV_DIR/bin"
```

**Option B — system packages (Ubuntu/Debian, requires `sudo`; tested with OpenCV 4.6):**
```sh
sudo apt-get update && sudo apt-get install -y libopencv-dev build-essential python3
export OPENCV_INCLUDE_DIR="/usr/include/opencv4"
export OPENCV_LIB_DIR="/usr/lib/x86_64-linux-gnu"
export OPENCV_BIN_DIR="/usr/bin"
```

#### Step 3. Install the JavaScript dependencies
From the `samples/js` directory, install the dependencies without running the addon's autobuild, then add a known-good `node-gyp` locally (the system `node-gyp` shipped by some distributions is broken):
```sh
cd samples/js
npm install --ignore-scripts
npm install --no-save --ignore-scripts node-gyp@latest
```

#### Step 4. Build the native OpenCV addon
Still in `samples/js`, put the local `node-gyp` first on `PATH`, disable the autobuild, and compile the addon against the OpenCV from Step 2 (the `OPENCV_*` variables must still be exported):
```sh
export PATH="$PWD/node_modules/.bin:$PATH"
export OPENCV4NODEJS_DISABLE_AUTOBUILD=1
( cd node_modules/@u4/opencv4nodejs && node bin/install.js rebuild )
```
Verify that the addon was produced:
```sh
ls node_modules/@u4/opencv4nodejs/build/Release/opencv4nodejs.node
```

#### Step 5. Run the sample
The `OPENCV_*` variables and `OPENCV4NODEJS_DISABLE_AUTOBUILD=1` must also be set at **run time**, because the addon re-resolves OpenCV when it loads. If the GenAI bindings were built locally, make their runtime available first (for example `source <openvino>/setupvars.sh`). Then, from `samples/js`, re-export the same three `OPENCV_*` variables as in Step 2 and run:
```sh
export OPENCV4NODEJS_DISABLE_AUTOBUILD=1
# OPENCV_INCLUDE_DIR / OPENCV_LIB_DIR / OPENCV_BIN_DIR as set in Step 2

node rag/image_video_embedding.js <MODEL_DIR> \
  --query "<QUERY>" \
  --images <IMAGE_PATH_1> [<IMAGE_PATH_2> ...] \
  --videos <VIDEO_PATH_1> [<VIDEO_PATH_2> ...] \
  [--num-video-frames 8] [--device CPU]
```
At least one `--images` or `--videos` input is required. Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#embedding-models) for the list of supported models.

#### Troubleshooting
- **`Cannot find module 'nopt'` or other `node-gyp` failures** — the system `node-gyp` is broken. Ensure `samples/js/node_modules/.bin` is first on `PATH` (Step 4) so the locally installed `node-gyp` is used.
- **`opencv2/core.hpp: No such file or directory`** — the `OPENCV_*` variables are not set, or the addon was built with a plain `npm install`/`node-gyp`. Export the three variables from Step 2 and rebuild with `node bin/install.js rebuild` (Step 4).
- **`No build found ... launch opencv-build-npm once`** — the `OPENCV_*` variables and `OPENCV4NODEJS_DISABLE_AUTOBUILD=1` are missing at run time. Export them before running the sample (Step 5).
- **`Cannot find module '.../build/Release/opencv4nodejs.node'`** — the addon was not built. Run Step 4.

# Text Embedding Pipeline Usage

```js
import { TextEmbeddingPipeline } from 'openvino-genai-node';

const pipeline = await TextEmbeddingPipeline(model_dir, "CPU");

const embeddings = await pipeline.embedDocuments(["document1", "document2"]);
```

# Text Rerank Pipeline Usage

```js
import { TextRerankPipeline } from 'openvino-genai-node';

const pipeline = await TextRerankPipeline(modelPath, { device: "CPU" });

const rerankResult = await pipeline.rerank(query, documents);
```

# Multimodal Embedding Pipeline Usage

```js
import { EmbeddingPipeline } from 'openvino-genai-node';

const pipeline = await EmbeddingPipeline(modelDir, "CPU");

const { embeddings } = await pipeline.embed("A query", { images: [imageTensor] });
```
