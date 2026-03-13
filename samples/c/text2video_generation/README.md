# Text-to-Video C API Sample

This sample demonstrates how to use the **OpenVINO™ GenAI C API** to
generate a video tensor from a text prompt using a **Text-to-Video
model** (for example, LTX-Video).

The example shows how to:

-   Initialize the `Text2VideoPipeline`
-   Configure video generation parameters using `VideoGenerationConfig`
-   Execute the generation process from a text prompt

------------------------------------------------------------------------

# Prerequisites

To run this sample, you need:

-   OpenVINO GenAI installed
-   A compatible **Text-to-Video model** exported to the **OpenVINO IR
    format**

------------------------------------------------------------------------

# 1. Install Dependencies

Install the required tools to download and export the model from Hugging
Face:

``` bash
pip install "optimum-intel[openvino,diffusers]"
```

This installs tools required to export models compatible with OpenVINO.

------------------------------------------------------------------------

# 2. Download and Convert the Model

Use `optimum-cli` to export a Text-to-Video model to the OpenVINO
format.

Example using the **LTX-Video** model from Hugging Face:

``` bash
optimum-cli export openvino   --model Lightricks/LTX-Video   --task text-to-video   --weight-format fp16   models/ltx-video
```

⚠️ **Note**

Video generation models can be very large (**5GB or more**). Make sure
you have enough disk space available.

If you are working in a constrained environment (for example GitHub
Codespaces), you may want to export the model to temporary or external
storage.

------------------------------------------------------------------------

# Building the Sample

If you build the OpenVINO GenAI repository from source, this sample will
be compiled automatically when **C samples** are enabled.

``` bash
mkdir build
cd build

cmake .. -DENABLE_SAMPLES=ON

cmake --build . --target text2video_generation_c -j $(nproc)
```

After building, the executable will be available in the build directory.

------------------------------------------------------------------------

# Running the Sample

The compiled program requires:

1.  The path to the exported OpenVINO model directory
2.  A text prompt describing the video

### Syntax

``` bash
./text2video_generation_c <MODEL_DIR> "<PROMPT>"
```

### Example

``` bash
./text2video_generation_c ../models/ltx-video "A robotic bird flying through a neon forest, 4k resolution"
```

------------------------------------------------------------------------

# Expected Output

When the sample runs successfully, it will initialize the pipeline,
configure the generation parameters, and generate the raw video tensor.

Example output:

    --- Initializing Pipeline ---
    --- Setting up Video Configuration ---
    --- Generating Video ---

    Prompt: 'A robotic bird flying through a neon forest, 4k resolution'

    Success! Video tensor generated.
    Tensor Shape: [1, 16, 3, 512, 512]

    --- Done ---

------------------------------------------------------------------------

# Disclaimer

This C sample prints the **generated tensor shape** to the console to
confirm successful execution of the pipeline.

The generated tensor represents raw video data in memory.

To visualize the video, the tensor must be encoded into a video format
such as:

-   MP4
-   GIF

This can be done using external libraries such as:

-   FFmpeg
-   OpenCV

------------------------------------------------------------------------

# Directory Structure

    samples/c/text2video_generation/
    │
    ├── CMakeLists.txt
    ├── main.c
    └── README.md

