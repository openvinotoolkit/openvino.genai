# Video-to-Video Style Transfer C++ Sample

This sample demonstrates how to run stable diffusion style transfer on video inputs using OpenVINO GenAI. Unlike image-to-image samples, this pipeline is optimized for temporal consistency and video throughput.

## Key Features

* **Full Video I/O**: Integrated `cv::VideoCapture` and `cv::VideoWriter` for direct MP4 processing.
* **Temporal Stabilization**: Implements weighted frame blending to reduce flickering between generated frames.
* **Performance Optimization**: Uses a "Skip & Blend" strategy (rendering keyframes and blending intermediates) to emulate real-time speeds on Edge hardware.
* **Visual Enhancement**: Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for better color definition in the output.

## Prerequisites

1.  **Install OpenVINO GenAI**: Ensure you have the OpenVINO GenAI C++ libraries installed.
2.  **Install OpenCV**: This sample requires OpenCV for video decoding/encoding.
    * **Windows**: Set `OpenCV_DIR` to your build folder during CMake configuration.
    * **Linux**: Install via `sudo apt-get install libopencv-dev`.
3.  **Prepare Input Video**: Place a video file named `input.mp4` in your build directory. You can download a [sample video here](https://storage.openvinotoolkit.org/data/test_data/videos/car-detection.mp4) (rename it to `input.mp4`) or use your own.

## Prepare the Model

This sample works with standard Stable Diffusion models (e.g., Dreamlike Anime, SD v1.5). You must first convert the model to OpenVINO format using `optimum-cli`.

```bash
# Install dependencies
pip install optimum[openvino]

# Download and convert a model (e.g., Dreamlike Anime)
optimum-cli export openvino --model dreamlike-art/dreamlike-anime-1.0 --task text-to-image models/dreamlike_anime
```

## Build Instructions

Linux:
```bash
mkdir build && cd build
cmake ..
make
```

Windows:
```bash
mkdir build && cd build
cmake .. -DOpenCV_DIR="C:\path\to\opencv\build"
cmake --build . --config Release
```

Examples: 

Run on GPU(Recommended for speed)
```bash
./video_style_transfer "models/dreamlike_anime" "input.mp4" "GPU"
```
Run on CPU:
```bash
./video_style_transfer "models/dreamlike_anime" "input.mp4" "CPU"
```

Output:
The processed video will be saved as output_style_transfer.mp4 in the execution directory.
