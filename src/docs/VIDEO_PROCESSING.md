# FFmpeg + oneVPL Video Processing Pipeline

## Overview

This module provides a video processing pipeline that combines FFmpeg for video decoding with Intel's oneVPL (Video Processing Library) for hardware-accelerated video post-processing.

## Architecture

The pipeline consists of three main components:

1. **FFmpeg Decoder**: Handles video file decoding
   - Supports multiple video formats (MP4, AVI, MKV, MOV, etc.)
   - Supports various codecs (H.264, H.265, VP9, AV1, etc.)
   - Extracts video frames and metadata

2. **oneVPL VPP (Video Post-Processing)**: Performs video processing operations
   - Scaling (resize to target resolution)
   - Format conversion (NV12, RGB, etc.)
   - Denoising
   - Detail enhancement
   - Hardware acceleration on Intel platforms

3. **Pipeline Controller**: Manages the data flow between FFmpeg and oneVPL
   - Converts frame formats between FFmpeg and oneVPL
   - Handles buffering and synchronization
   - Provides a simple API for processing

## Building

The video processing pipeline is disabled by default. To enable it:

```bash
cmake -DENABLE_FFMPEG_VPL=ON ..
cmake --build .
```

### Dependencies

- **FFmpeg libraries**: libavcodec, libavformat, libavutil
- **oneVPL**: Intel Video Processing Library

#### Installing Dependencies on Ubuntu

```bash
sudo apt-get install libavcodec-dev libavformat-dev libavutil-dev
# For oneVPL, download from https://github.com/intel/libvpl
```

#### Installing Dependencies on Windows

Use vcpkg:
```cmd
vcpkg install ffmpeg
# For oneVPL, download from https://github.com/intel/libvpl
```

## Usage

### C++ API

```cpp
#include "openvino/genai/video_processing/ffmpeg_vpl_pipeline.hpp"

// Configure the pipeline
ov::genai::VideoProcessingConfig config;
config.input_file = "input.mp4";
config.output_file = "output.yuv";
config.target_width = 1280;
config.target_height = 720;
config.denoise = true;
config.output_format = 0; // 0=NV12, 1=RGB

// Create and run the pipeline
ov::genai::FFmpegVPLPipeline pipeline(config);

// Option 1: Process entire video
pipeline.process();

// Option 2: Process frame by frame
ov::genai::VideoFrame frame;
while (pipeline.get_next_frame(frame)) {
    // Process frame.data
}
```

### Command-line Sample

```bash
# Basic usage
./video_processing_sample input.mp4 output.yuv

# With scaling and denoising
./video_processing_sample input.mp4 output.yuv --width 1920 --height 1080 --denoise

# Convert to RGB format
./video_processing_sample input.mp4 output.rgb --format rgb
```

## Configuration Options

### VideoProcessingConfig

- `input_file` (string): Path to input video file
- `output_file` (string): Path to output file for processed frames
- `target_width` (int): Target width for scaling (0 = keep original)
- `target_height` (int): Target height for scaling (0 = keep original)
- `denoise` (bool): Enable denoising filter
- `detail_enhance` (bool): Enable detail enhancement filter
- `output_format` (int): Output format (0=NV12, 1=RGB)

### VideoFrame

- `data` (vector<uint8_t>): Raw frame data
- `width` (int): Frame width
- `height` (int): Frame height
- `format` (int): Frame format (0=NV12, 1=RGB)
- `timestamp` (int64_t): Presentation timestamp

## Performance

oneVPL provides hardware acceleration on Intel platforms:

- **CPU**: Software fallback, slower but compatible
- **GPU (integrated)**: Hardware-accelerated, efficient for most use cases
- **GPU (discrete)**: Maximum performance for high-resolution video

Performance depends on:
- Input video resolution and codec
- Target resolution and processing operations
- Available hardware acceleration

## Limitations

- oneVPL hardware acceleration is only available on Intel platforms
- Some advanced VPP features may not be available on all hardware
- Output file format is raw video data (no container format)

## Testing

The pipeline includes comprehensive tests:

```bash
# Run all tests
./tests_continuous_batching --gtest_filter="*FFmpegVPL*"

# Run specific test
./tests_continuous_batching --gtest_filter="FFmpegVPLPipelineTest.ConstructorWithConfig"
```

## Examples

See the [video_processing sample](../samples/cpp/video_processing/) for complete examples.

## Future Enhancements

Potential improvements:
- Add support for encoding output to video containers
- Integrate with OpenVINO for inference on video frames
- Add more VPP filters (sharpening, color adjustment, etc.)
- Support for batch processing multiple videos
- Python bindings for the API

## References

- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [oneVPL Documentation](https://intel.github.io/libvpl/)
- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)
