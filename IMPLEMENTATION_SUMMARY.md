# FFmpeg + oneVPL Video Processing Pipeline - Implementation Summary

## Overview
This implementation adds a new video processing pipeline to OpenVINO GenAI that integrates FFmpeg for video decoding with Intel's oneVPL (Video Processing Library) for hardware-accelerated video post-processing.

## What Was Implemented

### 1. Core Pipeline Components

#### Header File: `src/cpp/include/openvino/genai/video_processing/ffmpeg_vpl_pipeline.hpp`
- `VideoProcessingConfig`: Configuration structure for video processing parameters
- `VideoFrame`: Data structure for video frame information
- `FFmpegVPLPipeline`: Main pipeline class with Pimpl pattern

#### Implementation: `src/cpp/src/video_processing/ffmpeg_vpl_pipeline.cpp`
- FFmpeg integration for video decoding (supports multiple formats and codecs)
- oneVPL VPP integration for video post-processing
- Frame-by-frame processing capability
- Batch processing of entire video files
- Video metadata extraction

### 2. Sample Application

#### Sample: `samples/cpp/video_processing/video_processing_sample.cpp`
A command-line application demonstrating the pipeline usage with options for:
- Video scaling (resize to target resolution)
- Denoising filter
- Detail enhancement
- Format conversion (NV12, RGB)

#### Documentation: `samples/cpp/video_processing/README.md`
Comprehensive guide on building and using the sample

#### Test Script: `samples/cpp/video_processing/test_pipeline.sh`
Integration test script that:
- Creates test video using FFmpeg
- Runs multiple processing scenarios
- Validates output files
- Demonstrates all key features

### 3. Tests

#### Test File: `tests/cpp/test_ffmpeg_vpl_pipeline.cpp`
Unit tests using Google Test framework:
- Configuration validation tests
- Pipeline construction tests
- Frame structure tests
- Integration tests (disabled by default, run when test video is available)
- Graceful handling when FFmpeg/VPL are not available

### 4. Build System Integration

#### Changes to CMake:
- `cmake/features.cmake`: Added `ENABLE_FFMPEG_VPL` option (default: OFF)
- `src/cpp/CMakeLists.txt`: 
  - Conditional compilation of video processing sources
  - FFmpeg and VPL library detection and linking
  - Compile definitions for conditional compilation
- `samples/CMakeLists.txt`: Added video_processing sample
- `samples/cpp/video_processing/CMakeLists.txt`: Sample build configuration

### 5. Documentation

#### Technical Documentation: `src/docs/VIDEO_PROCESSING.md`
Comprehensive documentation covering:
- Architecture overview
- Building instructions
- Usage examples (C++ API)
- Configuration options
- Performance considerations
- Limitations
- Future enhancements

#### Python Example: `samples/python/video_processing_example.py`
Conceptual Python API examples showing:
- Basic processing
- Scaling
- Filter application
- Frame-by-frame processing
- Integration with OpenVINO inference

## Key Features

1. **Flexible Input Support**: Works with any video format supported by FFmpeg (MP4, AVI, MKV, etc.)
2. **Hardware Acceleration**: Leverages oneVPL for GPU-accelerated processing on Intel platforms
3. **Multiple Processing Options**:
   - Video scaling/resizing
   - Denoising
   - Detail enhancement
   - Format conversion (NV12 ↔ RGB)
4. **Two Processing Modes**:
   - Batch mode: Process entire video file
   - Streaming mode: Process frames individually
5. **Graceful Degradation**: Code compiles and works even when FFmpeg/VPL are not available

## Build Instructions

### With FFmpeg and VPL Support (enabled)
```bash
# Install dependencies
sudo apt-get install libavcodec-dev libavformat-dev libavutil-dev
# Install oneVPL from https://github.com/intel/libvpl

# Build
cmake -DENABLE_FFMPEG_VPL=ON ..
cmake --build . --target video_processing_sample
```

### Without FFmpeg/VPL (default)
```bash
# Build normally - video processing will be excluded
cmake ..
cmake --build .
```

## Usage Example

```bash
# Basic processing
./video_processing_sample input.mp4 output.yuv

# Scale to 720p with denoising
./video_processing_sample input.mp4 output.yuv --width 1280 --height 720 --denoise

# Convert to RGB format
./video_processing_sample input.mp4 output.rgb --format rgb
```

## API Example

```cpp
#include "openvino/genai/video_processing/ffmpeg_vpl_pipeline.hpp"

// Configure
ov::genai::VideoProcessingConfig config;
config.input_file = "input.mp4";
config.output_file = "output.yuv";
config.target_width = 1280;
config.target_height = 720;
config.denoise = true;

// Process
ov::genai::FFmpegVPLPipeline pipeline(config);
pipeline.process();
```

## Testing

Run the unit tests:
```bash
./tests_continuous_batching --gtest_filter="*FFmpegVPL*"
```

Run the integration test script:
```bash
cd samples/cpp/video_processing
./test_pipeline.sh
```

## Files Created

1. **Headers** (1 file):
   - `src/cpp/include/openvino/genai/video_processing/ffmpeg_vpl_pipeline.hpp`

2. **Implementation** (1 file):
   - `src/cpp/src/video_processing/ffmpeg_vpl_pipeline.cpp`

3. **Sample** (4 files):
   - `samples/cpp/video_processing/video_processing_sample.cpp`
   - `samples/cpp/video_processing/CMakeLists.txt`
   - `samples/cpp/video_processing/README.md`
   - `samples/cpp/video_processing/test_pipeline.sh`

4. **Tests** (1 file):
   - `tests/cpp/test_ffmpeg_vpl_pipeline.cpp`

5. **Documentation** (2 files):
   - `src/docs/VIDEO_PROCESSING.md`
   - `samples/python/video_processing_example.py`

6. **Build System** (3 files modified):
   - `cmake/features.cmake`
   - `src/cpp/CMakeLists.txt`
   - `samples/CMakeLists.txt`

**Total: 12 files (9 new, 3 modified)**

## Design Decisions

1. **Conditional Compilation**: Pipeline is only built when `ENABLE_FFMPEG_VPL=ON` to avoid requiring dependencies by default
2. **Pimpl Pattern**: Implementation details are hidden behind Impl class to avoid exposing FFmpeg/VPL headers
3. **Two Processing Modes**: Support both batch and streaming to accommodate different use cases
4. **Raw Output**: Outputs raw frames (not in container) for maximum flexibility
5. **Minimal Dependencies**: Only requires FFmpeg and VPL when enabled, no other external dependencies

## Future Enhancements

Potential improvements for future development:
- Python bindings for the C++ API
- Support for encoding output to video containers (MP4, etc.)
- Integration examples with OpenVINO inference
- Additional VPP filters (sharpening, color adjustment)
- Batch processing of multiple videos
- Asynchronous processing support
- Memory pool optimization for better performance

## Limitations

1. **Platform Support**: oneVPL hardware acceleration only available on Intel platforms
2. **Output Format**: Raw video data only (no container format encoding)
3. **No Python Bindings**: Currently C++ only
4. **Requires External Libraries**: FFmpeg and oneVPL must be installed separately

## Conclusion

This implementation provides a solid foundation for video processing in OpenVINO GenAI. The pipeline can be used standalone or integrated with OpenVINO inference for video analytics applications. The modular design and conditional compilation ensure it doesn't impact users who don't need video processing capabilities.
