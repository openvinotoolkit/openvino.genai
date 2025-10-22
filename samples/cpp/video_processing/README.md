# FFmpeg + oneVPL Video Processing Pipeline Sample

This sample demonstrates how to use the FFmpeg + oneVPL video processing pipeline to decode video files using FFmpeg and process them using Intel's oneVPL (Video Processing Library).

## Description

The pipeline consists of two main components:

1. **FFmpeg Decoder**: Decodes video frames from various video formats (MP4, AVI, MKV, etc.)
2. **oneVPL VPP (Video Post-Processing)**: Processes the decoded frames with operations like:
   - Scaling (resize to target resolution)
   - Denoising
   - Detail enhancement
   - Format conversion (NV12, RGB)

## How It Works

1. FFmpeg decodes the input video frame by frame
2. Each decoded frame is passed to oneVPL VPP for processing
3. The processed frames are written to the output file

## Building the Sample

The sample requires FFmpeg and oneVPL libraries to be installed and the build configured with:

```bash
cmake -DENABLE_FFMPEG_VPL=ON ..
cmake --build . --target video_processing_sample
```

## Running the Sample

Basic usage:
```bash
./video_processing_sample input.mp4 output.yuv
```

With options:
```bash
./video_processing_sample input.mp4 output.yuv --width 1280 --height 720 --denoise
```

### Command-line Arguments

- `<input_video>` - Path to input video file (required)
- `<output_file>` - Path to output file where processed frames will be saved (required)
- `--width <width>` - Target width for scaling (optional, default: keep original)
- `--height <height>` - Target height for scaling (optional, default: keep original)
- `--denoise` - Enable denoising filter (optional)
- `--enhance` - Enable detail enhancement filter (optional)
- `--format <format>` - Output format: `nv12` (default) or `rgb` (optional)

## Examples

1. **Simple processing** (decode and re-encode):
   ```bash
   ./video_processing_sample video.mp4 output.yuv
   ```

2. **Scale video to 720p**:
   ```bash
   ./video_processing_sample video.mp4 output.yuv --width 1280 --height 720
   ```

3. **Scale with denoising**:
   ```bash
   ./video_processing_sample video.mp4 output.yuv --width 1920 --height 1080 --denoise
   ```

4. **Convert to RGB format**:
   ```bash
   ./video_processing_sample video.mp4 output.rgb --format rgb
   ```

## Output Format

The output file contains raw video frames in the specified format:
- **NV12** (default): YUV 4:2:0 format, widely used for video processing
- **RGB**: RGB24 format, suitable for image processing applications

## Dependencies

- **FFmpeg** (libavcodec, libavformat, libavutil): For video decoding
- **oneVPL**: Intel's Video Processing Library for hardware-accelerated video processing
- **OpenVINO GenAI**: The framework this sample is built on

## Notes

- oneVPL provides hardware acceleration on Intel platforms
- The pipeline supports various video codecs through FFmpeg (H.264, H.265, VP9, etc.)
- Performance depends on the video resolution, codec, and processing operations enabled
