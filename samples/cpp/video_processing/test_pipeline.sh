#!/bin/bash
# Test script for FFmpeg + oneVPL video processing pipeline
# This script demonstrates the expected usage of the pipeline

set -e

echo "================================"
echo "FFmpeg + oneVPL Pipeline Test"
echo "================================"
echo ""

# Check if sample is built
if [ ! -f "./video_processing_sample" ]; then
    echo "Error: video_processing_sample not found"
    echo "Please build the sample first with:"
    echo "  cmake -DENABLE_FFMPEG_VPL=ON .."
    echo "  cmake --build . --target video_processing_sample"
    exit 1
fi

# Check for test video
if [ ! -f "test_video.mp4" ]; then
    echo "Warning: test_video.mp4 not found"
    echo "Creating a simple test video using FFmpeg..."
    
    # Create a test video if ffmpeg is available
    if command -v ffmpeg &> /dev/null; then
        ffmpeg -f lavfi -i testsrc=duration=5:size=640x480:rate=30 \
               -pix_fmt yuv420p -c:v libx264 test_video.mp4 -y 2>&1 | tail -5
        echo "Test video created: test_video.mp4"
    else
        echo "Error: FFmpeg not available to create test video"
        echo "Please provide a test_video.mp4 file manually"
        exit 1
    fi
fi

echo ""
echo "Test 1: Basic processing (original resolution)"
echo "----------------------------------------------"
./video_processing_sample test_video.mp4 output_original.yuv
if [ -f "output_original.yuv" ]; then
    SIZE=$(stat -f%z "output_original.yuv" 2>/dev/null || stat -c%s "output_original.yuv")
    echo "✓ Output file created: output_original.yuv (${SIZE} bytes)"
else
    echo "✗ Test failed: output file not created"
    exit 1
fi

echo ""
echo "Test 2: Scaling to 1280x720"
echo "----------------------------"
./video_processing_sample test_video.mp4 output_720p.yuv --width 1280 --height 720
if [ -f "output_720p.yuv" ]; then
    SIZE=$(stat -f%z "output_720p.yuv" 2>/dev/null || stat -c%s "output_720p.yuv")
    echo "✓ Output file created: output_720p.yuv (${SIZE} bytes)"
else
    echo "✗ Test failed: output file not created"
    exit 1
fi

echo ""
echo "Test 3: Scaling with denoising"
echo "-------------------------------"
./video_processing_sample test_video.mp4 output_denoised.yuv --width 640 --height 480 --denoise
if [ -f "output_denoised.yuv" ]; then
    SIZE=$(stat -f%z "output_denoised.yuv" 2>/dev/null || stat -c%s "output_denoised.yuv")
    echo "✓ Output file created: output_denoised.yuv (${SIZE} bytes)"
else
    echo "✗ Test failed: output file not created"
    exit 1
fi

echo ""
echo "Test 4: RGB format conversion"
echo "------------------------------"
./video_processing_sample test_video.mp4 output.rgb --format rgb
if [ -f "output.rgb" ]; then
    SIZE=$(stat -f%z "output.rgb" 2>/dev/null || stat -c%s "output.rgb")
    echo "✓ Output file created: output.rgb (${SIZE} bytes)"
else
    echo "✗ Test failed: output file not created"
    exit 1
fi

echo ""
echo "================================"
echo "All tests passed! ✓"
echo "================================"
echo ""
echo "Generated files:"
ls -lh output_*.yuv output.rgb 2>/dev/null | awk '{print "  " $9 " - " $5}'

echo ""
echo "Cleanup? (y/n)"
read -r response
if [ "$response" = "y" ]; then
    rm -f output_*.yuv output.rgb test_video.mp4
    echo "Cleaned up test files"
fi
